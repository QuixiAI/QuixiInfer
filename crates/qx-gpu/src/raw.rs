//! Kernel launches through the CUDA driver API, on CubeCL's own stream, for the DeepSeek chunk path.
//!
//! CubeCL's launch costs about 120 us of host time (its client/server channel submission and its memory pool's
//! work for the per-launch info buffer) and a 28 us host-to-device copy of the info words on the compute stream
//! (2026-11-07 profiles: 55 dense launches and 93 such copies per slab-layer, 8.7 s of the 18.6 s of a 4,096-row
//! chunk spent enqueuing). A launch here is a memcpy into a pinned ring slot, one `cuMemcpyHtoDAsync` of the info
//! words into the device ring, and one `cuLaunchKernel` with the slot's address as the kernel's `const u64* info`:
//! a few microseconds of host time, no pool, no channel.
//!
//! The same flattened sources CubeCL compiles are compiled here with NVRTC to PTX (once per module per process)
//! and loaded with `cuModuleLoadData` (the driver's JIT, cached on disk like CubeCL's); the entry points are the
//! `extern "C"` names. Launches go to the first stream CubeCL created (`cubecl::cuda::QX_STREAMS`), so they stay
//! ordered with CubeCL's launches, uploads and syncs on that stream.
//!
//! Rules: the ring holds `SLOTS` launches; `Gpu::sync` (or any full stream sync) retires every slot, and the chunk
//! path syncs at least once per layer, so a layer must launch fewer than `SLOTS` kernels (a layer of the chunk path
//! launches a few thousand). Never during a CUDA graph capture: the captured copy would replay a stale slot.
use cudarc::driver::{result, sys};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Whether `Gpu::run` goes through the raw launcher (`QX_RAW_LAUNCH=1` and inside a chunk pass).
pub static RAW: AtomicBool = AtomicBool::new(false);

/// The info ring: slots of `WORDS` u64 each.
const SLOTS: usize = 1 << 16;
const WORDS: usize = 16;

struct Module {
    funcs: HashMap<&'static str, sys::CUfunction>,
    module: sys::CUmodule,
}

struct Launcher {
    stream: sys::CUstream,
    modules: HashMap<u64, Module>,
    host: *mut u64,
    dev: u64,
    next: usize,
    launches: u64,
}

// SAFETY: the launcher is used from the engine thread only (the mutex serializes any other caller); the raw
// pointers are a pinned host allocation and a device allocation it owns.
unsafe impl Send for Launcher {}

static LAUNCHER: Mutex<Option<Launcher>> = Mutex::new(None);

/// Whether the raw launcher is enabled for this process (`QX_RAW_LAUNCH=1`).
pub fn enabled() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_RAW_LAUNCH").is_ok_and(|v| v == "1"))
}

fn launcher() -> std::sync::MutexGuard<'static, Option<Launcher>> {
    let mut g = LAUNCHER.lock().unwrap();
    if g.is_none() {
        crate::mem::bind();
        let streams = cubecl::cuda::QX_STREAMS.lock().unwrap();
        let stream = *streams.first().expect("CubeCL has created no stream yet: launch something through it first") as sys::CUstream;
        drop(streams);
        let bytes = SLOTS * WORDS * 8;
        let host = unsafe { result::malloc_host(bytes, 0) }.expect("pinned info ring") as *mut u64;
        let dev = unsafe { result::malloc_sync(bytes) }.expect("device info ring");
        *g = Some(Launcher { stream, modules: HashMap::new(), host, dev, next: 0, launches: 0 });
    }
    g
}

/// Compiles and loads `module` (once), and launches `entry` with `grid` blocks of `threads` threads and `info`.
pub fn launch(module: &crate::kernel::Module, entry: &'static str, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
    assert!(info.len() <= WORDS, "raw launch of {entry}: {} info words, the ring slot holds {WORDS}", info.len());
    let mut g = launcher();
    let l = g.as_mut().unwrap();
    if !l.modules.contains_key(&module.source_hash) {
        if std::env::var("QX_DS_VERBOSE").is_ok() {
            eprintln!("raw: compiling {}.cu", module.name);
        }
        let opts = cudarc::nvrtc::CompileOptions { options: vec!["--gpu-architecture=compute_120a".into(), "-lineinfo".into()], ..Default::default() };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(module.source, opts).unwrap_or_else(|e| panic!("raw: kernels/{}.cu does not compile with NVRTC:\n{e}", module.name));
        let src = CString::new(ptx.to_src()).expect("PTX text");
        let m = unsafe { result::module::load_data(src.as_ptr() as *const c_void) }.unwrap_or_else(|e| panic!("raw: cuModuleLoadData of {}.cu: {e:?}", module.name));
        let mut funcs = HashMap::new();
        for &name in module.entries {
            let cname = CString::new(name).unwrap();
            let f = unsafe { result::module::get_function(m, cname) }.unwrap_or_else(|e| panic!("raw: {}.cu has no function {name}: {e:?}", module.name));
            funcs.insert(name, f);
        }
        l.modules.insert(module.source_hash, Module { funcs, module: m });
    }
    let f = *l.modules[&module.source_hash].funcs.get(entry).unwrap_or_else(|| panic!("raw: {}.cu has no entry {entry}", module.name));
    let slot = l.next;
    l.next = (l.next + 1) % SLOTS;
    l.launches += 1;
    let host = unsafe { std::slice::from_raw_parts_mut(l.host.add(slot * WORDS), WORDS) };
    host[..info.len()].copy_from_slice(info);
    let dev_slot = l.dev + (slot * WORDS * 8) as u64;
    unsafe {
        result::memcpy_htod_async(dev_slot, &host[..info.len()], l.stream).expect("raw: info copy");
        let mut arg = dev_slot;
        let mut params: [*mut c_void; 1] = [&mut arg as *mut u64 as *mut c_void];
        result::launch_kernel(f, grid, (threads, 1, 1), 0, l.stream, &mut params).unwrap_or_else(|e| panic!("raw: launch of {entry}: {e:?}"));
    }
}

/// A scope in which `Gpu::run` goes through the raw launcher (when `QX_RAW_LAUNCH=1`): the chunk pass. Dropping it
/// turns the raw path off again (the window path's graph captures must never see it).
pub struct Scope(bool);

impl Scope {
    pub fn enter() -> Self {
        let on = enabled();
        if on {
            RAW.store(true, Ordering::Relaxed);
        }
        Scope(on)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.0 {
            RAW.store(false, Ordering::Relaxed);
        }
    }
}

/// Launches so far through the raw path (for the chunk log).
pub fn launches() -> u64 {
    LAUNCHER.lock().unwrap().as_ref().map(|l| l.launches).unwrap_or(0)
}

impl Drop for Module {
    fn drop(&mut self) {
        // modules live for the process; nothing to unload before exit
        let _ = self.module;
    }
}
