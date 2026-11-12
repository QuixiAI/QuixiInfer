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
const WORDS: usize = 32;  // (the attention kernels take 17 info words)

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
    /// Side streams with an event each (`side_begin`), the event of the main stream they start behind, and the
    /// side stream the next launches go to (None: the main stream).
    side: Vec<(sys::CUstream, sys::CUevent)>,
    main_ev: Option<sys::CUevent>,
    cur: Option<usize>,
}

// SAFETY: the launcher is used from the engine thread only (the mutex serializes any other caller); the raw
// pointers are a pinned host allocation and a device allocation it owns.
unsafe impl Send for Launcher {}

static LAUNCHER: Mutex<Option<Launcher>> = Mutex::new(None);

/// Whether the raw launcher is enabled for this process: the default since 2026-11-08 (set D 31.470 -> 30.210 with it;
/// `QX_RAW_LAUNCH=0` sends the chunk path's launches through CubeCL again).
pub fn enabled() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_RAW_LAUNCH").is_ok_and(|v| v == "0"))
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
        *g = Some(Launcher { stream, modules: HashMap::new(), host, dev, next: 0, launches: 0, side: Vec::new(), main_ev: None, cur: None });
    }
    g
}

/// Compiles and loads `module` if it is not loaded yet.
fn ensure(l: &mut Launcher, module: &crate::kernel::Module) {
    if !l.modules.contains_key(&module.source_hash) {
        // CubeCL's context keeps one module per precompiled source (the vendored cubecl-cuda's `QX_MODULES`), and it
        // has usually loaded this one already: take it from there, or load it here and leave it there for CubeCL
        let shared = cubecl::cuda::qx_module_share();
        let have = if shared { cubecl::cuda::QX_MODULES.lock().unwrap().iter().find(|(h, _)| *h == module.source_hash).map(|&(_, m)| m) } else { None };
        let m = match have {
            Some(m) => m as sys::CUmodule,
            None => {
                // the same target CubeCL compiles with (the device's own, `sm_120a` on an RTX 50-series card; NVRTC
                // still hands back PTX)
                let arch = crate::mem::nvrtc_arch(false);
                let key = [arch.as_str(), "-lineinfo", "raw"];
                let t0 = std::time::Instant::now();
                let mut loaded = None;
                if let Some(ptx) = cubecl::cuda::qx_ptx_cached(module.source, &key) {
                    match unsafe { result::module::load_data(ptx.as_ptr() as *const c_void) } {
                        Ok(m) => loaded = Some((m, "PTX from the cache")),
                        Err(e) if e.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY => panic!("raw: cuModuleLoadData of {}.cu: out of device memory", module.name),
                        // PTX the driver refuses: drop the entry and compile again
                        Err(_) => cubecl::cuda::qx_ptx_forget(module.source, &key),
                    }
                }
                let (m, how) = loaded.unwrap_or_else(|| {
                    let opts = cudarc::nvrtc::CompileOptions { options: vec![arch.clone(), "-lineinfo".into()], ..Default::default() };
                    let ptx = cudarc::nvrtc::compile_ptx_with_opts(module.source, opts).unwrap_or_else(|e| panic!("raw: kernels/{}.cu does not compile with NVRTC:\n{e}", module.name));
                    let src = CString::new(ptx.to_src()).expect("PTX text");
                    let m = unsafe { result::module::load_data(src.as_ptr() as *const c_void) }.unwrap_or_else(|e| panic!("raw: cuModuleLoadData of {}.cu: {e:?}", module.name));
                    // stored after the load: PTX the driver rejects is not handed back on the next run
                    let chars: Vec<std::ffi::c_char> = src.as_bytes_with_nul().iter().map(|&b| b as std::ffi::c_char).collect();
                    cubecl::cuda::qx_ptx_store(module.source, &key, &chars);
                    (m, "compiled")
                });
                if std::env::var("QX_DS_VERBOSE").is_ok() {
                    eprintln!("raw: {}.cu {how}, loaded after {:.2} s", module.name, t0.elapsed().as_secs_f64());
                }
                if shared {
                    cubecl::cuda::QX_MODULES.lock().unwrap().push((module.source_hash, m as usize));
                }
                m
            }
        };
        let mut funcs = HashMap::new();
        for &name in module.entries {
            let cname = CString::new(name).unwrap();
            let f = unsafe { result::module::get_function(m, cname) }.unwrap_or_else(|e| panic!("raw: {}.cu has no function {name}: {e:?}", module.name));
            funcs.insert(name, f);
        }
        l.modules.insert(module.source_hash, Module { funcs, module: m });
    }
}

/// Loads `module` now (so a later launch from it allocates nothing): the chunk path loads its own modules before
/// cuBLAS takes memory for its handle and kernel libraries.
pub fn preload(module: &crate::kernel::Module) {
    let mut g = launcher();
    ensure(g.as_mut().unwrap(), module);
}

/// Compiles and loads `module` (once), and launches `entry` with `grid` blocks of `threads` threads and `info`.
pub fn launch(module: &crate::kernel::Module, entry: &'static str, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
    assert!(info.len() <= WORDS, "raw launch of {entry}: {} info words, the ring slot holds {WORDS}", info.len());
    let mut g = launcher();
    let l = g.as_mut().unwrap();
    ensure(l, module);
    let f = *l.modules[&module.source_hash].funcs.get(entry).unwrap_or_else(|| panic!("raw: {}.cu has no entry {entry}", module.name));
    let slot = l.next;
    l.next = (l.next + 1) % SLOTS;
    l.launches += 1;
    let host = unsafe { std::slice::from_raw_parts_mut(l.host.add(slot * WORDS), WORDS) };
    host[..info.len()].copy_from_slice(info);
    let dev_slot = l.dev + (slot * WORDS * 8) as u64;
    let stream = l.cur.map_or(l.stream, |i| l.side[i].0);
    unsafe {
        result::memcpy_htod_async(dev_slot, &host[..info.len()], stream).expect("raw: info copy");
        let mut arg = dev_slot;
        let mut params: [*mut c_void; 1] = [&mut arg as *mut u64 as *mut c_void];
        result::launch_kernel(f, grid, (threads, 1, 1), 0, stream, &mut params).unwrap_or_else(|e| panic!("raw: launch of {entry}: {e:?}"));
    }
}

/// Side streams for independent kernel sequences that should overlap on the GPU (the chunk path's experts: one
/// launch set per expert, each far too small to fill the card). `side_begin(n)`: `n` side streams exist and each
/// waits for everything launched on the main stream so far. `side_use(Some(i))`: the launches that follow go to side
/// stream `i` (`None`: the main stream again). `side_join(n)`: the main stream waits for everything launched on the
/// first `n` side streams so far, so a launch after it on the main stream runs after them.
pub fn side_begin(n: usize) {
    let mut g = launcher();
    let l = g.as_mut().unwrap();
    unsafe {
        while l.side.len() < n {
            let st = result::stream::create(result::stream::StreamKind::NonBlocking).expect("raw: side stream");
            let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("raw: side event");
            l.side.push((st, ev));
        }
        if l.main_ev.is_none() {
            l.main_ev = Some(result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("raw: main event"));
        }
        let me = l.main_ev.unwrap();
        result::event::record(me, l.stream).expect("raw: main event record");
        for &(st, _) in &l.side[..n] {
            result::stream::wait_event(st, me, sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT).expect("raw: side wait");
        }
    }
}

pub fn side_use(i: Option<usize>) {
    launcher().as_mut().unwrap().cur = i;
}

pub fn side_join(n: usize) {
    let mut g = launcher();
    let l = g.as_mut().unwrap();
    unsafe {
        for &(st, ev) in &l.side[..n.min(l.side.len())] {
            result::event::record(ev, st).expect("raw: side event record");
            result::stream::wait_event(l.stream, ev, sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT).expect("raw: main wait");
        }
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
