//! The vision tower on the GPU (`qx_vision::Vision` is the CPU form and the reference for this one).
//!
//! The matrix products are `qx_gemm16` (f16 operands, f32 accumulation on tensor cores), everything between them
//! the kernels of `kernels/vision.cu`. The tower runs on a thread and a stream of its own beside the engine's
//! windows: its launches go through the driver API with their info words in a ring of its own (as `qx_gpu::raw`
//! does for the chunk path), nothing is compiled or allocated after `GpuTower::new`, and the device's expert tier
//! gives up nothing for it: the layers' weights stay in pinned host memory and are copied into a staging area
//! layer by layer (26 MB a layer; the aligner's two matrices in parts of that size), the working buffers are
//! sized once for the largest image (about 330 MB in all), out of what the engine leaves free after its warm-up.
//!
//! Rules it keeps, because the engine shares the context:
//! - After `new`, no call that CUDA forbids beside another thread's global-mode graph capture (the engine captures
//!   window graphs lazily): every transfer is between pinned host memory and the device on the tower's stream,
//!   and the end of an encode is waited for by polling an event, never by a stream or context synchronize.
//! - The info ring is never reused within an encode (the largest image is about 12,000 launches, the ring holds 65,536,
//!   and every encode starts at slot 0 with the stream drained): a pinned ring's copy is asynchronous, so a slot
//!   rewritten before the stream reached it would hand a kernel another launch's words.
//! - Launches are enqueued in batches behind one copy of their info words (half the driver calls: the tower's
//!   thread and the engine's contend for the context).
use anyhow::{anyhow, bail, Result};
use cudarc::driver::{result, sys};
use qx_gpu::kernel::modules;
use qx_pack::gguf::Gguf;
use qx_vision::{Fitted, Vision};
use std::ffi::c_void;
use std::sync::Mutex;

/// Patches and LLM cells of the largest image (1,024 positions: at most 1,021 cells of 9 patches).
const NMAX: usize = 9216;
const CMAX: usize = 1024;
/// Rows per launch of the wide products, and the most query rows of an attention block.
const BLK: usize = 1024;
const QB: usize = 1024;
const SLOTS: usize = 1 << 16;
const WORDS: usize = 16;
/// The factor the attention's probabilities are stored with in f16 (see `qxv_softmax16`).
const PGAIN: f32 = 4096.0;
/// Bytes of an attention block's scores (all heads, f32) and of the partial sums of the sliced product of the
/// probabilities with the values.
const SC_BYTES: usize = 72 << 20;
const PARTS_BYTES: usize = 8 << 20;
/// Output rows of the aligner's matrices staged at a time.
const MM_PART: usize = 1280;

fn up32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

struct Fns {
    gemm: sys::CUfunction,
    gemm_sum: sys::CUfunction,
    rms: sys::CUfunction,
    qkv: sys::CUfunction,
    softmax: sys::CUfunction,
    add: sys::CUfunction,
    f16: sys::CUfunction,
    swiglu: sys::CUfunction,
    unfold: sys::CUfunction,
}

struct Inner {
    stream: sys::CUstream,
    event: sys::CUevent,
    f: Fns,
    ring_host: *mut u64,
    ring_dev: u64,
    next: usize,
    /// Launches written to the ring and not enqueued yet: (function, threads, grid, slot).
    pend: Vec<(sys::CUfunction, u32, (u32, u32, u32), usize)>,
    /// The matrices as the GGUF holds them (f16, `out` rows of `in`), pinned: per layer qkv, out, gate, up, down
    /// (offsets into `w_host`), then the aligner's two, then the patch embedding padded to `pk` columns.
    w_host: *mut u8,
    layer_off: Vec<[usize; 5]>,
    mm_off: [usize; 2],
    patch_off: usize,
    pk: usize,
    /// Pinned host buffers for an encode's inputs and output.
    patches_host: *mut u16,
    cs_host: *mut f32,
    out_host: *mut f32,
    hosts: Vec<*mut c_void>,
    /// The vectors (f32) on the device: per layer ln1, qkv bias, out bias, ln2; then patch bias, post norm, the
    /// aligner's two biases.
    layer_small: Vec<[u64; 4]>,
    tail_small: [u64; 4],
    stage: u64,
    // working buffers (several serve two purposes at different times: see `encode`)
    x: u64, h16: u64, q16: u64, k16: u64, vt16: u64, tmp: u64, tmp2: u64, g16: u64, o: u64, sc: u64, p16: u64, parts: u64, cs: u64,
    allocs: Vec<u64>,
    device_bytes: usize,
    dim: usize, heads: usize, ffn: usize, out_dim: usize, patch: usize, merge: usize, eps: f32, theta: f32, layers: usize,
}

// SAFETY: one encode at a time (the mutex); the raw pointers are allocations the tower owns for its life.
unsafe impl Send for Inner {}

pub struct GpuTower {
    inner: Mutex<Inner>,
}

impl Inner {
    /// Writes a launch's info words to the next ring slot; `flush` enqueues it.
    fn k(&mut self, f: sys::CUfunction, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
        assert!(info.len() <= WORDS && self.next < SLOTS, "vision: launch {} of an encode ({} info words)", self.next, info.len());
        let slot = self.next;
        self.next += 1;
        unsafe { std::slice::from_raw_parts_mut(self.ring_host.add(slot * WORDS), WORDS)[..info.len()].copy_from_slice(info) };
        self.pend.push((f, threads, grid, slot));
    }

    /// Enqueues the pending launches behind one copy of their info words.
    fn flush(&mut self) -> Result<()> {
        let Some(&(_, _, _, first)) = self.pend.first() else { return Ok(()) };
        let n = self.pend.len();
        unsafe {
            let host = std::slice::from_raw_parts(self.ring_host.add(first * WORDS), n * WORDS);
            result::memcpy_htod_async(self.ring_dev + (first * WORDS * 8) as u64, host, self.stream).map_err(|e| anyhow!("vision: info copy: {e:?}"))?;
            for (f, threads, grid, slot) in self.pend.drain(..) {
                let mut arg = self.ring_dev + (slot * WORDS * 8) as u64;
                let mut params: [*mut c_void; 1] = [&mut arg as *mut u64 as *mut c_void];
                result::launch_kernel(f, grid, (threads, 1, 1), 0, self.stream, &mut params).map_err(|e| anyhow!("vision: launch: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Y (t x n, rows `ldc` floats apart) = X16 (t x k) . W16^T (n rows of k).
    fn gemm(&mut self, x16: u64, w16: u64, y: u64, t: usize, n: usize, k: usize, ldc: usize) {
        let f = self.f.gemm;
        self.k(f, 256, ((n as u32).div_ceil(64), (t as u32).div_ceil(128), 1), &[x16, w16, y, t as u64, n as u64, k as u64, ldc as u64, 0]);
    }

    /// Copies `bytes` of the pinned matrices at `off` into the staging area at `at`, behind everything enqueued.
    fn stage_in(&mut self, off: usize, bytes: usize, at: usize) -> Result<u64> {
        self.flush()?;
        unsafe {
            let src = std::slice::from_raw_parts(self.w_host.add(off), bytes);
            result::memcpy_htod_async(self.stage + at as u64, src, self.stream).map_err(|e| anyhow!("vision: weights copy: {e:?}"))?;
        }
        Ok(self.stage + at as u64)
    }

    /// Waits for everything enqueued on the tower's stream, by polling an event (see the module's rules).
    fn wait(&mut self) -> Result<()> {
        self.flush()?;
        unsafe { result::event::record(self.event, self.stream) }.map_err(|e| anyhow!("vision: event: {e:?}"))?;
        let t0 = std::time::Instant::now();
        loop {
            match unsafe { result::event::query(self.event) } {
                Ok(()) => return Ok(()),
                Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {}
                Err(e) => bail!("vision: the tower's stream failed: {e:?}"),
            }
            if t0.elapsed().as_secs() > 120 {
                bail!("vision: the tower's stream did not finish in 120 s");
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

impl GpuTower {
    /// Loads the kernels, pins the matrices and allocates every buffer. Err (nothing kept) when the projector is
    /// not what this tower takes, or the device has no room for it beside `headroom` bytes left free for the
    /// engine: the caller then keeps the CPU tower. Call once the engine has warmed up and is idle.
    pub fn new(mmproj: &std::path::Path, v: &Vision, headroom: usize) -> Result<Self> {
        qx_gpu::mem::bind();
        let g = Gguf::open(mmproj)?;
        let (heads, ffn, eps, theta, layers) = v.shape();
        let (d, e, r) = (v.dim, v.out_dim, v.merge);
        let cin = d * r * r;
        if d % heads != 0 || d / heads > 64 || (d / heads) % 4 != 0 || ffn > 3 * d || e % MM_PART != 0 || e > 8 * MM_PART {
            bail!("the GPU tower's buffers are laid out for a head width of at most 64, a feed-forward width of at most 3 x the width and an output width in multiples of {MM_PART}");
        }
        // everything read and checked before anything is allocated
        let raw = |name: &str, want: usize| -> Result<&[u8]> {
            let t = g.tensor(name).ok_or_else(|| anyhow!("no tensor {name}"))?;
            if t.ty != qx_pack::ggml::GgmlType::F16 || t.elements() as usize != want {
                bail!("{name}: {:?} with {} elements; the GPU tower takes F16 matrices ({want} here)", t.ty, t.elements());
            }
            Ok(g.data(t))
        };
        let f32s = |name: &str, want: usize| -> Result<Vec<f32>> {
            let t = g.tensor(name).ok_or_else(|| anyhow!("no tensor {name}"))?;
            if t.ty != qx_pack::ggml::GgmlType::F32 || t.elements() as usize != want {
                bail!("{name}: {:?} with {} elements, not {want} F32", t.ty, t.elements());
            }
            Ok(g.data(t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let per_layer = [("attn_qkv.weight", 3 * d * d), ("attn_out.weight", d * d), ("ffn_gate.weight", ffn * d), ("ffn_up.weight", ffn * d), ("ffn_down.weight", d * ffn)];
        let pin_in = 3 * v.patch * v.patch;
        let pk = up32(pin_in);
        let mut mats: Vec<&[u8]> = Vec::new();
        for l in 0..layers {
            for (n, want) in &per_layer {
                mats.push(raw(&format!("v.blk.{l}.{n}"), *want)?);
            }
        }
        mats.push(raw("mm.1.weight", e * cin)?);
        mats.push(raw("mm.2.weight", e * e)?);
        let pw = raw("v.patch_embd.weight", d * pin_in)?;
        let mut small_host: Vec<f32> = Vec::new();
        let mut layer_small_off = Vec::new();
        for l in 0..layers {
            let mut o4 = [0usize; 4];
            for (i, (n, want)) in [("ln1.weight", d), ("attn_qkv.bias", 3 * d), ("attn_out.bias", d), ("ln2.weight", d)].iter().enumerate() {
                o4[i] = small_host.len();
                small_host.extend(f32s(&format!("v.blk.{l}.{n}"), *want)?);
            }
            layer_small_off.push(o4);
        }
        let mut tail_off = [0usize; 4];
        for (i, (n, want)) in [("v.patch_embd.bias", d), ("v.post_ln.weight", d), ("mm.1.bias", e), ("mm.2.bias", e)].iter().enumerate() {
            tail_off[i] = small_host.len();
            small_host.extend(f32s(n, *want)?);
        }
        // allocations; `undo` frees whatever exists
        let mut allocs: Vec<u64> = Vec::new();
        let mut hosts: Vec<*mut c_void> = Vec::new();
        let undo = |allocs: &Vec<u64>, hosts: &Vec<*mut c_void>, what: String| -> anyhow::Error {
            unsafe {
                for &a in allocs {
                    let _ = result::free_sync(a);
                }
                for &h in hosts {
                    let _ = result::free_host(h);
                }
            }
            anyhow!(what)
        };
        macro_rules! host {
            ($bytes:expr) => {
                match unsafe { result::malloc_host($bytes, 0) } {
                    Ok(p) => {
                        hosts.push(p);
                        p
                    }
                    Err(er) => return Err(undo(&allocs, &hosts, format!("{} bytes of pinned memory for the vision tower: {er:?}", $bytes))),
                }
            };
        }
        let mut device_bytes = 0usize;
        macro_rules! dev {
            ($bytes:expr) => {
                match unsafe { result::malloc_sync($bytes) } {
                    Ok(p) => {
                        allocs.push(p);
                        device_bytes += $bytes;
                        p
                    }
                    Err(er) => return Err(undo(&allocs, &hosts, format!("{} bytes of device memory for the vision tower: {er:?}", $bytes))),
                }
            };
        }
        let total: usize = mats.iter().map(|m| m.len()).sum::<usize>() + d * pk * 2;
        let w_host = host!(total) as *mut u8;
        let wh = unsafe { std::slice::from_raw_parts_mut(w_host, total) };
        let mut at = 0usize;
        let mut offs: Vec<usize> = Vec::new();
        for m in &mats {
            wh[at..at + m.len()].copy_from_slice(m);
            offs.push(at);
            at += m.len();
        }
        let layer_off: Vec<[usize; 5]> = (0..layers).map(|l| [offs[l * 5], offs[l * 5 + 1], offs[l * 5 + 2], offs[l * 5 + 3], offs[l * 5 + 4]]).collect();
        let mm_off = [offs[layers * 5], offs[layers * 5 + 1]];
        // the patch embedding with its rows padded to a multiple of 32 columns (the product's K)
        let patch_off = at;
        wh[at..].fill(0);
        for row in 0..d {
            wh[at + row * pk * 2..at + row * pk * 2 + pin_in * 2].copy_from_slice(&pw[row * pin_in * 2..(row + 1) * pin_in * 2]);
        }
        let ring_host = host!(SLOTS * WORDS * 8) as *mut u64;
        let patches_host = host!(NMAX * pk * 2) as *mut u16;
        let cs_host = host!(NMAX * (d / heads) * 4) as *mut f32;
        let out_host = host!(CMAX * e * 4) as *mut f32;
        let np = up32(NMAX);
        let layer_bytes = (3 * d * d + d * d + 3 * ffn * d) * 2;
        let ring_dev = dev!(SLOTS * WORDS * 8);
        let stage = dev!(layer_bytes.max(MM_PART * cin * 2).max(2 * MM_PART * e * 2).max(d * pk * 2));
        let x = dev!((NMAX * d * 4).max(CMAX * e * 4));
        let h16 = dev!(NMAX * d * 2);
        let q16 = dev!((NMAX * d * 2).max(NMAX * pk * 2).max(CMAX * e * 2));
        let k16 = dev!((NMAX * d * 2).max(CMAX * cin * 2));
        let vt16 = dev!(d * np * 2);
        let tmp = dev!(BLK * 3 * d * 4);
        let tmp2 = dev!(BLK * ffn * 4);
        let g16 = dev!(BLK * ffn * 2);
        let o = dev!((NMAX * d * 4).max(CMAX * e * 4));
        let sc = dev!(SC_BYTES);
        let p16 = dev!(SC_BYTES / 2);
        let parts = dev!(PARTS_BYTES);
        let cs = dev!(NMAX * (d / heads) * 4);
        let small = dev!(small_host.len() * 4);
        if let Err(er) = unsafe { result::memcpy_htod_sync(small, &small_host) } {
            return Err(undo(&allocs, &hosts, format!("vision: vectors upload: {er:?}")));
        }
        let free = qx_gpu::mem::mem_info().0;
        if free < headroom {
            return Err(undo(&allocs, &hosts, format!("with the tower's {:.0} MB the device would have {:.0} MB free, under the {:.0} MB kept for the engine (QX_VISION_HEADROOM_MIB)", device_bytes as f64 / 1048576.0, free as f64 / 1048576.0, headroom as f64 / 1048576.0)));
        }
        // a stream of the lowest priority: the engine's windows go first
        let stream = unsafe {
            let (mut lo, mut hi) = (0i32, 0i32);
            let _ = sys::cuCtxGetStreamPriorityRange(&mut lo, &mut hi);
            result::stream::create_with_priority(result::stream::StreamKind::NonBlocking, lo)
        };
        let stream = match stream {
            Ok(s) => s,
            Err(er) => return Err(undo(&allocs, &hosts, format!("vision: stream: {er:?}"))),
        };
        let event = match result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING) {
            Ok(ev) => ev,
            Err(er) => return Err(undo(&allocs, &hosts, format!("vision: event: {er:?}"))),
        };
        let fun = |m: &qx_gpu::kernel::Module, e: &'static str| qx_gpu::raw::function(m, e) as sys::CUfunction;
        let f = Fns {
            gemm: fun(&modules::GEMM, "qx_gemm16"),
            gemm_sum: fun(&modules::GEMM, "qx_gemm16_sum"),
            rms: fun(&modules::VISION, "qxv_rms16"),
            qkv: fun(&modules::VISION, "qxv_qkv"),
            softmax: fun(&modules::VISION, "qxv_softmax16"),
            add: fun(&modules::VISION, "qxv_add"),
            f16: fun(&modules::VISION, "qxv_f16"),
            swiglu: fun(&modules::VISION, "qxv_swiglu16"),
            unfold: fun(&modules::VISION, "qxv_unfold16"),
        };
        let at = |o: usize| small + (o * 4) as u64;
        Ok(Self {
            inner: Mutex::new(Inner {
                stream, event, f, ring_host, ring_dev, next: 0, pend: Vec::new(), w_host, layer_off, mm_off, patch_off, pk,
                patches_host, cs_host, out_host, hosts,
                layer_small: layer_small_off.iter().map(|o| [at(o[0]), at(o[1]), at(o[2]), at(o[3])]).collect(),
                tail_small: [at(tail_off[0]), at(tail_off[1]), at(tail_off[2]), at(tail_off[3])],
                stage, x, h16, q16, k16, vt16, tmp, tmp2, g16, o, sc, p16, parts, cs, allocs, device_bytes,
                dim: d, heads, ffn, out_dim: e, patch: v.patch, merge: r, eps, theta, layers,
            }),
        })
    }

    /// Device bytes the tower holds.
    pub fn device_bytes(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).device_bytes
    }

    /// The block of a fitted image, as `Vision::encode` returns it: (rows of `out_dim` floats, grid rows, columns).
    /// Err when the image is beyond the buffers, a driver call fails or a value of the block is not finite (an
    /// activation beyond f16's range): the caller's CPU tower takes the image.
    pub fn encode(&self, v: &Vision, im: &Fitted) -> Result<(Vec<f32>, usize, usize)> {
        qx_gpu::mem::bind();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        let res = Self::run(s, im);
        if res.is_err() {
            // whatever was enqueued finishes before the buffers are used again
            s.pend.clear();
            let _ = s.wait();
        }
        let (gh, gw, c) = res?;
        let e = s.out_dim;
        let feat = unsafe { std::slice::from_raw_parts(s.out_host, c * e) };
        if !feat.iter().all(|x| x.is_finite()) {
            bail!("the GPU tower's block has values that are not finite");
        }
        let (start, end, newline) = v.delimiters();
        let mut block = Vec::with_capacity(Vision::block_tokens(gh, gw) * e);
        block.extend_from_slice(start);
        for y in 0..gh {
            block.extend_from_slice(&feat[y * gw * e..(y + 1) * gw * e]);
            block.extend_from_slice(newline);
        }
        block.extend_from_slice(end);
        Ok((block, gh, gw))
    }

    /// Enqueues and waits for one image; the aligner's rows are in `out_host` afterwards. Returns (grid rows,
    /// columns, cells).
    fn run(s: &mut Inner, im: &Fitted) -> Result<(usize, usize, usize)> {
        let (p, d, heads, ffn, e, r) = (s.patch, s.dim, s.heads, s.ffn, s.out_dim, s.merge);
        let (nh, nw) = (im.h / p, im.w / p);
        let n = nh * nw;
        let (gh, gw) = (nh.div_ceil(r), nw.div_ceil(r));
        let c = gh * gw;
        if n == 0 || n > NMAX || c > CMAX || im.rgb.len() != im.w * im.h * 3 {
            bail!("an image of {n} patches is beyond the GPU tower's buffers");
        }
        let (hd, np, pk) = (d / heads, up32(n), s.pk);
        // host side, into pinned memory: the patches as f16 (pixels as the reference feeds them, rounded to bf16
        // first) and the rotary angles
        let px = |b: u8| -> u16 {
            let x = (b as f32 / 255.0 - 0.5) / 0.5;
            let bits = x.to_bits();
            half::f16::from_f32(f32::from_bits(bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)).to_bits()
        };
        let patches = unsafe { std::slice::from_raw_parts_mut(s.patches_host, n * pk) };
        for (i, dst) in patches.chunks_exact_mut(pk).enumerate() {
            let (gy, gx) = (i / nw, i % nw);
            for ch in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        dst[(ch * p + y) * p + x] = px(im.rgb[((gy * p + y) * im.w + gx * p + x) * 3 + ch]);
                    }
                }
            }
            dst[3 * p * p..].fill(0);
        }
        let nf = hd / 4;
        let inv: Vec<f32> = (0..nf).map(|j| 1.0 / s.theta.powf(2.0 * j as f32 / (2 * nf) as f32)).collect();
        let cs = unsafe { std::slice::from_raw_parts_mut(s.cs_host, n * hd) };
        for i in 0..n {
            let (row, col) = ((i / nw) as f32, (i % nw) as f32);
            for j in 0..nf {
                let (a, b) = (row * inv[j], col * inv[j]);
                cs[(i * 2 * nf + j) * 2] = a.cos();
                cs[(i * 2 * nf + j) * 2 + 1] = a.sin();
                cs[(i * 2 * nf + nf + j) * 2] = b.cos();
                cs[(i * 2 * nf + nf + j) * 2 + 1] = b.sin();
            }
        }
        // QX_VISION_PROF=1: the stream is waited for after each part and the parts' times printed (slower overall)
        let profiling = std::env::var("QX_VISION_PROF").is_ok_and(|v| v == "1");
        let mut times = [0f64; 5];
        let mut last = std::time::Instant::now();
        let mut prof = |s: &mut Inner, which: usize| -> Result<()> {
            if profiling {
                s.wait()?;
                times[which] += last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
            }
            Ok(())
        };
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = s.eps.to_bits() as u64;
        let st = s.stream;
        // (the stream is drained: every earlier encode ended in `wait`)
        s.next = 0;
        let (x, h16, q16, k16, vt16, tmp, tmp2, g16, o, scb, p16, parts) = (s.x, s.h16, s.q16, s.k16, s.vt16, s.tmp, s.tmp2, s.g16, s.o, s.sc, s.p16, s.parts);
        let (fadd, frms, fqkv, fsm, ff16, fsw, fsum, fgemm, fun) = (s.f.add, s.f.rms, s.f.qkv, s.f.softmax, s.f.f16, s.f.swiglu, s.f.gemm_sum, s.f.gemm, s.f.unfold);
        unsafe {
            result::memcpy_htod_async(q16, &*patches, st).map_err(|er| anyhow!("vision: patches: {er:?}"))?;
            result::memcpy_htod_async(s.cs, &*cs, st).map_err(|er| anyhow!("vision: angles: {er:?}"))?;
            sys::cuMemsetD8Async(vt16, 0, d * up32(NMAX) * 2, st).result().map_err(|er| anyhow!("vision: memset: {er:?}"))?;
        }
        // patch embedding (the patches sit in q16's region until the first layer writes q there)
        let w = s.stage_in(s.patch_off, d * pk * 2, 0)?;
        s.gemm(q16, w, x, n, d, pk, d);
        let b = s.tail_small[0];
        s.k(fadd, 256, (((n * d) as u32).div_ceil(256), 1, 1), &[x, 0, b, n as u64, d as u64]);
        for l in 0..s.layers {
            let lo = s.layer_off[l];
            let sizes = [3 * d * d * 2, d * d * 2, ffn * d * 2, ffn * d * 2, d * ffn * 2];
            let mut at = 0usize;
            let mut wp = [0u64; 5];
            for i in 0..5 {
                wp[i] = s.stage_in(lo[i], sizes[i], at)?;
                at += sizes[i];
            }
            let [ln1, qkv_b, out_b, ln2] = s.layer_small[l];
            // attention
            s.k(frms, 256, (n as u32, 1, 1), &[x, ln1, h16, d as u64, eps]);
            for r0 in (0..n).step_by(BLK) {
                let rows = BLK.min(n - r0);
                s.gemm(h16 + (r0 * d * 2) as u64, wp[0], tmp, rows, 3 * d, d, 3 * d);
                s.k(fqkv, 256, (rows as u32, 1, 1), &[tmp, qkv_b, q16, k16, vt16, d as u64, heads as u64, s.cs, r0 as u64, n as u64, np as u64]);
            }
            prof(s, 3)?;
            // per block of query rows, every head in one launch each (heads as the product's groups): the scores,
            // their softmax, and the probabilities' product with the values, K in slices (64 output columns alone
            // leave the card idle)
            let qb = (SC_BYTES / (heads * np * 4) / 128 * 128).clamp(128, QB);
            for q0 in (0..n).step_by(qb) {
                let rows = qb.min(n - q0);
                let (dq, dp) = (n * hd * 2, rows * np * 2);
                s.k(fgemm, 256, ((n as u32).div_ceil(64), (rows as u32).div_ceil(128), heads as u32), &[q16 + (q0 * hd * 2) as u64, k16, scb, rows as u64, n as u64, hd as u64, np as u64, hd as u64, hd as u64, 1, hd as u64, dq as u64, dq as u64, (rows * np * 4) as u64]);
                prof(s, 0)?;
                s.k(fsm, 256, ((heads * rows) as u32, 1, 1), &[scb, p16, n as u64, np as u64, scale.to_bits() as u64, PGAIN.to_bits() as u64]);
                prof(s, 1)?;
                let blocks = rows.div_ceil(128) * heads;
                let mut sl = ((320 + blocks / 2) / blocks).clamp(1, (np / 256).max(1));
                while sl > 1 && heads * sl * rows * hd * 4 > PARTS_BYTES {
                    sl -= 1;
                }
                let y = o + (q0 * d * 4) as u64;
                let grid = (1, (rows as u32).div_ceil(128), (heads * sl) as u32);
                if sl == 1 {
                    s.k(fgemm, 256, grid, &[p16, vt16, y, rows as u64, hd as u64, np as u64, d as u64, np as u64, np as u64, 1, np as u64, dp as u64, (hd * np * 2) as u64, (hd * 4) as u64]);
                } else {
                    let ks = np.div_ceil(sl).div_ceil(32) * 32;
                    s.k(fgemm, 256, grid, &[p16, vt16, parts, rows as u64, hd as u64, np as u64, hd as u64, np as u64, np as u64, sl as u64, ks as u64, dp as u64, (hd * np * 2) as u64, (rows * hd * 4) as u64]);
                    s.k(fsum, 256, (((heads * rows * hd) as u32).div_ceil(256), 1, 1), &[parts, y, rows as u64, hd as u64, d as u64, sl as u64, heads as u64, (hd * 4) as u64, 0, hd as u64]);
                }
                prof(s, 2)?;
            }
            // the attention's output in f16 (into h16, free since the fused projection), its projection, the residual
            s.k(ff16, 256, (((n * d) as u32).div_ceil(256), 1, 1), &[o, h16, n as u64, d as u64, 0, 0, (1.0 / PGAIN).to_bits() as u64]);
            for r0 in (0..n).step_by(BLK) {
                let rows = BLK.min(n - r0);
                s.gemm(h16 + (r0 * d * 2) as u64, wp[1], tmp, rows, d, d, d);
                s.k(fadd, 256, (((rows * d) as u32).div_ceil(256), 1, 1), &[x + (r0 * d * 4) as u64, tmp, out_b, rows as u64, d as u64]);
            }
            // feed-forward
            s.k(frms, 256, (n as u32, 1, 1), &[x, ln2, h16, d as u64, eps]);
            for r0 in (0..n).step_by(BLK) {
                let rows = BLK.min(n - r0);
                let hb = h16 + (r0 * d * 2) as u64;
                s.gemm(hb, wp[2], tmp, rows, ffn, d, ffn);
                s.gemm(hb, wp[3], tmp2, rows, ffn, d, ffn);
                s.k(fsw, 256, (((rows * ffn) as u32).div_ceil(256), 1, 1), &[tmp, tmp2, g16, (rows * ffn) as u64]);
                s.gemm(g16, wp[4], tmp, rows, d, ffn, d);
                s.k(fadd, 256, (((rows * d) as u32).div_ceil(256), 1, 1), &[x + (r0 * d * 4) as u64, tmp, 0, rows as u64, d as u64]);
            }
            prof(s, 4)?;
        }
        // the last norm and the aligner: the cells in k16's region, the first product in o's, its activation in
        // q16's, the output in x's
        let [_, post_ln, mm1_b, mm2_b] = s.tail_small;
        s.k(frms, 256, (n as u32, 1, 1), &[x, post_ln, h16, d as u64, eps]);
        let cin = d * r * r;
        s.k(fun, 256, (((c * cin) as u32).div_ceil(256), 1, 1), &[h16, k16, d as u64, r as u64, nh as u64, nw as u64, gw as u64, c as u64]);
        for part in 0..e / MM_PART {
            let w1 = s.stage_in(s.mm_off[0] + part * MM_PART * cin * 2, MM_PART * cin * 2, 0)?;
            s.gemm(k16, w1, o + (part * MM_PART * 4) as u64, c, MM_PART, cin, e);
        }
        s.k(ff16, 256, (((c * e) as u32).div_ceil(256), 1, 1), &[o, q16, c as u64, e as u64, mm1_b, 1, 1f32.to_bits() as u64]);
        for part in 0..e / (2 * MM_PART) {
            let w2 = s.stage_in(s.mm_off[1] + part * 2 * MM_PART * e * 2, 2 * MM_PART * e * 2, 0)?;
            s.gemm(q16, w2, x + (part * 2 * MM_PART * 4) as u64, c, 2 * MM_PART, e, e);
        }
        s.k(fadd, 256, (((c * e) as u32).div_ceil(256), 1, 1), &[x, 0, mm2_b, c as u64, e as u64]);
        s.flush()?;
        unsafe {
            let out = std::slice::from_raw_parts_mut(s.out_host, c * e);
            result::memcpy_dtoh_async(out, x, st).map_err(|er| anyhow!("vision: download: {er:?}"))?;
        }
        s.wait()?;
        if profiling {
            eprintln!("vision profile ({n} patches, {} launches): scores {:.3} s, softmax {:.3}, p.v {:.3}, norm and fused projection {:.3}, output projection and feed-forward {:.3}", s.next, times[0], times[1], times[2], times[3], times[4]);
        }
        Ok((gh, gw, c))
    }
}

impl Drop for GpuTower {
    fn drop(&mut self) {
        let mut i = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        i.pend.clear();
        let _ = i.wait();
        unsafe {
            for &a in &i.allocs {
                let _ = result::free_sync(a);
            }
            for &h in &i.hosts {
                let _ = result::free_host(h);
            }
        }
    }
}
