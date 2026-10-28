//! Typed launchers for the kernels in `kernels/`, one function per entry point family.
use cubecl::prelude::CubeDim;
use qx_gpu::kernel::modules;
use qx_gpu::Gpu;
use qx_pack::ggml::GgmlType;

fn suffix(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "f32",
        GgmlType::BF16 => "bf16",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q6_K => "q6_k",
        GgmlType::IQ4_NL => "iq4_nl",
        GgmlType::IQ3_S => "iq3_s",
        GgmlType::IQ2_S => "iq2_s",
        GgmlType::NVFP4 => "nvfp4",
        t => panic!("no GPU kernels for {t:?}"),
    }
}

fn entry(prefix: &str, t: GgmlType) -> &'static str {
    let name = format!("{prefix}_{}", suffix(t));
    *modules::GEMV.entries.iter().find(|e| **e == name).unwrap_or_else(|| panic!("no entry {name}"))
}

pub const GEMV_STORE: u64 = 0;
pub const GEMV_ADD: u64 = 1;

/// `y[r] = W[r] . x` (or `+=` with `GEMV_ADD`) for `rows` rows of `cols`, `w` the first row's address.
pub fn gemv(gpu: &Gpu, t: GgmlType, w: u64, x: u64, y: u64, rows: usize, cols: usize, mode: u64) {
    gemv_at(gpu, t, w, x, y, rows, cols, mode, 0, 0);
}

/// `gemv` with `y` moved by `*pos * y_stride` floats, `pos` a device u32 address (0: no move).
#[allow(clippy::too_many_arguments)]
pub fn gemv_at(gpu: &Gpu, t: GgmlType, w: u64, x: u64, y: u64, rows: usize, cols: usize, mode: u64, pos: u64, y_stride: usize) {
    let warps_per_block = 8u32;
    let grid = (rows as u32).div_ceil(warps_per_block);
    gpu.run(modules::GEMV.kernel(entry("qx_gemv", t), CubeDim::new_1d(32 * warps_per_block)), (grid, 1, 1), &[w, x, y, rows as u64, cols as u64, mode, 0, 0, pos, y_stride as u64]);
}

/// Decodes `rows` rows starting at `row0` (plus `*row_ptr` when `row_ptr` is a device u32 address) to f32.
#[allow(clippy::too_many_arguments)]
pub fn dequant(gpu: &Gpu, t: GgmlType, w: u64, out: u64, rows: usize, cols: usize, row0: usize, row_ptr: u64) {
    let grid = (rows as u32).div_ceil(8);
    gpu.run(modules::GEMV.kernel(entry("qx_dequant", t), CubeDim::new_1d(256)), (grid, 1, 1), &[w, out, rows as u64, cols as u64, row0 as u64, row_ptr, 0]);
}

/// Decodes one row per token: row `u32[tok + 4t]` of W to out + t * cols, for t < `rows` (an embedding lookup for
/// a window).
pub fn dequant_gather(gpu: &Gpu, t: GgmlType, w: u64, out: u64, rows: usize, cols: usize, tok: u64) {
    let grid = (rows as u32).div_ceil(8);
    gpu.run(modules::GEMV.kernel(entry("qx_dequant", t), CubeDim::new_1d(256)), (grid, 1, 1), &[w, out, rows as u64, cols as u64, 0, tok, 1]);
}

/// One matrix of a `qx_mgemv` launch: `y = W . x`. `qx` is the kernel type id (`qx_type` or a planar id),
/// `stride` the row stride in bytes (0: dense).
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub w: u64,
    pub y: u64,
    pub rows: usize,
    pub cols: usize,
    pub qx: u64,
    pub stride: usize,
}

pub fn qx_type(t: GgmlType) -> u64 {
    match t {
        GgmlType::F32 => 0,
        GgmlType::BF16 => 1,
        GgmlType::Q8_0 => 2,
        GgmlType::Q6_K => 3,
        GgmlType::IQ4_NL => 4,
        GgmlType::IQ3_S => 5,
        GgmlType::IQ2_S => 6,
        GgmlType::F16 => 7,
        GgmlType::NVFP4 => 10,
        t => panic!("no GPU decoder for {t:?}"),
    }
}

/// A fixed group of matrices sharing an input, with its descriptor tables already in device memory: one for
/// `qx_mgemv` (f32, up to 8 tokens a launch) and one for `qx_tmv` (bf16 tensor cores, up to 16 tokens).
pub struct MGemv {
    pub table: qx_gpu::mem::DeviceBuf,
    pub n: usize,
    pub blocks: u32,
    /// `table`'s entries for the segments that are not planar Q8_0 (entries, blocks, segments): beside the int8
    /// `qx_tmv`, these run in f32, each token with the bits of its one-token product.
    pub rest: (qx_gpu::mem::DeviceBuf, u32, usize),
    /// `qx_tmv`'s tables (entries, blocks, segments): every segment (bf16); the planar Q8_0 ones (int8 entry).
    pub ttable: [(qx_gpu::mem::DeviceBuf, u32, usize); 2],
    /// The input length every segment shares.
    pub tcols: usize,
    /// A segment is planar Q8_0: its rows take int8 inputs (`qx_mgemv_q8x`).
    pub q8: bool,
    /// `qx_tmv`'s split-K partial sums and tile counters.
    _tscratch: (qx_gpu::mem::DeviceBuf, qx_gpu::mem::DeviceBuf),
}

/// Whether a window of `t` tokens runs its products on tensor cores (`qx_tmv`) rather than in f32 (`qx_mgemv`).
/// Default: above 8 tokens (batched decode), where `qx_tmv` reads each weight once instead of once per 8 tokens;
/// up to 8 (one sequence, speculative windows, small batches) `qx_mgemv`, which is faster per token. The two
/// round differently, so a sequence in a window of more than 8 tokens gets slightly different logits than
/// alone. `QX_TMV=1`: `qx_tmv` for every width (each token's bits independent of its window); `QX_TMV=0`: never.
pub fn use_tmv(t: usize) -> bool {
    static U: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    match *U.get_or_init(|| std::env::var("QX_TMV").ok().map(|v| v != "0")) {
        Some(always) => always,
        None => t > 8 || (t > 4 && TMV_BATCHED.load(std::sync::atomic::Ordering::Relaxed)),
    }
}

/// Whether planar Q8_0 rows take int8 inputs (`QX_MGEMV_INT8=0`: the f32 path).
pub fn mgemv_int8() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_MGEMV_INT8").is_ok_and(|v| v == "0"))
}

/// The int8 inputs `qx_mgemv_q8x` writes (8 tokens of up to 16384 columns, and their scales), one buffer for
/// every launch: they run in order on one stream.
fn q8x_scratch(k: usize) -> (u64, u64) {
    static B: std::sync::OnceLock<(qx_gpu::mem::DeviceBuf, qx_gpu::mem::DeviceBuf)> = std::sync::OnceLock::new();
    let b = B.get_or_init(|| (qx_gpu::mem::DeviceBuf::new(8 * 16384), qx_gpu::mem::DeviceBuf::new(8 * 16384 / 32 * 4)));
    assert!(8 * k <= b.0.bytes, "int8 matvec input of {k} columns");
    (b.0.ptr, b.1.ptr)
}

/// Whether `qx_tmv` gives planar Q8_0 rows int8 inputs and integer products (`QX_TMV_INT8=0`: bf16).
fn tmv_int8() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_TMV_INT8").map_or(true, |v| v != "0"))
}

/// `qx_tmv`'s int8 inputs and their scales: 32 tokens of up to 16384 columns.
fn tmv_q8_scratch(k: usize) -> (u64, u64) {
    static B: std::sync::OnceLock<(qx_gpu::mem::DeviceBuf, qx_gpu::mem::DeviceBuf)> = std::sync::OnceLock::new();
    let b = B.get_or_init(|| (qx_gpu::mem::DeviceBuf::new(32 * 16384), qx_gpu::mem::DeviceBuf::new(32 * 16384 / 32 * 4)));
    assert!(32 * k <= b.0.bytes, "qx_tmv int8 input of {k} columns");
    (b.0.ptr, b.1.ptr)
}

/// The bf16 input rows `qx_tmv_prep` writes for `qx_tmv` (32 tokens of up to `k` columns), one buffer for every
/// launch: they run in order on one stream.
fn tmv_scratch(k: usize) -> u64 {
    static B: std::sync::OnceLock<qx_gpu::mem::DeviceBuf> = std::sync::OnceLock::new();
    let b = B.get_or_init(|| qx_gpu::mem::DeviceBuf::new(32 * 16384 * 2));
    assert!(32 * k * 2 <= b.bytes, "qx_tmv input of {k} columns");
    b.ptr
}

/// Set while launching (or capturing) a window of several sequences: those take `qx_tmv` from 5 tokens on, where
/// it is the faster; a speculative window of one sequence keeps `qx_mgemv` up to 8, so its tokens get exactly the
/// bits of one-token steps.
pub static TMV_BATCHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl MGemv {
    /// Picks each segment's work shape: split-K for matrices with few rows, fewer lanes per row for short rows.
    pub fn new(segs: &[Seg]) -> Self {
        let mut words = Vec::new();
        let mut blocks = 0u32;
        // the same entries for the segments that are not planar Q8_0 alone: `qx_tmv`'s int8 mode runs them here
        let (mut rwords, mut rblocks) = (Vec::new(), 0u32);
        for s in segs {
            // Row-per-warp needs enough blocks to fill 170 SMs several times; otherwise split each row's columns
            // across a block's warps.
            let lanes: u64 = if s.cols <= 512 {
                16
            } else if s.rows / 8 < 680 {
                0
            } else {
                32
            };
            let nb = if lanes == 0 { s.rows as u32 } else { (s.rows as u32).div_ceil(8 * (32 / lanes as u32)) };
            words.extend_from_slice(&[s.w, s.y, s.rows as u64, s.cols as u64, s.qx, lanes, blocks as u64, s.stride as u64]);
            blocks += nb;
            if s.qx != QX_Q8_0_P {
                rwords.extend_from_slice(&[s.w, s.y, s.rows as u64, s.cols as u64, s.qx, lanes, rblocks as u64, s.stride as u64]);
                rblocks += nb;
            }
        }
        // qx_tmv: warps per 16-row tile (1 to 8) and then blocks per tile (k split) until the launch has about 16
        // warps per SM, every warp keeping a 128-column chunk at least
        const TARGET: usize = 170 * 16;
        let plan: Vec<(usize, usize, usize)> = segs
            .iter()
            .map(|s| {
                assert!(s.cols % 32 == 0, "qx_tmv needs whole 32-column steps ({} columns)", s.cols);
                let (tiles, chunks) = (s.rows.div_ceil(16), (s.cols / 128).max(1));
                let mut ws = 1;
                while ws < 8 && tiles * ws < TARGET && ws * 2 <= chunks {
                    ws *= 2;
                }
                let ks = if ws == 8 { TARGET.div_ceil(tiles * 8).min(chunks / 8).max(1) } else { 1 };
                (tiles, ws, ks)
            })
            .collect();
        let (n_part, n_ctr): (usize, usize) = plan.iter().map(|&(t, _, k)| if k > 1 { (t * k * 512, t) } else { (0, 0) }).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let partials = qx_gpu::mem::DeviceBuf::new(n_part.max(1) * 4);
        let counters = qx_gpu::mem::DeviceBuf::new(n_ctr.max(1) * 4);
        // two tables of the same entries, each numbering its own blocks: every segment (bf16 inputs), and the
        // planar Q8_0 ones for the int8 entry
        let mut tw: [(Vec<u64>, u32); 2] = Default::default();
        let (mut po, mut co) = (0usize, 0usize);
        for (s, &(tiles, ws, ks)) in segs.iter().zip(&plan) {
            let nb = (tiles.div_ceil(8 / ws) * ks) as u32;
            let lists: &[usize] = if s.qx == QX_Q8_0_P { &[0, 1] } else { &[0] };
            for &l in lists {
                let (words, blocks) = &mut tw[l];
                words.extend_from_slice(&[s.w, s.y, s.rows as u64, s.cols as u64, s.qx, ks as u64, *blocks as u64, s.stride as u64, partials.ptr + (po * 4) as u64, counters.ptr + (co * 4) as u64, ws as u64, 0]);
                *blocks += nb;
            }
            if ks > 1 {
                po += tiles * ks * 512;
                co += tiles;
            }
        }
        let tcols = segs[0].cols;
        assert!(segs.iter().all(|s| s.cols == tcols), "a fused group shares its input");
        let q8 = segs.iter().any(|s| s.qx == QX_Q8_0_P) && tcols % 32 == 0;
        let tab = |l: usize| (qx_gpu::mem::DeviceBuf::from_slice(if tw[l].0.is_empty() { &[0u64][..] } else { &tw[l].0 }), tw[l].1, tw[l].0.len() / 12);
        let rn = rwords.len() / 8;
        let rtable = qx_gpu::mem::DeviceBuf::from_slice(if rn == 0 { &[0u64][..] } else { &rwords });
        Self { table: qx_gpu::mem::DeviceBuf::from_slice(&words), n: segs.len(), blocks, rest: (rtable, rblocks, rn), ttable: [tab(0), tab(1)], tcols, q8, _tscratch: (partials, counters) }
    }

    pub fn run(&self, gpu: &Gpu, x: u64) {
        self.run_x(gpu, x, 0, 0);
    }

    /// With x transformed as it loads: mode 1 silu(x * f32::from_bits(arg)), mode 2 silu(x) * x2 (arg = &x2).
    pub fn run_x(&self, gpu: &Gpu, x: u64, mode: u64, arg: u64) {
        self.run_t(gpu, x, mode, arg, 1, 0);
    }

    /// For `t` tokens with x rows `x_stride` floats apart (launches of up to 8 tokens each); segment s writes its
    /// token-n output at y_s + n * rows_s. Each token's result is bit for bit its `run_x` result.
    pub fn run_t(&self, gpu: &Gpu, x: u64, mode: u64, arg: u64, t: usize, x_stride: usize) {
        self.run_t_with(gpu, x, mode, arg, t, x_stride, use_tmv(t))
    }

    /// `run_t` with the kernel chosen by the caller (`tmv`: `qx_tmv`, else `qx_mgemv`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_t_with(&self, gpu: &Gpu, x: u64, mode: u64, arg: u64, t: usize, x_stride: usize, tmv: bool) {
        const ENTRIES: [[&str; 3]; 8] = [
            ["qx_mgemv", "qx_mgemv_silu", "qx_mgemv_silumul"],
            ["qx_mgemv_t2", "qx_mgemv_silu_t2", "qx_mgemv_silumul_t2"],
            ["qx_mgemv_t3", "qx_mgemv_silu_t3", "qx_mgemv_silumul_t3"],
            ["qx_mgemv_t4", "qx_mgemv_silu_t4", "qx_mgemv_silumul_t4"],
            ["qx_mgemv_t5", "qx_mgemv_silu_t5", "qx_mgemv_silumul_t5"],
            ["qx_mgemv_t6", "qx_mgemv_silu_t6", "qx_mgemv_silumul_t6"],
            ["qx_mgemv_t7", "qx_mgemv_silu_t7", "qx_mgemv_silumul_t7"],
            ["qx_mgemv_t8", "qx_mgemv_silu_t8", "qx_mgemv_silumul_t8"],
        ];
        assert!(t >= 1, "mgemv for no tokens");
        if tmv {
            assert!(t <= 32, "qx_tmv takes at most 32 tokens");
            let k = self.tcols;
            let xb = tmv_scratch(k);
            // planar Q8_0 segments take int8 inputs and the int8 entry; the bf16 inputs only for the other types
            let int8 = self.q8 && tmv_int8();
            let launch = |entry: &'static str, (tab, blocks, n): &(qx_gpu::mem::DeviceBuf, u32, usize), xq: u64, xd: u64| {
                if *n > 0 {
                    gpu.run(modules::GEMM.kernel(entry, CubeDim::new_1d(256)), (*blocks, 1, 1), &[xb, tab.ptr, *n as u64, 0, 0, t as u64, xq, xd]);
                }
            };
            if int8 {
                let (xq, xd) = tmv_q8_scratch(k);
                gpu.run(modules::MGEMV.kernel("qx_mgemv_q8x", CubeDim::new_1d(256)), ((k as u32).div_ceil(256), t as u32, 1), &[x, x_stride as u64, mode, arg, k as u64, xq, xd, 1]);
                const Q8: [&str; 4] = ["qx_tmv_q8_1", "qx_tmv_q8_2", "qx_tmv_q8_3", "qx_tmv_q8_4"];
                launch(Q8[t.div_ceil(8) - 1], &self.ttable[1], xq, xd);
            }
            if int8 {
                // the other types (the router's f32 rows among them) in f32, 8 tokens a launch
                let (rt, rb, rn) = &self.rest;
                if *rn > 0 {
                    for c in (0..t).step_by(8) {
                        let n = (t - c).min(8);
                        let off = (c * x_stride * 4) as u64;
                        let xarg = if mode == 2 { arg + off } else { arg };
                        gpu.run(modules::MGEMV.kernel(ENTRIES[n - 1][mode as usize], CubeDim::new_1d(256)), (*rb, 1, 1), &[x + off, rt.ptr, *rn as u64, x_stride as u64, xarg, n as u64, c as u64, 0, 0, k as u64]);
                    }
                }
            } else {
                gpu.run(modules::GEMM.kernel("qx_tmv_prep", CubeDim::new_1d(256)), (((32 * k / 2) as u32).div_ceil(256), 1, 1), &[x, x_stride as u64, arg, mode, t as u64, k as u64, xb]);
                launch("qx_tmv", &self.ttable[0], 0, 0);
            }
            return;
        }
        for c in (0..t).step_by(8) {
            let n = (t - c).min(8);
            let off = (c * x_stride * 4) as u64;
            let entry = ENTRIES[n - 1][mode as usize];
            let xarg = if mode == 2 { arg + off } else { arg };
            let (xq, xd) = if self.q8 && mgemv_int8() {
                let (xq, xd) = q8x_scratch(self.tcols);
                gpu.run(modules::MGEMV.kernel("qx_mgemv_q8x", CubeDim::new_1d(256)), ((self.tcols as u32).div_ceil(256), n as u32, 1), &[x + off, x_stride as u64, mode, xarg, self.tcols as u64, xq, xd, 0]);
                (xq, xd)
            } else {
                (0, 0)
            };
            gpu.run(modules::MGEMV.kernel(entry, CubeDim::new_1d(256)), (self.blocks, 1, 1), &[x + off, self.table.ptr, self.n as u64, x_stride as u64, xarg, n as u64, c as u64, xq, xd, self.tcols as u64]);
        }
    }
}

/// Kernel type ids of the planar layouts (kernels/include/qx_quant.cuh).
pub const QX_Q6_K_P: u64 = 8;
pub const QX_Q8_0_P: u64 = 9;

/// Regroups Q6_K or Q8_0 rows into the planar layout the GPU reads with aligned 8-byte loads; other types are
/// returned unchanged. Returns the bytes, the kernel type id and the row stride.
pub fn to_planar(t: GgmlType, data: &[u8], rows: usize, cols: usize) -> (Vec<u8>, u64, usize) {
    use rayon::prelude::*;
    let rb = t.bytes(cols);
    let (bb, planes): (usize, &[(usize, usize)]) = match t {
        // (offset in block, length): ql, qh, scales, d
        GgmlType::Q6_K => (210, &[(0, 128), (128, 64), (192, 16), (208, 2)]),
        GgmlType::Q8_0 => (34, &[(2, 32), (0, 2)]),
        _ => return (data.to_vec(), qx_type(t), rb),
    };
    let nb = rb / bb;
    let stride = rb.next_multiple_of(16);
    let mut out = vec![0u8; rows * stride];
    out.par_chunks_mut(stride).enumerate().for_each(|(r, dst)| {
        let src = &data[r * rb..(r + 1) * rb];
        let mut o = 0;
        for &(off, len) in planes {
            for b in 0..nb {
                dst[o..o + len].copy_from_slice(&src[b * bb + off..b * bb + off + len]);
                o += len;
            }
        }
    });
    let id = if t == GgmlType::Q6_K { QX_Q6_K_P } else { QX_Q8_0_P };
    (out, id, stride)
}
