//! The DeepSeek-V4.1-Flash decode path on the GPU, first version: one token at a time, eager launches, the routed
//! experts computed by the CPU pool from the GGUF (no VRAM cache yet), the host between layers for the MoE.
//! It exists to check the kernels in kernels/dsv41.cu against the CPU reference (`qx-ds-check`); the asynchronous,
//! graph-captured, tiered engine grows from it without changing what it computes.
pub mod driver;

use crate::ops::{self, MGemv, Seg};
use anyhow::{anyhow, Result};
use cubecl::prelude::CubeDim;
use qx_cpu::pool::{Expert, ExpertFmt, Pool};
use qx_gpu::kernel::modules::{DSV41, LAYER, MOE, STEP};
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;
use qx_pack::ggml::GgmlType;
use qx_pack::gguf::Gguf;
use qx_pack::pack::PackInfo;
use qx_ref::dsv41::Config;
use qx_ref::weights::W;
use std::os::unix::fs::FileExt;

/// The three expert tiers behind the DeepSeek path: a VRAM arena of whole records filled by rank (static for
/// now), the RAM tier and NVMe through `ExpertStore`, plus the device buffers the expert kernels use.
pub struct Tiers {
    /// The DMA staging area for the GPU's share of RAM-resident misses, its capacity in records, and the share.
    stage: DeviceBuf,
    stage_cap: usize,
    stage_share: u64,
    stage_share_decode: u64,
    stage_wide_min: usize,
    pub store: crate::tiers::ExpertStore,
    arena: DeviceBuf,
    /// Per global expert: its VRAM address or 0.
    table: Vec<u64>,
    /// VRAM slots by record size: (bytes, free slot addresses); and which expert holds each slot.
    classes: Vec<(usize, Vec<u64>)>,
    layer_class: Vec<usize>,
    holder: std::collections::HashMap<u64, usize>,
    /// Decayed routing counts per expert, the experts missed this token, and the policy.
    score: Vec<f64>,
    missed: Vec<usize>,
    decay: f64,
    pub promos_per_token: usize,
    pub margin: f64,
    pub adaptive: bool,
    pub vram_promos: u64,
    pub ram_promos: u64,
    pub promo_secs: f64,
    slots: DeviceBuf,
    gpu_out: DeviceBuf,
    xq: DeviceBuf,
    xd: DeviceBuf,
    gu: DeviceBuf,
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub nvme_misses: u64,
    pub pool_secs: f64,
    pub gpu_secs: f64,
}

/// A weight matrix in VRAM (planar for Q8_0 and Q6_K, see `ops::to_planar`).
pub struct Mat {
    pub buf: DeviceBuf,
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub qx: u64,
    pub stride: usize,
}

impl Mat {
    fn seg(&self, y: &DeviceBuf) -> Seg {
        Seg { w: self.buf.ptr, y: y.ptr, rows: self.rows, cols: self.cols, qx: self.qx, stride: self.stride }
    }
    /// Rows [r0, r0 + n) as a segment.
    fn seg_rows(&self, y: u64, r0: usize, n: usize) -> Seg {
        Seg { w: self.buf.ptr + (r0 * self.stride) as u64, y, rows: n, cols: self.cols, qx: self.qx, stride: self.stride }
    }
}

struct HcDs {
    /// The mix matrix as f32 rows (24 x hc*E).
    fn_: DeviceBuf,
    scale: DeviceBuf,
    base: DeviceBuf,
}

struct Experts<'a> {
    /// The expert matrices in a plain GGUF (None when loaded from a pack: the records come from the tiers).
    gate: Option<W<'a>>,
    up: Option<W<'a>>,
    down: Option<W<'a>>,
    fmt: ExpertFmt,
    gate_ty: GgmlType,
    down_ty: GgmlType,
    gate_row_bytes: usize,
    down_row_bytes: usize,
}

struct Layer<'a> {
    attn_norm: DeviceBuf,
    ffn_norm: DeviceBuf,
    q_a_norm: DeviceBuf,
    kv_norm: DeviceBuf,
    sinks: DeviceBuf,
    hc_attn: HcDs,
    hc_ffn: HcDs,
    /// On the attention input: wq_a, wkv, and where the layer has them the compressor's kv and gate and the
    /// indexer's head weights.
    g_x: MGemv,
    /// On qr: wq_b, and the indexer's query projection where the layer has it.
    g_qr: MGemv,
    wo_a: Vec<MGemv>,
    g_ob: MGemv,
    comp_norm: Option<DeviceBuf>,
    idx_k_norm: Option<DeviceBuf>,
    g_ik: Option<MGemv>,
    engram: Option<(MGemv, DeviceBuf)>,
    g_ffn: MGemv,
    g_shd: MGemv,
    router_bias: DeviceBuf,
    experts: Experts<'a>,
    has_comp: bool,
    has_idx_q: bool,
}

/// Device scratch for one token.
struct Scratch {
    tok: DeviceBuf,
    pos: DeviceBuf,
    slot: DeviceBuf,
    /// Host-mapped staging for the three per-step inputs above (ids, positions, slots: DS_TMAX u32 each): the host
    /// writes them and a copy kernel moves them in stream order, so a step never synchronizes the whole context
    /// (a synchronous upload waited for the driver's in-flight promotion copies: 1.4-7 ms a step at 256k, 2026-11-06).
    in_host: qx_gpu::mem::HostMapped,
    r: DeviceBuf,
    rn: DeviceBuf,
    e: DeviceBuf,
    mixes: DeviceBuf,
    /// The 24 summed mixes (for the activation dumps; `hc_terms` sums the partials in `mixes`).
    mixsum: DeviceBuf,
    pre_a: DeviceBuf,
    pre_f: DeviceBuf,
    pre_prev: DeviceBuf,
    pre_first: DeviceBuf,
    post: DeviceBuf,
    comb: DeviceBuf,
    xpre: DeviceBuf,
    x: DeviceBuf,
    qr_raw: DeviceBuf,
    qr: DeviceBuf,
    q: DeviceBuf,
    kv_raw: DeviceBuf,
    kv: DeviceBuf,
    ckv: DeviceBuf,
    cscore: DeviceBuf,
    comp_tmp: DeviceBuf,
    flag: DeviceBuf,
    ik: DeviceBuf,
    iq: DeviceBuf,
    iw: DeviceBuf,
    scores: DeviceBuf,
    bmax: DeviceBuf,
    inside: DeviceBuf,
    cand_sel: DeviceBuf,
    cand_count: DeviceBuf,
    attn: DeviceBuf,
    /// Per (head, key chunk) partials of the split attention: (max, sum, o[head_dim]).
    attn_part: DeviceBuf,
    /// The grouped output projection's result as the GEMVs write it, [group][DS_TMAX][o_lora]; regrouped per
    /// token into `oa` for the second projection.
    oag: DeviceBuf,
    oa: DeviceBuf,
    y: DeviceBuf,
    ekv: DeviceBuf,
    erows: DeviceBuf,
    /// The Engram rows the host gathers, pinned and device-mapped, one region per Engram layer: a kernel copies
    /// them to `erows` in stream order (a plain upload runs on the legacy stream, unordered against the queued
    /// kernels, and corrupts the earlier Engram layer's input when the host runs ahead of the device).
    erows_host: qx_gpu::mem::HostMapped,
    egates: DeviceBuf,
    logits_r: DeviceBuf,
    ids: DeviceBuf,
    wts: DeviceBuf,
    yk: DeviceBuf,
    ones_slots: DeviceBuf,
    shg: DeviceBuf,
    shu: DeviceBuf,
    shh: DeviceBuf,
    shy: DeviceBuf,
    logits: DeviceBuf,
    ones: DeviceBuf,
}

/// One sequence's state.
struct State {
    /// Per layer: the latent rings of every slot (slots x W x 512, slot s at s * W rows) and the table of their
    /// addresses (one per slot).
    rings: Vec<DeviceBuf>,
    ring_tables: Vec<DeviceBuf>,
    /// Per KV source layer: the compressed stores of every slot (slots x rows x 640), their table, the pending
    /// pairs of every slot, their table.
    stores: Vec<Option<(DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf)>>,
    /// Per top-k source layer: the selected rows and their count, per token of a window.
    sels: Vec<Option<(DeviceBuf, DeviceBuf)>>,
    /// Per slot: the tokens so far and the position.
    tokens: Vec<Vec<u32>>,
    pos: Vec<usize>,
}

/// A sequence moved out of its slot (`DsEngine::park`): what `unpark` needs to continue it in any slot.
pub struct Parked {
    pub tokens: Vec<u32>,
    /// Per layer: the used rows of its compressed store (empty for layers without one).
    stores: Vec<Vec<f32>>,
}

impl Parked {
    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        4 * (self.tokens.len() + self.stores.iter().map(Vec::len).sum::<usize>())
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Writes the state to `path`: a magic, the tokens, then every layer's store rows (little-endian).
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"QXPK0001")?;
        f.write_all(&(self.tokens.len() as u64).to_le_bytes())?;
        f.write_all(bytemuck::cast_slice(&self.tokens))?;
        f.write_all(&(self.stores.len() as u64).to_le_bytes())?;
        for s in &self.stores {
            f.write_all(&(s.len() as u64).to_le_bytes())?;
            f.write_all(bytemuck::cast_slice(s))?;
        }
        f.flush()
    }

    /// Reads a state `write` produced.
    pub fn read(path: &std::path::Path) -> std::io::Result<Parked> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"QXPK0001" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a QuixiInfer state file"));
        }
        let mut n = [0u8; 8];
        let rd_len = |f: &mut std::io::BufReader<std::fs::File>, n: &mut [u8; 8]| -> std::io::Result<usize> {
            f.read_exact(n)?;
            Ok(u64::from_le_bytes(*n) as usize)
        };
        let nt = rd_len(&mut f, &mut n)?;
        let mut tokens = vec![0u32; nt];
        f.read_exact(bytemuck::cast_slice_mut(&mut tokens))?;
        let nl = rd_len(&mut f, &mut n)?;
        let mut stores = Vec::with_capacity(nl);
        for _ in 0..nl {
            let len = rd_len(&mut f, &mut n)?;
            let mut v = vec![0f32; len];
            f.read_exact(bytemuck::cast_slice_mut(&mut v))?;
            stores.push(v);
        }
        Ok(Parked { tokens, stores })
    }
}

/// The device side of the tiers in the asynchronous mode (the host side lives in the driver thread).
pub struct AsyncBufs {
    _arena: DeviceBuf,
    table_dev: DeviceBuf,
    /// The RAM table and the share (in eighths) of RAM-resident misses the GPU reads in place over PCIe
    /// (`QX_DS_GPU_SHARE`, default 0: measured at about 7 GB/s, six times slower per expert than the pool).
    rtable_dev: DeviceBuf,
    gpu_share: u64,
    /// The share (eighths) of RAM-resident misses the driver stages into VRAM by DMA for the GPU's expert kernels
    /// (`QX_DS_STAGE_SHARE`), the staging area (`stage_cap` records of the largest size, reused per job), and the
    /// slot arrays: `slots_all` (nonzero where the GPU produces the output) and `slots_stg` (the staged addresses,
    /// written by the driver).
    stage_share: u64,
    stage_share_decode: u64,
    /// Windows of this width or more use `stage_share`; narrower ones `stage_share_decode`.
    stage_wide_min: usize,
    stage_cap: usize,
    _stage: DeviceBuf,
    slots_all: DeviceBuf,
    slots_stg: DeviceBuf,
    step: DeviceBuf,
    slots: DeviceBuf,
    gpu_out: DeviceBuf,
    xq: DeviceBuf,
    xd: DeviceBuf,
    gu: DeviceBuf,
}

pub struct DsEngine<'a> {
    pub cfg: Config,
    gpu: Gpu,
    embd: Mat,
    output: Mat,
    output_norm: DeviceBuf,
    g_out: MGemv,
    engram_table: Option<W<'a>>,
    layers: Vec<Layer<'a>>,
    s: Scratch,
    st: State,
    /// The pool for the experts the host dispatches (None in the asynchronous mode: the driver has its own).
    pool: Option<Pool>,
    /// Rows the compressed stores hold (the context in compressed positions).
    max_rows: usize,
    /// Key chunks of 12 per token in the split attention: window plus selected rows.
    attn_chunks: usize,
    ring_w: usize,
    /// Positions each slot holds; a layer's compressed store has `store_rows(max_ctx, ratio)` rows per slot.
    max_ctx: usize,
    pub exact_experts: bool,
    /// The tiers when loaded from a pack; else the experts are read from the GGUF by the pool.
    pub tiers: Option<std::cell::RefCell<Tiers>>,
    /// The asynchronous mode (`QX_DS_ASYNC`, default on with a pack): the driver thread and the device buffers.
    pub driver: Option<driver::Driver>,
    abufs: Option<AsyncBufs>,
    /// The pack file and the Engram section's geometry, for rows read on the host.
    engram_pack: Option<(std::fs::File, PackInfo)>,
    /// io_uring for the per-token Engram row reads (all rows of a token in one batch; under NVMe contention the
    /// serial preads cost 4 ms per token).
    engram_ring: Option<std::cell::RefCell<io_uring::IoUring>>,
    /// The latents of every position so far, per layer (the ring only keeps the last n_swa + DS_TMAX), so a
    /// Sequence slots: each has its own ring, compressed stores, position and tokens.
    pub slots: usize,
    /// The captured token (asynchronous mode), and whether the warm-up run before its capture happened.
    graphs: Vec<Option<cubecl_runtime::client::Graph>>,
    /// Per width: whether its compile pass ran (see `forward_segments`); `warm` marks that pass while it runs.
    warmed: Vec<bool>,
    warm: bool,
    /// Seconds in the last forward: all, experts (pool + GPU experts), the host's Engram row gather.
    pub t_forward: f64,
    pub t_engram: f64,
    /// Phases of the last forward, seconds: uploads, launch or replay, device wait, (unused), download.
    pub t_phase: [f64; 5],
    pub t_experts: f64,
}

const E: usize = 5120;
/// Tokens per window (a prefill chunk or a speculative verification); the mailbox takes up to this many.
pub const DS_TMAX: usize = 128;
/// Tokens per attention launch (the split attention's partials are sized for this many).
const ATTN_SLICE: usize = 32;
const _: () = assert!(DS_TMAX <= crate::mailbox::TMAX);
/// Blocks per output in `qx_ds_hc_mixes` (24 outputs x this many 256-thread blocks stream the 2 MB mixing matrix).
const HC_SPLIT: usize = 16;

/// A kernel name with a static lifetime, interned so repeated launches do not leak (the entry set is small).
fn static_name(s: String) -> &'static str {
    static NAMES: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut v = NAMES.lock().unwrap();
    if let Some(&n) = v.iter().find(|&&n| n == s) {
        return n;
    }
    let n: &'static str = Box::leak(s.into_boxed_str());
    v.push(n);
    n
}

/// The pool's CPUs: `QX_DS_CPUS` as `a-b` or `a,b,c` (default 0-15: one thread per physical core on a 16-core
/// part; the first is the driver thread, which is pool thread 0).
pub fn pool_cpus() -> Vec<usize> {
    let spec = std::env::var("QX_DS_CPUS").unwrap_or_else(|_| "0-15".into());
    let mut v = Vec::new();
    for part in spec.split(',') {
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (usize, usize) = (a.trim().parse().expect("QX_DS_CPUS"), b.trim().parse().expect("QX_DS_CPUS"));
            v.extend(a..=b);
        } else if !part.trim().is_empty() {
            v.push(part.trim().parse().expect("QX_DS_CPUS"));
        }
    }
    assert!(!v.is_empty(), "QX_DS_CPUS names no CPU");
    v
}

/// Pins the calling thread to one CPU.
pub fn pin_thread(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn bits(f: f32) -> u64 {
    f.to_bits() as u64
}

/// YaRN parameters of a layer's RoPE: (base, freq_scale, corr0, corr1, ext).
fn rope_params(c: &Config, compressed: bool) -> (f32, f32, f32, f32, u64) {
    let n = c.n_rot as f32;
    if compressed {
        let base = c.comp_rope_base;
        let dim = |n_rot: f32| n * (c.yarn_orig_ctx as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        (base, 1.0 / c.yarn_factor, dim(c.yarn_beta_fast).floor().max(0.0), dim(c.yarn_beta_slow).ceil().min(n - 1.0), 1)
    } else {
        (c.rope_base, 1.0, 0.0, 0.0, 0)
    }
}

/// `layer expert` lines, hottest first, as `qx-profile` and the routing trace write them.
fn read_profile(path: &std::path::Path, n_layer: usize, ne: usize) -> Option<Vec<(usize, usize)>> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: Vec<(usize, usize)> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .filter(|&(l, e)| l < n_layer && e < ne)
        .collect();
    (v.len() == n_layer * ne).then_some(v)
}

impl<'a> DsEngine<'a> {
    /// From a plain GGUF: experts computed by the pool from the mapped file.
    pub fn load(g: &'a Gguf, max_ctx: usize) -> Result<Self> {
        Self::load_with(g, max_ctx, None, 1)
    }

    /// From a pack (`g` opened on it): `vram_gib` of expert records in VRAM and `ram_gib` pinned, both filled by the
    /// ranking in `profile` (else layer-major), the rest read from the pack on NVMe.
    pub fn load_tiered(g: &'a Gguf, max_ctx: usize, vram_gib: f64, ram_gib: f64, profile: Option<&std::path::Path>) -> Result<Self> {
        Self::load_tiered_slots(g, max_ctx, vram_gib, ram_gib, profile, 1)
    }

    /// `load_tiered` with `slots` sequences, each with its own ring and compressed stores (`max_ctx` positions).
    pub fn load_tiered_slots(g: &'a Gguf, max_ctx: usize, vram_gib: f64, ram_gib: f64, profile: Option<&std::path::Path>, slots: usize) -> Result<Self> {
        let pack = PackInfo::of(g).ok_or_else(|| anyhow!("not a pack"))??;
        Self::load_with(g, max_ctx, Some((pack, vram_gib, ram_gib, profile)), slots.max(1))
    }

    fn load_with(g: &'a Gguf, max_ctx: usize, tiered: Option<(PackInfo, f64, f64, Option<&std::path::Path>)>, slots: usize) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let gpu = Gpu::new();
        let t = |n: &str| g.tensor(n).ok_or_else(|| anyhow!("missing tensor {n}"));
        let mat = |n: &str| -> Result<Mat> {
            let w = W::new(g, t(n)?);
            let (bytes, qx, stride) = ops::to_planar(w.ty, w.data, w.rows, w.cols);
            Ok(Mat { buf: DeviceBuf::from_slice(&bytes), ty: w.ty, rows: w.rows, cols: w.cols, qx, stride })
        };
        let raw_mat = |n: &str| -> Result<Mat> {
            let w = W::new(g, t(n)?);
            Ok(Mat { buf: DeviceBuf::from_slice(w.data), ty: w.ty, rows: w.rows, cols: w.cols, qx: ops::qx_type(w.ty), stride: w.row_bytes() })
        };
        let vecf = |n: &str| -> Result<DeviceBuf> { Ok(DeviceBuf::from_slice(&W::new(g, t(n)?).to_vec())) };
        let hc = cfg.hc;
        let hcd = hc * E;
        let buf = |n: usize| DeviceBuf::new(n * 4);
        let tm = DS_TMAX;
        let s = Scratch {
            tok: buf(tm),
            pos: buf(tm),
            slot: buf(tm),
            in_host: qx_gpu::mem::HostMapped::new(3 * tm * 4),
            r: buf(hcd * tm),
            rn: buf(hcd * tm),
            e: buf(E * tm),
            mixes: buf(24 * HC_SPLIT * tm),
            mixsum: buf(24 * tm),
            pre_a: buf(hc * tm),
            pre_f: buf(hc * tm),
            pre_prev: buf(hc * tm),
            pre_first: DeviceBuf::from_slice(&(0..tm * hc).map(|i| if i % hc == 0 { 1f32 } else { 0.0 }).collect::<Vec<f32>>()),
            post: buf(hc * tm),
            comb: buf(hc * hc * tm),
            xpre: buf(E * tm),
            x: buf(E * tm),
            qr_raw: buf(cfg.q_lora * tm),
            qr: buf(cfg.q_lora * tm),
            q: buf(cfg.n_head * cfg.head_dim * tm),
            kv_raw: buf(cfg.head_dim * tm),
            kv: buf(cfg.head_dim * tm),
            ckv: buf(cfg.head_dim * tm),
            cscore: buf(cfg.head_dim * tm),
            comp_tmp: buf(cfg.head_dim * tm),
            flag: buf(tm),
            ik: buf(cfg.idx_dim * tm),
            iq: buf(cfg.idx_heads * cfg.idx_dim * tm),
            iw: buf(cfg.idx_heads * tm),
            scores: buf(max_ctx.max(1) * tm),
            bmax: buf(max_ctx.div_ceil(cfg.cand_bs.max(1)).max(1) * tm),
            inside: buf(max_ctx.div_ceil(cfg.cand_bs.max(1) * 32).max(1) * tm),
            cand_sel: buf(cfg.cand_blocks.max(1) * tm),
            cand_count: buf(tm),
            attn: buf(cfg.n_head * cfg.head_dim * tm),
            attn_part: buf(cfg.n_head * (cfg.n_swa + cfg.idx_top_k).div_ceil(12) * (cfg.head_dim + 2) * ATTN_SLICE),
            oag: buf(cfg.o_groups * tm * cfg.o_lora),
            oa: buf(cfg.o_groups * cfg.o_lora * tm),
            y: buf(E * tm),
            ekv: buf((hc + 1) * E * tm),
            erows: buf(cfg.engram_cols() * cfg.engram_dim * tm),
            erows_host: qx_gpu::mem::HostMapped::new(cfg.engram_layers.len() * tm * cfg.engram_cols() * cfg.engram_dim * 4),
            egates: buf(hc * tm),
            logits_r: buf(cfg.n_expert * tm),
            ids: buf(cfg.n_expert_used * tm),
            wts: buf(cfg.n_expert_used * tm),
            yk: buf(cfg.n_expert_used * E * tm),
            ones_slots: DeviceBuf::from_slice(&vec![1u64; cfg.n_expert_used * tm]),
            shg: buf(cfg.ff_exp * tm),
            shu: buf(cfg.ff_exp * tm),
            shh: buf(cfg.ff_exp * tm),
            shy: buf(E * tm),
            logits: buf(cfg.n_vocab * tm),
            ones: DeviceBuf::from_slice(&vec![1f32; hcd]),
        };
        let ring_w = cfg.n_swa + DS_TMAX;
        let mut layers = Vec::new();
        let mut rings = Vec::new();
        let mut ring_tables = Vec::new();
        let mut stores = Vec::new();
        let mut sels = Vec::new();
        let row = cfg.head_dim + cfg.idx_dim;
        for l in 0..cfg.n_layer {
            let b = |n: &str| format!("blk.{l}.{n}");
            let has_comp = cfg.kv_src[l] == l as i32;
            let has_idx_q = cfg.topk_src[l] == l as i32;
            let hcw = |p: &str| -> Result<HcDs> {
                Ok(HcDs { fn_: vecf(&b(&format!("{p}_fn.weight")))?, scale: vecf(&b(&format!("{p}_scale.weight")))?, base: vecf(&b(&format!("{p}_base.weight")))? })
            };
            // the attention-input group
            let wq_a = mat(&b("attn_q_a.weight"))?;
            let wkv = mat(&b("attn_kv.weight"))?;
            let mut segs = vec![wq_a.seg(&s.qr_raw), wkv.seg(&s.kv_raw)];
            let mut keep: Vec<Mat> = vec![wq_a, wkv];
            if has_comp {
                let m = mat(&b("attn_compressor_kv.weight"))?;
                segs.push(m.seg(&s.ckv));
                keep.push(m);
                if cfg.ratio[l] > 1 {
                    let m = mat(&b("attn_compressor_gate.weight"))?;
                    segs.push(m.seg(&s.cscore));
                    keep.push(m);
                }
            }
            if has_idx_q {
                let m = mat(&b("indexer.proj.weight"))?;
                segs.push(m.seg(&s.iw));
                keep.push(m);
            }
            let g_x = MGemv::new(&segs);
            let wq_b = mat(&b("attn_q_b.weight"))?;
            let mut segs = vec![wq_b.seg(&s.q)];
            keep.push(wq_b);
            if has_idx_q {
                let m = mat(&b("indexer.attn_q_b.weight"))?;
                segs.push(m.seg(&s.iq));
                keep.push(m);
            }
            let g_qr = MGemv::new(&segs);
            let wo_a = mat(&b("attn_output_a.weight"))?;
            let gd = cfg.n_head / cfg.o_groups * cfg.head_dim;
            assert_eq!(wo_a.cols, gd);
            let wo_a_groups: Vec<MGemv> = (0..cfg.o_groups).map(|gi| MGemv::new(&[wo_a.seg_rows(s.oag.ptr + (gi * DS_TMAX * cfg.o_lora * 4) as u64, gi * cfg.o_lora, cfg.o_lora)])).collect();
            keep.push(wo_a);
            let wo_b = mat(&b("attn_output_b.weight"))?;
            let g_ob = MGemv::new(&[wo_b.seg(&s.y)]);
            keep.push(wo_b);
            let (comp_norm, idx_k_norm, g_ik) = if has_comp {
                let m = mat(&b("indexer.attn_k.weight"))?;
                let g_ik = MGemv::new(&[m.seg(&s.ik)]);
                keep.push(m);
                (Some(vecf(&b("attn_compressor_norm.weight"))?), Some(vecf(&b("indexer.k_norm.weight"))?), Some(g_ik))
            } else {
                (None, None, None)
            };
            let engram = if cfg.engram_layers.contains(&l) {
                let m = mat(&b("engram_kv.weight"))?;
                let g = MGemv::new(&[m.seg(&s.ekv)]);
                keep.push(m);
                Some((g, vecf(&b("engram_gate.weight"))?))
            } else {
                None
            };
            let router = mat(&b("ffn_gate_inp.weight"))?;
            let shg = mat(&b("ffn_gate_shexp.weight"))?;
            let shu = mat(&b("ffn_up_shexp.weight"))?;
            let g_ffn = MGemv::new(&[router.seg(&s.logits_r), shg.seg(&s.shg), shu.seg(&s.shu)]);
            keep.extend([router, shg, shu]);
            let shd = mat(&b("ffn_down_shexp.weight"))?;
            let g_shd = MGemv::new(&[shd.seg(&s.shy)]);
            keep.push(shd);
            let (gt, ut, dt) = (t(&b("ffn_gate_exps.weight"))?, t(&b("ffn_up_exps.weight"))?, t(&b("ffn_down_exps.weight"))?);
            let fmt = ExpertFmt { gate: crate::experts::wtype(gt.ty), down: crate::experts::wtype(dt.ty), clamp: cfg.clamp_exp[l] };
            let packed = gt.packed();
            let (gate, up, down) = if packed { (None, None, None) } else { (Some(W::new(g, gt)), Some(W::new(g, ut)), Some(W::new(g, dt))) };
            let (gate_row_bytes, down_row_bytes) = (gt.ty.bytes(gt.shape[0] as usize), dt.ty.bytes(dt.shape[0] as usize));
            // the Mats behind the groups must stay alive: their device buffers are referenced by the tables
            std::mem::forget(keep);
            layers.push(Layer {
                attn_norm: vecf(&b("attn_norm.weight"))?,
                ffn_norm: vecf(&b("ffn_norm.weight"))?,
                q_a_norm: vecf(&b("attn_q_a_norm.weight"))?,
                kv_norm: vecf(&b("attn_kv_a_norm.weight"))?,
                sinks: vecf(&b("attn_sinks.weight"))?,
                hc_attn: hcw("hc_attn")?,
                hc_ffn: hcw("hc_ffn")?,
                g_x,
                g_qr,
                wo_a: wo_a_groups,
                g_ob,
                comp_norm,
                idx_k_norm,
                g_ik,
                engram,
                g_ffn,
                g_shd,
                router_bias: vecf(&b("exp_probs_b.bias"))?,
                experts: Experts { gate, up, down, fmt, gate_ty: gt.ty, down_ty: dt.ty, gate_row_bytes, down_row_bytes },
                has_comp,
                has_idx_q,
            });
            let ring = buf(slots * ring_w * cfg.head_dim);
            ring_tables.push(DeviceBuf::from_slice(&(0..slots).map(|sl| ring.ptr + (sl * ring_w * cfg.head_dim * 4) as u64).collect::<Vec<u64>>()));
            rings.push(ring);
            stores.push(if has_comp {
                // rows per slot: the positions compress `ratio` to one row (row (pos + 1) / ratio - 1 is written), so
                // max_ctx / ratio rows are ever used; one spare row keeps every index in range. At 256k with the
                // 2:1 layers this is half the 2.7 GB a max_ctx-row store would take per slot (2026-11-06).
                let rows_l = Self::store_rows(max_ctx, cfg.ratio[l] as usize);
                let store = buf(slots * rows_l * row);
                let pending = buf(slots * cfg.ratio[l] * 2 * cfg.head_dim);
                let st = DeviceBuf::from_slice(&(0..slots).map(|sl| store.ptr + (sl * rows_l * row * 4) as u64).collect::<Vec<u64>>());
                let pt = DeviceBuf::from_slice(&(0..slots).map(|sl| pending.ptr + (sl * cfg.ratio[l] * 2 * cfg.head_dim * 4) as u64).collect::<Vec<u64>>());
                Some((store, st, pending, pt))
            } else {
                None
            });
            // the selected rows and their count per token of a window
            sels.push(if has_idx_q { Some((buf(cfg.idx_top_k * DS_TMAX), buf(DS_TMAX))) } else { None });
        }
        let output = mat("output.weight")?;
        let g_out = MGemv::new(&[output.seg(&s.logits)]);
        let cpus = pool_cpus();
        let n_expert = cfg.n_expert;
        let (tiers, engram_pack) = match tiered {
            None => (None, None),
            Some((pack, vram_gib, ram_gib, profile)) => {
                let order: Vec<(usize, usize)> = profile.and_then(|p| read_profile(p, cfg.n_layer, n_expert)).unwrap_or_else(|| {
                    eprintln!("tiers: no routing profile, filling in layer order");
                    (0..cfg.n_layer).flat_map(|l| (0..n_expert).map(move |e| (l, e))).collect()
                });
                // VRAM: the hottest records that fit; a slot per record at its own size
                // the expert tier takes at most what the context, slots and scratch left (everything else is
                // allocated by now), less headroom for the runtime's pools and the window graphs
                // the staging area for DMA-staged experts comes first (its slots are of the largest record)
                // the share for prompt windows (8 tokens and more) and for decode steps; both in eighths
                let knob = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).min(8);
                // 8/8 and 256 records since 2026-11-06 (were 6/8 and 96): at windows of 128 the GPU takes every RAM
                // miss by DMA and the pool idles, 10.8 ms per token against 12.3 at the old 64-chunk; 6/8 at 96 made
                // no difference at 64 and overflowed at 128 (perf/optimization_status.md, "128-chunk bundle")
                let stage_share = knob("QX_DS_STAGE_SHARE", 8);
                let stage_share_decode = knob("QX_DS_STAGE_SHARE_DECODE", 3);
                // The window share applies from width 2 (QX_DS_STAGE_WIDE_MIN; it was 8 until 2026-11-06): at widths
                // 2 and 4 the 6/8 share measured 4.7% and 8.2% faster per position than 3/8, outside the noise
                // (perf/optimization_status.md). Width 1 keeps the decode share: a staged expert's DMA costs about 0.7
                // ms against the pool's 0.22 when it serves one token, and 4/8 measured flat there.
                let stage_wide_min: usize = std::env::var("QX_DS_STAGE_WIDE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1);
                let stage_cap: usize = if stage_share > 0 || stage_share_decode > 0 { std::env::var("QX_DS_STAGE_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(256) } else { 1 };
                let max_rec = (0..cfg.n_layer).map(|l| pack.experts.record[l] as usize).max().unwrap_or(0);
                let stage = DeviceBuf::new(stage_cap * max_rec.max(256));
                let want = (vram_gib * 1073741824.0) as usize;
                let (free, _) = qx_gpu::mem::mem_info();
                // The headroom must hold what comes after the tier: CubeCL's pools, every kernel module, and one
                // instantiated graph per window width (14 in a server). 1.5 GiB was not enough on 2026-11-06: a module
                // first loaded after the graphs failed with CUDA_ERROR_OUT_OF_MEMORY (see the warm-up's free-memory
                // line in the server log, and QX_DS_VRAM_RESERVE_MIB to adjust).
                let reserve_mib: usize = std::env::var("QX_DS_VRAM_RESERVE_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(2560);
                let budget = want.min(free.saturating_sub(reserve_mib << 20));
                if budget < want {
                    eprintln!("tiers: VRAM budget cut from {:.1} to {:.1} GiB: the context and slots take the rest", vram_gib, budget as f64 / 1073741824.0);
                }
                let mut take: Vec<(usize, usize)> = Vec::new();
                let mut used = 0usize;
                let mut in_vram = vec![false; cfg.n_layer * n_expert];
                for &(l, e) in &order {
                    let b = pack.experts.record[l] as usize;
                    if in_vram[l * n_expert + e] || used + b > budget {
                        continue;
                    }
                    in_vram[l * n_expert + e] = true;
                    take.push((l, e));
                    used += b;
                }
                // slots by record size (one class per distinct size), 2 spare slots per class for promotions
                let mut sizes: Vec<usize> = Vec::new();
                let layer_class: Vec<usize> = (0..cfg.n_layer)
                    .map(|l| {
                        let b = pack.experts.record[l] as usize;
                        sizes.iter().position(|&x| x == b).unwrap_or_else(|| {
                            sizes.push(b);
                            sizes.len() - 1
                        })
                    })
                    .collect();
                let spare = 2usize;
                let arena = DeviceBuf::new(used + sizes.iter().sum::<usize>() * spare);
                let mut store = crate::tiers::ExpertStore::build(&pack, cfg.n_layer, &order, &in_vram, (ram_gib * 1073741824.0) as usize, 64)?;
                let t0 = std::time::Instant::now();
                let mut table = vec![0u64; cfg.n_layer * n_expert];
                let mut holder = std::collections::HashMap::new();
                let mut off = 0usize;
                // records not in RAM come through the store's staging, 32 at a time
                for chunk in take.chunks(32) {
                    let gs: Vec<usize> = chunk.iter().map(|&(l, e)| l * n_expert + e).collect();
                    let ptrs = store.fetch(&gs)?;
                    for (&g_id, &p) in gs.iter().zip(&ptrs) {
                        let n = store.record_bytes(g_id);
                        let src = unsafe { std::slice::from_raw_parts(p, n) };
                        arena.upload_at(off, src);
                        table[g_id] = arena.ptr + off as u64;
                        holder.insert(table[g_id], g_id);
                        off += n;
                    }
                }
                let mut classes: Vec<(usize, Vec<u64>)> = sizes.iter().map(|&b| (b, Vec::new())).collect();
                for (c, &b) in sizes.iter().enumerate() {
                    for _ in 0..spare {
                        classes[c].1.push(arena.ptr + off as u64);
                        off += b;
                    }
                }
                let mut score = vec![0f64; cfg.n_layer * n_expert];
                for (i, &(l, e)) in order.iter().enumerate() {
                    score[l * n_expert + e] = 1.0 / (1.0 + i as f64 / 2000.0);
                }
                eprintln!("tiers: {} records ({:.1} GiB) in VRAM in {} size classes, filled in {:.1}s", take.len(), used as f64 / 1073741824.0, sizes.len(), t0.elapsed().as_secs_f64());
                let k = cfg.n_expert_used;
                let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
                let tiers = Tiers {
                    stage,
                    stage_cap,
                    stage_share,
                    stage_share_decode,
                    stage_wide_min,
                    store,
                    arena,
                    table,
                    classes,
                    layer_class,
                    holder,
                    score,
                    missed: Vec::new(),
                    decay: 0.5f64.powf(1.0 / env("QX_DS_HALF_LIFE", 256.0)),
                    promos_per_token: env("QX_DS_PROMOS", 4.0) as usize,
                    margin: env("QX_DS_MARGIN", 0.2),
                    adaptive: !std::env::var("QX_DS_ADAPT").is_ok_and(|v| v == "0"),
                    vram_promos: 0,
                    ram_promos: 0,
                    promo_secs: 0.0,
                    slots: DeviceBuf::new(k * DS_TMAX * 8),
                    gpu_out: buf(k * E * DS_TMAX),
                    xq: DeviceBuf::new(E * DS_TMAX),
                    xd: buf(E / 256 * DS_TMAX),
                    gu: buf(k * 2 * cfg.ff_exp * DS_TMAX),
                    vram_hits: 0,
                    ram_hits: 0,
                    nvme_misses: 0,
                    pool_secs: 0.0,
                    gpu_secs: 0.0,
                };
                let f = std::fs::File::open(&pack.path)?;
                (Some(tiers), Some((f, pack)))
            }
        };
        let (tiers, driver, abufs) = match tiers {
            Some(t) if !std::env::var("QX_DS_ASYNC").is_ok_and(|v| v == "0") => {
                let layers_fmt: Vec<driver::LayerFmt> = layers.iter().map(|l| driver::LayerFmt { fmt: l.experts.fmt, gate_bytes: cfg.ff_exp * l.experts.gate_row_bytes, gate_row_bytes: l.experts.gate_row_bytes, down_row_bytes: l.experts.down_row_bytes }).collect();
                let table_dev = DeviceBuf::from_slice(&t.table);
                // the RAM-resident experts' mapped addresses: the GPU computes a share of them in place
                let rtable: Vec<u64> = (0..t.table.len()).map(|g| t.store.ram_device_addr(g)).collect();
                let rtable_dev = DeviceBuf::from_slice(&rtable);
                let gpu_share: u64 = std::env::var("QX_DS_GPU_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0u64).min(8);
                let slots_all = DeviceBuf::new(cfg.n_expert_used * DS_TMAX * 8);
                let slots_stg = DeviceBuf::new(cfg.n_expert_used * DS_TMAX * 8);
                let host = driver::Host {
                    store: t.store,
                    table_dev: table_dev.ptr,
                    table: t.table,
                    rtable_dev: rtable_dev.ptr,
                    rtable,
                    gpu_reads_ram: gpu_share > 0,
                    stage_base: t.stage.ptr,
                    stage_cap: t.stage_cap,
                    slots_stg_dev: slots_stg.ptr,
                    slots_all_dev: slots_all.ptr,
                    arena_ptr: t.arena.ptr,
                    classes: t.classes,
                    layer_class: t.layer_class,
                    holder: t.holder,
                    score: t.score,
                    decay: t.decay,
                    promos_per_token: t.promos_per_token,
                    margin: t.margin,
                    adaptive: t.adaptive,
                    layers: layers_fmt,
                    n_expert,
                    n_embd: E,
                    ff: cfg.ff_exp,
                };
                let abufs = AsyncBufs { _arena: t.arena, table_dev, rtable_dev, gpu_share, stage_share: t.stage_share, stage_share_decode: t.stage_share_decode, stage_wide_min: t.stage_wide_min, stage_cap: t.stage_cap, _stage: t.stage, slots_all, slots_stg, step: DeviceBuf::new(crate::mailbox::STEP_ROW * 4), slots: t.slots, gpu_out: t.gpu_out, xq: t.xq, xd: t.xd, gu: t.gu };
                abufs.step.zero();
                let d = driver::Driver::start(host, cpus.clone());
                (None, Some(d), Some(abufs))
            }
            other => (other, None, None),
        };
        if std::env::var("QX_DS_VERBOSE").is_ok() {
            if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
                eprintln!("device memory after load: {:.2} GiB free of {:.2} GiB", free as f64 / 1073741824.0, total as f64 / 1073741824.0);
            }
        }
        Ok(Self {
            embd: raw_mat("token_embd.weight")?,
            output,
            output_norm: vecf("output_norm.weight")?,
            g_out,
            engram_table: g.tensor("per_layer_token_embd.weight").filter(|t| !t.packed()).map(|t| W::new(g, t)),
            layers,
            s,
            st: State { rings, ring_tables, stores, sels, tokens: vec![Vec::new(); slots], pos: vec![0; slots] },
            slots,
            pool: if driver.is_some() { None } else { Some(Pool::new(&cpus)) },
            max_rows: max_ctx,
            attn_chunks: (cfg.n_swa + cfg.idx_top_k).div_ceil(12),
            ring_w,
            max_ctx,
            exact_experts: false,
            tiers: tiers.map(std::cell::RefCell::new),
            driver,
            abufs,
            engram_ring: engram_pack.as_ref().map(|_| std::cell::RefCell::new(io_uring::IoUring::new(128).expect("engram io_uring"))),
            engram_pack,
            graphs: (0..=DS_TMAX).map(|_| None).collect(),
            warmed: vec![false; DS_TMAX + 1],
            warm: false,
            t_engram: 0.0,
            t_phase: [0.0; 5],
            t_forward: 0.0,
            t_experts: 0.0,
            cfg,
            gpu,
        })
    }

    fn k(&self, entry: &'static str, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
        self.trace(entry);
        self.gpu.run(DSV41.kernel(entry, CubeDim::new_1d(threads)), grid, info);
        self.trace_done();
    }

    /// `QX_DS_TRACE=1`: name every launch and sync after it, so a device fault names its kernel.
    fn trace(&self, what: &str) {
        if std::env::var("QX_DS_TRACE").is_ok() {
            eprintln!("launch {what}");
        }
    }
    fn trace_done(&self) {
        if std::env::var("QX_DS_TRACE").is_ok() {
            self.gpu.sync();
        }
    }
    /// `QX_DS_SYNC=a,b,...`: a device sync at the named points of the forward (a race bisection aid; names:
    /// engram, post, resident, wait, combine, layer).
    fn dsync(&self, what: &str) {
        static POINTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let pts = POINTS.get_or_init(|| std::env::var("QX_DS_SYNC").map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default());
        if pts.iter().any(|p| p == what) {
            self.gpu.sync();
        }
    }
    fn mg(&self, g: &MGemv, x: u64, what: &str) {
        self.trace(what);
        g.run(&self.gpu, x);
        self.trace_done();
    }

    /// RMS norm of `t` rows of `dim` (rows `dim` apart in and out).
    fn rms(&self, x: u64, w: u64, out: u64, dim: usize, t: usize) {
        self.trace("qx_rms_groups");
        self.gpu.run(LAYER.kernel("qx_rms_groups", CubeDim::new_1d(1024)), (1, t as u32, 1), &[x, w, out, dim as u64, 0, bits(1e-20), 0, dim as u64, dim as u64]);
        self.trace_done();
    }
    /// A fused GEMV over `t` tokens whose x rows are `x_stride` floats apart (outputs per segment at y + n * rows).
    /// Always the f32 path (chunks of 8 tokens): every token's result is bit for bit a one-token step's, so windows,
    /// prefix rewinds and single steps agree; the tensor-core path's bf16 inputs would not (`QX_TMV=1` opts in).
    fn mgt(&self, g: &MGemv, x: u64, x_stride: usize, t: usize, what: &str) {
        self.trace(what);
        if std::env::var("QX_DS_TRACE").is_ok() {
            self.gpu.sync();
            eprintln!("  table {:x?} x {x:#x} stride {x_stride} t {t}", g.table_words());
        }
        let tmv = t > 8 && t <= 32 && std::env::var("QX_TMV").is_ok_and(|v| v == "1");
        g.run_t_with(&self.gpu, x, 0, 0, t, x_stride, tmv);
        self.trace_done();
    }

    fn rope(&self, x: u64, heads: usize, dim: usize, off: usize, compressed: bool, delta: i32, inverse: bool, t: usize) {
        let c = &self.cfg;
        let (base, fs, c0, c1, ext) = rope_params(c, compressed);
        self.k("qx_ds_rope", (c.n_rot / 2) as u32, (heads as u32, t as u32, 1), &[x, heads as u64, dim as u64, c.n_rot as u64, off as u64, self.s.pos.ptr, delta as u32 as u64, bits(base), bits(fs), bits(c0), bits(c1), ext, inverse as u64, (heads * dim) as u64, dim as u64]);
    }

    /// The hyper-connection mixes of the residual: pre into `pre`, post and comb into the scratch.
    fn hc_terms(&self, h: &HcDs, pre: &DeviceBuf, t: usize) {
        let hcd = self.cfg.hc * E;
        self.rms(self.s.r.ptr, self.s.ones.ptr, self.s.rn.ptr, hcd, t);
        self.k("qx_ds_hc_mixes", 256, (24, HC_SPLIT as u32, t as u32), &[h.fn_.ptr, self.s.rn.ptr, self.s.mixes.ptr, hcd as u64, 24]);
        self.k("qx_ds_hc_terms", 32, (t as u32, 1, 1), &[self.s.mixes.ptr, h.scale.ptr, h.base.ptr, pre.ptr, self.s.post.ptr, self.s.comb.ptr, bits(self.cfg.hc_eps), 20, HC_SPLIT as u64, self.s.mixsum.ptr]);
    }

    fn hc_pre(&self, pre: &DeviceBuf, t: usize) {
        self.k("qx_ds_hc_pre", 256, ((E as u32).div_ceil(256), t as u32, 1), &[self.s.r.ptr, pre.ptr, self.s.xpre.ptr, E as u64, self.cfg.hc as u64]);
    }

    fn hc_post(&self, t: usize) {
        self.k("qx_ds_hc_post", 256, ((E as u32).div_ceil(256), t as u32, 1), &[self.s.y.ptr, self.s.r.ptr, self.s.post.ptr, self.s.comb.ptr, E as u64, self.cfg.hc as u64]);
    }

    /// Runs the routed experts for the token on the CPU pool (or exactly in f32) from the GGUF and uploads the k
    /// outputs. Returns the slots to hand the combine (all "CPU").
    fn experts_gguf(&self, l: usize, x: &[f32], ids: &[i32]) {
        let c = &self.cfg;
        let ex = &self.layers[l].experts;
        let (gate, up, down) = (ex.gate.as_ref().unwrap(), ex.up.as_ref().unwrap(), ex.down.as_ref().unwrap());
        let (ff, ne) = (c.ff_exp, E);
        let (gb, db) = (ff * ex.gate_row_bytes, ne * ex.down_row_bytes);
        let mut y = vec![0f32; ids.len() * ne];
        if self.exact_experts {
            let limit = c.clamp_exp[l];
            for (a, &e) in ids.iter().enumerate() {
                let e = e as usize;
                let (mut g, mut u) = (vec![0f32; ff], vec![0f32; ff]);
                gate.matvec_rows(e * ff, x, &mut g);
                up.matvec_rows(e * ff, x, &mut u);
                let h: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| qx_cpu::math::qx_silu_mul(a.min(limit), b.clamp(-limit, limit))).collect();
                down.matvec_rows(e * ne, &h, &mut y[a * ne..(a + 1) * ne]);
            }
        } else {
            let experts: Vec<Expert> = ids
                .iter()
                .enumerate()
                .map(|(a, &e)| {
                    let e = e as usize;
                    Expert { gate: gate.data[e * gb..].as_ptr(), up: up.data[e * gb..].as_ptr(), down: down.data[e * db..].as_ptr(), out: a, tok: 0, scale: [1.0; 3] }
                })
                .collect();
            self.pool.as_ref().unwrap().run_multi(ex.fmt, &[x], ff, ex.gate_row_bytes, ex.down_row_bytes, &experts, &mut y);
        }
        self.s.yk.upload(&y);
    }

    /// The GPU's integer expert kernels for the resident selections (slot 0 entries are skipped by the kernels).
    fn launch_resident(&self, l: usize, k: usize, t: usize, slots: u64, xq: u64, xd: u64, gu: u64, out: u64) {
        let c = &self.cfg;
        let ex = &self.layers[l].experts;
        let (ff, gb) = (c.ff_exp, c.ff_exp * ex.gate_row_bytes);
        let (gsuf, dsuf) = (ops::suffix_of(ex.gate_ty), ops::suffix_of(ex.down_ty));
        let kg = static_name(format!("qx_moe_gateup_q8k_{gsuf}_w20"));
        let kd = static_name(format!("qx_moe_down_q8k_{dsuf}"));
        self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), (k * t) as u32, 1), &[slots, xq, xd, k as u64, gu, ff as u64, E as u64, ex.gate_row_bytes as u64]);
        self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(256)), ((E as u32).div_ceil(16), (k * t) as u32, 1), &[slots, gu, bits(c.clamp_exp[l]), 0, out, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64]);
    }

    /// The live expert ranking from the driver (hottest first; see `Shared::profile_req`), in `--profile` order:
    /// what to load the tiers from next time. None without the asynchronous driver or when it does not answer.
    pub fn live_profile(&self) -> Option<Vec<(usize, usize)>> {
        let d = self.driver.as_ref()?;
        *d.shared.profile.lock().unwrap() = None;
        d.shared.profile_req.store(true, std::sync::atomic::Ordering::Release);
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() < 2.0 {
            if let Some(p) = d.shared.profile.lock().unwrap().take() {
                return Some(p);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        None
    }

    /// Writes the live ranking as a profile file (`layer expert` per line, hottest first), atomically.
    pub fn save_profile(&self, path: &std::path::Path) -> Result<()> {
        let p = self.live_profile().ok_or_else(|| anyhow!("no live profile"))?;
        let mut text = String::with_capacity(p.len() * 8);
        for (l, e) in p {
            text.push_str(&format!("{l} {e}\n"));
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Tier statistics for reports: VRAM hits, RAM hits, NVMe misses, promotions, promotion time.
    pub fn tier_stats(&self) -> Option<String> {
        if let Some(d) = &self.driver {
            let sh = &d.shared;
            let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
            return Some(format!("  vram {} staged {} pcie {} ram {} nvme {} promos vram {} ram {} ({:.0} ms, pool {:.0} ms of which fetch {:.0}, stage wait {:.0})", g(&sh.vram_hits), g(&sh.staged_hits), g(&sh.mapped_hits), g(&sh.ram_hits), g(&sh.nvme_misses), g(&sh.vram_promos), g(&sh.ram_promos), g(&sh.promo_ns) as f64 / 1e6, g(&sh.pool_ns) as f64 / 1e6, g(&sh.fetch_ns) as f64 / 1e6, g(&sh.stage_ns) as f64 / 1e6));
        }
        self.tiers.as_ref().map(|t| {
            let t = t.borrow();
            format!("  vram {} ram {} nvme {} promos vram {} ram {} ({:.0} ms)", t.vram_hits, t.ram_hits, t.nvme_misses, t.vram_promos, t.ram_promos, t.promo_secs * 1e3)
        })
    }

    /// The tiered path: resident experts by the GPU's integer kernels (x quantized to Q8_K on the device), the
    /// others by the pool from RAM or NVMe records; outputs land in `gpu_out` and `yk`, the combine picks by slot.
    fn experts_tiered(&self, l: usize, x: &[f32], ids: &[i32]) -> Result<()> {
        let c = &self.cfg;
        let (ff, ne, k) = (c.ff_exp, E, ids.len());
        let ex = &self.layers[l].experts;
        let (gb, grb, drb) = (ff * ex.gate_row_bytes, ex.gate_row_bytes, ex.down_row_bytes);
        let mut t = self.tiers.as_ref().unwrap().borrow_mut();
        let slots: Vec<u64> = ids.iter().map(|&e| t.table[l * c.n_expert + e as usize]).collect();
        for (&e, &sl) in ids.iter().zip(&slots) {
            let g = l * c.n_expert + e as usize;
            t.score[g] += 1.0;
            if sl == 0 {
                t.missed.push(g);
            }
        }
        t.slots.upload(&slots);
        let n_res = slots.iter().filter(|&&p| p != 0).count();
        t.vram_hits += n_res as u64;
        let t0 = std::time::Instant::now();
        if n_res > 0 {
            self.gpu.run(MOE.kernel("qx_quant_q8k", CubeDim::new_1d(256)), ((E / 256) as u32, 1, 1), &[self.s.x.ptr, t.xq.ptr, t.xd.ptr]);
            self.launch_resident(l, k, 1, t.slots.ptr, t.xq.ptr, t.xd.ptr, t.gu.ptr, t.gpu_out.ptr);
        }
        // the misses, through the store (RAM places or NVMe staging) to the pool
        let miss: Vec<(usize, usize)> = ids.iter().enumerate().filter(|(a, _)| slots[*a] == 0).map(|(a, &e)| (a, l * c.n_expert + e as usize)).collect();
        if !miss.is_empty() {
            let gs: Vec<usize> = miss.iter().map(|m| m.1).collect();
            let in_ram = gs.iter().filter(|&&g| t.store.in_ram(g)).count();
            t.ram_hits += in_ram as u64;
            t.nvme_misses += (gs.len() - in_ram) as u64;
            let ptrs = t.store.fetch(&gs)?;
            let experts: Vec<Expert> = miss.iter().zip(&ptrs).map(|(&(a, _), &p)| Expert { gate: p, up: unsafe { p.add(gb) }, down: unsafe { p.add(2 * gb) }, out: a, tok: 0, scale: [1.0; 3] }).collect();
            let mut y = vec![0f32; k * ne];
            self.pool.as_ref().unwrap().run_multi(ex.fmt, &[x], ff, grb, drb, &experts, &mut y);
            self.s.yk.upload(&y);
        }
        self.gpu.sync();
        t.pool_secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// After a token: decay the scores, promote the hottest missed experts into VRAM (over the coldest residents
    /// of their size class) and into RAM (over the coldest RAM records), a few of each; the Qwen swapper's policy,
    /// here between tokens on the host. Everything the GPU may still touch has completed (the forward synced).
    fn adapt(&self) -> Result<()> {
        let Some(cell) = &self.tiers else { return Ok(()) };
        let mut t = cell.borrow_mut();
        if !t.adaptive {
            t.missed.clear();
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let decay = t.decay;
        for s in t.score.iter_mut() {
            *s *= decay;
        }
        let mut cand = std::mem::take(&mut t.missed);
        cand.sort_unstable();
        cand.dedup();
        cand.sort_by(|&a, &b| t.score[b].partial_cmp(&t.score[a]).unwrap());
        let ne = self.cfg.n_expert;
        let n = t.promos_per_token;
        // VRAM: the hottest candidates each take a free slot or evict the coldest resident of their class
        let mut into_vram: Vec<usize> = Vec::new();
        for &g in cand.iter().take(n) {
            let c = t.layer_class[g / ne];
            let addr = match t.classes[c].1.pop() {
                Some(a) => a,
                None => {
                    // the coldest resident of this class
                    let victim = t.holder.iter().filter(|(_, h)| t.layer_class[**h / ne] == c).min_by(|a, b| t.score[*a.1].partial_cmp(&t.score[*b.1]).unwrap()).map(|(a, h)| (*a, *h));
                    let Some((a, h)) = victim else { continue };
                    if t.score[g] <= t.score[h] * (1.0 + t.margin) {
                        continue;
                    }
                    t.table[h] = 0;
                    t.holder.remove(&a);
                    a
                }
            };
            into_vram.push(g);
            // the record: from its RAM place, else through the store's staging
            let ptr = t.store.fetch(&[g])?[0];
            let nb = t.store.record_bytes(g);
            let src = unsafe { std::slice::from_raw_parts(ptr, nb) };
            let off = (addr - t.arena.ptr) as usize;
            t.arena.upload_at(off, src);
            t.table[g] = addr;
            t.holder.insert(addr, g);
            t.vram_promos += 1;
        }
        // RAM: the hottest candidates on NVMe (not just promoted to VRAM) over the coldest RAM records
        let mut into_ram: Vec<usize> = Vec::new();
        for &g in cand.iter() {
            if into_ram.len() >= n {
                break;
            }
            if t.store.in_ram(g) || into_vram.contains(&g) {
                continue;
            }
            into_ram.push(g);
        }
        for &g in &into_ram {
            let c = t.layer_class[g / ne];
            if t.store.free_places()[c] == 0 {
                // evict the coldest RAM record of this class that is not in VRAM
                let victim = (0..t.table.len()).filter(|&h| t.store.in_ram(h) && t.layer_class[h / ne] == c && t.table[h] == 0).min_by(|&a, &b| t.score[a].partial_cmp(&t.score[b]).unwrap());
                let Some(h) = victim else { continue };
                if t.score[g] <= t.score[h] * (1.0 + t.margin) {
                    continue;
                }
                t.store.evict(h);
            }
            let skipped = t.store.promote(&[g])?;
            if skipped.is_empty() {
                t.ram_promos += 1;
            }
        }
        t.promo_secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// The Engram rows of a token: from the pack's page layout on the host when loaded from a pack.
    fn engram_row(&self, r: u64, out: &mut [f32]) {
        if let Some(t) = &self.engram_table {
            t.row(r as usize, out);
            return;
        }
        let (f, pack) = self.engram_pack.as_ref().expect("engram table");
        let rb = pack.ple.row_bytes as usize;
        let mut raw = vec![0u8; rb];
        f.read_exact_at(&mut raw, pack.ple_row(r)).expect("engram row read");
        let ty = qx_pack::ggml::GgmlType::from_id(13).unwrap(); // Q5_K
        qx_pack::dequant(ty, &raw, out);
    }

    /// Starts slot `slot` anew at position 0 (its ring, stores and pending pairs are overwritten as positions
    /// recur; the captured graphs stay valid).
    pub fn reset(&mut self, slot: usize) {
        self.gpu.sync();
        self.st.pos[slot] = 0;
        self.st.tokens[slot].clear();
    }

    /// Rows a compressed store holds per slot for `max_ctx` positions at compression `ratio`.
    fn store_rows(max_ctx: usize, ratio: usize) -> usize {
        max_ctx.div_ceil(ratio.max(1)) + 1
    }

    /// The tokens of slot `slot` so far.
    pub fn tokens(&self, slot: usize) -> &[u32] {
        &self.st.tokens[slot]
    }

    /// Slot `slot`'s position.
    pub fn pos(&self, slot: usize) -> usize {
        self.st.pos[slot]
    }

    /// Rewinds slot `slot` to position `p` (a group boundary for every compressor ratio, so an even position).
    /// The compressed stores and indexer keys are positional, so positions from `p` on are simply recomputed; the
    /// attention ring must hold the window [p - n_swa, p). If the ring still does (the rewind is within the last
    /// DS_TMAX positions), nothing is copied; otherwise the window is rebuilt by replaying the slot's own tokens
    /// from the group boundary at or below p - n_swa: identical inputs through the same deterministic kernels give
    /// the ring, the compressed rows and the keys bit for bit (about 130 tokens, one or two windows). This
    /// replaced a per-position host copy of every latent, 21.5 GB per slot at 256k (2026-11-06).
    pub fn rewind(&mut self, slot: usize, p: usize) {
        assert!(p <= self.st.pos[slot], "rewind past the end");
        let max_ratio = self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        assert!(p % max_ratio == 0, "rewind to a group boundary");
        let pos = self.st.pos[slot];
        if pos - p <= self.ring_w - self.cfg.n_swa {
            // the ring holds [p - n_swa, p) still: positions beyond p are overwritten as the slot advances
            self.st.pos[slot] = p;
            self.st.tokens[slot].truncate(p);
            return;
        }
        self.replay_to(slot, p);
    }

    /// Rebuilds slot `slot`'s attention window for position `p` by replaying its tokens from the group boundary
    /// at or below p - n_swa, and leaves the slot at `p` (tokens truncated). The slot's tokens must hold [0, p).
    fn replay_to(&mut self, slot: usize, p: usize) {
        let max_ratio = self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        let mut a = p.saturating_sub(self.cfg.n_swa);
        a -= a % max_ratio;
        let replay: Vec<u32> = self.st.tokens[slot][a..p].to_vec();
        self.st.tokens[slot].truncate(a);
        self.st.pos[slot] = a;
        let mut off = 0;
        while off < replay.len() {
            // widths whose kernels every server warms (the scheduler's set): never 5 or 7, whose kernels may not be
            // loaded and whose multi-GEMV is not bit-identical at width 5
            let left = replay.len() - off;
            let w = [128usize, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1].into_iter().find(|&w| w <= left.min(DS_TMAX)).unwrap();
            let segs: Vec<(usize, u32)> = replay[off..off + w].iter().map(|&t| (slot, t)).collect();
            let _ = self.forward_segments(&segs, None);
            off += w;
        }
        debug_assert_eq!(self.st.pos[slot], p);
    }

    /// Moves slot `slot`'s sequence out to host memory and leaves the slot empty (as after `reset`): its tokens
    /// and the used rows of its compressed stores (`pos / ratio` per store). `unpark` puts it back into any slot.
    /// The ring and the pending pairs are not kept: a restore goes through `rewind`, which rebuilds the attention
    /// window by replay at a group boundary, where no group is pending.
    pub fn park(&mut self, slot: usize) -> Parked {
        let p = self.snapshot(slot);
        self.st.tokens[slot].clear();
        self.st.pos[slot] = 0;
        p
    }

    /// Slot `slot`'s state as `park` would take it, without emptying the slot: the tokens and the used store rows.
    pub fn snapshot(&mut self, slot: usize) -> Parked {
        self.gpu.sync();
        let pos = self.st.pos[slot];
        let row = self.cfg.head_dim + self.cfg.idx_dim;
        let mut stores = Vec::with_capacity(self.cfg.n_layer);
        for l in 0..self.cfg.n_layer {
            stores.push(match &self.st.stores[l] {
                Some((store, _, _, _)) => {
                    let ratio = self.cfg.ratio[l].max(1) as usize;
                    let rows = pos / ratio;
                    store.download_at::<f32>(slot * Self::store_rows(self.max_ctx, ratio) * row * 4, rows * row)
                }
                None => Vec::new(),
            });
        }
        let tokens = self.st.tokens[slot].clone();
        Parked { tokens, stores }
    }

    /// Puts a parked sequence into slot `slot` (whatever the slot held is dropped) and rewinds it to `upto`, a
    /// group boundary at most its length (the longest prefix a new prompt shares with it). Exact: the restored
    /// slot computes what the slot that was parked would have.
    pub fn unpark(&mut self, slot: usize, p: Parked, upto: usize) {
        self.gpu.sync();
        let pos = p.tokens.len();
        assert!(upto <= pos, "unpark past the parked length");
        let row = self.cfg.head_dim + self.cfg.idx_dim;
        for l in 0..self.cfg.n_layer {
            if let Some((store, _, _, _)) = &self.st.stores[l] {
                let ratio = self.cfg.ratio[l].max(1) as usize;
                let rows = (upto.div_ceil(ratio)).min(p.stores[l].len() / row);
                store.upload_at(slot * Self::store_rows(self.max_ctx, ratio) * row * 4, &p.stores[l][..rows * row]);
            }
        }
        self.st.tokens[slot] = p.tokens;
        self.st.pos[slot] = pos;
        // the ring is empty whatever the distance: always replay the window before `upto`
        assert!(upto % self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize == 0, "unpark to a group boundary");
        self.replay_to(slot, upto);
    }

    /// One token of slot 0 at its position; returns its logits. `act` receives named activations (each one syncs).
    pub fn forward(&mut self, tok: u32, act: Option<&mut dyn FnMut(&str, Vec<f32>)>) -> Vec<f32> {
        self.forward_segments(&[(0, tok)], act)
    }

    /// A window of 1 to DS_TMAX consecutive tokens of slot 0; returns their logits, token n's row at n * n_vocab.
    pub fn forward_window(&mut self, toks: &[u32], act: Option<&mut dyn FnMut(&str, Vec<f32>)>) -> Vec<f32> {
        let segs: Vec<(usize, u32)> = toks.iter().map(|&t| (0, t)).collect();
        self.forward_segments(&segs, act)
    }

    /// A window of 1 to DS_TMAX tokens, each (slot, token), a slot's tokens consecutive and in order, at the slots'
    /// positions; returns the logits in window order, token n's row at n * n_vocab. Each token's logits are those
    /// of one-token steps. Windows of more than one token need the asynchronous expert driver; `act` (activation
    /// dumps) only with one token.
    pub fn forward_segments(&mut self, toks: &[(usize, u32)], act: Option<&mut dyn FnMut(&str, Vec<f32>)>) -> Vec<f32> {
        let tf = std::time::Instant::now();
        self.t_experts = 0.0;
        let t = toks.len();
        assert!((1..=DS_TMAX).contains(&t), "window of {t} tokens");
        assert!(t == 1 || (self.driver.is_some() && act.is_none()), "windows need the asynchronous driver and no dumps");
        // positions: each slot's tokens continue from its position, in order
        let mut positions: Vec<u32> = Vec::with_capacity(t);
        let mut slots_v: Vec<u32> = Vec::with_capacity(t);
        let mut ids: Vec<u32> = Vec::with_capacity(t);
        let mut counted: Vec<usize> = vec![0; self.slots];
        for &(sl, tok) in toks {
            assert!(sl < self.slots, "slot {sl} of {}", self.slots);
            assert!((tok as usize) < self.cfg.n_vocab, "token id {tok} outside the vocabulary ({}): the token file was made with another tokenizer", self.cfg.n_vocab);
            positions.push((self.st.pos[sl] + counted[sl]) as u32);
            slots_v.push(sl as u32);
            ids.push(tok);
            counted[sl] += 1;
            self.st.tokens[sl].push(tok);
        }
        let pos_min = positions.iter().copied().min().unwrap() as usize;
        let pos_max = positions.iter().copied().max().unwrap() as usize;
        let tp = std::time::Instant::now();
        // QX_DS_SYNC_UPLOADS=1 restores the synchronous uploads (each ends in cuCtxSynchronize) for an A/B
        static SYNC_UPLOADS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SYNC_UPLOADS.get_or_init(|| std::env::var("QX_DS_SYNC_UPLOADS").is_ok_and(|v| v == "1")) {
            self.s.tok.upload(&ids);
            self.s.pos.upload(&positions);
            self.s.slot.upload(&slots_v);
        } else {
            // the previous forward synced, so nothing reads the staging or the three buffers now; the copies run on
            // the compute stream ahead of this window's kernels (and ahead of a graph replay)
            let tm = DS_TMAX;
            self.s.in_host.slice::<u32>(0, t).copy_from_slice(&ids);
            self.s.in_host.slice::<u32>(tm * 4, t).copy_from_slice(&positions);
            self.s.in_host.slice::<u32>(2 * tm * 4, t).copy_from_slice(&slots_v);
            for (i, dst) in [&self.s.tok, &self.s.pos, &self.s.slot].into_iter().enumerate() {
                self.k("qx_ds_copy_f32", 256, ((t as u32).div_ceil(256), 1, 1), &[self.s.in_host.dptr + (i * tm * 4) as u64, dst.ptr, t as u64]);
            }
        }
        self.t_phase[0] = tp.elapsed().as_secs_f64();
        // the Engram rows of every Engram layer and token, gathered on the host into the mapped regions
        // (region ei holds token n's rows at n * cols * dim) before any launch
        let c = &self.cfg;
        let ed = c.engram_dim;
        let n_e = c.engram_cols() * ed;
        let te = std::time::Instant::now();
        let all_rows: Vec<Vec<u64>> = (0..c.engram_layers.len()).flat_map(|ei| (0..t).map(move |n| (ei, n))).map(|(ei, n)| c.engram_rows(ei, &self.st.tokens[toks[n].0], positions[n] as usize)).collect();
        if let (Some((f, pack)), Some(ring)) = (&self.engram_pack, &self.engram_ring) {
            use io_uring::{opcode, types};
            use std::os::fd::AsRawFd;
            let rb = pack.ple.row_bytes as usize;
            let flat: Vec<u64> = all_rows.iter().flatten().copied().collect();
            let mut raw = vec![0u8; flat.len() * rb];
            let mut ring = ring.borrow_mut();
            let mut next = 0;
            let mut done = 0;
            while done < flat.len() {
                while next < flat.len() && next - done < 120 {
                    let e = opcode::Read::new(types::Fd(f.as_raw_fd()), unsafe { raw.as_mut_ptr().add(next * rb) }, rb as u32).offset(pack.ple_row(flat[next])).build().user_data(next as u64);
                    unsafe { ring.submission().push(&e).expect("engram io_uring queue") };
                    next += 1;
                }
                ring.submit_and_wait(1).expect("engram io_uring submit");
                for cqe in ring.completion() {
                    assert!(cqe.result() == rb as i32, "engram row read returned {}", cqe.result());
                    done += 1;
                }
            }
            let ty = qx_pack::ggml::GgmlType::from_id(13).unwrap(); // Q5_K
            let mut i = 0;
            for (ri, rows) in all_rows.iter().enumerate() {
                let (ei, n) = (ri / t, ri % t);
                let dst = self.s.erows_host.slice::<f32>((ei * DS_TMAX + n) * n_e * 4, n_e);
                for j in 0..rows.len() {
                    qx_pack::dequant(ty, &raw[i * rb..(i + 1) * rb], &mut dst[j * ed..(j + 1) * ed]);
                    i += 1;
                }
            }
        } else {
            for (ri, rows) in all_rows.iter().enumerate() {
                let (ei, n) = (ri / t, ri % t);
                let dst = self.s.erows_host.slice::<f32>((ei * DS_TMAX + n) * n_e * 4, n_e);
                for (j, &r) in rows.iter().enumerate() {
                    self.engram_row(r, &mut dst[j * ed..(j + 1) * ed]);
                }
            }
        }
        self.t_engram = te.elapsed().as_secs_f64();
        // A captured graph per window width replays the launches (`QX_GRAPH=0` disables): only in the
        // asynchronous mode (no host work between launches), without activation dumps, and once every compressed
        // store has a visible row (pos + 1 >= the largest ratio), so the launch set is the same for every window
        // (the grids that depend on the context are sized from the maximum; the kernels bound by position). The
        // first eligible window of a width runs eagerly (the warm-up the capture needs), then is recorded without
        // running: prepare, eager run and recording stay together, as CubeCL keeps one prepared capture per stream.
        let tp = std::time::Instant::now();
        let max_ratio = c.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        // The first window of a width runs once as a dry pass: CubeCL compiles and loads a kernel on its first
        // launch, and a module load while an expert job is in flight deadlocks (the load waits for the device,
        // the device spins in a wait kernel for the driver, the driver's copies wait for the context lock the
        // load holds; seen as CubeCL's server thread in cuModuleLoadData with the GPU at 100%). The dry pass
        // launches the width's whole kernel set (positions clamped so the compressed-store kernels are included;
        // each kernel bounds itself by the true positions in `s.pos`) with the expert job replaced by zeroed slot
        // lists, then the device is drained. It writes the same ring and store positions the real pass rewrites.
        if self.driver.is_some() && !self.warmed[t] {
            if let Some(d) = &self.driver {
                d.shared.mb.slice::<f32>(crate::mailbox::mb_y(E), crate::mailbox::AMAX * E).fill(0.0);
            }
            self.warm = true;
            let p = max_ratio - 1;
            self.launch_window(t, pos_min.max(p), pos_max.max(p), None);
            self.warm = false;
            self.gpu.sync();
            self.warmed[t] = true;
        }
        let eligible = self.driver.is_some() && act.is_none() && pos_min + 1 >= max_ratio && !std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") && std::env::var("QX_DS_SYNC").is_err() && std::env::var("QX_DS_TRACE").is_err();
        if !eligible {
            self.launch_window(t, pos_min, pos_max, act);
        } else if let Some(g) = &self.graphs[t] {
            // SAFETY: every buffer the graph reads or writes is owned by `self` and outlives it.
            unsafe { g.replay().expect("graph replay") };
        } else {
            self.gpu.client.graph_prepare().expect("graph_prepare");
            self.launch_window(t, pos_min, pos_max, None);
            // Drain the eager run, so no expert job is in flight while the device records nothing, then keep the
            // driver out of its event syncs for the recording (see `Shared::capture`).
            self.gpu.sync();
            let shared = self.driver.as_ref().map(|d| d.shared.clone());
            let not_capturing = shared.as_ref().map(|s| s.capture.lock().unwrap());
            self.gpu.client.start_capture().expect("start_capture");
            self.launch_window(t, pos_min, pos_max, None);
            let g = self.gpu.client.stop_capture().expect("stop_capture");
            drop(not_capturing);
            if std::env::var("QX_DS_VERBOSE").is_ok() {
                eprintln!("window graph ({t} tokens) captured at position {pos_min}");
            }
            self.graphs[t] = Some(g);
        }
        self.t_phase[1] = tp.elapsed().as_secs_f64();
        let tp = std::time::Instant::now();
        let gpu = &self.gpu;
        gpu.sync();
        self.t_phase[2] = tp.elapsed().as_secs_f64();
        let tp = std::time::Instant::now();
        for sl in 0..self.slots {
            if counted[sl] > 0 {
                self.st.pos[sl] += counted[sl];
            }
        }
        self.t_phase[3] = tp.elapsed().as_secs_f64();
        if let Some(d) = &self.driver {
            d.shared.tokens_done.fetch_add(t as u64, std::sync::atomic::Ordering::Release);
        }
        self.adapt().expect("tier promotions");
        let tp = std::time::Instant::now();
        let out = self.s.logits.download(self.cfg.n_vocab * t);
        self.t_phase[4] = tp.elapsed().as_secs_f64();
        self.t_forward = tf.elapsed().as_secs_f64();
        out
    }

    /// Queues a window's launches (everything between the uploads and the final sync): `t` tokens whose
    /// positions lie in [pos, last] (the device arrays hold each token's own).
    fn launch_window(&mut self, t: usize, pos: usize, last: usize, mut act: Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        let c = &self.cfg;
        let (hc, hcd) = (c.hc, c.hc * E);
        let tt = t as u32;
        let gpu = &self.gpu;
        let dump = |name: String, b: &DeviceBuf, n: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>| {
            if let Some(f) = act {
                gpu.sync();
                f(&name, b.download(n));
            }
        };
        self.trace("dequant_gather");
        ops::dequant_gather(gpu, self.embd.ty, self.embd.buf.ptr, self.s.e.ptr, t, self.embd.cols, self.s.tok.ptr);
        self.trace_done();
        self.k("qx_ds_hc_init", 256, ((hcd as u32).div_ceil(256), tt, 1), &[self.s.r.ptr, self.s.e.ptr, E as u64, hc as u64]);
        dump("hc_init".into(), &self.s.r, hcd, &mut act);
        let mut have_prev = false;
        for l in 0..c.n_layer {
            let w = &self.layers[l];
            let ratio = c.ratio[l];
            if let Some((g_ekv, egate)) = &w.engram {
                let ei = c.engram_layers.iter().position(|&x| x == l).unwrap();
                let n = c.engram_cols() * c.engram_dim;
                self.k("qx_ds_copy_f32", 256, (((n * t) as u32).div_ceil(256), 1, 1), &[self.s.erows_host.dptr + (ei * DS_TMAX * n * 4) as u64, self.s.erows.ptr, (n * t) as u64]);
                self.mgt(g_ekv, self.s.erows.ptr, n, t, "engram_kv");
                dump(format!("engram_kv-{l}"), &self.s.ekv, (hc + 1) * E, &mut act);
                self.k("qx_ds_engram_gate", 256, (hc as u32, tt, 1), &[self.s.r.ptr, self.s.ekv.ptr, egate.ptr, E as u64, bits(1e-20), self.s.egates.ptr]);
                dump(format!("engram_gate-{l}"), &self.s.egates, hc, &mut act);
                dump(format!("engram_out-{l}"), &self.s.r, hcd, &mut act);
            }
            // attention
            self.hc_terms(&w.hc_attn, &self.s.pre_a, t);
            dump(format!("hc_mixes-{l}"), &self.s.mixsum, 24, &mut act);
            dump(format!("hc_pre-{l}"), &self.s.pre_a, hc, &mut act);
            dump(format!("hc_post-{l}"), &self.s.post, hc, &mut act);
            dump(format!("hc_comb-{l}"), &self.s.comb, hc * hc, &mut act);
            self.hc_pre(if have_prev { &self.s.pre_prev } else { &self.s.pre_first }, t);
            dump(format!("hc_attn_pre-{l}"), &self.s.xpre, E, &mut act);
            self.rms(self.s.xpre.ptr, w.attn_norm.ptr, self.s.x.ptr, E, t);
            dump(format!("attn_norm-{l}"), &self.s.x, E, &mut act);
            self.mgt(&w.g_x, self.s.x.ptr, E, t, "g_x");
            self.rms(self.s.qr_raw.ptr, w.q_a_norm.ptr, self.s.qr.ptr, c.q_lora, t);
            dump(format!("qr_norm-{l}"), &self.s.qr, c.q_lora, &mut act);
            self.mgt(&w.g_qr, self.s.qr.ptr, c.q_lora, t, "g_qr");
            let comp = ratio != 0;
            let nope = c.head_dim - c.n_rot;
            self.rope(self.s.q.ptr, c.n_head, c.head_dim, nope, comp, 0, false, t);
            dump(format!("q-{l}"), &self.s.q, c.n_head * c.head_dim, &mut act);
            self.rms(self.s.kv_raw.ptr, w.kv_norm.ptr, self.s.kv.ptr, c.head_dim, t);
            self.rope(self.s.kv.ptr, 1, c.head_dim, nope, comp, 0, false, t);
            dump(format!("kv-{l}"), &self.s.kv, c.head_dim, &mut act);
            self.k("qx_ds_kv_append", 256, (tt, 1, 1), &[self.s.kv.ptr, self.st.ring_tables[l].ptr, self.ring_w as u64, self.s.pos.ptr, self.s.slot.ptr, c.head_dim as u64]);
            let row = c.head_dim + c.idx_dim;
            if w.has_comp {
                let (_, st, _, pt) = self.st.stores[l].as_ref().unwrap();
                let score = if ratio > 1 { self.s.cscore.ptr } else { 0 };
                self.k("qx_ds_compress", 256, (1, 1, 1), &[self.s.ckv.ptr, score, pt.ptr, st.ptr, w.comp_norm.as_ref().unwrap().ptr, self.s.comp_tmp.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.head_dim as u64, row as u64, bits(1e-20), t as u64]);
                if (pos + 1) % ratio == 0 {
                    dump(format!("comp_latent-{l}"), &self.s.comp_tmp, c.head_dim, &mut act);
                }
                self.mgt(w.g_ik.as_ref().unwrap(), self.s.comp_tmp.ptr, c.head_dim, t, "g_ik");
                let (base, fs, c0, c1, ext) = rope_params(c, true);
                self.k("qx_ds_idx_key_commit", 128, (tt, 1, 1), &[self.s.ik.ptr, st.ptr, w.idx_k_norm.as_ref().unwrap().ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.idx_dim as u64, row as u64, c.head_dim as u64, bits(1e-20)]);
                // the latent at [448, 512) and the key at [512 + 64, 640), both at the group's first position
                self.k("qx_ds_rope_row", (c.n_rot / 2) as u32, (tt, 1, 1), &[st.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, row as u64, nope as u64, c.n_rot as u64, bits(base), bits(fs), bits(c0), bits(c1), ext]);
                self.k("qx_ds_rope_row", (c.n_rot / 2) as u32, (tt, 1, 1), &[st.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, row as u64, (c.head_dim + c.idx_dim - c.n_rot) as u64, c.n_rot as u64, bits(base), bits(fs), bits(c0), bits(c1), ext]);
            }
            // the last token's visible rows bound the window's grids; each kernel bounds its own token
            let n_visible = if comp { (last + 1) / ratio } else { 0 };
            let mut sel: Option<(u64, u64, u64)> = None; // (sel, count, store table)
            if comp && n_visible > 0 {
                let src = c.kv_src[l] as usize;
                let tsrc = c.topk_src[l] as usize;
                let store_table = self.st.stores[src].as_ref().unwrap().1.ptr;
                if w.has_idx_q {
                    self.rope(self.s.iq.ptr, c.idx_heads, c.idx_dim, c.idx_dim - c.n_rot, comp, 0, false, t);
                    let max_rows = self.max_rows as u64;
                    // QX_DS_IDX_KERNEL=1 keeps the original scorer; 2 (default) the shared-memory one, bit-identical
                    static IDX_KERNEL: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
                    let idx_entry = *IDX_KERNEL.get_or_init(|| if std::env::var("QX_DS_IDX_KERNEL").is_ok_and(|v| v == "1") { "qx_ds_idx_score" } else { "qx_ds_idx_score2" });
                    self.k(idx_entry, 256, ((self.max_rows as u32).div_ceil(8), tt, 1), &[self.s.iq.ptr, self.s.iw.ptr, store_table, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.idx_dim as u64, row as u64, c.head_dim as u64, self.s.scores.ptr, max_rows, bits(1.0 / ((c.idx_heads * c.idx_dim) as f32).sqrt())]);
                    dump(format!("idx_score-{l}"), &self.s.scores, n_visible, &mut act);
                    let (sb, cb) = self.st.sels[l].as_ref().unwrap();
                    let bs = c.cand_bs.max(1);
                    let n_blk = n_visible.div_ceil(bs);
                    let words = self.max_rows.div_ceil(bs * 32) as u64;
                    if c.cand_src == l as i32 {
                        let max_blocks = self.max_rows.div_ceil(bs) as u64;
                        self.k("qx_ds_block_max", 256, ((max_blocks as u32).div_ceil(256), tt, 1), &[self.s.scores.ptr, self.s.bmax.ptr, self.s.pos.ptr, ratio as u64, bs as u64, max_rows, max_blocks]);
                        dump(format!("cand_block_score-{l}"), &self.s.bmax, n_blk, &mut act);
                        // top blocks: the scores are the block maxima, "rows" are blocks: positions / (ratio * bs)
                        self.k("qx_ds_topk", 1024, (tt, 1, 1), &[self.s.bmax.ptr, c.cand_blocks as u64, self.s.cand_sel.ptr, self.s.cand_count.ptr, self.s.pos.ptr, (ratio * bs) as u64, max_blocks]);
                    } else if c.cand_src >= 0 && (c.cand_src as usize) < l {
                        self.k("qx_ds_zero_u32", 256, (((words * t as u64) as u32).div_ceil(256), 1, 1), &[self.s.inside.ptr, words * t as u64]);
                        self.k("qx_ds_mark_blocks", 256, ((c.cand_blocks as u32).div_ceil(256), tt, 1), &[self.s.cand_sel.ptr, self.s.cand_count.ptr, self.s.inside.ptr, c.cand_blocks as u64, words]);
                        self.k("qx_ds_cand_mask", 256, ((self.max_rows as u32).div_ceil(256), tt, 1), &[self.s.scores.ptr, self.s.inside.ptr, self.s.pos.ptr, ratio as u64, bs as u64, max_rows, words]);
                    }
                    self.k("qx_ds_topk", 1024, (tt, 1, 1), &[self.s.scores.ptr, c.idx_top_k as u64, sb.ptr, cb.ptr, self.s.pos.ptr, ratio as u64, max_rows]);
                }
                let (sb, cb) = self.st.sels[tsrc].as_ref().expect("top-k source");
                sel = Some((sb.ptr, cb.ptr, store_table));
            }
            let (sp, cp, stp) = sel.unwrap_or((0, 0, 0));
            // the attention in slices of ATTN_SLICE tokens (the partials buffer is sized for one slice); the
            // kernels index tokens from the pointers they get, so each slice passes offset pointers
            let hd = c.head_dim;
            for s0 in (0..t).step_by(ATTN_SLICE) {
                let ts = (t - s0).min(ATTN_SLICE);
                let (q, o, ps, sl) = (self.s.q.ptr + (s0 * c.n_head * hd * 4) as u64, self.s.attn.ptr + (s0 * c.n_head * hd * 4) as u64, self.s.pos.ptr + (s0 * 4) as u64, self.s.slot.ptr + (s0 * 4) as u64);
                let (sps, cps) = if sp != 0 { (sp + (s0 * c.idx_top_k * 4) as u64, cp + (s0 * 4) as u64) } else { (0, 0) };
                if std::env::var("QX_DS_ATTN").is_ok_and(|v| v == "old") {
                    self.k("qx_ds_attn", 256, (c.n_head as u32, ts as u32, 1), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64]);
                } else {
                    let chunks = self.attn_chunks as u64;
                    self.k("qx_ds_attn_part", 256, ((c.n_head / 8) as u32, chunks as u32, ts as u32), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64, self.s.attn_part.ptr, chunks]);
                    self.k("qx_ds_attn_merge", 256, (c.n_head as u32, ts as u32, 1), &[self.s.attn_part.ptr, w.sinks.ptr, o, chunks, hd as u64]);
                }
            }
            dump(format!("{}-{l}", if sel.is_some() { "attn_comp" } else { "attn_raw" }), &self.s.attn, c.n_head * c.head_dim, &mut act);
            self.rope(self.s.attn.ptr, c.n_head, c.head_dim, nope, comp, 0, true, t);
            dump(format!("attn_derope-{l}"), &self.s.attn, c.n_head * c.head_dim, &mut act);
            let gd = c.n_head / c.o_groups * c.head_dim;
            for (gi, g) in w.wo_a.iter().enumerate() {
                self.mgt(g, self.s.attn.ptr + (gi * gd * 4) as u64, c.n_head * c.head_dim, t, "wo_a");
            }
            // the groups' rows per token, contiguous for the second projection
            self.k("qx_ds_regroup", 256, (((c.o_groups * c.o_lora) as u32).div_ceil(256), tt, 1), &[self.s.oag.ptr, self.s.oa.ptr, c.o_groups as u64, c.o_lora as u64, DS_TMAX as u64]);
            dump(format!("attn_wo_a-{l}"), &self.s.oa, c.o_groups * c.o_lora, &mut act);
            self.mgt(&w.g_ob, self.s.oa.ptr, c.o_groups * c.o_lora, t, "g_ob");
            dump(format!("attn_out-{l}"), &self.s.y, E, &mut act);
            self.hc_post(t);
            dump(format!("hc_attn_post-{l}"), &self.s.r, hcd, &mut act);
            // FFN
            self.hc_terms(&w.hc_ffn, &self.s.pre_f, t);
            self.hc_pre(&self.s.pre_a, t);
            dump(format!("hc_ffn_pre-{l}"), &self.s.xpre, E, &mut act);
            self.rms(self.s.xpre.ptr, w.ffn_norm.ptr, self.s.x.ptr, E, t);
            dump(format!("ffn_norm-{l}"), &self.s.x, E, &mut act);
            self.mgt(&w.g_ffn, self.s.x.ptr, E, t, "g_ffn");
            dump(format!("ffn_moe_logits-{l}"), &self.s.logits_r, c.n_expert, &mut act);
            self.k("qx_ds_route", 32, (tt, 1, 1), &[self.s.logits_r.ptr, c.n_expert as u64, c.n_expert_used as u64, w.router_bias.ptr, self.s.ids.ptr, self.s.wts.ptr, bits(c.w_scale)]);
            let k = c.n_expert_used;
            // QX_DS_NOEXPERTS=1: skip the routed experts (a timing knob: the dense and attention cost alone)
            let no_experts = std::env::var("QX_DS_NOEXPERTS").is_ok_and(|v| v == "1");
            let mut shexp_done = false;
            let (combine_gpu, cpu_rows, slots_ptr) = if no_experts {
                self.s.yk.zero();
                (self.s.yk.ptr, self.s.yk.ptr, self.s.ones_slots.ptr)
            } else if let (Some(d), Some(ab)) = (&self.driver, &self.abufs) {
                // the misses to the driver's mailbox (x quantized to Q8_K on the way), the residents on the GPU,
                // then wait for the CPU's rows: no host involvement
                let mb = d.shared.mb.dptr;
                let stage_share = if t >= ab.stage_wide_min { ab.stage_share } else { ab.stage_share_decode };
                self.trace("qx_moe_post");
                gpu.run(STEP.kernel("qx_moe_post", CubeDim::new_1d(256)), (tt, 1, 1), &[mb, ab.table_dev.ptr, (l * c.n_expert) as u64, self.s.ids.ptr, self.s.wts.ptr, self.s.x.ptr, ab.step.ptr, l as u64, k as u64, E as u64, ab.slots.ptr, ab.xq.ptr, ab.xd.ptr, t as u64 | ((self.warm as u64) << 32), ab.rtable_dev.ptr, ab.gpu_share, stage_share, ab.slots_all.ptr, ab.slots_stg.ptr, ab.stage_cap as u64]);
                self.trace_done();
                self.dsync("post");
                self.launch_resident(l, k, t, ab.slots.ptr, ab.xq.ptr, ab.xd.ptr, ab.gu.ptr, ab.gpu_out.ptr);
                self.dsync("resident");
                // the shared expert while the pool works on the misses
                self.k("qx_ds_silu_mul_clamp", 256, (((c.ff_exp * t) as u32).div_ceil(256), 1, 1), &[self.s.shg.ptr, self.s.shu.ptr, self.s.shh.ptr, (c.ff_exp * t) as u64, bits(c.clamp_shexp[l])]);
                self.mgt(&w.g_shd, self.s.shh.ptr, c.ff_exp, t, "g_shd");
                if stage_share > 0 {
                    // the staged experts: their records and slot addresses land on the copy stream, then the
                    // same kernels over the staged slot array (zero where nothing is staged)
                    self.trace("qx_moe_wait_staged");
                    gpu.run(STEP.kernel("qx_moe_wait_staged", CubeDim::new_1d(1)), (1, 1, 1), &[mb, ab.step.ptr]);
                    self.trace_done();
                    self.launch_resident(l, k, t, ab.slots_stg.ptr, ab.xq.ptr, ab.xd.ptr, ab.gu.ptr, ab.gpu_out.ptr);
                }
                self.trace("qx_moe_wait");
                gpu.run(STEP.kernel("qx_moe_wait", CubeDim::new_1d(1)), (1, 1, 1), &[mb, ab.step.ptr]);
                self.trace_done();
                self.dsync("wait");
                shexp_done = true;
                (ab.gpu_out.ptr, mb + crate::mailbox::mb_y(E) as u64, ab.slots_all.ptr)
            } else {
                assert!(t == 1, "the host-driven expert paths take one token");
                gpu.sync();
                let ids: Vec<i32> = self.s.ids.download(k);
                let x: Vec<f32> = self.s.x.download(E);
                let te = std::time::Instant::now();
                let combine_gpu = if self.tiers.is_some() {
                    self.experts_tiered(l, &x, &ids).expect("experts");
                    self.tiers.as_ref().unwrap().borrow().gpu_out.ptr
                } else {
                    self.experts_gguf(l, &x, &ids);
                    self.s.yk.ptr
                };
                self.t_experts += te.elapsed().as_secs_f64();
                let slots_ptr = self.tiers.as_ref().map_or(self.s.ones_slots.ptr, |t| t.borrow().slots.ptr);
                (combine_gpu, self.s.yk.ptr, slots_ptr)
            };
            if !shexp_done {
                self.k("qx_ds_silu_mul_clamp", 256, (((c.ff_exp * t) as u32).div_ceil(256), 1, 1), &[self.s.shg.ptr, self.s.shu.ptr, self.s.shh.ptr, (c.ff_exp * t) as u64, bits(c.clamp_shexp[l])]);
                self.mgt(&w.g_shd, self.s.shh.ptr, c.ff_exp, t, "g_shd");
            }
            dump(format!("ffn_shexp-{l}"), &self.s.shy, E, &mut act);
            self.k("qx_ds_moe_combine", 256, ((E as u32).div_ceil(256), tt, 1), &[combine_gpu, k as u64, cpu_rows, self.s.y.ptr, E as u64, slots_ptr, self.s.wts.ptr, self.s.shy.ptr]);
            self.dsync("combine");
            dump(format!("ffn_out-{l}"), &self.s.y, E, &mut act);
            self.hc_post(t);
            dump(format!("l_out-{l}"), &self.s.r, hcd, &mut act);
            self.dsync("layer");
            std::mem::swap(&mut self.s.pre_prev, &mut self.s.pre_f);
            have_prev = true;
        }
        self.hc_pre(&self.s.pre_prev, t);
        dump("hc_out".into(), &self.s.xpre, E, &mut act);
        self.rms(self.s.xpre.ptr, self.output_norm.ptr, self.s.x.ptr, E, t);
        dump("result_norm".into(), &self.s.x, E, &mut act);
        self.mgt(&self.g_out, self.s.x.ptr, E, t, "g_out");
    }
}
