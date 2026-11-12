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
    /// The prefetch's speculative slots (2 x PRE_SLOTS records) and the device table of prestaged addresses.
    pred_stage: DeviceBuf,
    prestaged: DeviceBuf,
    rec_max: usize,
    stage_share: u64,
    stage_share_decode: u64,
    stage_wide_min: usize,
    /// The share for windows of width 2 up to `stage_wide_min` - 1 (a server's concurrent decode steps).
    stage_share_narrow: u64,
    pub store: crate::tiers::ExpertStore,
    arena: DeviceBuf,
    /// The arena's flex tail (address, bytes) that the chunk path's buffers view during a prompt.
    flex: Option<(u64, usize)>,
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
    /// The router alone, into the predictor's logits (`QX_DS_PREDICT`).
    g_route: MGemv,
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
    /// The router predictor's own scratch (`QX_DS_PREDICT`): it must write nothing the real pass reads.
    rn_p: DeviceBuf,
    mixes_p: DeviceBuf,
    mixsum_p: DeviceBuf,
    pre_p: DeviceBuf,
    post_p: DeviceBuf,
    comb_p: DeviceBuf,
    xpre_p: DeviceBuf,
    xp: DeviceBuf,
    logits_p: DeviceBuf,
    wts_p: DeviceBuf,
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
    /// Per layer: the slot's whole attention ring (`ring_w` latents), so `unpark` at the parked position needs no
    /// replay. Empty in files written before the format carried it (`QXPK0001`): those are restored by replay.
    rings: Vec<Vec<f32>>,
}

impl Parked {
    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        4 * (self.tokens.len() + self.stores.iter().map(Vec::len).sum::<usize>() + self.rings.iter().map(Vec::len).sum::<usize>())
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Writes the state to `path`: a magic, the tokens, every layer's store rows, then every layer's ring
    /// (little-endian; `QXPK0002`).
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"QXPK0002")?;
        f.write_all(&(self.tokens.len() as u64).to_le_bytes())?;
        f.write_all(bytemuck::cast_slice(&self.tokens))?;
        for part in [&self.stores, &self.rings] {
            f.write_all(&(part.len() as u64).to_le_bytes())?;
            for s in part {
                f.write_all(&(s.len() as u64).to_le_bytes())?;
                f.write_all(bytemuck::cast_slice(s))?;
            }
        }
        f.flush()
    }

    /// Reads a state `write` produced.
    pub fn read(path: &std::path::Path) -> std::io::Result<Parked> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        let has_rings = match &magic {
            b"QXPK0001" => false,
            b"QXPK0002" => true,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a QuixiInfer state file")),
        };
        let mut n = [0u8; 8];
        let rd_len = |f: &mut std::io::BufReader<std::fs::File>, n: &mut [u8; 8]| -> std::io::Result<usize> {
            f.read_exact(n)?;
            Ok(u64::from_le_bytes(*n) as usize)
        };
        let nt = rd_len(&mut f, &mut n)?;
        let mut tokens = vec![0u32; nt];
        f.read_exact(bytemuck::cast_slice_mut(&mut tokens))?;
        let mut parts: Vec<Vec<Vec<f32>>> = Vec::new();
        for _ in 0..(if has_rings { 2 } else { 1 }) {
            let nl = rd_len(&mut f, &mut n)?;
            let mut part = Vec::with_capacity(nl);
            for _ in 0..nl {
                let len = rd_len(&mut f, &mut n)?;
                let mut v = vec![0f32; len];
                f.read_exact(bytemuck::cast_slice_mut(&mut v))?;
                part.push(v);
            }
            parts.push(part);
        }
        let rings = if has_rings { parts.pop().unwrap() } else { Vec::new() };
        let stores = parts.pop().unwrap();
        Ok(Parked { tokens, stores, rings })
    }
}

/// The chunk path's per-row state (`forward_chunk`): what crosses the layer boundary (the hc streams `r`, the
/// FFN hc pre-mix of the previous layer `pre`), what the MoE phase needs for every row of the chunk at once (`x`,
/// `xq`/`xd`, `ids`/`wts`, the per-selection `gu` and `out`, the shared expert's `shy`, the FFN hc terms `post`/`comb`),
/// the token arrays, the per-expert selection lists and slot words, and the Engram rows of every row.
pub struct ChunkBufs {
    pub rows: usize,
    r: DeviceBuf,
    pre: DeviceBuf,
    post: DeviceBuf,
    comb: DeviceBuf,
    x: DeviceBuf,
    shy: DeviceBuf,
    ids: DeviceBuf,
    wts: DeviceBuf,
    xq: DeviceBuf,
    xd: DeviceBuf,
    gu: DeviceBuf,
    out: DeviceBuf,
    tok: DeviceBuf,
    pos: DeviceBuf,
    slot: DeviceBuf,
    sel: DeviceBuf,
    /// Two slot-word tables (u64 per expert), used by alternate layers: layer l + 1's prefetch job writes its words
    /// while the GPU may still be waiting on layer l's, and layer l + 2's job is posted only after layer l + 1's
    /// slab passes synced, which drains layer l.
    slots_tbl: [DeviceBuf; 2],
    erows_host: qx_gpu::mem::HostMapped,
    /// Per indexer layer: the selected rows and their count per row of the chunk (read by the layers above it,
    /// so they must outlive the slab pass), and the candidate blocks of the candidate source layer likewise.
    sels: Vec<Option<(DeviceBuf, DeviceBuf)>>,
    cand_sel: DeviceBuf,
    cand_count: DeviceBuf,
    /// The grouped kernels' quantized hidden rows per selection (`qx_moe_quant_h_sel`): ff int8 and ff/256 scales.
    hq: DeviceBuf,
    hd: DeviceBuf,
    /// A slot word that reads as "slot 0, generation 1" (the chunk kernels return at slot 0): the warm-up's target,
    /// and a batch descriptor pointing at it.
    warm_word: DeviceBuf,
    warm_desc: DeviceBuf,
    /// The batched kernels' per-expert descriptors (word address, selection list address, M, 0), n_expert of them.
    bdesc: DeviceBuf,
    /// The chunk path's later scratch (the f16 weights and inputs of the dense GEMMs, the chunk-wide FFN rows, the
    /// expert side streams' scratch), handed out by `cscratch`: in the flex tail like the rest, so it costs the
    /// expert tier nothing outside a chunk.
    extra: DeviceBuf,
    /// Whether every chunk kernel variant was compiled and loaded (`chunk_warm`).
    warmed: bool,
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
    /// The share for windows of width 2 up to `stage_wide_min` - 1 (a server's concurrent decode steps).
    stage_share_narrow: u64,
    stage_cap: usize,
    _stage: DeviceBuf,
    _pred_stage: DeviceBuf,
    prestaged: DeviceBuf,
    /// The prediction header counter (device side, see `qx_ds_pred_post`).
    pred_ctr: DeviceBuf,
    slots_all: DeviceBuf,
    slots_stg: DeviceBuf,
    /// The staged-ready words, one per stage stream (`driver::stage_ready_by_words`).
    stg_ready: DeviceBuf,
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
    /// The chunk path's buffers (`forward_chunk`), allocated at load when `QX_DS_CHUNK` is set.
    chunk: Option<ChunkBufs>,
    /// The Engram host rows' layout for the next launches: rows per Engram-layer region and the first row the
    /// window reads (DS_TMAX and 0 for windows; the chunk's rows and the slab's first row in chunk mode).
    erows_stride: usize,
    erows_row0: usize,
    /// The chunk jobs' generation counter (see `ChunkJob::gen`) and the global expert ordinal (`ChunkJob::ord0`).
    chunk_gen: u64,
    /// Bytes of `ChunkBufs::extra` handed out so far.
    chunk_extra_off: usize,
    /// The chunk-wide FFN scratch of `QX_DS_CHUNK_GEMM=1` (for `rows` rows): router logits, shared-expert gate, up and
    /// hidden rows, then the f16 inputs and the f16 weights of one matrix.
    chunk_ffn_bufs: Option<(usize, [DeviceBuf; 6])>,
    /// `QX_DS_CHUNK_GEMM=1`: the f16 weights of the current chunk layer's dense groups (by group address: offset in
    /// `blas_bufs.0`), decoded once per layer and chunk; `blas_bufs.1` holds a slab's f16 inputs.
    blas_map: std::cell::RefCell<std::collections::HashMap<usize, u64>>,
    blas_bufs: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
    /// Output scratch for the cuBLAS dry pass.
    blas_y: Option<DeviceBuf>,
    /// Per side stream of `QX_DS_CHUNK_STREAMS`: the expert scratch (gate/up rows, quantized hidden rows, scales).
    chunk_side: Vec<[DeviceBuf; 3]>,
    chunk_ord: u64,
    /// Set while `replay_to` rebuilds the rings: the windows skip the compressed-store and index-key writes (the
    /// stores already hold the exact rows; the replay's early positions are not exact and must not overwrite or
    /// be read back from them) and launch eagerly (no graph has that kernel set).
    replaying: bool,
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
    /// Per width: the graph of the KV-only variant (layers up to the last KV source, see `prompt_keep`).
    graphs_kv: Vec<Option<cubecl_runtime::client::Graph>>,
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

/// `QX_DS_SCHED_TRACE=path`: every engine call that advances or rewinds a slot is appended to `path` (`R slot`,
/// `W slot pos`, `F kv slot:tok,...`, `C slot kv tok,...`, `L` for a chunk release), each forward followed by `H` and
/// one FNV-1a hash per returned logits row. `qx-ds-replay` re-executes such a file and compares the hashes: a server
/// run's exact schedule, replayed deterministically (2026-11-09, for the late-token flip under three streams).
pub fn sched_trace_on() -> bool {
    sched_trace_file().is_some()
}

fn sched_trace_file() -> Option<&'static std::sync::Mutex<std::fs::File>> {
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    F.get_or_init(|| std::env::var("QX_DS_SCHED_TRACE").ok().map(|p| std::sync::Mutex::new(std::fs::File::create(&p).expect("QX_DS_SCHED_TRACE")))).as_ref()
}

/// `QX_DS_TRACE_LAYERS=1` (with `QX_DS_SCHED_TRACE`): windows run eagerly and every layer appends `Y layer` with a
/// hash of each row's residual streams, and the driver appends `S layer` with the selections it served on the pool
/// and by staging: enough to name the layer and the expert where two runs of one schedule part ways.
pub fn layer_trace_on() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *L.get_or_init(|| sched_trace_on() && std::env::var("QX_DS_TRACE_LAYERS").is_ok_and(|v| v == "1"))
}

pub fn sched_trace(line: &str) {
    if let Some(f) = sched_trace_file() {
        use std::io::Write;
        let mut f = f.lock().unwrap();
        let _ = writeln!(f, "{line}");
    }
}

/// One FNV-1a hash (hex) per row of `rows` equal rows of `out` (empty when `out` is).
pub fn row_hashes(out: &[f32], rows: usize) -> String {
    if out.is_empty() || rows == 0 {
        return String::new();
    }
    let nv = out.len() / rows;
    out.chunks(nv.max(1)).map(|r| {
        let mut h = 0xcbf29ce484222325u64;
        for v in r {
            h = (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }).collect::<Vec<_>>().join(" ")
}

/// `QX_DS_PREDICT`: 0 off, 1 `pre` (the layer's router on the pre-attention residual), 2 `next` (the next layer's
/// router on this layer's FFN input), 3 `mix` (`pre` computed after the layer's own attention-side hc terms, whose
/// pre-mix it reuses instead of recomputing: the same guess on engram-free layers, at `next`'s cost). Scored in the stats line. `pre` is also what the prefetch (`QX_DS_PREFETCH`)
/// consumes, so it is on whenever the prefetch is.
pub fn predict_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("QX_DS_PREDICT").as_deref() {
        Ok("pre") | Ok("1") => 1,
        Ok("next") | Ok("2") => 2,
        Ok("mix") | Ok("3") => 3,
        _ => if prefetch_k() > 0 { 1 } else { 0 },
    })
}

/// `QX_DS_PREFETCH=K`: the driver copies the predicted top-K experts per token that are RAM-resident but not in
/// VRAM into speculative staging slots during the layer's attention + dense phase, and `qx_moe_post` serves them
/// from there (presuppositions row 5: 76-88% of the host-served selections are predicted). 0 (default): off.
/// Width-1 and small windows only (<= 8 tokens): wider windows are bytes-bound and wrong guesses cost bytes.
pub fn prefetch_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("QX_DS_PREFETCH").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(crate::mailbox::PRED_K))
}

/// Speculative staging slots per layer parity (two parities: a layer's slots are reused two layers later).
pub const PRE_SLOTS: usize = 24;

/// The CPUs this process may run on (`sched_getaffinity`), ascending.
pub fn allowed_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return (0..std::thread::available_parallelism().map_or(16, |n| n.get())).collect();
        }
        (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect()
    }
}

/// The SMT siblings of `cpu` (itself excluded), from sysfs; empty without SMT or topology files.
pub fn siblings_of(cpu: usize) -> Vec<usize> {
    let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")) else { return Vec::new() };
    parse_cpu_list(s.trim()).into_iter().filter(|&c| c != cpu).collect()
}

fn parse_cpu_list(spec: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in spec.split(',') {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                v.extend(a..=b);
            }
        } else if let Ok(c) = part.trim().parse::<usize>() {
            v.push(c);
        }
    }
    v
}

/// The machine's physical cores this process may use, as the lowest allowed CPU id of each core (sysfs
/// `thread_siblings_list`, SMT siblings folded; offline CPUs and CPUs outside the affinity mask skipped), ascending.
pub fn physical_cores() -> Vec<usize> {
    static C: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    C.get_or_init(physical_cores_uncached).clone()
}

fn physical_cores_uncached() -> Vec<usize> {
    let allowed = allowed_cpus();
    let mut cores: Vec<usize> = Vec::new();
    for &cpu in &allowed {
        let first = match std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")) {
            Ok(s) => parse_cpu_list(s.trim()).into_iter().filter(|c| allowed.contains(c)).min().unwrap_or(cpu),
            Err(_) => cpu,  // no topology file (a container, a VM): every allowed CPU counts as a core
        };
        if first == cpu {
            cores.push(cpu);
        }
    }
    if cores.is_empty() {
        cores = allowed;
    }
    cores
}

/// The CPU the engine thread is pinned to (its sync spins must not share a physical core with a pool worker: on
/// this box that cost the pool 7% of decode256k, 2026-11-08). Default: the last physical core, which `pool_cpus`
/// leaves out; `QX_DS_MAIN_CPU=N` picks another, `QX_DS_MAIN_CPU=none` leaves the thread unpinned.
pub fn main_cpu() -> Option<usize> {
    static M: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *M.get_or_init(main_cpu_uncached)
}

fn main_cpu_uncached() -> Option<usize> {
    match std::env::var("QX_DS_MAIN_CPU") {
        Ok(v) if v == "none" || v == "-1" => None,
        Ok(v) => v.parse().ok(),
        Err(_) => {
            let cores = physical_cores();
            let pool = pool_cpus();
            cores.last().copied().filter(|c| !pool.contains(c) && cores.len() >= 4)
        }
    }
}

/// The pool's CPUs: `QX_DS_CPUS` as `a-b` or `a,b,c`, else every physical core but the last (the engine thread's,
/// `main_cpu`), at most 16 (the pool is near the DRAM ceiling at 16 cores: 43 of ~52-59 GB/s; more workers mean
/// more barrier waiters, not more bandwidth); one worker per physical core, never the SMT siblings (measured 1.5x
/// slower, 2026-11-08). The first is the driver thread, which is pool thread 0.
pub fn pool_cpus() -> Vec<usize> {
    // cached at first use: the topology is read from the calling thread's affinity, and `pin_engine_early` narrows
    // that to one CPU before the engine loads (a pool computed after the pin was one core, 2026-11-08)
    static P: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    P.get_or_init(pool_cpus_uncached).clone()
}

fn pool_cpus_uncached() -> Vec<usize> {
    let spec = std::env::var("QX_DS_CPUS").unwrap_or_else(|_| {
        let cores = physical_cores();
        let n = if cores.len() >= 4 { (cores.len() - 1).min(16) } else { cores.len() };
        cores[..n].iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
    });
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

/// Pins the calling thread to one CPU; false (with a warning) when the kernel refuses (the CPU is outside the
/// process's cpuset: then the thread keeps its previous mask).
pub fn pin_thread(cpu: usize) -> bool {
    set_affinity(0, &[cpu])
}

/// Sets thread `tid`'s (0: the caller's) affinity to `cpus`; false with a warning on failure.
fn set_affinity(tid: i32, cpus: &[usize]) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            eprintln!("cpus: sched_setaffinity({tid}, {cpus:?}) failed: {} (the process's cpuset excludes it?); thread left where it was", std::io::Error::last_os_error());
            return false;
        }
        true
    }
}

/// Pins the engine thread before CUDA initialises (call it first thing in the process or the engine thread, before
/// any CUDA use), after building rayon's global pool on every allowed core so the tier fill keeps its parallelism.
/// Measured 2026-11-08: CUDA's worker thread (`DSD-0-0`) created under a one-CPU mask stays idle while the engine
/// decodes (decode256k 42.0-42.7); created with the wide mask and moved later, it spins beside the engine thread
/// (44.2-44.7). The rayon workers spawned here keep the wide mask (threads inherit affinity at creation).
pub fn pin_engine_early() {
    // read (and cache) the topology, the pool's CPUs and the engine's CPU before the mask narrows
    let cores = physical_cores();
    let pool = pool_cpus();
    let Some(main) = main_cpu() else { return };
    // the host process's own threads (a server's main, HTTP and scheduler threads): `place_threads` leaves them alone
    let _ = HOST_TIDS.set(task_tids());
    debug_assert!(!pool.contains(&main));
    let _ = rayon::ThreadPoolBuilder::new().num_threads(cores.len().max(1)).build_global();
    let ok = pin_thread(main);
    if std::env::var("QX_DS_VERBOSE").is_ok() || !ok {
        eprintln!("cpus: engine thread on {main}{} before CUDA starts; rayon on {} cores", if ok { "" } else { " (pin failed)" }, cores.len());
    }
}

/// Places the engine's threads before the driver and its pool start: the calling (engine) thread on `main_cpu`,
/// every other thread that exists at this point (CUDA's driver and event threads, rayon's fill workers, a server's
/// HTTP and scheduler threads) on that core's SMT siblings (`QX_DS_HELPERS=free` leaves them unpinned; the sibling
/// placement is the default so that nothing spins on a pool worker's core). Done after the tier fill, so the fill
/// ran on every core, and before the pool exists, so pool workers (which pin themselves) are untouched. The old
/// form pinned the engine thread before CUDA initialised, which confined CUDA's threads and the fill to one CPU
/// (2026-11-08: the tier fill 8 -> 12 s, warm-up 15 -> 22 s).
/// Threads that existed before `pin_engine_early` belong to the host process (in the server: the main thread and
/// the HTTP runtime's workers) and are NOT moved unless `QX_DS_HELPERS=all`: on the engine's CPU they time-share
/// with a thread that spins for the whole request, and the server's time to first token at concurrency 3 went
/// bimodal (1.8 s or 2.8 s at 8k, 1.9 or 3.4 at 256k; 2026-11-09, the breaker's finding on the placement).
static HOST_TIDS: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();

fn task_tids() -> Vec<i32> {
    std::fs::read_dir("/proc/self/task").map(|rd| rd.flatten().filter_map(|e| e.file_name().to_string_lossy().parse::<i32>().ok()).collect()).unwrap_or_default()
}

pub fn place_threads() {
    let Some(main) = main_cpu() else { return };
    let pool = pool_cpus();  // before the pin narrows this thread's mask (physical_cores reads it)
    let mode = std::env::var("QX_DS_HELPERS").unwrap_or_default();
    let helpers_free = mode == "free";
    let host: &[i32] = if mode == "all" { &[] } else { HOST_TIDS.get().map_or(&[], |v| v.as_slice()) };
    let sib = siblings_of(main);
    // `same` (default): the helpers time-share the engine's own CPU (what the first placement did by accident and
    // what measured best: the spinning CUDA thread and the engine thread throttle each other); `sibling`: on the
    // engine core's SMT siblings (44.2); `free`: left unpinned (44.5)
    // measured 2026-11-08 (decode256k, three repeats each): same 42.0, sibling 44.2, free 44.5: `same` is the default
    let helper_cpus: Vec<usize> = if mode == "sibling" && !sib.is_empty() { sib } else { vec![main] };
    let me = unsafe { libc::gettid() };
    let (mut moved, mut kept) = (0usize, 0usize);
    if !helpers_free {
        for tid in task_tids() {
            if tid == me {
                continue;
            }
            if host.contains(&tid) {
                kept += 1;
            } else if set_affinity(tid, &helper_cpus) {
                moved += 1;
            }
        }
    }
    let ok = pin_thread(main);
    if std::env::var("QX_DS_VERBOSE").is_ok() || !ok {
        eprintln!("cpus: pool on {pool:?}, engine thread on {main}{}, {moved} other threads on {:?}, {kept} host threads left alone", if ok { "" } else { " (pin failed)" }, if helpers_free { vec![] } else { helper_cpus.clone() });
    }
}

/// Whether the chunk path's dense products run as half-precision tensor-core GEMMs with f32 accumulation (cuBLAS;
/// the weights decoded to f16 once per layer and chunk; the router and the vocabulary head stay f32): the default
/// since 2026-11-10 (the user's contract change: prompt processing KL-bounded against the exact path, as Strata's
/// is). `QX_DS_PREFILL_EXACT=1` (or `QX_DS_CHUNK_GEMM=0`) keeps the exact kernels, bit-identical to single steps.
/// Windows (`QX_DS_WINDOW`, the server's short prompts) are always exact. 16k prompt in 4096-row chunks: 2.17 ->
/// 1.78 ms per token (461 -> 562 tok/s); the 63 positions after the prompt: KL 0.0084, top-1 96.8%.
fn chunk_gemm() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("QX_DS_PREFILL_EXACT").is_ok_and(|v| v == "1") && !std::env::var("QX_DS_CHUNK_GEMM").is_ok_and(|v| v == "0"))
}

/// The 4-byte units of `ChunkBufs::extra` for chunks of `rows` rows: the chunk-wide FFN rows and their f16 inputs
/// and weights, one layer's attention-side dense weights as f16 with a slab's f16 inputs and outputs, and eight
/// expert side streams' scratch (an estimate from the shapes, with slack; `cscratch` falls back to an allocation).
fn chunk_extra_units(c: &Config, rows: usize) -> usize {
    let ff = c.ff_exp;
    let ffn = rows * (c.n_expert + 3 * ff) + rows * E.max(ff) / 2 + ff * E / 2;
    let qd = c.n_head * c.head_dim;
    let od = c.o_groups * c.o_lora;
    let layer_halves = (c.q_lora + 2 * c.head_dim + 1024) * E + qd * c.q_lora + qd * c.o_lora + E * od;
    let blas = layer_halves / 2 + DS_TMAX * qd.max(E).max(od) / 2 + DS_TMAX * qd;
    let side = 8 * chunk_side_rows(rows) * (2 * ff + ff / 4 + ff / 256);
    (ffn + blas + side) * 21 / 20 + 4096
}

fn verbose_types() -> bool {
    std::env::var("QX_DS_VERBOSE").is_ok()
}

/// Rows of expert scratch per side stream: an eighth of the chunk, at least 256.
fn chunk_side_rows(rows_max: usize) -> usize {
    (rows_max / 8).max(256)
}

/// `QX_DS_CHUNK_STREAMS=N`: the chunk path's experts over N side streams (default 8; 1: the main stream only; at
/// most 16). Measured 2026-11-10 on a 16k prompt in 4096-row chunks: 2.77 ms per token at 1, 2.38 at 2, 2.17 at 8,
/// 2.18 at 12 and 16, the logits after the prompt bit-identical at every count; prefill256k 5.8 -> 5.1.
fn chunk_streams() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QX_DS_CHUNK_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(8).clamp(1, 16))
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
            rn_p: buf(hcd * tm),
            mixes_p: buf(24 * HC_SPLIT * tm),
            mixsum_p: buf(24 * tm),
            pre_p: buf(hc * tm),
            post_p: buf(hc * tm),
            comb_p: buf(hc * hc * tm),
            xpre_p: buf(E * tm),
            xp: buf(E * tm),
            logits_p: buf(cfg.n_expert * tm),
            wts_p: buf(crate::mailbox::PRED_K * tm),
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
            let g_route = MGemv::new(&[router.seg(&s.logits_p)]);
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
                g_route,
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
        // QX_DS_CHUNK=N: the chunk path's buffers for N rows. With the tiers they are views into the VRAM arena's tail
        // (the coldest records' slots: evicted for a prompt's duration, back to the tiers after; `flex_check`), so a
        // prompt borrows the memory instead of taking it from the expert tier for good. Their size is known here.
        let chunk_rows = std::env::var("QX_DS_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
        let chunk_sel_fallback = std::env::var("QX_DS_CHUNK_GROUPED").is_ok_and(|v| v == "0");
        let chunk_sizes = |rows: usize| -> Vec<usize> {
            let k = cfg.n_expert_used;
            let mut v = vec![rows * hcd, rows * hc, rows * hc, rows * hc * hc, rows * E, rows * E, rows * k, rows * k, rows * E / 4, rows * (E / 256),
                             rows * if chunk_sel_fallback { k } else { 1 } * 2 * cfg.ff_exp, rows * k * E, rows, rows, rows, rows * k, cfg.cand_blocks.max(1) * rows, rows,
                             rows * if chunk_sel_fallback { k } else { 1 } * cfg.ff_exp / 4, rows * if chunk_sel_fallback { k } else { 1 } * (cfg.ff_exp / 256)];
            for l in 0..cfg.n_layer { if cfg.topk_src[l] == l as i32 { v.push(cfg.idx_top_k * rows); v.push(rows); } }
            v.push(chunk_extra_units(&cfg, rows));
            v.iter().map(|&n| (n * 4 + 255) / 256 * 256).collect()
        };
        let flex_bytes: usize = chunk_rows.map(|rows| chunk_sizes(rows).iter().sum()).unwrap_or(0);
        // (built once the tiers exist: the big buffers are views into the arena's flex tail when the tiers have one,
        // else their own allocations)
        let mk_chunk = |rows: usize, flex: Option<(u64, usize)>| -> ChunkBufs {
            let k = cfg.n_expert_used;
            let sel_fallback = chunk_sel_fallback;
            let mut off = 0usize;
            let mut fbuf = |n: usize| -> DeviceBuf {
                match flex {
                    Some((fp, fb)) => {
                        let bytes = (n * 4 + 255) / 256 * 256;
                        assert!(off + bytes <= fb, "chunk buffers overflow the flex tail");
                        let v = DeviceBuf::view(fp + off as u64, n * 4);
                        off += bytes;
                        v
                    }
                    None => buf(n),
                }
            };
            ChunkBufs {
                rows,
                r: fbuf(rows * hcd),
                pre: fbuf(rows * hc),
                post: fbuf(rows * hc),
                comb: fbuf(rows * hc * hc),
                x: fbuf(rows * E),
                shy: fbuf(rows * E),
                ids: fbuf(rows * k),
                wts: fbuf(rows * k),
                xq: fbuf(rows * E / 4),
                xd: fbuf(rows * (E / 256)),
                // gate/up and the quantized hidden rows are a per-pass scratch indexed by a selection's place in
                // its expert's list (at most `rows` of them in one pass); only the per-selection fallback
                // (QX_DS_CHUNK_GROUPED=0) indexes them by selection id and needs k times as much
                gu: fbuf(rows * if sel_fallback { k } else { 1 } * 2 * cfg.ff_exp),
                out: fbuf(rows * k * E),
                tok: fbuf(rows),
                pos: fbuf(rows),
                slot: fbuf(rows),
                sel: fbuf(rows * k),
                slots_tbl: [buf(2 * cfg.n_expert), buf(2 * cfg.n_expert)],
                erows_host: qx_gpu::mem::HostMapped::new(cfg.engram_layers.len() * rows * cfg.engram_cols() * cfg.engram_dim * 4),
                sels: (0..cfg.n_layer).map(|l| (cfg.topk_src[l] == l as i32).then(|| (fbuf(cfg.idx_top_k * rows), fbuf(rows)))).collect(),
                cand_sel: fbuf(cfg.cand_blocks.max(1) * rows),
                cand_count: fbuf(rows),
                hq: fbuf(rows * if sel_fallback { k } else { 1 } * cfg.ff_exp / 4),
                hd: fbuf(rows * if sel_fallback { k } else { 1 } * (cfg.ff_exp / 256)),
                extra: fbuf(chunk_extra_units(&cfg, rows)),
                warm_word: DeviceBuf::from_slice(&[1u64 << 48]),
                warm_desc: DeviceBuf::new(32),
                bdesc: DeviceBuf::new(rows * k * 32),
                warmed: false,
            }
        };
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
                // The window share applies from width 4 (QX_DS_STAGE_WIDE_MIN; it was 8 until 2026-11-06, then 2 until
                // 2026-11-10). Width 1 keeps the decode share: a staged expert's DMA costs about 0.7 ms against the
                // pool's 0.22 when it serves one token, and 2/8, 3/8, 5/8 measured 41.7, 41.8, 42.6 ms on decode256k.
                // Widths 2 and 3 (a 3-slot server's concurrent decode steps) have their own share
                // (QX_DS_STAGE_SHARE_NARROW, 4/8): on the window share the 15-core pool idled (136 staged records per
                // token, 0.1 pool) and the step was bound by the DMA lane; serve256k3 measured 106.3 ms at 8/8, 97.2 at
                // 3/8, 95.3 at 4/8, 96.1 at 5/8, 98.9 at 6/8, and concurrency 2 74 -> 67 (perf/optimization_status.md,
                // 2026-11-10). Width 4 and up is unmeasured since the pool got its own cores: it keeps the window share.
                let stage_wide_min: usize = std::env::var("QX_DS_STAGE_WIDE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
                let stage_share_narrow = knob("QX_DS_STAGE_SHARE_NARROW", 4);
                let stage_cap: usize = if stage_share > 0 || stage_share_decode > 0 { std::env::var("QX_DS_STAGE_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(256) } else { 1 };
                let max_rec = (0..cfg.n_layer).map(|l| pack.experts.record[l] as usize).max().unwrap_or(0);
                let stage = DeviceBuf::new(stage_cap * max_rec.max(256));
                let pred_stage = DeviceBuf::new(if prefetch_k() > 0 { 2 * PRE_SLOTS * max_rec } else { 256 });
                let prestaged = DeviceBuf::new(cfg.n_layer * n_expert * 8);  // zeroed: nothing prestaged
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
                // the flex tail: the chunk buffers' bytes at the end of the arena; the records that fill it are the coldest
                let arena_bytes = used + sizes.iter().sum::<usize>() * spare;
                let flex = if flex_bytes > 0 && arena_bytes >= flex_bytes { Some(flex_bytes) } else { None };
                let arena = DeviceBuf::new(arena_bytes);
                let flex = flex.map(|fb| (arena.ptr + (arena_bytes - fb) as u64, fb));
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
                    pred_stage,
                    prestaged,
                    rec_max: max_rec,
                    stage_share,
                    stage_share_decode,
                    stage_wide_min,
                    stage_share_narrow,
                    store,
                    arena,
                    flex,
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
        let chunk = chunk_rows.map(|rows| mk_chunk(rows, tiers.as_ref().and_then(|t| t.flex)));
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
                let stg_ready = DeviceBuf::new(8 * 8);
                stg_ready.zero();
                let host = driver::Host {
                    store: t.store,
                    table_dev: table_dev.ptr,
                    table: t.table,
                    rtable_dev: rtable_dev.ptr,
                    rtable,
                    gpu_reads_ram: gpu_share > 0,
                    stage_base: t.stage.ptr,
                    stage_cap: t.stage_cap,
                    pred_stage_base: t.pred_stage.ptr,
                    pre_slots: if prefetch_k() > 0 { PRE_SLOTS } else { 0 },
                    prestaged_dev: t.prestaged.ptr,
                    rec_max: t.rec_max,
                    slots_stg_dev: slots_stg.ptr,
                    slots_all_dev: slots_all.ptr,
                    stg_ready_dev: stg_ready.ptr,
                    arena_ptr: t.arena.ptr,
                    flex_ptr: t.flex.map(|f| f.0).unwrap_or(0),
                    flex_bytes: t.flex.map(|f| f.1).unwrap_or(0),
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
                let abufs = AsyncBufs { _arena: t.arena, table_dev, rtable_dev, gpu_share, stage_share: t.stage_share, stage_share_decode: t.stage_share_decode, stage_share_narrow: t.stage_share_narrow, stage_wide_min: t.stage_wide_min, stage_cap: t.stage_cap, _stage: t.stage, _pred_stage: t.pred_stage, prestaged: t.prestaged, pred_ctr: DeviceBuf::new(8), slots_all, slots_stg, stg_ready, step: DeviceBuf::new(crate::mailbox::STEP_ROW * 4), slots: t.slots, gpu_out: t.gpu_out, xq: t.xq, xd: t.xd, gu: t.gu };
                abufs.step.zero();
                place_threads();
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
            chunk,
            erows_stride: DS_TMAX,
            erows_row0: 0,
            chunk_gen: 0,
            chunk_extra_off: 0,
            chunk_ffn_bufs: None,
            blas_map: Default::default(),
            blas_bufs: None,
            blas_y: None,
            chunk_side: Vec::new(),
            chunk_ord: 0,
            replaying: false,
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
            graphs_kv: (0..=DS_TMAX).map(|_| None).collect(),
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
        if let Some(&base) = self.blas_map.borrow().get(&(g as *const MGemv as usize)) {
            // the chunk path with QX_DS_CHUNK_GEMM=1: this layer's weights are in f16 already
            let bb = self.blas_bufs.as_ref().unwrap();
            g.run_blas_cached(&self.gpu, x, x_stride, t, bb.1.ptr, base, bb.2.ptr, bb.2.bytes);
            self.trace_done();
            return;
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

    /// Router prediction (`QX_DS_PREDICT=pre|next`, a measurement; presuppositions row 5): the layer's routing
    /// guessed before its experts are known, written to the mailbox's prediction region for the driver to score
    /// against the real selections (`pred6/8/12` in the stats line). Writes only the predictor's own scratch and
    /// that region: the real pass's bits are untouched.
    /// `pre`: layer `l`'s own router on the residual before `l`'s attention (the FFN pre-mix terms `pre_a` are
    /// computable there; what is missing is the attention's update of `r`). The guess is available one attention +
    /// dense phase ahead of the real routing.
    /// `shared` (`QX_DS_PREDICT=mix`): called from `layer_attn` right after the real `hc_terms` wrote `pre_a` (the
    /// FFN input's pre-mix, computed on the residual before the attention's update): only the mix application, the
    /// norm and the router are the predictor's own.
    fn predict_pre(&self, l: usize, t: usize, shared: bool) {
        let Some(d) = self.driver.as_ref() else { return };
        if prefetch_k() > 0 && t > 8 {
            return;
        }
        let (c, w, s) = (&self.cfg, &self.layers[l], &self.s);
        let hcd = c.hc * E;
        if shared {
            self.k("qx_ds_hc_pre", 256, ((E as u32).div_ceil(256), t as u32, 1), &[s.r.ptr, s.pre_a.ptr, s.xpre_p.ptr, E as u64, c.hc as u64]);
        } else {
            self.rms(s.r.ptr, s.ones.ptr, s.rn_p.ptr, hcd, t);
            self.k("qx_ds_hc_mixes", 256, (24, HC_SPLIT as u32, t as u32), &[w.hc_attn.fn_.ptr, s.rn_p.ptr, s.mixes_p.ptr, hcd as u64, 24]);
            self.k("qx_ds_hc_terms", 32, (t as u32, 1, 1), &[s.mixes_p.ptr, w.hc_attn.scale.ptr, w.hc_attn.base.ptr, s.pre_p.ptr, s.post_p.ptr, s.comb_p.ptr, bits(c.hc_eps), 20, HC_SPLIT as u64, s.mixsum_p.ptr]);
            self.k("qx_ds_hc_pre", 256, ((E as u32).div_ceil(256), t as u32, 1), &[s.r.ptr, s.pre_p.ptr, s.xpre_p.ptr, E as u64, c.hc as u64]);
        }
        self.rms(s.xpre_p.ptr, w.ffn_norm.ptr, s.xp.ptr, E, t);
        self.mgt(&w.g_route, s.xp.ptr, E, t, "route_pred");
        let dst = d.shared.mb.dptr + crate::mailbox::mb_pred(E, l) as u64;
        self.k("qx_ds_route", 32, (t as u32, 1, 1), &[s.logits_p.ptr, c.n_expert as u64, crate::mailbox::PRED_K as u64, w.router_bias.ptr, dst, s.wts_p.ptr, bits(c.w_scale)]);
        if prefetch_k() > 0 && !self.warm {
            // (never from a dry pass: its routing is not a real job's)
            // tell the driver: layer and tokens; the kernel adds a device-side counter so every prediction is
            // distinct, also when this launch is replayed from a window's CUDA graph
            if let Some(ab) = self.abufs.as_ref() {
                // the kernel lives in step.cu (STEP), not dsv41.cu, which `self.k` launches from
                self.trace("qx_ds_pred_post");
                self.gpu.run(STEP.kernel("qx_ds_pred_post", CubeDim::new_1d(1)), (1, 1, 1), &[d.shared.mb.dptr, l as u64 | ((t as u64) << 8), ab.pred_ctr.ptr]);
                self.trace_done();
            }
        }
    }

    /// `next`: layer `l + 1`'s router on layer `l`'s FFN input (`xpre`, before `l`'s experts; what is missing is
    /// `l`'s FFN output and `l + 1`'s attention). Available a whole expert phase plus an attention ahead.
    fn predict_next(&self, l: usize, t: usize) {
        let Some(d) = self.driver.as_ref() else { return };
        if l + 1 >= self.layers.len() {
            return;
        }
        let (c, w1, s) = (&self.cfg, &self.layers[l + 1], &self.s);
        self.rms(s.xpre.ptr, w1.ffn_norm.ptr, s.xp.ptr, E, t);
        self.mgt(&w1.g_route, s.xp.ptr, E, t, "route_pred");
        let dst = d.shared.mb.dptr + crate::mailbox::mb_pred(E, l + 1) as u64;
        self.k("qx_ds_route", 32, (t as u32, 1, 1), &[s.logits_p.ptr, c.n_expert as u64, crate::mailbox::PRED_K as u64, w1.router_bias.ptr, dst, s.wts_p.ptr, bits(c.w_scale)]);
        if prefetch_k() > 0 && !self.warm && t <= 8 {
            // the prefetch on the cheap predictor: one norm and one router GEMV, a whole expert phase of lead
            if let Some(ab) = self.abufs.as_ref() {
                self.trace("qx_ds_pred_post");
                self.gpu.run(STEP.kernel("qx_ds_pred_post", CubeDim::new_1d(1)), (1, 1, 1), &[d.shared.mb.dptr, (l + 1) as u64 | ((t as u64) << 8), ab.pred_ctr.ptr]);
                self.trace_done();
            }
        }
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
            let calls = g(&sh.tl_calls).max(1) as f64;
            let us = |a: &std::sync::atomic::AtomicU64| g(a) as f64 / 1e3 / calls;
            let pre = if g(&sh.pre_issued) > 0 || g(&sh.pre_used) > 0 {
                format!(" | prefetch issued {} used {} nvme {} skipped {}", g(&sh.pre_issued), g(&sh.pre_used), g(&sh.pre_nvme), g(&sh.pre_skipped))
            } else {
                String::new()
            };
            let pred = if g(&sh.pred_total) > 0 {
                format!(" | pred6 {}/{} pred8 {} pred12 {} non-vram pred6 {}/{} pred8 {} pred12 {}", g(&sh.pred_hits[0]), g(&sh.pred_total), g(&sh.pred_hits[1]), g(&sh.pred_hits[2]), g(&sh.pred_miss_hits[0]), g(&sh.pred_miss_total), g(&sh.pred_miss_hits[1]), g(&sh.pred_miss_hits[2]))
            } else {
                String::new()
            };
            return Some(format!("  vram {} staged {} pcie {} ram {} nvme {} promos vram {} ram {} ({:.0} ms, pool {:.0} ms of which fetch {:.0}, stage wait {:.0}) spec {}/{} non-vram {}/{} | serve {} calls: read {:.0} us, issue {:.0}, total {:.0}, gap {:.0}{pred}{pre}{}", g(&sh.vram_hits), g(&sh.staged_hits), g(&sh.mapped_hits), g(&sh.ram_hits), g(&sh.nvme_misses), g(&sh.vram_promos), g(&sh.ram_promos), g(&sh.promo_ns) as f64 / 1e6, g(&sh.pool_ns) as f64 / 1e6, g(&sh.fetch_ns) as f64 / 1e6, g(&sh.stage_ns) as f64 / 1e6, g(&sh.spec_hits), g(&sh.spec_total), g(&sh.spec_miss_hits), g(&sh.spec_miss_total), g(&sh.tl_calls), us(&sh.tl_read_ns), us(&sh.tl_issue_ns), us(&sh.tl_total_ns), us(&sh.tl_gap_ns), if g(&sh.stage_fallbacks) > 0 || g(&sh.ram_dedups) > 0 { format!("{}{}", if g(&sh.stage_fallbacks) > 0 { format!(" | stage fallbacks {}", g(&sh.stage_fallbacks)) } else { String::new() }, if g(&sh.ram_dedups) > 0 { format!(" | ram dedups {}", g(&sh.ram_dedups)) } else { String::new() }) } else { String::new() }));
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
        sched_trace(&format!("R {slot}"));
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
    /// attention ring must hold the window [p - n_swa, p) at every layer. If the ring still does (the rewind is
    /// within the last DS_TMAX positions), nothing is copied; otherwise the rings are rebuilt by replaying the
    /// slot's own tokens (`replay_to`). This replaced a per-position host copy of every latent, 21.5 GB per slot
    /// at 256k (2026-11-06).
    pub fn rewind(&mut self, slot: usize, p: usize) {
        sched_trace(&format!("W {slot} {p}"));
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

    /// Tokens a replay must cover so that every layer's ring is exact at its end: a layer's keys at a position
    /// come from the layer below's output there, which needs that layer's own window of n_swa positions, so each
    /// layer adds n_swa - 1 positions of warm-up; n_layer layers deep, n_swa + n_layer * (n_swa - 1) (5,208 here).
    pub fn replay_span(&self) -> usize {
        self.cfg.n_swa + self.cfg.n_layer * (self.cfg.n_swa - 1)
    }

    /// Rebuilds slot `slot`'s attention rings for position `p` by replaying its tokens from the group boundary at
    /// or below p - replay_span(), and leaves the slot at `p` (tokens truncated). The slot's tokens must hold
    /// [0, p). The replay runs in `replaying` mode: the compressed stores and index keys are neither written nor
    /// disturbed (they hold the exact rows of every position below `p` already; the replay's early positions are
    /// not exact, and a later replayed position reads the rows of the earlier ones), only the rings are rebuilt,
    /// and the chain of n_swa - 1 positions per layer makes the last n_swa positions exact at every layer by the
    /// end. (Before 2026-11-06 the replay covered n_swa tokens, which left every layer above the first inexact;
    /// a first fix saved and restored the rows but let the replay read its own wrong ones.)
    fn replay_to(&mut self, slot: usize, p: usize) {
        let max_ratio = self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        let mut a = p.saturating_sub(self.replay_span());
        a -= a % max_ratio;
        let replay: Vec<u32> = self.st.tokens[slot][a..p].to_vec();
        self.st.tokens[slot].truncate(a);
        self.st.pos[slot] = a;
        self.gpu.sync();
        self.replaying = true;
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
        self.replaying = false;
        debug_assert_eq!(self.st.pos[slot], p);
        self.gpu.sync();
    }

    /// Moves slot `slot`'s sequence out to host memory and leaves the slot empty (as after `reset`): its tokens,
    /// the used rows of its compressed stores (`pos / ratio` per store) and its attention rings. `unpark` puts it
    /// back into any slot. The pending pairs are not kept: a park happens at a group boundary, where no group is
    /// pending.
    pub fn park(&mut self, slot: usize) -> Parked {
        let p = self.snapshot(slot);
        self.st.tokens[slot].clear();
        self.st.pos[slot] = 0;
        p
    }

    /// Slot `slot`'s state as `park` would take it, without emptying the slot: the tokens, the used store rows
    /// and the rings.
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
        let n_ring = self.ring_w * self.cfg.head_dim;
        let rings = (0..self.cfg.n_layer).map(|l| self.st.rings[l].download_at::<f32>(slot * n_ring * 4, n_ring)).collect();
        Parked { tokens, stores, rings }
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
        assert!(upto % self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize == 0, "unpark to a group boundary");
        if p.rings.len() == self.cfg.n_layer {
            // the rings as parked: exact at `pos`, and `rewind` keeps its fast path within the last DS_TMAX
            let n_ring = self.ring_w * self.cfg.head_dim;
            for l in 0..self.cfg.n_layer {
                self.st.rings[l].upload_at(slot * n_ring * 4, &p.rings[l][..n_ring]);
            }
            self.rewind(slot, upto);
        } else {
            // a state without rings (QXPK0001): rebuild them by replay
            self.replay_to(slot, upto);
        }
    }

    /// The Engram rows of every Engram layer and token of `toks` (at `positions`), gathered on the host into the
    /// mapped regions of `dst` (region ei holds token n's rows at (ei * stride + n) * cols * dim). Sets `t_engram`.
    /// A prompt chunk's Engram rows on a background thread, engram layer by engram layer (`done` counts the layers
    /// whose rows are in `dst`): the reads are 196,608 random pages of the table for 4,096 rows, 0.4-1.0 s on a disk
    /// the expert stream keeps busy, however they are issued (one ring, six rings, twelve threads of positioned
    /// reads, direct or buffered: the same time, 2026-11-11), so they overlap the card's work instead of preceding
    /// it. Direct page reads, a few rings; the same rows and the same decode as `gather_engram`. None without a
    /// pack (QX_DS_ENGRAM_THREADS=0 too): the caller gathers in line.
    fn gather_engram_start(&self, toks: &[(usize, u32)], positions: &[u32], dst: &qx_gpu::mem::HostMapped, stride: usize) -> Option<(std::thread::JoinHandle<()>, std::sync::Arc<std::sync::atomic::AtomicUsize>)> {
        let (_, pack) = self.engram_pack.as_ref()?;
        let n_thr = std::env::var("QX_DS_ENGRAM_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);
        if n_thr == 0 {
            return None;
        }
        let t = toks.len();
        let c = &self.cfg;
        let (ed, n_layers) = (c.engram_dim, c.engram_layers.len());
        let n_e = c.engram_cols() * ed;
        let rb = pack.ple.row_bytes as usize;
        // (file offset, float offset in dst), engram layer by engram layer
        let mut jobs: Vec<(u64, usize)> = Vec::with_capacity(n_layers * t * c.engram_cols());
        for ei in 0..n_layers {
            for n in 0..t {
                for (j, r) in c.engram_rows(ei, &self.st.tokens[toks[n].0], positions[n] as usize).into_iter().enumerate() {
                    jobs.push((pack.ple_row(r), (ei * stride + n) * n_e + j * ed));
                }
            }
        }
        assert!((n_layers * stride * n_e) * 4 <= dst.bytes);
        let base = dst.slice::<f32>(0, 0).as_mut_ptr() as usize;
        let path = pack.path.clone();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let done2 = done.clone();
        let per_layer = jobs.len() / n_layers.max(1);
        let h = std::thread::Builder::new().name("qx-engram".into()).spawn(move || {
            use io_uring::{opcode, types};
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::OpenOptionsExt;
            // (a thread inherits its creator's mask, and the engine thread is pinned to one CPU: these wait on the
            // disk and decode rows, so they go wherever a CPU is free; nothing here spins)
            let n_cpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) }.max(1) as usize);
            set_affinity(0, &(0..n_cpu).collect::<Vec<_>>());
            let f = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path).expect("engram table, direct");
            let fd = f.as_raw_fd();
            let ty = qx_pack::ggml::GgmlType::from_id(13).unwrap(); // Q5_K
            for layer in jobs.chunks(per_layer.max(1)) {
                let per = layer.len().div_ceil(n_thr);
                std::thread::scope(|sc| {
                    for part in layer.chunks(per.max(1)) {
                        sc.spawn(move || {
                            const QD: usize = 256;
                            const BUF: usize = 8192;
                            // the row's page (two when it straddles)
                            let layout = std::alloc::Layout::from_size_align(QD * BUF, 4096).unwrap();
                            let pool = unsafe { std::alloc::alloc(layout) };
                            assert!(!pool.is_null());
                            let mut free: Vec<usize> = (0..QD).collect();
                            let mut ring = io_uring::IoUring::new(QD as u32).expect("engram io_uring");
                            let (mut next, mut fin) = (0usize, 0usize);
                            while fin < part.len() {
                                while next < part.len() && !free.is_empty() {
                                    let b = free.pop().unwrap();
                                    let off = part[next].0;
                                    let (a0, a1) = (off & !4095, (off + rb as u64 + 4095) & !4095);
                                    let e = opcode::Read::new(types::Fd(fd), unsafe { pool.add(b * BUF) }, (a1 - a0) as u32).offset(a0).build().user_data(((next as u64) << 16) | b as u64);
                                    unsafe { ring.submission().push(&e).expect("engram io_uring queue") };
                                    next += 1;
                                }
                                ring.submit_and_wait(1).expect("engram io_uring submit");
                                for cqe in ring.completion() {
                                    let (i, b) = ((cqe.user_data() >> 16) as usize, (cqe.user_data() & 0xffff) as usize);
                                    let (off, o) = part[i];
                                    let in_page = (off & 4095) as usize;
                                    assert!(cqe.result() >= (in_page + rb) as i32, "engram row read returned {}", cqe.result());
                                    // (each row's region is written by exactly one thread; the caller reads a layer's
                                    // rows only after `done` passed it)
                                    let out = unsafe { std::slice::from_raw_parts_mut((base + o * 4) as *mut f32, ed) };
                                    let raw = unsafe { std::slice::from_raw_parts(pool.add(b * BUF + in_page), rb) };
                                    qx_pack::dequant(ty, raw, out);
                                    free.push(b);
                                    fin += 1;
                                }
                            }
                            unsafe { std::alloc::dealloc(pool, layout) };
                        });
                    }
                });
                done2.fetch_add(1, std::sync::atomic::Ordering::Release);
            }
        }).expect("engram thread");
        Some((h, done))
    }

    fn gather_engram(&self, toks: &[(usize, u32)], positions: &[u32], dst: &qx_gpu::mem::HostMapped, stride: usize) -> f64 {
        let t = toks.len();
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
                let dst = dst.slice::<f32>((ei * stride + n) * n_e * 4, n_e);
                for j in 0..rows.len() {
                    qx_pack::dequant(ty, &raw[i * rb..(i + 1) * rb], &mut dst[j * ed..(j + 1) * ed]);
                    i += 1;
                }
            }
        } else {
            for (ri, rows) in all_rows.iter().enumerate() {
                let (ei, n) = (ri / t, ri % t);
                let dst = dst.slice::<f32>((ei * stride + n) * n_e * 4, n_e);
                for (j, &r) in rows.iter().enumerate() {
                    self.engram_row(r, &mut dst[j * ed..(j + 1) * ed]);
                }
            }
        }
        te.elapsed().as_secs_f64()
    }

    /// The dry pass of width `t` (see `forward_segments`), once per width: the window's whole kernel set with the
    /// expert job replaced by zeroed slot lists, then the device drained. The token arrays must hold the window
    /// the real pass runs next (the dry pass writes the same ring and store positions, which the real pass rewrites).
    fn warm_width(&mut self, t: usize, pos_min: usize, pos_max: usize) {
        if self.driver.is_some() && !self.warmed[t] {
            // A dry window can cross a compressor-group boundary. Its later tokens overwrite pending
            // entries that the real window's first token still needs (e.g. positions 1,2 at ratio 2).
            // Preserve the incoming pairs; rings and committed store rows are rewritten by the real pass.
            self.gpu.sync();
            let pending: Vec<Option<Vec<u8>>> = self.st.stores.iter()
                .map(|s| s.as_ref().map(|(_, _, p, _)| p.download(p.bytes))).collect();
            let max_ratio = self.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
            if let Some(d) = &self.driver {
                d.shared.mb.slice::<f32>(crate::mailbox::mb_y(E), crate::mailbox::AMAX * E).fill(0.0);
            }
            self.warm = true;
            let p = max_ratio - 1;
            self.launch_window(t, pos_min.max(p), pos_max.max(p), None, false);
            self.warm = false;
            self.gpu.sync();
            for (store, saved) in self.st.stores.iter().zip(&pending) {
                if let (Some((_, _, p, _)), Some(bytes)) = (store, saved) {
                    p.upload(bytes);
                }
            }
            self.warmed[t] = true;
        }
    }

    /// Points the per-row scratch of `s` at rows [r0, r0 + t) of the chunk buffers (the layer parts read and write
    /// `s.*` by pointer). `pre_prev` and `pre_f` alias one region: the attention part reads it, the FFN part then
    /// writes the next layer's value. Not `shy`: the shared expert's GEMV group was bound to the scratch buffer at
    /// load, so the slab pass copies it out instead.
    fn chunk_views(&mut self, r0: usize, t: usize) {
        let cb = self.chunk.as_ref().expect("chunk buffers");
        let (hc, hcd, k) = (self.cfg.hc, self.cfg.hc * E, self.cfg.n_expert_used);
        let v = |b: &DeviceBuf, per_row: usize| DeviceBuf::view(b.ptr + (r0 * per_row * 4) as u64, t * per_row * 4);
        let views = [
            (v(&cb.r, hcd), 0usize), (v(&cb.pre, hc), 1), (v(&cb.pre, hc), 2), (v(&cb.post, hc), 3), (v(&cb.comb, hc * hc), 4),
            (v(&cb.x, E), 5), (v(&cb.ids, k), 7), (v(&cb.wts, k), 8), (v(&cb.tok, 1), 9), (v(&cb.pos, 1), 10), (v(&cb.slot, 1), 11),
        ];
        let topk = self.cfg.idx_top_k;
        let cand = (v(&cb.cand_sel, self.cfg.cand_blocks.max(1)), v(&cb.cand_count, 1));
        let sels: Vec<Option<(DeviceBuf, DeviceBuf)>> = cb.sels.iter().map(|o| o.as_ref().map(|(sb, cnt)| (v(sb, topk), v(cnt, 1)))).collect();
        for (view, which) in views {
            let slot: &mut DeviceBuf = match which {
                0 => &mut self.s.r, 1 => &mut self.s.pre_prev, 2 => &mut self.s.pre_f, 3 => &mut self.s.post, 4 => &mut self.s.comb,
                5 => &mut self.s.x, 7 => &mut self.s.ids, 8 => &mut self.s.wts, 9 => &mut self.s.tok, 10 => &mut self.s.pos, _ => &mut self.s.slot,
            };
            *slot = view;
        }
        self.s.cand_sel = cand.0;
        self.s.cand_count = cand.1;
        for (l, o) in sels.into_iter().enumerate() {
            if let Some(pair) = o {
                self.st.sels[l] = Some(pair);
            }
        }
        self.erows_row0 = r0;
    }

    /// Compiles and loads every kernel the chunk path launches, once, while no expert job is in flight: a module
    /// load while the GPU spins in a wait kernel and the driver's copies are in flight deadlocks (the window path's
    /// dry pass exists for the same reason; the chunk path's first use of a format's grouped kernel at layer 10 hung
    /// a run on 2026-11-06). Every launch sees "slot 0" and returns; the wait sees its generation at once.
    /// Asks the driver for the arena's flex tail (the chunk buffers live there) and waits until its records are out
    /// of the VRAM table on host and device. A no-op without the tiers or when already held.
    /// Every writer of the chunk buffers must hold the arena's flex tail (else it writes live expert records).
    fn assert_flex_held(&self, who: &str) {
        if let Some(d) = self.driver.as_ref() {
            if d.shared.flex_bytes > 0 {
                assert!(d.shared.flex_want.load(std::sync::atomic::Ordering::Acquire), "{who}: the chunk buffers written without the flex tail held");
            }
        }
    }

    fn flex_enter(&self) {
        let Some(d) = self.driver.as_ref() else { return };
        let sh = &d.shared;
        if sh.flex_want.load(std::sync::atomic::Ordering::Acquire) && sh.flex_ack.load(std::sync::atomic::Ordering::Acquire) == sh.flex_req.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        sh.flex_want.store(true, std::sync::atomic::Ordering::Release);
        let req = sh.flex_req.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
        let t0 = std::time::Instant::now();
        while sh.flex_ack.load(std::sync::atomic::Ordering::Acquire) != req {
            assert!(t0.elapsed().as_secs() < 120, "the driver did not give the flex tail to the chunk path");
            std::thread::yield_now();
        }
    }

    /// Gives the arena's flex tail back to the tiers (the chunk buffers are dead): call once a prompt's chunks are
    /// done and no other prompt is about to run its chunks. The driver applies it at its next turn.
    pub fn chunk_release(&self) {
        sched_trace("L");
        if let Some(d) = self.driver.as_ref() {
            let sh = &d.shared;
            if sh.flex_want.swap(false, std::sync::atomic::Ordering::AcqRel) {
                sh.flex_req.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
        }
    }

    pub fn chunk_warm(&mut self) {
        if self.chunk.as_ref().is_none_or(|cb| cb.warmed) {
            return;
        }
        // the dry pass writes the chunk buffers, which are the arena tail's records until the tail is taken: take it
        // (a no-op inside a chunk pass, which holds it already). The server's warm-up ran this without it on
        // 2026-11-08 and corrupted the coldest records at start-up; the caller releases the tail after.
        self.flex_enter();
        self.assert_flex_held("chunk_warm");
        let d = self.driver.as_ref().unwrap().shared.clone();
        let cb = self.chunk.as_ref().unwrap();
        let (k, ff) = (self.cfg.n_expert_used, self.cfg.ff_exp);
        let word = cb.warm_word.ptr;
        cb.warm_desc.upload(&[word, cb.sel.ptr, 0u64, 0u64]);
        let mut seen: Vec<(GgmlType, GgmlType)> = Vec::new();
        for l in 0..self.cfg.n_layer {
            let ex = &self.layers[l].experts;
            if seen.contains(&(ex.gate_ty, ex.down_ty)) {
                continue;
            }
            seen.push((ex.gate_ty, ex.down_ty));
            let (gsuf, dsuf) = (ops::suffix_of(ex.gate_ty), ops::suffix_of(ex.down_ty));
            for (name, info) in [
                (format!("qx_moe_gateup_grp_{gsuf}"), vec![word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 1, cb.sel.ptr]),
                (format!("qx_moe_down_grp_{dsuf}"), vec![word, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, 0, ex.down_row_bytes as u64, cb.sel.ptr, 1]),
                (format!("qx_moe_gateup_mma_{gsuf}"), vec![word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 1, cb.sel.ptr]),
                (format!("qx_moe_down_mma_{dsuf}"), vec![word, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, 0, ex.down_row_bytes as u64, cb.sel.ptr, 1]),
                (format!("qx_moe_gateup_grpb_{gsuf}"), vec![0, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 0, cb.warm_desc.ptr, 1]),
                (format!("qx_moe_down_grpb_{dsuf}"), vec![0, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, 0, ex.down_row_bytes as u64, cb.warm_desc.ptr, 1]),
                (format!("qx_moe_gateup_q8k_{gsuf}_w20_sel"), vec![word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 0, cb.sel.ptr]),
                (format!("qx_moe_down_q8k_{dsuf}_sel"), vec![word, cb.gu.ptr, 0, 0, cb.out.ptr, ff as u64, E as u64, 0, ex.down_row_bytes as u64, cb.sel.ptr]),
            ] {
                let threads = if name.contains("_mma_") { 128 } else { 256 };
                self.gpu.run(MOE.kernel(static_name(name), CubeDim::new_1d(threads)), (1, 1, 1), &info);
            }
        }
        self.gpu.run(MOE.kernel("qx_moe_wait_word", CubeDim::new_1d(1)), (1, 1, 1), &[word, 1]);
        self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (1, 1, 1), &[cb.gu.ptr, 0, cb.hq.ptr, cb.hd.ptr, ff as u64, cb.sel.ptr, 1]);
        self.gpu.run(MOE.kernel("qx_moe_q8k_rows", CubeDim::new_1d(256)), (1, 1, 1), &[cb.x.ptr, cb.xq.ptr, cb.xd.ptr, E as u64]);
        let done_now = d.chunk_done.atomic_u64(0).load(std::sync::atomic::Ordering::Acquire);
        self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, done_now]);
        self.gpu.sync();
        self.chunk.as_mut().unwrap().warmed = true;
        if std::env::var("QX_DS_VERBOSE").is_ok() {
            eprintln!("chunk kernels warmed: {} expert format pairs", seen.len());
        }
    }

    /// A prompt chunk of `toks` (1 to the chunk buffers' rows) for slot `slot`, layer-major: every layer runs its
    /// attention and FFN-pre parts over the chunk in slabs of DS_TMAX rows, then the layer's routed experts over
    /// every row of the chunk at once, each distinct expert streamed into VRAM once (the driver's chunk job: VRAM
    /// residents first, then RAM records through the staging ring, then NVMe records), then the FFN-post parts.
    /// The slab parts are the window kernels on views of the chunk buffers, so every row computes what windows
    /// compute, bit for bit; the expert kernels are the window kernels indexed by selection lists. Launches are
    /// eager (no graph). Returns the last row's logits. Needs the asynchronous driver and `QX_DS_CHUNK`.
    pub fn forward_chunk(&mut self, slot: usize, toks: &[u32]) -> Vec<f32> {
        self.forward_chunk_opt(slot, toks, false)
    }

    /// `forward_chunk`, KV-only when `kv_only` (the layer skip of `forward_segments_opt`, for a chunk at least
    /// `prompt_keep` positions before its prompt's end): layers up to the last KV source, whose slab passes stop
    /// after the ring, store and index-key writes, and no experts there; returns no logits.
    pub fn forward_chunk_opt(&mut self, slot: usize, toks: &[u32], kv_only: bool) -> Vec<f32> {
        self.forward_chunk_keep(slot, toks, if kv_only { toks.len() } else { 0 })
    }

    /// `forward_chunk` with the full stack from row `keep_from` on only: the rows before it stop at the last KV
    /// source (as a KV-only chunk's do), the rows from it on run every layer, and the last row's logits come back
    /// (none when `keep_from` is the chunk's length: a KV-only chunk). One chunk in place of a KV-only chunk and a
    /// full one: the layers up to the last KV source stream their experts once for both parts. The same bits as the
    /// two chunks give.
    pub fn forward_chunk_keep(&mut self, slot: usize, toks: &[u32], keep_from: usize) -> Vec<f32> {
        let kv_only = keep_from >= toks.len();
        let out = self.forward_chunk_inner(slot, toks, keep_from.min(toks.len()));
        if sched_trace_on() {
            let t: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
            // (`K slot keep_from tokens`: a chunk with the full stack from row keep_from on; `C` for the two pure forms)
            if kv_only || keep_from == 0 {
                sched_trace(&format!("C {slot} {} {}", kv_only as u8, t.join(",")));
            } else {
                sched_trace(&format!("K {slot} {keep_from} {}", t.join(",")));
            }
            sched_trace(&format!("H {}", row_hashes(&out, 1)));
        }
        out
    }

    fn forward_chunk_inner(&mut self, slot: usize, toks: &[u32], keep_from: usize) -> Vec<f32> {
        let tf = std::time::Instant::now();
        self.t_experts = 0.0;
        let rows = toks.len();
        let kv_only = keep_from >= rows;
        let rows_max = self.chunk.as_ref().map(|c| c.rows).expect("QX_DS_CHUNK=N sets the chunk buffers at load");
        assert!((1..=rows_max).contains(&rows), "chunk of {rows} rows (buffers for {rows_max})");
        assert!(self.driver.is_some() && self.abufs.is_some(), "the chunk path needs the asynchronous driver");
        assert!(slot < self.slots);
        let (k, ne, ff, n_layer, n_vocab) = (self.cfg.n_expert_used, self.cfg.n_expert, self.cfg.ff_exp, self.cfg.n_layer, self.cfg.n_vocab);
        let clamp_exp: Vec<f32> = self.cfg.clamp_exp.clone();
        let pos0 = self.st.pos[slot];
        let positions: Vec<u32> = (0..rows).map(|i| (pos0 + i) as u32).collect();
        let slots_v: Vec<u32> = vec![slot as u32; rows];
        let segs: Vec<(usize, u32)> = toks.iter().map(|&t| (slot, t)).collect();
        for &t in toks {
            assert!((t as usize) < n_vocab, "token id {t} outside the vocabulary");
            self.st.tokens[slot].push(t);
        }
        // the Engram rows of every row: on a thread from here on, while the dry passes and the first layers run
        // (each engram layer waits for its rows)
        let mut engram_job = {
            let cb = self.chunk.as_ref().unwrap();
            let job = if rows >= 256 { self.gather_engram_start(&segs, &positions, &cb.erows_host, rows_max) } else { None };
            self.t_engram = if job.is_none() { self.gather_engram(&segs, &positions, &cb.erows_host, rows_max) } else { 0.0 };
            job
        };
        // the slabs: full ones of DS_TMAX, then the remainder in the widths every server warms (never 5 or 7:
        // their kernels are not loaded behind a sealed kernel set, and width 5's multi-GEMV is not bit-identical)
        // (a slab ends at `keep_from`: the rows before it and the rows from it on part ways at the last KV source)
        let mut slabs: Vec<(usize, usize)> = Vec::new();
        let mut r0 = 0;
        while r0 < rows {
            let left = if r0 < keep_from { keep_from - r0 } else { rows - r0 };
            let t = [128usize, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1].into_iter().find(|&w| w <= left.min(DS_TMAX)).unwrap();
            slabs.push((r0, t));
            r0 += t;
        }
        // the dry passes of the slab widths, on the ordinary scratch with the first rows' arrays (the previous
        // forward synced, so nothing is in flight)
        // (the widest slab's worth of arrays: a dry pass of width w writes the ring and store cells of the w positions
        // its arrays name, and the real pass rewrites exactly those only if they are the chunk's first w. With the
        // first slab narrower than a later one (a mixed chunk whose KV-only part is a few rows) the wider dry
        // passes ran on stale positions and overwrote other positions' stored keys: 0 of 64 rows, 2026-11-11)
        let t0 = slabs.iter().map(|&(_, t)| t).max().unwrap();
        self.s.tok.upload(&toks[..t0]);
        self.s.pos.upload(&positions[..t0]);
        self.s.slot.upload(&slots_v[..t0]);
        let mut widths: Vec<usize> = slabs.iter().map(|&(_, t)| t).collect();
        widths.sort_unstable();
        widths.dedup();
        for &w in &widths {
            self.warm_width(w, pos0, pos0 + w - 1);
        }
        // QX_RAW_LAUNCH=1: this pass's launches through the driver API on CubeCL's stream (qx_gpu::raw)
        let _raw = qx_gpu::raw::Scope::enter();
        self.flex_enter();
        self.assert_flex_held("forward_chunk_opt");
        self.chunk_warm();
        // the chunk arrays and the Engram rows of every row
        {
            let cb = self.chunk.as_ref().unwrap();
            cb.tok.upload(toks);
            cb.pos.upload(&positions);
            cb.slot.upload(&slots_v);
        }

        let saved = [
            std::mem::replace(&mut self.s.r, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.pre_prev, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.pre_f, DeviceBuf::view(0, 0)),
            std::mem::replace(&mut self.s.post, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.comb, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.x, DeviceBuf::view(0, 0)),
            std::mem::replace(&mut self.s.ids, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.wts, DeviceBuf::view(0, 0)),
            std::mem::replace(&mut self.s.tok, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.pos, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.slot, DeviceBuf::view(0, 0)),
        ];
        let saved_cand = (std::mem::replace(&mut self.s.cand_sel, DeviceBuf::view(0, 0)), std::mem::replace(&mut self.s.cand_count, DeviceBuf::view(0, 0)));
        let saved_sels: Vec<Option<(DeviceBuf, DeviceBuf)>> = self.st.sels.iter_mut().map(|o| o.take()).collect();
        // the chunk's Engram rows in place of the window's region while the slab parts run
        std::mem::swap(&mut self.s.erows_host, &mut self.chunk.as_mut().unwrap().erows_host);
        self.erows_stride = rows_max;
        let n_slabs = slabs.len();
        let slab = |si: usize| -> (usize, usize) { slabs[si] };
        let d = self.driver.as_ref().unwrap().shared.clone();
        let mut t_moe = 0f64;
        let (mut t_enq, mut t_sync, mut t_post) = (0f64, 0f64, 0f64);
        let mut n_distinct = 0usize;
        let verbose = std::env::var("QX_DS_VERBOSE").is_ok();
        let trace = std::env::var("QX_DS_CHUNK_TRACE").is_ok();
        let kv_last = self.kv_last();
        let n_layers_run = if kv_only { kv_last + 1 } else { n_layer };
        // QX_DS_CHUNK_PREFETCH=0 disables the prefetch job (the experts stream only after the routing is known)
        let prefetch_on = !std::env::var("QX_DS_CHUNK_PREFETCH").is_ok_and(|v| v == "0");
        let n_ring = self.abufs.as_ref().map(|ab| ab.stage_cap as usize).unwrap_or(0);
        if chunk_streams() > 1 && self.chunk_side.len() < chunk_streams() {
            let ff = self.cfg.ff_exp;
            while self.chunk_side.len() < chunk_streams() {
                // (an expert's scratch is indexed by the row's place in its list: `chunk_side_rows` rows per stream;
                // an expert with more rows runs on the main stream with the chunk's own scratch)
                let cap = chunk_side_rows(rows_max);
                let bufs = [self.cscratch(cap * 2 * ff * 4), self.cscratch(cap * ff), self.cscratch(cap * (ff / 256) * 4)];
                self.chunk_side.push(bufs);
            }
        }
        // QX_DS_CHUNK_GEMM=1: the FFN's dense products once over the chunk's rows (cuBLAS f16 GEMMs), not per slab
        // (our own modules are loaded first: cuBLAS's handle and kernel libraries come out of the same headroom, and
        // a module load after them failed in a three-slot 256k server, 2026-11-10)
        // The handle and its kernel libraries (about 1.4 GiB in all) come out of the headroom at the first chunk:
        // a one-slot engine has it (0.7 GiB left after the dry pass); a three-slot 256k server did not (0.45 GiB
        // free after its warm-up until 2026-11-12; 1.21 GiB since the kernel modules are loaded once per source, not
        // re-measured with cuBLAS), and its prompts keep the exact kernels (`cuBLAS: no handle`), as they do when
        // the launches do not go through the raw launcher (cuBLAS queues on the stream directly).
        let mut chunk_ffn_wide = chunk_gemm() && qx_gpu::raw::RAW.load(std::sync::atomic::Ordering::Relaxed) && self.layers.iter().all(|w| w.g_ffn.blas_ok() && w.g_shd.blas_ok()) && {
            for m in [&MOE, &DSV41, &LAYER, &qx_gpu::kernel::modules::GEMV, &qx_gpu::kernel::modules::MGEMV, &qx_gpu::kernel::modules::GEMM] {
                qx_gpu::raw::preload(m);
            }
            !ops::gemm16_cublas() || qx_gpu::blas::try_init()
        };
        if verbose_types() && pos0 == 0 {
            let w = &self.layers[0];
            let t = |g: &MGemv| g.segs.iter().map(|s| format!("{}:{}x{}/{}", s.qx, s.rows, s.cols, s.stride)).collect::<Vec<_>>().join(" ");
            eprintln!("dense groups (type:rows x cols/stride) g_x [{}] g_qr [{}] wo_a [{}] g_ob [{}] g_ffn [{}] g_shd [{}]; ffn wide {}", t(&w.g_x), t(&w.g_qr), t(&w.wo_a[0]), t(&w.g_ob), t(&w.g_ffn), t(&w.g_shd), chunk_ffn_wide);
            let mut kinds = std::collections::BTreeMap::new();
            for w in &self.layers { for g in [&w.g_x, &w.g_qr, &w.g_ob, &w.g_ffn, &w.g_shd] { for s in &g.segs { *kinds.entry(s.qx).or_insert(0usize) += 1; } } }
            eprintln!("dense segment types over all layers: {kinds:?}");
            for w in &self.layers { if let Some((g, _)) = &w.engram { eprintln!("engram group [{}]", t(g)); break; } }
        }
        if chunk_ffn_wide && self.blas_bufs.is_none() {
            let groups = |w: &Layer| -> usize { [&w.g_x, &w.g_qr, &w.g_ob].iter().map(|g| g.w16_total()).sum::<usize>() + w.wo_a.iter().map(|g| g.w16_total()).sum::<usize>() };
            let halves = self.layers.iter().map(groups).max().unwrap_or(0);
            let kmax = self.layers.iter().flat_map(|w| [w.g_x.tcols, w.g_qr.tcols, w.g_ob.tcols].into_iter().chain(w.wo_a.iter().map(|g| g.tcols))).max().unwrap_or(E);
            // (a slab's f16 inputs: the widest are the attention rows the grouped output projection reads whole; the
            // K-sliced products' partials: `ops::gemm16_planes`)
            let qd = self.cfg.n_head * self.cfg.head_dim;
            let bufs = (self.cscratch(halves * 2 + 256), self.cscratch(DS_TMAX * kmax.max(qd) * 2 + 256), self.cscratch(DS_TMAX * qd * 4));
            self.blas_bufs = Some(bufs);
        }
        let mut dry_ok = true;
        if chunk_ffn_wide && ops::gemm16_cublas() {
            // cuBLAS loads a kernel library the first time a GEMM shape needs it, and that load must never happen
            // while an expert job is posted (a kernel waiting for a staged record keeps the card busy, the driver
            // thread sits behind the load's lock: the run stopped in layer 2 on 2026-11-10). So every shape this
            // chunk will use runs once here, on scratch, before any job exists: the standing dry-pass rule.
            self.chunk_ffn_alloc(rows_max);
            let (w16, x16s) = { let b = self.blas_bufs.as_ref().unwrap(); (b.0.ptr, b.1.ptr) };
            let [_, gate_scratch, _, _, x16c, w16c] = { let (_, b) = self.chunk_ffn_bufs.as_ref().unwrap(); [b[0].ptr, b[1].ptr, b[2].ptr, b[3].ptr, b[4].ptr, b[5].ptr] };
            let mut seen = std::collections::HashSet::new();
            let mut widths: Vec<usize> = slabs.iter().map(|&(_, t)| t).collect();
            widths.sort_unstable();
            widths.dedup();
            let ymax = DS_TMAX * self.layers.iter().flat_map(|w| [&w.g_x, &w.g_qr, &w.g_ob].into_iter().chain(w.wo_a.iter())).flat_map(|g| g.segs.iter().map(|s| s.rows)).max().unwrap_or(1);
            if self.blas_y.as_ref().is_none_or(|b| b.bytes < ymax * 4) {
                let b = self.cscratch(ymax * 4);
                self.blas_y = Some(b);
            }
            let ys = self.blas_y.as_ref().unwrap().ptr;
            for w in &self.layers {
                for g in [&w.g_x, &w.g_qr, &w.g_ob].into_iter().chain(w.wo_a.iter()) {
                    for sg in g.segs.iter().filter(|sg| sg.qx != 0) {
                        for &t in &widths {
                            if dry_ok && seen.insert((sg.rows, sg.cols, t)) {
                                dry_ok = qx_gpu::blas::try_gemm_hss(w16, sg.cols, sg.rows, x16s, t, ys, sg.rows).is_ok();
                            }
                        }
                    }
                }
                for g in [&w.g_ffn, &w.g_shd] {
                    for sg in g.segs.iter().filter(|sg| sg.qx != 0) {
                        for rows in [rows, rows - keep_from] {
                            if rows > 0 && seen.insert((sg.rows, sg.cols, rows)) {
                                // (the output lands in the gate scratch or, for the down product, would not fit it: E > ff)
                                let y = if sg.rows <= self.cfg.ff_exp { gate_scratch } else { self.chunk.as_ref().unwrap().shy.ptr };
                                dry_ok = dry_ok && qx_gpu::blas::try_gemm_hss(w16c, sg.cols, sg.rows, x16c, rows, y, sg.rows).is_ok();
                            }
                        }
                    }
                }
            }
            self.gpu.sync();
            // cuBLAS must leave the engine its headroom (graphs of new window widths, modules): with less than
            // QX_DS_BLAS_MIN_FREE_MIB (384) free after the dry pass, or a failed load, it is given up for the
            // process and prompt chunks keep the exact kernels
            let min_free = std::env::var("QX_DS_BLAS_MIN_FREE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(384) << 20;
            let free = qx_gpu::mem::mem_info().0;
            if !dry_ok || free < min_free {
                eprintln!("cuBLAS: given up ({} MiB free after its dry pass, {} needed{}): prompt chunks keep the exact kernels", free >> 20, min_free >> 20, if dry_ok { "" } else { "; a kernel library did not load" });
                qx_gpu::blas::disable();
                chunk_ffn_wide = false;
            } else if verbose && pos0 == 0 {
                eprintln!("cuBLAS: prompt chunks in f16 ({} MiB free after the dry pass)", free >> 20);
            }
        }
        for l in 0..n_layers_run {
            if chunk_ffn_wide {
                // this layer's attention-side dense groups as f16, once for the chunk (cuBLAS GEMMs per slab from them)
                let base = self.blas_bufs.as_ref().unwrap().0.ptr;
                let w = &self.layers[l];
                let mut map = self.blas_map.borrow_mut();
                map.clear();
                let mut off = 0u64;
                for g in [&w.g_x, &w.g_qr, &w.g_ob].into_iter().chain(w.wo_a.iter()) {
                    if g.blas_ok() && g.w16_total() > 0 {
                        g.prep_w16(&self.gpu, base + off);
                        map.insert(g as *const MGemv as usize, base + off);
                        off += (g.w16_total() * 2) as u64;
                    }
                }
            }
            if let (Some((h, done)), Some(ei)) = (&engram_job, self.cfg.engram_layers.iter().position(|&x| x == l)) {
                let tw = std::time::Instant::now();
                while done.load(std::sync::atomic::Ordering::Acquire) <= ei {
                    // (a failed read panics the thread: the engine stops with it instead of waiting for rows that never come)
                    assert!(!h.is_finished() || done.load(std::sync::atomic::Ordering::Acquire) > ei, "the Engram thread stopped before layer {l}'s rows were read");
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                self.t_engram += tw.elapsed().as_secs_f64();
            }
            let te = std::time::Instant::now();
            let mut went_on = true;
            // the layer's hottest non-resident experts stream into the ring while the slab passes run (the layer
            // above the last KV source of a KV-only chunk has no experts to prefetch)
            let generation = self.chunk_gen % 65535 + 1;
            self.chunk_gen = generation;
            let ord0_pre = self.chunk_ord;
            // the rows this layer's experts and the layers above run on: all of them up to the last KV source
            let ra = if l >= kv_last { keep_from } else { 0 };
            let n_act = rows - ra;
            let prefetch = if prefetch_on && n_ring > 0 && !(kv_only && l == kv_last) { n_ring } else { 0 };
            if prefetch > 0 {
                let cb = self.chunk.as_ref().unwrap();
                *d.chunk_pick.lock().unwrap() = None;
                d.chunk_jobs.lock().unwrap().push_back(crate::dsv41::driver::ChunkJob { layer: l, experts: Vec::new(), prefetch, counts: Vec::new(), extra_scores: Vec::new(), slots_dev: cb.slots_tbl[l % 2].ptr, generation, ord0: ord0_pre });
                self.chunk_ord += prefetch as u64;
            }
            for si in 0..n_slabs {
                let (r0, t) = slab(si);
                if l > kv_last && r0 < keep_from {
                    continue;
                }
                self.chunk_views(r0, t);
                if l == 0 {
                    self.layer_init(t, &mut None);
                }
                if predict_mode() == 1 {
                    self.predict_pre(l, t, false);
                }
                went_on = self.layer_attn(l, t, pos0 + r0, pos0 + r0 + t - 1, &mut None, r0 < keep_from);
                if !went_on {
                    continue;
                }
                if chunk_ffn_wide {
                    // the norm only: the router, the shared expert and the selections run once over the chunk below
                    let w = &self.layers[l];
                    self.hc_terms(&w.hc_ffn, &self.s.pre_f, t);
                    self.hc_pre(&self.s.pre_a, t);
                    self.rms(self.s.xpre.ptr, w.ffn_norm.ptr, self.s.x.ptr, E, t);
                    continue;
                }
                self.layer_ffn_pre(l, t, &mut None);
                if trace && l < 8 {
                    self.gpu.sync();
                    let x: Vec<f32> = self.s.x.download(E);
                    let r: Vec<f32> = self.s.r.download(E);
                    let st = |v: &[f32]| (v.iter().filter(|a| a.is_nan()).count(), v.iter().map(|a| a.abs()).fold(0f32, f32::max), v.iter().sum::<f32>());
                    eprintln!("    trace l{l} slab{si} after ffn_pre: x nan/max/sum {:?}  r {:?}", st(&x), st(&r));
                }
                self.layer_shexp(l, t);
                // the shared expert's rows out of the bound scratch into the chunk buffer
                let shy_c = self.chunk.as_ref().unwrap().shy.ptr + (r0 * E * 4) as u64;
                self.k("qx_ds_copy_f32", 256, (((t * E) as u32).div_ceil(256), 1, 1), &[self.s.shy.ptr, shy_c, (t * E) as u64]);
            }
            let _ = went_on;
            if kv_only && l == kv_last {
                // KV-only: the last KV source's ring, store and keys are written; nothing above runs
                break;
            }
            if chunk_ffn_wide {
                self.chunk_ffn(l, ra, n_act, rows_max);
            }
            // the experts of the whole chunk
            let cb = self.chunk.as_ref().unwrap();
            // (the experts' rows are numbered from `ra`: xq, xd, the selection lists and the outputs)
            self.gpu.run(MOE.kernel("qx_moe_q8k_rows", CubeDim::new_1d(256)), (n_act as u32, 1, 1), &[cb.x.ptr + (ra * E * 4) as u64, cb.xq.ptr, cb.xd.ptr, E as u64]);
            t_enq += te.elapsed().as_secs_f64();
            let ts = std::time::Instant::now();
            self.gpu.sync();
            t_sync += ts.elapsed().as_secs_f64();
            let tm = std::time::Instant::now();
            let ids: Vec<i32> = cb.ids.download_at::<i32>(ra * k * 4, n_act * k);
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
            for (a, &e) in ids.iter().enumerate() {
                lists[e as usize].push(a as u32);
            }
            // the prefetched experts (the driver's pick, in its copy order), then the used ones it did not cover
            let picked: Vec<u32> = if prefetch > 0 {
                let t0 = std::time::Instant::now();
                loop {
                    if let Some(p) = d.chunk_pick.lock().unwrap().take() {
                        break p;
                    }
                    assert!(t0.elapsed().as_secs() < 60, "the driver did not pick the prefetch");
                    std::thread::yield_now();
                }
            } else {
                Vec::new()
            };
            let mut in_pick = vec![false; ne];
            for &e in &picked {
                in_pick[e as usize] = true;
            }
            let mut rest: Vec<usize> = (0..ne).filter(|&e| !lists[e].is_empty() && !in_pick[e]).collect();
            rest.sort_by_key(|&e| (d.tier_of[l * ne + e].load(std::sync::atomic::Ordering::Relaxed), e));
            // the launch order: the pick (used or not: an unused one only marks its ordinal, freeing its slot), then the rest
            let launch: Vec<(usize, u64)> = picked.iter().enumerate().map(|(i, &e)| (e as usize, ord0_pre + i as u64 + 1)).chain(rest.iter().enumerate().map(|(j, &e)| (e, self.chunk_ord + j as u64 + 1))).collect();
            let ord0_rest = self.chunk_ord;
            self.chunk_ord += rest.len() as u64;
            let mut sel: Vec<u32> = Vec::with_capacity(rows * k);
            let mut offs: Vec<usize> = vec![0; ne];
            for &(e, _) in &launch {
                offs[e] = sel.len();
                sel.extend_from_slice(&lists[e]);
            }
            cb.sel.upload(&sel);
            let extra_scores: Vec<(u32, u32)> = picked.iter().filter(|&&e| !lists[e as usize].is_empty()).map(|&e| (e, lists[e as usize].len() as u32)).collect();
            let n_pick_used = extra_scores.len();
            d.chunk_jobs.lock().unwrap().push_back(crate::dsv41::driver::ChunkJob { layer: l, experts: rest.iter().map(|&e| e as u32).collect(), prefetch: 0, counts: rest.iter().map(|&e| lists[e].len() as u32).collect(), extra_scores, slots_dev: cb.slots_tbl[l % 2].ptr, generation, ord0: ord0_rest });
            let ex = &self.layers[l].experts;
            let gb = ff * ex.gate_row_bytes;
            // the expert formats differ per layer (the UD quant mixes them): the kernel variants per layer
            let (gsuf, dsuf) = (ops::suffix_of(ex.gate_ty), ops::suffix_of(ex.down_ty));
            // QX_DS_CHUNK_GROUPED=0: the per-selection kernels (each selection re-reads its expert) instead of the
            // grouped ones (a weight block decoded once per warp for up to 8 rows; bit-identical)
            let grouped = !std::env::var("QX_DS_CHUNK_GROUPED").is_ok_and(|v| v == "0");
            // QX_DS_CHUNK_MMA=0: the grouped CUDA-core kernels instead of the int8 tensor-core ones (both exact)
            static MMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let mma = grouped && *MMA.get_or_init(|| !std::env::var("QX_DS_CHUNK_MMA").is_ok_and(|v| v == "0"));
            let kg = static_name(if mma { format!("qx_moe_gateup_mma_{gsuf}") } else if grouped { format!("qx_moe_gateup_grp_{gsuf}") } else { format!("qx_moe_gateup_q8k_{gsuf}_w20_sel") });
            let kd = static_name(if mma { format!("qx_moe_down_mma_{dsuf}") } else if grouped { format!("qx_moe_down_grp_{dsuf}") } else { format!("qx_moe_down_q8k_{dsuf}_sel") });
            let mut n_used = 0usize;
            // QX_DS_CHUNK_BATCH=B (default 0 = one launch set per expert): the batched grouped kernels take B
            // experts per launch, each block waiting on its own expert's word. Measured slower at 16 (2026-11-06:
            // 8.4 against 7.0 ms per token at 2048 rows), kept for the record.
            static BATCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let batch = *BATCH.get_or_init(|| std::env::var("QX_DS_CHUNK_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
            // (the batched path's h quantizer runs over a batch's contiguous list range, which the per-pass scratch
            // layout does not hold: the path is off unless the per-selection buffers were allocated)
            let batch = if std::env::var("QX_DS_CHUNK_GROUPED").is_ok_and(|v| v == "0") { batch } else { 0 };
            if grouped && batch > 0 {
                let kgb = static_name(format!("qx_moe_gateup_grpb_{gsuf}"));
                let kdb = static_name(format!("qx_moe_down_grpb_{dsuf}"));
                let used: Vec<(usize, u64)> = launch.iter().copied().filter(|&(e, _)| !lists[e].is_empty()).collect();
                n_used = used.len();
                // one descriptor per (expert, group of 8 rows): every block of a batched launch has work
                let mut desc: Vec<u64> = Vec::new();
                let mut first_entry: Vec<usize> = Vec::with_capacity(used.len() + 1);
                for &(e, _) in &used {
                    first_entry.push(desc.len() / 4);
                    let m = lists[e].len();
                    for g in (0..m).step_by(8) {
                        desc.extend_from_slice(&[cb.slots_tbl[l % 2].ptr + 8 * e as u64, cb.sel.ptr + ((offs[e] + g) * 4) as u64, (m - g).min(8) as u64, g as u64]);
                    }
                }
                first_entry.push(desc.len() / 4);
                cb.bdesc.upload(&desc);
                for (bi, group) in used.chunks(batch).enumerate() {
                    let (i0, i1) = (bi * batch, bi * batch + group.len());
                    let entries = (first_entry[i1] - first_entry[i0]) as u32;
                    let (e0, el) = (group[0].0, group[group.len() - 1].0);
                    let total_sel = (offs[el] + lists[el].len() - offs[e0]) as u32;
                    let dptr = cb.bdesc.ptr + (first_entry[i0] * 32) as u64;
                    let sel0 = cb.sel.ptr + (offs[e0] * 4) as u64;
                    self.gpu.run(MOE.kernel(kgb, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), entries, 1), &[0, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 0, dptr, generation]);
                    self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (total_sel, 1, 1), &[cb.gu.ptr, bits(clamp_exp[l]), cb.hq.ptr, cb.hd.ptr, ff as u64, sel0, total_sel as u64]);
                    self.gpu.run(MOE.kernel(kdb, CubeDim::new_1d(256)), ((E as u32).div_ceil(8), entries, 1), &[0, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, dptr, generation]);
                    // the mark covers every launch entry up to this group's last expert (the unused picks between them too)
                    let last_ord = group[group.len() - 1].1;
                    self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, last_ord]);
                }
                // and the entries after the last used one (unused picks at the tail)
                if let Some(&(_, ord)) = launch.last() {
                    self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, ord]);
                }
            } else {
            // QX_DS_CHUNK_STREAMS=N (default 1): each expert's launch set on one of N side streams, so N experts
            // overlap on the card (one expert's kernels are ~130 blocks: far too few to fill it). Each stream has its
            // own gate/up and hidden scratch; the outputs are per selection. The progress mark is written on the
            // main stream after it joined the side streams, every N used experts: it still means "every expert up
            // to this ordinal is done". The same kernels on the same inputs: the same bits.
            // (not with a staging ring too small to hold several groups of N, nor for a chunk of a few rows: the
            // driver frees a ring slot on the mark, which now comes once per N experts)
            let ns = if mma && qx_gpu::raw::RAW.load(std::sync::atomic::Ordering::Relaxed) && n_ring >= 8 * chunk_streams() && n_act >= 256 { chunk_streams() } else { 1 };
            let side: Vec<[u64; 3]> = if ns > 1 { self.chunk_side[..ns].iter().map(|b| [b[0].ptr, b[1].ptr, b[2].ptr]).collect() } else { Vec::new() };
            if ns > 1 {
                qx_gpu::raw::side_begin(ns);
            }
            let mut pending: Option<u64> = None;
            let side_cap = chunk_side_rows(rows_max);
            let mut n_side = 0usize;
            for &(e, ord) in &launch {
                let m = lists[e].len() as u32;
                if m > 0 && ns > 1 && m as usize > side_cap {
                    // too many rows for a side stream's scratch: on the main stream, after the side streams
                    n_used += 1;
                    let word = cb.slots_tbl[l % 2].ptr + 8 * e as u64;
                    let sl = cb.sel.ptr + (offs[e] * 4) as u64;
                    let passes = m.div_ceil(64);
                    qx_gpu::raw::side_join(ns);
                    self.gpu.run(MOE.kernel("qx_moe_wait_word", CubeDim::new_1d(1)), (1, 1, 1), &[word, generation]);
                    self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(128)), (((2 * ff) as u32).div_ceil(64), passes, 1), &[word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, m as u64, sl]);
                    self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (m, 1, 1), &[cb.gu.ptr, bits(clamp_exp[l]), cb.hq.ptr, cb.hd.ptr, ff as u64, sl, m as u64]);
                    self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(128)), ((E as u32).div_ceil(64), passes, 1), &[word, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, sl, m as u64]);
                    self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, ord]);
                    pending = None;
                    n_side = 0;
                    continue;
                }
                if m > 0 && ns > 1 {
                    let si = n_side % ns;
                    n_side += 1;
                    n_used += 1;
                    let word = cb.slots_tbl[l % 2].ptr + 8 * e as u64;
                    let sl = cb.sel.ptr + (offs[e] * 4) as u64;
                    let [gu, hq, hd] = side[si];
                    let passes = m.div_ceil(64);
                    qx_gpu::raw::side_use(Some(si));
                    self.gpu.run(MOE.kernel("qx_moe_wait_word", CubeDim::new_1d(1)), (1, 1, 1), &[word, generation]);
                    self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(128)), (((2 * ff) as u32).div_ceil(64), passes, 1), &[word, cb.xq.ptr, cb.xd.ptr, k as u64, gu, ff as u64, E as u64, ex.gate_row_bytes as u64, m as u64, sl]);
                    self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (m, 1, 1), &[gu, bits(clamp_exp[l]), hq, hd, ff as u64, sl, m as u64]);
                    self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(128)), ((E as u32).div_ceil(64), passes, 1), &[word, hq, hd, 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, sl, m as u64]);
                    qx_gpu::raw::side_use(None);
                    pending = Some(ord);
                    if n_side % ns == 0 {
                        qx_gpu::raw::side_join(ns);
                        self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, ord]);
                        pending = None;
                    }
                    continue;
                }
                if ns > 1 {
                    // an unused pick: its ordinal is covered by the next mark
                    pending = Some(ord);
                    continue;
                }
                if m > 0 {
                    n_used += 1;
                    let word = cb.slots_tbl[l % 2].ptr + 8 * e as u64;
                    let sl = cb.sel.ptr + (offs[e] * 4) as u64;
                    self.gpu.run(MOE.kernel("qx_moe_wait_word", CubeDim::new_1d(1)), (1, 1, 1), &[word, generation]);
                    if mma {
                        let passes = m.div_ceil(64);
                        self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(128)), (((2 * ff) as u32).div_ceil(64), passes, 1), &[word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, m as u64, sl]);
                        self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (m, 1, 1), &[cb.gu.ptr, bits(clamp_exp[l]), cb.hq.ptr, cb.hd.ptr, ff as u64, sl, m as u64]);
                        self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(128)), ((E as u32).div_ceil(64), passes, 1), &[word, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, sl, m as u64]);
                    } else if grouped {
                        let groups = m.div_ceil(8);
                        self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), groups, 1), &[word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, m as u64, sl]);
                        self.gpu.run(MOE.kernel("qx_moe_quant_h_sel", CubeDim::new_1d(256)), (m, 1, 1), &[cb.gu.ptr, bits(clamp_exp[l]), cb.hq.ptr, cb.hd.ptr, ff as u64, sl, m as u64]);
                        self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(256)), ((E as u32).div_ceil(8), groups, 1), &[word, cb.hq.ptr, cb.hd.ptr, 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, sl, m as u64]);
                    } else {
                        self.gpu.run(MOE.kernel(kg, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), m, 1), &[word, cb.xq.ptr, cb.xd.ptr, k as u64, cb.gu.ptr, ff as u64, E as u64, ex.gate_row_bytes as u64, 0, sl]);
                        self.gpu.run(MOE.kernel(kd, CubeDim::new_1d(256)), ((E as u32).div_ceil(16), m, 1), &[word, cb.gu.ptr, bits(clamp_exp[l]), 0, cb.out.ptr, ff as u64, E as u64, (2 * gb) as u64, ex.down_row_bytes as u64, sl]);
                    }
                }
                self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, ord]);
            }
            if ns > 1 {
                // the main stream runs on only after every side stream's experts (the combine reads their outputs)
                qx_gpu::raw::side_join(ns);
                if let Some(ord) = pending {
                    self.gpu.run(MOE.kernel("qx_moe_signal", CubeDim::new_1d(1)), (1, 1, 1), &[d.chunk_done.dptr, ord]);
                }
            }
            }
            let order_len = n_used;
            n_distinct += order_len;
            if verbose && pos0 == 0 {
                eprintln!("  chunk layer {l}: {} used experts, {} prefetched ({} of them used), {} streamed after the routing", n_used, picked.len(), n_pick_used, rest.len());
            }
            t_moe += tm.elapsed().as_secs_f64();
            // QX_DS_CHUNK_TIMING=1: a sync after the experts, so the layer's phases can be told apart (the next
            // layer's slab enqueue otherwise absorbs the experts' GPU time); costs the overlap it removes
            if std::env::var("QX_DS_CHUNK_TIMING").is_ok() {
                let host = tm.elapsed().as_secs_f64();
                self.gpu.sync();
                eprintln!("  timing layer {l}: slabs {:.1} ms (to the sync), experts {:.1} ms ({:.1} of host work before the sync; {} used, {} rows), layer start to here {:.1} ms", (tm - te).as_secs_f64() * 1e3, tm.elapsed().as_secs_f64() * 1e3, host * 1e3, n_used, n_act, te.elapsed().as_secs_f64() * 1e3);
            }
            let tp = std::time::Instant::now();
            let (out_ptr, shy_ptr) = (cb.out.ptr, cb.shy.ptr);
            let ones = self.s.ones_slots.ptr;
            for si in 0..n_slabs {
                let (r0, t) = slab(si);
                if r0 < ra {
                    continue;
                }
                self.chunk_views(r0, t);
                let o = out_ptr + ((r0 - ra) * k * E * 4) as u64;
                self.layer_ffn_post(l, t, o, o, ones, shy_ptr + (r0 * E * 4) as u64, &mut None);
                if trace && l < 8 {
                    self.gpu.sync();
                    let y: Vec<f32> = self.s.y.download(E);
                    let r: Vec<f32> = self.s.r.download(E);
                    let out: Vec<f32> = DeviceBuf::view(o, k * E * 4).download(k * E);
                    let st = |v: &[f32]| (v.iter().filter(|a| a.is_nan()).count(), v.iter().map(|a| a.abs()).fold(0f32, f32::max), v.iter().sum::<f32>());
                    eprintln!("    trace l{l} slab{si} after ffn_post: experts(row0) {:?} y {:?} r {:?}", st(&out), st(&y), st(&r));
                }
            }
            t_post += tp.elapsed().as_secs_f64();
        }
        // (the Engram thread writes the chunk's host rows: it ends before the buffers change hands)
        if let Some((h, _)) = engram_job.take() {
            h.join().expect("engram thread");
        }
        let (r0, t) = slab(n_slabs - 1);
        self.chunk_views(r0, t);
        if !kv_only {
            self.layer_head(t, &mut None);
        }
        self.gpu.sync();
        let logits = if kv_only { Vec::new() } else { self.s.logits.download_at::<f32>((t - 1) * n_vocab * 4, n_vocab) };
        // the window scratch back
        self.blas_map.borrow_mut().clear();
        let [r, pre_prev, pre_f, post, comb, x, ids_b, wts, tok, pos, slot_b] = saved;
        self.s.r = r; self.s.pre_prev = pre_prev; self.s.pre_f = pre_f; self.s.post = post; self.s.comb = comb; self.s.x = x;
        self.s.ids = ids_b; self.s.wts = wts; self.s.tok = tok; self.s.pos = pos; self.s.slot = slot_b;
        std::mem::swap(&mut self.s.erows_host, &mut self.chunk.as_mut().unwrap().erows_host);
        self.s.cand_sel = saved_cand.0;
        self.s.cand_count = saved_cand.1;
        self.st.sels = saved_sels;
        self.erows_stride = DS_TMAX;
        self.erows_row0 = 0;
        self.st.pos[slot] += rows;
        d.tokens_done.fetch_add(rows as u64, std::sync::atomic::Ordering::Release);
        self.adapt().expect("tier promotions");
        self.t_phase = [0.0, t_moe, 0.0, 0.0, 0.0];
        self.t_forward = tf.elapsed().as_secs_f64();
        if std::env::var("QX_DS_VERBOSE").is_ok() {
            eprintln!("chunk of {rows} rows at {pos0}{}: {:.0} ms ({:.2} ms per row; host: slab enqueue {:.0} ms, sync on the slab passes {:.0}, expert enqueue {:.0}, post enqueue {:.0}; {} distinct experts per layer, engram {:.0} ms)", if kv_only { " KV-only".to_string() } else if keep_from > 0 { format!(" (full stack from row {keep_from})") } else { String::new() }, self.t_forward * 1e3, self.t_forward * 1e3 / rows as f64, t_enq * 1e3, t_sync * 1e3, t_moe * 1e3, t_post * 1e3, n_distinct / n_layers_run.max(1), self.t_engram * 1e3);
        }
        logits
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
        self.forward_segments_opt(toks, act, false)
    }

    /// The last layer whose compressed store and index keys are its own (the KV source of every layer above it).
    fn kv_last(&self) -> usize {
        (0..self.cfg.n_layer).filter(|&l| self.cfg.kv_src[l] == l as i32).max().unwrap_or(self.cfg.n_layer - 1)
    }

    /// Rows the chunk path can take at once (`QX_DS_CHUNK` at load), 0 without chunk buffers.
    pub fn chunk_rows(&self) -> usize {
        self.chunk.as_ref().map_or(0, |c| c.rows)
    }

    /// Prompt positions that must still follow a window for it to run KV-only (`forward_segments_opt`): the layers
    /// above the last KV source see earlier positions only through that source's store and keys and their own
    /// sliding windows, so a position needs the full stack only if a later position within n_swa does at the layer
    /// above, and so on up the stack: n_swa + (n_layer - kv_last) * (n_swa - 1) positions before the prompt's end,
    /// plus n_swa so that a rewind within the last DS_TMAX positions still finds exact rings at every layer
    /// (2,796 here). Windows of a prompt with at least this many positions after them run KV-only exactly: every
    /// kept position computes what the full pass computes, bit for bit.
    pub fn prompt_keep(&self) -> usize {
        2 * self.cfg.n_swa + (self.cfg.n_layer - self.kv_last()) * (self.cfg.n_swa - 1)
    }

    /// `forward_segments`, KV-only when `kv_only`: the window runs the layers up to the last KV source and stops
    /// after that layer's ring, store and index-key writes (no attention output, no experts there, nothing
    /// above), and returns no logits. Only for prompt windows at least `prompt_keep` positions before the end of
    /// their slot's prompt, all tokens in one slot's prompt; the caller decides.
    pub fn forward_segments_opt(&mut self, toks: &[(usize, u32)], act: Option<&mut dyn FnMut(&str, Vec<f32>)>, kv_only: bool) -> Vec<f32> {
        let out = self.forward_segments_inner(toks, act, kv_only);
        if sched_trace_on() {
            let segs: Vec<String> = toks.iter().map(|&(s, t)| format!("{s}:{t}")).collect();
            sched_trace(&format!("F {} {}", kv_only as u8, segs.join(",")));
            sched_trace(&format!("H {}", row_hashes(&out, toks.len())));
        }
        out
    }

    fn forward_segments_inner(&mut self, toks: &[(usize, u32)], act: Option<&mut dyn FnMut(&str, Vec<f32>)>, kv_only: bool) -> Vec<f32> {
        let tf = std::time::Instant::now();
        self.t_experts = 0.0;
        let t = toks.len();
        assert!((1..=DS_TMAX).contains(&t), "window of {t} tokens");
        assert!(t == 1 || (self.driver.is_some() && act.is_none()), "windows need the asynchronous driver and no dumps");
        assert!(!kv_only || (self.driver.is_some() && act.is_none()), "KV-only windows need the asynchronous driver and no dumps");
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
        self.t_engram = self.gather_engram(toks, &positions, &self.s.erows_host, DS_TMAX);
        let c = &self.cfg;
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
        self.warm_width(t, pos_min, pos_max);
        let eligible = self.driver.is_some() && act.is_none() && pos_min + 1 >= max_ratio && !self.replaying && !layer_trace_on() && !std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") && std::env::var("QX_DS_SYNC").is_err() && std::env::var("QX_DS_TRACE").is_err();
        let captured = if kv_only { &self.graphs_kv[t] } else { &self.graphs[t] };
        if !eligible {
            self.launch_window(t, pos_min, pos_max, act, kv_only);
        } else if let Some(g) = captured {
            // SAFETY: every buffer the graph reads or writes is owned by `self` and outlives it.
            unsafe { g.replay().expect("graph replay") };
        } else {
            self.gpu.client.graph_prepare().expect("graph_prepare");
            self.launch_window(t, pos_min, pos_max, None, kv_only);
            // Drain the eager run, so no expert job is in flight while the device records nothing, then keep the
            // driver out of its event syncs for the recording (see `Shared::capture`).
            self.gpu.sync();
            let shared = self.driver.as_ref().map(|d| d.shared.clone());
            let not_capturing = shared.as_ref().map(|s| s.capture.lock().unwrap());
            self.gpu.client.start_capture().expect("start_capture");
            self.launch_window(t, pos_min, pos_max, None, kv_only);
            let g = self.gpu.client.stop_capture().expect("stop_capture");
            drop(not_capturing);
            if std::env::var("QX_DS_VERBOSE").is_ok() {
                eprintln!("window graph ({t} tokens{}) captured at position {pos_min}", if kv_only { ", KV-only" } else { "" });
            }
            if kv_only {
                self.graphs_kv[t] = Some(g);
            } else {
                self.graphs[t] = Some(g);
            }
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
        let out = if kv_only { Vec::new() } else { self.s.logits.download(self.cfg.n_vocab * t) };
        self.t_phase[4] = tp.elapsed().as_secs_f64();
        self.t_forward = tf.elapsed().as_secs_f64();
        out
    }

    /// Queues a window's launches (everything between the uploads and the final sync): `t` tokens whose
    /// positions lie in [pos, last] (the device arrays hold each token's own).
    /// Downloads `n` floats of `b` to the activation dump `act` (each dump syncs the device).
    fn dump(&self, name: String, b: &DeviceBuf, n: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        if let Some(f) = act {
            self.gpu.sync();
            f(&name, b.download(n));
        }
    }

    /// Queues a window's launches (everything between the uploads and the final sync): `t` tokens whose
    /// positions lie in [pos, last] (the device arrays hold each token's own). The layer body is in four parts
    /// (`layer_attn`, `layer_ffn_pre`, the experts, `layer_ffn_post`) that the chunk path (`forward_chunk`)
    /// reuses slab by slab, layer-major.
    fn launch_window(&mut self, t: usize, pos: usize, last: usize, mut act: Option<&mut dyn FnMut(&str, Vec<f32>)>, kv_only: bool) {
        self.layer_init(t, &mut act);
        for l in 0..self.cfg.n_layer {
            if predict_mode() == 1 {
                self.predict_pre(l, t, false);
            }
            if !self.layer_attn(l, t, pos, last, &mut act, kv_only) {
                return;
            }
            self.layer_ffn_pre(l, t, &mut act);
            let (combine_gpu, cpu_rows, slots_ptr) = self.layer_moe_window(l, t, &mut act);
            self.layer_ffn_post(l, t, combine_gpu, cpu_rows, slots_ptr, self.s.shy.ptr, &mut act);
            if layer_trace_on() && !self.warm {
                // `Y layer hash...`: the residual streams of every row after this layer (eager windows only)
                self.gpu.sync();
                let hcd = self.cfg.hc * E;
                let r: Vec<f32> = self.s.r.download(t * hcd);
                sched_trace(&format!("Y {l} {}", row_hashes(&r, t)));
            }
            std::mem::swap(&mut self.s.pre_prev, &mut self.s.pre_f);
        }
        self.layer_head(t, &mut act);
    }

    /// The embeddings of the window's tokens into the hyper-connection streams.
    fn layer_init(&self, t: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        let c = &self.cfg;
        let hcd = c.hc * E;
        self.trace("dequant_gather");
        ops::dequant_gather(&self.gpu, self.embd.ty, self.embd.buf.ptr, self.s.e.ptr, t, self.embd.cols, self.s.tok.ptr);
        self.trace_done();
        self.k("qx_ds_hc_init", 256, ((hcd as u32).div_ceil(256), t as u32, 1), &[self.s.r.ptr, self.s.e.ptr, E as u64, c.hc as u64]);
        self.dump("hc_init".into(), &self.s.r, hcd, act);
    }

    /// The final norm and the vocabulary head over the window's rows (logits in `s.logits`).
    fn layer_head(&self, t: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        self.hc_pre(&self.s.pre_prev, t);
        self.dump("hc_out".into(), &self.s.xpre, E, act);
        self.rms(self.s.xpre.ptr, self.output_norm.ptr, self.s.x.ptr, E, t);
        self.dump("result_norm".into(), &self.s.x, E, act);
        self.mgt(&self.g_out, self.s.x.ptr, E, t, "g_out");
    }

    /// Layer `l`'s Engram and attention sublayer for `t` rows at positions [pos, last], through its hc_post.
    /// Returns false when a KV-only window stops here (the last KV source's ring, store and keys are written).
    fn layer_attn(&self, l: usize, t: usize, pos: usize, last: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>, kv_only: bool) -> bool {
        let kv_last = self.kv_last();
        let c = &self.cfg;
        let (hc, hcd) = (c.hc, c.hc * E);
        let tt = t as u32;
        let w = &self.layers[l];
        let ratio = c.ratio[l];
        if let Some((g_ekv, egate)) = &w.engram {
            let ei = c.engram_layers.iter().position(|&x| x == l).unwrap();
            let n = c.engram_cols() * c.engram_dim;
            self.k("qx_ds_copy_f32", 256, (((n * t) as u32).div_ceil(256), 1, 1), &[self.s.erows_host.dptr + (ei * self.erows_stride * n * 4) as u64 + (self.erows_row0 * n * 4) as u64, self.s.erows.ptr, (n * t) as u64]);
            self.mgt(g_ekv, self.s.erows.ptr, n, t, "engram_kv");
            self.dump(format!("engram_kv-{l}"), &self.s.ekv, (hc + 1) * E, act);
            self.k("qx_ds_engram_gate", 256, (hc as u32, tt, 1), &[self.s.r.ptr, self.s.ekv.ptr, egate.ptr, E as u64, bits(1e-20), self.s.egates.ptr]);
            self.dump(format!("engram_gate-{l}"), &self.s.egates, hc, act);
            self.dump(format!("engram_out-{l}"), &self.s.r, hcd, act);
        }
        // attention
        self.hc_terms(&w.hc_attn, &self.s.pre_a, t);
        if predict_mode() == 3 {
            self.predict_pre(l, t, true);
        }
        self.dump(format!("hc_mixes-{l}"), &self.s.mixsum, 24, act);
        self.dump(format!("hc_pre-{l}"), &self.s.pre_a, hc, act);
        self.dump(format!("hc_post-{l}"), &self.s.post, hc, act);
        self.dump(format!("hc_comb-{l}"), &self.s.comb, hc * hc, act);
        self.hc_pre(if l > 0 { &self.s.pre_prev } else { &self.s.pre_first }, t);
        self.dump(format!("hc_attn_pre-{l}"), &self.s.xpre, E, act);
        self.rms(self.s.xpre.ptr, w.attn_norm.ptr, self.s.x.ptr, E, t);
        self.dump(format!("attn_norm-{l}"), &self.s.x, E, act);
        self.mgt(&w.g_x, self.s.x.ptr, E, t, "g_x");
        self.rms(self.s.qr_raw.ptr, w.q_a_norm.ptr, self.s.qr.ptr, c.q_lora, t);
        self.dump(format!("qr_norm-{l}"), &self.s.qr, c.q_lora, act);
        self.mgt(&w.g_qr, self.s.qr.ptr, c.q_lora, t, "g_qr");
        let comp = ratio != 0;
        let nope = c.head_dim - c.n_rot;
        self.rope(self.s.q.ptr, c.n_head, c.head_dim, nope, comp, 0, false, t);
        self.dump(format!("q-{l}"), &self.s.q, c.n_head * c.head_dim, act);
        self.rms(self.s.kv_raw.ptr, w.kv_norm.ptr, self.s.kv.ptr, c.head_dim, t);
        self.rope(self.s.kv.ptr, 1, c.head_dim, nope, comp, 0, false, t);
        self.dump(format!("kv-{l}"), &self.s.kv, c.head_dim, act);
        self.k("qx_ds_kv_append", 256, (tt, 1, 1), &[self.s.kv.ptr, self.st.ring_tables[l].ptr, self.ring_w as u64, self.s.pos.ptr, self.s.slot.ptr, c.head_dim as u64]);
        let row = c.head_dim + c.idx_dim;
        if w.has_comp && !self.replaying {
            let (_, st, _, pt) = self.st.stores[l].as_ref().unwrap();
            let score = if ratio > 1 { self.s.cscore.ptr } else { 0 };
            self.k("qx_ds_compress", 256, (1, 1, 1), &[self.s.ckv.ptr, score, pt.ptr, st.ptr, w.comp_norm.as_ref().unwrap().ptr, self.s.comp_tmp.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.head_dim as u64, row as u64, bits(1e-20), t as u64]);
            if (pos + 1) % ratio == 0 {
                self.dump(format!("comp_latent-{l}"), &self.s.comp_tmp, c.head_dim, act);
            }
            self.mgt(w.g_ik.as_ref().unwrap(), self.s.comp_tmp.ptr, c.head_dim, t, "g_ik");
            let (base, fs, c0, c1, ext) = rope_params(c, true);
            self.k("qx_ds_idx_key_commit", 128, (tt, 1, 1), &[self.s.ik.ptr, st.ptr, w.idx_k_norm.as_ref().unwrap().ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.idx_dim as u64, row as u64, c.head_dim as u64, bits(1e-20)]);
            // the latent at [448, 512) and the key at [512 + 64, 640), both at the group's first position
            self.k("qx_ds_rope_row", (c.n_rot / 2) as u32, (tt, 1, 1), &[st.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, row as u64, nope as u64, c.n_rot as u64, bits(base), bits(fs), bits(c0), bits(c1), ext]);
            self.k("qx_ds_rope_row", (c.n_rot / 2) as u32, (tt, 1, 1), &[st.ptr, self.s.flag.ptr, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, row as u64, (c.head_dim + c.idx_dim - c.n_rot) as u64, c.n_rot as u64, bits(base), bits(fs), bits(c0), bits(c1), ext]);
        }
        // KV-only: the last KV source's ring, store and keys are written; its attention output, its experts
        // and every layer above feed only this position's own logits, which nothing needs (`prompt_keep`)
        if kv_only && l == kv_last {
            return false;
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
                // QX_DS_TOPK_KERNEL=1 keeps the original one-block top-k; 2 (default) the tiled one, the same k indices in
                // the same order (ds-checks, the 16k tails, the 256k decode's tokens)
                static TOPK_KERNEL: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
                let topk_entry = *TOPK_KERNEL.get_or_init(|| if std::env::var("QX_DS_TOPK_KERNEL").is_ok_and(|v| v == "1") { "qx_ds_topk" } else { "qx_ds_topk2" });
                self.k(idx_entry, 256, ((self.max_rows as u32).div_ceil(8), tt, 1), &[self.s.iq.ptr, self.s.iw.ptr, store_table, self.s.pos.ptr, self.s.slot.ptr, ratio as u64, c.idx_dim as u64, row as u64, c.head_dim as u64, self.s.scores.ptr, max_rows, bits(1.0 / ((c.idx_heads * c.idx_dim) as f32).sqrt())]);
                self.dump(format!("idx_score-{l}"), &self.s.scores, n_visible, act);
                let (sb, cb) = self.st.sels[l].as_ref().unwrap();
                let bs = c.cand_bs.max(1);
                let n_blk = n_visible.div_ceil(bs);
                let words = self.max_rows.div_ceil(bs * 32) as u64;
                if c.cand_src == l as i32 {
                    let max_blocks = self.max_rows.div_ceil(bs) as u64;
                    self.k("qx_ds_block_max", 256, ((max_blocks as u32).div_ceil(256), tt, 1), &[self.s.scores.ptr, self.s.bmax.ptr, self.s.pos.ptr, ratio as u64, bs as u64, max_rows, max_blocks]);
                    self.dump(format!("cand_block_score-{l}"), &self.s.bmax, n_blk, act);
                    // top blocks: the scores are the block maxima, "rows" are blocks: positions / (ratio * bs)
                    self.k(topk_entry, 1024, (tt, 1, 1), &[self.s.bmax.ptr, c.cand_blocks as u64, self.s.cand_sel.ptr, self.s.cand_count.ptr, self.s.pos.ptr, (ratio * bs) as u64, max_blocks]);
                } else if c.cand_src >= 0 && (c.cand_src as usize) < l {
                    self.k("qx_ds_zero_u32", 256, (((words * t as u64) as u32).div_ceil(256), 1, 1), &[self.s.inside.ptr, words * t as u64]);
                    self.k("qx_ds_mark_blocks", 256, ((c.cand_blocks as u32).div_ceil(256), tt, 1), &[self.s.cand_sel.ptr, self.s.cand_count.ptr, self.s.inside.ptr, c.cand_blocks as u64, words]);
                    self.k("qx_ds_cand_mask", 256, ((self.max_rows as u32).div_ceil(256), tt, 1), &[self.s.scores.ptr, self.s.inside.ptr, self.s.pos.ptr, ratio as u64, bs as u64, max_rows, words]);
                }
                self.k(topk_entry, 1024, (tt, 1, 1), &[self.s.scores.ptr, c.idx_top_k as u64, sb.ptr, cb.ptr, self.s.pos.ptr, ratio as u64, max_rows]);
            }
            let (sb, cb) = self.st.sels[tsrc].as_ref().expect("top-k source");
            sel = Some((sb.ptr, cb.ptr, store_table));
        }
        let (sp, cp, stp) = sel.unwrap_or((0, 0, 0));
        // the attention in slices of ATTN_SLICE tokens (the partials buffer is sized for one slice); the
        // kernels index tokens from the pointers they get, so each slice passes offset pointers
        let hd = c.head_dim;
        // the slab kernel holds the partials buffer only as scratch (12 scores per (token, head, chunk)), so a call
        // covers the whole window: 4x the blocks of a 32-token slice (256 -> 1,024 per call at 128 tokens)
        // prompt chunks in f16 (the layer's dense weights are decoded: `blas_map` is filled): the attention on tensor
        // cores too, one launch for the slab (`qx_ds_attn16`; QX_DS_ATTN16=0 keeps the exact kernels)
        let attn16 = attn16_on() && hd == 512 && c.n_head % 16 == 0 && !self.blas_map.borrow().is_empty();
        let slice = if attn16 || (attn_tile() && self.attn_chunks <= 64) { t.max(1) } else { ATTN_SLICE };
        for s0 in (0..t).step_by(slice) {
            let ts = (t - s0).min(slice);
            let (q, o, ps, sl) = (self.s.q.ptr + (s0 * c.n_head * hd * 4) as u64, self.s.attn.ptr + (s0 * c.n_head * hd * 4) as u64, self.s.pos.ptr + (s0 * 4) as u64, self.s.slot.ptr + (s0 * 4) as u64);
            let (sps, cps) = if sp != 0 { (sp + (s0 * c.idx_top_k * 4) as u64, cp + (s0 * 4) as u64) } else { (0, 0) };
            if attn16 {
                self.trace("qx_ds_attn16");
                self.gpu.run(qx_gpu::kernel::modules::GEMM.kernel("qx_ds_attn16", CubeDim::new_1d(256)), ((c.n_head / 16) as u32, ts as u32, 1), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64]);
                self.trace_done();
            } else if std::env::var("QX_DS_ATTN").is_ok_and(|v| v == "old") {
                self.k("qx_ds_attn", 256, (c.n_head as u32, ts as u32, 1), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64]);
            } else if attn_tile() && self.attn_chunks <= 64 {
                // the slab form: one block per (8 heads, token) over all the chunks, the merge in-block (the same bits;
                // the partials buffer holds the chunks' scores). QX_DS_ATTN=tile
                let chunks = self.attn_chunks as u64;
                self.k("qx_ds_attn_tile", 256, ((c.n_head / 8) as u32, ts as u32, 1), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64, self.s.attn_part.ptr, chunks]);
            } else {
                let chunks = self.attn_chunks as u64;
                self.k("qx_ds_attn_part", 256, ((c.n_head / 8) as u32, chunks as u32, ts as u32), &[q, self.st.ring_tables[l].ptr, self.ring_w as u64, ps, sl, sps, cps, stp, row as u64, w.sinks.ptr, o, bits(1.0 / (hd as f32).sqrt()), c.n_swa as u64, c.idx_top_k as u64, hd as u64, self.s.attn_part.ptr, chunks]);
                self.k("qx_ds_attn_merge", 256, (c.n_head as u32, ts as u32, 1), &[self.s.attn_part.ptr, w.sinks.ptr, o, chunks, hd as u64]);
            }
        }
        self.dump(format!("{}-{l}", if sel.is_some() { "attn_comp" } else { "attn_raw" }), &self.s.attn, c.n_head * c.head_dim, act);
        self.rope(self.s.attn.ptr, c.n_head, c.head_dim, nope, comp, 0, true, t);
        self.dump(format!("attn_derope-{l}"), &self.s.attn, c.n_head * c.head_dim, act);
        let gd = c.n_head / c.o_groups * c.head_dim;
        // (prompt chunks in f16: the groups' products in one launch, the weights being one after the other)
        let wo_base = {
            let map = self.blas_map.borrow();
            let s0 = &w.wo_a[0].segs[0];
            let step = (s0.rows * s0.cols * 2) as u64;
            let base = map.get(&(&w.wo_a[0] as *const MGemv as usize)).copied();
            base.filter(|&b| !ops::gemm16_cublas() && w.wo_a.iter().enumerate().all(|(gi, g)| g.segs.len() == 1 && g.segs[0].qx != 0 && g.segs[0].rows == s0.rows && g.segs[0].cols == gd && g.segs[0].y == s0.y + (gi * DS_TMAX * s0.rows * 4) as u64 && map.get(&(g as *const MGemv as usize)) == Some(&(b + gi as u64 * step))))
        };
        if let Some(base) = wo_base {
            let bb = self.blas_bufs.as_ref().unwrap();
            self.trace("wo_a");
            ops::run_blas_groups(&self.gpu, &w.wo_a, self.s.attn.ptr, c.n_head * c.head_dim, gd, t, bb.1.ptr, base, bb.2.ptr, bb.2.bytes);
            self.trace_done();
        } else {
            for (gi, g) in w.wo_a.iter().enumerate() {
                self.mgt(g, self.s.attn.ptr + (gi * gd * 4) as u64, c.n_head * c.head_dim, t, "wo_a");
            }
        }
        // the groups' rows per token, contiguous for the second projection
        self.k("qx_ds_regroup", 256, (((c.o_groups * c.o_lora) as u32).div_ceil(256), tt, 1), &[self.s.oag.ptr, self.s.oa.ptr, c.o_groups as u64, c.o_lora as u64, DS_TMAX as u64]);
        self.dump(format!("attn_wo_a-{l}"), &self.s.oa, c.o_groups * c.o_lora, act);
        self.mgt(&w.g_ob, self.s.oa.ptr, c.o_groups * c.o_lora, t, "g_ob");
        self.dump(format!("attn_out-{l}"), &self.s.y, E, act);
        self.hc_post(t);
        self.dump(format!("hc_attn_post-{l}"), &self.s.r, hcd, act);
        true
    }

    /// Layer `l`'s FFN sublayer up to the routing: the hc terms (into `pre_f`), the norm into `x`, the router
    /// logits and the selections (`ids`, `wts`).
    fn layer_ffn_pre(&self, l: usize, t: usize, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        let c = &self.cfg;
        let w = &self.layers[l];
        self.hc_terms(&w.hc_ffn, &self.s.pre_f, t);
        self.hc_pre(&self.s.pre_a, t);
        self.dump(format!("hc_ffn_pre-{l}"), &self.s.xpre, E, act);
        self.rms(self.s.xpre.ptr, w.ffn_norm.ptr, self.s.x.ptr, E, t);
        if predict_mode() == 2 {
            self.predict_next(l, t);
        }
        self.dump(format!("ffn_norm-{l}"), &self.s.x, E, act);
        self.mgt(&w.g_ffn, self.s.x.ptr, E, t, "g_ffn");
        self.dump(format!("ffn_moe_logits-{l}"), &self.s.logits_r, c.n_expert, act);
        self.k("qx_ds_route", 32, (t as u32, 1, 1), &[self.s.logits_r.ptr, c.n_expert as u64, c.n_expert_used as u64, w.router_bias.ptr, self.s.ids.ptr, self.s.wts.ptr, bits(c.w_scale)]);
    }

    /// The FFN's dense part for every row of a chunk at once (`QX_DS_CHUNK_GEMM=1`): the router logits (f32) and the
    /// shared expert's gate and up from the chunk's normed rows, the selections, and the shared expert's down
    /// product into the chunk's `shy`: half-precision tensor-core GEMMs with f32 accumulation over all the rows,
    /// the weights decoded to f16 once per matrix. KL-bounded against the per-slab exact path.
    /// `bytes` of chunk-path scratch: from the flex tail's spare region while it lasts, else its own allocation.
    fn cscratch(&mut self, bytes: usize) -> DeviceBuf {
        let bytes = bytes.div_ceil(256) * 256;
        if let Some(cb) = self.chunk.as_ref() {
            if self.chunk_extra_off + bytes <= cb.extra.bytes {
                let v = DeviceBuf::view(cb.extra.ptr + self.chunk_extra_off as u64, bytes);
                self.chunk_extra_off += bytes;
                return v;
            }
        }
        if std::env::var("QX_DS_VERBOSE").is_ok() {
            eprintln!("chunk scratch: {} MiB outside the flex tail ({} of {} MiB of it handed out)", bytes >> 20, self.chunk_extra_off >> 20, self.chunk.as_ref().map_or(0, |c| c.extra.bytes >> 20));
        }
        DeviceBuf::new(bytes)
    }

    fn chunk_ffn_alloc(&mut self, rows_max: usize) {
        let (ne, ff) = (self.cfg.n_expert, self.cfg.ff_exp);
        if self.chunk_ffn_bufs.as_ref().is_none_or(|(n, _)| *n < rows_max) {
            let w16 = self.layers.iter().map(|w| w.g_ffn.w16_halves().max(w.g_shd.w16_halves())).max().unwrap_or(0);
            let bufs = [self.cscratch(rows_max * ne * 4), self.cscratch(rows_max * ff * 4), self.cscratch(rows_max * ff * 4), self.cscratch(rows_max * ff * 4), self.cscratch(rows_max * E.max(ff) * 2 + 4), self.cscratch(w16 * 2 + 4)];
            self.chunk_ffn_bufs = Some((rows_max, bufs));
        }
    }

    fn chunk_ffn(&mut self, l: usize, ra: usize, rows: usize, rows_max: usize) {
        self.chunk_ffn_alloc(rows_max);
        let [lg, g, u, h, x16, w16] = { let (_, b) = self.chunk_ffn_bufs.as_ref().unwrap(); [b[0].ptr, b[1].ptr, b[2].ptr, b[3].ptr, b[4].ptr, b[5].ptr] };
        let c = &self.cfg;
        let w = &self.layers[l];
        let cb = self.chunk.as_ref().unwrap();
        // (rows `ra` on: the chunk's rows from which this layer runs)
        let ku = (ra * c.n_expert_used * 4) as u64;
        w.g_ffn.run_blas_to(&self.gpu, cb.x.ptr + (ra * E * 4) as u64, rows, &[lg, g, u], x16, w16);
        self.k("qx_ds_route", 32, (rows as u32, 1, 1), &[lg, c.n_expert as u64, c.n_expert_used as u64, w.router_bias.ptr, cb.ids.ptr + ku, cb.wts.ptr + ku, bits(c.w_scale)]);
        self.k("qx_ds_silu_mul_clamp", 256, (((c.ff_exp * rows) as u32).div_ceil(256), 1, 1), &[g, u, h, (c.ff_exp * rows) as u64, bits(c.clamp_shexp[l])]);
        w.g_shd.run_blas_to(&self.gpu, h, rows, &[cb.shy.ptr + (ra * E * 4) as u64], x16, w16);
    }

    /// The shared expert of layer `l` (`shg`, `shu` from the FFN GEMV group into `shy`).
    fn layer_shexp(&self, l: usize, t: usize) {
        let c = &self.cfg;
        self.k("qx_ds_silu_mul_clamp", 256, (((c.ff_exp * t) as u32).div_ceil(256), 1, 1), &[self.s.shg.ptr, self.s.shu.ptr, self.s.shh.ptr, (c.ff_exp * t) as u64, bits(c.clamp_shexp[l])]);
        self.mgt(&self.layers[l].g_shd, self.s.shh.ptr, c.ff_exp, t, "g_shd");
    }

    /// The routed experts of a window (the asynchronous driver, or the host paths for one token), with the
    /// shared expert run where it overlaps best. Returns (GPU outputs, CPU rows, slot flags) for the combine.
    fn layer_moe_window(&mut self, l: usize, t: usize, _act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) -> (u64, u64, u64) {
        let c = &self.cfg;
        let tt = t as u32;
        let gpu = &self.gpu;
        let k = c.n_expert_used;
        // QX_DS_NOEXPERTS=1: skip the routed experts (a timing knob: the dense and attention cost alone)
        let no_experts = std::env::var("QX_DS_NOEXPERTS").is_ok_and(|v| v == "1");
        let mut shexp_done = false;
        let out = if no_experts {
            self.s.yk.zero();
            (self.s.yk.ptr, self.s.yk.ptr, self.s.ones_slots.ptr)
        } else if let (Some(d), Some(ab)) = (&self.driver, &self.abufs) {
            // the misses to the driver's mailbox (x quantized to Q8_K on the way), the residents on the GPU,
            // then wait for the CPU's rows: no host involvement
            let mb = d.shared.mb.dptr;
            let stage_share = if t >= ab.stage_wide_min { ab.stage_share } else if t >= 2 { ab.stage_share_narrow } else { ab.stage_share_decode };
            self.trace("qx_moe_post");
            gpu.run(STEP.kernel("qx_moe_post", CubeDim::new_1d(256)), (tt, 1, 1), &[mb, ab.table_dev.ptr, (l * c.n_expert) as u64, self.s.ids.ptr, self.s.wts.ptr, self.s.x.ptr, ab.step.ptr, l as u64, k as u64, E as u64, ab.slots.ptr, ab.xq.ptr, ab.xd.ptr, t as u64 | ((self.warm as u64) << 32), ab.rtable_dev.ptr, ab.gpu_share, stage_share, ab.slots_all.ptr, ab.slots_stg.ptr, ab.stage_cap as u64, ab.prestaged.ptr]);
            self.trace_done();
            self.dsync("post");
            self.launch_resident(l, k, t, ab.slots.ptr, ab.xq.ptr, ab.xd.ptr, ab.gu.ptr, ab.gpu_out.ptr);
            self.dsync("resident");
            // the shared expert while the pool works on the misses
            self.layer_shexp(l, t);
            if stage_share > 0 {
                // the staged experts: their records and slot addresses land on the copy stream, then the
                // same kernels over the staged slot array (zero where nothing is staged)
                if driver::stage_ready_by_words() {
                    self.trace("qx_moe_wait_staged2");
                    gpu.run(STEP.kernel("qx_moe_wait_staged2", CubeDim::new_1d(1)), (1, 1, 1), &[mb, ab.step.ptr, ab.stg_ready.ptr, driver::n_stage_streams() as u64]);
                } else {
                    self.trace("qx_moe_wait_staged");
                    gpu.run(STEP.kernel("qx_moe_wait_staged", CubeDim::new_1d(1)), (1, 1, 1), &[mb, ab.step.ptr]);
                }
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
            self.layer_shexp(l, t);
        }
        out
    }

    /// Layer `l`'s FFN sublayer after the experts: the combine of the k outputs with the shared expert into `y`,
    /// then hc_post into the streams.
    fn layer_ffn_post(&self, l: usize, t: usize, combine_gpu: u64, cpu_rows: u64, slots_ptr: u64, shy: u64, act: &mut Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        let c = &self.cfg;
        let hcd = c.hc * E;
        self.dump(format!("ffn_shexp-{l}"), &self.s.shy, E, act);
        self.k("qx_ds_moe_combine", 256, ((E as u32).div_ceil(256), t as u32, 1), &[combine_gpu, c.n_expert_used as u64, cpu_rows, self.s.y.ptr, E as u64, slots_ptr, self.s.wts.ptr, shy]);
        self.dsync("combine");
        self.dump(format!("ffn_out-{l}"), &self.s.y, E, act);
        self.hc_post(t);
        self.dump(format!("l_out-{l}"), &self.s.r, hcd, act);
        self.dsync("layer");
    }
}

/// Whether the attention over a window runs the slab kernel (`QX_DS_ATTN=tile`: one block per 8 heads and token over
/// all the key chunks, the merge in-block), the same bits as the chunked kernels.
fn attn16_on() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_DS_ATTN16").is_ok_and(|v| v == "0"))
}

fn attn_tile() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_DS_ATTN").is_ok_and(|v| v == "tile"))
}
