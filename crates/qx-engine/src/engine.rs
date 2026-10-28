//! Phase 3 engine: the forward pass on the GPU, with the routed experts on the CPU.
//!
//! Correct before fast: one token per call, eager launches, a device sync around every CPU step, f32 KV
//! cache in VRAM. Later phases replace pieces of this without changing what it computes.
use crate::experts::{ExpertCache, LayerExperts};
use crate::host_experts::HostExperts;
use crate::mailbox::{self, CacheState, Driver, LayerPtrs, PleSource, SwapConfig};
use crate::ops::{self, MGemv, Seg, GEMV_STORE};
use qx_cpu::pool::GateFmt;
use qx_gpu::kernel::modules::{ATTN, STEP};
use std::sync::atomic::Ordering;
use anyhow::{anyhow, Result};
use cubecl::prelude::CubeDim;
use qx_gpu::kernel::modules::{LAYER, MOE};
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;
use qx_pack::ggml::GgmlType;
use qx_pack::gguf::Gguf;
use qx_ref::model::{Config, E, HC, HCD};
use qx_ref::weights::W;

const EPS: f32 = 1e-6;
/// Tokens per speculative window, tokens per window across sequences, u32 step words per sequence slot
/// (kernels/include/qx_sys.cuh).
pub const WMAX: usize = mailbox::WMAX;
pub const TMAX: usize = mailbox::TMAX;
const STEP_ROW: usize = mailbox::STEP_ROW;
/// Entries of the lane ring: the host queues at most that many windows ahead of the GPU.
const LANE_RING: usize = 64;
/// The fused GEMV's token count for a window of t tokens (every count from 1 to 8 has its own entry).
fn tpad(t: usize) -> usize {
    t
}
/// Slots of the indexer's raw-key ring with the FP8 cache: a block plus a window's rejected cells.
pub(crate) const IDX_RING: usize = 16;

#[path = "prefill.rs"]
mod prefill;
use prefill::PrefillBufs;
#[path = "drafter.rs"]
mod drafter;
use drafter::Drafter;
#[path = "prefix.rs"]
pub mod prefix;

/// A weight matrix in VRAM: GGUF bytes, Q6_K and Q8_0 regrouped into the planar layout (`ops::to_planar`).
pub struct Mat {
    pub buf: DeviceBuf,
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Kernel type id and row stride of the bytes in `buf`.
    pub qx: u64,
    pub stride: usize,
}

struct HcW {
    norm: DeviceBuf,
    down: Mat,
    up: Mat,
    inject: Option<Mat>,
}

struct GdnW {
    qkv: Mat,
    gate: Mat,
    beta: Mat,
    alpha: Mat,
    dt_bias: DeviceBuf,
    a: DeviceBuf,
    conv: DeviceBuf,
    norm: DeviceBuf,
    out: Mat,
}

struct AttnW {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    iq: Mat,
    ik: Mat,
    iq_norm: DeviceBuf,
    ik_norm: DeviceBuf,
}

struct PleW {
    key: Mat,
    value: Mat,
    norm_key: DeviceBuf,
    norm_query: DeviceBuf,
    norm_conv: DeviceBuf,
    conv: DeviceBuf,
}

enum MixerW {
    Gdn(GdnW),
    Attn(AttnW),
}

struct LayerW<'a> {
    hc_attn: HcW,
    hc_ffn: HcW,
    mixer: MixerW,
    router: Mat,
    experts: LayerExperts<'a>,
    sh_gate: Mat,
    sh_up: Mat,
    sh_down: Mat,
    sh_router: Mat,
    ple: Option<PleW>,
}

/// The fused matrix-vector launches of one layer (and of the head, in `head_g`).
struct Groups {
    hc_attn: (MGemv, MGemv),
    hc_ffn: (MGemv, MGemv),
    mixer_in: MGemv,
    mixer_out: MGemv,
    moe_in: MGemv,
    sh_down: MGemv,
    ple: Option<MGemv>,
}

/// The second half of a split window: its window buffers (the slots' state shared with the first half's), the
/// launches that write them, and its own MoE job words (the second mailbox's counter).
struct Half {
    s: Scratch,
    groups: Vec<Groups>,
    head_hc: (MGemv, MGemv),
    head_out: MGemv,
    words: DeviceBuf,
}

/// The narrowest window of several sequences that runs split in two halves (`Engine::launch_split`), each half's
/// GPU work overlapping the other's CPU experts (`QX_SPLIT`, default 4; above TMAX: never).
fn split_min() -> usize {
    static U: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_SPLIT").ok().and_then(|s| s.parse().ok()).map_or(4, |v: usize| if v == 0 { usize::MAX } else { v.max(2) }))
}

/// One attention layer's FP8 KV cache for every sequence slot: E4M3 K and V with `SCALE_GROUPS` f32 scales per
/// (cell, kv head), one per 32 dims.
/// A slot's host pages (`KvPages`) hold every cell; its VRAM ring holds the most recent `window` cells at slot
/// `cell % window`.
pub struct Fp8Kv {
    /// Slot s's ring at s * (window * kv * dim) bytes (scales: s * window * kv * SCALE_GROUPS floats).
    pub ring_k: DeviceBuf,
    pub ring_v: DeviceBuf,
    pub ring_ks: DeviceBuf,
    pub ring_vs: DeviceBuf,
    /// Per slot, the eight words the kernels read (kernels/include/qx_kv.cuh): ring K, V, K scales, V scales, the
    /// slot's page table, the layer's K, V and scales offsets within a page.
    pub tab: DeviceBuf,
}

impl Fp8Kv {
    /// `tables(s)` gives slot s's page table address; `offsets` this layer's within a page.
    pub fn new(slots: usize, window: usize, nkv: usize, hd: usize, tables: impl Fn(usize) -> u64, offsets: (u64, u64, u64)) -> Self {
        let mut kv = Self {
            ring_k: DeviceBuf::new(slots * window * nkv * hd),
            ring_v: DeviceBuf::new(slots * window * nkv * hd),
            ring_ks: DeviceBuf::new(slots * window * nkv * 4 * crate::kvpages::SCALE_GROUPS),
            ring_vs: DeviceBuf::new(slots * window * nkv * 4 * crate::kvpages::SCALE_GROUPS),
            tab: DeviceBuf::new(slots * 64),
        };
        let words: Vec<u64> = (0..slots)
            .flat_map(|s| {
                let (r, rs) = ((s * window * nkv * hd) as u64, (s * window * nkv * 4 * crate::kvpages::SCALE_GROUPS) as u64);
                [kv.ring_k.ptr + r, kv.ring_v.ptr + r, kv.ring_ks.ptr + rs, kv.ring_vs.ptr + rs, tables(s), offsets.0, offsets.1, offsets.2]
            })
            .collect();
        kv.tab = DeviceBuf::from_slice(&words);
        kv
    }

    /// Slot s's eight addresses (as in `tab`).
    pub fn slot(&self, s: usize) -> [u64; 8] {
        let w: Vec<u64> = self.tab.download_at(s * 64, 8);
        w.try_into().unwrap()
    }
}

/// Per-layer sequence state: `conv`, `s`, `kr`, `kp`, the FP8 KV per sequence slot (see `Strides`), the rest shared.
enum LayerState {
    /// `qkv`, `cv`, `beta`, `alpha`: the layer's window inputs (TMAX rows each), kept for a rollback;
    /// `s0`, `conv0`: the states before the window (of the one sequence a speculative window runs).
    Gdn { conv: DeviceBuf, s: DeviceBuf, qkv: DeviceBuf, cv: DeviceBuf, beta: DeviceBuf, alpha: DeviceBuf, s0: DeviceBuf, conv0: DeviceBuf },
    /// `k`, `v`: f32 caches in VRAM (QX_KV=f32), or empty with `fp8` set. `kr`: raw indexer keys (a ring of one
    /// block with FP8), `kp`: pooled block keys.
    Attn { k: DeviceBuf, v: DeviceBuf, kr: DeviceBuf, kp: DeviceBuf, fp8: Option<Fp8Kv> },
}

/// Scratch vectors, reused by every layer.
struct Scratch {
    e: DeviceBuf,
    r: DeviceBuf,
    xn: DeviceBuf,
    lo: DeviceBuf,
    gate: DeviceBuf,
    x: DeviceBuf,
    inject: DeviceBuf,
    y: DeviceBuf,
    qkv: DeviceBuf,
    z: DeviceBuf,
    beta: DeviceBuf,
    alpha: DeviceBuf,
    cv: DeviceBuf,
    o: DeviceBuf,
    on: DeviceBuf,
    qg: DeviceBuf,
    q: DeviceBuf,
    ag: DeviceBuf,
    kraw: DeviceBuf,
    knew: DeviceBuf,
    vnew: DeviceBuf,
    kr: DeviceBuf,
    iq: DeviceBuf,
    qi: DeviceBuf,
    scores: DeviceBuf,
    cells: DeviceBuf,
    att: DeviceBuf,
    att_sc: DeviceBuf,
    att_po: DeviceBuf,
    att_pm: DeviceBuf,
    att_ps: DeviceBuf,
    router: DeviceBuf,
    sh_g: DeviceBuf,
    sh_u: DeviceBuf,
    sh_h: DeviceBuf,
    sh_y: DeviceBuf,
    sg: DeviceBuf,
    ids: DeviceBuf,
    slots: DeviceBuf,
    /// The selections grouped by slot (`qx_moe_groups`): count, starts, lengths, members.
    grp: DeviceBuf,
    wts: DeviceBuf,
    gu: DeviceBuf,
    yk: DeviceBuf,
    xq: DeviceBuf,
    xd: DeviceBuf,
    hq: DeviceBuf,
    hd: DeviceBuf,
    ycpu: DeviceBuf,
    logits: DeviceBuf,
    emb: DeviceBuf,
    pkey: DeviceBuf,
    pval: DeviceBuf,
    kn: DeviceBuf,
    qn: DeviceBuf,
    pg: DeviceBuf,
    /// The PLE history of each slot.
    hist: DeviceBuf,
    /// The PLE history before the window and each token's cn, for a rollback.
    hist0: DeviceBuf,
    cn: DeviceBuf,
    /// The window's lanes (u32 per token): positions and tokens (written by qx_win_begin) and sequence slots
    /// (written by the host, see `set_lanes`).
    pos: DeviceBuf,
    tok: DeviceBuf,
    slot: DeviceBuf,
    /// QSA selection: each token's cell count.
    count: DeviceBuf,
    /// The window's samples (u32 x TMAX).
    samples: DeviceBuf,
}

/// Floats per sequence slot in the slot-strided state buffers.
#[derive(Clone, Copy)]
pub struct Strides {
    conv: usize,
    gdn: usize,
    kr: usize,
    kp: usize,
    hist: usize,
}

/// The host's view of a sequence slot other than the current one (the current one's is `Engine::pos`/`tokens`).
#[derive(Clone, Default)]
struct SeqHost {
    pos: usize,
    tokens: Vec<u32>,
}

pub struct Engine<'a> {
    pub cfg: Config,
    gpu: Gpu,
    embd: Mat,
    output: Mat,
    head: HcW,
    layers: Vec<LayerW<'a>>,
    groups: Vec<Groups>,
    head_hc: (MGemv, MGemv),
    head_out: MGemv,
    state: Vec<LayerState>,
    s: Scratch,
    /// The second half of a split window (`launch_split`), and which half the launches act on now (0: `s` and
    /// `groups`).
    half2: Option<Half>,
    half: std::cell::Cell<usize>,
    /// The last window's first half when it ran split (0: it did not), for its samples and logits.
    last_split: std::cell::Cell<usize>,
    // Field order is drop order: the driver thread reads the cache and the host experts, so it goes first.
    driver: Driver,
    pub cache: ExpertCache,
    pub host: HostExperts,
    /// Device step state, a row of STEP_ROW words per slot (kernels/step.cu), and the context tokens by position
    /// (`max_ctx + 1` per slot).
    step: DeviceBuf,
    ctx: DeviceBuf,
    /// Sampling parameters per slot (8 words): temperature, top_p, min_p (f32), top_k (u32), seed (u64).
    sample: DeviceBuf,
    /// The host KV pages and the slots' page tables, and the prefix cache over them.
    pub kv: std::cell::RefCell<crate::kvpages::KvPages>,
    pub prefix: std::cell::RefCell<prefix::PrefixCache>,
    /// Sequence slots, and the one the single-sequence calls (`stream`, `forward`, `prefill`, ...) act on.
    pub slots: usize,
    pub cur: usize,
    seqs: Vec<SeqHost>,
    strides: Strides,
    /// Token masks for the sampler (kernels/step.cu `qx_sample` [7]): a flag per lane, then a bit vector per lane.
    /// The host fills it for a window before queueing it; a window with masked lanes must not be queued before
    /// the previous one has completed.
    pub masks: qx_gpu::mem::HostMapped,
    /// The lane ring the windows take their lanes from (`qx_lanes_take`): LANE_RING entries of TMAX u32, then the
    /// published count of windows run; and the windows queued so far.
    lanes: qx_gpu::mem::HostMapped,
    lane_n: std::cell::Cell<u64>,
    /// Download the routing every layer (a sync each) to fill the counters below.
    pub route_stats: bool,
    /// Routing counts per (layer, expert), and how many selections the VRAM cache served.
    pub route_counts: std::cell::RefCell<Vec<u64>>,
    pub route_hits: std::cell::Cell<u64>,
    pub route_total: std::cell::Cell<u64>,
    /// When set, every routing is appended as (layer, 10 expert ids).
    pub route_trace: std::cell::RefCell<Option<Vec<u16>>>,
    /// Cells of the FP8 KV cache's VRAM ring.
    pub kv_window: usize,
    /// One captured decode step, replayed for every token once recorded.
    graph: Option<cubecl_runtime::client::Graph>,
    /// Captured windows by width (index t), for speculative decoding; and the windows of several sequences of 5 to 8
    /// tokens, whose products run on tensor cores (`ops::TMV_BATCHED`).
    win_graphs: Vec<Option<cubecl_runtime::client::Graph>>,
    batch_graphs: Vec<Option<cubecl_runtime::client::Graph>>,
    /// Captured split windows (`launch_split`) by width.
    split_graphs: Vec<Option<cubecl_runtime::client::Graph>>,
    pub max_ctx: usize,
    pub pos: usize,
    pub tokens: Vec<u32>,
    /// Batched-prefill scratch (`QX_PREFILL_CHUNK` tokens per chunk, 0 disables).
    prefill: Option<PrefillBufs>,
    /// The other tier of prompt buffers when there are two (`QX_PREFILL_BIG`): both are carved from the cache's
    /// tail, and `prefill_tier` swaps the one wanted into `prefill`. `prefill_big`: `prefill` is the big tier.
    prefill_alt: Option<PrefillBufs>,
    prefill_big: bool,
    /// Staging groups the prefill copy thread has finished.
    stage_ready: std::sync::atomic::AtomicU32,
    /// Hand each prefill's routing to the adaptive cache (off for the warm-up's throwaway text).
    pub prefill_feeds_cache: bool,
    /// The MTP drafter for speculative decoding (`QX_MTP`: its GGUF; 0 disables).
    drafter: Option<Drafter>,
    /// Speculative timing (QX_SPEC_PROF): setup, windows, accept and rollback, drafts; window time and count by width.
    spec_prof: std::cell::RefCell<[f64; 24]>,
    /// Decode speculatively in `stream` when a drafter is loaded (default on; `QX_SPEC=0` turns it off).
    pub spec: bool,
    /// The slot whose sequence the drafter's state follows (prefill catch-up runs only for it).
    draft_slot: Option<usize>,
    /// Save a prefix cache entry after every prompt prefill (`prefill_prompt`).
    pub snapshot_prefill: bool,
    /// The prompt buffers live in the expert cache's tail and are lent for each prompt (Strata's borrowing):
    /// whether they do, whether the tail is lent now, and whether a prefill keeps it lent for the caller to give
    /// back (the server, between the chunks of a prompt).
    lend: bool,
    lent: std::cell::Cell<bool>,
    pub keep_lent: bool,
}

/// A speculative sequence between windows: the drafts for the next one, and running window costs by width.
pub struct SpecRun {
    mtp: Vec<u32>,
    cost: [f64; WMAX + 1],
    tau: f64,
    k: usize,
    /// Windows, drafts proposed, drafts kept.
    pub stats: [usize; 3],
}

/// How the next token is chosen.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, seed: 0 }
    }
}

/// How many drafts to verify: the window width with the most expected tokens per second. Draft j (of the
/// probabilities `p`) is kept with probability p_1 .. p_j; `cost[w]` is the measured seconds of a window of
/// width w (0: not yet measured, extrapolated from its neighbours); every width also pays `extra` (the draft
/// pass). Drafts whose chance of all being kept falls below `tau` are not considered.
fn pick_width(p: &[f32], cost: &[f64], extra: f64, tau: f64) -> usize {
    let at = |w: usize| -> f64 {
        if cost[w] > 0.0 {
            return cost[w];
        }
        let lo = (1..w).rev().find(|&i| cost[i] > 0.0);
        let base = lo.map_or(0.012, |i| cost[i]);
        let step = match lo {
            Some(i) if i >= 2 && cost[i - 1] > 0.0 => (cost[i] - cost[i - 1]).max(0.1 * base),
            _ => 0.4 * base,
        };
        base + step * (w - lo.unwrap_or(1)) as f64
    };
    let (mut best, mut best_v, mut e, mut c) = (0usize, 1.0 / (at(1) + extra), 1.0f64, 1.0f64);
    for (j, &pj) in p.iter().enumerate() {
        if j + 2 >= cost.len() {
            break;
        }
        c *= pj as f64;
        if c < tau {
            break;
        }
        e += c;
        let v = e / (at(j + 2) + extra);
        // explore an unmeasured width now and then, so its cost gets measured
        if v > best_v || (cost[j + 2] <= 0.0 && c > 0.5) {
            best = j + 1;
            best_v = v;
        }
    }
    best
}

/// `QX_EXACT_EXPERTS=1`: CPU experts in f32 (slow), so the engine can be compared with the exact reference.
/// NVFP4 resident experts grouped by slot (`QX_MOE_GROUPED=0`: one weight read per selection).
fn moe_grouped() -> bool {
    static X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X.get_or_init(|| !std::env::var("QX_MOE_GROUPED").is_ok_and(|v| v == "0"))
}

fn exact_experts() -> bool {
    static X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X.get_or_init(|| std::env::var("QX_EXACT_EXPERTS").is_ok_and(|v| v == "1"))
}

fn bits(f: f32) -> u64 {
    f.to_bits() as u64
}

fn blocks(n: usize, t: u32) -> (u32, u32, u32) {
    ((n as u32).div_ceil(t), 1, 1)
}

impl<'a> Engine<'a> {
    /// `cache_budget` bytes of VRAM hold routed experts, filled in `cache_order`; the rest run on the CPU.
    /// `load_slots` with `QX_SLOTS` sequence slots (default 1).
    pub fn load(g: &'a Gguf, max_ctx: usize, cache_budget: usize, cache_order: &[(usize, usize)]) -> Result<Self> {
        let slots = std::env::var("QX_SLOTS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        Self::load_slots(g, max_ctx, cache_budget, cache_order, slots)
    }

    /// Loads the model with state for `slots` sequences decoded together (`decode_batch`). Each slot holds a
    /// whole context of `max_ctx` tokens.
    pub fn load_slots(g: &'a Gguf, max_ctx: usize, cache_budget: usize, cache_order: &[(usize, usize)], slots: usize) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let gpu = Gpu::new();
        let t = |n: &str| g.tensor(n).ok_or_else(|| anyhow!("missing tensor {n}"));
        let mat = |n: &str| -> Result<Mat> {
            let w = W::new(g, t(n)?);
            let (bytes, qx, stride) = ops::to_planar(w.ty, w.data, w.rows, w.cols);
            Ok(Mat { buf: DeviceBuf::from_slice(&bytes), ty: w.ty, rows: w.rows, cols: w.cols, qx, stride })
        };
        // The embedding table is read a row at a time by the dequant kernel, in GGUF layout.
        let raw_mat = |n: &str| -> Result<Mat> {
            let w = W::new(g, t(n)?);
            Ok(Mat { buf: DeviceBuf::from_slice(w.data), ty: w.ty, rows: w.rows, cols: w.cols, qx: ops::qx_type(w.ty), stride: w.row_bytes() })
        };
        let vecf = |n: &str| -> Result<DeviceBuf> { Ok(DeviceBuf::from_slice(&W::new(g, t(n)?).to_vec())) };
        let hc = |p: &str, inject: bool| -> Result<HcW> {
            Ok(HcW {
                norm: vecf(&format!("{p}_norm.weight"))?,
                down: mat(&format!("{p}_down.weight"))?,
                up: mat(&format!("{p}_up.weight"))?,
                inject: if inject { Some(mat(&format!("{p}_inject.weight"))?) } else { None },
            })
        };
        let mut layers = Vec::new();
        let mut state = Vec::new();
        let ch = 2 * cfg.ssm_k_heads * cfg.ssm_dim + cfg.ssm_v_heads * cfg.ssm_dim;
        let kv_f32 = std::env::var("QX_KV").is_ok_and(|v| v == "f32");
        let window: usize = std::env::var("QX_KV_WINDOW").ok().and_then(|s| s.parse().ok()).unwrap_or(4096).min(max_ctx).max(cfg.kpool);
        let ns = slots.max(1);
        assert!(ns == 1 || !kv_f32, "sequence slots need the FP8 KV cache");
        let n_attn = (0..cfg.n_layer).filter(|&l| cfg.is_attn(l)).count();
        let extra_pages: usize = std::env::var("QX_KV_EXTRA_PAGES").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
        // QX_KV_GIB caps the pinned page arena below slots x contexts (the server reserves pages at admission)
        let kv_cap = std::env::var("QX_KV_GIB").ok().and_then(|s| s.parse::<f64>().ok()).map(|gib| (gib * 1073741824.0) as usize / crate::kvpages::KvPages::page_bytes_for(n_attn, cfg.n_head_kv, cfg.head_dim));
        let kvp = crate::kvpages::KvPages::new(n_attn, cfg.n_head_kv, cfg.head_dim, ns, max_ctx, if kv_f32 { 0 } else { extra_pages }, kv_cap);
        eprintln!("kv pages: {} of {:.2} MiB ({:.2} GiB pinned), {} per sequence", kvp.pages(), kvp.page_bytes as f64 / 1048576.0, (kvp.pages() * kvp.page_bytes) as f64 / 1073741824.0, kvp.max_pages);
        let mut l_attn = 0;
        let strides = Strides {
            conv: 3 * ch,
            gdn: cfg.ssm_v_heads * cfg.ssm_dim * cfg.ssm_dim,
            kr: IDX_RING * cfg.idx_dim,
            kp: max_ctx.div_ceil(cfg.kpool) * cfg.idx_dim,
            hist: (cfg.ple_conv_k - 1) * cfg.ple_ngram * HCD,
        };
        for l in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{l}.{s}");
            let mixer = if cfg.is_attn(l) {
                let (nkv, hd) = (cfg.n_head_kv, cfg.head_dim);
                let fp8 = if kv_f32 { None } else { Some(Fp8Kv::new(ns, window, nkv, hd, |s| kvp.table(s), kvp.offsets(l_attn))) };
                l_attn += 1;
                let full = if kv_f32 { max_ctx * nkv * hd * 4 } else { 0 };
                state.push(LayerState::Attn {
                    k: DeviceBuf::new(full),
                    v: DeviceBuf::new(full),
                    kr: DeviceBuf::new(if kv_f32 { max_ctx * cfg.idx_dim } else { ns * strides.kr } * 4),
                    kp: DeviceBuf::new(ns * strides.kp * 4),
                    fp8,
                });
                MixerW::Attn(AttnW {
                    q: mat(&b("attn_q.weight"))?,
                    k: mat(&b("attn_k.weight"))?,
                    v: mat(&b("attn_v.weight"))?,
                    o: mat(&b("attn_output.weight"))?,
                    q_norm: vecf(&b("attn_q_norm.weight"))?,
                    k_norm: vecf(&b("attn_k_norm.weight"))?,
                    iq: mat(&b("indexer.q_proj.weight"))?,
                    ik: mat(&b("indexer.k_proj.weight"))?,
                    iq_norm: vecf(&b("indexer.q_norm.weight"))?,
                    ik_norm: vecf(&b("indexer.k_norm.weight"))?,
                })
            } else {
                state.push(LayerState::Gdn {
                    conv: DeviceBuf::new(ns * strides.conv * 4),
                    s: DeviceBuf::new(ns * strides.gdn * 4),
                    qkv: DeviceBuf::new(TMAX * ch * 4),
                    cv: DeviceBuf::new(TMAX * ch * 4),
                    beta: DeviceBuf::new(TMAX * cfg.ssm_v_heads * 4),
                    alpha: DeviceBuf::new(TMAX * cfg.ssm_v_heads * 4),
                    s0: DeviceBuf::new(cfg.ssm_v_heads * cfg.ssm_dim * cfg.ssm_dim * 4),
                    conv0: DeviceBuf::new(3 * ch * 4),
                });
                MixerW::Gdn(GdnW {
                    qkv: mat(&b("attn_qkv.weight"))?,
                    gate: mat(&b("attn_gate.weight"))?,
                    beta: mat(&b("ssm_beta.weight"))?,
                    alpha: mat(&b("ssm_alpha.weight"))?,
                    dt_bias: vecf(&b("ssm_dt.bias"))?,
                    a: vecf(&b("ssm_a"))?,
                    conv: vecf(&b("ssm_conv1d.weight"))?,
                    norm: vecf(&b("ssm_norm.weight"))?,
                    out: mat(&b("ssm_out.weight"))?,
                })
            };
            let ple = if l == cfg.ple_layer {
                Some(PleW {
                    key: mat(&b("ple_key.weight"))?,
                    value: mat(&b("ple_value.weight"))?,
                    norm_key: vecf(&b("ple_norm_key.weight"))?,
                    norm_query: vecf(&b("ple_norm_query.weight"))?,
                    norm_conv: vecf(&b("ple_norm_conv.weight"))?,
                    conv: vecf(&b("ple_conv1d.weight"))?,
                })
            } else {
                None
            };
            layers.push(LayerW {
                hc_attn: hc(&b("hc_attn"), true)?,
                hc_ffn: hc(&b("hc_ffn"), true)?,
                mixer,
                router: mat(&b("ffn_gate_inp.weight"))?,
                experts: LayerExperts::load(g, l, cfg.ff_exp, E)?,
                sh_gate: mat(&b("ffn_gate_shexp.weight"))?,
                sh_up: mat(&b("ffn_up_shexp.weight"))?,
                sh_down: mat(&b("ffn_down_shexp.weight"))?,
                sh_router: mat(&b("ffn_gate_inp_shexp.weight"))?,
                ple,
            });
        }
        let f = |n: usize| DeviceBuf::new(n * 4);
        let ff = cfg.ff_exp;
        // Token-major buffers for a window of up to TMAX tokens: token t's row follows token t-1's.
        let w = TMAX;
        let sel = cfg.idx_top_k + cfg.kpool;
        let splits = sel.div_ceil(32);
        let scratch = |hist: Option<&DeviceBuf>| Scratch {
            e: f(w * E),
            r: f(w * HCD),
            xn: f(w * HCD),
            lo: f(w * 512),
            gate: f(w * HCD),
            x: f(w * E),
            inject: f(w * HC),
            y: f(w * E),
            qkv: f(ch),
            z: f(w * cfg.ssm_v_heads * cfg.ssm_dim),
            beta: f(cfg.ssm_v_heads),
            alpha: f(cfg.ssm_v_heads),
            cv: f(ch),
            o: f(w * cfg.ssm_v_heads * cfg.ssm_dim),
            on: f(w * cfg.ssm_v_heads * cfg.ssm_dim),
            qg: f(w * 2 * cfg.n_head * cfg.head_dim),
            q: f(w * cfg.n_head * cfg.head_dim),
            ag: f(w * cfg.n_head * cfg.head_dim),
            kraw: f(w * cfg.n_head_kv * cfg.head_dim),
            knew: f(w * cfg.n_head_kv * cfg.head_dim),
            vnew: f(w * cfg.n_head_kv * cfg.head_dim),
            kr: f(w * cfg.idx_dim),
            iq: f(w * cfg.idx_heads * cfg.idx_dim),
            qi: f(w * cfg.idx_heads * cfg.idx_dim),
            scores: f(w * max_ctx.div_ceil(cfg.kpool)),
            cells: f((w * sel).max(max_ctx)),
            att: f(w * cfg.n_head * cfg.head_dim),
            att_sc: f(cfg.n_head * sel),
            att_po: f(w * splits * cfg.n_head * cfg.head_dim),
            att_pm: f(w * splits * cfg.n_head),
            att_ps: f(w * splits * cfg.n_head),
            router: f(w * cfg.n_expert),
            sh_g: f(w * ff),
            sh_u: f(w * ff),
            sh_h: f(w * ff),
            sh_y: f(w * E),
            sg: f(w),
            ids: f(w * cfg.n_expert_used),
            slots: f(2 * w * cfg.n_expert_used),
            grp: f(1 + 3 * mailbox::AMAX),
            wts: f(w * cfg.n_expert_used),
            gu: f(w * cfg.n_expert_used * 2 * ff),
            yk: f(w * cfg.n_expert_used * E),
            xq: f(w * E / 4),
            xd: f(w * E / 256),
            hq: f(cfg.n_expert_used * ff / 4),
            hd: f(cfg.n_expert_used * ff / 32),
            ycpu: f(E),
            logits: f(w * cfg.n_vocab),
            emb: f(w * E),
            pkey: f(w * HCD),
            pval: f(w * E),
            kn: f(w * HCD),
            qn: f(w * HCD),
            pg: f(w * HC),
            hist: hist.map_or_else(|| f(ns * strides.hist), |h| DeviceBuf::view(h.ptr, h.bytes)),
            hist0: f(strides.hist),
            cn: f(w * HCD),
            pos: f(w),
            tok: f(w),
            slot: f(w),
            count: f(w),
            samples: f(w),
        };
        let s = scratch(None);
        // the second half of a split window: its own window buffers, the slots' PLE history shared
        let s2 = (ns > 1 && split_min() <= TMAX).then(|| scratch(Some(&s.hist)));
        let le: Vec<LayerExperts> = (0..cfg.n_layer).map(|l| LayerExperts::load(g, l, cfg.ff_exp, E)).collect::<Result<_>>()?;
        let t_host = std::time::Instant::now();
        let pack = qx_pack::pack::PackInfo::of(g).transpose()?;
        let host = HostExperts::build(&le, cfg.n_expert, pack.as_ref())?;
        eprintln!("host experts: {:.1} GiB pinned in {:.1}s", host.bytes() as f64 / 1073741824.0, t_host.elapsed().as_secs_f64());
        // Prefill scratch comes out of the expert cache's budget.
        // at most the PLE ring's positions: a chunk waits for all its embeddings at once (4096 deadlocked)
        let chunk: usize = std::env::var("QX_PREFILL_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(2048).min(mailbox::PLE_CAP);
        let hc_lo = layers[0].hc_attn.down.rows.max(layers[0].hc_ffn.down.rows);
        let stage_bytes = std::env::var("QX_PREFILL_STAGE_MIB").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(384) << 20;
        // the buffers' size (allocated once to measure it, then freed)
        let prefill_bytes = if chunk > 0 { PrefillBufs::new(&cfg, chunk.min(max_ctx), max_ctx, hc_lo, stage_bytes, None).bytes } else { 0 };
        // A second, bigger tier for long prompts: fewer chunks, so fewer passes over the experts the cache does
        // not hold (each chunk streams them over PCIe), but a bigger lent tail. Lent only for a long run.
        let big = std::env::var("QX_PREFILL_BIG").ok().and_then(|s| s.parse().ok()).unwrap_or(8192).min(mailbox::PLE_CAP).min(max_ctx);
        let big_bytes = if chunk > 0 && big > chunk.min(max_ctx) { PrefillBufs::new(&cfg, big, max_ctx, hc_lo, stage_bytes, None).bytes } else { 0 };
        // lent from the cache when the swapper runs (it evicts and restores the tail's experts) and the cache
        // is large enough; otherwise allocated beside it, out of the cache's budget
        let lend = chunk > 0 && std::env::var("QX_ADAPT").map(|v| v != "0").unwrap_or(true) && !std::env::var("QX_PREFILL_LEND").is_ok_and(|v| v == "0") && cache_budget > 2 * prefill_bytes;
        let mut prefill = (chunk > 0 && !lend).then(|| PrefillBufs::new(&cfg, chunk.min(max_ctx), max_ctx, hc_lo, stage_bytes, None));
        let cache_budget = cache_budget.saturating_sub(prefill.as_ref().map_or(0, |p| p.bytes));
        // The drafter, before the cache takes what is left.
        let mtp_path = match std::env::var("QX_MTP") {
            Ok(v) if v == "0" => None,
            Ok(v) => Some(std::path::PathBuf::from(v)),
            // an mtp-*.gguf next to the model, or one directory up (the GGUF shards sit in a quant directory)
            Err(_) => g.shards[0].path.ancestors().skip(1).take(2).filter_map(|d| std::fs::read_dir(d).ok()).find_map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("mtp-") && n.to_string_lossy().ends_with(".gguf")))),
        };
        let drafter = match mtp_path {
            Some(p) => {
                let d = Drafter::load(&p, &cfg, &s, max_ctx, window)?;
                eprintln!("drafter: {} ({:.2} GiB in VRAM)", p.display(), d.bytes as f64 / 1073741824.0);
                Some(d)
            }
            None => None,
        };
        let cache_budget = cache_budget.saturating_sub(drafter.as_ref().map_or(0, |d| d.bytes));
        let spare = std::env::var("QX_SPARE_SLOTS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let dbg_mem = |what: &str| if std::env::var("QX_LOAD_DEBUG").is_ok() { eprintln!("  VRAM free {what}: {:.3} GiB", qx_gpu::mem::mem_info().0 as f64 / 1073741824.0) };
        // everything else that lives in VRAM is allocated before the cache takes what is left
        let embd = raw_mat("token_embd.weight")?;
        dbg_mem("before the cache");
        // At most what is free now less the headroom the runtime takes later (modules, graphs, pools: 0.9 GiB
        // measured after warm-up with one slot; the batch graphs grow with the slots: 1.25 GiB was not enough
        // for 4), whatever the caller's estimate. Every GiB here is 380 NVFP4 experts less in VRAM.
        let headroom_gib = std::env::var("QX_VRAM_HEADROOM_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.5 + 0.15 * (ns - 1) as f64);
        let headroom = (headroom_gib * 1073741824.0) as usize;
        let cache_budget = cache_budget.min(qx_gpu::mem::mem_info().0.saturating_sub(headroom));
        let cache = ExpertCache::build(&host, cfg.n_layer, cfg.n_expert, cache_budget, cache_order, spare);
        dbg_mem("after the cache");
        eprintln!("expert cache: {} of {} experts in {:.2} GiB of VRAM", cache.resident, cfg.n_layer * cfg.n_expert, cache.used as f64 / 1073741824.0);
        // the lent region: the arena's last bytes
        let lend_region = lend.then(|| {
            let end = cache.arena.ptr + cache.used as u64;
            let lo = (end - prefill_bytes as u64) / 256 * 256;
            (lo, (end - lo) as usize)
        });
        let mut prefill_alt = None;
        if let Some((lo, n)) = lend_region {
            prefill = Some(PrefillBufs::new(&cfg, chunk.min(max_ctx), max_ctx, hc_lo, stage_bytes, Some((lo, n))));
            eprintln!("prefill: chunks of {} tokens, {:.2} GiB of buffers lent from the expert cache's tail", chunk.min(max_ctx), n as f64 / 1073741824.0);
            if big_bytes > 0 {
                let end = cache.arena.ptr + cache.used as u64;
                if cache.used > big_bytes + big_bytes / 4 {
                    let lo = (end - big_bytes as u64) / 256 * 256;
                    prefill_alt = Some(PrefillBufs::new(&cfg, big, max_ctx, hc_lo, stage_bytes, Some((lo, (end - lo) as usize))));
                    eprintln!("prefill: chunks of {big} tokens for long prompts, {:.2} GiB lent while they run", (end - lo) as f64 / 1073741824.0);
                }
            }
        } else if let Some(p) = &prefill {
            eprintln!("prefill: chunks of {} tokens, {:.2} GiB of buffers", p.c, p.bytes as f64 / 1073741824.0);
        }
        let seg = |m: &Mat, y: &DeviceBuf| Seg { w: m.buf.ptr, y: y.ptr, rows: m.rows, cols: m.cols, qx: m.qx, stride: m.stride };
        let hc_groups = |h: &HcW, s: &Scratch| -> (MGemv, MGemv) {
            let mut first = vec![seg(&h.down, &s.lo)];
            if let Some(i) = &h.inject {
                first.push(seg(i, &s.inject));
            }
            (MGemv::new(&first), MGemv::new(&[seg(&h.up, &s.gate)]))
        };
        let groups_for = |s: &Scratch| -> Vec<Groups> {
            layers
            .iter()
            .zip(&state)
            .map(|(w, st)| {
                let (mixer_in, mixer_out) = match (&w.mixer, st) {
                    (MixerW::Gdn(g), LayerState::Gdn { qkv, beta, alpha, .. }) => (MGemv::new(&[seg(&g.qkv, qkv), seg(&g.gate, &s.z), seg(&g.beta, beta), seg(&g.alpha, alpha)]), MGemv::new(&[seg(&g.out, &s.y)])),
                    (MixerW::Attn(a), _) => (
                        MGemv::new(&[seg(&a.q, &s.qg), seg(&a.k, &s.kraw), seg(&a.v, &s.vnew), seg(&a.ik, &s.kr), seg(&a.iq, &s.iq)]),
                        MGemv::new(&[seg(&a.o, &s.y)]),
                    ),
                    _ => unreachable!(),
                };
                Groups {
                    hc_attn: hc_groups(&w.hc_attn, &s),
                    hc_ffn: hc_groups(&w.hc_ffn, &s),
                    mixer_in,
                    mixer_out,
                    moe_in: MGemv::new(&[seg(&w.router, &s.router), seg(&w.sh_gate, &s.sh_g), seg(&w.sh_up, &s.sh_u), seg(&w.sh_router, &s.sg)]),
                    sh_down: MGemv::new(&[seg(&w.sh_down, &s.sh_y)]),
                    ple: w.ple.as_ref().map(|p| MGemv::new(&[seg(&p.key, &s.pkey), seg(&p.value, &s.pval)])),
                }
            })
            .collect()
        };
        let groups = groups_for(&s);
        let head = hc("output_hc", false)?;
        let output = mat("output.weight")?;
        let mut drafter = drafter;
        if let Some(d) = drafter.as_mut() {
            d.set_head(&output, &s.logits);
        }
        let head_hc = hc_groups(&head, &s);
        let head_out = MGemv::new(&[seg(&output, &s.logits)]);
        let half2 = s2.map(|s2| Half { groups: groups_for(&s2), head_hc: hc_groups(&head, &s2), head_out: MGemv::new(&[seg(&output, &s2.logits)]), words: DeviceBuf::new(STEP_ROW * 4), s: s2 });
        let ptrs: Vec<LayerPtrs> = le
            .iter()
            .enumerate()
            .map(|(l, x)| LayerPtrs {
                rec0: host.record(l, 0),
                rec_stride: host.layers[l].1,
                rec_bytes: host.record_bytes(l),
                gate_bytes: x.gate_bytes(),
                gate_row_bytes: x.gate.row_bytes(),
                down_row_bytes: x.down.row_bytes(),
                fmt: match x.gate_ty() {
                    GgmlType::IQ3_S => GateFmt::Iq3s,
                    GgmlType::IQ2_S => GateFmt::Iq2s,
                    GgmlType::NVFP4 => GateFmt::Nvfp4,
                    t => panic!("no CPU expert kernels for {t:?}"),
                },
            })
            .collect();
        let adapt = if std::env::var("QX_ADAPT").map(|v| v != "0").unwrap_or(true) {
            let cs = CacheState {
                table: cache.table.ptr,
                host_table: cache.host_table.clone(),
                n_expert: cfg.n_expert,
                free: cache.classes.iter().map(|c| c.free.clone()).collect(),
                slot_bytes: cache.classes.iter().map(|c| c.slot_bytes).collect(),
                layer_class: cache.layer_class.clone(),
            };
            let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
            let sc = SwapConfig { half_life_tokens: env("QX_HALF_LIFE", 256.0), max_swaps_per_token: env("QX_SWAPS", 32.0) as usize, margin: env("QX_MARGIN", 0.2), save: (std::env::var("QX_SAVE_ROUTING").as_deref() != Ok("0")).then(|| crate::routing::profile_path(g)).flatten() };
            Some((cs, sc, cache_order.to_vec()))
        } else {
            None
        };
        let pt = t("per_layer_token_embd.weight")?;
        let shard = &g.shards[pt.shard];
        let ple_src = PleSource {
            cfg: Config::from_gguf(g)?,
            table: match &pack {
                Some(p) => crate::ple_io::PleTable { path: p.path.clone(), offset: p.offset[2], ty: pt.ty, row_bytes: p.ple.row_bytes as usize, rows_per_page: p.ple.rows_per_page as usize },
                None => crate::ple_io::PleTable { path: shard.path.clone(), offset: shard.data_start + pt.offset, ty: pt.ty, row_bytes: pt.ty.bytes(pt.shape[0] as usize), rows_per_page: 0 },
            },
            cache_rows: std::env::var("QX_PLE_CACHE_ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 18),
        };
        // The expert pool's CPUs: `QX_POOL_CPUS` as a count (the first n) or a list (`0-15,16-23`); default the 16
        // physical cores (16-31 are their SMT siblings).
        let cpus: Vec<usize> = match std::env::var("QX_POOL_CPUS") {
            Ok(v) if v.contains(['-', ',']) => v
                .split(',')
                .flat_map(|r| match r.split_once('-') {
                    Some((a, b)) => (a.trim().parse::<usize>().unwrap()..=b.trim().parse::<usize>().unwrap()).collect::<Vec<_>>(),
                    None => vec![r.trim().parse::<usize>().unwrap()],
                })
                .collect(),
            Ok(v) => (0..v.trim().parse::<usize>().unwrap_or(16)).collect(),
            Err(_) => (0..16).collect(),
        };
        let driver = Driver::start(ptrs, cfg.ff_exp, E, ple_src, max_ctx, ns, cpus, exact_experts(), adapt);
        if let Some((lo, n)) = lend_region {
            driver.shared.lend_lo.store(lo, Ordering::Relaxed);
            driver.shared.lend_hi.store(lo + n as u64, Ordering::Relaxed);
        }
        let route_counts = std::cell::RefCell::new(vec![0u64; cfg.n_layer * cfg.n_expert]);
        let masks = qx_gpu::mem::HostMapped::new(TMAX * 4 + TMAX * cfg.n_vocab.div_ceil(32) * 4);
        let prefix_cache = prefix::PrefixCache::new(&cfg, &strides, max_ctx, (std::env::var("QX_PREFIX_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(if ns > 1 { 8.0 } else { 2.0 }) * 1073741824.0) as usize);
        let step = DeviceBuf::new(ns * STEP_ROW * 4);
        let ctx = DeviceBuf::new(ns * (max_ctx + 1) * 4);
        let sample = DeviceBuf::new(ns * 32);
        Ok(Self { cache, host, driver, step, ctx, sample, route_stats: false, route_counts, route_hits: Default::default(), route_total: Default::default(), route_trace: Default::default(), embd, output, head, layers, groups, head_hc, head_out, state, s, half2, half: Default::default(), last_split: Default::default(), split_graphs: (0..=TMAX).map(|_| None).collect(), gpu, cfg, kv_window: window, graph: None, win_graphs: (0..=TMAX).map(|_| None).collect(), batch_graphs: (0..=WMAX).map(|_| None).collect(), max_ctx, pos: 0, tokens: Vec::new(), prefill, prefill_alt, prefill_big: false, stage_ready: Default::default(), prefill_feeds_cache: true, drafter, spec_prof: std::cell::RefCell::new([0.0; 24]), spec: !std::env::var("QX_SPEC").is_ok_and(|v| v == "0"), prefix: std::cell::RefCell::new(prefix_cache), kv: std::cell::RefCell::new(kvp), slots: ns, cur: 0, seqs: vec![SeqHost::default(); ns], strides, masks, lanes: qx_gpu::mem::HostMapped::new(LANE_RING * TMAX * 4 + 64), lane_n: Default::default(), draft_slot: None, snapshot_prefill: false, lend, lent: std::cell::Cell::new(false), keep_lent: false })
    }

    fn k(&self, entry: &'static str, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
        self.gpu.run(LAYER.kernel(entry, CubeDim::new_1d(threads)), grid, info);
    }

    /// RMS norm of `groups` groups of `dim` per token, token rows `xs` (x) and `os` (out) floats apart.
    #[allow(clippy::too_many_arguments)]
    fn rms_groups(&self, x: u64, w: u64, out: u64, groups: usize, dim: usize, w_stride: usize, t: usize, xs: usize, os: usize) {
        self.k("qx_rms_groups", 1024, (groups as u32, t as u32, 1), &[x, w, out, dim as u64, w_stride as u64, bits(EPS), 0, xs as u64, os as u64]);
    }

    /// HC pre-mix of the residual into `s.x` (and `s.inject` when the module has one), for t tokens. `xn_ready`:
    /// the combine before it already wrote the normalised residual for this module's norm.
    fn mix(&self, r: u64, w: &HcW, g: &(MGemv, MGemv), xn_ready: bool, t: usize) {
        let s = self.sc();
        if !xn_ready {
            self.rms_groups(r, w.norm.ptr, s.xn.ptr, HC, E, E, t, HCD, HCD);
        }
        g.0.run_t(&self.gpu, s.xn.ptr, 0, 0, tpad(t), HCD);
        g.1.run_t(&self.gpu, s.lo.ptr, 1, bits(1.0 / HC as f32), tpad(t), w.down.rows);
        self.k("qx_hc_mix_out", 256, (E.div_ceil(256) as u32, t as u32, 1), &[s.xn.ptr, s.gate.ptr, s.x.ptr, E as u64, HC as u64]);
    }

    /// R += y * w(inject), then the normalised residual for the HC module whose norm is `next_norm`.
    fn combine(&self, r: u64, next_norm: &DeviceBuf, t: usize) {
        let s = self.sc();
        self.k("qx_hc_combine_norm", 1024, (HC as u32, t as u32, 1), &[r, s.y.ptr, s.inject.ptr, next_norm.ptr, s.xn.ptr, E as u64, bits(EPS), HC as u64]);
    }

    fn gdn(&self, w: &GdnW, st: &LayerState, g: &Groups, t: usize) {
        let (c, s) = (&self.cfg, self.sc());
        let LayerState::Gdn { conv, s: sst, qkv, cv, beta, alpha, s0, conv0 } = st else { unreachable!() };
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_dim);
        let ch = w.qkv.rows;
        g.mixer_in.run_t(&self.gpu, s.x.ptr, 0, 0, tpad(t), E);
        let st = &self.strides;
        self.k("qx_gdn_conv_l2", d as u32, ((ch / d) as u32, t as u32, 1), &[qkv.ptr, conv.ptr, w.conv.ptr, cv.ptr, ch as u64, d as u64, kh as u64, bits(EPS), conv0.ptr, t as u64, s.slot.ptr, st.conv as u64]);
        self.k("qx_gdn_recur", 128, (vh as u32, 4, t as u32), &[cv.ptr, sst.ptr, s.o.ptr, beta.ptr, alpha.ptr, w.dt_bias.ptr, w.a.ptr, kh as u64, d as u64, t as u64, ch as u64, vh as u64, s0.ptr, 0, s.slot.ptr, st.gdn as u64]);
        self.k("qx_gdn_out", 128, (vh as u32, t as u32, 1), &[s.o.ptr, w.norm.ptr, s.z.ptr, s.on.ptr, d as u64, bits(EPS)]);
        g.mixer_out.run_t(&self.gpu, s.on.ptr, 0, 0, tpad(t), vh * d);
    }

    /// Per-head norm and RoPE at the window's positions; token rows `xt` (x) and `ot` (out) floats apart, and
    /// the output row moves `out_pos_stride` floats per position.
    #[allow(clippy::too_many_arguments)]
    fn norm_rope(&self, x: u64, xs: usize, w: u64, out: u64, os: usize, heads: usize, dim: usize, out_pos_stride: usize, t: usize, xt: usize, ot: usize, pos: u64) {
        let c = &self.cfg;
        self.k("qx_norm_rope", dim as u32, (heads as u32, t as u32, 1), &[x, xs as u64, w, out, os as u64, dim as u64, c.n_rot as u64, pos, bits(c.rope_base), bits(EPS), 1, out_pos_stride as u64, xt as u64, ot as u64]);
    }

    /// Attention for t tokens at positions `pos` (device u32 x t) of sequence slots `slots` (device u32 x t).
    fn attn(&self, w: &AttnW, st: &LayerState, g: &Groups, t: usize, pos: u64, slots: u64) {
        let (c, s) = (&self.cfg, self.sc());
        let LayerState::Attn { k, v, kr, kp, fp8 } = st else { unreachable!() };
        let (nh, nkv, hd, id) = (c.n_head, c.n_head_kv, c.head_dim, c.idx_dim);
        assert!(t == 1 || fp8.is_some(), "windows need the FP8 KV cache");
        let n_top = c.idx_top_k / c.kpool;
        g.mixer_in.run_t(&self.gpu, s.x.ptr, 0, 0, tpad(t), E);
        self.norm_rope(s.qg.ptr, 2 * hd, w.q_norm.ptr, s.q.ptr, hd, nh, hd, 0, t, 2 * nh * hd, nh * hd, pos);
        self.k("qx_take_halves", 256, blocks(t * nh * hd, 256), &[s.qg.ptr, s.ag.ptr, hd as u64, (t * nh * hd) as u64]);
        if let Some(f) = fp8 {
            self.norm_rope(s.kraw.ptr, hd, w.k_norm.ptr, s.knew.ptr, hd, nkv, hd, 0, t, nkv * hd, nkv * hd, pos);
            self.gpu.run(ATTN.kernel("qx_kv_append_fp8", CubeDim::new_1d((hd / 2) as u32)), (nkv as u32, 2, t as u32), &[s.knew.ptr, s.vnew.ptr, f.tab.ptr, slots, 0, 0, 0, 0, 0, 0, pos, self.kv_window as u64, nkv as u64, hd as u64]);
            // indexer: each token's raw key into its slot's ring, a pooled key whenever a block completes
            let st = &self.strides;
            self.k("qx_idx_ring_pool", id as u32, (t as u32, 1, 1), &[s.kr.ptr, pos, w.ik_norm.ptr, kp.ptr, id as u64, c.n_rot as u64, bits(c.rope_base), bits(EPS), c.kpool as u64, kr.ptr, IDX_RING as u64, t as u64, st.kr as u64, st.kp as u64, slots]);
        } else {
            self.norm_rope(s.kraw.ptr, hd, w.k_norm.ptr, k.ptr, hd, nkv, hd, nkv * hd, 1, 0, 0, pos);
            self.k("qx_copy", 256, blocks(nkv * hd, 256), &[s.vnew.ptr, v.ptr, (nkv * hd) as u64, pos, (nkv * hd) as u64]);
            self.k("qx_copy", 256, blocks(id, 256), &[s.kr.ptr, kr.ptr, id as u64, pos, id as u64]);
            self.k("qx_idx_pool", id as u32, (1, 1, 1), &[kr.ptr, pos, w.ik_norm.ptr, kp.ptr, id as u64, c.n_rot as u64, bits(c.rope_base), bits(EPS), c.kpool as u64, 0]);
        }
        self.norm_rope(s.iq.ptr, id, w.iq_norm.ptr, s.qi.ptr, id, c.idx_heads, id, 0, t, c.idx_heads * id, c.idx_heads * id, pos);
        let max_blocks = self.max_ctx.div_ceil(c.kpool);
        let sel = c.idx_top_k + c.kpool;
        self.k("qx_idx_score", 256, ((max_blocks * 32).div_ceil(256) as u32, t as u32, 1), &[s.qi.ptr, kp.ptr, s.scores.ptr, pos, c.idx_heads as u64, id as u64, c.kpool as u64, n_top as u64, max_blocks as u64, slots, self.strides.kp as u64]);
        self.k("qx_qsa_select", 1024, (t as u32, 1, 1), &[s.scores.ptr, pos, c.kpool as u64, n_top as u64, s.cells.ptr, s.count.ptr, max_blocks as u64, sel as u64]);
        let splits = sel.div_ceil(32);
        if let Some(f) = fp8 {
            self.gpu.run(
                ATTN.kernel("qxp_attn_split_fp8", CubeDim::new_1d(256)),
                (splits as u32, nkv as u32, t as u32),
                &[s.q.ptr, 0, 0, s.cells.ptr, s.count.ptr, s.att_po.ptr, s.att_pm.ptr, s.att_ps.ptr, nkv as u64, nh as u64, hd as u64, bits(1.0 / (hd as f32).sqrt()), f.tab.ptr, slots, 0, 0, 0, 0, 0, 0, pos, self.kv_window as u64, sel as u64, 0],
            );
            self.gpu.run(ATTN.kernel("qxp_attn_merge", CubeDim::new_1d(hd as u32)), (nh as u32, t as u32, 1), &[s.att_po.ptr, s.att_pm.ptr, s.att_ps.ptr, s.count.ptr, s.att.ptr, nh as u64, hd as u64, splits as u64]);
        } else {
            self.gpu.run(ATTN.kernel("qx_attn_split", CubeDim::new_1d(256)), (splits as u32, nkv as u32, 1), &[s.q.ptr, k.ptr, v.ptr, s.cells.ptr, s.count.ptr, s.att_po.ptr, s.att_pm.ptr, s.att_ps.ptr, nkv as u64, nh as u64, hd as u64, bits(1.0 / (hd as f32).sqrt())]);
            self.gpu.run(ATTN.kernel("qx_attn_merge", CubeDim::new_1d(hd as u32)), (nh as u32, 1, 1), &[s.att_po.ptr, s.att_pm.ptr, s.att_ps.ptr, s.count.ptr, s.att.ptr, nh as u64, hd as u64]);
        }
        self.k("qx_mul_sigmoid", 256, blocks(t * nh * hd, 256), &[s.att.ptr, s.ag.ptr, (t * nh * hd) as u64]);
        g.mixer_out.run_t(&self.gpu, s.att.ptr, 0, 0, tpad(t), nh * hd);
    }

    /// The MoE block up to the CPU's answer: routing, the misses posted to the CPU, the resident and shared
    /// experts. `moe_back` collects the answer.
    fn moe_front(&self, w: &LayerW, l: usize, t: usize) {
        let (c, s) = (&self.cfg, self.sc());
        let (ff, k) = (c.ff_exp, c.n_expert_used);
        let g = self.grp(l);
        let a = (t * k) as u32;
        g.moe_in.run_t(&self.gpu, s.x.ptr, 0, 0, tpad(t), E);
        self.gpu.run(MOE.kernel("qx_moe_route", CubeDim::new_1d(32)), (t as u32, 1, 1), &[s.router.ptr, c.n_expert as u64, k as u64, s.ids.ptr, s.wts.ptr]);
        let le = &w.experts;
        let tab = self.cache.table.ptr;
        let base = (l * c.n_expert) as u64;
        let (mb, words) = self.mailbox();
        // Hand the misses to the CPU first, so the pool works while the GPU computes the resident experts.
        self.gpu.run(STEP.kernel("qx_moe_post", CubeDim::new_1d(256)), (t as u32, 1, 1), &[mb.dptr, tab, base, s.ids.ptr, s.wts.ptr, s.x.ptr, words, l as u64, k as u64, E as u64, s.slots.ptr, s.xq.ptr, s.xd.ptr, t as u64]);
        let gb = le.gate_bytes() as u64;
        // the scale tail (NVFP4) sits after the down rows
        let tail = 2 * gb + le.down_bytes() as u64;
        if exact_experts() {
            let (gu_entry, down_entry) = match le.gate_ty() {
                GgmlType::IQ2_S => ("qx_moe_gateup_iq2_s", "qx_moe_down_iq4_nl"),
                GgmlType::IQ3_S => ("qx_moe_gateup_iq3_s", "qx_moe_down_iq4_nl"),
                GgmlType::NVFP4 => ("qx_moe_gateup_nvfp4", "qx_moe_down_nvfp4"),
                t => panic!("no expert kernel for {t:?}"),
            };
            self.gpu.run(MOE.kernel(gu_entry, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), a, 1), &[s.slots.ptr, k as u64, 0, s.x.ptr, s.gu.ptr, ff as u64, E as u64, le.gate.row_bytes() as u64, tail]);
            self.gpu.run(MOE.kernel(down_entry, CubeDim::new_1d(256)), ((E as u32).div_ceil(8), a, 1), &[s.slots.ptr, 0, 0, 0, s.gu.ptr, s.yk.ptr, ff as u64, E as u64, 2 * gb, le.down.row_bytes() as u64, tail]);
        } else {
            let (gu_entry, down_entry) = match le.gate_ty() {
                GgmlType::IQ2_S => ("qx_moe_gateup_q8k_iq2_s", "qx_moe_down_q80_iq4_nl"),
                GgmlType::IQ3_S => ("qx_moe_gateup_q8k_iq3_s", "qx_moe_down_q80_iq4_nl"),
                GgmlType::NVFP4 => ("qx_moe_gateup_q8k_nvfp4", "qx_moe_down_q80_nvfp4"),
                t => panic!("no expert kernel for {t:?}"),
            };
            if le.gate_ty() == GgmlType::NVFP4 && moe_grouped() {
                // each resident expert's weights read once for all the window's tokens that chose it
                self.gpu.run(MOE.kernel("qx_moe_groups", CubeDim::new_1d(512)), (1, 1, 1), &[s.slots.ptr, a as u64, s.grp.ptr]);
                self.gpu.run(MOE.kernel("qx_moe_gateup_q8k_nvfp4_g", CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), a, 1), &[s.slots.ptr, s.xq.ptr, s.xd.ptr, k as u64, s.gu.ptr, ff as u64, E as u64, le.gate.row_bytes() as u64, tail, s.grp.ptr]);
                self.gpu.run(MOE.kernel("qx_moe_down_q80_nvfp4_g", CubeDim::new_1d(256)), ((E as u32).div_ceil(8 * 2), a, 1), &[s.slots.ptr, s.gu.ptr, 0, 0, s.yk.ptr, ff as u64, E as u64, 2 * gb, le.down.row_bytes() as u64, tail, s.grp.ptr]);
            } else {
                self.gpu.run(MOE.kernel(gu_entry, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), a, 1), &[s.slots.ptr, s.xq.ptr, s.xd.ptr, k as u64, s.gu.ptr, ff as u64, E as u64, le.gate.row_bytes() as u64, tail]);
                self.gpu.run(MOE.kernel(down_entry, CubeDim::new_1d(256)), ((E as u32).div_ceil(8 * 2), a, 1), &[s.slots.ptr, s.gu.ptr, 0, 0, s.yk.ptr, ff as u64, E as u64, 2 * gb, le.down.row_bytes() as u64, tail]);
            }
        }
        g.sh_down.run_t(&self.gpu, s.sh_g.ptr, 2, s.sh_u.ptr, tpad(t), ff);
    }

    /// Waits for the CPU's answer to `moe_front`'s job and combines every expert's output into `s.y`.
    fn moe_back(&self, l: usize, t: usize) {
        let (c, s) = (&self.cfg, self.sc());
        let k = c.n_expert_used;
        let (mb, words) = self.mailbox();
        self.gpu.run(STEP.kernel("qx_moe_wait", CubeDim::new_1d(1)), (1, 1, 1), &[mb.dptr, words]);
        self.gpu.run(MOE.kernel("qx_moe_combine", CubeDim::new_1d(256)), (E.div_ceil(256) as u32, t as u32, 1), &[s.yk.ptr, k as u64, mb.dptr + mailbox::mb_y(E) as u64, s.y.ptr, E as u64, s.slots.ptr, s.wts.ptr, s.sh_y.ptr, s.sg.ptr]);
        if self.route_stats {
            self.gpu.sync();
            let ids: Vec<i32> = s.ids.download(t * k);
            let slots: Vec<u64> = s.slots.download(t * k);
            let hits = slots.iter().filter(|&&p| p != 0).count();
            let mut rc = self.route_counts.borrow_mut();
            for &id in &ids {
                rc[l * c.n_expert + id as usize] += 1;
            }
            self.route_total.set(self.route_total.get() + (t * k) as u64);
            self.route_hits.set(self.route_hits.get() + hits as u64);
            if let Some(tr) = self.route_trace.borrow_mut().as_mut() {
                for tk in 0..t {
                    tr.push(l as u16);
                    tr.extend(ids[tk * k..(tk + 1) * k].iter().map(|&i| i as u16));
                }
            }
        }
    }

    /// The scratch, groups and MoE mailbox (and its job words) of the half the launches act on.
    fn sc(&self) -> &Scratch {
        match self.half.get() {
            0 => &self.s,
            _ => &self.half2.as_ref().unwrap().s,
        }
    }

    fn grp(&self, l: usize) -> &Groups {
        match self.half.get() {
            0 => &self.groups[l],
            _ => &self.half2.as_ref().unwrap().groups[l],
        }
    }

    fn head_groups(&self) -> (&(MGemv, MGemv), &MGemv) {
        match self.half.get() {
            0 => (&self.head_hc, &self.head_out),
            _ => {
                let h = self.half2.as_ref().unwrap();
                (&h.head_hc, &h.head_out)
            }
        }
    }

    fn mailbox(&self) -> (&qx_gpu::mem::HostMapped, u64) {
        match self.half.get() {
            0 => (&self.driver.shared.moe, self.step.ptr),
            _ => (&self.driver.shared.moe2, self.half2.as_ref().unwrap().words.ptr),
        }
    }

    fn ple(&self, w: &PleW, g: &MGemv, t: usize) {
        let s = self.sc();
        let sh = &self.driver.shared;
        self.gpu.run(STEP.kernel("qx_ple_wait", CubeDim::new_1d(256)), (1, 1, 1), &[sh.ple.dptr, s.pos.ptr, s.emb.ptr, E as u64, mailbox::PLE_CAP as u64, t as u64, s.slot.ptr, sh.ple_stride as u64]);
        g.run_t(&self.gpu, s.emb.ptr, 0, 0, tpad(t), E);
        self.rms_groups(s.pkey.ptr, w.norm_key.ptr, s.kn.ptr, HC, E, E, t, HCD, HCD);
        self.rms_groups(s.r.ptr, w.norm_query.ptr, s.qn.ptr, HC, E, E, t, HCD, HCD);
        self.k("qx_ple_gate", 256, (HC as u32, t as u32, 1), &[s.kn.ptr, s.qn.ptr, s.pg.ptr, E as u64]);
        self.k("qx_ple_apply", 256, (HC as u32, t as u32, 1), &[s.pval.ptr, s.pg.ptr, w.norm_conv.ptr, w.conv.ptr, s.hist.ptr, s.r.ptr, E as u64, bits(EPS), 0, t as u64, s.hist0.ptr, s.cn.ptr, s.slot.ptr, self.strides.hist as u64]);
    }

    /// PLE rows asked for, served from the row cache, and read from the drive.
    pub fn ple_stats(&self) -> (u64, u64, u64) {
        let s = &self.driver.ple_stats;
        (s.rows.load(Ordering::Relaxed), s.cache_hits.load(Ordering::Relaxed), s.reads.load(Ordering::Relaxed))
    }

    /// What the engine shares with its CPU threads (counters included).
    pub fn shared(&self) -> &mailbox::Shared {
        &self.driver.shared
    }

    /// Hits and routings the driver has counted, and swaps it issued.
    pub fn cache_stats(&self) -> (u64, u64, u64) {
        let sh = &self.driver.shared;
        (sh.hits.load(Ordering::Relaxed), sh.routed.load(Ordering::Relaxed), sh.swaps.load(Ordering::Relaxed))
    }

    /// Clears the current slot's sequence state. The MoE job counter keeps counting, so it stays in step with the
    /// driver.
    pub fn reset(&mut self) {
        self.gpu.sync();
        let (c, st) = (self.cur, self.strides);
        for ls in &self.state {
            match ls {
                LayerState::Gdn { conv, s, .. } => {
                    conv.zero_range(c * st.conv * 4, st.conv * 4);
                    s.zero_range(c * st.gdn * 4, st.gdn * 4);
                }
                LayerState::Attn { k, v, kr, kp, fp8 } => {
                    if fp8.is_none() {
                        k.zero();
                        v.zero();
                        kr.zero();
                        kp.zero();
                    } else {
                        kr.zero_range(c * st.kr * 4, st.kr * 4);
                        kp.zero_range(c * st.kp * 4, st.kp * 4);
                    }
                }
            }
        }
        self.s.hist.zero_range(c * st.hist * 4, st.hist * 4);
        {
            let mut kv = self.kv.borrow_mut();
            for p in std::mem::take(&mut kv.slot_pages[c]) {
                kv.release(p);
            }
            let (mp, tables) = (kv.max_pages, kv.tables.ptr);
            let _ = tables;
            kv.tables.zero_range(c * mp * 8, mp * 8);
        }
        if let Some(d) = &self.drafter {
            d.reset();
        }
        self.step.upload_at(c * STEP_ROW * 4, &[0u32; 4]);
        let sh = &self.driver.shared;
        let q = &sh.seqs[c];
        q.paused.store(true, Ordering::SeqCst);
        q.epoch.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(1));
        q.forced.lock().unwrap().clear();
        q.n_forced.store(0, Ordering::SeqCst);
        q.allow_sampled.store(false, Ordering::SeqCst);
        q.ple_rewind.store(0, Ordering::SeqCst);
        sh.count(c).store(0, Ordering::SeqCst);
        sh.ple.atomic_u64(sh.ple_at(c) + mailbox::PLE_CONSUMED).store(0, Ordering::SeqCst);
        sh.ple.atomic_u64(sh.ple_at(c) + mailbox::PLE_READY).store(0, Ordering::SeqCst);
        q.ple_next.store(0, Ordering::SeqCst);
        q.paused.store(false, Ordering::SeqCst);
        self.pos = 0;
        self.tokens.clear();
    }

    /// What the host shares with the PLE thread about the current slot.
    fn seq(&self) -> &mailbox::SeqShared {
        &self.driver.shared.seqs[self.cur]
    }

    /// The current slot's step words.
    fn step_words(&self) -> Vec<u32> {
        self.step.download_at(self.cur * STEP_ROW * 4, STEP_ROW)
    }

    fn set_step_words(&self, st: &[u32]) {
        self.step.upload_at(self.cur * STEP_ROW * 4, st);
    }

    /// Byte offset of slot `slot`'s context array.
    fn ctx_at(&self, slot: usize) -> usize {
        slot * (self.max_ctx + 1) * 4
    }

    /// Makes `slot` the one the single-sequence calls act on, keeping the host view of the one before.
    pub fn select(&mut self, slot: usize) {
        assert!(slot < self.slots);
        if slot == self.cur {
            return;
        }
        let old = SeqHost { pos: self.pos, tokens: std::mem::take(&mut self.tokens) };
        self.seqs[self.cur] = old;
        let new = std::mem::take(&mut self.seqs[slot]);
        self.pos = new.pos;
        self.tokens = new.tokens;
        self.cur = slot;
    }

    /// Queues the lanes (one sequence slot per token, the rest of the TMAX entries unused) of the next window to
    /// run: exactly one call per window executed (a graph recording is not one). Waits while the GPU is LANE_RING
    /// windows behind.
    fn queue_lanes(&self, lanes: &[u32]) {
        let n = self.lane_n.get();
        let published = self.lanes.atomic_u64(LANE_RING * TMAX * 4);
        while published.load(Ordering::Acquire) + LANE_RING as u64 <= n {
            std::hint::spin_loop();
        }
        self.lanes.slice::<u32>((n as usize % LANE_RING) * TMAX * 4, TMAX).copy_from_slice(lanes);
        self.lane_n.set(n + 1);
    }

    /// Lanes for a window of the current slot alone.
    fn lanes_single(&self) {
        self.queue_lanes(&[self.cur as u32; TMAX]);
    }

    /// Gives `slot` host KV pages for cells below `end`, their table entries written by queued kernels (so the
    /// windows queued after them find them). Panics when the arena is out of pages.
    pub fn touch_pages(&self, slot: usize, end: usize) {
        let mut kv = self.kv.borrow_mut();
        let need = end.min(self.max_ctx).div_ceil(crate::kvpages::PAGE);
        while kv.slot_pages[slot].len() < need {
            let i = kv.slot_pages[slot].len();
            let p = match kv.alloc() {
                Some(p) => p,
                None => {
                    drop(kv);
                    assert!(self.prefix_evict_one(), "out of KV pages");
                    kv = self.kv.borrow_mut();
                    kv.alloc().expect("out of KV pages")
                }
            };
            kv.slot_pages[slot].push(p);
            self.k("qx_set_u64", 1, (1, 1, 1), &[kv.table(slot) + (i * 8) as u64, kv.addr(p)]);
        }
    }

    /// KV pages a sequence of `len` tokens occupies (with room for a speculative window past its end).
    pub fn kv_pages_for(&self, len: usize) -> usize {
        (len.min(self.max_ctx) + WMAX).div_ceil(crate::kvpages::PAGE)
    }

    /// KV page accounting for admission: (free pages, pages the prefix cache alone holds and can give up, total).
    pub fn kv_pages(&self) -> (usize, usize, usize) {
        let kv = self.kv.borrow();
        (kv.free_pages(), self.prefix_evictable_pages(&kv), kv.pages())
    }

    /// Pages slot `slot` holds now.
    pub fn kv_slot_pages(&self, slot: usize) -> usize {
        self.kv.borrow().slot_pages[slot].len()
    }

    /// Sampling for the following steps; temperature 0 is greedy.
    pub fn set_sampling(&mut self, p: &Sampling) {
        self.gpu.sync();
        let mut w = [0u32; 8];
        w[0] = p.temperature.to_bits();
        w[1] = p.top_p.to_bits();
        w[2] = p.min_p.to_bits();
        w[3] = p.top_k;
        w[4] = p.seed as u32;
        w[5] = (p.seed >> 32) as u32;
        self.sample.upload_at(self.cur * 32, &w);
    }

    /// Forces `tokens` as the context from position 0, beyond what is already there.
    pub fn set_forced(&mut self, tokens: &[u32]) {
        assert!(tokens.len() <= self.max_ctx);
        self.gpu.sync();
        self.ctx.upload_at(self.ctx_at(self.cur), tokens);
        let mut st = self.step_words();
        st[3] = tokens.len() as u32;
        self.set_step_words(&st);
        let sh = &self.driver.shared;
        *sh.seqs[self.cur].forced.lock().unwrap() = tokens.to_vec();
        sh.seqs[self.cur].n_forced.store(tokens.len() as u32, Ordering::Release);
    }

    /// Queues one whole step on the GPU; nothing waits on the host.
    fn launch_step(&self, debug: Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        self.launch_window(1, debug);
    }

    /// Queues a window of t tokens at positions pos .. pos + t - 1 (the step state's pos): every token through
    /// every layer, its logits (rows of `s.logits`) and its sample. Each token's logits are bit for bit those of
    /// one-token steps. Advances pos by t.
    fn launch_window(&self, t: usize, mut debug: Option<&mut dyn FnMut(&str, Vec<f32>)>) {
        assert!((1..=TMAX).contains(&t));
        let mut dump = |name: String, b: &DeviceBuf, n: usize| {
            if let Some(d) = debug.as_mut() {
                self.gpu.sync();
                d(&name, b.download(n));
            }
        };
        self.take_lanes();
        self.win_start(t);
        for l in 0..self.layers.len() {
            self.layer_front(l, t, &mut dump);
            self.layer_back(l, t, &mut dump);
        }
        self.win_finish(t, 0, &mut dump);
    }

    /// A window of several sequences split in two halves, the first `ta` tokens and the other `tb` (whole
    /// segments each), interleaved layer by layer so each half's GPU work runs while the CPU computes the
    /// other's missing experts: half A's front of layer l (its misses posted), half B's back of layer l - 1 and
    /// front of l, then A's back of l. Each half has its own buffers and MoE mailbox.
    fn launch_split(&self, ta: usize, tb: usize) {
        assert!(ta >= 1 && tb >= 1 && ta + tb <= TMAX && self.half2.is_some());
        let mut none = |_: String, _: &DeviceBuf, _: usize| {};
        let (a, b) = (0, 1);
        self.take_lanes();
        let sb = &self.half2.as_ref().unwrap().s;
        self.gpu.run(STEP.kernel("qx_lanes_split", CubeDim::new_1d(1)), (1, 1, 1), &[self.s.slot.ptr, sb.slot.ptr, ta as u64, TMAX as u64]);
        self.half.set(a);
        self.win_start(ta);
        self.half.set(b);
        self.win_start(tb);
        let nl = self.layers.len();
        for l in 0..nl {
            self.half.set(a);
            self.layer_front(l, ta, &mut none);
            if l > 0 {
                self.half.set(b);
                self.layer_back(l - 1, tb, &mut none);
            }
            self.half.set(b);
            self.layer_front(l, tb, &mut none);
            self.half.set(a);
            self.layer_back(l, ta, &mut none);
        }
        self.win_finish(ta, 0, &mut none);
        self.half.set(b);
        self.layer_back(nl - 1, tb, &mut none);
        self.win_finish(tb, ta, &mut none);
        self.half.set(a);
    }

    /// The window's lanes into the first half's `slot` (one ring entry per window).
    fn take_lanes(&self) {
        self.gpu.run(STEP.kernel("qx_lanes_take", CubeDim::new_1d(1)), (1, 1, 1), &[self.lanes.dptr, self.s.slot.ptr, LANE_RING as u64, TMAX as u64, self.step.ptr + 9 * 4, self.lanes.dptr + (LANE_RING * TMAX * 4) as u64]);
    }

    /// The current half's tokens and positions, their embeddings and the residual streams.
    fn win_start(&self, t: usize) {
        let s = self.sc();
        let ctx_stride = (self.max_ctx + 1) as u64;
        self.gpu.run(STEP.kernel("qx_win_begin", CubeDim::new_1d(1)), (1, 1, 1), &[self.step.ptr, self.ctx.ptr, s.tok.ptr, s.pos.ptr, t as u64, s.slot.ptr, ctx_stride]);
        ops::dequant_gather(&self.gpu, self.embd.ty, self.embd.buf.ptr, s.e.ptr, t, self.embd.cols, s.tok.ptr);
        self.k("qx_hc_init", 256, (HCD.div_ceil(256) as u32, t as u32, 1), &[s.r.ptr, s.e.ptr, E as u64, HC as u64]);
    }

    /// Layer l up to the CPU's experts: PLE, the mixer, and the MoE block's front.
    fn layer_front(&self, l: usize, t: usize, dump: &mut dyn FnMut(String, &DeviceBuf, usize)) {
        let (s, w, g) = (self.sc(), &self.layers[l], self.grp(l));
        if let (Some(p), Some(pg)) = (&w.ple, &g.ple) {
            self.ple(p, pg, t);
        }
        // The previous layer's combine normalised the residual for this mix, unless PLE changed it since.
        self.mix(s.r.ptr, &w.hc_attn, &g.hc_attn, l > 0 && w.ple.is_none(), t);
        dump(format!("hc_mixed-{l}"), &s.x, E);
        match &w.mixer {
            MixerW::Gdn(gw) => self.gdn(gw, &self.state[l], g, t),
            MixerW::Attn(a) => self.attn(a, &self.state[l], g, t, s.pos.ptr, s.slot.ptr),
        }
        dump(format!("{}-{l}", if self.cfg.is_attn(l) { "attn_output" } else { "linear_attn_out" }), &s.y, E);
        self.combine(s.r.ptr, &w.hc_ffn.norm, t);
        self.mix(s.r.ptr, &w.hc_ffn, &g.hc_ffn, true, t);
        dump(format!("hc_mixed-{l}"), &s.x, E);
        self.moe_front(w, l, t);
    }

    /// Layer l from the CPU's experts on: the MoE output into the residual.
    fn layer_back(&self, l: usize, t: usize, dump: &mut dyn FnMut(String, &DeviceBuf, usize)) {
        let s = self.sc();
        self.moe_back(l, t);
        dump(format!("ffn_out-{l}"), &s.y, E);
        let next_norm = if l + 1 < self.layers.len() { &self.layers[l + 1].hc_attn.norm } else { &self.head.norm };
        self.combine(s.r.ptr, next_norm, t);
        dump(format!("l_last-{l}"), &s.r, HCD);
    }

    /// The head, the samples (mask lanes from `lane0`) and the window's end for the current half.
    fn win_finish(&self, t: usize, lane0: usize, dump: &mut dyn FnMut(String, &DeviceBuf, usize)) {
        let s = self.sc();
        let sh = &self.driver.shared;
        let ctx_stride = (self.max_ctx + 1) as u64;
        let (head_hc, head_out) = self.head_groups();
        self.mix(s.r.ptr, &self.head, head_hc, true, t);
        dump("result_norm".into(), &s.x, E);
        head_out.run_t(&self.gpu, s.x.ptr, 0, 0, tpad(t), E);
        self.gpu.run(STEP.kernel("qx_sample", CubeDim::new_1d(1024)), (t as u32, 1, 1), &[s.logits.ptr, self.cfg.n_vocab as u64, 0, self.sample.ptr, s.samples.ptr, s.pos.ptr, s.slot.ptr, self.masks.dptr, lane0 as u64]);
        self.gpu.run(STEP.kernel("qx_win_end", CubeDim::new_1d(1)), (1, 1, 1), &[self.step.ptr, self.ctx.ptr, sh.out.dptr, sh.out_count_off as u64, s.samples.ptr, t as u64, s.slot.ptr, ctx_stride, sh.out_stride as u64]);
    }

    /// One forced token; returns the logits. `debug`, when given, receives named activations (each one syncs).
    pub fn forward(&mut self, tok: u32, debug: Option<&mut dyn FnMut(&str, Vec<f32>)>) -> Vec<f32> {
        self.touch_pages(self.cur, self.pos + 1);
        self.lanes_single();
        assert!(self.pos < self.max_ctx, "context full");
        self.tokens.push(tok);
        let toks = self.tokens.clone();
        self.seq().allow_sampled.store(false, Ordering::Release);
        self.set_forced(&toks);
        self.launch_step(debug);
        self.pos += 1;
        self.gpu.sync();
        self.s.logits.download(self.cfg.n_vocab)
    }

    /// Forced tokens as one window (at most WMAX); returns each token's logits.
    pub fn forward_window(&mut self, toks: &[u32]) -> Vec<Vec<f32>> {
        self.touch_pages(self.cur, self.pos + toks.len());
        self.lanes_single();
        let t = toks.len();
        assert!((1..=WMAX).contains(&t) && self.pos + t <= self.max_ctx);
        self.tokens.extend_from_slice(toks);
        let all = self.tokens.clone();
        self.seq().allow_sampled.store(false, Ordering::Release);
        self.set_forced(&all);
        self.launch_window(t, None);
        self.pos += t;
        self.gpu.sync();
        let n = self.cfg.n_vocab;
        let flat: Vec<f32> = self.s.logits.download(t * n);
        flat.chunks(n).map(|c| c.to_vec()).collect()
    }

    /// Queues one step: a replay of the captured graph when there is one (`QX_GRAPH=0` disables graphs).
    /// The first call without a graph runs the step eagerly as the warm-up the capture needs, then records the
    /// next step; that recording is not executed, so it is replayed at once.
    fn queue_step(&mut self) {
        self.lanes_single();
        if std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") {
            self.launch_step(None);
            return;
        }
        if let Some(g) = &self.graph {
            // SAFETY: every buffer the graph reads or writes is owned by `self` and outlives it.
            unsafe { g.replay().expect("graph replay") };
            return;
        }
        self.gpu.client.graph_prepare().expect("graph_prepare");
        self.launch_step(None);
        self.gpu.client.start_capture().expect("start_capture");
        self.launch_step(None);
        let g = self.gpu.client.stop_capture().expect("stop_capture");
        self.lanes_single();
        unsafe { g.replay().expect("graph replay") };
        self.graph = Some(g);
    }

    /// Runs `prompt` then `n_new` greedy tokens, all queued back to back; returns the generated tokens.
    pub fn generate(&mut self, prompt: &[u32], n_new: usize) -> Vec<u32> {
        self.generate_timed(prompt, n_new).0
    }

    /// `generate`, also returning when the prompt's last step and the last step completed, measured from the
    /// call by polling the mapped step count.
    pub fn generate_timed(&mut self, prompt: &[u32], n_new: usize) -> (Vec<u32>, f64, f64) {
        let t0 = std::time::Instant::now();
        assert!(self.pos == 0, "generate starts from an empty sequence");
        assert!(prompt.len() + n_new <= self.max_ctx);
        self.set_forced(prompt);
        self.touch_pages(self.cur, prompt.len() + n_new);
        self.seq().allow_sampled.store(true, Ordering::Release);
        let steps = prompt.len() + n_new - 1;
        let mut queued = self.prefill_prompt(prompt);
        while queued < steps {
            if self.graph.is_none() && steps - queued < 2 {
                self.lanes_single();
                self.launch_step(None);
                queued += 1;
                continue;
            }
            let had_graph = self.graph.is_some();
            self.queue_step();
            // a first capture runs two steps: the eager warm-up and the replayed recording
            queued += if had_graph || std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") { 1 } else { 2 };
        }
        let (mut t_prompt, t_end);
        {
            let sh = &self.driver.shared;
            let count = sh.count(self.cur);
            let want_prompt = (prompt.len() - 1) as u64;
            t_prompt = 0.0;
            loop {
                let c = count.load(Ordering::Acquire);
                if t_prompt == 0.0 && c >= want_prompt {
                    t_prompt = t0.elapsed().as_secs_f64();
                }
                if c >= steps as u64 {
                    t_end = t0.elapsed().as_secs_f64();
                    break;
                }
                std::hint::spin_loop();
            }
        }
        self.gpu.sync();
        let sh = &self.driver.shared;
        let out: Vec<u32> = sh.outs(self.cur, steps).to_vec();
        self.pos = steps;
        self.tokens = prompt.to_vec();
        self.tokens.extend_from_slice(&out[prompt.len() - 1..]);
        (out[prompt.len() - 1..].to_vec(), t_prompt, t_end)
    }

    /// Compiles and loads every kernel of the prefill and decode paths (and captures the step graph) on a
    /// throwaway sequence, so the first real request does not pay for it.
    pub fn warm_up(&mut self) {
        let n = 160.min(self.max_ctx / 2);
        let toks: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
        self.prefill_feeds_cache = false;
        let _ = self.generate(&toks, 4);
        self.reset();
        self.capture_windows();
        if self.drafter.is_some() && !std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") {
            self.capture_drafts();
            self.reset();
        }
        self.prefill_feeds_cache = true;
        if std::env::var("QX_LOAD_DEBUG").is_ok() {
            eprintln!("  VRAM free after warm-up: {:.3} GiB", qx_gpu::mem::mem_info().0 as f64 / 1073741824.0);
        }
    }

    /// Captures one window graph per width on a throwaway sequence (each capture needs an executed warm-up run
    /// first), then clears the sequence. `QX_GRAPH=0` skips it; windows then launch eagerly.
    fn capture_windows(&mut self) {
        if std::env::var("QX_GRAPH").is_ok_and(|v| v == "0") {
            return;
        }
        let wmax = self.max_window();
        // forced tokens for the windows run before their captures (the recordings are not run); the throwaway
        // sequence restarts when the context would overflow
        // (width, kind): 0 one sequence's window, 1 several sequences' (5 to 8 tokens), 2 split in halves
        let split = if self.half2.is_some() { split_min()..=self.slots.min(TMAX) } else { 1..=0 };
        let widths: Vec<(usize, u8)> = (1..=wmax).map(|t| (t, 0)).chain(if self.slots > 1 { (5..=WMAX).map(|t| (t, 1)).collect() } else { Vec::new() }).chain(split.map(|t| (t, 2))).collect();
        let n = (widths.iter().map(|w| w.0).sum::<usize>() + 8).min(self.max_ctx - 1);
        let toks: Vec<u32> = (0..n as u32).map(|i| 2000 + (i * 6151) % 40000).collect();
        self.seq().allow_sampled.store(false, Ordering::Release);
        self.set_forced(&toks);
        self.touch_pages(self.cur, n);
        let mut at = 0;
        for (t, kind) in widths {
            if at + t + 1 > n {
                self.gpu.sync();
                self.reset();
                self.seq().allow_sampled.store(false, Ordering::Release);
                self.set_forced(&toks);
                self.touch_pages(self.cur, n);
                at = 0;
            }
            ops::TMV_BATCHED.store(kind > 0, Ordering::Relaxed);
            self.gpu.client.graph_prepare().expect("graph_prepare");
            self.lanes_single();
            // a split window of the one slot runs both halves at the same positions: its state is thrown away
            let launch = |e: &Self| if kind == 2 { e.launch_split(t.div_ceil(2), t / 2) } else { e.launch_window(t, None) };
            launch(self);
            self.gpu.client.start_capture().expect("start_capture");
            launch(self);
            let g = self.gpu.client.stop_capture().expect("stop_capture");
            match kind {
                0 => self.win_graphs[t] = Some(g),
                1 => self.batch_graphs[t] = Some(g),
                _ => self.split_graphs[t] = Some(g),
            }
            at += t;
        }
        ops::TMV_BATCHED.store(false, Ordering::Relaxed);
        self.gpu.sync();
        self.reset();
    }

    /// Queues a window of t tokens of the current slot: its captured graph, or eager launches.
    fn run_window(&self, t: usize) {
        self.lanes_single();
        match &self.win_graphs[t] {
            // SAFETY: every buffer the graph uses is owned by `self` and outlives it.
            Some(g) => unsafe { g.replay().expect("graph replay") },
            None => self.launch_window(t, None),
        }
    }

    /// Writes the step state words: position, last sampled token, forced count.
    fn set_step(&self, pos: usize, sampled: u32, forced: usize) {
        let mut st = self.step_words();
        st[0] = pos as u32;
        st[2] = sampled;
        st[3] = forced as u32;
        self.set_step_words(&st);
    }

    /// Makes the forced context `self.tokens[..from]` followed by `toks` (positions from `from` on), for the
    /// device and the PLE thread.
    fn force_tail(&mut self, from: usize, toks: &[u32]) {
        let sh = &self.driver.shared;
        let mut f = sh.seqs[self.cur].forced.lock().unwrap();
        f.truncate(from);
        if f.len() < from {
            f.clear();
            f.extend_from_slice(&self.tokens[..from]);
        }
        f.extend_from_slice(toks);
        sh.seqs[self.cur].n_forced.store(f.len() as u32, Ordering::Release);
        let n = f.len();
        drop(f);
        self.ctx.upload_at(self.ctx_at(self.cur) + from * 4, toks);
        let _ = n;
    }

    /// One speculative step. `self.tokens` holds the committed tokens; the last, at position `self.pos`, is not yet
    /// processed. Runs it and `drafts` (at most WMAX - 1) as one window, keeps the drafts the target's own samples
    /// confirm plus the sample after them, and rolls the sequence state back to that point. Returns the kept
    /// samples: what plain decoding would have produced at those positions, bit for bit.
    pub fn spec_step(&mut self, drafts: &[u32]) -> Vec<u32> {
        let p0 = self.pos;
        assert_eq!(self.tokens.len(), p0 + 1, "spec_step needs the last committed token unprocessed");
        let k = drafts.len().min(WMAX - 1).min(self.max_ctx - p0 - 1);
        let mut w = vec![self.tokens[p0]];
        w.extend_from_slice(&drafts[..k]);
        let t = w.len();
        let c0 = std::time::Instant::now();
        self.gpu.sync();
        self.touch_pages(self.cur, p0 + t);
        self.force_tail(p0, &w);
        self.set_step(p0, self.tokens[p0], p0 + t);
        let c1 = std::time::Instant::now();
        self.run_window(t);
        self.gpu.sync();
        let c2 = std::time::Instant::now();
        let samples: Vec<u32> = self.s.samples.download(t);
        let mut keep = 1;
        while keep < t && w[keep] == samples[keep - 1] {
            keep += 1;
        }
        let kept = samples[..keep].to_vec();
        if keep < t {
            self.rollback(keep);
        }
        let pos = p0 + keep;
        self.tokens.truncate(p0 + 1);
        self.tokens.extend_from_slice(&kept);
        self.pos = pos;
        // the forced context ends before the new last token; the PLE thread restarts there when drafts were wrong
        {
            let sh = &self.driver.shared;
            let mut f = sh.seqs[self.cur].forced.lock().unwrap();
            f.truncate(pos);
            sh.seqs[self.cur].n_forced.store(f.len() as u32, Ordering::Release);
        }
        self.set_step(pos, kept[keep - 1], pos);
        let sh = &self.driver.shared;
        sh.count(self.cur).store(pos as u64, Ordering::Release);
        if keep < t {
            sh.seqs[self.cur].ple_rewind.store(pos as u32 + 1, Ordering::Release);
            while sh.seqs[self.cur].ple_rewind.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
        }
        let c3 = std::time::Instant::now();
        let mut pr = self.spec_prof.borrow_mut();
        pr[0] += (c1 - c0).as_secs_f64();
        pr[1] += (c2 - c1).as_secs_f64();
        pr[2] += (c3 - c2).as_secs_f64();
        pr[4 + t] += (c2 - c1).as_secs_f64();
        pr[13 + t] += 1.0;
        kept
    }

    /// The sequence state after only the first `keep` tokens of the window just run: conv states and the PLE
    /// history are rebuilt from their saved starts and the window's inputs, the DeltaNet states by replaying the
    /// kept tokens from the saved start (the same kernel, so the same bits).
    fn rollback(&self, keep: usize) {
        let (c, s) = (&self.cfg, &self.s);
        let sd = self.strides;
        let slot = self.cur;
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_dim);
        for (l, st) in self.state.iter().enumerate() {
            let LayerState::Gdn { conv, s: sst, qkv, cv, beta, alpha, s0, conv0 } = st else { continue };
            let MixerW::Gdn(gw) = &self.layers[l].mixer else { unreachable!() };
            let ch = gw.qkv.rows;
            self.k("qx_conv_rollback", 256, blocks(ch, 256), &[conv.ptr + (slot * sd.conv * 4) as u64, conv0.ptr, qkv.ptr, ch as u64, keep as u64]);
            // the window's lanes are still set: its first `keep` tokens are one segment of this slot
            self.k("qx_gdn_recur", 128, (vh as u32, 4, 1), &[cv.ptr, sst.ptr, s.o.ptr, beta.ptr, alpha.ptr, gw.dt_bias.ptr, gw.a.ptr, kh as u64, d as u64, keep as u64, ch as u64, vh as u64, 0, s0.ptr, s.slot.ptr, sd.gdn as u64]);
        }
        self.k("qx_ple_hist_rollback", 256, blocks(HCD, 256), &[s.hist.ptr + (slot * sd.hist * 4) as u64, s.hist0.ptr, s.cn.ptr, HCD as u64, keep as u64]);
    }

    /// Speculative generation: `prompt`, then up to `n_new` tokens, each window verified exactly, so the tokens
    /// are those plain decoding with the same sampling would produce. `draft(context, k)` proposes up to k
    /// tokens to follow the context. Returns the generated tokens and (windows, drafts proposed, drafts kept).
    pub fn generate_spec(&mut self, prompt: &[u32], n_new: usize, sp: &Sampling, k: usize, stop: &[u32], draft: Option<&mut dyn FnMut(&[u32], usize) -> Vec<u32>>) -> (Vec<u32>, [usize; 3]) {
        let mut out = Vec::new();
        let (_, _, stats) = self.spec_core(prompt, n_new, sp, k, draft, &mut |t| {
            if stop.contains(&t) {
                return false;
            }
            out.push(t);
            true
        });
        (out, stats)
    }

    /// Drafts per speculative window (0 without a drafter).
    pub fn drafter_k(&self) -> usize {
        self.drafter.as_ref().map_or(0, |d| d.k)
    }

    /// Whether `stream` decodes speculatively: a drafter is loaded and `QX_SPEC` is not 0.
    pub fn spec_enabled(&self) -> bool {
        self.drafter.is_some() && self.spec
    }

    /// The speculative loop: `on_token` gets each new token in order and returns false to stop. Returns (seconds
    /// until the prompt was read, seconds in all, [windows, drafts, drafts kept]).
    fn spec_core(&mut self, prompt: &[u32], n_new: usize, sp: &Sampling, k: usize, mut draft: Option<&mut dyn FnMut(&[u32], usize) -> Vec<u32>>, on_token: &mut dyn FnMut(u32) -> bool) -> (f64, f64, [usize; 3]) {
        let t0 = std::time::Instant::now();
        let n_new = n_new.min(self.max_ctx - prompt.len());
        let mut run = self.spec_begin(prompt, sp, k);
        let t_prompt = t0.elapsed().as_secs_f64();
        let (mut n_out, mut stopped) = (0usize, false);
        while n_out < n_new && !stopped {
            let kept = self.spec_next(&mut run, n_new - n_out, draft.as_mut().map(|f| &mut **f as &mut dyn FnMut(&[u32], usize) -> Vec<u32>));
            for &tk in &kept {
                if n_out >= n_new || !on_token(tk) {
                    stopped = true;
                    break;
                }
                n_out += 1;
            }
        }
        if std::env::var("QX_SPEC_PROF").is_ok() {
            let pr = self.spec_prof.borrow();
            eprintln!("spec: setup {:.3}s, windows {:.3}s, accept+rollback {:.3}s, drafts {:.3}s", pr[0], pr[1], pr[2], pr[3]);
            let sh = &self.driver.shared;
            let (ns, jobs, miss) = (sh.cpu_ns.swap(0, Ordering::Relaxed), sh.cpu_jobs.swap(0, Ordering::Relaxed), sh.cpu_misses.swap(0, Ordering::Relaxed));
            eprintln!("  cpu pool: {:.3}s over {jobs} jobs, {miss} misses ({:.1} per job, {:.1} us per miss)", ns as f64 * 1e-9, miss as f64 / jobs.max(1) as f64, ns as f64 * 1e-3 / miss.max(1) as f64);
            for t in 1..=WMAX {
                if pr[13 + t] > 0.0 {
                    eprintln!("  width {t}: {} windows, {:.2} ms each", pr[13 + t], 1e3 * pr[4 + t] / pr[13 + t]);
                }
            }
        }
        *self.spec_prof.borrow_mut() = [0.0; 24];
        (t_prompt, t0.elapsed().as_secs_f64(), run.stats)
    }

    /// Starts speculative decoding in the current slot (empty: `reset` first): forces `prompt` and runs all of it
    /// but the last token (prefill when long enough, then windows), the drafter catching up throughout. Up to `k`
    /// drafts per window.
    pub fn spec_begin(&mut self, prompt: &[u32], sp: &Sampling, k: usize) -> SpecRun {
        // pos > 0: a restored prefix of the prompt
        assert!(self.pos < prompt.len() && !prompt.is_empty() && self.tokens.len() >= self.pos && self.tokens[..self.pos] == prompt[..self.pos]);
        self.draft_slot = Some(self.cur);
        self.set_sampling(sp);
        self.set_forced(prompt);
        self.seq().allow_sampled.store(false, Ordering::Release);
        let n = prompt.len() - 1;
        let done = self.prefill_prompt(prompt).max(self.pos);
        // the rest of the prompt (all of it when not prefilled) in windows, without drafts
        self.tokens = prompt[..done].to_vec();
        let mut p = done;
        while p < n {
            let t = (n - p).min(WMAX);
            self.tokens.extend_from_slice(&prompt[p..p + t]);
            // the step state is the host's to change only once the last window and catch-up have run
            self.gpu.sync();
            self.set_step(p, 0, prompt.len());
            self.touch_pages(self.cur, p + t);
            self.run_window(t);
            if self.drafter.is_some() {
                // the window left the step state at p + t
                self.draft_catch_up(p, t, self.s.r.ptr, self.ctx.ptr + (self.ctx_at(self.cur) + (p + 1) * 4) as u64, false, p + t);
            }
            p += t;
        }
        self.gpu.sync();
        self.pos = n;
        self.tokens = prompt.to_vec();
        {
            let sh = &self.driver.shared;
            let mut f = sh.seqs[self.cur].forced.lock().unwrap();
            f.truncate(n);
            sh.seqs[self.cur].n_forced.store(n as u32, Ordering::Release);
        }
        let tau = std::env::var("QX_SPEC_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);
        SpecRun { mtp: Vec::new(), cost: [0.0; WMAX + 1], tau, k, stats: [0; 3] }
    }

    /// One speculative window of the current slot, at most `want` tokens wanted: the tokens it keeps, each the one
    /// plain decoding would sample there. `draft(context, k)` proposes drafts instead of the MTP drafter.
    pub fn spec_next(&mut self, run: &mut SpecRun, want: usize, draft: Option<&mut dyn FnMut(&[u32], usize) -> Vec<u32>>) -> Vec<u32> {
        let room = want.min(self.max_ctx - self.pos - 1);
        let kk = run.k.min(room.saturating_sub(1)).min(WMAX - 1);
        let own = draft.is_none();
        let d = match draft {
            Some(f) if kk > 0 => f(&self.tokens, kk),
            Some(_) => Vec::new(),
            None => std::mem::take(&mut run.mtp).into_iter().take(kk).collect(),
        };
        let t_win = std::time::Instant::now();
        let tw = d.len().min(WMAX - 1) + 1;
        let kept = self.spec_step(&d);
        let dt = t_win.elapsed().as_secs_f64();
        // running cost of a window by width
        run.cost[tw] = if run.cost[tw] > 0.0 { 0.8 * run.cost[tw] + 0.2 * dt } else { dt };
        if own && self.drafter.is_some() && run.k > 0 {
            let c0 = std::time::Instant::now();
            let (toks, probs) = self.mtp_drafts(kept.len());
            let t_draft = c0.elapsed().as_secs_f64();
            self.spec_prof.borrow_mut()[3] += t_draft;
            run.mtp = toks[..pick_width(&probs[..kk.min(probs.len())], &run.cost, t_draft, run.tau)].to_vec();
        }
        run.stats[0] += 1;
        run.stats[1] += d.len().min(WMAX - 1);
        run.stats[2] += kept.len() - 1;
        kept
    }

    /// Hands the current slot's speculative sequence to `step_segments`: its next token is its last sample, as a
    /// batched slot's is. (The drafter stops following it.)
    pub fn spec_to_batch(&mut self) {
        self.seq().allow_sampled.store(true, Ordering::Release);
        self.draft_slot = None;
    }

    /// Runs all but the last token of a forced prompt (already `set_forced` from position 0) with the batched
    /// prefill, when it is enabled and the prompt has at least `QX_PREFILL_MIN` tokens (default 96: shorter ones
    /// are cheaper token by token than staging every missing expert). Returns the steps it covered.
    fn prefill_prompt(&mut self, prompt: &[u32]) -> usize {
        let min: usize = std::env::var("QX_PREFILL_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(96);
        let n = prompt.len() - 1;
        let from = self.pos;
        if !self.can_prefill() || n < from + min.max(1) {
            return from;
        }
        self.tokens = prompt[..n].to_vec();
        self.prefill(n - from);
        if self.snapshot_prefill {
            self.prefix_snapshot(self.cur, 1);
        }
        n
    }

    /// The widest window: TMAX tokens with several slots, else a speculative window's WMAX.
    pub fn max_window(&self) -> usize {
        if self.slots > 1 { TMAX } else { WMAX }
    }

    /// Starts a sequence in `slot` (made current): clears it, sets its sampling and forces `prompt`, and runs
    /// all but the prompt's last token with the batched prefill when the prompt is long enough. Returns the
    /// slot's position: the prompt tokens from there on go through `step_segments`.
    pub fn begin_seq(&mut self, slot: usize, prompt: &[u32], sp: &Sampling) -> usize {
        self.begin_seq_deferred(slot, prompt, sp, false)
    }

    /// `begin_seq`; with `defer`, no prefill now: the caller feeds the prompt through `prefill_more` and
    /// `step_segments` (chunked prefill interleaved with other sequences' steps).
    pub fn begin_seq_deferred(&mut self, slot: usize, prompt: &[u32], sp: &Sampling, defer: bool) -> usize {
        assert!(!prompt.is_empty() && prompt.len() < self.max_ctx);
        self.select(slot);
        if self.draft_slot == Some(slot) {
            self.draft_slot = None;
        }
        self.reset();
        self.set_sampling(sp);
        self.set_forced(prompt);
        self.seq().allow_sampled.store(true, Ordering::Release);
        let done = if defer { 0 } else { self.prefill_prompt(prompt) };
        self.pos = done;
        self.tokens = prompt.to_vec();
        done
    }

    /// Runs the next `n` forced tokens of `slot` (positions pos .. pos + n, all before the prompt's last token)
    /// with the batched prefill, in chunks of at most `prefill_chunk` tokens. Waits for the GPU.
    pub fn prefill_more(&mut self, slot: usize, n: usize) {
        self.select(slot);
        assert!(self.pos + n < self.tokens.len(), "prefill past the prompt's last token");
        self.prefill(n);
        // a chunk's snapshot only while the cache has a free entry: a long prompt's chunks would otherwise cycle
        // a small cache (8 entries of 497 MiB at 262k) and evict every prompt in it; the prompt's end is saved
        // by the server regardless
        if self.snapshot_prefill && self.prefix_has_room(self.pos) {
            self.prefix_snapshot(slot, 1);
        }
    }

    /// Borrows the expert cache's tail for the prompt buffers (a no-op when they have their own memory or the
    /// tail is lent already): the swapper evicts its experts, then the queued work that may still read them
    /// finishes.
    pub fn prefill_lend(&self) {
        if !self.lend || self.lent.get() {
            return;
        }
        let sh = &self.driver.shared;
        let (lo, n) = self.prefill.as_ref().unwrap().region.expect("lent prompt buffers have a region");
        sh.lend_lo.store(lo, Ordering::Relaxed);
        sh.lend_hi.store(lo + n as u64, Ordering::Relaxed);
        sh.lend_req.store(1, Ordering::Release);
        while sh.lend_state.load(Ordering::Acquire) != 1 {
            std::hint::spin_loop();
        }
        self.gpu.sync();
        self.lent.set(true);
    }

    /// Gives the tail back once the prompt work that used it is done: the swapper copies its experts back in.
    pub fn prefill_release(&self) {
        if !self.lent.get() {
            return;
        }
        self.gpu.sync();
        let sh = &self.driver.shared;
        sh.lend_req.store(0, Ordering::Release);
        while sh.lend_state.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        self.lent.set(false);
    }

    /// Tokens per prefill chunk (0: no batched prefill), of the small tier when there are two.
    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill.iter().chain(self.prefill_alt.iter()).map(|p| p.c).min().unwrap_or(0)
    }

    /// The longest chunk worth running for a prompt with `left` tokens still to read: the big tier's when there
    /// is one and `left` fills at least one of its chunks.
    pub fn prefill_chunk_for(&self, left: usize) -> usize {
        let big = self.prefill.iter().chain(self.prefill_alt.iter()).map(|p| p.c).max().unwrap_or(0);
        if left >= big { big } else { self.prefill_chunk_tokens() }
    }

    /// Makes the big (or small) tier of prompt buffers the current one, giving back a lent tail of the other.
    pub(crate) fn prefill_tier(&mut self, big: bool) {
        if self.prefill_alt.is_none() || big == self.prefill_big {
            return;
        }
        self.prefill_release();
        self.gpu.sync();
        std::mem::swap(&mut self.prefill, &mut self.prefill_alt);
        self.prefill_big = big;
        // the staging handshake counts groups over every chunk (`stage_ready` is the engine's): the tier taking
        // over continues the other's count
        let (g, f) = {
            let o = self.prefill_alt.as_ref().unwrap();
            (o.generation, o.freed.atomic_u64(0).load(Ordering::Acquire))
        };
        let p = self.prefill.as_mut().unwrap();
        p.generation = g;
        p.freed.atomic_u64(0).store(f, Ordering::Release);
    }

    /// The host's position of `slot`: the next position its windows run.
    pub fn slot_pos(&self, slot: usize) -> usize {
        if slot == self.cur { self.pos } else { self.seqs[slot].pos }
    }

    /// One window over several slots: `segs` lists (slot, tokens) in order, TMAX tokens at most. A slot's tokens
    /// run at its next positions, each its forced token or, for a first token past the forced ones, its last
    /// sample. Returns every token's sample (a slot's next token is its segment's last). Each token's logits
    /// equal those of the slot's tokens run alone.
    pub fn step_segments(&mut self, segs: &[(usize, usize)]) -> Vec<u32> {
        let t = self.step_segments_async(segs);
        self.gpu.sync();
        match self.last_split.get() {
            0 => self.s.samples.download(t),
            ta => [self.s.samples.download(ta), self.half2.as_ref().unwrap().s.samples.download(t - ta)].concat(),
        }
    }

    /// `step_segments` without waiting: the window is queued and the slots' positions advanced. Its samples
    /// appear in the slots' output rings (`sample_ready`, `sample_at`). Returns the window's token count.
    pub fn step_segments_async(&mut self, segs: &[(usize, usize)]) -> usize {
        let mut lanes: Vec<u32> = segs.iter().flat_map(|&(s, n)| std::iter::repeat_n(s as u32, n)).collect();
        let t = lanes.len();
        assert!((1..=self.max_window()).contains(&t), "a window of {t} tokens");
        let last = *lanes.last().unwrap();
        lanes.resize(TMAX, last);
        for &(slot, n) in segs {
            // a segment past the forced tokens must be one token: later tokens would wait on their own samples
            let forced = if slot == self.cur { self.tokens.len() } else { self.seqs[slot].tokens.len() };
            assert!(n == 1 || self.slot_pos(slot) + n <= forced, "slot {slot}: a window of {n} tokens past its forced tokens");
            self.touch_pages(slot, self.slot_pos(slot) + n);
        }
        // split in two halves of whole segments, the first ceil(t / 2) tokens
        let ta = t.div_ceil(2);
        let split = self.half2.is_some() && segs.len() > 1 && t >= split_min() && segs.iter().scan(0, |n, s| { *n += s.1; Some(*n) }).any(|n| n == ta);
        if split {
            self.queue_lanes(&lanes);
            ops::TMV_BATCHED.store(true, Ordering::Relaxed);
            match &self.split_graphs[t] {
                // SAFETY: every buffer the graph uses is owned by `self` and outlives it.
                Some(g) => unsafe { g.replay().expect("graph replay") },
                None => self.launch_split(ta, t - ta),
            }
            ops::TMV_BATCHED.store(false, Ordering::Relaxed);
            self.last_split.set(ta);
        } else if segs.len() > 1 && t > 4 && t <= WMAX {
            self.queue_lanes(&lanes);
            match &self.batch_graphs[t] {
                // SAFETY: every buffer the graph uses is owned by `self` and outlives it.
                Some(g) => unsafe { g.replay().expect("graph replay") },
                None => {
                    ops::TMV_BATCHED.store(true, Ordering::Relaxed);
                    self.launch_window(t, None);
                    ops::TMV_BATCHED.store(false, Ordering::Relaxed);
                }
            }
        } else if segs.len() == 1 && segs[0].0 == self.cur {
            self.run_window(t);
        } else {
            self.queue_lanes(&lanes);
            match &self.win_graphs[t] {
                Some(g) => unsafe { g.replay().expect("graph replay") },
                None => self.launch_window(t, None),
            }
        }
        if !split {
            self.last_split.set(0);
        }
        // launches reach the GPU now rather than at the next sync
        self.gpu.client.flush().expect("flush");
        for &(slot, n) in segs {
            if slot == self.cur {
                self.pos += n;
            } else {
                self.seqs[slot].pos += n;
            }
        }
        t
    }

    /// Sets lane `lane`'s token mask for the next window: `allowed` is a bit vector over the vocabulary (bit i:
    /// token i may be sampled), or None for no mask. Only for windows queued after the previous one completed.
    pub fn set_mask(&self, lane: usize, allowed: Option<&[u32]>) {
        let words = self.cfg.n_vocab.div_ceil(32);
        match allowed {
            Some(bits) => {
                let dst = self.masks.slice::<u32>(TMAX * 4 + lane * words * 4, words);
                dst.fill(0);
                dst[..bits.len().min(words)].copy_from_slice(&bits[..bits.len().min(words)]);
                self.masks.slice::<u32>(lane * 4, 1)[0] = 1;
            }
            None => self.masks.slice::<u32>(lane * 4, 1)[0] = 0,
        }
    }

    /// Whether the sample of `slot`'s token at position `pos` has been published.
    pub fn sample_ready(&self, slot: usize, pos: usize) -> bool {
        self.driver.shared.count(slot).load(Ordering::Acquire) > pos as u64
    }

    /// The token sampled after `slot`'s position `pos` (once `sample_ready`).
    pub fn sample_at(&self, slot: usize, pos: usize) -> u32 {
        self.driver.shared.outs(slot, pos + 1)[pos]
    }

    /// The logits rows of the last window (`n_vocab` floats per token).
    pub fn window_logits(&self, t: usize) -> Vec<f32> {
        let n = self.cfg.n_vocab;
        match self.last_split.get() {
            0 => self.s.logits.download(t * n),
            ta => [self.s.logits.download(ta * n), self.half2.as_ref().unwrap().s.logits.download((t - ta) * n)].concat(),
        }
    }

    /// Streams up to `max_new` sampled tokens after `prompt` (the sequence must be empty: call `reset` first).
    /// `on_token` gets each generated token and returns false to stop; tokens in `stop` end the generation
    /// without being passed on. Steps are queued a few ahead of the tokens produced, so stopping wastes at most
    /// that many. Returns (tokens passed on, seconds until the prompt was read, seconds in all).
    pub fn stream(&mut self, prompt: &[u32], max_new: usize, sp: &Sampling, stop: &[u32], mut on_token: impl FnMut(u32) -> bool) -> (usize, f64, f64) {
        assert!(!prompt.is_empty() && max_new > 0);
        if self.spec_enabled() {
            let k = self.drafter.as_ref().unwrap().k;
            let mut passed = 0;
            let (tp, ta, _) = self.spec_core(prompt, max_new, sp, k, None, &mut |t| {
                if stop.contains(&t) || !on_token(t) {
                    return false;
                }
                passed += 1;
                true
            });
            return (passed, tp, ta);
        }
        assert!(self.pos == 0, "the plain stream starts from an empty sequence");
        let max_new = max_new.min(self.max_ctx - prompt.len());
        let t0 = std::time::Instant::now();
        self.set_sampling(sp);
        self.set_forced(prompt);
        self.touch_pages(self.cur, prompt.len() + max_new);
        self.seq().allow_sampled.store(true, Ordering::Release);
        let total = prompt.len() - 1 + max_new;
        let first_gen = prompt.len() - 1;
        const AHEAD: usize = 6;
        let prefilled = self.prefill_prompt(prompt);
        let (mut queued, mut emitted, mut passed) = (prefilled, first_gen, 0usize);
        let mut t_prompt = 0.0;
        let mut stopped = false;
        loop {
            let c = {
                let sh = &self.driver.shared;
                sh.count(self.cur).load(Ordering::Acquire) as usize
            };
            if t_prompt == 0.0 && c >= first_gen {
                t_prompt = t0.elapsed().as_secs_f64();
            }
            while emitted < c && !stopped {
                let tok = self.driver.shared.outs(self.cur, emitted + 1)[emitted];
                emitted += 1;
                if stop.contains(&tok) || !on_token(tok) {
                    stopped = true;
                } else {
                    passed += 1;
                }
            }
            if stopped || emitted >= total {
                break;
            }
            let target = (c.max(first_gen) + AHEAD).min(total);
            if queued < target {
                if self.graph.is_none() && target - queued < 2 {
                    self.lanes_single();
                    self.launch_step(None);
                    queued += 1;
                } else {
                    let had = self.graph.is_some() || std::env::var("QX_GRAPH").is_ok_and(|v| v == "0");
                    self.queue_step();
                    queued += if had { 1 } else { 2 };
                }
            } else {
                std::hint::spin_loop();
            }
        }
        self.gpu.sync();
        self.pos = queued;
        (passed, t_prompt, t0.elapsed().as_secs_f64())
    }
}
