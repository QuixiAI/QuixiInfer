//! The CPU side of a free-running step: one driver thread that answers the GPU's MoE jobs with the expert
//! pool and posts each position's PLE embedding once its token is known.
//!
//! Layouts mirror kernels/include/qx_sys.cuh.
use qx_cpu::pool::{Expert, GateFmt, Pool};
use qx_gpu::mem::HostMapped;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Tokens per speculative window (one sequence), tokens per window (all sequences), selections per window and
/// u32 step words per sequence slot (kernels/include/qx_sys.cuh).
pub const WMAX: usize = 8;
pub const TMAX: usize = 32;
pub const AMAX: usize = 320;
pub const STEP_ROW: usize = 16;
const _: () = assert!(qx_cpu::pool::MAX_EXPERTS >= AMAX, "the CPU pool must take a whole window's misses");
pub const MB_SEQ: usize = 0;
pub const MB_DONE: usize = 64;
pub const MB_LAYER: usize = 128;
pub const MB_NMISS: usize = 132;
pub const MB_NTOK: usize = 136;
pub const MB_KPT: usize = 140;
pub const MB_IDS: usize = 256;
pub const MB_KIDX: usize = 1536;
pub const MB_ALL: usize = 2816;
pub const MB_X: usize = 4096;
/// AMAX rows of E floats: the unweighted output of the missing expert with selection index a at row a.
pub fn mb_y(e: usize) -> usize {
    MB_X + 4 * TMAX * e
}
pub fn mb_all(_e: usize) -> usize {
    MB_ALL
}
pub fn mb_bytes(e: usize) -> usize {
    mb_y(e) + 4 * AMAX * e + 64
}
pub const PLE_CONSUMED: usize = 0;
pub const PLE_READY: usize = 64;
pub const PLE_EMB: usize = 128;
/// Slots of each sequence's PLE embedding ring (at least a prefill chunk).
pub const PLE_CAP: usize = 8192;

/// Where one layer's routed experts are in the pinned host copy: record e at `rec0 + e * rec_stride`, holding
/// gate rows, up rows (each `gate_bytes`), then down rows.
#[derive(Clone, Copy)]
pub struct LayerPtrs {
    pub rec0: *const u8,
    pub rec_stride: usize,
    pub rec_bytes: usize,
    pub gate_bytes: usize,
    pub gate_row_bytes: usize,
    pub down_row_bytes: usize,
    pub fmt: GateFmt,
}
unsafe impl Send for LayerPtrs {}

impl LayerPtrs {
    fn rec(&self, e: usize) -> *const u8 {
        unsafe { self.rec0.add(e * self.rec_stride) }
    }
    /// Expert `e`'s per-tensor scales {gate, up, down} from its record tail (1 for the formats without one).
    fn scales(&self, e: usize) -> [f32; 3] {
        if self.fmt != GateFmt::Nvfp4 {
            return [1.0; 3];
        }
        let p = unsafe { self.rec(e).add(self.rec_bytes - 16) };
        let f = |i: usize| f32::from_le_bytes(unsafe { std::slice::from_raw_parts(p.add(4 * i), 4) }.try_into().unwrap());
        [f(0), f(1), f(2)]
    }
}

/// Adaptive VRAM residency: decayed routing frequency decides which experts the cache holds.
pub struct SwapConfig {
    pub half_life_tokens: f64,
    pub max_swaps_per_token: usize,
    pub margin: f64,
    /// Where the swapper keeps its ranking (hottest first) for the next start; see `routing::default_order`.
    pub save: Option<std::path::PathBuf>,
}

/// The cache state the driver owns while decoding.
pub struct CacheState {
    pub table: u64,
    pub host_table: Vec<u64>,
    pub n_expert: usize,
    pub free: Vec<Vec<u64>>,
    pub slot_bytes: Vec<usize>,
    pub layer_class: Vec<usize>,
}
unsafe impl Send for CacheState {}

struct Swapper {
    cs: CacheState,
    cfg: SwapConfig,
    score: Vec<f64>,
    scale: f64,
    decay: f64,
    /// Experts routed this token that the cache does not hold.
    missed: Vec<usize>,
    /// Evicted slots, reusable once `ticks` reaches the first field, with the expert that evicted the slot's
    /// last one: it is copied in as the slot comes out (it used to wait to be missed again, a round trip of at
    /// least three windows for every swap).
    quarantine: std::collections::VecDeque<(u64, usize, u64, Option<usize>)>,
    /// Calls to `end_token`. The GPU's step count cannot time quarantine: resets and rollbacks move it back.
    ticks: u64,
    stream: cudarc::driver::sys::CUstream,
    /// Pinned words the table writes copy from; a ring large enough that a word is never reused in flight.
    words: HostMapped,
    word_next: usize,
    /// Experts a prefill routed to that the cache lacks, hottest first, swapped in a burst over the next tokens,
    /// and the swaps (or evictions) the burst may still spend.
    bulk: std::collections::VecDeque<usize>,
    burst: usize,
    /// While the cache's tail is lent: the experts evicted from it (expert, slot) and its free slots (class, slot).
    lent: Vec<(usize, u64)>,
    lent_free: Vec<(usize, u64)>,
}

impl Swapper {
    fn new(cs: CacheState, cfg: SwapConfig, ranking: &[(usize, usize)]) -> Self {
        let n = cs.host_table.len();
        let mut score = vec![0f64; n];
        for (i, &(l, e)) in ranking.iter().enumerate() {
            score[l * cs.n_expert + e] = 1.0 / (1.0 + i as f64 / 2000.0);
        }
        let stream = unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("copy stream");
        let decay = 0.5f64.powf(1.0 / cfg.half_life_tokens);
        Self { cs, cfg, score, scale: 1.0, decay, missed: Vec::new(), quarantine: Default::default(), ticks: 0, stream, words: HostMapped::new(1 << 20), word_next: 0, lent: Vec::new(), lent_free: Vec::new(), bulk: Default::default(), burst: 0 }
    }

    fn observe(&mut self, l: usize, ids: &[i32]) {
        for &e in ids {
            let g = l * self.cs.n_expert + e as usize;
            self.score[g] += self.scale;
            if self.cs.host_table[g] == 0 {
                self.missed.push(g);
            }
        }
    }

    /// Takes a prefill's routing counts: they join the scores as if just observed, and the experts among them
    /// the cache lacks become burst candidates.
    fn observe_bulk(&mut self, counts: &[u32]) {
        let mut cand = Vec::new();
        for (g, &c) in counts.iter().enumerate() {
            if c > 0 {
                self.score[g] += self.scale * c as f64;
                if self.cs.host_table[g] == 0 {
                    cand.push(g);
                }
            }
        }
        cand.sort_by(|&a, &b| self.score[b].partial_cmp(&self.score[a]).unwrap());
        self.bulk = cand.into();
        self.burst = std::env::var("QX_BULK_SWAPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3072);
    }

    /// Writes table entry `g` on the copy stream, after everything queued on it so far.
    fn write_entry(&mut self, g: usize, v: u64) {
        let i = self.word_next;
        self.word_next = (self.word_next + 1) % (self.words.bytes / 8);
        self.words.slice::<u64>(i * 8, 1)[0] = v;
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(self.cs.table + 8 * g as u64, self.words.ptr.add(i * 8) as *const _, 8, self.stream).result().expect("table write");
        }
        self.cs.host_table[g] = v;
    }

    /// Once per token: decay, release quarantined slots, then swap the hottest missed experts in over the
    /// coldest resident ones of their size class. Returns the swaps issued.
    /// Lends the slots overlapping [lo, hi): evicts their experts (table entries to 0), takes the free and
    /// quarantined ones out of circulation, and waits until the table writes have landed.
    fn lend(&mut self, lo: u64, hi: u64) {
        let ne = self.cs.n_expert;
        let in_tail = |a: u64, bytes: usize| a < hi && a + bytes as u64 > lo;
        for g in 0..self.cs.host_table.len() {
            let a = self.cs.host_table[g];
            let c = self.cs.layer_class[g / ne];
            if a != 0 && in_tail(a, self.cs.slot_bytes[c]) {
                self.write_entry(g, 0);
                self.lent.push((g, a));
            }
        }
        for c in 0..self.cs.free.len() {
            let sb = self.cs.slot_bytes[c];
            let (keep, take): (Vec<u64>, Vec<u64>) = self.cs.free[c].iter().partition(|&&a| !in_tail(a, sb));
            self.cs.free[c] = keep;
            self.lent_free.extend(take.into_iter().map(|a| (c, a)));
        }
        let sb = self.cs.slot_bytes.clone();
        let mut q = std::mem::take(&mut self.quarantine);
        q.retain(|&(_, c, a, _)| {
            if in_tail(a, sb[c]) {
                self.lent_free.push((c, a));
                false
            } else {
                true
            }
        });
        self.quarantine = q;
        unsafe { cudarc::driver::sys::cuStreamSynchronize(self.stream).result().expect("lend") };
    }

    /// Gives the lent slots back: each evicted expert the cache has not taken in elsewhere is copied back into its
    /// slot (table entry written after the copy, on the same stream); the other slots become free.
    fn reclaim(&mut self, layers: &[LayerPtrs]) {
        let ne = self.cs.n_expert;
        for (g, a) in std::mem::take(&mut self.lent) {
            let c = self.cs.layer_class[g / ne];
            if self.cs.host_table[g] == 0 {
                let lp = &layers[g / ne];
                unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(a, lp.rec(g % ne) as *const _, lp.rec_bytes, self.stream).result().expect("expert copy");
                }
                self.write_entry(g, a);
            } else {
                self.cs.free[c].push(a);
            }
        }
        for (c, a) in std::mem::take(&mut self.lent_free) {
            self.cs.free[c].push(a);
        }
    }

    /// Writes the ranking, hottest first, as `layer expert` lines.
    fn save(&self) {
        let Some(path) = &self.cfg.save else { return };
        let ne = self.cs.n_expert;
        let mut g: Vec<usize> = (0..self.score.len()).collect();
        g.sort_by(|&a, &b| self.score[b].partial_cmp(&self.score[a]).unwrap());
        let text: String = g.iter().map(|&g| format!("{} {}\n", g / ne, g % ne)).collect();
        let tmp = path.with_extension("tmp");
        if std::fs::create_dir_all(path.parent().unwrap()).and_then(|_| std::fs::write(&tmp, text)).and_then(|_| std::fs::rename(&tmp, path)).is_err() {
            eprintln!("routing profile: cannot write {}", path.display());
        }
    }

    fn end_token(&mut self, layers: &[LayerPtrs], ntok: usize) -> usize {
        self.ticks += 1;
        let mut issued = 0;
        let step = self.ticks;
        if step % 4096 == 0 {
            self.save();
        }
        self.scale /= self.decay.powi(ntok as i32);
        if self.scale > 1e200 {
            for s in self.score.iter_mut() {
                *s /= self.scale;
            }
            self.scale = 1.0;
        }
        let ne = self.cs.n_expert;
        while let Some(&(when, c, addr, next)) = self.quarantine.front() {
            if when > step {
                break;
            }
            self.quarantine.pop_front();
            match next {
                Some(g) if self.cs.host_table[g] == 0 && admit_on_release() => {
                    let lp = &layers[g / ne];
                    unsafe {
                        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(addr, lp.rec(g % ne) as *const _, lp.rec_bytes, self.stream).result().expect("expert copy");
                    }
                    self.write_entry(g, addr);
                    issued += 1;
                }
                _ => self.cs.free[c].push(addr),
            }
        }
        let mut cand = std::mem::take(&mut self.missed);
        cand.sort_unstable();
        cand.dedup();
        cand.sort_by(|&a, &b| self.score[b].partial_cmp(&self.score[a]).unwrap());
        // a window of several tokens observed as many routings: it may swap as many times over
        cand.truncate((self.cfg.max_swaps_per_token * ntok).min(64));
        // after a prefill, a burst of its hottest missing experts on top
        let mut from_bulk = std::collections::HashSet::new();
        let mut extra = 56usize.min(self.burst);
        while extra > 0 {
            let Some(g) = self.bulk.pop_front() else { break };
            if self.cs.host_table[g] == 0 && !cand.contains(&g) {
                cand.push(g);
                from_bulk.insert(g);
                extra -= 1;
                self.burst -= 1;
            }
        }
        if self.burst == 0 {
            self.bulk.clear();
        }
        // Candidates with a free slot go in now; the rest each evict one cold resident, whose slot frees up
        // after quarantine, and come back while they stay hot.
        let mut need: Vec<Vec<usize>> = vec![Vec::new(); self.cs.free.len()];
        for g in cand {
            let (l, e) = (g / ne, g % ne);
            let c = self.cs.layer_class[l];
            if let Some(slot) = self.cs.free[c].pop() {
                let lp = &layers[l];
                unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(slot, lp.rec(e) as *const _, lp.rec_bytes, self.stream).result().expect("expert copy");
                }
                self.write_entry(g, slot);
                issued += 1;
            } else {
                need[c].push(g);
            }
        }
        for (c, gs) in need.into_iter().enumerate() {
            if gs.is_empty() {
                continue;
            }
            let mut res: Vec<usize> = (0..self.cs.host_table.len()).filter(|&h| self.cs.host_table[h] != 0 && self.cs.layer_class[h / ne] == c).collect();
            let k = gs.len().min(res.len());
            if k == 0 {
                continue;
            }
            res.select_nth_unstable_by(k - 1, |&a, &b| self.score[a].partial_cmp(&self.score[b]).unwrap());
            res.truncate(k);
            res.sort_by(|&a, &b| self.score[a].partial_cmp(&self.score[b]).unwrap());
            for (g, v) in gs.into_iter().zip(res) {
                if self.score[g] > self.score[v] * (1.0 + self.cfg.margin) {
                    let addr = self.cs.host_table[v];
                    self.write_entry(v, 0);
                    self.quarantine.push_back((step + 2, c, addr, Some(g)));
                    // a burst candidate takes the slot once it is out of quarantine
                    if from_bulk.contains(&g) {
                        self.bulk.push_front(g);
                    }
                }
            }
        }
        issued
    }
}

/// Whether an expert that won an eviction is copied in as the slot leaves quarantine (`QX_ADMIT=1`) or waits to
/// be missed again (the default).
fn admit_on_release() -> bool {
    static X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X.get_or_init(|| std::env::var("QX_ADMIT").is_ok_and(|v| v == "1"))
}

/// What the PLE reader needs: the table on disk and the hash geometry.
pub struct PleSource {
    pub cfg: qx_ref::model::Config,
    pub table: crate::ple_io::PleTable,
    pub cache_rows: usize,
}

/// What the host and the PLE thread share about one sequence slot.
#[derive(Default)]
pub struct SeqShared {
    /// Tokens the host forces, by position, and how many.
    pub forced: std::sync::Mutex<Vec<u32>>,
    pub n_forced: AtomicU32,
    /// Position the PLE embedding is due for next; reset with the sequence.
    pub ple_next: AtomicU32,
    /// While set, the PLE thread leaves the slot alone (and forgets its tokens): the host is resetting it.
    pub paused: AtomicBool,
    /// Positions past the forced ones take the sampled token (generation); off, they wait for the host.
    pub allow_sampled: AtomicBool,
    /// 1 + the position the PLE thread is to restart from (0: none); the thread clears it once done.
    pub ple_rewind: AtomicU32,
    /// Bumped by every reset, so the PLE thread drops an embedding it read across one.
    pub epoch: AtomicU32,
}

pub struct Shared {
    /// The MoE mailbox, and the second half's of a split window (`Engine::launch_split`): the GPU posts the
    /// halves' jobs alternately and the driver answers them in that order.
    pub moe: HostMapped,
    pub moe2: HostMapped,
    /// Each slot's PLE mailbox, `ple_stride` bytes apart.
    pub ple: HostMapped,
    pub ple_stride: usize,
    /// Each slot's output, `out_stride` bytes apart: u32 per position, the token sampled at that step; offset
    /// `out_count_off` holds the slot's step count (u64).
    pub out: HostMapped,
    pub out_stride: usize,
    pub out_count_off: usize,
    pub seqs: Vec<SeqShared>,
    pub stop: AtomicBool,
    /// Routed experts the GPU served, routed experts in all, swaps issued.
    pub hits: std::sync::atomic::AtomicU64,
    pub routed: std::sync::atomic::AtomicU64,
    pub swaps: std::sync::atomic::AtomicU64,
    /// Routing counts per (layer, expert) of a batched prefill, for the swapper to take at its next token.
    pub bulk: std::sync::Mutex<Option<Vec<u32>>>,
    /// Driver accounting: nanoseconds in the pool, jobs answered, misses computed.
    pub cpu_ns: std::sync::atomic::AtomicU64,
    pub cpu_jobs: std::sync::atomic::AtomicU64,
    pub cpu_misses: std::sync::atomic::AtomicU64,
    /// Lending the expert cache's tail [lend_lo, lend_hi) to the prompt buffers: the engine sets `lend_req`
    /// (1 lend, 0 give back), the driver evicts or restores the experts there and acknowledges in `lend_state`.
    pub lend_req: AtomicU32,
    pub lend_state: AtomicU32,
    pub lend_lo: std::sync::atomic::AtomicU64,
    pub lend_hi: std::sync::atomic::AtomicU64,
}

impl Shared {
    /// Slot `s`'s step count: the positions whose sampled tokens are in its output ring.
    pub fn count(&self, s: usize) -> &std::sync::atomic::AtomicU64 {
        self.out.atomic_u64(s * self.out_stride + self.out_count_off)
    }
    /// Slot `s`'s output ring, positions 0 .. n.
    pub fn outs(&self, s: usize, n: usize) -> &mut [u32] {
        self.out.slice::<u32>(s * self.out_stride, n)
    }
    /// The byte offset of slot `s`'s PLE mailbox.
    pub fn ple_at(&self, s: usize) -> usize {
        s * self.ple_stride
    }
}

pub struct Driver {
    pub shared: Arc<Shared>,
    pub ple_stats: Arc<crate::ple_io::Stats>,
    thread: Option<std::thread::JoinHandle<()>>,
    ple_thread: Option<std::thread::JoinHandle<()>>,
}

impl Driver {
    /// `exact`: compute the experts in f32 instead of with the pool, for comparisons with the exact reference.
    /// `adapt`: the cache state and policy for adaptive residency, with the ranking the cache was filled from.
    #[allow(clippy::too_many_arguments)]
    pub fn start(layers: Vec<LayerPtrs>, ff: usize, n_embd: usize, ple: PleSource, max_ctx: usize, slots: usize, cpus: Vec<usize>, exact: bool, adapt: Option<(CacheState, SwapConfig, Vec<(usize, usize)>)>) -> Self {
        let out_count_off = (max_ctx * 4).next_multiple_of(64);
        let out_stride = out_count_off + 64;
        let ple_stride = (PLE_EMB + PLE_CAP * 4 * n_embd + 64).next_multiple_of(4096);
        let shared = Arc::new(Shared {
            moe: HostMapped::new(mb_bytes(n_embd)),
            moe2: HostMapped::new(mb_bytes(n_embd)),
            ple: HostMapped::new(slots * ple_stride),
            ple_stride,
            out: HostMapped::new(slots * out_stride),
            out_stride,
            out_count_off,
            seqs: (0..slots).map(|_| SeqShared::default()).collect(),
            stop: AtomicBool::new(false),
            hits: Default::default(),
            routed: Default::default(),
            swaps: Default::default(),
            bulk: Default::default(),
            cpu_ns: Default::default(),
            cpu_jobs: Default::default(),
            cpu_misses: Default::default(),
            lend_req: Default::default(),
            lend_state: Default::default(),
            lend_lo: Default::default(),
            lend_hi: Default::default(),
        });
        let ple_stats = Arc::new(crate::ple_io::Stats { rows: Default::default(), cache_hits: Default::default(), reads: Default::default() });
        let (sh, st) = (shared.clone(), ple_stats.clone());
        let ple_thread = std::thread::spawn(move || crate::ple_io::run(sh, ple.table, ple.cfg, n_embd, ple.cache_rows, st));
        let sh = shared.clone();
        let thread = std::thread::spawn(move || run(sh, layers, ff, n_embd, cpus, exact, adapt));
        Self { shared, ple_stats, thread: Some(thread), ple_thread: Some(ple_thread) }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        if let Some(t) = self.ple_thread.take() {
            let _ = t.join();
        }
    }
}

/// f32 experts: row k of y = down_e(silu(gate_e x) * up_e x), every row decoded and dotted in f32.
fn exact_experts(lp: &LayerPtrs, ids: &[i32], kidx: &[i32], x: &[f32], ff: usize, y: &mut [f32]) {
    let gb = lp.gate_bytes;
    use qx_pack::ggml::GgmlType;
    let (gt, dt) = match lp.fmt {
        GateFmt::Iq3s => (GgmlType::IQ3_S, GgmlType::IQ4_NL),
        GateFmt::Iq2s => (GgmlType::IQ2_S, GgmlType::IQ4_NL),
        GateFmt::Nvfp4 => (GgmlType::NVFP4, GgmlType::NVFP4),
    };
    let n = x.len();
    let (mut row, mut row2) = (vec![0f32; n], vec![0f32; ff]);
    for (&e, &k) in ids.iter().zip(kidx) {
        let e = e as usize;
        let sc = lp.scales(e);
        let mut h = vec![0f32; ff];
        for r in 0..ff {
            let g = unsafe { std::slice::from_raw_parts(lp.rec(e).add(r * lp.gate_row_bytes), lp.gate_row_bytes) };
            let u = unsafe { std::slice::from_raw_parts(lp.rec(e).add(gb + r * lp.gate_row_bytes), lp.gate_row_bytes) };
            qx_pack::dequant(gt, g, &mut row);
            let gv = qx_ref::weights::dot(&row, x) * sc[0];
            qx_pack::dequant(gt, u, &mut row);
            let uv = qx_ref::weights::dot(&row, x) * sc[1];
            h[r] = qx_cpu::math::qx_silu_mul(gv, uv);
        }
        for r in 0..n {
            let d = unsafe { std::slice::from_raw_parts(lp.rec(e).add(2 * gb + r * lp.down_row_bytes), lp.down_row_bytes) };
            qx_pack::dequant(dt, d, &mut row2);
            y[k as usize * n + r] = qx_ref::weights::dot(&row2, &h) * sc[2];
        }
    }
}

/// Answers the MoE job `seq` posted in mailbox `mb`: the misses through the pool (or in f32), then the swapper's
/// bookkeeping.
#[allow(clippy::too_many_arguments)]
fn serve(sh: &Shared, mb: &HostMapped, seq: u64, layers: &[LayerPtrs], pool: &Pool, ff: usize, n: usize, exact: bool, swapper: Option<&mut Swapper>) {
    let n_layer = layers.len();
    let l = *mb.slice::<u32>(MB_LAYER, 1).first().unwrap() as usize;
    let nm = *mb.slice::<u32>(MB_NMISS, 1).first().unwrap() as usize;
    let ntok = (*mb.slice::<u32>(MB_NTOK, 1).first().unwrap() as usize).max(1);
    let kpt = *mb.slice::<u32>(MB_KPT, 1).first().unwrap() as usize;
    let ids: Vec<i32> = mb.slice::<i32>(MB_IDS, nm).to_vec();
    let kidx: Vec<i32> = mb.slice::<i32>(MB_KIDX, nm).to_vec();
    let lp = layers[l];
    let y = mb.slice::<f32>(mb_y(n), AMAX * n);
    let tj = std::time::Instant::now();
    // one pool run for the whole window: each miss reads its token's input
    let xs: Vec<Vec<f32>> = (0..ntok).map(|t| mb.slice::<f32>(MB_X + 4 * t * n, n).to_vec()).collect();
    if exact {
        for t in 0..ntok {
            let mine: Vec<usize> = (0..nm).filter(|&i| kidx[i] as usize / kpt.max(1) == t).collect();
            let (tids, tk): (Vec<i32>, Vec<i32>) = mine.iter().map(|&i| (ids[i], kidx[i])).unzip();
            exact_experts(&lp, &tids, &tk, &xs[t], ff, y);
        }
    } else if nm > 0 {
        let experts: Vec<Expert> = ids
            .iter()
            .zip(&kidx)
            .map(|(&e, &a)| unsafe {
                let r = lp.rec(e as usize);
                Expert { gate: r, up: r.add(lp.gate_bytes), down: r.add(2 * lp.gate_bytes), out: a as usize, tok: a as usize / kpt.max(1), scale: lp.scales(e as usize) }
            })
            .collect();
        let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        pool.run_multi(lp.fmt, &xr, ff, lp.gate_row_bytes, lp.down_row_bytes, &experts, y);
    }
    mb.atomic_u64(MB_DONE).store(seq, Ordering::Release);
    sh.cpu_ns.fetch_add(tj.elapsed().as_nanos() as u64, Ordering::Relaxed);
    sh.cpu_jobs.fetch_add(1, Ordering::Relaxed);
    sh.cpu_misses.fetch_add(nm as u64, Ordering::Relaxed);
    let sel = (ntok * kpt) as u64;
    sh.routed.fetch_add(sel, Ordering::Relaxed);
    sh.hits.fetch_add(sel - nm as u64, Ordering::Relaxed);
    if let Some(sw) = swapper {
        let all: Vec<i32> = mb.slice::<i32>(MB_ALL, ntok * kpt).to_vec();
        sw.observe(l, &all);
        if l == n_layer - 1 {
            if let Some(counts) = sh.bulk.lock().unwrap().take() {
                sw.observe_bulk(&counts);
            }
            let k = sw.end_token(layers, ntok);
            sh.swaps.fetch_add(k as u64, Ordering::Relaxed);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(sh: Arc<Shared>, layers: Vec<LayerPtrs>, ff: usize, n: usize, cpus: Vec<usize>, exact: bool, adapt: Option<(CacheState, SwapConfig, Vec<(usize, usize)>)>) {
    qx_gpu::mem::bind();
    let pool = Pool::new(&cpus);
    let mut swapper = adapt.map(|(cs, cfg, ranking)| Swapper::new(cs, cfg, &ranking));
    let mut last_job = [0u64; 2];
    let mut next_mb = 0;
    let mut idle = 0u32;
    loop {
        if sh.stop.load(Ordering::Relaxed) {
            if let Some(sw) = &swapper {
                // short runs (the checks) leave the saved ranking alone
                if sw.ticks >= 4096 {
                    sw.save();
                }
            }
            return;
        }
        let mut worked = false;
        // the prompt buffers borrow the cache's tail, or give it back
        let want = sh.lend_req.load(Ordering::Acquire);
        if want != sh.lend_state.load(Ordering::Relaxed) {
            if let Some(sw) = swapper.as_mut() {
                if want == 1 {
                    sw.lend(sh.lend_lo.load(Ordering::Relaxed), sh.lend_hi.load(Ordering::Relaxed));
                } else {
                    sw.reclaim(&layers);
                }
            }
            sh.lend_state.store(want, Ordering::Release);
            worked = true;
        }
        // MoE jobs: the mailbox not answered last first, so a split window's halves alternate
        for i in 0..2 {
            let h = (next_mb + i) % 2;
            let mb = if h == 0 { &sh.moe } else { &sh.moe2 };
            let seq = mb.atomic_u64(MB_SEQ).load(Ordering::Acquire);
            if seq != last_job[h] {
                last_job[h] = seq;
                serve(&sh, mb, seq, &layers, &pool, ff, n, exact, swapper.as_mut());
                next_mb = 1 - h;
                worked = true;
                break;
            }
        }
        if worked {
            idle = 0;
        } else {
            idle += 1;
            if idle < 2_000_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(20));
            }
        }
    }
}
