//! The DeepSeek driver thread: answers the MoE mailbox the GPU posts each layer (`qx_moe_post`), computing the
//! missed experts on the CPU pool from their RAM places or, through the expert store's staging, from NVMe; and
//! between tokens promotes the hottest missed experts into VRAM (over the coldest residents of their size
//! class, quarantining the freed slots for two tokens so no in-flight job reads a slot being refilled) and into
//! RAM (over the coldest RAM records). The Qwen engine's `Swapper` policy, with a third tier underneath.
use crate::mailbox::{self, MB_DONE, MB_IDS, MB_KIDX, MB_KPT, MB_LAYER, MB_NMISS, MB_NTOK, MB_SEQ, MB_X};
use crate::tiers::ExpertStore;
use qx_cpu::pool::{Expert, ExpertFmt, Pool};
use qx_gpu::mem::HostMapped;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// What the GPU side shares with the driver.
pub struct Shared {
    pub mb: HostMapped,
    pub stop: AtomicBool,
    /// The arena's flex tail size (0: none, and then nothing need hold it).
    pub flex_bytes: usize,
    /// The chunk path's request for the arena's flex tail: wanted (a prompt runs in chunks) or not; the request
    /// counter and the driver's acknowledgement of the last applied request.
    pub flex_want: AtomicBool,
    pub flex_req: AtomicU64,
    pub flex_ack: AtomicU64,
    pub vram_hits: AtomicU64,
    pub ram_hits: AtomicU64,
    pub nvme_misses: AtomicU64,
    pub vram_promos: AtomicU64,
    pub ram_promos: AtomicU64,
    /// RAM copies of VRAM residents evicted to make room (`QX_DS_RAM_DEDUP`).
    pub ram_dedups: AtomicU64,
    pub pool_ns: AtomicU64,
    /// Of `pool_ns`, the part spent fetching records (NVMe reads) before the pool runs.
    pub fetch_ns: AtomicU64,
    pub promo_ns: AtomicU64,
    /// Selections the GPU computed in place from device-mapped RAM.
    pub mapped_hits: AtomicU64,
    /// Selections staged into VRAM by DMA for the GPU, and the time the driver waited for those copies after
    /// its pool share.
    pub staged_hits: AtomicU64,
    /// How often a layer's routed experts were routed by the previous token too (the premise of a speculative
    /// prefetch): all selections, and the selections not served from VRAM (the ones a prefetch could have hidden).
    pub spec_hits: AtomicU64,
    pub spec_total: AtomicU64,
    pub spec_miss_hits: AtomicU64,
    pub spec_miss_total: AtomicU64,
    /// The per-layer timeline of `serve` (host side): calls, nanoseconds reading the mailbox and scoring, issuing the
    /// staged copies, the whole call, and the gap from the previous call's end to this call's start (the GPU's
    /// time between two jobs, as the host sees it).
    pub tl_calls: AtomicU64,
    pub tl_read_ns: AtomicU64,
    pub tl_issue_ns: AtomicU64,
    pub tl_total_ns: AtomicU64,
    pub tl_gap_ns: AtomicU64,
    /// Router prediction scoring (`QX_DS_PREDICT`): of every routed selection, how many the layer's prediction's top
    /// 6 / 8 / 12 contained; the same over the host-served selections (pool + staged), the ones a prefetch could hide.
    pub pred_hits: [AtomicU64; 3],
    pub pred_total: AtomicU64,
    pub pred_miss_hits: [AtomicU64; 3],
    pub pred_miss_total: AtomicU64,
    /// The prefetch: speculative copies issued, selections the post kernel served from them, NVMe records read early.
    /// Staged selections that fell back to the pool because their record left RAM between the post and the serve.
    pub stage_fallbacks: AtomicU64,
    pub pre_issued: AtomicU64,
    pub pre_used: AtomicU64,
    pub pre_nvme: AtomicU64,
    pub pre_skipped: AtomicU64,
    pub stage_ns: AtomicU64,
    /// Chunk jobs, for the chunk's verbose line (ns and counts): [time inside serve_chunk, of it waiting for a ring
    /// slot, of it waiting for NVMe reads, records staged from RAM, records staged from NVMe, time in the copy
    /// calls, waiting for landing places, in the read submits, picks skipped].
    pub chunk_lane: [AtomicU64; 9],
    /// The routing of the chunk layer whose prefetch job has generation `chunk_used_gen`: which experts have rows
    /// (the engine publishes it as soon as it has the lists; the prefetch job then skips the picks without rows).
    pub chunk_used: std::sync::Mutex<(u64, Vec<bool>)>,
    pub chunk_used_gen: AtomicU64,
    /// Set by the host after each token (the sampled token is known): the driver may run its promotions.
    pub tokens_done: AtomicU64,
    /// A request for the live ranking (every expert, hottest first: VRAM residents, then RAM, then NVMe, each by
    /// decayed score), answered by the driver into `profile`.
    pub profile_req: AtomicBool,
    pub profile: std::sync::Mutex<Option<Vec<(usize, usize)>>>,
    /// Held by the engine thread while it records a CUDA graph. CubeCL captures in global mode, under which a
    /// `cuEventSynchronize` or `cuEventQuery` from any other thread fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
    /// and invalidates the capture (seen in the server's warm-up, where nine widths are captured back to back while
    /// the previous window's promotion copies are still in flight). The driver takes this lock around each such
    /// call; a capture never waits for the driver, so neither side can deadlock.
    pub capture: std::sync::Mutex<()>,
    /// The chunk path (`DsEngine::forward_chunk`): one job per layer, posted by the engine and served here, which
    /// writes each expert's slot word (its VRAM address, or the staging slot its record is copied into) behind the
    /// copy; the GPU's per-expert launches wait on the word. `chunk_done` is the host-mapped word the GPU marks
    /// with the ordinal of the last expert it finished (release), which gates the staging slots' reuse.
    pub chunk_jobs: std::sync::Mutex<std::collections::VecDeque<ChunkJob>>,
    pub chunk_done: HostMapped,
    /// The experts a prefetch job picked (see `ChunkJob::prefetch`), in copy order.
    pub chunk_pick: std::sync::Mutex<Option<Vec<u32>>>,
    /// Per global expert: 0 in VRAM, 1 in RAM, 2 on NVMe (kept by the driver; the engine orders a chunk's experts
    /// by it so the GPU meets the fast ones first).
    pub tier_of: Vec<std::sync::atomic::AtomicU8>,
}

/// A chunk's MoE for one layer: the distinct experts in the order the GPU's launches wait for them, each with its
/// selection count (for the scores), and the device table (u64 per expert of the layer) of slot words to fill.
pub struct ChunkJob {
    pub layer: usize,
    /// The experts in the order the GPU's launches wait for them. Empty with `prefetch` > 0: the driver picks up
    /// to that many of the layer's hottest non-resident experts itself (the engine does not know the scores) and
    /// reports the list in `Shared::chunk_pick`; the engine launches them in that order, skipping (but still
    /// marking) the ones the routing did not select.
    pub experts: Vec<u32>,
    pub prefetch: usize,
    /// Per expert of `experts`: its selection count, for the scores (empty: no scoring).
    pub counts: Vec<u32>,
    /// Further (expert, count) pairs to score without copying (the prefetched experts the routing used).
    pub extra_scores: Vec<(u32, u32)>,
    pub slots_dev: u64,
    /// The job's generation (1..65535, in the words' top 16 bits): the GPU waits for it, so the table is never
    /// zeroed between jobs.
    pub generation: u64,
    /// The ordinal of the job's first expert (global, monotonic across jobs): the GPU marks `ord0 + i + 1` in
    /// `chunk_done` after expert i, and the staging ring frees a slot once the mark passed its last user.
    pub ord0: u64,
}

/// Per layer: the expert formats and record geometry.
#[derive(Clone, Copy)]
pub struct LayerFmt {
    pub fmt: ExpertFmt,
    pub gate_bytes: usize,
    pub gate_row_bytes: usize,
    pub down_row_bytes: usize,
}

/// The driver's own state: the tiers' host side.
pub struct Host {
    pub store: ExpertStore,
    /// Device table (u64 per global expert) and its host mirror.
    pub table_dev: u64,
    pub table: Vec<u64>,
    /// Device table of the RAM-resident experts' mapped addresses (u64 per global expert, 0: not in RAM), which the
    /// post kernel hands a share of to the GPU's expert kernels; its host mirror.
    pub rtable_dev: u64,
    pub rtable: Vec<u64>,
    /// Whether the GPU reads RAM records in place (then evicted places are quarantined two ticks).
    pub gpu_reads_ram: bool,
    /// The VRAM staging area for DMA-staged experts (`stage_cap` records of the largest size, reused per job)
    /// and the device array the staged slot addresses go to.
    pub stage_base: u64,
    pub stage_cap: usize,
    /// The prefetch's speculative slots (`pre_slots` per layer parity, records of `rec_max` bytes) and the device
    /// table of prestaged addresses the post kernel reads (`QX_DS_PREFETCH`).
    pub pred_stage_base: u64,
    pub pre_slots: usize,
    pub prestaged_dev: u64,
    pub rec_max: usize,
    pub slots_stg_dev: u64,
    pub slots_all_dev: u64,
    /// The staged-ready words (u64 per stage stream; `stage_ready_by_words`): the job's sequence number is written
    /// into each behind that stream's copies, and `qx_moe_wait_staged2` polls them.
    pub stg_ready_dev: u64,
    pub arena_ptr: u64,
    /// The arena's flex tail (address, bytes; 0 bytes: none): the chunk path's buffers live there during a prompt,
    /// and the records there leave the VRAM table for the prompt's duration (`flex_check`).
    pub flex_ptr: u64,
    pub flex_bytes: usize,
    /// VRAM slots by size class: (bytes, free addresses); the holder of each slot.
    pub classes: Vec<(usize, Vec<u64>)>,
    pub layer_class: Vec<usize>,
    pub holder: std::collections::HashMap<u64, usize>,
    pub score: Vec<f64>,
    pub decay: f64,
    pub promos_per_token: usize,
    pub margin: f64,
    pub adaptive: bool,
    pub layers: Vec<LayerFmt>,
    pub n_expert: usize,
    pub n_embd: usize,
    pub ff: usize,
}

pub struct Driver {
    pub shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Driver {
    pub fn start(host: Host, cpus: Vec<usize>) -> Self {
        let shared = Arc::new(Shared {
            mb: HostMapped::new(mailbox::mb_bytes(host.n_embd)),
            stop: AtomicBool::new(false),
            flex_want: AtomicBool::new(false),
            flex_bytes: host.flex_bytes,
            flex_req: AtomicU64::new(0),
            flex_ack: AtomicU64::new(0),
            vram_hits: Default::default(),
            ram_hits: Default::default(),
            nvme_misses: Default::default(),
            vram_promos: Default::default(),
            ram_promos: Default::default(),
            ram_dedups: Default::default(),
            pool_ns: Default::default(),
            fetch_ns: Default::default(),
            promo_ns: Default::default(),
            mapped_hits: Default::default(),
            staged_hits: Default::default(),
            spec_hits: Default::default(),
            spec_total: Default::default(),
            spec_miss_hits: Default::default(),
            spec_miss_total: Default::default(),
            stage_ns: Default::default(),
            chunk_lane: Default::default(),
            chunk_used: std::sync::Mutex::new((0, Vec::new())),
            chunk_used_gen: AtomicU64::new(0),
            tl_calls: Default::default(),
            tl_read_ns: Default::default(),
            tl_issue_ns: Default::default(),
            tl_total_ns: Default::default(),
            tl_gap_ns: Default::default(),
            pred_hits: Default::default(),
            pred_total: Default::default(),
            pred_miss_hits: Default::default(),
            pred_miss_total: Default::default(),
            stage_fallbacks: Default::default(),
            pre_issued: Default::default(),
            pre_used: Default::default(),
            pre_nvme: Default::default(),
            pre_skipped: Default::default(),
            tokens_done: Default::default(),
            profile_req: AtomicBool::new(false),
            profile: std::sync::Mutex::new(None),
            capture: std::sync::Mutex::new(()),
            chunk_jobs: std::sync::Mutex::new(Default::default()),
            chunk_done: HostMapped::new(64),
            chunk_pick: std::sync::Mutex::new(None),
            tier_of: (0..host.table.len()).map(|g| std::sync::atomic::AtomicU8::new(if host.table[g] != 0 { 0 } else if host.rtable[g] != 0 { 1 } else { 2 })).collect(),
        });
        let sh = shared.clone();
        // the thread finishes its CUDA set-up (streams, events, pinned words) before any window can post a job:
        // a host allocation that synchronizes with the device would otherwise deadlock against the first post
        // kernel spinning on the mailbox
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || run(sh, host, cpus, ready_tx));
        let _ = ready_rx.recv();
        Self { shared, thread: Some(thread) }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Loop {
    host: Host,
    pool: Pool,
    /// For the tier mirror (`note_tier`); set once the loop runs.
    shared: Option<Arc<Shared>>,
    /// The chunk path's staging ring: the next slot to fill and, per slot, the ordinal of its last user (the GPU's
    /// mark in `chunk_done` must pass it before the slot is refilled). Persists across jobs.
    ring_next: usize,
    ring_user: Vec<u64>,
    stream: cudarc::driver::sys::CUstream,
    /// Pinned words the table writes copy from (a ring, never reused in flight).
    words: HostMapped,
    word_next: usize,
    missed: Vec<usize>,
    /// The experts each layer routed for the previous token (or window): the speculative-prefetch guess.
    last_sel: Vec<Vec<i32>>,
    /// (free at tick, class, address).
    quarantine: std::collections::VecDeque<(u64, usize, u64)>,
    ticks: u64,
    promoted_tokens: u64,
    /// The VRAM copies issued at the last token boundary: the event that marks their completion and the RAM
    /// records they read from (not to be evicted before it).
    copy_event: cudarc::driver::sys::CUevent,
    inflight: Vec<usize>,
    /// Whether the flex tail is given to the chunk path; while it is, the tail's empty slots (class, address) and
    /// the records evicted from it (class, address, record), copied back at release if still in RAM.
    flex_on: bool,
    flex_slots: Vec<(usize, u64)>,
    flex_evicted: Vec<(usize, u64, usize)>,
    /// RAM promotion candidates left from the last boundary, hottest first, done one at a time between jobs.
    pending_ram: Vec<usize>,
    /// RAM places taken from evicted records, free again two ticks later (a window in flight may read them in
    /// place): (free at tick, class, place).
    ram_quarantine: std::collections::VecDeque<(u64, usize, usize)>,
    /// VRAM victims without a RAM copy, waiting for a RAM place (`QX_DS_RAM_DEDUP`).
    wb: Vec<usize>,
    /// Victims being copied from their VRAM slot into a RAM place: (expert, place), adopted when the copies land.
    d2h: Vec<(usize, usize)>,
    /// The staging copies' streams (round robin: one copy engine queue is 14 GB/s, several approach the link) and
    /// an event per stream.
    stage_streams: Vec<cudarc::driver::sys::CUstream>,
    stage_events: Vec<cudarc::driver::sys::CUevent>,
    /// The chunk path's NVMe batches overlap the RAM copies (`chunk_overlap`): an event per stream recorded behind
    /// each batch's copies, up to eight sets used in turn (batch j records set j % nsets), and whether a set has
    /// copies that were not waited for. A class's landing places are handed out in rotation and a batch takes at
    /// most 1/nsets of them, so batch j shares no place with the nsets - 1 batches before it; before its reads are
    /// submitted the set of batch j - nsets is waited for (the streams are FIFO: every earlier copy is then done
    /// too). `chunk_batch_no` is j.
    chunk_events: [Vec<cudarc::driver::sys::CUevent>; 8],
    chunk_events_pending: [bool; 8],
    chunk_batch_no: usize,
    /// A chunk job staged copies since the last `chunk_settle`.
    chunk_staged: bool,
    /// Whether the GPU learns of the staged records' landing from the ready words (`stage_ready_by_words`) rather
    /// than from a done word this thread posts after waiting for the copies (and after the pool's share).
    stg_words: bool,
    /// When the previous `serve` call returned (for the timeline's gap).
    last_serve_end: Option<std::time::Instant>,
    /// The prefetch: predicted experts per token to fetch (0: off), the last prediction header seen, and per layer
    /// parity the live speculative slots (record, slot) and the free ones.
    pref_k: usize,
    pref_cap: usize,
    pref_last: u64,
    /// The speculative copies' own stream and event, whether a batch is still in flight, and the records it reads
    /// (their RAM places must not be evicted until it lands).
    pre_stream: cudarc::driver::sys::CUstream,
    pre_event: cudarc::driver::sys::CUevent,
    pre_busy: bool,
    pre_inflight: Vec<usize>,
    pre_live: [Vec<(usize, u64)>; 2],
    pre_free: [Vec<u64>; 2],
}

impl Loop {
    fn write_entry(&mut self, g: usize, v: u64) {
        let i = self.word_next;
        self.word_next = (self.word_next + 1) % (self.words.bytes / 8);
        self.words.slice::<u64>(i * 8, 1)[0] = v;
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(self.host.table_dev + 8 * g as u64, self.words.ptr.add(i * 8) as *const _, 8, self.stream).result().expect("table write");
        }
        self.host.table[g] = v;
        self.note_tier(g);
    }

    /// Keeps `Shared::tier_of` in step with the tables.
    fn note_tier(&self, g: usize) {
        if let Some(sh) = &self.shared {
            sh.tier_of[g].store(if self.host.table[g] != 0 { 0 } else if self.host.rtable[g] != 0 { 1 } else { 2 }, Ordering::Relaxed);
        }
    }

    /// Writes one u64 to device memory on the copy stream (from the pinned words ring).
    fn write_word(&mut self, dst: u64, v: u64) {
        let st = self.stream;
        self.write_word_on(st, dst, v);
    }

    /// `write_word` on a given stream.
    fn write_word_on(&mut self, stream: cudarc::driver::sys::CUstream, dst: u64, v: u64) {
        let i = self.word_next;
        self.word_next = (self.word_next + 1) % (self.words.bytes / 8);
        self.words.slice::<u64>(i * 8, 1)[0] = v;
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, self.words.ptr.add(i * 8) as *const _, 8, stream).result().expect("device word write");
        }
    }

    /// Writes the RAM table's entry for `g` (the mapped address of its place, or 0) on the copy stream.
    fn write_rentry(&mut self, g: usize, v: u64) {
        let i = self.word_next;
        self.word_next = (self.word_next + 1) % (self.words.bytes / 8);
        self.words.slice::<u64>(i * 8, 1)[0] = v;
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(self.host.rtable_dev + 8 * g as u64, self.words.ptr.add(i * 8) as *const _, 8, self.stream).result().expect("ram table write");
        }
        self.host.rtable[g] = v;
        self.note_tier(g);
    }

    /// Serves a chunk job: every expert's slot word is written behind what makes it readable. VRAM residents at
    /// once; RAM residents copied into the staging area (a ring of `stage_cap` slots over the copy streams, a
    /// slot reused only once the GPU's progress mark passed its last user); NVMe records read into the store's
    /// landing places in batches of the queue, then copied the same way. The engine ordered the experts VRAM, RAM,
    /// NVMe, so the GPU meets each tier's records as they land.
    fn serve_chunk(&mut self, sh: &Shared, mut job: ChunkJob) {
        let (l, ne) = (job.layer, self.host.n_expert);
        for (&e, &c) in job.experts.iter().zip(&job.counts) {
            self.host.score[l * ne + e as usize] += c as f64;
        }
        for &(e, c) in &job.extra_scores {
            self.host.score[l * ne + e as usize] += c as f64;
            if self.host.table[l * ne + e as usize] == 0 {
                self.missed.push(l * ne + e as usize);
            }
        }
        let n_slots = self.host.stage_cap.max(1);
        if self.ring_user.len() != n_slots {
            self.ring_user = vec![0; n_slots];
            self.ring_next = 0;
        }
        if job.experts.is_empty() && job.prefetch > 0 {
            // the layer's non-resident experts, NVMe ones first (the slow reads must start while the slab passes
            // run; the RAM copies stream after the routing at the link's speed), hottest first within a tier,
            // up to the ring's size
            let mut cand: Vec<usize> = (0..ne).filter(|&e| self.host.table[l * ne + e] == 0).collect();
            let tier = |h: &Host, g: usize| if h.store.in_ram(g) { 1 } else { 0 };
            cand.sort_by(|&a, &b| tier(&self.host, l * ne + a).cmp(&tier(&self.host, l * ne + b)).then(self.host.score[l * ne + b].partial_cmp(&self.host.score[l * ne + a]).unwrap_or(std::cmp::Ordering::Equal)).then(a.cmp(&b)));
            cand.truncate(job.prefetch.min(n_slots));
            if chunk_overlap() && !cand.is_empty() {
                // the NVMe reads and the RAM copies run side by side: a share of the RAM records ahead of each
                // NVMe batch (their copies fill the batch's read time), in the order the GPU will meet them
                let half = (self.host.store.stage_places(l * ne) / 2).clamp(1, chunk_batch());
                let (nv, ram): (Vec<usize>, Vec<usize>) = cand.iter().partition(|&&e| tier(&self.host, l * ne + e) == 0);
                let nb = nv.len().div_ceil(half);
                let share = if nb > 0 { ram.len().div_ceil(nb) } else { ram.len() };
                let mut order = Vec::with_capacity(cand.len());
                for b in 0..nb {
                    order.extend_from_slice(&ram[(b * share).min(ram.len())..((b + 1) * share).min(ram.len())]);
                    order.extend_from_slice(&nv[b * half..((b + 1) * half).min(nv.len())]);
                }
                order.extend_from_slice(&ram[(nb * share).min(ram.len())..]);
                cand = order;
            }
            job.experts = cand.iter().map(|&e| e as u32).collect();
            *sh.chunk_pick.lock().unwrap() = Some(job.experts.clone());
        }
        let rec_max = self.host.classes.iter().map(|c| c.0).max().unwrap_or(0) as u64;
        let mut slot_user = std::mem::take(&mut self.ring_user);
        let mut next_slot = self.ring_next;
        let mut next_stream = 0usize;
        let done = sh.chunk_done.atomic_u64(0);
        let t0 = std::time::Instant::now();
        let (mut n_vram, mut n_ram, mut n_nvme) = (0u64, 0u64, 0u64);
        // the copy of one record from `src` into the next staging slot, its word written behind it
        let stage = |lp: &mut Loop, g: usize, e: u32, src: *const u8, ord: u64, slot_user: &mut Vec<u64>, next_slot: &mut usize, next_stream: &mut usize| {
            let s = *next_slot;
            *next_slot = (*next_slot + 1) % n_slots;
            // the slot's previous user must be done on the GPU (its ordinal marked)
            if done.load(Ordering::Acquire) < slot_user[s] {
                let tw = std::time::Instant::now();
                while done.load(Ordering::Acquire) < slot_user[s] {
                    std::hint::spin_loop();
                }
                sh.chunk_lane[1].fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            slot_user[s] = ord;
            let dst = lp.host.stage_base + s as u64 * rec_max;
            let nb = lp.host.store.record_bytes(g);
            let si = *next_stream % lp.stage_streams.len();
            *next_stream += 1;
            let tc = std::time::Instant::now();
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src as *const _, nb, lp.stage_streams[si]).result().expect("chunk stage copy");
            }
            let st = lp.stage_streams[si];
            lp.write_word_on(st, job.slots_dev + 8 * e as u64, dst | (job.generation << 48));
            sh.chunk_lane[5].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
        };
        // (two consecutive batches must not share a landing place: a class with a single one keeps the old form)
        let places = self.host.store.stage_places(l * ne);
        if chunk_overlap() && places >= 2 {
            // The next batch of NVMe records is read while the records before it are staged: its reads are
            // submitted as soon as the previous batch's copies are enqueued.
            let n = job.experts.len();
            // (a pick the routing left without rows is skipped once the engine has published the routing: its
            // ordinal is still marked by the engine, and nothing reads its word)
            let can_skip = job.counts.is_empty() && chunk_skip();
            let mut skip: Vec<bool> = Vec::new();
            let mut staged = vec![false; n];
            let is_nvme = |lp: &Loop, i: usize| { let g = l * ne + job.experts[i] as usize; lp.host.table[g] == 0 && !lp.host.store.in_ram(g) };
            let mut inflight: Option<(Vec<usize>, Vec<*const u8>, Vec<(usize, usize)>)> = None;
            let mut more_nvme = true;
            // (at most half the class's places, and within the store's read queue)
            let half = (places / 2).clamp(1, chunk_batch());
            // Batch j lands in places no batch after j - places / half used: before its reads go out, the copies
            // of batch j - nsets (and so of every older one) must have left theirs. Event set j % nsets is that
            // batch's, and then this one's (`QX_DS_CHUNK_SETS=2`: the two-set form).
            let nsets = (places / half).clamp(2, chunk_sets());
            // the next batch of NVMe records at or after `from` (their job indices and record ids): up to the
            // first record after them that is staged from RAM, skipped picks passed over
            let next_run = |lp: &Loop, from: usize, skip: &Vec<bool>| -> Option<(Vec<usize>, Vec<usize>)> {
                let skipped = |j: usize| !skip.is_empty() && skip[j];
                let mut j = from;
                while j < n && (skipped(j) || !is_nvme(lp, j)) {
                    j += 1;
                }
                if j == n {
                    return None;
                }
                let (mut idxs, mut batch): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
                while j < n && batch.len() < half && (skipped(j) || is_nvme(lp, j)) {
                    if !skipped(j) {
                        idxs.push(j);
                        batch.push(l * ne + job.experts[j] as usize);
                    }
                    j += 1;
                }
                Some((idxs, batch))
            };
            let mut nv_from = 0usize;
            let mut i = 0usize;
            while i < n {
                if can_skip && skip.is_empty() && sh.chunk_used_gen.load(Ordering::Acquire) == job.generation {
                    // (under the lock: the engine may have gone on to the next layer's routing since the load)
                    let used = sh.chunk_used.lock().unwrap();
                    if used.0 == job.generation {
                        skip = job.experts.iter().map(|&e| !used.1[e as usize]).collect();
                    }
                }
                if inflight.is_none() && more_nvme {
                    match next_run(self, i.max(nv_from), &skip) {
                        None => more_nvme = false,
                        Some((idxs, batch)) => {
                            // every batch but the last staged one must have left its landing places
                            let tc = std::time::Instant::now();
                            self.chunk_wait_set(self.chunk_batch_no % nsets);
                            sh.chunk_lane[6].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            let tc = std::time::Instant::now();
                            let (ptrs, jobs) = self.host.store.fetch_submit(&batch).expect("chunk NVMe fetch");
                            sh.chunk_lane[7].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            nv_from = idxs[idxs.len() - 1] + 1;
                            inflight = Some((idxs, ptrs, jobs));
                        }
                    }
                }
                let e = job.experts[i];
                let g = l * ne + e as usize;
                let ord = job.ord0 + i as u64 + 1;
                if inflight.as_ref().is_some_and(|f| f.0[0] == i) {
                    let (idxs, ptrs, jobs) = inflight.take().unwrap();
                    let tw = std::time::Instant::now();
                    self.host.store.fetch_wait(&jobs).expect("chunk NVMe fetch");
                    sh.fetch_ns.fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    sh.chunk_lane[2].fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    for (bi, &ix) in idxs.iter().enumerate() {
                        let ej = job.experts[ix];
                        stage(self, l * ne + ej as usize, ej, ptrs[bi], job.ord0 + ix as u64 + 1, &mut slot_user, &mut next_slot, &mut next_stream);
                        staged[ix] = true;
                    }
                    let p = self.chunk_batch_no % nsets;
                    for si in 0..self.stage_streams.len() {
                        unsafe { cudarc::driver::result::event::record(self.chunk_events[p][si], self.stage_streams[si]).expect("event record") };
                    }
                    self.chunk_events_pending[p] = true;
                    self.chunk_batch_no += 1;
                    n_nvme += idxs.len() as u64;
                    i += 1;
                } else if staged[i] {
                    i += 1;
                } else if !skip.is_empty() && skip[i] {
                    sh.chunk_lane[8].fetch_add(1, Ordering::Relaxed);
                    i += 1;
                } else if self.host.table[g] != 0 {
                    let v = self.host.table[g] | (job.generation << 48);
                    self.write_word(job.slots_dev + 8 * e as u64, v);
                    n_vram += 1;
                    i += 1;
                } else if self.host.store.in_ram(g) {
                    let src = self.host.store.place[g] as *const u8;
                    stage(self, g, e, src, ord, &mut slot_user, &mut next_slot, &mut next_stream);
                    n_ram += 1;
                    i += 1;
                } else {
                    panic!("an NVMe record with no read in flight");
                }
            }
            assert!(inflight.is_none());
        } else {
        // NVMe records: the ones not in RAM when their turn comes, read in batches of the landing places
        let mut i = 0usize;
        while i < job.experts.len() {
            let e = job.experts[i];
            let g = l * ne + e as usize;
            let ord = job.ord0 + i as u64 + 1;
            if self.host.table[g] != 0 {
                let v = self.host.table[g] | (job.generation << 48);
                self.write_word(job.slots_dev + 8 * e as u64, v);
                n_vram += 1;
                i += 1;
            } else if self.host.store.in_ram(g) {
                let src = self.host.store.place[g] as *const u8;
                stage(self, g, e, src, ord, &mut slot_user, &mut next_slot, &mut next_stream);
                n_ram += 1;
                i += 1;
            } else {
                // a batch of consecutive NVMe records from here (the engine put them last)
                let mut batch: Vec<usize> = Vec::new();
                let mut j = i;
                while j < job.experts.len() && batch.len() < 48 {
                    let gj = l * ne + job.experts[j] as usize;
                    if self.host.table[gj] != 0 || self.host.store.in_ram(gj) {
                        break;
                    }
                    batch.push(gj);
                    j += 1;
                }
                let tw = std::time::Instant::now();
                let ptrs = self.host.store.fetch(&batch).expect("chunk NVMe fetch");
                sh.fetch_ns.fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
                for (bi, &gj) in batch.iter().enumerate() {
                    let ej = job.experts[i + bi];
                    stage(self, gj, ej, ptrs[bi], job.ord0 + (i + bi) as u64 + 1, &mut slot_user, &mut next_slot, &mut next_stream);
                }
                // the landing places are reused by the next batch: its reads must not start before these copies read them
                for si in 0..self.stage_streams.len() {
                    unsafe {
                        cudarc::driver::result::event::record(self.stage_events[si], self.stage_streams[si]).expect("event record");
                        cudarc::driver::result::event::synchronize(self.stage_events[si]).expect("chunk stage copies");
                    }
                }
                n_nvme += batch.len() as u64;
                i = j;
            }
        }
        }
        self.chunk_staged |= n_ram + n_nvme > 0;
        self.ring_user = slot_user;
        self.ring_next = next_slot;
        sh.vram_hits.fetch_add(n_vram, Ordering::Relaxed);
        sh.staged_hits.fetch_add(n_ram, Ordering::Relaxed);
        sh.nvme_misses.fetch_add(n_nvme, Ordering::Relaxed);
        sh.chunk_lane[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        sh.chunk_lane[3].fetch_add(n_ram, Ordering::Relaxed);
        sh.chunk_lane[4].fetch_add(n_nvme, Ordering::Relaxed);
        sh.stage_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !job.counts.is_empty() {
            self.missed.extend(job.experts.iter().map(|&e| l * ne + e as usize).filter(|&g| self.host.table[g] == 0));
        }
    }

    /// Waits for the copies behind chunk event set `p` (and so for every staging copy enqueued before them).
    fn chunk_wait_set(&mut self, p: usize) {
        if self.chunk_events_pending[p] {
            for si in 0..self.stage_streams.len() {
                unsafe { cudarc::driver::result::event::synchronize(self.chunk_events[p][si]).expect("chunk stage copies") };
            }
            self.chunk_events_pending[p] = false;
        }
    }

    /// Waits for every staging copy the chunk jobs enqueued (out of the NVMe landing places and into the staging
    /// area): whoever uses either next (a window's misses) may then overwrite them.
    fn chunk_settle(&mut self) {
        if self.chunk_staged {
            for si in 0..self.stage_streams.len() {
                unsafe {
                    cudarc::driver::result::event::record(self.stage_events[si], self.stage_streams[si]).expect("event record");
                    cudarc::driver::result::event::synchronize(self.stage_events[si]).expect("chunk stage copies");
                }
            }
            self.chunk_staged = false;
            self.chunk_events_pending = [false; 8];
        }
    }

    /// Answers one MoE job.
    fn serve(&mut self, sh: &Shared, seq: u64) {
        self.chunk_settle();
        let t_start = std::time::Instant::now();
        if let Some(prev) = self.last_serve_end {
            sh.tl_gap_ns.fetch_add(t_start.duration_since(prev).as_nanos() as u64, Ordering::Relaxed);
        }
        let mb = &sh.mb;
        let n = self.host.n_embd;
        let l = *mb.slice::<u32>(MB_LAYER, 1).first().unwrap() as usize;
        let nm = *mb.slice::<u32>(MB_NMISS, 1).first().unwrap() as usize;
        let nmap = *mb.slice::<u32>(mailbox::MB_NMAP, 1).first().unwrap() as usize;
        let nstage = *mb.slice::<u32>(mailbox::MB_NSTAGE, 1).first().unwrap() as usize;
        if self.pref_k > 0 {
            sh.pre_used.fetch_add(*mb.slice::<u32>(mailbox::MB_NPRE, 1).first().unwrap() as u64, Ordering::Relaxed);
        }
        let ntok = (*mb.slice::<u32>(MB_NTOK, 1).first().unwrap() as usize).max(1);
        let kpt = (*mb.slice::<u32>(MB_KPT, 1).first().unwrap() as usize).max(1);
        let mut ids: Vec<i32> = mb.slice::<i32>(MB_IDS, nm).to_vec();
        let mut kidx: Vec<i32> = mb.slice::<i32>(MB_KIDX, nm).to_vec();
        // every routed expert of this layer scores (the post kernel lists them all)
        let all: Vec<i32> = mb.slice::<i32>(MB_ALL_OFF, ntok * kpt).to_vec();
        let ne = self.host.n_expert;
        for &e in &all {
            self.host.score[l * ne + e as usize] += 1.0;
        }
        {
            // the speculative-prefetch premise: would the previous token's experts of this layer have guessed these?
            if self.last_sel.len() < self.host.layers.len() {
                self.last_sel.resize(self.host.layers.len(), Vec::new());
            }
            let prev = &self.last_sel[l];
            let mut cur: Vec<i32> = all.clone();
            cur.sort_unstable();
            cur.dedup();
            let hits = cur.iter().filter(|e| prev.binary_search(e).is_ok()).count();
            sh.spec_hits.fetch_add(hits as u64, Ordering::Relaxed);
            sh.spec_total.fetch_add(cur.len() as u64, Ordering::Relaxed);
            // the selections the host serves (pool + staged): the ones a prefetch could hide
            let mut miss: Vec<i32> = ids.clone();
            if nstage > 0 {
                miss.extend_from_slice(mb.slice::<i32>(mailbox::MB_SIDS, nstage));
            }
            miss.sort_unstable();
            miss.dedup();
            let mh = miss.iter().filter(|e| prev.binary_search(e).is_ok()).count();
            sh.spec_miss_hits.fetch_add(mh as u64, Ordering::Relaxed);
            sh.spec_miss_total.fetch_add(miss.len() as u64, Ordering::Relaxed);
            self.last_sel[l] = cur;
        }
        if crate::dsv41::predict_mode() != 0 {
            // the layer's predicted top-PRED_K per token against the real selections and the host-served ones
            let pk = mailbox::PRED_K;
            let pred: Vec<i32> = mb.slice::<i32>(mailbox::mb_pred(n, l), ntok * pk).to_vec();
            let ks = [6usize, 8, 12];
            for (tok, chunk) in all.chunks(kpt).enumerate() {
                let p = &pred[tok * pk..(tok + 1) * pk];
                for (i, &k) in ks.iter().enumerate() {
                    let h = chunk.iter().filter(|e| p[..k.min(pk)].contains(e)).count() as u64;
                    sh.pred_hits[i].fetch_add(h, Ordering::Relaxed);
                }
            }
            sh.pred_total.fetch_add(all.len() as u64, Ordering::Relaxed);
            let mut host_served: Vec<(i32, i32)> = ids.iter().copied().zip(kidx.iter().copied()).collect();
            if nstage > 0 {
                host_served.extend(mb.slice::<i32>(mailbox::MB_SIDS, nstage).iter().copied().zip(mb.slice::<i32>(mailbox::MB_SKIDX, nstage).iter().copied()));
            }
            for &(e, a) in &host_served {
                let tok = a as usize / kpt;
                let p = &pred[tok * pk..(tok + 1) * pk];
                for (i, &k) in ks.iter().enumerate() {
                    if p[..k.min(pk)].contains(&e) {
                        sh.pred_miss_hits[i].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            sh.pred_miss_total.fetch_add(host_served.len() as u64, Ordering::Relaxed);
        }
        sh.vram_hits.fetch_add((all.len() - nm - nmap - nstage) as u64, Ordering::Relaxed);
        sh.mapped_hits.fetch_add(nmap as u64, Ordering::Relaxed);
        sh.staged_hits.fetch_add(nstage as u64, Ordering::Relaxed);
        sh.tl_read_ns.fetch_add(t_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_read = std::time::Instant::now();
        // the staged share: each record copied from its RAM place into the staging area on the copy stream, its
        // address written to the staged slot array behind the copy, then the staged-done word once all landed
        // (after the pool share below, so the copies overlap it). The area is reused per job: the GPU finished
        // the previous job's staged kernels before it posted this one.
        let mut staged: Vec<usize> = Vec::new();
        let mut used_streams: Vec<usize> = Vec::new();
        if nstage > 0 {
            let sids: Vec<i32> = mb.slice::<i32>(mailbox::MB_SIDS, nstage).to_vec();
            let skidx: Vec<i32> = mb.slice::<i32>(mailbox::MB_SKIDX, nstage).to_vec();
            let ne = self.host.n_expert;
            let mut off = 0u64;
            // one copy per distinct record (a window selects an expert for several tokens): (slot, its stream)
            let mut placed: std::collections::HashMap<usize, (u64, usize)> = Default::default();
            let mut next_stream = 0usize;
            let mut fallbacks = 0u64;
            for (&e, &a) in sids.iter().zip(&skidx) {
                let g = l * ne + e as usize;
                let (dst, si) = match placed.get(&g) {
                    Some(&p) => p,
                    None => {
                        if !self.host.store.in_ram(g) {
                            // evicted between the post and now: the pool computes it, and the combine takes its row
                            ids.push(e);
                            kidx.push(a);
                            self.write_word(self.host.slots_all_dev + 8 * a as u64, 0);
                            fallbacks += 1;
                            continue;
                        }
                        let nb = self.host.store.record_bytes(g);
                        let dst = self.host.stage_base + off;
                        off += nb as u64;
                        let si = next_stream % self.stage_streams.len();
                        next_stream += 1;
                        if !used_streams.contains(&si) {
                            used_streams.push(si);
                        }
                        let src = self.host.store.place[g] as *const u8;
                        unsafe {
                            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src as *const _, nb, self.stage_streams[si]).result().expect("stage copy");
                        }
                        staged.push(g);
                        placed.insert(g, (dst, si));
                        (dst, si)
                    }
                };
                let st = self.stage_streams[si];
                self.write_word_on(st, self.host.slots_stg_dev + 8 * a as u64, dst);
            }
            if self.stg_words {
                // the ready words behind the copies, on every stage stream (the GPU waits for all of them)
                for si in 0..self.stage_streams.len() {
                    let st = self.stage_streams[si];
                    self.write_word_on(st, self.host.stg_ready_dev + 8 * si as u64, seq);
                }
            } else {
                for &si in &used_streams {
                    unsafe { cudarc::driver::result::event::record(self.stage_events[si], self.stage_streams[si]).expect("event record") };
                }
            }
            self.missed.extend_from_slice(&staged);
            if fallbacks > 0 {
                // The fallback's slot words are asynchronous writes on the copy stream, and the combine kernel reads
                // them as soon as this job is acknowledged: they must have landed first. Until 2026-11-09 nothing
                // waited, and a combine that ran ahead of the write took the GPU's output row for a selection no
                // GPU kernel had computed (stale but deterministic bits: the late-token flip under three streams,
                // found by replaying a server run's schedule, `qx-ds-replay`).
                let _not_capturing = sh.capture.lock().unwrap();
                unsafe { cudarc::driver::result::stream::synchronize(self.stream).expect("fallback slot words") };
                drop(_not_capturing);
                sh.stage_fallbacks.fetch_add(fallbacks, Ordering::Relaxed);
            }
        }
        sh.tl_issue_ns.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if crate::dsv41::layer_trace_on() {
            // the lanes of this layer's host-served selections: (expert, selection index); the rest ran from VRAM
            let st: Vec<(i32, i32)> = if nstage > 0 { mb.slice::<i32>(mailbox::MB_SIDS, nstage).iter().copied().zip(mb.slice::<i32>(mailbox::MB_SKIDX, nstage).iter().copied()).collect() } else { Vec::new() };
            let pl: Vec<(i32, i32)> = ids.iter().copied().zip(kidx.iter().copied()).collect();
            let ram: Vec<bool> = ids.iter().map(|&e| self.host.store.in_ram(l * ne + e as usize)).collect();
            crate::dsv41::sched_trace(&format!("S {l} kpt {kpt} pool {pl:?} inram {ram:?} staged {st:?} all {all:?}"));
        }
        let nm = ids.len();
        if nm > 0 {
            let t0 = std::time::Instant::now();
            let gs: Vec<usize> = ids.iter().map(|&e| l * ne + e as usize).collect();
            let in_ram = gs.iter().filter(|&&g| self.host.store.in_ram(g)).count();
            sh.ram_hits.fetch_add(in_ram as u64, Ordering::Relaxed);
            sh.nvme_misses.fetch_add((gs.len() - in_ram) as u64, Ordering::Relaxed);
            self.missed.extend_from_slice(&gs);
            // the NVMe reads are in flight while the pool computes the experts already in RAM
            let from_ram: Vec<bool> = gs.iter().map(|&g| self.host.store.in_ram(g)).collect();
            let (ptrs, pending) = self.host.store.fetch_submit(&gs).expect("expert fetch");
            let lf = self.host.layers[l];
            let xs: Vec<Vec<f32>> = (0..ntok).map(|t| mb.slice::<f32>(MB_X + 4 * t * n, n).to_vec()).collect();
            let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
            let experts: Vec<Expert> = kidx.iter().zip(&ptrs).map(|(&a, &p)| Expert { gate: p, up: unsafe { p.add(lf.gate_bytes) }, down: unsafe { p.add(2 * lf.gate_bytes) }, out: a as usize, tok: a as usize / kpt, scale: [1.0; 3] }).collect();
            let (mut ram, mut nvme): (Vec<Expert>, Vec<Expert>) = (Vec::new(), Vec::new());
            for (e, &r) in experts.iter().zip(&from_ram) {
                if r { ram.push(*e) } else { nvme.push(*e) }
            }
            let y = mb.slice::<f32>(mailbox::mb_y(n), mailbox::AMAX * n);
            if !ram.is_empty() {
                self.pool.run_multi(lf.fmt, &xr, self.host.ff, lf.gate_row_bytes, lf.down_row_bytes, &ram, y);
            }
            let tw = std::time::Instant::now();
            self.host.store.fetch_wait(&pending).expect("expert fetch");
            sh.fetch_ns.fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if !nvme.is_empty() {
                self.pool.run_multi(lf.fmt, &xr, self.host.ff, lf.gate_row_bytes, lf.down_row_bytes, &nvme, y);
            }
            sh.pool_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if nstage > 0 && !self.stg_words {
            let tw = std::time::Instant::now();
            let _not_capturing = sh.capture.lock().unwrap();
            for &si in &used_streams {
                unsafe { cudarc::driver::result::event::synchronize(self.stage_events[si]).expect("stage copies") };
            }
            drop(_not_capturing);
            sh.stage_ns.fetch_add(tw.elapsed().as_nanos() as u64, Ordering::Relaxed);
            mb.atomic_u64(mailbox::MB_STG_DONE).store(seq, Ordering::Release);
        }
        mb.atomic_u64(MB_DONE).store(seq, Ordering::Release);
        sh.tl_calls.fetch_add(1, Ordering::Relaxed);
        sh.tl_total_ns.fetch_add(t_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.last_serve_end = Some(std::time::Instant::now());
    }

    /// Between tokens: decay, release quarantined slots, promote the hottest misses into VRAM and RAM.
    /// `dt`: tokens since the last call (a window's width): the promotion counts scale with it.
    fn adapt(&mut self, sh: &Shared, dt: usize) {
        let t0 = std::time::Instant::now();
        self.ticks += 1;
        let h = &mut self.host;
        // the half-life is in tokens: a window's call decays `dt` tokens' worth (before 2026-11-06 it decayed once
        // per call, so a 128-token window kept a 128x longer memory than decode; QX_DS_DECAY_PER_STEP=1 restores that)
        static PER_STEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let per_step = *PER_STEP.get_or_init(|| std::env::var("QX_DS_DECAY_PER_STEP").is_ok_and(|v| v == "1"));
        let decay = if per_step { h.decay } else { h.decay.powi(dt.max(1) as i32) };
        for s in h.score.iter_mut() {
            *s *= decay;
        }
        while let Some(&(when, c, a)) = self.quarantine.front() {
            if when > self.ticks {
                break;
            }
            self.quarantine.pop_front();
            self.host.classes[c].1.push(a);
        }
        while let Some(&(when, c, p)) = self.ram_quarantine.front() {
            if when > self.ticks {
                break;
            }
            self.ram_quarantine.pop_front();
            self.host.store.free_place(c, p);
        }
        let mut cand = std::mem::take(&mut self.missed);
        if !self.host.adaptive {
            return;
        }
        cand.sort_unstable();
        cand.dedup();
        let h = &self.host;
        cand.sort_by(|&a, &b| h.score[b].partial_cmp(&h.score[a]).unwrap());
        let ne = h.n_expert;
        let n = (h.promos_per_token * dt.max(1)).min(512);
        // QX_DS_VRAM_PROMOS / QX_DS_RAM_PROMOS override the per-tier counts (diagnostics)
        let knob = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let (n_vram, n_ram) = (knob("QX_DS_VRAM_PROMOS").map_or(n, |v| v * dt.max(1)), knob("QX_DS_RAM_PROMOS").map_or(n, |v| v * dt.max(1)));
        // VRAM: free slots first, then evictions of colder residents (their slots quarantined). Only records in RAM
        // are copied: a staging place could be reused by a later fetch while the asynchronous copy still reads it.
        // NVMe candidates reach RAM below and VRAM at a later token.
        // Two phases: the victims' table entries are zeroed first (small stream writes that land at once), then
        // the copies and the new entries. Queued behind the copies, a zero could land after the slot's two-tick
        // quarantine had expired and the slot been refilled: the GPU would read a slot being overwritten.
        // While the previous tick's copies are still in flight no new ones are issued (the driver never blocks).
        self.finish_copies(sh, false);
        let vram_ok = self.inflight.is_empty() && self.d2h.is_empty();
        let mut plan: Vec<(usize, u64)> = Vec::new();
        // Evictions at a prompt chunk's tick (`dt` over a window's width): a victim's slot is quarantined two ticks,
        // so its candidate enters VRAM only if another slot is free now, and during a prompt's chunks the flex tail
        // holds the free slots: a chunk's tick evicted hundreds of residents and promoted none, and the decode
        // ticks after it refill at the per-token budget. Window and decode ticks are not capped: their bursts are
        // refilled by the next ticks' own budgets (capping them too read prefill128 10.5 -> 15.8 and decode256k
        // 38.6 -> 39.4, 2026-11-11). QX_DS_EVICT_CAP=N (default the per-token budget; 0: no cap, as before).
        static EVICT_CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let evict_cap = if dt <= crate::dsv41::DS_TMAX {
            usize::MAX
        } else {
            match *EVICT_CAP.get_or_init(|| std::env::var("QX_DS_EVICT_CAP").ok().and_then(|v| v.parse::<usize>().ok())) {
                Some(0) => usize::MAX,
                Some(n) => n,
                None => self.host.promos_per_token.max(1),
            }
        };
        let mut evicted_now = 0usize;
        for &g in cand.iter() {
            if !vram_ok || plan.len() >= n_vram {
                break;
            }
            if self.host.table[g] != 0 || !self.host.store.in_ram(g) {
                continue;
            }
            let c = self.host.layer_class[g / ne];
            let addr = match self.host.classes[c].1.pop() {
                Some(a) => a,
                None => {
                    let h = &self.host;
                    let victim = h.holder.iter().filter(|(_, x)| h.layer_class[**x / ne] == c).min_by(|a, b| h.score[*a.1].partial_cmp(&h.score[*b.1]).unwrap()).map(|(a, x)| (*a, *x));
                    let Some((a, v)) = victim else { continue };
                    if h.score[g] <= h.score[v] * (1.0 + h.margin) {
                        continue;
                    }
                    if evicted_now >= evict_cap {
                        continue;
                    }
                    evicted_now += 1;
                    self.write_entry(v, 0);
                    // a victim with no RAM copy (an initial resident, or one whose copy was deduplicated) would be
                    // NVMe-only: queue it for RAM ahead of the candidates (QX_DS_RAM_DEDUP)
                    if ram_dedup() && !self.host.store.in_ram(v) {
                        // Swap: the victim's bytes go from its slot (quarantined two ticks, read here on the copy
                        // stream ahead of any refill) into the RAM place of a duplicate: the hottest VRAM resident
                        // of the class that still has a RAM copy nobody reads (not being copied, not in the flex
                        // tail, whose refill reads it). The victim counts as in RAM once the copy landed
                        // (`finish_copies`). No duplicate (or the GPU reads RAM in place): the NVMe queue.
                        let mut swapped = false;
                        if !self.host.gpu_reads_ram && swap_mode() {
                            let h = &self.host;
                            let (fp, fb) = (h.flex_ptr, h.flex_bytes);
                            let (inflight, pre_inflight, d2h) = (&self.inflight, &self.pre_inflight, &self.d2h);
                            let dup = h.holder.values().copied().filter(|&x| x != v && h.layer_class[x / ne] == c && h.table[x] != 0 && h.store.in_ram(x) && !(fb > 0 && h.table[x] + h.classes[c].0 as u64 > fp) && !inflight.contains(&x) && !pre_inflight.contains(&x) && !d2h.iter().any(|d| d.0 == x)).max_by(|&p, &q| h.score[p].partial_cmp(&h.score[q]).unwrap());
                            if let Some(x) = dup {
                                if let Some((_, p)) = self.host.store.evict_deferred(x) {
                                    self.write_rentry(x, 0);
                                    let nb = self.host.store.record_bytes(v);
                                    unsafe {
                                        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(p as *mut _, a, nb, self.stream).result().expect("victim copy");
                                    }
                                    self.d2h.push((v, p));
                                    sh.ram_dedups.fetch_add(1, Ordering::Relaxed);
                                    swapped = true;
                                }
                            }
                        }
                        if !swapped && self.wb.len() < 512 {
                            self.wb.push(v);
                        }
                    }
                    self.host.holder.remove(&a);
                    self.quarantine.push_back((self.ticks + 2, c, a));
                    // the slot is in quarantine: take a free one if any, else wait for the next token
                    match self.host.classes[c].1.pop() {
                        Some(f) => f,
                        None => continue,
                    }
                }
            };
            self.host.holder.insert(addr, g);
            plan.push((g, addr));
        }
        // The copies and table writes are stream-ordered (the table entry lands after its record), so the next
        // token's post kernel can read the table at once; only the RAM places being read must stay put until the
        // event says the copies are done. The driver serves the next token without waiting.
        let mut into_vram: Vec<usize> = Vec::new();
        for &(g, addr) in &plan {
            let ptr = self.host.store.place[g] as *const u8;
            let nb = self.host.store.record_bytes(g);
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(addr, ptr as *const _, nb, self.stream).result().expect("expert copy");
            }
            self.write_entry(g, addr);
            into_vram.push(g);
            sh.vram_promos.fetch_add(1, Ordering::Relaxed);
        }
        // only when this call issued copies (`vram_ok`): with the previous tick's copies still in flight nothing
        // was issued, and `inflight` must keep naming them (the breaker, 2026-11-10: overwritten with the empty
        // plan, their RAM places passed for unread duplicates and were reused under the queued copies)
        if vram_ok && (!into_vram.is_empty() || !self.d2h.is_empty()) {
            unsafe { cudarc::driver::result::event::record(self.copy_event, self.stream).expect("event record") };
            self.inflight = into_vram;
        }
        // RAM: NVMe records among the candidates, hottest first, promoted one per idle gap by `promote_one`
        let in_vram = &self.inflight;
        self.pending_ram = cand.iter().copied().filter(|&g| !self.host.store.in_ram(g) && !in_vram.contains(&g)).take(n_ram).collect();
        self.pending_ram.reverse(); // pop from the back: hottest first
        if !self.wb.is_empty() {
            let h = &self.host;
            let inflight = &self.inflight;
            // (a place already assigned means the read is pending or done: not queued again)
            self.wb.retain(|&g| h.table[g] == 0 && h.store.place[g] == 0 && !inflight.contains(&g));
            let wb = self.wb.clone();
            self.pending_ram.retain(|g| !wb.contains(g));
            self.pending_ram.extend(wb); // popped before the candidates
        }
        sh.promo_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Applies a pending flex request (`Shared::flex_*`): giving the arena's tail to the chunk path evicts every record
    /// in it (table 0 on host and device, slot out of its class's free list) after the in-flight promotion copies
    /// landed; taking it back returns the tail's slots to the free lists, empty, for the next promotions. The
    /// acknowledgement is written after the device table words landed on this stream.
    fn flex_check(&mut self, sh: &Shared) {
        let req = sh.flex_req.load(Ordering::Acquire);
        if req == sh.flex_ack.load(Ordering::Acquire) {
            return;
        }
        let want = sh.flex_want.load(Ordering::Acquire);
        if want != self.flex_on && self.host.flex_bytes > 0 {
            let fp = self.host.flex_ptr;
            // a slot is in the tail when any of its bytes are (the tail's start cuts through a slot of whichever class
            // lies there: that record would be overwritten while still in the table if only starts were tested)
            let in_tail = |a: u64, bytes: usize| a + bytes as u64 > fp;
            if want {
                // every promotion copy in flight has landed before the tail is written by the chunk path
                self.finish_copies(sh, true);
                let ne = self.host.n_expert;
                let class_bytes: Vec<usize> = self.host.classes.iter().map(|c| c.0).collect();
                let evict: Vec<(u64, usize)> = self.host.holder.iter().filter(|&(&a, &g)| in_tail(a, class_bytes[self.host.layer_class[g / ne]])).map(|(&a, &g)| (a, g)).collect();
                for &(a, g) in &evict {
                    self.write_entry(g, 0);
                    self.host.holder.remove(&a);
                    self.flex_evicted.push((self.host.layer_class[g / ne], a, g));
                }
                for (c, (bytes, free)) in self.host.classes.iter_mut().enumerate() {
                    let bytes = *bytes;
                    let (tail, keep): (Vec<u64>, Vec<u64>) = free.iter().partition(|&&a| in_tail(a, bytes));
                    *free = keep;
                    self.flex_slots.extend(tail.into_iter().map(|a| (c, a)));
                }
                // quarantined tail slots would return to the free lists later: keep them out too
                let (tail_q, keep_q): (Vec<(u64, usize, u64)>, Vec<(u64, usize, u64)>) = self.quarantine.drain(..).partition(|&(_, c, a)| in_tail(a, class_bytes[c]));
                self.quarantine.extend(keep_q);
                self.flex_slots.extend(tail_q.into_iter().map(|(_, c, a)| (c, a)));
                unsafe { cudarc::driver::result::stream::synchronize(self.stream).expect("flex: table words") };
                if std::env::var("QX_DS_VERBOSE").is_ok() {
                    let straddle = evict.iter().filter(|&&(a, _)| a < fp).count();
                    eprintln!("flex: {} records out of the tail ({} slots to the chunk path; {} record(s) straddle its start)", evict.len(), evict.len() + self.flex_slots.len(), straddle);
                }
            } else {
                // the evicted records go straight back into their slots (one to two GB, a few tens of ms on the
                // copy stream, and the decode after a prompt pays nothing: without this, the promotions would
                // refill the tail over hundreds of tokens); the tail's empty slots return to the free lists
                self.finish_copies(sh, true);
                let mut refilled = 0usize;
                let evicted = std::mem::take(&mut self.flex_evicted);
                for &(c, a, g) in &evicted {
                    if self.host.table[g] == 0 && self.host.store.in_ram(g) && !self.host.holder.contains_key(&a) {
                        let ptr = self.host.store.place[g] as *const u8;
                        let nb = self.host.store.record_bytes(g);
                        unsafe {
                            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(a, ptr as *const _, nb, self.stream).result().expect("flex refill copy");
                        }
                        self.write_entry(g, a);
                        self.host.holder.insert(a, g);
                        self.inflight.push(g);
                        refilled += 1;
                    } else {
                        self.host.classes[c].1.push(a);
                    }
                }
                if !self.inflight.is_empty() {
                    unsafe { cudarc::driver::result::event::record(self.copy_event, self.stream).expect("event record") };
                }
                for &(c, a) in &self.flex_slots {
                    self.host.classes[c].1.push(a);
                }
                if std::env::var("QX_DS_VERBOSE").is_ok() {
                    eprintln!("flex: {} records back into the tail, {} slots back to the tiers empty", refilled, evicted.len() - refilled + self.flex_slots.len());
                }
                self.flex_slots.clear();
            }
            self.flex_on = want;
        }
        sh.flex_ack.store(req, Ordering::Release);
    }

    /// The prefetch (`QX_DS_PREFETCH=K`): on a new router prediction for layer `l` (posted by the GPU at the top of
    /// the layer, before its attention), copies the predicted top-K experts per token that are in RAM but not in
    /// VRAM into this parity's speculative slots on the copy stream and writes their addresses into the prestaged
    /// table behind the copies, so `qx_moe_post` finds them resident when the real routing arrives ~0.5 ms later.
    /// The slots of layer `l - 2` (same parity) are recycled first: their kernels ran before the post of `l - 1`,
    /// which precedes this prediction on the GPU's stream; their table entries are cleared (the next lookup of
    /// those experts is the next token's layer `l - 2`). Predicted experts on NVMe only are queued for a RAM
    /// promotion. True when it did anything.
    fn prefetch_check(&mut self, sh: &Shared) -> bool {
        if self.pref_k == 0 || self.host.pre_slots == 0 {
            return false;
        }
        let hdr = sh.mb.atomic_u64(mailbox::MB_PRED_HDR).load(Ordering::Acquire);
        if hdr == self.pref_last {
            return false;
        }
        self.pref_last = hdr;
        // No backlog, ever: a new batch (clears, copies, entries) is issued only once the previous one has fully
        // landed, so everything issued here starts at once. That is what makes recycling safe (the clears land
        // within microseconds, long before any later job can look those experts up; a backlog could let a job
        // find an entry whose slot is then overwritten under it) and it keeps the copies inside the layer's idle
        // bus time. A prediction that arrives while a batch is in flight is skipped: its slots stay as they are,
        // still valid residents.
        if self.pre_busy {
            let _not_capturing = sh.capture.lock().unwrap();
            let done = unsafe { cudarc::driver::result::event::query(self.pre_event).is_ok() };
            drop(_not_capturing);
            if !done {
                sh.pre_skipped.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            self.pre_busy = false;
            self.pre_inflight.clear();
        }
        let l = (hdr & 0xff) as usize;
        let ntok = ((hdr >> 8) & 0xffff) as usize;
        let ne = self.host.n_expert;
        if l >= self.host.layers.len() || ntok == 0 {
            return true;
        }
        let par = l & 1;
        // the predicted experts of this layer, distinct, best-ranked first (rank, then token)
        let pk = mailbox::PRED_K;
        let pred: Vec<i32> = sh.mb.slice::<i32>(mailbox::mb_pred(self.host.n_embd, l), ntok * pk).to_vec();
        let mut want: Vec<usize> = Vec::with_capacity(ntok * self.pref_k);
        for r in 0..self.pref_k.min(pk) {
            for tok in 0..ntok {
                let e = pred[tok * pk + r];
                if e >= 0 && (e as usize) < ne {
                    let g = l * ne + e as usize;
                    if !want.contains(&g) {
                        want.push(g);
                    }
                }
            }
        }
        // keep what this parity already holds for this layer (an entry from an earlier token is a valid resident);
        // recycle the rest, clears first
        let live = std::mem::take(&mut self.pre_live[par]);
        let mut issued_any = false;
        for (g, slot) in live {
            if want.contains(&g) {
                self.pre_live[par].push((g, slot));
            } else {
                self.write_entry_pre(g, 0);
                self.pre_free[par].push(slot);
                issued_any = true;
            }
        }
        let mut issued = 0u64;
        for g in want {
            if issued as usize >= self.pref_cap {
                break;
            }
            if self.host.table[g] != 0 || self.pre_live[par].iter().any(|&(x, _)| x == g) {
                continue; // resident already (in the tier or in a speculative slot)
            }
            if !self.host.store.in_ram(g) {
                // NVMe only: a RAM promotion now, for the next time this expert is predicted or routed
                if !self.pending_ram.contains(&g) {
                    self.pending_ram.push(g);
                    sh.pre_nvme.fetch_add(1, Ordering::Relaxed);
                }
                continue;
            }
            let Some(slot) = self.pre_free[par].pop() else { break };
            let ptr = self.host.store.place[g] as *const u8;
            let nb = self.host.store.record_bytes(g);
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(slot, ptr as *const _, nb, self.pre_stream).result().expect("prefetch copy");
            }
            self.write_entry_pre(g, slot);
            self.pre_inflight.push(g);
            self.pre_live[par].push((g, slot));
            issued += 1;
        }
        if issued > 0 || issued_any {
            unsafe { cudarc::driver::result::event::record(self.pre_event, self.pre_stream).expect("event record") };
            self.pre_busy = true;
            sh.pre_issued.fetch_add(issued, Ordering::Relaxed);
        }
        true
    }

    /// Writes the prestaged table's entry for `g` on the copy stream (behind the record's copy when nonzero).
    fn write_entry_pre(&mut self, g: usize, v: u64) {
        let dst = self.host.prestaged_dev + 8 * g as u64;
        let st = self.pre_stream;
        self.write_word_on(st, dst, v);
    }

    /// Clears the in-flight set once the copies landed (`wait` blocks for them).
    fn finish_copies(&mut self, sh: &Shared, wait: bool) {
        if self.inflight.is_empty() && self.d2h.is_empty() {
            return;
        }
        let _not_capturing = sh.capture.lock().unwrap();
        let done = if wait {
            unsafe { cudarc::driver::result::event::synchronize(self.copy_event).expect("event sync") };
            true
        } else {
            unsafe { cudarc::driver::result::event::query(self.copy_event).is_ok() }
        };
        if done {
            self.inflight.clear();
            // the swapped-out victims landed in their RAM places
            for (v, p) in std::mem::take(&mut self.d2h) {
                let c = self.host.layer_class[v / self.host.n_expert];
                if self.host.store.adopt(v, p) {
                    let a = self.host.store.ram_device_addr(v);
                    self.write_rentry(v, a);
                } else {
                    self.host.store.free_place(c, p);
                }
            }
        }
    }

    /// Starts one RAM promotion from `pending_ram` (an asynchronous NVMe read, at most four in flight) over the
    /// coldest RAM record of its class that is neither a VRAM resident nor being copied to VRAM. False when idle.
    fn promote_one(&mut self, sh: &Shared) -> bool {
        // completed promotion reads: their records are readable, by the pool and in place by the GPU
        for g in self.host.store.promote_poll().expect("promotion reads") {
            let a = self.host.store.ram_device_addr(g);
            self.write_rentry(g, a);
        }
        if self.pending_ram.is_empty() || self.host.store.promotions_pending() >= 4 {
            return false;
        }
        let g = self.pending_ram.pop().unwrap();
        let t0 = std::time::Instant::now();
        self.finish_copies(sh, false);
        let ne = self.host.n_expert;
        let c = self.host.layer_class[g / ne];
        if self.host.store.free_places()[c] == 0 {
            // no free place: evict the coldest RAM record of the class. When the GPU reads RAM records in place
            // the place is quarantined two ticks (a window in flight may still read it) and the promotion waits;
            // at most 64 places of a class wait at once. Otherwise the place is reused at once.
            if self.host.gpu_reads_ram && self.ram_quarantine.iter().filter(|&&(_, cc, _)| cc == c).count() >= 64 {
                self.pending_ram.push(g);
                return false;
            }
            let h = &self.host;
            let inflight = &self.inflight;
            let pre_inflight = &self.pre_inflight;
            let victim = (0..h.table.len()).filter(|&x| h.store.in_ram(x) && h.layer_class[x / ne] == c && h.table[x] == 0 && !inflight.contains(&x) && !pre_inflight.contains(&x)).min_by(|&a, &b| h.score[a].partial_cmp(&h.score[b]).unwrap());
            // QX_DS_RAM_DEDUP=1: a RAM copy of a record that is resident in VRAM serves nobody (the tiers are
            // inclusive only for promoted records: the initial VRAM residents have no RAM copy, and a VRAM victim
            // without one falls to NVMe while its replacement holds a slot and a place). Such a copy is evicted
            // first, the hottest resident's first (the least likely to leave VRAM), without the margin test.
            // Residents of the flex tail keep theirs: the refill after a chunk reads it.
            let dup = if ram_dedup() && !h.gpu_reads_ram {
                let (fp, fb) = (h.flex_ptr, h.flex_bytes);
                let in_tail = |x: usize| fb > 0 && h.table[x] + h.classes[c].0 as u64 > fp;
                (0..h.table.len()).filter(|&x| h.table[x] != 0 && h.layer_class[x / ne] == c && h.store.in_ram(x) && !inflight.contains(&x) && !pre_inflight.contains(&x) && !in_tail(x)).max_by(|&a, &b| h.score[a].partial_cmp(&h.score[b]).unwrap())
            } else {
                None
            };
            let victim = if dup.is_some() { dup } else { victim };
            let Some(v) = victim else { return true };
            if dup.is_none() && h.score[g] <= h.score[v] * (1.0 + h.margin) {
                return true;
            }
            if dup.is_some() {
                sh.ram_dedups.fetch_add(1, Ordering::Relaxed);
            }
            self.write_rentry(v, 0);
            if self.host.gpu_reads_ram {
                if let Some((cc, p)) = self.host.store.evict_deferred(v) {
                    self.ram_quarantine.push_back((self.ticks + 2, cc, p));
                }
                self.pending_ram.push(g);
                return true;
            }
            self.host.store.evict(v);
        }
        if self.host.store.promote_submit(g).expect("promote") {
            sh.ram_promos.fetch_add(1, Ordering::Relaxed);
        }
        sh.promo_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }
}

/// Whether the RAM copies of VRAM residents are reclaimed (`QX_DS_RAM_DEDUP`, default 2; 0 = the tiers as before
/// 2026-11-10, when an evicted initial resident fell to NVMe and its replacement held a slot and a RAM place:
/// over half of decode256k's NVMe fetches). Measured on decode256k / serve256k3: off 42.0 / 93.0, 1 (RAM
/// promotions evict such copies first, victims queued for an NVMe read) 41.2 / 88.3, 2 (victims copied back
/// from their slot) 38.3 / 87.6.
pub fn ram_dedup() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QX_DS_RAM_DEDUP").map_or(true, |v| v != "0"))
}

/// `QX_DS_RAM_DEDUP=2` (the default): VRAM victims without a RAM copy are copied back from their slot (1: NVMe queue only).
pub fn swap_mode() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QX_DS_RAM_DEDUP").map_or(true, |v| v == "2" || v == "swap"))
}

/// The mailbox offset of the "all routed experts" list (`MB_ALL` in qx_sys.cuh).
const MB_ALL_OFF: usize = mailbox::MB_ALL;

/// The copy streams the staged records are spread over (`QX_DS_STAGE_STREAMS`, default 4, at most 8).
/// `QX_DS_CHUNK_OVERLAP=0`: a chunk job's NVMe batches are read and copied one after another, ahead of its RAM
/// records (the form until 2026-11-12). Default: the reads of the next NVMe batch run while RAM records copy.
pub fn chunk_overlap() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_DS_CHUNK_OVERLAP").is_ok_and(|v| v == "0"))
}

pub fn n_stage_streams() -> usize {
    std::env::var("QX_DS_STAGE_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 8)
}

/// The NVMe records of one read batch of a chunk job (`QX_DS_CHUNK_READS`, 1 to 32; 8 by default). A batch is read
/// blind once submitted, so small ones let the routing cancel more of a prefetch, and 8 records are already some
/// 600 requests of 128 KiB in flight, the disk's full rate. 2026-11-12, chunk of a warmed 2,048-token prompt: 3.26 s
/// with 8, 3.26 with 16, 3.28 with 4, 3.51 with 32 (3.96 with 8 and `QX_DS_CHUNK_SKIP=0`); prefill256k's scored
/// chunk 16.11-16.13 s with 8, 16.29-16.30 with 32.
pub fn chunk_batch() -> usize {
    static U: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_DS_CHUNK_READS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).clamp(1, 32))
}

/// The event sets of the chunk jobs' NVMe batches (`QX_DS_CHUNK_SETS`, 2 to 8).
pub fn chunk_sets() -> usize {
    static U: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("QX_DS_CHUNK_SETS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).clamp(2, 8))
}

/// Whether a prefetch job skips the picks the routing left without rows (`QX_DS_CHUNK_SKIP=0`: it stages them all).
pub fn chunk_skip() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_DS_CHUNK_SKIP").is_ok_and(|v| v == "0"))
}

/// Whether the GPU waits for the staged records on per-stream ready words written behind the copies
/// (`qx_moe_wait_staged2`, `QX_DS_STG_WAIT=words`) instead of the done word the driver posts after the copies
/// and its pool share (the default: the words measured equal, decode256k 44.7/44.6 vs 44.8/44.9, 2026-11-08,
/// because the pool share finishes no sooner than the copies and the staged kernels are small).
pub fn stage_ready_by_words() -> bool {
    std::env::var("QX_DS_STG_WAIT").is_ok_and(|v| v == "words")
}

fn run(sh: Arc<Shared>, host: Host, cpus: Vec<usize>, ready: std::sync::mpsc::Sender<()>) {
    qx_gpu::mem::bind();
    let pool = Pool::new(&cpus);
    let stream = unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("copy stream");
    let pre_stream = unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("prefetch stream");
    let pre_event = cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("prefetch event");
    let copy_event = cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("copy event");
    let n_stage_streams = n_stage_streams();
    let stage_streams: Vec<cudarc::driver::sys::CUstream> = (0..n_stage_streams).map(|_| unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("stage stream")).collect();
    let stage_events: Vec<cudarc::driver::sys::CUevent> = (0..n_stage_streams).map(|_| cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("stage event")).collect();
    let mk_events = || -> Vec<cudarc::driver::sys::CUevent> { (0..n_stage_streams).map(|_| cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("stage event")).collect() };
    let chunk_events = std::array::from_fn(|_| mk_events());
    let mut lp = Loop { host, pool, shared: Some(sh.clone()), chunk_events, chunk_events_pending: [false; 8], chunk_batch_no: 0, chunk_staged: false, ring_next: 0, ring_user: Vec::new(), stream, words: HostMapped::new(1 << 20), word_next: 0, missed: Vec::new(), last_sel: Vec::new(), quarantine: Default::default(), ticks: 0, promoted_tokens: 0, copy_event, inflight: Vec::new(), flex_on: false, flex_slots: Vec::new(), flex_evicted: Vec::new(), pending_ram: Vec::new(), wb: Vec::new(), d2h: Vec::new(), ram_quarantine: Default::default(), stage_streams, stage_events, stg_words: stage_ready_by_words(), last_serve_end: None, pref_k: crate::dsv41::prefetch_k(), pref_cap: std::env::var("QX_DS_PREFETCH_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(2), pref_last: 0, pre_stream, pre_event, pre_busy: false, pre_inflight: Vec::new(), pre_live: [Vec::new(), Vec::new()], pre_free: [Vec::new(), Vec::new()] };
    for par in 0..2 {
        lp.pre_free[par] = (0..lp.host.pre_slots).map(|i| lp.host.pred_stage_base + ((par * lp.host.pre_slots + i) * lp.host.rec_max) as u64).collect();
    }
    // exercise every copy path once before a window can wait on this thread: whatever the CUDA driver
    // loads or allocates on a stream's first copy then happens while the device is idle
    let scratch = qx_gpu::mem::DeviceBuf::new(64);
    for si in 0..lp.stage_streams.len() {
        let st = lp.stage_streams[si];
        lp.write_word_on(st, scratch.ptr, 0);
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(scratch.ptr + 8, lp.words.ptr as *const _, 8, st).result().expect("warm copy");
            cudarc::driver::result::event::record(lp.stage_events[si], st).expect("event record");
            cudarc::driver::result::event::synchronize(lp.stage_events[si]).expect("warm copy");
        }
    }
    lp.write_word(scratch.ptr, 0);
    unsafe {
        cudarc::driver::result::event::record(lp.copy_event, lp.stream).expect("event record");
        cudarc::driver::result::event::synchronize(lp.copy_event).expect("warm copy");
    }
    let _ = ready.send(());
    let mut last_job = 0u64;
    let mut idle = 0u32;
    let mut idle_since = std::time::Instant::now();
    let mut idle_reported = false;
    let verbose = std::env::var("QX_DS_VERBOSE").is_ok();
    loop {
        if sh.stop.load(Ordering::Relaxed) {
            return;
        }
        let mut worked = false;
        lp.flex_check(&sh);
        if lp.prefetch_check(&sh) {
            worked = true;
        }
        let seq = sh.mb.atomic_u64(MB_SEQ).load(Ordering::Acquire);
        if seq != last_job {
            last_job = seq;
            lp.serve(&sh, seq);
            worked = true;
        }
        let chunk = sh.chunk_jobs.lock().unwrap().pop_front();
        if let Some(job) = chunk {
            lp.serve_chunk(&sh, job);
            worked = true;
        }
        let td = sh.tokens_done.load(Ordering::Acquire);
        if td > lp.promoted_tokens {
            let dt = (td - lp.promoted_tokens) as usize;
            lp.promoted_tokens = td;
            lp.adapt(&sh, dt);
            worked = true;
        }
        if !worked && lp.promote_one(&sh) {
            worked = true;
        }
        if sh.profile_req.swap(false, Ordering::AcqRel) {
            let h = &lp.host;
            let ne = h.n_expert;
            let tier = |g: usize| if h.table[g] != 0 { 0 } else if h.store.in_ram(g) { 1 } else { 2 };
            let mut order: Vec<usize> = (0..h.table.len()).collect();
            order.sort_by(|&a, &b| tier(a).cmp(&tier(b)).then(h.score[b].partial_cmp(&h.score[a]).unwrap_or(std::cmp::Ordering::Equal)).then(a.cmp(&b)));
            *sh.profile.lock().unwrap() = Some(order.into_iter().map(|g| (g / ne, g % ne)).collect());
            worked = true;
        }
        if worked {
            idle = 0;
            idle_since = std::time::Instant::now();
            idle_reported = false;
        } else {
            idle += 1;
            if idle < 2_000_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(20));
            }
            // QX_DS_VERBOSE: a long idle while a window is in flight is a stuck mailbox; print its state once
            if verbose && !idle_reported && idle_since.elapsed().as_secs() >= 15 {
                idle_reported = true;
                let w = |o: usize| *sh.mb.slice::<u32>(o, 1).first().unwrap();
                eprintln!("driver idle 15 s: MB_SEQ {} MB_DONE {} MB_STG_DONE {} last_job {} layer {} nmiss {} nstage {} tokens_done {}", sh.mb.atomic_u64(MB_SEQ).load(Ordering::Acquire), sh.mb.atomic_u64(MB_DONE).load(Ordering::Acquire), sh.mb.atomic_u64(mailbox::MB_STG_DONE).load(Ordering::Acquire), last_job, w(MB_LAYER), w(MB_NMISS), w(mailbox::MB_NSTAGE), sh.tokens_done.load(Ordering::Acquire));
            }
        }
    }
}
