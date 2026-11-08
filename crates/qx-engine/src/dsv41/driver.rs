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
    pub stage_ns: AtomicU64,
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
    /// The staging copies' streams (round robin: one copy engine queue is 14 GB/s, several approach the link) and
    /// an event per stream.
    stage_streams: Vec<cudarc::driver::sys::CUstream>,
    stage_events: Vec<cudarc::driver::sys::CUevent>,
    /// Whether the GPU learns of the staged records' landing from the ready words (`stage_ready_by_words`) rather
    /// than from a done word this thread posts after waiting for the copies (and after the pool's share).
    stg_words: bool,
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
        let mut stage = |lp: &mut Loop, g: usize, e: u32, src: *const u8, ord: u64, slot_user: &mut Vec<u64>, next_slot: &mut usize, next_stream: &mut usize| {
            let s = *next_slot;
            *next_slot = (*next_slot + 1) % n_slots;
            // the slot's previous user must be done on the GPU (its ordinal marked)
            while done.load(Ordering::Acquire) < slot_user[s] {
                std::hint::spin_loop();
            }
            slot_user[s] = ord;
            let dst = lp.host.stage_base + s as u64 * rec_max;
            let nb = lp.host.store.record_bytes(g);
            let si = *next_stream % lp.stage_streams.len();
            *next_stream += 1;
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src as *const _, nb, lp.stage_streams[si]).result().expect("chunk stage copy");
            }
            let st = lp.stage_streams[si];
            lp.write_word_on(st, job.slots_dev + 8 * e as u64, dst | (job.generation << 48));
        };
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
        self.ring_user = slot_user;
        self.ring_next = next_slot;
        sh.vram_hits.fetch_add(n_vram, Ordering::Relaxed);
        sh.staged_hits.fetch_add(n_ram, Ordering::Relaxed);
        sh.nvme_misses.fetch_add(n_nvme, Ordering::Relaxed);
        sh.stage_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !job.counts.is_empty() {
            self.missed.extend(job.experts.iter().map(|&e| l * ne + e as usize).filter(|&g| self.host.table[g] == 0));
        }
    }

    /// Answers one MoE job.
    fn serve(&mut self, sh: &Shared, seq: u64) {
        let mb = &sh.mb;
        let n = self.host.n_embd;
        let l = *mb.slice::<u32>(MB_LAYER, 1).first().unwrap() as usize;
        let nm = *mb.slice::<u32>(MB_NMISS, 1).first().unwrap() as usize;
        let nmap = *mb.slice::<u32>(mailbox::MB_NMAP, 1).first().unwrap() as usize;
        let nstage = *mb.slice::<u32>(mailbox::MB_NSTAGE, 1).first().unwrap() as usize;
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
        sh.vram_hits.fetch_add((all.len() - nm - nmap - nstage) as u64, Ordering::Relaxed);
        sh.mapped_hits.fetch_add(nmap as u64, Ordering::Relaxed);
        sh.staged_hits.fetch_add(nstage as u64, Ordering::Relaxed);
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
        let vram_ok = self.inflight.is_empty();
        let mut plan: Vec<(usize, u64)> = Vec::new();
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
                    self.write_entry(v, 0);
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
        if !into_vram.is_empty() {
            unsafe { cudarc::driver::result::event::record(self.copy_event, self.stream).expect("event record") };
            self.inflight = into_vram;
        }
        // RAM: NVMe records among the candidates, hottest first, promoted one per idle gap by `promote_one`
        let in_vram = &self.inflight;
        self.pending_ram = cand.iter().copied().filter(|&g| !self.host.store.in_ram(g) && !in_vram.contains(&g)).take(n_ram).collect();
        self.pending_ram.reverse(); // pop from the back: hottest first
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

    /// Clears the in-flight set once the copies landed (`wait` blocks for them).
    fn finish_copies(&mut self, sh: &Shared, wait: bool) {
        if self.inflight.is_empty() {
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
            let victim = (0..h.table.len()).filter(|&x| h.store.in_ram(x) && h.layer_class[x / ne] == c && h.table[x] == 0 && !inflight.contains(&x)).min_by(|&a, &b| h.score[a].partial_cmp(&h.score[b]).unwrap());
            let Some(v) = victim else { return true };
            if h.score[g] <= h.score[v] * (1.0 + h.margin) {
                return true;
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

/// The mailbox offset of the "all routed experts" list (`MB_ALL` in qx_sys.cuh).
const MB_ALL_OFF: usize = mailbox::MB_ALL;

/// The copy streams the staged records are spread over (`QX_DS_STAGE_STREAMS`, default 4, at most 8).
pub fn n_stage_streams() -> usize {
    std::env::var("QX_DS_STAGE_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 8)
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
    let copy_event = cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("copy event");
    let n_stage_streams = n_stage_streams();
    let stage_streams: Vec<cudarc::driver::sys::CUstream> = (0..n_stage_streams).map(|_| unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("stage stream")).collect();
    let stage_events: Vec<cudarc::driver::sys::CUevent> = (0..n_stage_streams).map(|_| cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("stage event")).collect();
    let mut lp = Loop { host, pool, shared: Some(sh.clone()), ring_next: 0, ring_user: Vec::new(), stream, words: HostMapped::new(1 << 20), word_next: 0, missed: Vec::new(), last_sel: Vec::new(), quarantine: Default::default(), ticks: 0, promoted_tokens: 0, copy_event, inflight: Vec::new(), flex_on: false, flex_slots: Vec::new(), flex_evicted: Vec::new(), pending_ram: Vec::new(), ram_quarantine: Default::default(), stage_streams, stage_events, stg_words: stage_ready_by_words() };
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
