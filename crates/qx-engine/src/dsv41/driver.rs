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
    pub arena_ptr: u64,
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
            stage_ns: Default::default(),
            tokens_done: Default::default(),
            profile_req: AtomicBool::new(false),
            profile: std::sync::Mutex::new(None),
            capture: std::sync::Mutex::new(()),
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
    stream: cudarc::driver::sys::CUstream,
    /// Pinned words the table writes copy from (a ring, never reused in flight).
    words: HostMapped,
    word_next: usize,
    missed: Vec<usize>,
    /// (free at tick, class, address).
    quarantine: std::collections::VecDeque<(u64, usize, u64)>,
    ticks: u64,
    promoted_tokens: u64,
    /// The VRAM copies issued at the last token boundary: the event that marks their completion and the RAM
    /// records they read from (not to be evicted before it).
    copy_event: cudarc::driver::sys::CUevent,
    inflight: Vec<usize>,
    /// RAM promotion candidates left from the last boundary, hottest first, done one at a time between jobs.
    pending_ram: Vec<usize>,
    /// RAM places taken from evicted records, free again two ticks later (a window in flight may read them in
    /// place): (free at tick, class, place).
    ram_quarantine: std::collections::VecDeque<(u64, usize, usize)>,
    /// The staging copies' streams (round robin: one copy engine queue is 14 GB/s, several approach the link) and
    /// an event per stream.
    stage_streams: Vec<cudarc::driver::sys::CUstream>,
    stage_events: Vec<cudarc::driver::sys::CUevent>,
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
            for &si in &used_streams {
                unsafe { cudarc::driver::result::event::record(self.stage_events[si], self.stage_streams[si]).expect("event record") };
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
        if nstage > 0 {
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

fn run(sh: Arc<Shared>, host: Host, cpus: Vec<usize>, ready: std::sync::mpsc::Sender<()>) {
    qx_gpu::mem::bind();
    let pool = Pool::new(&cpus);
    let stream = unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("copy stream");
    let copy_event = cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("copy event");
    let n_stage_streams: usize = std::env::var("QX_DS_STAGE_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 8);
    let stage_streams: Vec<cudarc::driver::sys::CUstream> = (0..n_stage_streams).map(|_| unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("stage stream")).collect();
    let stage_events: Vec<cudarc::driver::sys::CUevent> = (0..n_stage_streams).map(|_| cudarc::driver::result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("stage event")).collect();
    let mut lp = Loop { host, pool, stream, words: HostMapped::new(1 << 20), word_next: 0, missed: Vec::new(), quarantine: Default::default(), ticks: 0, promoted_tokens: 0, copy_event, inflight: Vec::new(), pending_ram: Vec::new(), ram_quarantine: Default::default(), stage_streams, stage_events };
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
        let seq = sh.mb.atomic_u64(MB_SEQ).load(Ordering::Acquire);
        if seq != last_job {
            last_job = seq;
            lp.serve(&sh, seq);
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
