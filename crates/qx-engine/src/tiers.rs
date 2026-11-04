//! The routed experts below VRAM: a pinned RAM tier of whole records and the pack on NVMe for the rest.
//!
//! `HostExperts` (host_experts.rs) holds every record in RAM; a model whose experts exceed RAM uses this store
//! instead. Each layer's records have one size; records of the same size share a class with its own pinned arena
//! of places (4 KiB-aligned, so the pack's records read straight into them with O_DIRECT). A global expert id
//! `g = layer * n_expert + e` maps to a place address or 0 (on NVMe only). Misses are read through io_uring into a
//! ring of staging places of the class, so the CPU pool computes them from pinned memory like any other record;
//! `promote` reads a record into a place of its own, `evict` frees one. The caller (the driver thread) owns the
//! store; nothing here is shared between threads.
use anyhow::{anyhow, bail, Result};
use io_uring::{opcode, types, IoUring};
use qx_gpu::mem::HostMapped;
use qx_pack::pack::PackInfo;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;

const PAGE: usize = 4096;
/// Reads in flight at most (the drive's knee is below this; the pool starts on RAM misses meanwhile).
const QUEUE: usize = 64;

struct Class {
    bytes: usize,
    arena: HostMapped,
    /// Free places (addresses).
    free: Vec<usize>,
    /// Staging places for misses, used round robin.
    stage: Vec<usize>,
    stage_next: usize,
    places: usize,
}

pub struct Stats {
    pub ram_hits: u64,
    pub nvme_reads: u64,
    pub nvme_bytes: u64,
    pub promotions: u64,
}

pub struct ExpertStore {
    pack: PackInfo,
    pub n_layer: usize,
    pub n_expert: usize,
    classes: Vec<Class>,
    pub layer_class: Vec<usize>,
    /// Per global expert: its RAM place (address) or 0.
    pub place: Vec<usize>,
    /// Per place address: the expert it holds (for eviction bookkeeping).
    holder: std::collections::HashMap<usize, usize>,
    file: std::fs::File,
    ring: IoUring,
    /// Promotion reads on their own ring (so `fetch_wait` sees only its reads) and the (expert, place) pairs in
    /// flight: a record counts as in RAM only once its read completed.
    pring: IoUring,
    pending: Vec<(usize, usize)>,
    pub stats: Stats,
}

impl ExpertStore {
    /// Builds the RAM tier: places for the hottest experts of `order` that `skip` does not exclude (those in VRAM),
    /// `budget` bytes in all, plus `stage` staging places per class; fills them from the pack.
    pub fn build(pack: &PackInfo, n_layer: usize, order: &[(usize, usize)], skip: &[bool], budget: usize, stage: usize) -> Result<Self> {
        let ne = pack.experts.n_expert;
        if pack.experts.record.len() != n_layer {
            bail!("the pack has {} expert layers, the model {n_layer}", pack.experts.record.len());
        }
        let mut sizes: Vec<usize> = Vec::new();
        let mut layer_class = Vec::new();
        for l in 0..n_layer {
            let b = pack.experts.record[l] as usize;
            let c = sizes.iter().position(|&s| s == b).unwrap_or_else(|| {
                sizes.push(b);
                sizes.len() - 1
            });
            layer_class.push(c);
        }
        // the ranking prefix that fits beside the staging places
        let stage_bytes: usize = sizes.iter().map(|s| s * stage).sum();
        let mut take: Vec<(usize, usize)> = Vec::new();
        let mut per_class = vec![0usize; sizes.len()];
        let mut used = 0usize;
        let mut seen = vec![false; n_layer * ne];
        for &(l, e) in order {
            let g = l * ne + e;
            let b = sizes[layer_class[l]];
            if seen[g] || skip.get(g).copied().unwrap_or(false) || used + b + stage_bytes > budget {
                continue;
            }
            seen[g] = true;
            take.push((l, e));
            per_class[layer_class[l]] += 1;
            used += b;
        }
        let mut classes: Vec<Class> = sizes
            .iter()
            .zip(&per_class)
            .map(|(&bytes, &n)| {
                let places = n + stage;
                let arena = HostMapped::new((places * bytes).max(PAGE));
                let base = arena.ptr as usize;
                let stage_places: Vec<usize> = (n..places).map(|i| base + i * bytes).collect();
                Class { bytes, arena, free: (0..n).map(|i| base + i * bytes).collect(), stage: stage_places, stage_next: 0, places }
            })
            .collect();
        let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&pack.path)?;
        let ring = IoUring::new(QUEUE as u32)?;
        let pring = IoUring::new(QUEUE as u32)?;
        let mut me = Self { pack: pack.clone(), n_layer, n_expert: ne, classes: Vec::new(), layer_class, place: vec![0; n_layer * ne], holder: Default::default(), file, ring, pring, pending: Vec::new(), stats: Stats { ram_hits: 0, nvme_reads: 0, nvme_bytes: 0, promotions: 0 } };
        // assign places, then fill them in file order with large parallel reads
        let mut fills: Vec<(usize, usize)> = Vec::new(); // (g, place)
        for &(l, e) in &take {
            let c = &mut classes[me.layer_class[l]];
            let p = c.free.pop().unwrap();
            let g = l * ne + e;
            me.place[g] = p;
            me.holder.insert(p, g);
            fills.push((g, p));
        }
        me.classes = classes;
        fills.sort_by_key(|&(g, _)| me.file_offset(g));
        let t0 = std::time::Instant::now();
        {
            use rayon::prelude::*;
            let f = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&pack.path)?;
            let me_ref = &me;
            fills.par_iter().try_for_each(|&(g, p)| -> Result<()> {
                let len = me_ref.record_bytes(g);
                let dst = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
                read_exact_at(&f, dst, me_ref.file_offset(g))
            })?;
        }
        eprintln!(
            "expert store: {} records ({:.1} GiB) in pinned RAM, {} staging places per class, filled in {:.1}s; {} records stay on NVMe",
            take.len(),
            used as f64 / 1073741824.0,
            stage,
            t0.elapsed().as_secs_f64(),
            n_layer * ne - take.len() - skip.iter().filter(|&&s| s).count()
        );
        Ok(me)
    }

    /// Bytes of expert `g`'s record (padded to 4 KiB, as the pack stores it).
    pub fn record_bytes(&self, g: usize) -> usize {
        self.pack.experts.record[g / self.n_expert] as usize
    }

    fn file_offset(&self, g: usize) -> u64 {
        let (l, e) = (g / self.n_expert, g % self.n_expert);
        self.pack.offset[1] + self.pack.experts.layer_offset[l] + e as u64 * self.pack.experts.record[l]
    }

    /// Whether `g` is in RAM.
    pub fn in_ram(&self, g: usize) -> bool {
        self.place[g] != 0 && !self.pending.iter().any(|&(h, _)| h == g)
    }

    /// Starts promoting `g` into a free place of its class: the NVMe read goes on the promotion ring, and the
    /// record counts as in RAM once `promote_poll` sees it complete. False when the class has no free place.
    pub fn promote_submit(&mut self, g: usize) -> Result<bool> {
        if self.place[g] != 0 {
            return Ok(true);
        }
        let c = self.layer_class[g / self.n_expert];
        let Some(p) = self.classes[c].free.pop() else { return Ok(false) };
        self.place[g] = p;
        self.holder.insert(p, g);
        let len = self.record_bytes(g);
        let e = opcode::Read::new(types::Fd(self.file.as_raw_fd()), p as *mut u8, len as u32).offset(self.file_offset(g)).build().user_data(g as u64);
        unsafe { self.pring.submission().push(&e).map_err(|_| anyhow!("promotion queue full"))? };
        self.pring.submit()?;
        self.pending.push((g, p));
        self.stats.promotions += 1;
        Ok(true)
    }

    /// Reaps the completed promotion reads; returns the experts now in RAM.
    pub fn promote_poll(&mut self) -> Result<Vec<usize>> {
        let mut done = Vec::new();
        if self.pending.is_empty() {
            return Ok(done);
        }
        let cqes: Vec<_> = self.pring.completion().collect();
        for c in cqes {
            let g = c.user_data() as usize;
            if c.result() < 0 || c.result() as usize != self.record_bytes(g) {
                bail!("expert {g}: NVMe promotion read returned {}", c.result());
            }
            self.pending.retain(|&(h, _)| h != g);
            self.stats.nvme_reads += 1;
            self.stats.nvme_bytes += self.record_bytes(g) as u64;
            done.push(g);
        }
        Ok(done)
    }

    /// Promotion reads still in flight.
    pub fn promotions_pending(&self) -> usize {
        self.pending.len()
    }

    /// Frees expert `g`'s RAM place without returning it to the free list: (class, place), for a caller that
    /// quarantines places a GPU window in flight may still read (`free_place` later).
    pub fn evict_deferred(&mut self, g: usize) -> Option<(usize, usize)> {
        let p = self.place[g];
        if p == 0 {
            return None;
        }
        self.place[g] = 0;
        self.holder.remove(&p);
        Some((self.layer_class[g / self.n_expert], p))
    }

    /// Returns a place `evict_deferred` took out to its class's free list.
    pub fn free_place(&mut self, c: usize, p: usize) {
        self.classes[c].free.push(p);
    }

    /// The device-mapped address of expert `g`'s RAM place (0 when not in RAM).
    pub fn ram_device_addr(&self, g: usize) -> u64 {
        if !self.in_ram(g) {
            return 0;
        }
        self.device_addr(self.place[g] as *const u8)
    }

    /// Issues the reads of `jobs` (expert, destination) and waits for all of them.
    fn read_all(&mut self, jobs: &[(usize, usize)]) -> Result<()> {
        let fd = self.file.as_raw_fd();
        let mut next = 0;
        let mut done = 0;
        while done < jobs.len() {
            while next < jobs.len() && next - done < QUEUE {
                let (g, dst) = jobs[next];
                let len = self.record_bytes(g);
                let e = opcode::Read::new(types::Fd(fd), dst as *mut u8, len as u32).offset(self.file_offset(g)).build().user_data(next as u64);
                unsafe { self.ring.submission().push(&e).map_err(|_| anyhow!("io_uring queue full"))? };
                next += 1;
            }
            self.ring.submit_and_wait(1)?;
            let cqes: Vec<_> = self.ring.completion().collect();
            for c in cqes {
                let (g, _) = jobs[c.user_data() as usize];
                if c.result() < 0 || c.result() as usize != self.record_bytes(g) {
                    bail!("expert {g}: NVMe read returned {}", c.result());
                }
                done += 1;
                self.stats.nvme_reads += 1;
                self.stats.nvme_bytes += self.record_bytes(g) as u64;
            }
        }
        Ok(())
    }

    /// Record addresses for a window's missed experts: RAM places where they exist, staging places read from NVMe
    /// for the rest (all reads in one batch). More misses in one class than it has staging places is an error:
    /// size the ring for the window.
    pub fn fetch(&mut self, gs: &[usize]) -> Result<Vec<*const u8>> {
        let mut out = vec![std::ptr::null(); gs.len()];
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        let mut used = vec![0usize; self.classes.len()];
        for (i, &g) in gs.iter().enumerate() {
            if self.in_ram(g) {
                out[i] = self.place[g] as *const u8;
                self.stats.ram_hits += 1;
                continue;
            }
            // the same expert twice in a window (two tokens): one read
            if let Some(j) = jobs.iter().position(|&(h, _)| h == g) {
                out[i] = jobs[j].1 as *const u8;
                continue;
            }
            let c = self.layer_class[g / self.n_expert];
            let cl = &mut self.classes[c];
            if used[c] >= cl.stage.len() {
                bail!("more NVMe misses in one window than staging places ({})", cl.stage.len());
            }
            used[c] += 1;
            let p = cl.stage[cl.stage_next];
            cl.stage_next = (cl.stage_next + 1) % cl.stage.len();
            out[i] = p as *const u8;
            jobs.push((g, p));
        }
        self.read_all(&jobs)?;
        Ok(out)
    }

    /// `fetch` in two halves: this one returns the pointers at once with the NVMe reads submitted (the staging
    /// places they fill are not readable until `fetch_wait`); records in RAM are readable immediately.
    pub fn fetch_submit(&mut self, gs: &[usize]) -> Result<(Vec<*const u8>, Vec<(usize, usize)>)> {
        let mut out = vec![std::ptr::null(); gs.len()];
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        let mut used = vec![0usize; self.classes.len()];
        for (i, &g) in gs.iter().enumerate() {
            if self.in_ram(g) {
                out[i] = self.place[g] as *const u8;
                self.stats.ram_hits += 1;
                continue;
            }
            if let Some(j) = jobs.iter().position(|&(h, _)| h == g) {
                out[i] = jobs[j].1 as *const u8;
                continue;
            }
            let c = self.layer_class[g / self.n_expert];
            let cl = &mut self.classes[c];
            if used[c] >= cl.stage.len() {
                bail!("more NVMe misses in one window than staging places ({})", cl.stage.len());
            }
            used[c] += 1;
            let p = cl.stage[cl.stage_next];
            cl.stage_next = (cl.stage_next + 1) % cl.stage.len();
            out[i] = p as *const u8;
            jobs.push((g, p));
        }
        if jobs.len() > QUEUE {
            bail!("more NVMe misses in one window than the queue ({QUEUE})");
        }
        let fd = self.file.as_raw_fd();
        for (i, &(g, dst)) in jobs.iter().enumerate() {
            let len = self.record_bytes(g);
            let e = opcode::Read::new(types::Fd(fd), dst as *mut u8, len as u32).offset(self.file_offset(g)).build().user_data(i as u64);
            unsafe { self.ring.submission().push(&e).map_err(|_| anyhow!("io_uring queue full"))? };
        }
        if !jobs.is_empty() {
            self.ring.submit()?;
        }
        Ok((out, jobs))
    }

    /// Waits for the reads `fetch_submit` issued.
    pub fn fetch_wait(&mut self, jobs: &[(usize, usize)]) -> Result<()> {
        let mut done = 0;
        while done < jobs.len() {
            self.ring.submit_and_wait(1)?;
            let cqes: Vec<_> = self.ring.completion().collect();
            for c in cqes {
                let (g, _) = jobs[c.user_data() as usize];
                if c.result() < 0 || c.result() as usize != self.record_bytes(g) {
                    bail!("expert {g}: NVMe read returned {}", c.result());
                }
                done += 1;
                self.stats.nvme_reads += 1;
                self.stats.nvme_bytes += self.record_bytes(g) as u64;
            }
        }
        Ok(())
    }

    /// Frees expert `g`'s RAM place.
    pub fn evict(&mut self, g: usize) {
        let p = self.place[g];
        if p != 0 {
            self.place[g] = 0;
            self.holder.remove(&p);
            let c = self.layer_class[g / self.n_expert];
            self.classes[c].free.push(p);
        }
    }

    /// Reads experts `gs` (on NVMe) into free places of their classes; those with no free place are skipped and
    /// returned.
    pub fn promote(&mut self, gs: &[usize]) -> Result<Vec<usize>> {
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        let mut skipped = Vec::new();
        for &g in gs {
            if self.place[g] != 0 {
                continue;
            }
            let c = self.layer_class[g / self.n_expert];
            match self.classes[c].free.pop() {
                Some(p) => {
                    self.place[g] = p;
                    self.holder.insert(p, g);
                    jobs.push((g, p));
                }
                None => skipped.push(g),
            }
        }
        self.read_all(&jobs)?;
        self.stats.promotions += jobs.len() as u64;
        Ok(skipped)
    }

    /// Free RAM places per class.
    pub fn free_places(&self) -> Vec<usize> {
        self.classes.iter().map(|c| c.free.len()).collect()
    }

    /// Places per class (RAM-resident records plus staging).
    pub fn places(&self) -> Vec<(usize, usize)> {
        self.classes.iter().map(|c| (c.bytes, c.places)).collect()
    }

    /// Device address of a RAM place (the arenas are mapped), for DMA into VRAM slots.
    pub fn device_addr(&self, host: *const u8) -> u64 {
        let h = host as usize;
        for c in &self.classes {
            let base = c.arena.ptr as usize;
            if h >= base && h < base + c.arena.bytes {
                return c.arena.dptr + (h - base) as u64;
            }
        }
        panic!("not a store address");
    }
}

fn read_exact_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.read_at(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            bail!("short read at {off}");
        }
        done += n;
    }
    Ok(())
}
