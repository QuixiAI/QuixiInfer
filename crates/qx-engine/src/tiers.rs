//! The routed experts below VRAM: a pinned RAM tier of whole records and the pack on NVMe for the rest.
//!
//! `HostExperts` (host_experts.rs) holds every record in RAM; a model whose experts exceed RAM uses this store
//! instead. Each layer's records have one size; records of the same size share a class with its own pinned arena
//! of places (4 KiB-aligned, so the pack's records read straight into them with O_DIRECT). A global expert id
//! `g = layer * n_expert + e` maps to a place address or 0 (on NVMe only). Misses are read through io_uring into a
//! ring of staging places of the class, so the CPU pool computes them from pinned memory like any other record;
//! `promote` reads a record into a place of its own, `evict` frees one. The caller (the driver thread) owns the
//! store; nothing here is shared between threads.
//!
//! The elastic part (`build_elastic`, QX_DS_RAM_YIELD): the coldest records of the budget get their places in a
//! second arena per class, which `elastic_release` hands back to the operating system (the hottest records move
//! into the first arena, the coldest fall to NVMe) and `elastic_restore` allocates again. A host that needs the
//! memory for a while (a virtual machine) then costs NVMe reads for that while and nothing after.
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
    /// Places in `arena` that hold records (the rest of it is the staging places).
    n_base: usize,
    /// The elastic arena (None: released, or the class has none) and its places.
    ext: Option<HostMapped>,
    ext_n: usize,
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
    /// The records a store built with its elastic part out did not read (coldest first).
    pub elastic_owed: Vec<usize>,
    pub stats: Stats,
}

impl ExpertStore {
    /// Builds the RAM tier: places for the hottest experts of `order` that `skip` does not exclude (those in VRAM),
    /// `budget` bytes in all, plus `stage` staging places per class; fills them from the pack.
    pub fn build(pack: &PackInfo, n_layer: usize, order: &[(usize, usize)], skip: &[bool], budget: usize, stage: usize) -> Result<Self> {
        Self::build_elastic(pack, n_layer, order, skip, budget, stage, 0, false)
    }

    /// `build` with the coldest `elastic` bytes of the budget in arenas of their own (see the module's head).
    /// `out`: the store starts with that part released (the host needs the memory already): its records are not
    /// read, and `elastic_owed` lists them for the caller to bring in after `elastic_restore`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_elastic(pack: &PackInfo, n_layer: usize, order: &[(usize, usize)], skip: &[bool], budget: usize, stage: usize, elastic: usize, out: bool) -> Result<Self> {
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
        // the elastic part: the ranking's tail, per class
        let mut ext_n = vec![0usize; sizes.len()];
        let mut acc = 0usize;
        for &(l, _) in take.iter().rev() {
            let b = sizes[layer_class[l]];
            if acc + b > elastic {
                break;
            }
            acc += b;
            ext_n[layer_class[l]] += 1;
        }
        let mut classes: Vec<Class> = sizes
            .iter()
            .zip(&per_class)
            .zip(&ext_n)
            .map(|((&bytes, &n), &xn)| {
                let n_base = n - xn;
                let arena = HostMapped::new(((n_base + stage) * bytes).max(PAGE));
                let base = arena.ptr as usize;
                let stage_places: Vec<usize> = (n_base..n_base + stage).map(|i| base + i * bytes).collect();
                let ext = (xn > 0 && !out).then(|| HostMapped::new(xn * bytes));
                // the free list is popped from its end: the first arena's places (in the order a store without an
                // elastic part has always had) go to the ranking's head, the elastic arena's to its tail
                let mut free: Vec<usize> = ext.iter().flat_map(|x| (0..xn).rev().map(move |i| x.ptr as usize + i * bytes)).collect();
                free.extend((0..n_base).map(|i| base + i * bytes));
                Class { bytes, arena, free, stage: stage_places, stage_next: 0, places: n + stage, n_base, ext, ext_n: xn }
            })
            .collect();
        let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&pack.path)?;
        let ring = IoUring::new(QUEUE as u32)?;
        let pring = IoUring::new(QUEUE as u32)?;
        // a store that starts with the elastic part out owes the ranking's tail (coldest first)
        let mut owed: Vec<usize> = Vec::new();
        if out {
            let tail: usize = ext_n.iter().sum();
            owed = take.split_off(take.len() - tail).into_iter().rev().map(|(l, e)| l * ne + e).collect();
            used -= acc;
        }
        let mut me = Self { pack: pack.clone(), n_layer, n_expert: ne, classes: Vec::new(), layer_class, place: vec![0; n_layer * ne], holder: Default::default(), file, ring, pring, pending: Vec::new(), elastic_owed: owed, stats: Stats { ram_hits: 0, nvme_reads: 0, nvme_bytes: 0, promotions: 0 } };
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

    /// The number of NVMe landing (staging) places of `g`'s size class.
    pub fn stage_places(&self, g: usize) -> usize {
        self.classes[self.layer_class[g / self.n_expert]].stage.len()
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

    /// Gives expert `g` the place `p` (taken out by `evict_deferred`, already holding `g`'s bytes). False when `g`
    /// got a place meanwhile.
    pub fn adopt(&mut self, g: usize, p: usize) -> bool {
        if self.place[g] != 0 {
            return false;
        }
        self.place[g] = p;
        self.holder.insert(p, g);
        true
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
            if let Some(x) = &c.ext {
                let base = x.ptr as usize;
                if h >= base && h < base + x.bytes {
                    return x.dptr + (h - base) as u64;
                }
            }
        }
        panic!("not a store address");
    }

    /// Bytes of the elastic part (0: the store was built without one), released or not.
    pub fn elastic_bytes(&self) -> usize {
        self.classes.iter().map(|c| c.ext_n * c.bytes).sum()
    }

    /// Whether the elastic arenas are released.
    pub fn elastic_out(&self) -> bool {
        self.classes.iter().any(|c| c.ext_n > 0 && c.ext.is_none())
    }

    /// Hands the elastic arenas back to the operating system. Per class the records that fit the first arena stay,
    /// the hottest by `score`: those of them in the elastic arena are copied into the places the coldest leave.
    /// Returns (the records that lost their place, coldest first; the records that moved). The caller guarantees
    /// that nothing reads or writes a place meanwhile: no promotion read in flight, no copy queued out of or into
    /// a place, every place it took out with `evict_deferred` returned, and no window running.
    pub fn elastic_release(&mut self, score: &[f64]) -> (Vec<usize>, Vec<usize>) {
        assert!(self.pending.is_empty(), "elastic_release with promotion reads in flight");
        let ne = self.n_expert;
        let (mut dropped, mut moved) = (Vec::new(), Vec::new());
        let mut copies: Vec<(usize, usize, usize)> = Vec::new(); // (from, to, bytes)
        for c in 0..self.classes.len() {
            let Some((lo, hi)) = self.classes[c].ext.as_ref().map(|x| (x.ptr as usize, x.ptr as usize + x.bytes)) else { continue };
            let in_ext = |p: usize| p >= lo && p < hi;
            let mut recs: Vec<usize> = self.holder.values().copied().filter(|&g| self.layer_class[g / ne] == c).collect();
            recs.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            let n_drop = recs.len().saturating_sub(self.classes[c].n_base);
            let mut avail: Vec<usize> = self.classes[c].free.drain(..).filter(|&p| !in_ext(p)).collect();
            for &g in &recs[..n_drop] {
                let p = self.place[g];
                self.place[g] = 0;
                self.holder.remove(&p);
                if !in_ext(p) {
                    avail.push(p);
                }
                dropped.push(g);
            }
            for &g in &recs[n_drop..] {
                let p = self.place[g];
                if in_ext(p) {
                    let np = avail.pop().expect("a place in the first arena for every record that stays");
                    copies.push((p, np, self.classes[c].bytes));
                    self.holder.remove(&p);
                    self.place[g] = np;
                    self.holder.insert(np, g);
                    moved.push(g);
                }
            }
            self.classes[c].free = avail;
        }
        {
            use rayon::prelude::*;
            copies.par_iter().for_each(|&(from, to, n)| unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, n) });
        }
        for c in self.classes.iter_mut() {
            c.ext = None;
        }
        // coldest first over all classes
        dropped.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        (dropped, moved)
    }

    /// Allocates the elastic arenas again (their places free). False, with nothing changed, when the host cannot
    /// pin the memory.
    pub fn elastic_restore(&mut self) -> bool {
        let mut got: Vec<(usize, HostMapped)> = Vec::new();
        for (i, c) in self.classes.iter().enumerate() {
            if c.ext_n > 0 && c.ext.is_none() {
                // not zeroed: a place is read (`in_ram`, the GPU's table) only once its record's read completed
                match HostMapped::try_new_opt(c.ext_n * c.bytes, false) {
                    Some(x) => got.push((i, x)),
                    None => return false,
                }
            }
        }
        for (i, x) in got {
            let c = &mut self.classes[i];
            let (base, bytes) = (x.ptr as usize, c.bytes);
            // under the first arena's free places (the list is popped from its end)
            let mut free: Vec<usize> = (0..c.ext_n).map(|k| base + k * bytes).collect();
            free.append(&mut c.free);
            c.free = free;
            c.ext = Some(x);
        }
        true
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
