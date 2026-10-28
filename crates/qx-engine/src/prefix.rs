//! The prefix cache: a sequence's state after its first `pos` tokens, kept so a later prompt that starts with the
//! same tokens restarts from there. An entry holds the recurrent state (DeltaNet S and conv states, PLE history,
//! indexer raw-key ring and pooled keys) in a pinned arena and references to the sequence's host KV pages
//! (`KvPages`, shared by reference count; a partial last page is copied on restore). The VRAM KV ring is refilled
//! from the pages. Entries are evicted least recently used.
use super::*;
use qx_gpu::mem::HostMapped;

/// Where the parts of an entry's state sit in its arena range (byte offsets). Everything before `kp` has a fixed
/// size; the pooled indexer keys that follow are sized by the entry's position (`kp_rows` rows of `idx_dim`
/// floats per attention layer, 256-byte aligned each), not by the context: at 262k a full-context entry was
/// 497 MiB and 4 GiB held 8 of them.
#[derive(Clone, Copy)]
struct Layout {
    gdn: usize,
    conv: usize,
    hist: usize,
    kr: usize,
    kp: usize,
    n_attn: usize,
    idx_dim: usize,
    kpool: usize,
}

impl Layout {
    fn kp_rows(&self, p: usize) -> usize {
        p.div_ceil(self.kpool)
    }
    fn kp_part(&self, kp_rows: usize) -> usize {
        (kp_rows * self.idx_dim * 4).next_multiple_of(256)
    }
    /// Bytes of an entry saved at position `p`.
    fn entry_bytes(&self, p: usize) -> usize {
        (self.kp + self.n_attn * self.kp_part(self.kp_rows(p))).next_multiple_of(4096)
    }
}

pub struct Entry {
    pub tokens: Vec<u32>,
    pub pages: Vec<u32>,
    /// The entry's range of the arena.
    off: usize,
    bytes: usize,
    kp_rows: usize,
    last_used: u64,
}

pub struct PrefixCache {
    arena: HostMapped,
    layout: Layout,
    /// Free ranges (offset, bytes), sorted by offset, neighbours coalesced.
    free: Vec<(usize, usize)>,
    pub entries: Vec<Entry>,
    tick: u64,
    pub hits: u64,
    pub tokens_restored: u64,
    pub snapshots: u64,
    pub evictions: u64,
}

impl PrefixCache {
    /// `budget` bytes of pinned memory for entries' recurrent state (about 160 MiB each at a 32k context).
    pub fn new(cfg: &Config, strides: &Strides, max_ctx: usize, budget: usize) -> Self {
        let n_gdn = (0..cfg.n_layer).filter(|&l| !cfg.is_attn(l)).count();
        let n_attn = cfg.n_layer - n_gdn;
        let _ = max_ctx;
        let mut o = 0usize;
        let mut take = |n: usize| {
            let at = o;
            o += n.next_multiple_of(256);
            at
        };
        let layout = Layout { gdn: take(n_gdn * strides.gdn * 4), conv: take(n_gdn * strides.conv * 4), hist: take(strides.hist * 4), kr: take(n_attn * strides.kr * 4), kp: 0, n_attn, idx_dim: cfg.idx_dim, kpool: cfg.kpool };
        let layout = Layout { kp: o, ..layout };
        let bytes = budget.max(layout.entry_bytes(max_ctx)) / 4096 * 4096;
        eprintln!("prefix cache: {:.2} GiB; entries of {:.1} MiB plus {:.1} KiB per 1k tokens ({:.1} MiB at the full context)", bytes as f64 / 1073741824.0, layout.kp as f64 / 1048576.0, (n_attn * layout.idx_dim * 4) as f64 * 1024.0 / layout.kpool as f64 / 1024.0, layout.entry_bytes(max_ctx) as f64 / 1048576.0);
        Self { arena: HostMapped::new(bytes), layout, free: vec![(0, bytes)], entries: Vec::new(), tick: 0, hits: 0, tokens_restored: 0, snapshots: 0, evictions: 0 }
    }

    /// Whether `bytes` fit in one free range.
    fn has_room(&self, bytes: usize) -> bool {
        self.free.iter().any(|&(_, l)| l >= bytes)
    }

    /// First fit.
    fn alloc(&mut self, bytes: usize) -> Option<usize> {
        let i = self.free.iter().position(|&(_, l)| l >= bytes)?;
        let (off, len) = self.free[i];
        if len == bytes {
            self.free.remove(i);
        } else {
            self.free[i] = (off + bytes, len - bytes);
        }
        Some(off)
    }

    /// Returns a range, merging it with free neighbours.
    fn release(&mut self, off: usize, bytes: usize) {
        let i = self.free.partition_point(|&(o, _)| o < off);
        self.free.insert(i, (off, bytes));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    /// The longest entry that is a proper prefix of `prompt` (at least `min` tokens).
    pub fn lookup(&self, prompt: &[u32], min: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.tokens.len();
            if n >= min && n < prompt.len() && best.is_none_or(|b| n > self.entries[b].tokens.len()) && prompt[..n] == e.tokens[..] {
                best = Some(i);
            }
        }
        best
    }

    fn lru(&self) -> Option<usize> {
        self.entries.iter().enumerate().min_by_key(|(_, e)| e.last_used).map(|(i, _)| i)
    }
}

impl<'a> Engine<'a> {
    /// A pointer into an entry's arena range (`base`), `off` bytes in.
    fn arena_at(&self, pc: &PrefixCache, base: usize, off: usize, n: usize) -> &'static mut [u8] {
        let s = pc.arena.slice::<u8>(base + off, n);
        // SAFETY: the arena outlives the engine's use of it; the slice is consumed within the calling method.
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr(), n) }
    }

    /// The per-layer state regions of `slot`: (device address, bytes, arena offset) for every part, the pooled
    /// keys limited to `kp_rows` rows per attention layer.
    fn state_parts(&self, slot: usize, layout: &Layout, kp_rows: usize) -> Vec<(u64, usize, usize)> {
        let st = self.strides;
        let mut parts = Vec::new();
        let (mut ig, mut ia) = (0, 0);
        for ls in &self.state {
            match ls {
                LayerState::Gdn { conv, s, .. } => {
                    parts.push((s.ptr + (slot * st.gdn * 4) as u64, st.gdn * 4, layout.gdn + ig * st.gdn * 4));
                    parts.push((conv.ptr + (slot * st.conv * 4) as u64, st.conv * 4, layout.conv + ig * st.conv * 4));
                    ig += 1;
                }
                LayerState::Attn { kr, kp, .. } => {
                    parts.push((kr.ptr + (slot * st.kr * 4) as u64, st.kr * 4, layout.kr + ia * st.kr * 4));
                    parts.push((kp.ptr + (slot * st.kp * 4) as u64, kp_rows * layout.idx_dim * 4, layout.kp + ia * layout.kp_part(kp_rows)));
                    ia += 1;
                }
            }
        }
        parts.push((self.s.hist.ptr + (slot * st.hist * 4) as u64, st.hist * 4, layout.hist));
        parts
    }

    /// Drops the least recently used entry (its pages go back when no sequence shares them).
    pub(super) fn prefix_evict_one(&self) -> bool {
        let mut pc = self.prefix.borrow_mut();
        let Some(i) = pc.lru() else { return false };
        let e = pc.entries.swap_remove(i);
        pc.release(e.off, e.bytes);
        pc.evictions += 1;
        let mut kv = self.kv.borrow_mut();
        for p in e.pages {
            kv.release(p);
        }
        true
    }

    /// Pages only prefix entries reference (one reference each): what evicting the whole cache would free.
    pub(super) fn prefix_evictable_pages(&self, kv: &crate::kvpages::KvPages) -> usize {
        self.prefix.borrow().entries.iter().flat_map(|e| e.pages.iter()).filter(|&&p| kv.refs[p as usize] == 1).count()
    }

    /// Whether an entry saved at position `p` fits without evicting one.
    pub fn prefix_has_room(&self, p: usize) -> bool {
        let pc = self.prefix.borrow();
        pc.has_room(pc.layout.entry_bytes(p))
    }

    /// Evicts entries until `n` pages are free.
    pub fn prefix_make_room(&self, n: usize) {
        while self.kv.borrow().free_pages() < n && self.prefix_evict_one() {}
    }

    /// Saves `slot`'s state after its `slot_pos` tokens (the tokens come from its context array) as an entry,
    /// replacing one with the same tokens. Waits for the GPU. Nothing is saved under `min` tokens.
    pub fn prefix_snapshot(&mut self, slot: usize, min: usize) {
        let p = self.slot_pos(slot);
        if p < min || p > self.max_ctx {
            return;
        }
        self.gpu.sync();
        let tokens: Vec<u32> = self.ctx.download_at(self.ctx_at(slot), p);
        let layout = self.prefix.borrow().layout;
        let (bytes, kp_rows) = (layout.entry_bytes(p), layout.kp_rows(p));
        // the arena range: a free one (the entry with these tokens is dropped first), evicting least recently
        // used entries until one fits
        let base = {
            let mut pc = self.prefix.borrow_mut();
            pc.tick += 1;
            if let Some(i) = pc.entries.iter().position(|e| e.tokens == tokens) {
                let e = pc.entries.swap_remove(i);
                let mut kv = self.kv.borrow_mut();
                for pg in e.pages {
                    kv.release(pg);
                }
                pc.release(e.off, e.bytes);
            }
            drop(pc);
            loop {
                if let Some(a) = self.prefix.borrow_mut().alloc(bytes) {
                    break a;
                }
                if !self.prefix_evict_one() {
                    return;
                }
            }
        };
        for (dev, n, off) in self.state_parts(slot, &layout, kp_rows) {
            let dst = self.arena_at(&self.prefix.borrow(), base, off, n);
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, dev) }.expect("state download");
        }
        let pages: Vec<u32> = {
            let mut kv = self.kv.borrow_mut();
            let need = p.div_ceil(crate::kvpages::PAGE);
            let pages: Vec<u32> = kv.slot_pages[slot][..need].to_vec();
            for &pg in &pages {
                kv.retain(pg);
            }
            pages
        };
        let mut pc = self.prefix.borrow_mut();
        let tick = pc.tick;
        pc.entries.push(Entry { tokens, pages, off: base, bytes, kp_rows, last_used: tick });
        pc.snapshots += 1;
    }

    /// The entry index and length of the longest cached prefix of `prompt` of at least `min` tokens.
    pub fn prefix_lookup(&self, prompt: &[u32], min: usize) -> Option<(usize, usize)> {
        let pc = self.prefix.borrow();
        pc.lookup(prompt, min).map(|i| (i, pc.entries[i].tokens.len()))
    }

    /// Starts `prompt` in `slot` (made current) from entry `e`: the slot is reset, its sampling and forced tokens
    /// set, the entry's state and pages restored, the VRAM ring refilled, and its position is the entry's length.
    /// The rest of the prompt follows through `prefill_more`, `step_segments`, or `spec_begin`.
    pub fn prefix_restore(&mut self, slot: usize, e: usize, prompt: &[u32], sp: &Sampling) -> usize {
        self.select(slot);
        if self.draft_slot == Some(slot) {
            self.draft_slot = None;
        }
        self.reset();
        self.set_sampling(sp);
        self.set_forced(prompt);
        let (p, layout, base, kp_rows) = {
            let mut pc = self.prefix.borrow_mut();
            pc.tick += 1;
            let tick = pc.tick;
            let layout = pc.layout;
            let en = &mut pc.entries[e];
            en.last_used = tick;
            assert!(prompt.starts_with(&en.tokens), "restore of a prefix the prompt does not start with");
            (en.tokens.len(), layout, en.off, en.kp_rows)
        };
        for (dev, n, off) in self.state_parts(slot, &layout, kp_rows) {
            let src = self.arena_at(&self.prefix.borrow(), base, off, n);
            unsafe { cudarc::driver::result::memcpy_htod_sync(dev, src) }.expect("state upload");
        }
        // the pages: shared with the entry, but the one the slot keeps writing (a partial last page) is copied
        {
            let pages: Vec<u32> = self.prefix.borrow().entries[e].pages.clone();
            // one page for the partial last page's copy; growth takes pages as it goes (`touch_pages` evicts)
            self.prefix_make_room(1);
            let mut kv = self.kv.borrow_mut();
            let mut mine = Vec::with_capacity(pages.len());
            for (i, &pg) in pages.iter().enumerate() {
                if i + 1 == pages.len() && p % crate::kvpages::PAGE != 0 {
                    let np = kv.alloc().expect("out of KV pages");
                    unsafe { std::ptr::copy_nonoverlapping(kv.host_ptr(pg), kv.host_ptr(np), kv.page_bytes) };
                    mine.push(np);
                } else {
                    kv.retain(pg);
                    mine.push(pg);
                }
            }
            let addrs: Vec<u64> = mine.iter().map(|&pg| kv.addr(pg)).collect();
            kv.tables.upload_at(slot * kv.max_pages * 8, &addrs);
            kv.slot_pages[slot] = mine;
        }
        // the VRAM ring: the most recent cells of each attention layer
        let (c, lo) = (&self.cfg, p.saturating_sub(self.kv_window));
        if p > lo {
            for ls in &self.state {
                if let LayerState::Attn { fp8: Some(f), .. } = ls {
                    self.gpu.run(ATTN.kernel("qx_kv_ring_fill", CubeDim::new_1d((c.head_dim / 2) as u32)), ((p - lo) as u32, 2, 1), &[f.tab.ptr + (slot * 64) as u64, lo as u64, p as u64, self.kv_window as u64, c.n_head_kv as u64, c.head_dim as u64]);
                }
            }
        }
        self.gpu.sync();
        // position p: the step words, the PLE thread and the output count
        self.set_step(p, 0, prompt.len());
        let sh = &self.driver.shared;
        let q = &sh.seqs[slot];
        q.ple_next.store(p as u32, Ordering::Release);
        sh.ple.atomic_u64(sh.ple_at(slot) + mailbox::PLE_CONSUMED).store(p as u64, Ordering::Release);
        sh.ple.atomic_u64(sh.ple_at(slot) + mailbox::PLE_READY).store(p as u64, Ordering::Release);
        sh.count(slot).store(p as u64, Ordering::Release);
        q.allow_sampled.store(true, Ordering::Release);
        self.pos = p;
        self.tokens = prompt.to_vec();
        let mut pc = self.prefix.borrow_mut();
        pc.hits += 1;
        pc.tokens_restored += p as u64;
        p
    }

    /// (entries, hits, tokens restored, snapshots, evictions)
    pub fn prefix_stats(&self) -> (usize, u64, u64, u64, u64) {
        let pc = self.prefix.borrow();
        (pc.entries.len(), pc.hits, pc.tokens_restored, pc.snapshots, pc.evictions)
    }
}
