//! The PLE n-gram table read straight from the NVMe drive.
//!
//! The table (26.8 GiB of 90-byte IQ4_NL rows) is never mapped: rows come from O_DIRECT 4 KiB reads through
//! io_uring, so the page cache holds none of it. A set-associative row cache in RAM keeps rows already read.
//! One thread produces each position's embedding as soon as its token is known and posts it to the PLE mailbox;
//! for forced tokens it also reads rows for the next positions while the GPU works.
use crate::mailbox::{Shared, PLE_CAP, PLE_CONSUMED, PLE_EMB, PLE_READY};
use io_uring::{opcode, types, IoUring};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const PAGE: usize = 4096;
const WAYS: usize = 8;

/// Where the table is: file, byte offset of row 0, bytes per row, and whole rows per 4 KiB page (a pack's
/// layout: no row crosses a page) or 0 (rows back to back, as in the GGUF).
#[derive(Clone)]
pub struct PleTable {
    pub path: std::path::PathBuf,
    pub offset: u64,
    pub ty: qx_pack::ggml::GgmlType,
    pub row_bytes: usize,
    pub rows_per_page: usize,
}

impl PleTable {
    /// File offset of row `r`.
    fn row_at(&self, r: u64) -> u64 {
        match self.rows_per_page {
            0 => self.offset + r * self.row_bytes as u64,
            n => self.offset + r / n as u64 * PAGE as u64 + r % n as u64 * self.row_bytes as u64,
        }
    }
}

struct RowCache {
    sets: usize,
    keys: Vec<u64>,
    data: Vec<u8>,
    next: Vec<u8>,
    rb: usize,
}

impl RowCache {
    fn new(rows: usize, rb: usize) -> Self {
        let sets = (rows / WAYS).max(1);
        Self { sets, keys: vec![u64::MAX; sets * WAYS], data: vec![0; sets * WAYS * rb], next: vec![0; sets], rb }
    }
    fn set(&self, row: u64) -> usize {
        ((row.wrapping_mul(0x9E3779B97F4A7C15) >> 17) % self.sets as u64) as usize
    }
    fn get(&self, row: u64) -> Option<&[u8]> {
        let s = self.set(row);
        (0..WAYS).find(|&w| self.keys[s * WAYS + w] == row).map(|w| &self.data[(s * WAYS + w) * self.rb..(s * WAYS + w + 1) * self.rb])
    }
    fn put(&mut self, row: u64, bytes: &[u8]) {
        let s = self.set(row);
        if (0..WAYS).any(|w| self.keys[s * WAYS + w] == row) {
            return;
        }
        let w = self.next[s] as usize;
        self.next[s] = ((w + 1) % WAYS) as u8;
        self.keys[s * WAYS + w] = row;
        self.data[(s * WAYS + w) * self.rb..(s * WAYS + w + 1) * self.rb].copy_from_slice(bytes);
    }
}

/// Page-aligned scratch for O_DIRECT reads.
struct Aligned {
    ptr: *mut u8,
    bytes: usize,
}
impl Aligned {
    fn new(bytes: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).unwrap();
        Self { ptr: unsafe { std::alloc::alloc_zeroed(layout) }, bytes }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.bytes, PAGE).unwrap()) };
    }
}

pub struct Stats {
    pub rows: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub reads: std::sync::atomic::AtomicU64,
}

/// Reads `rows` (row indices) into `out` (row_bytes each), through the cache, with every missing page read
/// in one batch of io_uring requests.
fn fetch(ring: &mut IoUring, fd: i32, t: &PleTable, cache: &mut RowCache, buf: &Aligned, rows: &[u64], out: &mut [u8], stats: &Stats) {
    let rb = t.row_bytes;
    let mut missing: Vec<(usize, u64, u64)> = Vec::new(); // (row slot, page offset, bytes)
    for (i, &r) in rows.iter().enumerate() {
        stats.rows.fetch_add(1, Ordering::Relaxed);
        if let Some(b) = cache.get(r) {
            out[i * rb..(i + 1) * rb].copy_from_slice(b);
            stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let start = t.row_at(r);
        let page = start / PAGE as u64 * PAGE as u64;
        let end = (start + rb as u64).div_ceil(PAGE as u64) * PAGE as u64;
        missing.push((i, page, end - page));
    }
    if missing.is_empty() {
        return;
    }
    // Each missing row gets its own 8 KiB of scratch (a row can straddle two pages).
    for (k, &(_, off, len)) in missing.iter().enumerate() {
        let dst = unsafe { buf.ptr.add(k * 2 * PAGE) };
        let e = opcode::Read::new(types::Fd(fd), dst, len as u32).offset(off).build().user_data(k as u64);
        unsafe { ring.submission().push(&e).expect("io_uring queue full") };
    }
    ring.submit_and_wait(missing.len()).expect("io_uring submit");
    let mut done = 0;
    while done < missing.len() {
        let cqes: Vec<_> = ring.completion().collect();
        for c in cqes {
            assert!(c.result() >= 0, "PLE read failed: {}", c.result());
            done += 1;
        }
        if done < missing.len() {
            ring.submit_and_wait(1).expect("io_uring wait");
        }
    }
    stats.reads.fetch_add(missing.len() as u64, Ordering::Relaxed);
    for (k, &(i, off, _)) in missing.iter().enumerate() {
        let start = t.row_at(rows[i]);
        let src = unsafe { std::slice::from_raw_parts(buf.ptr.add(k * 2 * PAGE + (start - off) as usize), rb) };
        out[i * rb..(i + 1) * rb].copy_from_slice(src);
        cache.put(rows[i], src);
    }
}

/// The reader thread: posts the embedding of each position in turn. `cfg` gives the hash and row geometry.
pub fn run(sh: Arc<Shared>, t: PleTable, cfg: qx_ref::model::Config, n_embd: usize, cache_rows: usize, stats: Arc<Stats>) {
    let f = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&t.path).expect("open PLE table with O_DIRECT");
    let fd = f.as_raw_fd();
    let mut ring = IoUring::new(64).expect("io_uring");
    let mut cache = RowCache::new(cache_rows, t.row_bytes);
    let buf = Aligned::new(64 * 2 * PAGE);
    let pd = cfg.ple_dim;
    let nrows = cfg.ple_heads_per_gram * (cfg.ple_ngram - 1);
    let mut raw = vec![0u8; nrows * t.row_bytes];
    let mut emb = vec![0f32; n_embd];
    let ns = sh.seqs.len();
    // each slot's tokens so far (the n-gram hashes look back), rebuilt from the forced and sampled tokens
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); ns];
    let mut idle = 0u32;
    loop {
        if sh.stop.load(Ordering::Relaxed) {
            return;
        }
        let mut worked = false;
        for slot in 0..ns {
            let q = &sh.seqs[slot];
            let mb = sh.ple_at(slot);
            if q.paused.load(Ordering::Acquire) {
                tokens[slot].clear();
                continue;
            }
            let epoch = q.epoch.load(Ordering::Acquire);
            // A rejected speculative window: restart at position r, whose embedding (and every later one) used a
            // draft token. Done before the host launches anything that waits on these positions.
            let rw = q.ple_rewind.load(Ordering::Acquire);
            if rw > 0 {
                let r = rw as usize - 1;
                tokens[slot].truncate(r);
                q.ple_next.store(r as u32, Ordering::Release);
                sh.ple.atomic_u64(mb + PLE_READY).store(r as u64, Ordering::Release);
                sh.ple.atomic_u64(mb + PLE_CONSUMED).store(r as u64, Ordering::Release);
                q.ple_rewind.store(0, Ordering::Release);
                worked = true;
                continue;
            }
            let p = q.ple_next.load(Ordering::Acquire) as usize;
            let nf = q.n_forced.load(Ordering::Acquire) as usize;
            let done_steps = sh.count(slot).load(Ordering::Acquire) as usize;
            let tok = if p < nf {
                q.forced.lock().unwrap().get(p).copied()
            } else if p > 0 && done_steps >= p && q.allow_sampled.load(Ordering::Acquire) {
                Some(sh.outs(slot, p)[p - 1])
            } else {
                None
            };
            let Some(tk) = tok else { continue };
            // the ring slot is free once the GPU has consumed position p - cap
            if (sh.ple.atomic_u64(mb + PLE_CONSUMED).load(Ordering::Acquire) as usize) + PLE_CAP <= p {
                continue;
            }
            let toks = &mut tokens[slot];
            toks.truncate(p);
            if toks.len() < p {
                // no history (a pause, or a restored prefix): the forced tokens are it
                let f = q.forced.lock().unwrap();
                if f.len() < p {
                    continue;
                }
                toks.clear();
                toks.extend_from_slice(&f[..p]);
            }
            toks.push(tk);
            let rows = qx_ref::model::ple_rows(&cfg, toks);
            fetch(&mut ring, fd, &t, &mut cache, &buf, &rows, &mut raw, &stats);
            for h in 0..nrows {
                qx_pack::dequant(t.ty, &raw[h * t.row_bytes..(h + 1) * t.row_bytes], &mut emb[h * pd..(h + 1) * pd]);
            }
            // a reset while this was read: drop it
            if q.paused.load(Ordering::Acquire) || q.epoch.load(Ordering::Acquire) != epoch {
                continue;
            }
            sh.ple.slice::<f32>(mb + PLE_EMB + (p % PLE_CAP) * 4 * n_embd, n_embd).copy_from_slice(&emb);
            sh.ple.atomic_u64(mb + PLE_READY).store(p as u64 + 1, Ordering::Release);
            q.ple_next.store(p as u32 + 1, Ordering::Release);
            worked = true;
        }
        if worked {
            idle = 0;
        } else {
            idle += 1;
            if idle < 1_000_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(20));
            }
        }
    }
}
