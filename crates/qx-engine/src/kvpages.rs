//! The FP8 KV cache's host copy as pages (kernels/include/qx_kv.cuh): a page holds `PAGE` cells of every attention
//! layer, [K | V | K scales | V scales] per layer, in one pinned, mapped arena. Each sequence slot has a page
//! table in device memory (u64 page addresses by cell / PAGE); pages are reference-counted, so a prefix cache
//! entry can share a sequence's pages. Each page has a twin in VRAM for the indexer's pooled keys (`kp`), found
//! through a second table per slot, so they cost what the context in use costs rather than slots x contexts.
use qx_gpu::mem::{DeviceBuf, HostMapped};

pub const PAGE: usize = 256;
/// f32 scales per (cell, kv head) for K and for V: one per 32 of the 256 dims (QX_KV_SG in kernels/include/qx_kv.cuh).
pub const SCALE_GROUPS: usize = 8;

pub struct KvPages {
    pub host: HostMapped,
    pub page_bytes: usize,
    /// Byte offsets of a layer's K, V and K scales within a page (the V scales follow the K scales).
    pub region: usize,
    pub kv_bytes: usize,
    free: Vec<u32>,
    pub(crate) refs: Vec<u32>,
    /// Device page tables: slot s at s * max_pages u64, and the page ids behind them.
    pub tables: DeviceBuf,
    pub max_pages: usize,
    pub slot_pages: Vec<Vec<u32>>,
    /// The pooled keys' VRAM pages, `kp_page` bytes each: [attention layer][block within the page][dim] f32.
    pub kp: DeviceBuf,
    pub kp_page: usize,
    /// Device kp page tables (u64 addresses), laid out as `tables`.
    pub kp_tables: DeviceBuf,
}

impl KvPages {
    /// Bytes of one page: every attention layer's K, V, K scales and V scales for `PAGE` cells.
    pub fn page_bytes_for(n_attn: usize, nkv: usize, hd: usize) -> usize {
        let kv_bytes = PAGE * nkv * hd;
        (n_attn * (2 * kv_bytes + 2 * PAGE * nkv * 4 * SCALE_GROUPS)).next_multiple_of(4096)
    }

    /// `extra` pages beyond what `slots` full contexts need (for the prefix cache); `cap` replaces that total
    /// (a pool smaller than slots x contexts: the server then admits requests against the free pages).
    /// `kp_layer`: bytes of one attention layer's pooled keys in a page.
    #[allow(clippy::too_many_arguments)]
    pub fn new(n_attn: usize, nkv: usize, hd: usize, slots: usize, max_ctx: usize, extra: usize, cap: Option<usize>, kp_layer: usize) -> Self {
        let kv_bytes = PAGE * nkv * hd;
        let region = 2 * kv_bytes + 2 * PAGE * nkv * 4 * SCALE_GROUPS;
        let page_bytes = Self::page_bytes_for(n_attn, nkv, hd);
        let max_pages = max_ctx.div_ceil(PAGE);
        // at least one full context and a drafter's worth
        let n = cap.unwrap_or(slots * max_pages + extra).max(2 * max_pages + 2);
        Self {
            host: HostMapped::new(n * page_bytes),
            page_bytes,
            region,
            kv_bytes,
            free: (0..n as u32).rev().collect(),
            refs: vec![0; n],
            tables: DeviceBuf::new(slots * max_pages * 8),
            max_pages,
            slot_pages: vec![Vec::new(); slots],
            kp: DeviceBuf::new(n * n_attn * kp_layer),
            kp_page: n_attn * kp_layer,
            kp_tables: DeviceBuf::new(slots * max_pages * 8),
        }
    }

    pub fn pages(&self) -> usize {
        self.refs.len()
    }
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// (K offset, V offset, K scales offset) of attention layer `l_attn` within a page.
    pub fn offsets(&self, l_attn: usize) -> (u64, u64, u64) {
        let k = (l_attn * self.region) as u64;
        (k, k + self.kv_bytes as u64, k + 2 * self.kv_bytes as u64)
    }

    /// Device address of slot `s`'s page table.
    pub fn table(&self, s: usize) -> u64 {
        self.tables.ptr + (s * self.max_pages * 8) as u64
    }

    /// Device address of slot `s`'s kp page table.
    pub fn kp_table(&self, s: usize) -> u64 {
        self.kp_tables.ptr + (s * self.max_pages * 8) as u64
    }

    pub fn kp_addr(&self, page: u32) -> u64 {
        self.kp.ptr + page as u64 * self.kp_page as u64
    }

    pub fn addr(&self, page: u32) -> u64 {
        self.host.dptr + page as u64 * self.page_bytes as u64
    }

    pub fn host_ptr(&self, page: u32) -> *mut u8 {
        unsafe { self.host.ptr.add(page as usize * self.page_bytes) }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        let p = self.free.pop()?;
        self.refs[p as usize] = 1;
        Some(p)
    }

    pub fn retain(&mut self, page: u32) {
        self.refs[page as usize] += 1;
    }

    pub fn release(&mut self, page: u32) {
        let r = &mut self.refs[page as usize];
        *r -= 1;
        if *r == 0 {
            self.free.push(page);
        }
    }

    pub fn refs(&self, page: u32) -> u32 {
        self.refs[page as usize]
    }
}
