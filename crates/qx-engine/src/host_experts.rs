//! Every routed expert in pinned host memory, one contiguous [gate | up | down] record per expert.
//!
//! The memory is an anonymous mapping backed by transparent huge pages, filled from the GGUF and then
//! registered with the driver, so the CPU pool reads it with few TLB misses and VRAM swaps DMA from it
//! directly. A file mapping cannot be registered (the driver refuses read-only file pages). From a pack the
//! records are already in this layout on disk: one O_DIRECT read fills the memory, and the page cache keeps no
//! second copy of them.
use crate::experts::LayerExperts;
use anyhow::{bail, Result};
use rayon::prelude::*;

pub struct HostExperts {
    base: *mut u8,
    bytes: usize,
    /// Per layer: offset of expert 0, bytes per expert record, the gate (= up) and down part sizes, and the
    /// scale tail bytes.
    pub layers: Vec<(usize, usize, usize, usize, usize)>,
    pub n_expert: usize,
}

unsafe impl Send for HostExperts {}
unsafe impl Sync for HostExperts {}

const ALIGN: usize = 4096;

impl HostExperts {
    pub fn build(layers: &[LayerExperts], n_expert: usize, pack: Option<&qx_pack::pack::PackInfo>) -> Result<Self> {
        let mut table = Vec::new();
        let mut off = 0usize;
        for le in layers {
            let rec = le.expert_bytes().next_multiple_of(ALIGN);
            table.push((off, rec, le.gate_bytes(), le.down_bytes(), le.tail_bytes()));
            off += rec * n_expert;
        }
        let bytes = off.next_multiple_of(2 << 20);
        let base = unsafe {
            let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
            if p == libc::MAP_FAILED {
                bail!("mmap of {bytes} bytes for host experts failed");
            }
            libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
            p as *mut u8
        };
        let me = Self { base, bytes, layers: table, n_expert };
        if let Some(p) = pack {
            let e = &p.experts;
            let same = e.n_expert == n_expert && e.bytes as usize == bytes && me.layers.iter().enumerate().all(|(l, t)| t.0 as u64 == e.layer_offset[l] && t.1 as u64 == e.record[l]);
            if !same {
                bail!("the pack's expert records are not laid out as this build lays them out in memory");
            }
            unsafe { qx_pack::pack::read_into(&p.path, p.offset[1], base, bytes)? };
            return me.register();
        }
        // Fill in parallel: one task per (layer, expert block of 32).
        let base_addr = base as usize;
        let jobs: Vec<(usize, usize)> = (0..layers.len()).flat_map(|l| (0..n_expert).step_by(32).map(move |e| (l, e))).collect();
        jobs.par_iter().for_each(|&(l, e0)| {
            let (off, rec, gb, db, tb) = me.layers[l];
            for e in e0..(e0 + 32).min(n_expert) {
                let [g, u, d] = layers[l].parts(e);
                let tail = layers[l].tail(e);
                unsafe {
                    let dst = (base_addr + off + e * rec) as *mut u8;
                    std::ptr::copy_nonoverlapping(g.as_ptr(), dst, gb);
                    std::ptr::copy_nonoverlapping(u.as_ptr(), dst.add(gb), gb);
                    std::ptr::copy_nonoverlapping(d.as_ptr(), dst.add(2 * gb), d.len());
                    std::ptr::copy_nonoverlapping(tail.as_ptr(), dst.add(2 * gb + db), tb);
                }
            }
        });
        me.register()
    }

    fn register(self) -> Result<Self> {
        let (base, bytes) = (self.base, self.bytes);
        qx_gpu::mem::bind();
        // Mapped too: the prefill's grouped GEMMs read non-resident experts in place over PCIe (UVA: the device
        // address equals the host address).
        let r = unsafe { cudarc::driver::sys::cuMemHostRegister_v2(base as *mut _, bytes, cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE | cudarc::driver::sys::CU_MEMHOSTREGISTER_DEVICEMAP) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            bail!("cuMemHostRegister of {:.1} GiB failed: {r:?}", bytes as f64 / 1073741824.0);
        }
        Ok(self)
    }

    /// Expert (l, e)'s record: gate rows, then up rows, then down rows, then the scale tail.
    pub fn record(&self, l: usize, e: usize) -> *const u8 {
        let (off, rec, _, _, _) = self.layers[l];
        unsafe { self.base.add(off + e * rec) }
    }

    pub fn record_bytes(&self, l: usize) -> usize {
        let (_, _, gb, db, tb) = self.layers[l];
        2 * gb + db + tb
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for HostExperts {
    fn drop(&mut self) {
        qx_gpu::mem::bind();
        unsafe {
            let _ = cudarc::driver::sys::cuMemHostUnregister(self.base as *mut _);
            libc::munmap(self.base as *mut _, self.bytes);
        }
    }
}
