//! The VRAM expert cache: fixed-size slots in one arena, and the device table (`layer * n_expert + e` to slot
//! address, 0 when not resident) the MoE post kernel snapshots every layer. The driver thread's `Swapper`
//! (mailbox.rs) changes residency while decoding; this module allocates the slots and fills them first.
use crate::host_experts::HostExperts;
use anyhow::Result;
use qx_gpu::mem::DeviceBuf;
use qx_pack::ggml::GgmlType;
use qx_pack::gguf::Gguf;
use qx_ref::weights::W;

/// Where one layer's routed experts live in the GGUF, and their geometry.
pub struct LayerExperts<'a> {
    pub gate: W<'a>,
    pub up: W<'a>,
    pub down: W<'a>,
    pub ff: usize,
    pub n_embd: usize,
    /// NVFP4 experts: the per-tensor scales {gate, up, down, 0} of each expert, the 16-byte record tail
    /// (qx_pack::pack); empty for the other formats.
    pub scales: Vec<[f32; 4]>,
}

impl<'a> LayerExperts<'a> {
    pub fn load(g: &'a Gguf, l: usize, ff: usize, n_embd: usize) -> Result<Self> {
        let w = |s: &str| W::new(g, g.tensor(&format!("blk.{l}.{s}")).unwrap());
        let gate = w("ffn_gate_exps.weight");
        let n_expert = gate.rows / ff;
        let scales = if gate.ty == GgmlType::NVFP4 { qx_pack::pack::expert_scales(g, l, n_expert)? } else { Vec::new() };
        Ok(Self { gate, up: w("ffn_up_exps.weight"), down: w("ffn_down_exps.weight"), ff, n_embd, scales })
    }
    /// Bytes of the scale tail after the down rows (16 for NVFP4, else 0).
    pub fn tail_bytes(&self) -> usize {
        if self.scales.is_empty() {
            0
        } else {
            16
        }
    }
    /// The record tail of expert `e` (all zero when the format has none).
    pub fn tail(&self, e: usize) -> [u8; 16] {
        self.scales.get(e).map(qx_pack::pack::tail_bytes).unwrap_or([0; 16])
    }
    pub fn gate_ty(&self) -> GgmlType {
        self.gate.ty
    }
    /// Bytes of one expert's gate (= up) rows and down rows.
    pub fn gate_bytes(&self) -> usize {
        self.ff * self.gate.row_bytes()
    }
    pub fn down_bytes(&self) -> usize {
        self.n_embd * self.down.row_bytes()
    }
    /// Bytes of one expert's record before padding: gate, up, down and the scale tail.
    pub fn expert_bytes(&self) -> usize {
        2 * self.gate_bytes() + self.down_bytes() + self.tail_bytes()
    }
    /// The three byte ranges of expert `e` in the mapped GGUF.
    pub fn parts(&self, e: usize) -> [&'a [u8]; 3] {
        let (gb, db) = (self.gate_bytes(), self.down_bytes());
        [&self.gate.data[e * gb..(e + 1) * gb], &self.up.data[e * gb..(e + 1) * gb], &self.down.data[e * db..(e + 1) * db]]
    }
}

/// Slots of one record size.
pub struct SlotClass {
    pub slot_bytes: usize,
    /// Slots holding nothing yet.
    pub free: Vec<u64>,
}

pub struct ExpertCache {
    pub arena: DeviceBuf,
    pub table: DeviceBuf,
    /// The table as last written, mirrored on the host.
    pub host_table: Vec<u64>,
    pub n_expert: usize,
    pub classes: Vec<SlotClass>,
    /// Size class of each layer.
    pub layer_class: Vec<usize>,
    pub used: usize,
    pub resident: usize,
}

impl ExpertCache {
    /// Carves `budget` bytes into slots for the experts first in `order` (as many as fit), plus `spare` free
    /// slots per class for swaps to start from, and fills them from `host`.
    pub fn build(host: &HostExperts, n_layer: usize, n_expert: usize, budget: usize, order: &[(usize, usize)], spare: usize) -> Self {
        let mut sizes: Vec<usize> = Vec::new();
        let mut layer_class = Vec::new();
        for l in 0..n_layer {
            let b = host.record_bytes(l).next_multiple_of(256);
            let c = sizes.iter().position(|&s| s == b).unwrap_or_else(|| {
                sizes.push(b);
                sizes.len() - 1
            });
            layer_class.push(c);
        }
        // Take the ranking prefix that fits, keeping the spares' room.
        let spare_bytes: usize = sizes.iter().map(|s| s * spare).sum();
        let mut take: Vec<(usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut seen = vec![false; n_layer * n_expert];
        for &(l, e) in order {
            let b = sizes[layer_class[l]];
            if seen[l * n_expert + e] || used + b + spare_bytes > budget {
                continue;
            }
            seen[l * n_expert + e] = true;
            take.push((l, e));
            used += b;
        }
        let total = used + spare_bytes;
        let arena = DeviceBuf::new(total.max(256));
        let mut host_table = vec![0u64; n_layer * n_expert];
        let mut off = 0usize;
        for &(l, e) in &take {
            let b = sizes[layer_class[l]];
            let src = unsafe { std::slice::from_raw_parts(host.record(l, e), host.record_bytes(l)) };
            arena.upload_at(off, src);
            host_table[l * n_expert + e] = arena.ptr + off as u64;
            off += b;
        }
        let mut classes: Vec<SlotClass> = sizes.iter().map(|&s| SlotClass { slot_bytes: s, free: Vec::new() }).collect();
        for c in classes.iter_mut() {
            for _ in 0..spare {
                c.free.push(arena.ptr + off as u64);
                off += c.slot_bytes;
            }
        }
        let table = DeviceBuf::from_slice(&host_table);
        Self { arena, table, host_table, n_expert, classes, layer_class, used: total, resident: take.len() }
    }

    pub fn is_resident(&self, l: usize, e: usize) -> bool {
        self.host_table[l * self.n_expert + e] != 0
    }
}
