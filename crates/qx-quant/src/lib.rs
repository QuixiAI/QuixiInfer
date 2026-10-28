//! GGUF block formats on the CPU: plain dequantizers, the reference every faster path is checked against.
//!
//! Each `dequant_*` decodes whole blocks of one format into f32, in the element order ggml defines.
pub mod tables;

use half::{bf16, f16};
use tables::{FP4_VALUES2, IQ2S_GRID, IQ3S_GRID, IQ4NL_VALUES};

pub const QK_K: usize = 256;

#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

/// Q8_0: 34 bytes per 32: f16 scale, 32 int8.
pub fn dequant_q8_0(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
        let d = h(b, 0);
        for i in 0..32 {
            o[i] = d * (b[2 + i] as i8) as f32;
        }
    }
}

/// Q6_K: 210 bytes per 256: ql[128], qh[64], int8 scales[16], f16 d.
pub fn dequant_q6_k(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(210).zip(out.chunks_exact_mut(QK_K)) {
        let (ql, rest) = b.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (sc, d) = rest.split_at(16);
        let d = h(d, 0);
        for half in 0..2 {
            let ql = &ql[64 * half..];
            let qh = &qh[32 * half..];
            let sc = &sc[8 * half..];
            let o = &mut o[128 * half..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// IQ4_NL: 18 bytes per 32: f16 d, 16 bytes of nibbles; low nibbles are elements 0..16, high 16..32.
pub fn dequant_iq4_nl(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
        let d = h(b, 0);
        for j in 0..16 {
            o[j] = d * IQ4NL_VALUES[(b[2 + j] & 0xf) as usize] as f32;
            o[j + 16] = d * IQ4NL_VALUES[(b[2 + j] >> 4) as usize] as f32;
        }
    }
}

/// An unsigned E4M3 scale as ggml decodes it for NVFP4: value / 2, so that the doubled FP4 codes multiply back
/// to the real product. 0 and 0x7f (NaN) decode to 0. Exact: a power of two times (1 + m/8), or m * 2^-10.
pub const fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let (e, m) = (((x >> 3) & 0xf) as u32, (x & 7) as u32);
    if e == 0 {
        m as f32 * (1.0 / 1024.0)
    } else {
        // (1 + m/8) * 2^(e-7) / 2 = (1 + m/8) * 2^(e-8): exponent field e - 8 + 127
        f32::from_bits(((e + 119) << 23) | (m << 20))
    }
}

/// `ue4m3_half` of every byte.
pub static UE4M3_HALF: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = ue4m3_half(i as u8);
        i += 1;
    }
    t
};

/// NVFP4: 36 bytes per 64: four UE4M3 scales, then 32 bytes of packed E2M1 codes; sub-block s (16 values) is
/// bytes 8s .. 8s+7, low nibble value j, high nibble value j + 8.
pub fn dequant_nvfp4(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(36).zip(out.chunks_exact_mut(64)) {
        for s in 0..4 {
            let d = ue4m3_half(b[s]);
            for j in 0..8 {
                let q = b[4 + 8 * s + j];
                o[16 * s + j] = d * FP4_VALUES2[(q & 0xf) as usize] as f32;
                o[16 * s + j + 8] = d * FP4_VALUES2[(q >> 4) as usize] as f32;
            }
        }
    }
}

/// IQ3_S: 110 bytes per 256: f16 d, qs[64], qh[8], signs[32], scales[4].
/// Element group g of 8 (0..32) uses grid indices qs[2g], qs[2g+1] (9 bits with qh), sign byte g, and the
/// scale nibble of its 32-element sub-block: `d * (1 + 2 * scale)`.
pub fn dequant_iq3_s(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(110).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let qs = &b[2..66];
        let qh = &b[66..74];
        let signs = &b[74..106];
        let scales = &b[106..110];
        for g in 0..32 {
            let sub = g / 4; // 32-element sub-block
            let sc = (scales[sub / 2] >> (4 * (sub % 2))) & 0xf;
            let db = d * (1 + 2 * sc as i32) as f32;
            for half in 0..2 {
                let k = 2 * g + half;
                let idx = qs[k] as usize | ((((qh[k / 8] >> (k % 8)) & 1) as usize) << 8);
                let grid = IQ3S_GRID[idx];
                for j in 0..4 {
                    let e = 4 * half + j;
                    let m = ((grid >> (8 * j)) & 0xff) as f32;
                    let s = if (signs[g] >> e) & 1 != 0 { -1.0 } else { 1.0 };
                    o[8 * g + e] = db * m * s;
                }
            }
        }
    }
}

/// IQ2_S: 82 bytes per 256: f16 d, qs[32], signs[32], qh[8], scales[8].
/// Element group g of 8 uses grid index qs[g] (10 bits with qh), sign byte g, and the scale nibble of its
/// 16-element sub-block: `d * (0.5 + scale) * 0.25`.
pub fn dequant_iq2_s(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(82).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let qs = &b[2..34];
        let signs = &b[34..66];
        let qh = &b[66..74];
        let scales = &b[74..82];
        for g in 0..32 {
            let sub = g / 2; // 16-element sub-block
            let sc = (scales[sub / 2] >> (4 * (sub % 2))) & 0xf;
            let db = d * (0.5 + sc as f32) * 0.25;
            let idx = qs[g] as usize | ((((qh[g / 4] >> (2 * (g % 4))) & 3) as usize) << 8);
            let grid = IQ2S_GRID[idx];
            for e in 0..8 {
                let m = ((grid >> (8 * e)) & 0xff) as f32;
                let s = if (signs[g] >> e) & 1 != 0 { -1.0 } else { 1.0 };
                o[8 * g + e] = db * m * s;
            }
        }
    }
}

pub fn dequant_f16(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(2).zip(out.iter_mut()) {
        *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
    }
}

pub fn dequant_bf16(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(2).zip(out.iter_mut()) {
        *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
    }
}

pub fn dequant_f32(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(4).zip(out.iter_mut()) {
        *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
}

/// IQ4_NL quantization of whole blocks (ggml's `quantize_row_iq4_nl` scheme, weights x^2): per 32, the scale
/// with the best weighted fit over 15 trial scales around -max/127, each value at its nearest table level.
/// Writes 18 bytes per 32: f16 scale, 16 bytes of 4-bit indices (low nibble j, high nibble j + 16).
pub fn quantize_iq4_nl(src: &[f32], out: &mut [u8]) {
    let v = IQ4NL_VALUES;
    // the table is sorted: binary search, then the closer neighbour
    let nearest = |x: f32| -> usize {
        if x <= v[0] as f32 {
            return 0;
        }
        if x >= v[15] as f32 {
            return 15;
        }
        let (mut lo, mut hi) = (0usize, 15usize);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x < v[mid] as f32 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        if x - v[lo] as f32 <= v[hi] as f32 - x { lo } else { hi }
    };
    for (x, o) in src.chunks_exact(32).zip(out.chunks_exact_mut(18)) {
        let (mut amax, mut max) = (0f32, 0f32);
        for &a in x {
            if a.abs() > amax {
                amax = a.abs();
                max = a;
            }
        }
        let mut idx = [0usize; 32];
        let mut d = 0f32;
        if amax > 0.0 {
            let w: Vec<f32> = x.iter().map(|a| a * a).collect();
            let fit = |id: f32, idx: &mut [usize; 32]| -> (f32, f32) {
                let (mut sqx, mut sq2) = (0f32, 0f32);
                for j in 0..32 {
                    idx[j] = nearest(id * x[j]);
                    let q = v[idx[j]] as f32;
                    sqx += w[j] * q * x[j];
                    sq2 += w[j] * q * q;
                }
                (sqx, sq2)
            };
            let (sqx, sq2) = fit(v[0] as f32 / max, &mut idx);
            d = if sq2 > 0.0 { sqx / sq2 } else { 0.0 };
            let mut best = if sq2 > 0.0 { sqx * sqx / sq2 } else { 0.0 };
            for itry in -7i32..=7 {
                let mut ti = [0usize; 32];
                let (sqx, sq2) = fit((itry as f32 + v[0] as f32) / max, &mut ti);
                if sq2 > 0.0 && sqx * sqx > best * sq2 {
                    best = sqx * sqx / sq2;
                    d = sqx / sq2;
                    idx = ti;
                }
            }
        }
        o[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            o[2 + j] = (idx[j] as u8) | ((idx[j + 16] as u8) << 4);
        }
    }
}
