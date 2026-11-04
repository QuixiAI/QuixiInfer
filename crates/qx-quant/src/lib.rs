//! GGUF block formats on the CPU: plain dequantizers, the reference every faster path is checked against.
//!
//! Each `dequant_*` decodes whole blocks of one format into f32, in the element order ggml defines.
pub mod tables;

use half::{bf16, f16};
use tables::{FP4_VALUES2, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES, KSIGNS};

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

#[inline]
fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn sign(bits: u8, e: usize) -> f32 {
    if (bits >> e) & 1 != 0 { -1.0 } else { 1.0 }
}

/// IQ2_XXS: 66 bytes per 256: f16 d, then 8 sub-blocks of 32 elements as two u32 each: the first holds four
/// 8-bit grid indices (one per 8 elements), the second four 7-bit sign-pattern indices into `KSIGNS` and the
/// 4-bit scale in its top bits: `d * (0.5 + scale) * 0.25`.
pub fn dequant_iq2_xxs(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(66).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        for i in 0..8 {
            let a = u32le(b, 2 + 8 * i);
            let s = u32le(b, 6 + 8 * i);
            let db = d * (0.5 + (s >> 28) as f32) * 0.25;
            for g in 0..4 {
                let grid = IQ2XXS_GRID[((a >> (8 * g)) & 0xff) as usize];
                let sg = KSIGNS[((s >> (7 * g)) & 0x7f) as usize];
                for e in 0..8 {
                    let m = ((grid >> (8 * e)) & 0xff) as f32;
                    o[32 * i + 8 * g + e] = db * m * sign(sg, e);
                }
            }
        }
    }
}

/// IQ2_XS: 74 bytes per 256: f16 d, qs[32] of u16 (9-bit grid index, 7-bit sign-pattern index), scales[8] of
/// nibbles per 16-element sub-block: `d * (0.5 + scale) * 0.25`.
pub fn dequant_iq2_xs(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(74).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let scales = &b[66..74];
        for g in 0..32 {
            let sub = g / 2;
            let sc = (scales[sub / 2] >> (4 * (sub % 2))) & 0xf;
            let db = d * (0.5 + sc as f32) * 0.25;
            let q = u16le(b, 2 + 2 * g);
            let grid = IQ2XS_GRID[(q & 511) as usize];
            let sg = KSIGNS[(q >> 9) as usize];
            for e in 0..8 {
                let m = ((grid >> (8 * e)) & 0xff) as f32;
                o[8 * g + e] = db * m * sign(sg, e);
            }
        }
    }
}

/// IQ3_XXS: 98 bytes per 256: f16 d, qs[64] of 8-bit grid indices (4 elements each), then 8 u32 per 32-element
/// sub-block holding four 7-bit sign-pattern indices and the 4-bit scale in the top bits: `d * (0.5 + scale) * 0.5`.
pub fn dequant_iq3_xxs(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(98).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let qs = &b[2..66];
        for i in 0..8 {
            let s = u32le(b, 66 + 4 * i);
            let db = d * (0.5 + (s >> 28) as f32) * 0.5;
            for g in 0..4 {
                let sg = KSIGNS[((s >> (7 * g)) & 0x7f) as usize];
                for half in 0..2 {
                    let grid = IQ3XXS_GRID[qs[8 * i + 2 * g + half] as usize];
                    for j in 0..4 {
                        let e = 4 * half + j;
                        let m = ((grid >> (8 * j)) & 0xff) as f32;
                        o[32 * i + 8 * g + e] = db * m * sign(sg, e);
                    }
                }
            }
        }
    }
}

/// IQ1_S: 50 bytes per 256: f16 d, qs[32] of 8-bit grid indices, qh[8] of u16 per 32-element sub-block: three
/// extra index bits for each of its four groups, a 3-bit scale (`d * (2 * scale + 1)`) and a sign bit choosing
/// the offset +-0.125 added to every grid value (-1, 0, 1).
pub fn dequant_iq1_s(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(50).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let qs = &b[2..34];
        for i in 0..8 {
            let qh = u16le(b, 34 + 2 * i);
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 == 0 { 0.125 } else { -0.125 };
            for g in 0..4 {
                let idx = qs[4 * i + g] as usize | ((((qh >> (3 * g)) & 7) as usize) << 8);
                let grid = IQ1S_GRID[idx];
                for e in 0..8 {
                    let v = ((grid >> (8 * e)) & 0xff) as f32 - 1.0;
                    o[32 * i + 8 * g + e] = dl * (v + delta);
                }
            }
        }
    }
}

/// Q2_K: 84 bytes per 256: scales[16] (low nibble scale, high nibble min, per 16 elements), qs[64] of 2-bit
/// values, f16 d, f16 dmin. Element e of the block is bits `2 * ((e % 128) / 32)` of byte `32 * (e / 128) + e % 32`.
pub fn dequant_q2_k(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(84).zip(out.chunks_exact_mut(QK_K)) {
        let scales = &b[0..16];
        let qs = &b[16..80];
        let d = h(b, 80);
        let dmin = h(b, 82);
        for e in 0..QK_K {
            let j = e / 16;
            let dl = d * (scales[j] & 0xf) as f32;
            let ml = dmin * (scales[j] >> 4) as f32;
            let q = (qs[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3;
            o[e] = dl * q as f32 - ml;
        }
    }
}

/// The 6-bit scale and min of sub-block `j` (0..8) from a K-quant's packed 12 scale bytes.
#[inline]
fn k_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        ((scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))
    }
}

/// Q5_K: 176 bytes per 256: f16 d, f16 dmin, scales[12] (6-bit scale and min per 32 elements), qh[32] (one high
/// bit per element, bit j for sub-block j), qs[128] of nibbles (sub-block 2c in the low nibbles of chunk c, 2c + 1
/// in the high).
pub fn dequant_q5_k(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(176).zip(out.chunks_exact_mut(QK_K)) {
        let d = h(b, 0);
        let dmin = h(b, 2);
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        for j in 0..8 {
            let (sc, m) = k_scale_min(scales, j);
            let (dj, mj) = (d * sc as f32, dmin * m as f32);
            for k in 0..32 {
                let ql = (qs[32 * (j / 2) + k] >> (4 * (j % 2))) & 0xf;
                let hb = (qh[k] >> j) & 1;
                o[32 * j + k] = dj * (ql | (hb << 4)) as f32 - mj;
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
