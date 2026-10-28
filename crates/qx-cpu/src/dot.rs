//! Row dot products of quantized weights with quantized activations. Each format has a scalar version
//! (the reference) and an AVX-512 VNNI version; both sum the same integers and scale them the same way.
use crate::act::{Q80, Q8K};
use qx_quant::tables::{FP4_VALUES2, IQ2S_GRID, IQ3S_GRID, IQ4NL_VALUES};
use qx_quant::UE4M3_HALF;

/// |FP4_VALUES2|: the E2M1 magnitudes (doubled) by code; codes 8..15 are the negatives of 0..7.
static FP4_MAG: [u8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12];

#[inline]
fn f16(b: &[u8], o: usize) -> f32 {
    half::f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

// ---- scalar references -------------------------------------------------------------------------

/// IQ2_S row (82-byte blocks) . Q8_K. Block value: d * dq / 8 * sum over 16-element sub-blocks of
/// (1 + 2*scale) * isum.
pub fn iq2s_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(82).zip(x) {
        let d = f16(b, 0);
        let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
        let mut sum = 0i32;
        for sub in 0..16 {
            let sc = ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) as i32;
            let mut isum = 0i32;
            for g in 2 * sub..2 * sub + 2 {
                let grid = IQ2S_GRID[qs[g] as usize | ((((qh[g / 4] >> (2 * (g % 4))) & 3) as usize) << 8)];
                for e in 0..8 {
                    let m = ((grid >> (8 * e)) & 0xff) as i32;
                    let v = q.qs[8 * g + e] as i32;
                    isum += if (signs[g] >> e) & 1 != 0 { -m * v } else { m * v };
                }
            }
            sum += (2 * sc + 1) * isum;
        }
        acc += d * q.d * sum as f32 * 0.125;
    }
    acc
}

/// IQ3_S row (110-byte blocks) . Q8_K. Block value: d * dq * sum over 32-element sub-blocks of (1 + 2*scale) * isum.
pub fn iq3s_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(110).zip(x) {
        let d = f16(b, 0);
        let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
        let mut sum = 0i32;
        for sub in 0..8 {
            let sc = ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) as i32;
            let mut isum = 0i32;
            for g in 4 * sub..4 * sub + 4 {
                for half in 0..2 {
                    let k = 2 * g + half;
                    let grid = IQ3S_GRID[qs[k] as usize | ((((qh[k / 8] >> (k % 8)) & 1) as usize) << 8)];
                    for j in 0..4 {
                        let e = 4 * half + j;
                        let m = ((grid >> (8 * j)) & 0xff) as i32;
                        let v = q.qs[8 * g + e] as i32;
                        isum += if (signs[g] >> e) & 1 != 0 { -m * v } else { m * v };
                    }
                }
            }
            sum += (2 * sc + 1) * isum;
        }
        acc += d * q.d * sum as f32;
    }
    acc
}

/// IQ4_NL row (18-byte blocks) . Q8_0.
pub fn iq4nl_q80_scalar(row: &[u8], x: &[Q80]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(18).zip(x) {
        let d = f16(b, 0);
        let mut isum = 0i32;
        for j in 0..16 {
            isum += IQ4NL_VALUES[(b[2 + j] & 0xf) as usize] as i32 * q.qs[j] as i32;
            isum += IQ4NL_VALUES[(b[2 + j] >> 4) as usize] as i32 * q.qs[j + 16] as i32;
        }
        acc += d * q.d * isum as f32;
    }
    acc
}

/// The sum of 16 sub-block values in the fixed pairwise order every NVFP4 kernel uses (CPU scalar, CPU AVX-512
/// and the GPU's shuffle tree): pairs (s, s^1), then (s, s^2), (s, s^4), (s, s^8).
#[inline]
pub fn tree16(v: &[f32; 16]) -> f32 {
    let mut t = *v;
    let mut step = 1;
    while step < 16 {
        for s in 0..16 {
            if s & step == 0 {
                t[s] += t[s ^ step];
            }
        }
        step *= 2;
    }
    t[0]
}

/// NVFP4 row (36-byte blocks of 64: 4 UE4M3 sub-block scales, 32 code bytes) . Q8_K. Per 256: the 16 sub-block
/// values `d_s * isum_s` summed by `tree16`, times the activation scale; chunks accumulated in order. The
/// per-tensor scale is applied by the caller.
pub fn nvfp4_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (chunk, q) in row.chunks_exact(144).zip(x) {
        let mut v = [0f32; 16];
        for (s, vs) in v.iter_mut().enumerate() {
            let b = &chunk[36 * (s / 4)..36 * (s / 4) + 36];
            let d = UE4M3_HALF[b[s % 4] as usize];
            let codes = &b[4 + 8 * (s % 4)..4 + 8 * (s % 4) + 8];
            let mut isum = 0i32;
            for (j, &c) in codes.iter().enumerate() {
                isum += FP4_VALUES2[(c & 0xf) as usize] as i32 * q.qs[16 * s + j] as i32;
                isum += FP4_VALUES2[(c >> 4) as usize] as i32 * q.qs[16 * s + j + 8] as i32;
            }
            *vs = d * isum as f32;
        }
        acc += q.d * tree16(&v);
    }
    acc
}

/// NVFP4 row . Q8_0: Q8_0 block b covers sub-blocks 2b and 2b + 1; block value
/// `dq * (d_a * isum_a + d_b * isum_b)`, blocks accumulated in order.
pub fn nvfp4_q80_scalar(row: &[u8], x: &[Q80]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in x.iter().enumerate() {
        let blk = &row[36 * (b / 2)..36 * (b / 2) + 36];
        let a0 = 2 * (b % 2);
        let (da, db) = (UE4M3_HALF[blk[a0] as usize], UE4M3_HALF[blk[a0 + 1] as usize]);
        let codes = &blk[4 + 8 * a0..4 + 8 * a0 + 16];
        let (mut ia, mut ib) = (0i32, 0i32);
        for j in 0..8 {
            ia += FP4_VALUES2[(codes[j] & 0xf) as usize] as i32 * q.qs[j] as i32;
            ia += FP4_VALUES2[(codes[j] >> 4) as usize] as i32 * q.qs[j + 8] as i32;
            ib += FP4_VALUES2[(codes[8 + j] & 0xf) as usize] as i32 * q.qs[16 + j] as i32;
            ib += FP4_VALUES2[(codes[8 + j] >> 4) as usize] as i32 * q.qs[24 + j] as i32;
        }
        acc += q.d * (da * ia as f32 + db * ib as f32);
    }
    acc
}

// ---- AVX-512 ------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// 64 weight magnitudes (one byte each) from 8 IQ2_S grid entries; groups 8*i .. 8*i+8.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn hsum_epi32(v: __m512i) -> i32 {
        _mm512_reduce_add_epi32(v)
    }

    /// IQ2_S . Q8_K, 64 elements (8 groups, 4 sub-blocks of 16) per step.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq2s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(82).zip(x) {
            let d = f16(b, 0);
            let p = b.as_ptr();
            let (qs, signs, qh, scales) = (p.add(2), p.add(34), p.add(66), p.add(74));
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let mut g8 = [0u64; 8];
                for i in 0..8 {
                    let g = 8 * step + i;
                    g8[i] = IQ2S_GRID[*qs.add(g) as usize | ((((*qh.add(g / 4) >> (2 * (g % 4))) & 3) as usize) << 8)];
                }
                let w = _mm512_loadu_si512(g8.as_ptr() as *const _);
                let sgn = (signs.add(8 * step) as *const u64).read_unaligned();
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, sgn, _mm512_setzero_si512(), a);
                // 16 lanes of 4-element sums; sub-block s of this step covers lanes 4s .. 4s+4.
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                let s0 = *scales.add(2 * step);
                let s1 = *scales.add(2 * step + 1);
                let sc = [(s0 & 0xf) as i32, (s0 >> 4) as i32, (s1 & 0xf) as i32, (s1 >> 4) as i32];
                let scv = _mm512_set_epi32(
                    2 * sc[3] + 1, 2 * sc[3] + 1, 2 * sc[3] + 1, 2 * sc[3] + 1,
                    2 * sc[2] + 1, 2 * sc[2] + 1, 2 * sc[2] + 1, 2 * sc[2] + 1,
                    2 * sc[1] + 1, 2 * sc[1] + 1, 2 * sc[1] + 1, 2 * sc[1] + 1,
                    2 * sc[0] + 1, 2 * sc[0] + 1, 2 * sc[0] + 1, 2 * sc[0] + 1,
                );
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ2_S . Q8_K with the 8 grid indices of a step computed in vector registers and fetched by one gather.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq2s_q8k_gather(row: &[u8], x: &[Q8K]) -> f32 {
        let grid = IQ2S_GRID.as_ptr() as *const i64;
        // qh byte (g/4) holds 2 bits per group at 2*(g%4)
        let shifts = _mm512_set_epi64(6, 4, 2, 0, 6, 4, 2, 0);
        let three = _mm512_set1_epi64(3);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(82).zip(x) {
            let d = f16(b, 0);
            let p = b.as_ptr();
            let (qs, signs, qh, scales) = (p.add(2), p.add(34), p.add(66), p.add(74));
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let lo = _mm512_cvtepu8_epi64(_mm_loadl_epi64(qs.add(8 * step) as *const _));
                let h0 = *qh.add(2 * step) as i64;
                let h1 = *qh.add(2 * step + 1) as i64;
                let hv = _mm512_set_epi64(h1, h1, h1, h1, h0, h0, h0, h0);
                let hi = _mm512_and_si512(_mm512_srlv_epi64(hv, shifts), three);
                let idx = _mm512_or_si512(lo, _mm512_slli_epi64(hi, 8));
                let w = _mm512_i64gather_epi64::<8>(idx, grid);
                let sgn = (signs.add(8 * step) as *const u64).read_unaligned();
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, sgn, _mm512_setzero_si512(), a);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                let s0 = *scales.add(2 * step) as i32;
                let s1 = *scales.add(2 * step + 1) as i32;
                let (c0, c1, c2, c3) = (2 * (s0 & 0xf) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 0xf) + 1, 2 * (s1 >> 4) + 1);
                let scv = _mm512_set_epi32(c3, c3, c3, c3, c2, c2, c2, c2, c1, c1, c1, c1, c0, c0, c0, c0);
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ3_S . Q8_K, 64 elements (8 groups, 2 sub-blocks of 32) per step.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq3s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(110).zip(x) {
            let d = f16(b, 0);
            let p = b.as_ptr();
            let (qs, qh, signs, scales) = (p.add(2), p.add(66), p.add(74), p.add(106));
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let mut g16 = [0u32; 16];
                for i in 0..16 {
                    let k = 16 * step + i;
                    g16[i] = IQ3S_GRID[*qs.add(k) as usize | ((((*qh.add(k / 8) >> (k % 8)) & 1) as usize) << 8)];
                }
                let w = _mm512_loadu_si512(g16.as_ptr() as *const _);
                let sgn = (signs.add(8 * step) as *const u64).read_unaligned();
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, sgn, _mm512_setzero_si512(), a);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                let s = *scales.add(step);
                let (c0, c1) = (2 * (s & 0xf) as i32 + 1, 2 * (s >> 4) as i32 + 1);
                let scv = _mm512_set_epi32(c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0, c0, c0, c0, c0);
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32;
        }
        acc
    }

    /// NVFP4 . Q8_K, bit-equal to `nvfp4_q8k_scalar`: integer sub-block sums by VNNI, the 16 values of a chunk
    /// in a vector, the pairwise tree by lane permutes.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
    pub unsafe fn nvfp4_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(FP4_MAG.as_ptr() as *const _));
        let m4 = _mm256_set1_epi8(0xf);
        let bit3 = _mm512_set1_epi8(8);
        // [lo0 lo1 lo2 lo3 | hi0 hi1 hi2 hi3] qwords to element order [lo0 hi0 lo1 hi1 ...]
        let qperm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
        // lanes 0, 4, 8, 12 of a, then of b
        let pick4 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
        let lo8 = _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0);
        let mut acc = 0f32;
        for (chunk, q) in row.chunks_exact(144).zip(x) {
            let mut sums = [_mm512_setzero_si512(); 4];
            let mut dv = [0f32; 16];
            for (blk, sum) in sums.iter_mut().enumerate() {
                let b = chunk.as_ptr().add(36 * blk);
                for s in 0..4 {
                    dv[4 * blk + s] = UE4M3_HALF[*b.add(s) as usize];
                }
                let codes = _mm256_loadu_si256(b.add(4) as *const _);
                let lo = _mm256_and_si256(codes, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), m4);
                let c = _mm512_permutexvar_epi64(qperm, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
                let mag = _mm512_shuffle_epi8(lut, c);
                let neg = _mm512_test_epi8_mask(c, bit3);
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * blk) as *const _);
                let a = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
                *sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, a);
            }
            // lane 4s of each block vector: that sub-block's integer sum
            let r = |v: __m512i| {
                let t = _mm512_add_epi32(v, _mm512_srli_epi64(v, 32));
                _mm512_add_epi32(t, _mm512_shuffle_epi32::<0x4e>(t))
            };
            let i01 = _mm512_permutex2var_epi32(r(sums[0]), pick4, r(sums[1]));
            let i23 = _mm512_permutex2var_epi32(r(sums[2]), pick4, r(sums[3]));
            let all = _mm512_permutex2var_epi32(i01, lo8, i23);
            let v = _mm512_mul_ps(_mm512_loadu_ps(dv.as_ptr()), _mm512_cvtepi32_ps(all));
            let v = _mm512_add_ps(v, _mm512_permute_ps::<0b10_11_00_01>(v));
            let v = _mm512_add_ps(v, _mm512_permute_ps::<0b01_00_11_10>(v));
            let v = _mm512_add_ps(v, _mm512_shuffle_f32x4::<0b10_11_00_01>(v, v));
            let v = _mm512_add_ps(v, _mm512_shuffle_f32x4::<0b01_00_11_10>(v, v));
            acc += q.d * _mm512_cvtss_f32(v);
        }
        acc
    }

    /// NVFP4 . Q8_0, bit-equal to `nvfp4_q80_scalar`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
    pub unsafe fn nvfp4_q80(row: &[u8], x: &[Q80]) -> f32 {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(FP4_MAG.as_ptr() as *const _));
        let m4 = _mm_set1_epi8(0xf);
        let bit3 = _mm256_set1_epi8(8);
        let hsum = |v: __m128i| {
            let t = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
            let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0b10_11_00_01>(t));
            _mm_cvtsi128_si32(t)
        };
        let mut acc = 0f32;
        for (b, q) in x.iter().enumerate() {
            let blk = row.as_ptr().add(36 * (b / 2));
            let a0 = 2 * (b % 2);
            let (da, db) = (UE4M3_HALF[*blk.add(a0) as usize], UE4M3_HALF[*blk.add(a0 + 1) as usize]);
            let codes = _mm_loadu_si128(blk.add(4 + 8 * a0) as *const _);
            let lo = _mm_and_si128(codes, m4);
            let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), m4);
            // element order: sub a low nibbles, sub a high, sub b low, sub b high
            let c = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            let mag = _mm256_shuffle_epi8(lut, c);
            let neg = _mm256_test_epi8_mask(c, bit3);
            let a = _mm256_loadu_si256(q.qs.as_ptr() as *const _);
            let a = _mm256_mask_sub_epi8(a, neg, _mm256_setzero_si256(), a);
            let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), mag, a);
            let (ia, ib) = (hsum(_mm256_castsi256_si128(dp)), hsum(_mm256_extracti128_si256::<1>(dp)));
            acc += q.d * (da * ia as f32 + db * ib as f32);
        }
        acc
    }

    /// IQ4_NL . Q8_0, two 32-element blocks per step (one 64-byte activation load).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
    pub unsafe fn iq4nl_q80(row: &[u8], x: &[Q80]) -> f32 {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(IQ4NL_VALUES.as_ptr() as *const _));
        let m4 = _mm_set1_epi8(0xf);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(18).zip(x) {
            let d = f16(b, 0);
            let raw = _mm_loadu_si128(b.as_ptr().add(2) as *const _);
            let lo = _mm_and_si128(raw, m4);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            let idx = _mm256_set_m128i(hi, lo); // elements 0..16 low nibbles, 16..32 high nibbles
            let w = _mm256_shuffle_epi8(_mm512_castsi512_si256(lut), idx);
            let a = _mm256_loadu_si256(q.qs.as_ptr() as *const _);
            // signed x signed via |w| . (a * sign(w))
            let wabs = _mm256_abs_epi8(w);
            let a = _mm256_sign_epi8(a, w);
            let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), wabs, a);
            let s = _mm256_extracti128_si256(dp, 1);
            let t = _mm_add_epi32(_mm256_castsi256_si128(dp), s);
            let t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b01_00_11_10));
            let t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b10_11_00_01));
            acc += d * q.d * _mm_cvtsi128_si32(t) as f32;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::act::{quantize_q8_0, quantize_q8_k};

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Random blocks with a sane f16 scale in the first two bytes of each block.
    fn blocks(n: usize, bb: usize, seed: &mut u64) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n * bb).map(|_| rng(seed) as u8).collect();
        for b in v.chunks_exact_mut(bb) {
            let d = half::f16::from_f32(((rng(seed) % 1000) as f32 + 1.0) * 1e-4);
            b[..2].copy_from_slice(&d.to_le_bytes());
        }
        v
    }

    fn act(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| ((rng(seed) % 20001) as f32 - 10000.0) * 1e-3).collect()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1e-3)
    }

    #[test]
    fn iq2s_avx_matches_scalar() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..200 {
            let w = blocks(10, 82, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq2s_q8k_scalar(&w, &q), unsafe { avx512::iq2s_q8k(&w, &q) });
            assert!(close(a, b), "{a} vs {b}");
            let c = unsafe { avx512::iq2s_q8k_gather(&w, &q) };
            assert!(close(a, c), "gather {a} vs {c}");
        }
    }

    #[test]
    fn iq3s_avx_matches_scalar() {
        let mut s = 0x0fed_cba9_8765_4321u64;
        for _ in 0..200 {
            let w = blocks(10, 110, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq3s_q8k_scalar(&w, &q), unsafe { avx512::iq3s_q8k(&w, &q) });
            assert!(close(a, b), "{a} vs {b}");
        }
    }

    #[test]
    fn iq4nl_avx_matches_scalar() {
        let mut s = 0x5555_aaaa_3333_ccccu64;
        for _ in 0..200 {
            let w = blocks(20, 18, &mut s);
            let mut q = Vec::new();
            quantize_q8_0(&act(640, &mut s), &mut q);
            let (a, b) = (iq4nl_q80_scalar(&w, &q), unsafe { avx512::iq4nl_q80(&w, &q) });
            assert!(close(a, b), "{a} vs {b}");
        }
    }

    /// Random NVFP4 blocks: scale bytes below 0x7f (the NaN code decodes to 0 either way).
    fn nvfp4_blocks(n: usize, seed: &mut u64) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n * 36).map(|_| rng(seed) as u8).collect();
        for b in v.chunks_exact_mut(36) {
            for s in 0..4 {
                b[s] = 0x20 + (rng(seed) % 0x30) as u8;
            }
        }
        v
    }

    #[test]
    fn nvfp4_avx_equals_scalar() {
        let mut s = 0x2468_ace0_1357_9bdfu64;
        for _ in 0..200 {
            let w = nvfp4_blocks(40, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (nvfp4_q8k_scalar(&w, &q), unsafe { avx512::nvfp4_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "q8k {a} vs {b}");
            let w = nvfp4_blocks(10, &mut s);
            let mut q0 = Vec::new();
            quantize_q8_0(&act(640, &mut s), &mut q0);
            let (a, b) = (nvfp4_q80_scalar(&w, &q0), unsafe { avx512::nvfp4_q80(&w, &q0) });
            assert!(a.to_bits() == b.to_bits(), "q80 {a} vs {b}");
        }
    }

    #[test]
    fn nvfp4_scalar_matches_dequant() {
        let mut s = 0x1357_9bdf_2468_ace0u64;
        let w = nvfp4_blocks(40, &mut s);
        let x = act(2560, &mut s);
        let mut qk = Vec::new();
        quantize_q8_k(&x, &mut qk);
        let xk: Vec<f32> = qk.iter().flat_map(|b| b.qs.iter().map(move |&v| v as f32 * b.d)).collect();
        let mut wf = vec![0f32; 2560];
        qx_quant::dequant_nvfp4(&w, &mut wf);
        let r: f64 = wf.iter().zip(&xk).map(|(a, b)| *a as f64 * *b as f64).sum();
        let got = nvfp4_q8k_scalar(&w, &qk);
        assert!(close(r as f32, got), "q8k {r} {got}");
        let w = nvfp4_blocks(10, &mut s);
        let mut q0 = Vec::new();
        quantize_q8_0(&x[..640], &mut q0);
        let x0q: Vec<f32> = q0.iter().flat_map(|b| b.qs.iter().map(move |&v| v as f32 * b.d)).collect();
        let mut wf = vec![0f32; 640];
        qx_quant::dequant_nvfp4(&w, &mut wf);
        let r: f64 = wf.iter().zip(&x0q).map(|(a, b)| *a as f64 * *b as f64).sum();
        let got = nvfp4_q80_scalar(&w, &q0);
        assert!(close(r as f32, got), "q80 {r} {got}");
    }

    /// The scalar integer dots equal dequantize-then-dot on the rounded activations.
    #[test]
    fn scalar_matches_dequant() {
        let mut s = 0x9999_1111_7777_3333u64;
        let w2 = blocks(10, 82, &mut s);
        let w3 = blocks(10, 110, &mut s);
        let w4 = blocks(20, 18, &mut s);
        let x = act(2560, &mut s);
        let mut qk = Vec::new();
        quantize_q8_k(&x, &mut qk);
        let xk: Vec<f32> = qk.iter().flat_map(|b| b.qs.iter().map(move |&v| v as f32 * b.d)).collect();
        let mut wf = vec![0f32; 2560];
        qx_quant::dequant_iq2_s(&w2, &mut wf);
        let r2: f32 = wf.iter().zip(&xk).map(|(a, b)| a * b).sum();
        assert!(close(r2, iq2s_q8k_scalar(&w2, &qk)), "iq2s {r2} {}", iq2s_q8k_scalar(&w2, &qk));
        qx_quant::dequant_iq3_s(&w3, &mut wf);
        let r3: f32 = wf.iter().zip(&xk).map(|(a, b)| a * b).sum();
        assert!(close(r3, iq3s_q8k_scalar(&w3, &qk)), "iq3s {r3} {}", iq3s_q8k_scalar(&w3, &qk));
        let x0 = &x[..640];
        let mut q0 = Vec::new();
        quantize_q8_0(x0, &mut q0);
        let x0q: Vec<f32> = q0.iter().flat_map(|b| b.qs.iter().map(move |&v| v as f32 * b.d)).collect();
        let mut wf4 = vec![0f32; 640];
        qx_quant::dequant_iq4_nl(&w4, &mut wf4);
        let r4: f32 = wf4.iter().zip(&x0q).map(|(a, b)| a * b).sum();
        assert!(close(r4, iq4nl_q80_scalar(&w4, &q0)), "iq4nl {r4} {}", iq4nl_q80_scalar(&w4, &q0));
    }
}
