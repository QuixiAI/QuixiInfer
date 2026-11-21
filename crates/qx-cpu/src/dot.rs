//! Row dot products of quantized weights with quantized activations. Each format has a scalar version
//! (the reference) and an AVX-512 VNNI or AVX2 version; all of them sum the same integers and scale them the same
//! way, so a row's value is bit-identical whichever kernel set runs.
use crate::act::{Q80, Q8K};
use qx_quant::tables::{FP4_VALUES2, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES, KSIGNS};
use qx_quant::UE4M3_HALF;
use std::sync::OnceLock;

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

#[inline]
fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// IQ2_XXS row (66-byte blocks) . Q8_K. Per 32-element sub-block two u32: four grid indices, then four 7-bit
/// `KSIGNS` indices and the 4-bit scale. Block value: d * dq / 8 * sum of (1 + 2*scale) * isum.
pub fn iq2xxs_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(66).zip(x) {
        let d = f16(b, 0);
        let mut sum = 0i32;
        for i in 0..8 {
            let a = u32le(b, 2 + 8 * i);
            let s = u32le(b, 6 + 8 * i);
            let mut isum = 0i32;
            for g in 0..4 {
                let grid = IQ2XXS_GRID[((a >> (8 * g)) & 0xff) as usize];
                let sg = KSIGNS[((s >> (7 * g)) & 0x7f) as usize];
                for e in 0..8 {
                    let m = ((grid >> (8 * e)) & 0xff) as i32;
                    let v = q.qs[32 * i + 8 * g + e] as i32;
                    isum += if (sg >> e) & 1 != 0 { -m * v } else { m * v };
                }
            }
            sum += (2 * (s >> 28) as i32 + 1) * isum;
        }
        acc += d * q.d * sum as f32 * 0.125;
    }
    acc
}

/// IQ2_XS row (74-byte blocks) . Q8_K: u16 per 8 elements (9-bit grid index, 7-bit sign index), scale nibbles
/// per 16. Block value: d * dq / 8 * sum of (1 + 2*scale) * isum.
pub fn iq2xs_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(74).zip(x) {
        let d = f16(b, 0);
        let scales = &b[66..74];
        let mut sum = 0i32;
        for sub in 0..16 {
            let sc = ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) as i32;
            let mut isum = 0i32;
            for g in 2 * sub..2 * sub + 2 {
                let qg = u16le(b, 2 + 2 * g);
                let grid = IQ2XS_GRID[(qg & 511) as usize];
                let sg = KSIGNS[(qg >> 9) as usize];
                for e in 0..8 {
                    let m = ((grid >> (8 * e)) & 0xff) as i32;
                    let v = q.qs[8 * g + e] as i32;
                    isum += if (sg >> e) & 1 != 0 { -m * v } else { m * v };
                }
            }
            sum += (2 * sc + 1) * isum;
        }
        acc += d * q.d * sum as f32 * 0.125;
    }
    acc
}

/// IQ3_XXS row (98-byte blocks) . Q8_K: a grid byte per 4 elements, then per 32-element sub-block a u32 of four
/// 7-bit sign indices and the 4-bit scale. Block value: d * dq / 4 * sum of (1 + 2*scale) * isum.
pub fn iq3xxs_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(98).zip(x) {
        let d = f16(b, 0);
        let qs = &b[2..66];
        let mut sum = 0i32;
        for i in 0..8 {
            let s = u32le(b, 66 + 4 * i);
            let mut isum = 0i32;
            for g in 0..4 {
                let sg = KSIGNS[((s >> (7 * g)) & 0x7f) as usize];
                for half in 0..2 {
                    let grid = IQ3XXS_GRID[qs[8 * i + 2 * g + half] as usize];
                    for j in 0..4 {
                        let e = 4 * half + j;
                        let m = ((grid >> (8 * j)) & 0xff) as i32;
                        let v = q.qs[32 * i + 8 * g + e] as i32;
                        isum += if (sg >> e) & 1 != 0 { -m * v } else { m * v };
                    }
                }
            }
            sum += (2 * (s >> 28) as i32 + 1) * isum;
        }
        acc += d * q.d * sum as f32 * 0.25;
    }
    acc
}

/// IQ1_S row (50-byte blocks) . Q8_K. Grid values are -1, 0, 1 (stored as 0, 1, 2) plus a per-sub-block offset of
/// +-1/8, so with `isum` the grid dot and `qsum` the activation sum the sub-block is exactly
/// `(1 + 2*scale) * (8*isum +- qsum) / 8`. Block value: d * dq / 8 * the sum over sub-blocks.
pub fn iq1s_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(50).zip(x) {
        let d = f16(b, 0);
        let qs = &b[2..34];
        let mut sum = 0i32;
        for i in 0..8 {
            let qh = u16le(b, 34 + 2 * i);
            let (mut isum, mut qsum) = (0i32, 0i32);
            for g in 0..4 {
                let grid = IQ1S_GRID[qs[4 * i + g] as usize | ((((qh >> (3 * g)) & 7) as usize) << 8)];
                for e in 0..8 {
                    let v = q.qs[32 * i + 8 * g + e] as i32;
                    isum += (((grid >> (8 * e)) & 0xff) as i32 - 1) * v;
                    qsum += v;
                }
            }
            let val = 8 * isum + if qh & 0x8000 == 0 { qsum } else { -qsum };
            sum += (2 * ((qh >> 12) & 7) as i32 + 1) * val;
        }
        acc += d * q.d * sum as f32 * 0.125;
    }
    acc
}

/// Q2_K row (84-byte blocks) . Q8_K: 2-bit values with a 4-bit scale and 4-bit min per 16 elements.
/// Block value: dq * (d * sum of scale * isum - dmin * sum of min * qsum).
pub fn q2k_q8k_scalar(row: &[u8], x: &[Q8K]) -> f32 {
    let mut acc = 0f32;
    for (b, q) in row.chunks_exact(84).zip(x) {
        let scales = &b[0..16];
        let qs = &b[16..80];
        let d = f16(b, 80);
        let dmin = f16(b, 82);
        let (mut sum, mut msum) = (0i32, 0i32);
        for j in 0..16 {
            let (mut isum, mut qsum) = (0i32, 0i32);
            for k in 0..16 {
                let e = 16 * j + k;
                let w = ((qs[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3) as i32;
                let v = q.qs[e] as i32;
                isum += w * v;
                qsum += v;
            }
            sum += (scales[j] & 0xf) as i32 * isum;
            msum += (scales[j] >> 4) as i32 * qsum;
        }
        acc += q.d * (d * sum as f32 - dmin * msum as f32);
    }
    acc
}

// ---- dispatch -----------------------------------------------------------------------------------

/// Which kernel set the CPU runs. AVX-512 is chosen only when every feature any AVX-512 kernel lists is present
/// (f, bw, vnni and vl), so no AVX-512 kernel can be reached on a CPU that lacks one of its instructions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Isa {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
        Isa::Avx512
    } else if is_x86_feature_detected!("avx2") {
        Isa::Avx2
    } else {
        Isa::Scalar
    }
}

/// The kernel set this CPU runs: detected on first use and cached, so a call is an atomic load and a branch.
///
/// `QX_CPU_ISA` (`scalar`, `avx2` or `avx512`) overrides the detected set for A/B runs and for bisecting a wrong
/// answer on a user's machine. It is read here, once, so every later call sees the same set. A set the CPU cannot
/// run is refused, not tried: one line on stderr names the set that runs instead.
pub fn isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        static ISA: OnceLock<Isa> = OnceLock::new();
        *ISA.get_or_init(|| {
            let detected = detect();
            let want = match std::env::var("QX_CPU_ISA").ok().as_deref() {
                None => return detected,
                Some("scalar") => Isa::Scalar,
                Some("avx2") => Isa::Avx2,
                Some("avx512") => Isa::Avx512,
                Some(other) => {
                    eprintln!("QX_CPU_ISA={other} is not scalar, avx2 or avx512; running {detected:?}");
                    return detected;
                }
            };
            let supported = match want {
                Isa::Scalar => true,
                Isa::Avx2 => detected != Isa::Scalar,
                Isa::Avx512 => detected == Isa::Avx512,
            };
            if supported {
                eprintln!("QX_CPU_ISA: running {want:?}");
                want
            } else {
                eprintln!("QX_CPU_ISA asks for {want:?}, which this CPU lacks; running {detected:?}");
                detected
            }
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Isa::Scalar
    }
}

pub type Q8kDot = unsafe fn(&[u8], &[Q8K]) -> f32;
pub type Q80Dot = unsafe fn(&[u8], &[Q80]) -> f32;

/// One entry per format the pool dispatches. A job copies the table once, so the row loop pays only an indirect
/// call per row, never a feature detection. The AVX-512 IQ2_S entry is the gather kernel, the one the pool used
/// before dispatch existed; the AVX2 and scalar entries use the plain one.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub iq2s_q8k: Q8kDot,
    pub iq3s_q8k: Q8kDot,
    pub nvfp4_q8k: Q8kDot,
    pub iq2xxs_q8k: Q8kDot,
    pub iq2xs_q8k: Q8kDot,
    pub iq3xxs_q8k: Q8kDot,
    pub iq1s_q8k: Q8kDot,
    pub q2k_q8k: Q8kDot,
    pub nvfp4_q80: Q80Dot,
    pub iq4nl_q80: Q80Dot,
}

fn scalar_kernels() -> Kernels {
    Kernels {
        iq2s_q8k: iq2s_q8k_scalar,
        iq3s_q8k: iq3s_q8k_scalar,
        nvfp4_q8k: nvfp4_q8k_scalar,
        iq2xxs_q8k: iq2xxs_q8k_scalar,
        iq2xs_q8k: iq2xs_q8k_scalar,
        iq3xxs_q8k: iq3xxs_q8k_scalar,
        iq1s_q8k: iq1s_q8k_scalar,
        q2k_q8k: q2k_q8k_scalar,
        nvfp4_q80: nvfp4_q80_scalar,
        iq4nl_q80: iq4nl_q80_scalar,
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn kernels_for(isa: Isa) -> Kernels {
    match isa {
        Isa::Avx512 => avx512::kernels(),
        Isa::Avx2 => avx2::kernels(),
        Isa::Scalar => scalar_kernels(),
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn kernels_for(_isa: Isa) -> Kernels {
    scalar_kernels()
}

/// The detected kernel table, built once.
pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| kernels_for(isa()))
}

/// Whether the CPU itself has the AVX-512 features the kernels need, regardless of `QX_CPU_ISA`.
#[cfg(test)]
pub(crate) fn cpu_has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the CPU itself has AVX2, regardless of `QX_CPU_ISA`.
#[cfg(test)]
pub(crate) fn cpu_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 82-byte blocks.
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 82-byte blocks.
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 110-byte blocks.
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 144-byte chunks.
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `36 * (x.len() + 1) / 2` bytes.
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
            let blk = &row[36 * (b / 2)..36 * (b / 2) + 36];
            let a0 = 2 * (b % 2);
            let (da, db) = (UE4M3_HALF[blk[a0] as usize], UE4M3_HALF[blk[a0 + 1] as usize]);
            let codes = _mm_loadu_si128(blk.as_ptr().add(4 + 8 * a0) as *const _);
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
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 18-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
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

    /// (1 + 2*scale) per lane for two 32-element sub-blocks: lanes 0..8 the first, 8..16 the second.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn scv2(c0: i32, c1: i32) -> __m512i {
        _mm512_set_epi32(c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0, c0, c0, c0, c0)
    }

    /// Per lane for four 16-element sub-blocks (4 lanes each).
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn scv4(c: [i32; 4]) -> __m512i {
        _mm512_set_epi32(c[3], c[3], c[3], c[3], c[2], c[2], c[2], c[2], c[1], c[1], c[1], c[1], c[0], c[0], c[0], c[0])
    }

    /// The 8 sign bytes of two 32-element sub-blocks whose sign words are `s0` and `s1`, as a 64-bit lane mask.
    #[inline]
    fn signs2(s0: u32, s1: u32) -> u64 {
        let mut m = 0u64;
        for g in 0..4 {
            m |= (KSIGNS[((s0 >> (7 * g)) & 0x7f) as usize] as u64) << (8 * g);
            m |= (KSIGNS[((s1 >> (7 * g)) & 0x7f) as usize] as u64) << (32 + 8 * g);
        }
        m
    }

    /// IQ2_XXS . Q8_K, 64 elements (two sub-blocks of 32) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 66-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq2xxs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(66).zip(x) {
            let d = f16(b, 0);
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let (a0, s0) = (u32le(b, 2 + 16 * step), u32le(b, 6 + 16 * step));
                let (a1, s1) = (u32le(b, 10 + 16 * step), u32le(b, 14 + 16 * step));
                let mut g8 = [0u64; 8];
                for g in 0..4 {
                    g8[g] = IQ2XXS_GRID[((a0 >> (8 * g)) & 0xff) as usize];
                    g8[4 + g] = IQ2XXS_GRID[((a1 >> (8 * g)) & 0xff) as usize];
                }
                let w = _mm512_loadu_si512(g8.as_ptr() as *const _);
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, signs2(s0, s1), _mm512_setzero_si512(), a);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv2(2 * (s0 >> 28) as i32 + 1, 2 * (s1 >> 28) as i32 + 1)));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ2_XS . Q8_K, 64 elements (8 groups, 4 sub-blocks of 16) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 74-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq2xs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(74).zip(x) {
            let d = f16(b, 0);
            let scales = &b[66..74];
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let mut g8 = [0u64; 8];
                let mut sgn = 0u64;
                for i in 0..8 {
                    let qg = u16le(b, 2 + 2 * (8 * step + i));
                    g8[i] = IQ2XS_GRID[(qg & 511) as usize];
                    sgn |= (KSIGNS[(qg >> 9) as usize] as u64) << (8 * i);
                }
                let w = _mm512_loadu_si512(g8.as_ptr() as *const _);
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, sgn, _mm512_setzero_si512(), a);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                let (s0, s1) = (scales[2 * step] as i32, scales[2 * step + 1] as i32);
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv4([2 * (s0 & 0xf) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 0xf) + 1, 2 * (s1 >> 4) + 1])));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ3_XXS . Q8_K, 64 elements (two sub-blocks of 32) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 98-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq3xxs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(98).zip(x) {
            let d = f16(b, 0);
            let qs = &b[2..66];
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let mut g16 = [0u32; 16];
                for i in 0..16 {
                    g16[i] = IQ3XXS_GRID[qs[16 * step + i] as usize];
                }
                let (s0, s1) = (u32le(b, 66 + 8 * step), u32le(b, 70 + 8 * step));
                let w = _mm512_loadu_si512(g16.as_ptr() as *const _);
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let a = _mm512_mask_sub_epi8(a, signs2(s0, s1), _mm512_setzero_si512(), a);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(dp, scv2(2 * (s0 >> 28) as i32 + 1, 2 * (s1 >> 28) as i32 + 1)));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.25;
        }
        acc
    }

    /// IQ1_S . Q8_K, bit-equal to `iq1s_q8k_scalar`: the grid bytes (0, 1, 2) dotted as unsigned, the activation
    /// sum subtracted once and added or subtracted again for the +-1/8 offset, all in integers.
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 50-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn iq1s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let ones = _mm512_set1_epi8(1);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(50).zip(x) {
            let d = f16(b, 0);
            let qs = &b[2..34];
            let mut sumv = _mm512_setzero_si512();
            for step in 0..4 {
                let (h0, h1) = (u16le(b, 34 + 4 * step), u16le(b, 36 + 4 * step));
                let mut g8 = [0u64; 8];
                for g in 0..4 {
                    g8[g] = IQ1S_GRID[qs[8 * step + g] as usize | ((((h0 >> (3 * g)) & 7) as usize) << 8)];
                    g8[4 + g] = IQ1S_GRID[qs[8 * step + 4 + g] as usize | ((((h1 >> (3 * g)) & 7) as usize) << 8)];
                }
                let w = _mm512_loadu_si512(g8.as_ptr() as *const _);
                let a = _mm512_loadu_si512(q.qs.as_ptr().add(64 * step) as *const _);
                let bsum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a);
                let qsum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, a);
                // 8 * (bsum - qsum) +- qsum = 8 * bsum + (-8 +- 1) * qsum
                let (c0, c1) = (if h0 & 0x8000 == 0 { -7 } else { -9 }, if h1 & 0x8000 == 0 { -7 } else { -9 });
                let val = _mm512_add_epi32(_mm512_slli_epi32::<3>(bsum), _mm512_mullo_epi32(qsum, scv2(c0, c1)));
                sumv = _mm512_add_epi32(sumv, _mm512_mullo_epi32(val, scv2(2 * ((h0 >> 12) & 7) as i32 + 1, 2 * ((h1 >> 12) & 7) as i32 + 1)));
            }
            acc += d * q.d * hsum_epi32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// Q2_K . Q8_K, bit-equal to `q2k_q8k_scalar`: per 32-byte chunk and 2-bit plane, 32 values against 32
    /// activations (two sub-blocks of 16, lanes 0..4 and 4..8), scales and mins applied as integers.
    ///
    /// # Safety
    ///
    /// The CPU has AVX-512 F, BW, VNNI and VL, and `row` holds exactly `x.len()` 84-byte blocks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
    pub unsafe fn q2k_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let m3 = _mm256_set1_epi8(3);
        let ones = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(84).zip(x) {
            let scales = &b[0..16];
            let d = f16(b, 80);
            let dmin = f16(b, 82);
            let (mut sumv, mut minv) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for c in 0..2 {
                let bytes = _mm256_loadu_si256(b.as_ptr().add(16 + 32 * c) as *const _);
                for s in 0..4 {
                    let shifted = match s {
                        0 => bytes,
                        1 => _mm256_srli_epi16::<2>(bytes),
                        2 => _mm256_srli_epi16::<4>(bytes),
                        _ => _mm256_srli_epi16::<6>(bytes),
                    };
                    let w = _mm256_and_si256(shifted, m3);
                    let a = _mm256_loadu_si256(q.qs.as_ptr().add(128 * c + 32 * s) as *const _);
                    let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, a);
                    let qs = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, a);
                    let j = 8 * c + 2 * s;
                    let (sc0, sc1) = ((scales[j] & 0xf) as i32, (scales[j + 1] & 0xf) as i32);
                    let (m0, m1) = ((scales[j] >> 4) as i32, (scales[j + 1] >> 4) as i32);
                    sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp, _mm256_set_epi32(sc1, sc1, sc1, sc1, sc0, sc0, sc0, sc0)));
                    minv = _mm256_add_epi32(minv, _mm256_mullo_epi32(qs, _mm256_set_epi32(m1, m1, m1, m1, m0, m0, m0, m0)));
                }
            }
            let hsum = |v: __m256i| -> i32 {
                let t = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0b01_00_11_10>(t));
                let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0b10_11_00_01>(t));
                _mm_cvtsi128_si32(t)
            };
            acc += q.d * (d * hsum(sumv) as f32 - dmin * hsum(minv) as f32);
        }
        acc
    }

    /// The AVX-512 row of the dispatch table.
    pub fn kernels() -> super::Kernels {
        super::Kernels {
            iq2s_q8k: iq2s_q8k_gather,
            iq3s_q8k: iq3s_q8k,
            nvfp4_q8k: nvfp4_q8k,
            iq2xxs_q8k: iq2xxs_q8k,
            iq2xs_q8k: iq2xs_q8k,
            iq3xxs_q8k: iq3xxs_q8k,
            iq1s_q8k: iq1s_q8k,
            q2k_q8k: q2k_q8k,
            nvfp4_q80: nvfp4_q80,
            iq4nl_q80: iq4nl_q80,
        }
    }
}

// ---- AVX2 ---------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// A group's 8 sign bits as 8 control bytes for `_mm256_sign_epi8`: 0xff where the bit is set (negate), 0x01
    /// where clear (keep). Three shifts and masks spread bit e of the byte to bit 0 of byte e; multiplying by 0xff
    /// then fills each byte without a carry, since every byte is 0 or 1.
    #[inline]
    fn sign_ctl(s: u8) -> u64 {
        let mut x = s as u64;
        x = (x | (x << 28)) & 0x0000_000f_0000_000f;
        x = (x | (x << 14)) & 0x0003_0003_0003_0003;
        x = (x | (x << 7)) & 0x0101_0101_0101_0101;
        x.wrapping_mul(0xff) | 0x0101_0101_0101_0101
    }

    /// 32 magnitudes times 32 signed activations, four elements summed per i32 lane. Every weight byte is a grid
    /// magnitude (<= 62; IQ4_NL <= 127), so a pair of products is at most 2 * 127 * 127 = 32258 and cannot
    /// saturate the i16 of `maddubs`; `grid_magnitudes_cannot_saturate` holds the bound over the tables.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dp32(w: __m256i, a: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum32(v: __m256i) -> i32 {
        let t = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0b01_00_11_10>(t));
        let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0b10_11_00_01>(t));
        _mm_cvtsi128_si32(t)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lanes8(v: __m256i) -> [i32; 8] {
        std::mem::transmute::<__m256i, [i32; 8]>(v)
    }

    /// IQ2_S . Q8_K, 64 elements (8 groups, 4 sub-blocks of 16) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 82-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq2s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(82).zip(x) {
            let d = f16(b, 0);
            let p = b.as_ptr();
            let (qs, signs, qh, scales) = (p.add(2), p.add(34), p.add(66), p.add(74));
            let mut sumv = _mm256_setzero_si256();
            for step in 0..4 {
                let mut g8 = [0u64; 8];
                let mut ctl = [0u64; 8];
                for i in 0..8 {
                    let g = 8 * step + i;
                    g8[i] = IQ2S_GRID[*qs.add(g) as usize | ((((*qh.add(g / 4) >> (2 * (g % 4))) & 3) as usize) << 8)];
                    ctl[i] = sign_ctl(*signs.add(g));
                }
                let s0 = *scales.add(2 * step) as i32;
                let s1 = *scales.add(2 * step + 1) as i32;
                let (c0, c1) = (2 * (s0 & 0xf) + 1, 2 * (s0 >> 4) + 1);
                let (c2, c3) = (2 * (s1 & 0xf) + 1, 2 * (s1 >> 4) + 1);
                let a0 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step) as *const __m256i);
                let a1 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step + 32) as *const __m256i);
                let dp0 = dp32(_mm256_loadu_si256(g8.as_ptr() as *const __m256i), _mm256_sign_epi8(a0, _mm256_loadu_si256(ctl.as_ptr() as *const __m256i)));
                let dp1 = dp32(_mm256_loadu_si256(g8.as_ptr().add(4) as *const __m256i), _mm256_sign_epi8(a1, _mm256_loadu_si256(ctl.as_ptr().add(4) as *const __m256i)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp0, _mm256_set_epi32(c1, c1, c1, c1, c0, c0, c0, c0)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp1, _mm256_set_epi32(c3, c3, c3, c3, c2, c2, c2, c2)));
            }
            acc += d * q.d * hsum32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ3_S . Q8_K, 64 elements (8 groups, 2 sub-blocks of 32) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 110-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq3s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(110).zip(x) {
            let d = f16(b, 0);
            let p = b.as_ptr();
            let (qs, qh, signs, scales) = (p.add(2), p.add(66), p.add(74), p.add(106));
            let mut sumv = _mm256_setzero_si256();
            for step in 0..4 {
                let mut g16 = [0u32; 16];
                let mut ctl = [0u64; 8];
                for i in 0..16 {
                    let k = 16 * step + i;
                    g16[i] = IQ3S_GRID[*qs.add(k) as usize | ((((*qh.add(k / 8) >> (k % 8)) & 1) as usize) << 8)];
                }
                for i in 0..8 {
                    ctl[i] = sign_ctl(*signs.add(8 * step + i));
                }
                let a0 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step) as *const __m256i);
                let a1 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step + 32) as *const __m256i);
                let dp0 = dp32(_mm256_loadu_si256(g16.as_ptr() as *const __m256i), _mm256_sign_epi8(a0, _mm256_loadu_si256(ctl.as_ptr() as *const __m256i)));
                let dp1 = dp32(_mm256_loadu_si256(g16.as_ptr().add(8) as *const __m256i), _mm256_sign_epi8(a1, _mm256_loadu_si256(ctl.as_ptr().add(4) as *const __m256i)));
                let s = *scales.add(step) as i32;
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp0, _mm256_set1_epi32(2 * (s & 0xf) + 1)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp1, _mm256_set1_epi32(2 * (s >> 4) + 1)));
            }
            acc += d * q.d * hsum32(sumv) as f32;
        }
        acc
    }

    /// NVFP4 . Q8_K, equal to `nvfp4_q8k_scalar`: the sub-block integer sums by `maddubs`, the 16 `d_s * isum_s`
    /// values and their fixed pairwise tree the scalar way (16 float ops per chunk).
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 144-byte chunks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn nvfp4_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(FP4_MAG.as_ptr() as *const __m128i));
        let m4 = _mm256_set1_epi8(0xf);
        let bit3 = _mm256_set1_epi8(8);
        let one = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for (chunk, q) in row.chunks_exact(144).zip(x) {
            let mut isum = [0i32; 16];
            let mut dv = [0f32; 16];
            for blk in 0..4 {
                let b = chunk.as_ptr().add(36 * blk);
                for s in 0..4 {
                    dv[4 * blk + s] = UE4M3_HALF[*b.add(s) as usize];
                }
                let codes = _mm256_loadu_si256(b.add(4) as *const __m256i);
                let lo = _mm256_and_si256(codes, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(codes), m4);
                // per 128-bit lane unpacklo/hi pair [L0 H0], [L2 H2] and [L1 H1], [L3 H3]; one permute per
                // 256-bit register puts the four 16-element sub-blocks back in element order
                let t0 = _mm256_unpacklo_epi64(lo, hi);
                let t1 = _mm256_unpackhi_epi64(lo, hi);
                for (h, idx) in [_mm256_permute2x128_si256::<0x20>(t0, t1), _mm256_permute2x128_si256::<0x31>(t0, t1)].into_iter().enumerate() {
                    let mag = _mm256_shuffle_epi8(lut, idx);
                    let neg = _mm256_cmpeq_epi8(_mm256_and_si256(idx, bit3), bit3);
                    let a = _mm256_loadu_si256(q.qs.as_ptr().add(64 * blk + 32 * h) as *const __m256i);
                    let dp = lanes8(dp32(mag, _mm256_sign_epi8(a, _mm256_or_si256(neg, one))));
                    isum[4 * blk + 2 * h] = dp[0] + dp[1] + dp[2] + dp[3];
                    isum[4 * blk + 2 * h + 1] = dp[4] + dp[5] + dp[6] + dp[7];
                }
            }
            let mut v = [0f32; 16];
            for s in 0..16 {
                v[s] = dv[s] * isum[s] as f32;
            }
            acc += q.d * tree16(&v);
        }
        acc
    }

    /// NVFP4 . Q8_0, equal to `nvfp4_q80_scalar`.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `36 * (x.len() + 1) / 2` bytes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn nvfp4_q80(row: &[u8], x: &[Q80]) -> f32 {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(FP4_MAG.as_ptr() as *const __m128i));
        let m4 = _mm_set1_epi8(0xf);
        let bit3 = _mm256_set1_epi8(8);
        let one = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for (b, q) in x.iter().enumerate() {
            let blk = &row[36 * (b / 2)..36 * (b / 2) + 36];
            let a0 = 2 * (b % 2);
            let (da, db) = (UE4M3_HALF[blk[a0] as usize], UE4M3_HALF[blk[a0 + 1] as usize]);
            let codes = _mm_loadu_si128(blk.as_ptr().add(4 + 8 * a0) as *const __m128i);
            let lo = _mm_and_si128(codes, m4);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(codes), m4);
            // [lo 8 bytes | hi 8 bytes] is sub a (16 elements), then sub b
            let idx = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            let mag = _mm256_shuffle_epi8(lut, idx);
            let neg = _mm256_cmpeq_epi8(_mm256_and_si256(idx, bit3), bit3);
            let a = _mm256_loadu_si256(q.qs.as_ptr() as *const __m256i);
            let dp = lanes8(dp32(mag, _mm256_sign_epi8(a, _mm256_or_si256(neg, one))));
            let (ia, ib) = (dp[0] + dp[1] + dp[2] + dp[3], dp[4] + dp[5] + dp[6] + dp[7]);
            acc += q.d * (da * ia as f32 + db * ib as f32);
        }
        acc
    }

    /// IQ4_NL . Q8_0, equal to `iq4nl_q80_scalar`: signed table values as `|w| . (a * sign(w))`.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 18-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq4nl_q80(row: &[u8], x: &[Q80]) -> f32 {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(IQ4NL_VALUES.as_ptr() as *const __m128i));
        let m4 = _mm_set1_epi8(0xf);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(18).zip(x) {
            let d = f16(b, 0);
            let raw = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
            let lo = _mm_and_si128(raw, m4);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), m4);
            let idx = _mm256_set_m128i(hi, lo); // low nibbles 0..16, high nibbles 16..32
            let w = _mm256_shuffle_epi8(lut, idx);
            let a = _mm256_loadu_si256(q.qs.as_ptr() as *const __m256i);
            let dp = dp32(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
            acc += d * q.d * hsum32(dp) as f32;
        }
        acc
    }

    /// IQ2_XXS . Q8_K, one 32-element sub-block per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 66-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq2xxs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(66).zip(x) {
            let d = f16(b, 0);
            let mut sumv = _mm256_setzero_si256();
            for i in 0..8 {
                let a = u32le(b, 2 + 8 * i);
                let s = u32le(b, 6 + 8 * i);
                let mut g8 = [0u64; 4];
                let mut ctl = [0u64; 4];
                for g in 0..4 {
                    g8[g] = IQ2XXS_GRID[((a >> (8 * g)) & 0xff) as usize];
                    ctl[g] = sign_ctl(KSIGNS[((s >> (7 * g)) & 0x7f) as usize]);
                }
                let av = _mm256_loadu_si256(q.qs.as_ptr().add(32 * i) as *const __m256i);
                let dp = dp32(_mm256_loadu_si256(g8.as_ptr() as *const __m256i), _mm256_sign_epi8(av, _mm256_loadu_si256(ctl.as_ptr() as *const __m256i)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp, _mm256_set1_epi32(2 * (s >> 28) as i32 + 1)));
            }
            acc += d * q.d * hsum32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ2_XS . Q8_K, 64 elements (8 groups, 4 sub-blocks of 16) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 74-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq2xs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(74).zip(x) {
            let d = f16(b, 0);
            let scales = &b[66..74];
            let mut sumv = _mm256_setzero_si256();
            for step in 0..4 {
                let mut g8 = [0u64; 8];
                let mut ctl = [0u64; 8];
                for i in 0..8 {
                    let qg = u16le(b, 2 + 2 * (8 * step + i));
                    g8[i] = IQ2XS_GRID[(qg & 511) as usize];
                    ctl[i] = sign_ctl(KSIGNS[(qg >> 9) as usize]);
                }
                let s0 = scales[2 * step] as i32;
                let s1 = scales[2 * step + 1] as i32;
                let (c0, c1) = (2 * (s0 & 0xf) + 1, 2 * (s0 >> 4) + 1);
                let (c2, c3) = (2 * (s1 & 0xf) + 1, 2 * (s1 >> 4) + 1);
                let a0 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step) as *const __m256i);
                let a1 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step + 32) as *const __m256i);
                let dp0 = dp32(_mm256_loadu_si256(g8.as_ptr() as *const __m256i), _mm256_sign_epi8(a0, _mm256_loadu_si256(ctl.as_ptr() as *const __m256i)));
                let dp1 = dp32(_mm256_loadu_si256(g8.as_ptr().add(4) as *const __m256i), _mm256_sign_epi8(a1, _mm256_loadu_si256(ctl.as_ptr().add(4) as *const __m256i)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp0, _mm256_set_epi32(c1, c1, c1, c1, c0, c0, c0, c0)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp1, _mm256_set_epi32(c3, c3, c3, c3, c2, c2, c2, c2)));
            }
            acc += d * q.d * hsum32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// IQ3_XXS . Q8_K, 64 elements (two sub-blocks of 32) per step.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 98-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq3xxs_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(98).zip(x) {
            let d = f16(b, 0);
            let qs = &b[2..66];
            let mut sumv = _mm256_setzero_si256();
            for step in 0..4 {
                let mut g16 = [0u32; 16];
                let mut ctl = [0u64; 8];
                for i in 0..16 {
                    g16[i] = IQ3XXS_GRID[qs[16 * step + i] as usize];
                }
                let (s0, s1) = (u32le(b, 66 + 8 * step), u32le(b, 70 + 8 * step));
                for g in 0..4 {
                    ctl[g] = sign_ctl(KSIGNS[((s0 >> (7 * g)) & 0x7f) as usize]);
                    ctl[4 + g] = sign_ctl(KSIGNS[((s1 >> (7 * g)) & 0x7f) as usize]);
                }
                let a0 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step) as *const __m256i);
                let a1 = _mm256_loadu_si256(q.qs.as_ptr().add(64 * step + 32) as *const __m256i);
                let dp0 = dp32(_mm256_loadu_si256(g16.as_ptr() as *const __m256i), _mm256_sign_epi8(a0, _mm256_loadu_si256(ctl.as_ptr() as *const __m256i)));
                let dp1 = dp32(_mm256_loadu_si256(g16.as_ptr().add(8) as *const __m256i), _mm256_sign_epi8(a1, _mm256_loadu_si256(ctl.as_ptr().add(4) as *const __m256i)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp0, _mm256_set1_epi32(2 * (s0 >> 28) as i32 + 1)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp1, _mm256_set1_epi32(2 * (s1 >> 28) as i32 + 1)));
            }
            acc += d * q.d * hsum32(sumv) as f32 * 0.25;
        }
        acc
    }

    /// IQ1_S . Q8_K, equal to `iq1s_q8k_scalar`: the grid bytes (0, 1, 2) dotted as unsigned, the activation sum
    /// scaled by -7 or -9 for the +-1/8 offset, all in integers.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 50-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn iq1s_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let ones = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(50).zip(x) {
            let d = f16(b, 0);
            let qs = &b[2..34];
            let mut sumv = _mm256_setzero_si256();
            for i in 0..8 {
                let qh = u16le(b, 34 + 2 * i);
                let mut g8 = [0u64; 4];
                for g in 0..4 {
                    g8[g] = IQ1S_GRID[qs[4 * i + g] as usize | ((((qh >> (3 * g)) & 7) as usize) << 8)];
                }
                let a = _mm256_loadu_si256(q.qs.as_ptr().add(32 * i) as *const __m256i);
                let w = _mm256_loadu_si256(g8.as_ptr() as *const __m256i);
                let bsum = dp32(w, a);
                let qsum = dp32(ones, a);
                let c = if qh & 0x8000 == 0 { -7 } else { -9 };
                let val = _mm256_add_epi32(_mm256_slli_epi32::<3>(bsum), _mm256_mullo_epi32(qsum, _mm256_set1_epi32(c)));
                sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(val, _mm256_set1_epi32(2 * ((qh >> 12) & 7) as i32 + 1)));
            }
            acc += d * q.d * hsum32(sumv) as f32 * 0.125;
        }
        acc
    }

    /// Q2_K . Q8_K, equal to `q2k_q8k_scalar`: per 32-byte chunk and 2-bit plane, 32 values against 32
    /// activations (two sub-blocks of 16), scales and mins applied as integers.
    ///
    /// # Safety
    ///
    /// The CPU has AVX2, and `row` holds exactly `x.len()` 84-byte blocks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn q2k_q8k(row: &[u8], x: &[Q8K]) -> f32 {
        let m3 = _mm256_set1_epi8(3);
        let ones = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for (b, q) in row.chunks_exact(84).zip(x) {
            let scales = &b[0..16];
            let d = f16(b, 80);
            let dmin = f16(b, 82);
            let (mut sumv, mut minv) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for c in 0..2 {
                let bytes = _mm256_loadu_si256(b.as_ptr().add(16 + 32 * c) as *const __m256i);
                for s in 0..4 {
                    let shifted = match s {
                        0 => bytes,
                        1 => _mm256_srli_epi16::<2>(bytes),
                        2 => _mm256_srli_epi16::<4>(bytes),
                        _ => _mm256_srli_epi16::<6>(bytes),
                    };
                    let a = _mm256_loadu_si256(q.qs.as_ptr().add(128 * c + 32 * s) as *const __m256i);
                    let dp = dp32(_mm256_and_si256(shifted, m3), a);
                    let qs = dp32(ones, a);
                    let j = 8 * c + 2 * s;
                    let (sc0, sc1) = ((scales[j] & 0xf) as i32, (scales[j + 1] & 0xf) as i32);
                    let (m0, m1) = ((scales[j] >> 4) as i32, (scales[j + 1] >> 4) as i32);
                    sumv = _mm256_add_epi32(sumv, _mm256_mullo_epi32(dp, _mm256_set_epi32(sc1, sc1, sc1, sc1, sc0, sc0, sc0, sc0)));
                    minv = _mm256_add_epi32(minv, _mm256_mullo_epi32(qs, _mm256_set_epi32(m1, m1, m1, m1, m0, m0, m0, m0)));
                }
            }
            acc += q.d * (d * hsum32(sumv) as f32 - dmin * hsum32(minv) as f32);
        }
        acc
    }

    /// The AVX2 row of the dispatch table.
    pub fn kernels() -> super::Kernels {
        super::Kernels {
            iq2s_q8k,
            iq3s_q8k,
            nvfp4_q8k,
            iq2xxs_q8k,
            iq2xs_q8k,
            iq3xxs_q8k,
            iq1s_q8k,
            q2k_q8k,
            nvfp4_q80,
            iq4nl_q80,
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
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

    /// Whether the selected kernel set is AVX-512.
    fn have_avx512() -> bool {
        isa() == Isa::Avx512
    }

    fn have_avx2() -> bool {
        cpu_has_avx2()
    }

    fn skip_avx512(name: &str) -> bool {
        if !cpu_has_avx512() {
            eprintln!("skipping {name}: no AVX-512 F/BW/VNNI/VL");
            return true;
        }
        if !have_avx512() {
            eprintln!("skipping {name}: QX_CPU_ISA selects {:?}, not Avx512", isa());
            return true;
        }
        false
    }

    fn skip_avx2(name: &str) -> bool {
        if !have_avx2() {
            eprintln!("skipping {name}: no AVX2");
            return true;
        }
        false
    }

    #[test]
    fn iq2s_avx_matches_scalar() {
        if skip_avx512("iq2s_avx_matches_scalar") {
            return;
        }
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..200 {
            let w = blocks(10, 82, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq2s_q8k_scalar(&w, &q), unsafe { avx512::iq2s_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
            let c = unsafe { avx512::iq2s_q8k_gather(&w, &q) };
            assert!(a.to_bits() == c.to_bits(), "gather {a} vs {c}");
        }
    }

    #[test]
    fn iq3s_avx_matches_scalar() {
        if skip_avx512("iq3s_avx_matches_scalar") {
            return;
        }
        let mut s = 0x0fed_cba9_8765_4321u64;
        for _ in 0..200 {
            let w = blocks(10, 110, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq3s_q8k_scalar(&w, &q), unsafe { avx512::iq3s_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
        }
    }

    #[test]
    fn iq4nl_avx_matches_scalar() {
        if skip_avx512("iq4nl_avx_matches_scalar") {
            return;
        }
        let mut s = 0x5555_aaaa_3333_ccccu64;
        for _ in 0..200 {
            let w = blocks(20, 18, &mut s);
            let mut q = Vec::new();
            quantize_q8_0(&act(640, &mut s), &mut q);
            let (a, b) = (iq4nl_q80_scalar(&w, &q), unsafe { avx512::iq4nl_q80(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
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
        if skip_avx512("nvfp4_avx_equals_scalar") {
            return;
        }
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

    /// Random Q2_K blocks: scales first, d and dmin at bytes 80 and 82.
    fn q2k_blocks(n: usize, seed: &mut u64) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n * 84).map(|_| rng(seed) as u8).collect();
        for b in v.chunks_exact_mut(84) {
            for o in [80, 82] {
                let d = half::f16::from_f32(((rng(seed) % 1000) as f32 + 1.0) * 1e-4);
                b[o..o + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        v
    }

    /// An edge row: every byte `fill`; the block's f16 scale `d` (Q2_K's `dmin` at 82 too); every scale field
    /// overwritten with `scale`. A fill of 0xff puts every grid index, sign bit and qh bit at its maximum. The
    /// scale fields are those the scalar reader of each format consumes; NVFP4's four UE4M3 bytes are `scale`.
    fn edge_row(name: &str, bb: usize, fill: u8, scale: u8, d: f32, dmin: f32) -> Vec<u8> {
        fn set_f16(v: &mut [u8], o: usize, x: f32) {
            v[o..o + 2].copy_from_slice(&half::f16::from_f32(x).to_le_bytes());
        }
        let mut v = vec![fill; bb];
        match name {
            "iq2s" => {
                set_f16(&mut v, 0, d);
                v[74..82].fill(scale);
            }
            "iq3s" => {
                set_f16(&mut v, 0, d);
                v[106..110].fill(scale);
            }
            "iq2xs" => {
                set_f16(&mut v, 0, d);
                v[66..74].fill(scale);
            }
            // the 4-bit scale is the top nibble of the second u32 of each 8-element pair
            "iq2xxs" => {
                set_f16(&mut v, 0, d);
                for i in 0..8 {
                    v[9 + 8 * i] = (scale << 4) | (v[9 + 8 * i] & 0x0f);
                }
            }
            "iq3xxs" => {
                set_f16(&mut v, 0, d);
                for i in 0..8 {
                    v[69 + 4 * i] = (scale << 4) | (v[69 + 4 * i] & 0x0f);
                }
            }
            // three scale bits at 12 of each qh u16; bit 15 is the +-1/8 offset and 0..11 the grid high bits
            "iq1s" => {
                set_f16(&mut v, 0, d);
                for i in 0..8 {
                    v[35 + 2 * i] = (v[35 + 2 * i] & 0x8f) | ((scale & 7) << 4);
                }
            }
            "q2k" => {
                v[0..16].fill(scale);
                set_f16(&mut v, 80, d);
                set_f16(&mut v, 82, dmin);
            }
            // no f16 scale: the four UE4M3 sub-block scale bytes are the scale field
            "nvfp4" => {
                for b in v.chunks_exact_mut(36) {
                    b[0..4].fill(scale);
                }
            }
            "iq4nl" => set_f16(&mut v, 0, d),
            _ => unreachable!(),
        }
        v
    }

    /// Every DeepSeek expert format: the AVX-512 dot equals the scalar one bit for bit, and the scalar one
    /// equals dequantize-then-dot on the rounded activations.
    #[test]
    fn deepseek_formats_match() {
        if skip_avx512("deepseek_formats_match") {
            return;
        }
        type Sc = fn(&[u8], &[Q8K]) -> f32;
        type Av = unsafe fn(&[u8], &[Q8K]) -> f32;
        type Dq = fn(&[u8], &mut [f32]);
        let cases: [(&str, usize, Sc, Av, Dq); 5] = [
            ("iq2xxs", 66, iq2xxs_q8k_scalar, avx512::iq2xxs_q8k, qx_quant::dequant_iq2_xxs),
            ("iq2xs", 74, iq2xs_q8k_scalar, avx512::iq2xs_q8k, qx_quant::dequant_iq2_xs),
            ("iq3xxs", 98, iq3xxs_q8k_scalar, avx512::iq3xxs_q8k, qx_quant::dequant_iq3_xxs),
            ("iq1s", 50, iq1s_q8k_scalar, avx512::iq1s_q8k, qx_quant::dequant_iq1_s),
            ("q2k", 84, q2k_q8k_scalar, avx512::q2k_q8k, qx_quant::dequant_q2_k),
        ];
        let mut s = 0x7777_8888_9999_aaaau64;
        for (name, bb, scalar, avx, dq) in cases {
            for _ in 0..200 {
                let n = 9; // 2304 elements: DeepSeek's expert width
                let w = if bb == 84 { q2k_blocks(n, &mut s) } else { blocks(n, bb, &mut s) };
                let x = act(256 * n, &mut s);
                let mut qk = Vec::new();
                quantize_q8_k(&x, &mut qk);
                let (a, b) = (scalar(&w, &qk), unsafe { avx(&w, &qk) });
                assert!(a.to_bits() == b.to_bits(), "{name}: avx {b} vs scalar {a}");
                let xk: Vec<f32> = qk.iter().flat_map(|b| b.qs.iter().map(move |&v| v as f32 * b.d)).collect();
                let mut wf = vec![0f32; 256 * n];
                dq(&w, &mut wf);
                // the dequantized weights carry f32 rounding, so the tolerance follows the terms, not the sum
                let r: f64 = wf.iter().zip(&xk).map(|(a, b)| *a as f64 * *b as f64).sum();
                let l1: f64 = wf.iter().zip(&xk).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                assert!((r - a as f64).abs() <= 1e-6 * l1 + 1e-6, "{name}: dequant {r} vs scalar {a} (l1 {l1})");
            }
        }
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

    // ---- AVX2 -----------------------------------------------------------------------------------

    #[test]
    fn iq2s_avx2_matches_scalar() {
        if skip_avx2("iq2s_avx2_matches_scalar") {
            return;
        }
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..200 {
            let w = blocks(10, 82, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq2s_q8k_scalar(&w, &q), unsafe { avx2::iq2s_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
        }
    }

    #[test]
    fn iq3s_avx2_matches_scalar() {
        if skip_avx2("iq3s_avx2_matches_scalar") {
            return;
        }
        let mut s = 0x0fed_cba9_8765_4321u64;
        for _ in 0..200 {
            let w = blocks(10, 110, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (iq3s_q8k_scalar(&w, &q), unsafe { avx2::iq3s_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
        }
    }

    #[test]
    fn iq4nl_avx2_matches_scalar() {
        if skip_avx2("iq4nl_avx2_matches_scalar") {
            return;
        }
        let mut s = 0x5555_aaaa_3333_ccccu64;
        for _ in 0..200 {
            let w = blocks(20, 18, &mut s);
            let mut q = Vec::new();
            quantize_q8_0(&act(640, &mut s), &mut q);
            let (a, b) = (iq4nl_q80_scalar(&w, &q), unsafe { avx2::iq4nl_q80(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "{a} vs {b}");
        }
    }

    #[test]
    fn nvfp4_avx2_equals_scalar() {
        if skip_avx2("nvfp4_avx2_equals_scalar") {
            return;
        }
        let mut s = 0x2468_ace0_1357_9bdfu64;
        for _ in 0..200 {
            let w = nvfp4_blocks(40, &mut s);
            let mut q = Vec::new();
            quantize_q8_k(&act(2560, &mut s), &mut q);
            let (a, b) = (nvfp4_q8k_scalar(&w, &q), unsafe { avx2::nvfp4_q8k(&w, &q) });
            assert!(a.to_bits() == b.to_bits(), "q8k {a} vs {b}");
            let w = nvfp4_blocks(10, &mut s);
            let mut q0 = Vec::new();
            quantize_q8_0(&act(640, &mut s), &mut q0);
            let (a, b) = (nvfp4_q80_scalar(&w, &q0), unsafe { avx2::nvfp4_q80(&w, &q0) });
            assert!(a.to_bits() == b.to_bits(), "q80 {a} vs {b}");
        }
    }

    /// The AVX2 dot of every DeepSeek expert format equals the scalar one bit for bit.
    #[test]
    fn deepseek_formats_avx2_match() {
        if skip_avx2("deepseek_formats_avx2_match") {
            return;
        }
        let cases: [(&str, usize, fn(&[u8], &[Q8K]) -> f32, unsafe fn(&[u8], &[Q8K]) -> f32); 5] = [
            ("iq2xxs", 66, iq2xxs_q8k_scalar, avx2::iq2xxs_q8k),
            ("iq2xs", 74, iq2xs_q8k_scalar, avx2::iq2xs_q8k),
            ("iq3xxs", 98, iq3xxs_q8k_scalar, avx2::iq3xxs_q8k),
            ("iq1s", 50, iq1s_q8k_scalar, avx2::iq1s_q8k),
            ("q2k", 84, q2k_q8k_scalar, avx2::q2k_q8k),
        ];
        let mut s = 0x7777_8888_9999_aaaau64;
        for (name, bb, scalar, avx) in cases {
            for _ in 0..200 {
                let n = 9;
                let w = if bb == 84 { q2k_blocks(n, &mut s) } else { blocks(n, bb, &mut s) };
                let mut qk = Vec::new();
                quantize_q8_k(&act(256 * n, &mut s), &mut qk);
                let (a, b) = (scalar(&w, &qk), unsafe { avx(&w, &qk) });
                assert!(a.to_bits() == b.to_bits(), "{name}: avx2 {b} vs scalar {a}");
            }
        }
    }

    /// Every pool kernel's AVX2 and AVX-512 versions on the same input are bit-identical: random rows as the
    /// other tests make them, plus rows with every grid index, sign bit and scale at its maximum, with the scale
    /// fields zero, and with the block scale zero, against activations at +127 and -127 (the quantizers clamp to
    /// 127, see `act.rs`, never -128), and against a zero activation scale.
    #[test]
    fn avx2_matches_avx512_bits() {
        if skip_avx512("avx2_matches_avx512_bits") || skip_avx2("avx2_matches_avx512_bits") {
            return;
        }
        type Q = unsafe fn(&[u8], &[Q8K]) -> f32;
        let qk: [(&str, usize, Q, Q); 8] = [
            ("iq2s", 82, avx2::iq2s_q8k, avx512::iq2s_q8k_gather),
            ("iq3s", 110, avx2::iq3s_q8k, avx512::iq3s_q8k),
            ("nvfp4", 144, avx2::nvfp4_q8k, avx512::nvfp4_q8k),
            ("iq2xxs", 66, avx2::iq2xxs_q8k, avx512::iq2xxs_q8k),
            ("iq2xs", 74, avx2::iq2xs_q8k, avx512::iq2xs_q8k),
            ("iq3xxs", 98, avx2::iq3xxs_q8k, avx512::iq3xxs_q8k),
            ("iq1s", 50, avx2::iq1s_q8k, avx512::iq1s_q8k),
            ("q2k", 84, avx2::q2k_q8k, avx512::q2k_q8k),
        ];
        let mut s = 0x0bad_f00d_dead_beefu64;
        let mut n_cmp = 0usize;
        for (name, bb, a2, a5) in qk {
            let nb = if matches!(name, "iq2s" | "iq3s" | "nvfp4") { 10 } else { 9 };
            for _ in 0..200 {
                let w = match name {
                    "nvfp4" => nvfp4_blocks(4 * nb, &mut s),
                    "q2k" => q2k_blocks(nb, &mut s),
                    _ => blocks(nb, bb, &mut s),
                };
                let mut q = Vec::new();
                quantize_q8_k(&act(256 * nb, &mut s), &mut q);
                let (x, y) = (unsafe { a2(&w, &q) }, unsafe { a5(&w, &q) });
                assert!(x.to_bits() == y.to_bits(), "{name}: avx2 {x} vs avx512 {y}");
                n_cmp += 1;
            }
            // for NVFP4 the UE4M3 bytes are the scale, so the zero-block-scale row zeroes them
            let zero_block = if name == "nvfp4" { 0 } else { 0xff };
            let edges = [
                ("max indices and scales", edge_row(name, bb, 0xff, 0xff, 1000.0, 1000.0)),
                ("zero scales", edge_row(name, bb, 0xff, 0, 1000.0, 1000.0)),
                ("zero block scale", edge_row(name, bb, 0xff, zero_block, 0.0, 0.0)),
            ];
            for (variant, w) in edges {
                for (sign, ad) in [(127i8, 1.0f32), (-127, 1.0), (127, 0.0)] {
                    let q = vec![Q8K { qs: [sign; 256], d: ad }];
                    let (x, y) = (unsafe { a2(&w, &q) }, unsafe { a5(&w, &q) });
                    assert!(x.to_bits() == y.to_bits(), "{name} {variant} act {sign} d {ad}: avx2 {x} vs avx512 {y}");
                    n_cmp += 1;
                }
            }
            eprintln!("{name}: random and edge rows bit-identical avx2/avx512");
        }
        type O = unsafe fn(&[u8], &[Q80]) -> f32;
        let q80: [(&str, usize, O, O); 2] = [("nvfp4 q80", 36, avx2::nvfp4_q80, avx512::nvfp4_q80), ("iq4nl", 18, avx2::iq4nl_q80, avx512::iq4nl_q80)];
        for (name, bb, a2, a5) in q80 {
            for _ in 0..200 {
                let w = if bb == 36 { nvfp4_blocks(10, &mut s) } else { blocks(20, bb, &mut s) };
                let mut q = Vec::new();
                quantize_q8_0(&act(640, &mut s), &mut q);
                let (x, y) = (unsafe { a2(&w, &q) }, unsafe { a5(&w, &q) });
                assert!(x.to_bits() == y.to_bits(), "{name}: avx2 {x} vs avx512 {y}");
                n_cmp += 1;
            }
            let f = if bb == 36 { "nvfp4" } else { "iq4nl" };
            // IQ4_NL's only scale is the f16 block scale, so its two edge cases are the large one and zero
            let edges: Vec<(&str, Vec<u8>)> = if bb == 36 {
                vec![
                    ("max indices and scales", edge_row(f, bb, 0xff, 0xff, 1000.0, 1000.0)),
                    ("zero scales", edge_row(f, bb, 0xff, 0, 1000.0, 1000.0)),
                    ("zero block scale", edge_row(f, bb, 0xff, 0, 0.0, 0.0)),
                ]
            } else {
                vec![("max indices", edge_row(f, bb, 0xff, 0, 1000.0, 1000.0)), ("zero block scale", edge_row(f, bb, 0xff, 0, 0.0, 0.0))]
            };
            for (variant, w) in edges {
                for (sign, ad) in [(127i8, 1.0f32), (-127, 1.0), (127, 0.0)] {
                    let q = vec![Q80 { qs: [sign; 32], d: ad }; if bb == 36 { 2 } else { 1 }];
                    let (x, y) = (unsafe { a2(&w, &q) }, unsafe { a5(&w, &q) });
                    assert!(x.to_bits() == y.to_bits(), "{name} {variant} act {sign} d {ad}: avx2 {x} vs avx512 {y}");
                    n_cmp += 1;
                }
            }
            eprintln!("{name}: random and edge rows bit-identical avx2/avx512");
        }
        eprintln!("{n_cmp} avx2/avx512 comparisons bit-identical");
    }

    /// The detection picks the strongest set the CPU supports, and the cached table is the one that set builds.
    #[test]
    fn dispatch_picks_the_expected_set() {
        let k = kernels();
        let ptr = |f: Q8kDot| f as usize;
        match isa() {
            Isa::Avx512 => {
                assert!(have_avx512());
                assert_eq!(ptr(k.iq2s_q8k), ptr(avx512::iq2s_q8k_gather));
                assert_eq!(ptr(k.q2k_q8k), ptr(avx512::q2k_q8k));
            }
            Isa::Avx2 => {
                assert!(have_avx2());
                assert_eq!(ptr(k.iq2s_q8k), ptr(avx2::iq2s_q8k));
                assert_eq!(ptr(k.q2k_q8k), ptr(avx2::q2k_q8k));
                assert_eq!(k.iq4nl_q80 as usize, avx2::iq4nl_q80 as Q80Dot as usize);
            }
            Isa::Scalar => {
                assert_eq!(ptr(k.iq1s_q8k), ptr(iq1s_q8k_scalar as Q8kDot));
            }
        }
        eprintln!("kernel set: {:?}", isa());
    }

    /// The AVX2 kernels feed grid bytes to `_mm256_maddubs_epi16` as the unsigned operand and Q8 values (|v| <= 127)
    /// as the signed one. Two products per i16 lane saturate at 32767, so every table byte must be <= 129; the
    /// largest is IQ3_XXS's 62 and IQ4_NL's 127, so no lane can saturate.
    #[test]
    fn grid_magnitudes_cannot_saturate() {
        let mut max = 0u64;
        {
            let mut scan = |t: &[u64], w: usize| {
                for &v in t {
                    for k in 0..w {
                        max = max.max((v >> (8 * k)) & 0xff);
                    }
                }
            };
            scan(&qx_quant::tables::IQ2S_GRID, 8);
            scan(&qx_quant::tables::IQ2XS_GRID, 8);
            scan(&qx_quant::tables::IQ2XXS_GRID, 8);
            scan(&qx_quant::tables::IQ1S_GRID, 8);
        }
        {
            let mut scan32 = |t: &[u32], w: usize| {
                for &v in t {
                    for k in 0..w {
                        max = max.max(((v >> (8 * k)) & 0xff) as u64);
                    }
                }
            };
            scan32(&qx_quant::tables::IQ3S_GRID, 4);
            scan32(&qx_quant::tables::IQ3XXS_GRID, 4);
        }
        assert!(max <= 129, "grid byte {max} * 127 * 2 would saturate maddubs");
        let m = qx_quant::tables::IQ4NL_VALUES.iter().map(|v| v.unsigned_abs()).max().unwrap();
        assert!(m <= 127 && 2 * 127u32 * 127 <= 32767, "IQ4_NL magnitude {m} would saturate maddubs");
        assert!(FP4_MAG.iter().copied().max().unwrap() <= 127);
        eprintln!("largest grid byte {max}, largest IQ4_NL magnitude {m}");
    }
}
