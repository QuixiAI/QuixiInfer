//! Activation quantization for the CPU expert pool, matching ggml's reference rounding.

/// 256 int8 values with one f32 scale, and the sums of each 16 (unused by the IQ dots, kept for layout parity).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Q8K {
    pub qs: [i8; 256],
    pub d: f32,
}

/// 32 int8 values with one scale (f16-rounded, as ggml stores it).
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Q80 {
    pub qs: [i8; 32],
    pub d: f32,
}

pub fn quantize_q8_k(x: &[f32], out: &mut Vec<Q8K>) {
    out.clear();
    for xs in x.chunks_exact(256) {
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xs {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let mut b = Q8K { qs: [0; 256], d: 0.0 };
        if amax != 0.0 {
            let iscale = -127.0 / max;
            for (q, v) in b.qs.iter_mut().zip(xs) {
                *q = ((iscale * v).round_ties_even() as i32).min(127) as i8;
            }
            b.d = 1.0 / iscale;
        }
        out.push(b);
    }
}

pub fn quantize_q8_0(x: &[f32], out: &mut Vec<Q80>) {
    out.clear();
    for xs in x.chunks_exact(32) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut b = Q80 { qs: [0; 32], d: half::f16::from_f32(d).to_f32() };
        for (q, v) in b.qs.iter_mut().zip(xs) {
            *q = (v * id).round() as i8;
        }
        out.push(b);
    }
}
