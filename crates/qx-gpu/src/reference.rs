//! Plain CPU references for the GPU kernels. Slow and obvious on purpose.

/// f32 to OCP FP8 E4M3 (bias 7, max 448, no infinities), round to nearest even, saturating; NaN to 0x7f.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    // Candidates: every finite E4M3 magnitude; pick nearest, ties to even mantissa.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..0x7f {
        let v = e4m3_to_f32(code);
        let err = (v - a).abs();
        if err < best_err || (err == best_err && code & 1 == 0) {
            best = code;
            best_err = err;
        }
    }
    sign | best
}

pub fn e4m3_to_f32(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((c >> 3) & 0xf) as i32;
    let m = (c & 7) as f32;
    if e == 0xf && (c & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 { sign * m / 8.0 * 2f32.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

pub fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

pub fn round_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}
