//! Deterministic float math shared bit for bit with the GPU (kernels/include/qx_math.cuh): the same correctly
//! rounded adds, multiplies and fused multiply-adds, in the same order.

/// exp(x): Cody-Waite reduction by ln 2, a degree-6 polynomial, scaling by 2^n. Within 2 ulp of expf.
/// The constants are bit patterns so both compilers see identical values.
#[inline]
pub fn qx_expf(x: f32) -> f32 {
    let b = f32::from_bits;
    let x = x.max(-87.0).min(88.0);
    let n = (x * b(0x3fb8aa3b)).round_ties_even();
    let r = n.mul_add(b(0xbf317200), x);
    let r = n.mul_add(b(0xb5bfbe8e), r);
    let mut p = b(0x39506963);
    p = p.mul_add(r, b(0x3ab743cf));
    p = p.mul_add(r, b(0x3c088908));
    p = p.mul_add(r, b(0x3d2aa9c1));
    p = p.mul_add(r, b(0x3e2aaaaa));
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r * r, r);
    p += 1.0;
    p * f32::from_bits((((n as i32) + 127) << 23) as u32)
}

/// silu(g) * u, as the GPU expert kernels compute it: g / (1 + exp(-g)) * u.
#[inline]
pub fn qx_silu_mul(g: f32, u: f32) -> f32 {
    g / (1.0 + qx_expf(-g)) * u
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expf_close_to_libm() {
        let mut worst = 0f32;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (a, b) = (qx_expf(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 3e-7, "relative error {worst}");
    }
}
