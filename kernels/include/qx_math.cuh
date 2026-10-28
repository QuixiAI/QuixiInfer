// Deterministic float math shared bit for bit with the CPU (crates/qx-cpu/src/math.rs): every operation is a
// correctly rounded add, multiply or fused multiply-add, written out so the compiler cannot contract or reorder.
#pragma once
#include "qx_common.cuh"

/// exp(x): Cody-Waite reduction by ln 2, a degree-6 polynomial, scaling by 2^n. Within 2 ulp of expf.
/// The constants are bit patterns so both compilers see identical values.
QX_DEV float qx_expf(float x) {
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    const float n = rintf(__fmul_rn(x, __uint_as_float(0x3fb8aa3bu)));
    float r = __fmaf_rn(n, __uint_as_float(0xbf317200u), x);
    r = __fmaf_rn(n, __uint_as_float(0xb5bfbe8eu), r);
    float p = __uint_as_float(0x39506963u);
    p = __fmaf_rn(p, r, __uint_as_float(0x3ab743cfu));
    p = __fmaf_rn(p, r, __uint_as_float(0x3c088908u));
    p = __fmaf_rn(p, r, __uint_as_float(0x3d2aa9c1u));
    p = __fmaf_rn(p, r, __uint_as_float(0x3e2aaaaau));
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, __fmul_rn(r, r), r);
    p = __fadd_rn(p, 1.0f);
    return __fmul_rn(p, __int_as_float(((int)n + 127) << 23));
}

/// silu(g) * u, as the expert pool computes it: g / (1 + exp(-g)) * u.
QX_DEV float qx_silu_mul(float g, float u) { return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, qx_expf(-g))), u); }
