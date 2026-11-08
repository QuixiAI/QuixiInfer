// Shared definitions for QuixiInfer kernels. Self-contained: no toolkit or system headers, so
// NVRTC and nvcc see exactly the same text. C++17.
#pragma once

typedef signed char i8;
typedef unsigned char u8;
typedef short i16;
typedef unsigned short u16;
typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;

#define QX_DEV __device__ __forceinline__

// ---- bf16 and f16 as raw bits -------------------------------------------------------------

QX_DEV float bf16_to_f32(u16 h) { return __uint_as_float(((u32)h) << 16); }

/// Round to nearest even; NaN stays NaN.
QX_DEV u16 f32_to_bf16(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}

QX_DEV float f16_to_f32(u16 h) {
    float f;
    asm("{ .reg .f16 t; mov.b16 t, %1; cvt.f32.f16 %0, t; }" : "=f"(f) : "h"(h));
    return f;
}

QX_DEV u16 f32_to_f16(float f) {
    u16 h;
    asm("{ .reg .f16 t; cvt.rn.f16.f32 t, %1; mov.b16 %0, t; }" : "=h"(h) : "f"(f));
    return h;
}

// ---- FP8 E4M3 -----------------------------------------------------------------------------

// The conversions are Ada's (sm_89) and later cards' `cvt` instructions; on older cards (the RTX 30 series, sm_86)
// the same values come from the `_sw` forms below; `kernels/bench/e4m3_check.cu` compares them bit for bit with the
// instructions over every float. E4M3 here is the OCP FP8 E4M3FN encoding: bias 7, no infinities,
// 0x7f / 0xff NaN, largest finite 448; the f32 conversion saturates to +-448 and turns any NaN into 0x7f (the
// checker found the instruction drops a NaN's sign both ways; every other pattern matched at the first try).

/// One float to one E4M3 byte in software (round to nearest even, saturating to the largest finite value).
QX_DEV u8 f32_to_e4m3_sw(float f) {
    const u32 u = __float_as_uint(f);
    const u8 s = (u8)((u >> 24) & 0x80);
    const u32 a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return 0x7f;                    // NaN: the instruction drops the sign
    if (a >= 0x43e00000u) return s | 0x7e;               // >= 448 (infinity included): the largest finite
    if (a < 0x3c800000u) {                               // below 2^-6: subnormal, units of 2^-9 (rint: ties to even)
        const int m = __float2int_rn(__uint_as_float(a) * 512.0f);
        return s | (u8)m;                                // m == 8 is the smallest normal, 0x08
    }
    const u32 e = a >> 23, mant = a & 0x7fffffu;
    u32 r = ((e - 127 + 7) << 3) | (mant >> 20);
    const u32 rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (r & 1))) ++r;
    return s | (u8)r;
}

/// One E4M3 byte to a float in software (exact; a NaN byte gives what the instructions give through f16: the
/// positive all-ones NaN, 0x7fffffff, whatever the byte's sign).
QX_DEV float e4m3_to_f32_sw(u8 v) {
    const u32 s = ((u32)v & 0x80) << 24;
    const u32 e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return __uint_as_float(0x7fffffffu);
    const float mag = e == 0 ? (float)m * (1.0f / 512.0f) : ldexpf(1.0f + (float)m * 0.125f, (int)e - 7);
    return __uint_as_float(s | __float_as_uint(mag));
}

QX_DEV u16 f32x2_to_e4m3x2_sw(float lo, float hi) {
    return (u16)f32_to_e4m3_sw(lo) | ((u16)f32_to_e4m3_sw(hi) << 8);
}

QX_DEV void e4m3x2_to_f32x2_sw(u16 v, float& lo, float& hi) {
    lo = e4m3_to_f32_sw((u8)(v & 0xff));
    hi = e4m3_to_f32_sw((u8)(v >> 8));
}

/// Two floats to two E4M3 bytes (lo in bits 0..7), round to nearest, saturating.
QX_DEV u16 f32x2_to_e4m3x2(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    u16 r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
#else
    return f32x2_to_e4m3x2_sw(lo, hi);
#endif
}

/// Two E4M3 bytes (lo in bits 0..7) to two floats.
QX_DEV void e4m3x2_to_f32x2(u16 v, float& lo, float& hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    u32 h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
    lo = f16_to_f32((u16)(h2 & 0xffff));
    hi = f16_to_f32((u16)(h2 >> 16));
#else
    e4m3x2_to_f32x2_sw(v, lo, hi);
#endif
}

// ---- warp reductions ----------------------------------------------------------------------

QX_DEV float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

QX_DEV float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---- launch arguments ---------------------------------------------------------------------

/// Info word `i` as a typed device pointer.
template <typename T> QX_DEV T* arg_ptr(const u64* info, int i) { return reinterpret_cast<T*>(info[i]); }
