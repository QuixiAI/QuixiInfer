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

/// Two floats to two E4M3 bytes (lo in bits 0..7), round to nearest, saturating.
QX_DEV u16 f32x2_to_e4m3x2(float lo, float hi) {
    u16 r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

/// Two E4M3 bytes (lo in bits 0..7) to two floats.
QX_DEV void e4m3x2_to_f32x2(u16 v, float& lo, float& hi) {
    u32 h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
    lo = f16_to_f32((u16)(h2 & 0xffff));
    hi = f16_to_f32((u16)(h2 >> 16));
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
