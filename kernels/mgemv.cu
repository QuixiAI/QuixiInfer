// Several matrix-vector products sharing one input x, in one launch: y_s = W_s . x for each segment s.
//
// A segment is 8 u64 words in device memory: [0] W, [1] y, [2] rows, [3] cols, [4] type (QxType),
// [5] lanes per row (32, 16 or 8; 0: split-K, one block of 8 warps per row), [6] first block,
// [7] row stride in bytes (0: dense). Blocks are 256 threads. A normal segment gives each block 8 warps x
// (32 / lanes) rows; a split-K segment gives each block one row, its warps taking every 8th 256-element chunk.
// Lanes beyond the row's end are masked, so rows whose length is not a multiple of 256 work in every mode.
//
// info: [0] x, [1] segments, [2] segment count, [3] x stride per token, [4] x argument, [5] tokens. Each entry
// transforms x as it is loaded: qx_mgemv plain; qx_mgemv_silu silu(x * s), s = info[4] as f32 bits; qx_mgemv_silumul silu(x) * x2,
// x2 = info[4] an address.
#include "qx_quant.cuh"
#include "qx_block.cuh"

template <int T> QX_DEV void dec(const u8* row, u32 cols, u32 base, float* w) {
    if constexpr (T == QX_Q6_K_P) decode8_q6k_p(row, cols / 256, base / 256, (base % 256) / 8, w);
    else if constexpr (T == QX_Q8_0_P) decode8_q80_p(row, cols / 32, base / 256, (base % 256) / 8, w);
    else decode8<T>(row, base / 256, (base % 256) / 8, w);
}

template <int XM> struct XIn {
    const float* x;
    u64 arg;
    /// The token's input as int8 blocks of 32 with an f32 scale each (`qx_mgemv_q8x`), or null: Q8_0 rows then
    /// take integer dot products instead of decoding to f32 against f32 inputs.
    const i8* xq;
    const float* xd;
};

template <int XM> QX_DEV float dot8(const float* w, const XIn<XM>& xi, u32 base) {
    float4 x0 = *reinterpret_cast<const float4*>(xi.x + base);
    float4 x1 = *reinterpret_cast<const float4*>(xi.x + base + 4);
    if constexpr (XM == 1) {
        const float s = __uint_as_float((u32)xi.arg);
        x0 = make_float4(siluf_(x0.x * s), siluf_(x0.y * s), siluf_(x0.z * s), siluf_(x0.w * s));
        x1 = make_float4(siluf_(x1.x * s), siluf_(x1.y * s), siluf_(x1.z * s), siluf_(x1.w * s));
    } else if constexpr (XM == 2) {
        const float* x2 = reinterpret_cast<const float*>(xi.arg) + base;
        const float4 u0 = *reinterpret_cast<const float4*>(x2), u1 = *reinterpret_cast<const float4*>(x2 + 4);
        x0 = make_float4(siluf_(x0.x) * u0.x, siluf_(x0.y) * u0.y, siluf_(x0.z) * u0.z, siluf_(x0.w) * u0.w);
        x1 = make_float4(siluf_(x1.x) * u1.x, siluf_(x1.y) * u1.y, siluf_(x1.z) * u1.z, siluf_(x1.w) * u1.w);
    }
    return w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
}

/// The lane's share of one row for NT tokens at once: steps st = step0, step0 + dstep, ... of 8*lanes elements
/// each. Full steps run four at a time with all their loads issued before any use, so each warp keeps several
/// chunks in flight. Every token's arithmetic is exactly the single-token sequence, whatever NT: a window of
/// tokens gets the same bits as one token at a time.
template <int T, int XM, int NT> QX_DEV void row_partial(const u8* row, const XIn<XM>* x, u32 cols, u32 lane, u32 lanes, u32 step0, u32 dstep, float* out) {
    const u32 per = 8 * lanes;
    const u32 full = cols / per;  // steps with every lane in range
    float a0[NT], a1[NT], a2[NT], a3[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) a0[n] = a1[n] = a2[n] = a3[n] = 0.f;
    u32 st = step0;
    for (; st + 3 * dstep < full; st += 4 * dstep) {
        float w0[8], w1[8], w2[8], w3[8];
        const u32 b0 = st * per + 8 * lane, b1 = b0 + dstep * per, b2 = b1 + dstep * per, b3 = b2 + dstep * per;
        dec<T>(row, cols, b0, w0);
        dec<T>(row, cols, b1, w1);
        dec<T>(row, cols, b2, w2);
        dec<T>(row, cols, b3, w3);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            a0[n] += dot8(w0, x[n], b0);
            a1[n] += dot8(w1, x[n], b1);
            a2[n] += dot8(w2, x[n], b2);
            a3[n] += dot8(w3, x[n], b3);
        }
    }
    for (; st * per < cols; st += dstep) {
        const u32 base = st * per + 8 * lane;
        if (base < cols) {
            float w[8];
            dec<T>(row, cols, base, w);
#pragma unroll
            for (int n = 0; n < NT; ++n) a0[n] += dot8(w, x[n], base);
        }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) out[n] = (a0[n] + a1[n]) + (a2[n] + a3[n]);
}

/// `row_partial` for planar Q8_0 rows against int8 inputs: per 8 elements two dp4a and one scale product, the
/// steps in the same order as `row_partial`, so every token's bits are the same whatever NT. The inputs take a
/// quarter of the f32 path's load traffic, which held a 5-token window to a third of the memory bandwidth.
template <int NT> QX_DEV void row_partial_q8(const u8* row, const XIn<0>* x, u32 cols, u32 lane, u32 lanes, u32 step0, u32 dstep, float* out) {
    const u32 per = 8 * lanes, nb = cols / 32;
    const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
    float a0[NT], a1[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) a0[n] = a1[n] = 0.f;
    // steps alternate between two accumulators (two loads in flight); the order depends on the step only
    u32 st = step0;
    for (; (st + dstep) * per + 8 * lane < cols; st += 2 * dstep) {
        const u32 b0 = st * per + 8 * lane, b1 = b0 + dstep * per;
        const int2 w0 = *reinterpret_cast<const int2*>(row + b0), w1 = *reinterpret_cast<const int2*>(row + b1);
        const float d0 = f16_to_f32(dw[b0 / 32]), d1 = f16_to_f32(dw[b1 / 32]);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int2 q0 = *reinterpret_cast<const int2*>(x[n].xq + b0), q1 = *reinterpret_cast<const int2*>(x[n].xq + b1);
            a0[n] += d0 * x[n].xd[b0 / 32] * (float)__dp4a(w0.y, q0.y, __dp4a(w0.x, q0.x, 0));
            a1[n] += d1 * x[n].xd[b1 / 32] * (float)__dp4a(w1.y, q1.y, __dp4a(w1.x, q1.x, 0));
        }
    }
    const u32 b = st * per + 8 * lane;
    if (b < cols) {
        const int2 w = *reinterpret_cast<const int2*>(row + b);
        const float d = f16_to_f32(dw[b / 32]);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int2 q = *reinterpret_cast<const int2*>(x[n].xq + b);
            a0[n] += d * x[n].xd[b / 32] * (float)__dp4a(w.y, q.y, __dp4a(w.x, q.x, 0));
        }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) out[n] = a0[n] + a1[n];
}

template <int XM, int NT> QX_DEV void partial(u32 ty, const u8* row, const XIn<XM>* x, u32 cols, u32 lane, u32 lanes, u32 step0, u32 dstep, float* out) {
    if (ty == QX_Q8_0_P && x[0].xq) {
        // the int8 inputs already carry the transform (silu, silu * x2)
        XIn<0> xi[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) xi[n] = XIn<0>{x[n].x, 0, x[n].xq, x[n].xd};
        row_partial_q8<NT>(row, xi, cols, lane, lanes, step0, dstep, out);
        return;
    }
    switch (ty) {
#define C(t) case t: row_partial<t, XM, NT>(row, x, cols, lane, lanes, step0, dstep, out); return;
        C(QX_F32) C(QX_BF16) C(QX_Q8_0) C(QX_Q6_K) C(QX_IQ4_NL) C(QX_IQ3_S) C(QX_IQ2_S) C(QX_Q6_K_P) C(QX_Q8_0_P) C(QX_NVFP4)
#undef C
        default:
#pragma unroll
            for (int n = 0; n < NT; ++n) out[n] = 0.f;
    }
}

/// info: [0] x, [1] segments, [2] segment count, [3] x row stride per token (floats), [4] x argument,
/// [5] tokens (1 to 8): token n reads x + n * stride (and x2 + n * stride) and writes y + (n0 + n) * rows, [6] n0
/// (a later chunk of a window's tokens), [7] the int8 inputs (`qx_mgemv_q8x`: token n at + n * cols) or 0,
/// [8] their scales (token n at + n * cols / 32), [9] cols of the int8 rows.
template <int XM, int NT> QX_DEV void mgemv(const u64* info) {
    XIn<XM> x[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        x[n].x = reinterpret_cast<const float*>(info[0]) + (u64)n * info[3];
        x[n].arg = XM == 2 ? info[4] + 4ull * n * info[3] : info[4];
        x[n].xq = info[7] ? reinterpret_cast<const i8*>(info[7]) + (u64)n * info[9] : nullptr;
        x[n].xd = info[7] ? reinterpret_cast<const float*>(info[8]) + (u64)n * (info[9] / 32) : nullptr;
    }
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 8 + 6]) ++s;
    const u64* d = segs + s * 8;
    const u32 rows = (u32)d[2], cols = (u32)d[3], ty = (u32)d[4], lanes = (u32)d[5];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, cols);
    const u32 blk = blockIdx.x - (u32)d[6];
    const u32 warp = threadIdx.x / 32, wl = threadIdx.x % 32;
    float* y = reinterpret_cast<float*>(d[1]);
    float a[NT];
    if (lanes == 0) {
        // split-K: one row per block
        __shared__ float part[8][NT];
        const u32 r = blk;
        partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, wl, 32, warp, 8, a);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float v = warp_sum(a[n]);
            if (wl == 0) part[warp][n] = v;
        }
        __syncthreads();
        if (threadIdx.x < NT) {
            const u32 n = threadIdx.x;
            float t = 0.f;
            for (int i = 0; i < 8; ++i) t += part[i][n];
            y[(info[6] + n) * rows + r] = t;
        }
        return;
    }
    const u32 rpw = 32 / lanes;  // rows per warp
    const u32 r = (blk * 8 + warp) * rpw + wl / lanes;
    const u32 lane = wl % lanes;
    if (r < rows) partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, lane, lanes, 0, 1, a);
    else
#pragma unroll
        for (int n = 0; n < NT; ++n) a[n] = 0.f;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        // reduce within each group of `lanes` lanes
        float v = a[n];
        for (u32 o = lanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (r < rows && lane == 0) y[(info[6] + n) * rows + r] = v;
    }
}

// One entry per token count: a shared entry would give the one-token case the registers of the eight-token one.
extern "C" __global__ void __launch_bounds__(256) qx_mgemv(const u64* info) { mgemv<0, 1>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu(const u64* info) { mgemv<1, 1>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul(const u64* info) { mgemv<2, 1>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t2(const u64* info) { mgemv<0, 2>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t2(const u64* info) { mgemv<1, 2>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t2(const u64* info) { mgemv<2, 2>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t3(const u64* info) { mgemv<0, 3>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t3(const u64* info) { mgemv<1, 3>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t3(const u64* info) { mgemv<2, 3>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t4(const u64* info) { mgemv<0, 4>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t4(const u64* info) { mgemv<1, 4>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t4(const u64* info) { mgemv<2, 4>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t5(const u64* info) { mgemv<0, 5>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t5(const u64* info) { mgemv<1, 5>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t5(const u64* info) { mgemv<2, 5>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t6(const u64* info) { mgemv<0, 6>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t6(const u64* info) { mgemv<1, 6>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t6(const u64* info) { mgemv<2, 6>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t7(const u64* info) { mgemv<0, 7>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t7(const u64* info) { mgemv<1, 7>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t7(const u64* info) { mgemv<2, 7>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_t8(const u64* info) { mgemv<0, 8>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silu_t8(const u64* info) { mgemv<1, 8>(info); }
extern "C" __global__ void __launch_bounds__(256) qx_mgemv_silumul_t8(const u64* info) { mgemv<2, 8>(info); }

/// The int8 inputs of a launch: token n's x (transformed as the launch's mode says) in blocks of 32, d = amax / 127,
/// q = round(x / d). info: [0] x, [1] x row stride (floats), [2] mode (0 plain, 1 silu(x * s), 2 silu(x) * x2),
/// [3] arg (s's bits, or x2), [4] cols, [5] q out (token n at + n * cols), [6] d out (token n at + n * cols / 32;
/// with [7] set, transposed for `qx_tmv`: block b's 32 tokens at + 32 b). Grid (ceil(cols / 256), tokens), 256
/// threads: one warp per block of 32.
extern "C" __global__ void qx_mgemv_q8x(const u64* info) {
    const u32 n = blockIdx.y, cols = (u32)info[4], i = blockIdx.x * 256 + threadIdx.x;
    const float* x = reinterpret_cast<const float*>(info[0]) + (u64)n * info[1];
    float v = 0.f;
    if (i < cols) {
        v = x[i];
        if (info[2] == 1) v = siluf_(v * __uint_as_float((u32)info[3]));
        else if (info[2] == 2) v = siluf_(v) * (reinterpret_cast<const float*>(info[3]) + (u64)n * info[1])[i];
    }
    const float amax = warp_max(fabsf(v));
    const float d = __fdiv_rn(amax, 127.f);
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    if (i < cols) {
        reinterpret_cast<i8*>(info[5])[(u64)n * cols + i] = (i8)roundf(__fmul_rn(v, id));
        if (i % 32 == 0) reinterpret_cast<float*>(info[6])[info[7] ? (u64)(i / 32) * 32 + n : (u64)n * (cols / 32) + i / 32] = d;
    }
}
