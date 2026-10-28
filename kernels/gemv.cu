// Matrix-vector products for one token: y[r] (op)= W[row0 + r] . x, f32 accumulation.
//
// One warp per output row, one 256-element chunk per step (see qx_quant.cuh). Rows whose length is not a
// multiple of 256 (32-element formats only) mask the lanes past the end.
//
// info: [0] W row 0 address, [1] x, [2] y, [3] rows, [4] cols, [5] mode (0 store, 1 add, 2 store scaled by
// info[6] as f32 bits), [6] scale bits, [7] row stride in bytes (0: dense rows), [8] address of a u32
// position (0: none), [9] floats y moves per position (y += pos * info[9])
#include "qx_quant.cuh"

template <int T> QX_DEV void gemv_row(const u64* info) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 rows = (u32)info[3], cols = (u32)info[4];
    if (warp >= rows) return;
    const u64 stride = info[7] ? info[7] : row_bytes<T>(cols);
    const u8* row = reinterpret_cast<const u8*>(info[0]) + (u64)warp * stride;
    const float* x = reinterpret_cast<const float*>(info[1]);
    float acc = 0.f;
    const u32 nchunk = (cols + 255) / 256;
    for (u32 c = 0; c < nchunk; ++c) {
        const u32 base = c * 256 + 8 * lane;
        if (base < cols) {
            float w[8];
            decode8<T>(row, c, lane, w);
            const float4 x0 = *reinterpret_cast<const float4*>(x + base);
            const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            acc += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float* y = reinterpret_cast<float*>(info[2]) + (info[8] ? (u64)*reinterpret_cast<const u32*>(info[8]) * info[9] : 0);
        const u32 mode = (u32)info[5];
        if (mode == 1) y[warp] += acc;
        else if (mode == 2) y[warp] = acc * __uint_as_float((u32)info[6]);
        else y[warp] = acc;
    }
}

extern "C" __global__ void qx_gemv_f32(const u64* info) { gemv_row<QX_F32>(info); }
extern "C" __global__ void qx_gemv_bf16(const u64* info) { gemv_row<QX_BF16>(info); }
extern "C" __global__ void qx_gemv_q8_0(const u64* info) { gemv_row<QX_Q8_0>(info); }
extern "C" __global__ void qx_gemv_q6_k(const u64* info) { gemv_row<QX_Q6_K>(info); }
extern "C" __global__ void qx_gemv_iq4_nl(const u64* info) { gemv_row<QX_IQ4_NL>(info); }
extern "C" __global__ void qx_gemv_iq3_s(const u64* info) { gemv_row<QX_IQ3_S>(info); }
extern "C" __global__ void qx_gemv_iq2_s(const u64* info) { gemv_row<QX_IQ2_S>(info); }
extern "C" __global__ void qx_gemv_nvfp4(const u64* info) { gemv_row<QX_NVFP4>(info); }

/// Decodes rows [row0, row0 + rows) to f32: out[r * cols + i]. info: [0] W, [1] out, [2] rows, [3] cols,
/// [4] row0, [5] address of a u32 added to row0 (0: none), e.g. the token id for an embedding lookup,
/// [6] gather (1): output row r is row row0 + u32[info[5] + 4r], an embedding lookup per token of a window.
template <int T> QX_DEV void dequant_rows(const u64* info) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 rows = (u32)info[2], cols = (u32)info[3];
    if (warp >= rows) return;
    const bool gather = info[6] == 1;
    const u64 r0 = info[4] + (info[5] ? reinterpret_cast<const u32*>(info[5])[gather ? warp : 0] : 0);
    const u8* row = reinterpret_cast<const u8*>(info[0]) + (u64)((gather ? 0 : warp) + r0) * row_bytes<T>(cols);
    float* out = reinterpret_cast<float*>(info[1]) + (u64)warp * cols;
    for (u32 c = 0; c * 256 < cols; ++c) {
        const u32 base = c * 256 + 8 * lane;
        if (base < cols) {
            float w[8];
            decode8<T>(row, c, lane, w);
#pragma unroll
            for (int i = 0; i < 8; ++i) out[base + i] = w[i];
        }
    }
}
extern "C" __global__ void qx_dequant_f32(const u64* info) { dequant_rows<QX_F32>(info); }
extern "C" __global__ void qx_dequant_bf16(const u64* info) { dequant_rows<QX_BF16>(info); }
extern "C" __global__ void qx_dequant_q8_0(const u64* info) { dequant_rows<QX_Q8_0>(info); }
extern "C" __global__ void qx_dequant_q6_k(const u64* info) { dequant_rows<QX_Q6_K>(info); }
extern "C" __global__ void qx_dequant_iq4_nl(const u64* info) { dequant_rows<QX_IQ4_NL>(info); }
extern "C" __global__ void qx_dequant_iq3_s(const u64* info) { dequant_rows<QX_IQ3_S>(info); }
extern "C" __global__ void qx_dequant_iq2_s(const u64* info) { dequant_rows<QX_IQ2_S>(info); }
extern "C" __global__ void qx_dequant_nvfp4(const u64* info) { dequant_rows<QX_NVFP4>(info); }
