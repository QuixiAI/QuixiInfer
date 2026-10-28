// Smoke kernels used by the qx-gpu tests: exercise the launch path and the shared header.
#include "qx_common.cuh"

/// x[i] += info[1] for i < info[0].
extern "C" __global__ void qx_add_scalar(float* x, const u64* info) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (u32)info[0]) x[i] += (float)info[1];
}

/// Writes i * 0.5 + info[2] to the raw address in info[0], for i < info[1].
extern "C" __global__ void qx_write_ptr(const u64* info) {
    float* dst = arg_ptr<float>(info, 0);
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (u32)info[1]) dst[i] = (float)i * 0.5f + (float)info[2];
}

/// Round-trips x through E4M3, BF16 and F16: out = [e4m3(x), bf16(x), f16(x)] per element.
extern "C" __global__ void qx_convert_roundtrip(const float* x, float* out, const u64* info) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    u32 n = (u32)info[0];
    if (i >= n) return;
    float v = x[i], a, b;
    e4m3x2_to_f32x2(f32x2_to_e4m3x2(v, 0.f), a, b);
    out[3 * i + 0] = a;
    out[3 * i + 1] = bf16_to_f32(f32_to_bf16(v));
    out[3 * i + 2] = f16_to_f32(f32_to_f16(v));
}

#include "qx_math.cuh"
/// out[i] = qx_expf(x[i]), out[n + i] = qx_silu_mul(x[i], 1.5). info: [0] n.
extern "C" __global__ void qx_math_check(const float* x, float* out, const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, n = (u32)info[0];
    if (i >= n) return;
    out[i] = qx_expf(x[i]);
    out[n + i] = qx_silu_mul(x[i], 1.5f);
}

/// Streams n float4 and writes one sum per block (bandwidth probe). info: [0] src, [1] n, [2] out.
extern "C" __global__ void qx_read_probe(const u64* info) {
    const float4* s = reinterpret_cast<const float4*>(info[0]);
    const u64 n = info[1];
    float acc = 0.f;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
        const float4 v = s[i];
        acc += v.x + v.y + v.z + v.w;
    }
    acc = warp_sum(acc);
    if (threadIdx.x % 32 == 0) atomicAdd(reinterpret_cast<float*>(info[2]), acc);
}

/// L2 prefetch of byte ranges: ranges[2i] = address, ranges[2i+1] = bytes; every thread issues prefetches for
/// 128-byte lines striding over all ranges. The loads complete in the background while later kernels run.
/// info: [0] ranges, [1] range count.
extern "C" __global__ void qx_prefetch_l2(const u64* info) {
    const u64* r = reinterpret_cast<const u64*>(info[0]);
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x, nt = (u64)gridDim.x * blockDim.x;
    for (u32 k = 0; k < (u32)info[1]; ++k) {
        const u64 base = r[2 * k], lines = (r[2 * k + 1] + 127) / 128;
        for (u64 i = tid; i < lines; i += nt) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(base + 128 * i));
    }
}
