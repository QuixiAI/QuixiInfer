// The software E4M3 conversions (qx_common.cuh `_sw`, used on cards before sm_89) against the hardware `cvt`
// instructions, over every one of the 2^32 float bit patterns (f32 -> E4M3) and every byte (E4M3 -> f32). Needs an
// sm_89+ card for the instructions. Prints the mismatch counts and the first few mismatches; exit 0 when none.
//   /usr/local/cuda-13.3/bin/nvcc -arch=sm_120a -std=c++17 -O3 -I kernels/include -o ~/qx-bins/e4m3_check kernels/bench/e4m3_check.cu && ~/qx-bins/e4m3_check
#include <cstdio>
#include <cstdlib>
#include "qx_common.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

__device__ u16 hw_f32x2_to_e4m3x2(float lo, float hi) {
    u16 r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ void hw_e4m3x2_to_f32x2(u16 v, float& lo, float& hi) {
    u32 h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
    lo = f16_to_f32((u16)(h2 & 0xffff));
    hi = f16_to_f32((u16)(h2 >> 16));
}

// every float: hardware and software f32 -> E4M3 of the pattern (in the low lane; the high lane carries its bitwise
// complement so both lanes are exercised)
__global__ void check_to(unsigned long long* bad, u32* first, u32 base) {
    const u32 u = base + (blockIdx.x * blockDim.x + threadIdx.x);
    const float lo = __uint_as_float(u), hi = __uint_as_float(~u);
    const u16 h = hw_f32x2_to_e4m3x2(lo, hi), s = f32x2_to_e4m3x2_sw(lo, hi);
    if (h != s) {
        const unsigned long long n = atomicAdd(bad, 1ull);
        if (n < 8) { first[3 * n] = u; first[3 * n + 1] = h; first[3 * n + 2] = s; }
    }
}

// every byte pair: hardware and software E4M3 -> f32 (NaN equal to NaN whatever the payload)
__global__ void check_from(unsigned long long* bad, u32* first) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. 65535
    float hl, hh, sl, sh;
    hw_e4m3x2_to_f32x2((u16)v, hl, hh);
    e4m3x2_to_f32x2_sw((u16)v, sl, sh);
    const bool okl = (__float_as_uint(hl) == __float_as_uint(sl)) || (isnan(hl) && isnan(sl) && signbit(hl) == signbit(sl));
    const bool okh = (__float_as_uint(hh) == __float_as_uint(sh)) || (isnan(hh) && isnan(sh) && signbit(hh) == signbit(sh));
    if (!okl || !okh) {
        const unsigned long long n = atomicAdd(bad, 1ull);
        if (n < 8) { first[3 * n] = v; first[3 * n + 1] = __float_as_uint(okl ? hh : hl); first[3 * n + 2] = __float_as_uint(okl ? sh : sl); }
    }
}

int main() {
    unsigned long long* bad; u32* first;
    CK(cudaMallocManaged(&bad, 16)); CK(cudaMallocManaged(&first, 2 * 8 * 3 * 4));
    bad[0] = bad[1] = 0;
    for (unsigned long long base = 0; base < (1ull << 32); base += 1ull << 28) {
        check_to<<<(1u << 28) / 256, 256>>>(bad, first, (u32)base);
        CK(cudaGetLastError());
    }
    check_from<<<65536 / 256, 256>>>(bad + 1, first + 24);
    CK(cudaDeviceSynchronize());
    std::printf("f32 -> E4M3: %llu mismatches of 4294967296\n", bad[0]);
    for (unsigned i = 0; i < 8 && i < bad[0]; ++i) std::printf("  float %08x: hw %04x sw %04x\n", first[3 * i], first[3 * i + 1], first[3 * i + 2]);
    std::printf("E4M3 -> f32: %llu mismatches of 65536 pairs\n", bad[1]);
    for (unsigned i = 0; i < 8 && i < bad[1]; ++i) std::printf("  pair %04x: hw %08x sw %08x\n", first[24 + 3 * i], first[24 + 3 * i + 1], first[24 + 3 * i + 2]);
    return bad[0] || bad[1] ? 1 : 0;
}
