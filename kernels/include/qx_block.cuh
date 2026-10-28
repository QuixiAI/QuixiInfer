// Block-level helpers.
#pragma once
#include "qx_common.cuh"

/// Sum over the whole block; every thread gets the result. `scratch` holds 32 floats.
QX_DEV float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = (blockDim.x + 31) / 32;
    __syncthreads();
    if (l == 0) scratch[w] = v;
    __syncthreads();
    float t = l < nw ? scratch[l] : 0.f;
    t = warp_sum(t);
    return t;
}

QX_DEV float block_max(float v, float* scratch) {
    v = warp_max(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = (blockDim.x + 31) / 32;
    __syncthreads();
    if (l == 0) scratch[w] = v;
    __syncthreads();
    float t = l < nw ? scratch[l] : -3.402823466e38f;
    return warp_max(t);
}

QX_DEV float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
QX_DEV float siluf_(float x) { return x / (1.f + __expf(-x)); }
QX_DEV float softplusf_(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
