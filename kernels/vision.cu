// The vision tower's own kernels (DeepSeek-V4.1-Flash's ViT and aligner on the GPU): everything between the f16
// matrix products, which are `qx_gemm16` (gemm.cu). Activations are f32 where they accumulate (the residual, the
// products' outputs) and f16 where they feed a product, as the model's reference runs the tower in bf16.
// The arithmetic follows crates/qx-vision/src/lib.rs (the CPU tower) step for step.
#include "qx_common.cuh"
#include "qx_block.cuh"

#define F(i) reinterpret_cast<float*>(info[i])
#define CF(i) reinterpret_cast<const float*>(info[i])
#define H(i) reinterpret_cast<u16*>(info[i])
#define U(i) ((u32)info[i])

/// RMS norm into f16: y[n][d] = f16(w[d] * x[n][d] / sqrt(mean_d(x[n]^2) + eps)).
/// info: [0] x (N x D), [1] w (D), [2] y16 (N x D), [3] D, [4] eps bits. Grid N, 256 threads.
extern "C" __global__ void qxv_rms16(const u64* info) {
    __shared__ float sc[32];
    const u32 n = blockIdx.x, D = U(3);
    const float* x = CF(0) + (u64)n * D;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sc);
    const float s = 1.f / sqrtf(ss / (float)D + __uint_as_float(U(4)));
    u16* y = H(2) + (u64)n * D;
    for (u32 i = threadIdx.x; i < D; i += blockDim.x) y[i] = f32_to_f16(CF(1)[i] * (x[i] * s));
}

/// The fused projection's rows into the attention's operands: q and k with the bias added and rotated by the
/// patch's 2D position (pair (j, j + hd/2) of a head by angle j: out_lo = lo cos - hi sin, out_hi = hi cos + lo sin),
/// v with the bias added, transposed. info: [0] qkv (R x 3D), [1] bias (3D), [2] q16 (heads x N x hd),
/// [3] k16 (heads x N x hd), [4] vt16 (heads x hd x Np), [5] D, [6] heads, [7] cos/sin (N x hd/2 pairs),
/// [8] the first row's patch r0, [9] N, [10] Np. Grid R, 256 threads.
extern "C" __global__ void qxv_qkv(const u64* info) {
    const u32 D = U(5), heads = U(6), hd = D / heads, half = hd / 2, N = U(9), Np = U(10);
    const u32 n = U(8) + blockIdx.x;
    const float* row = CF(0) + (u64)blockIdx.x * 3 * D;
    const float* b = CF(1);
    const float* cs = CF(7) + (u64)n * half * 2;
    for (u32 i = threadIdx.x; i < D; i += blockDim.x) {
        const u32 h = i / hd, j = i % hd, jj = j % half, lo = h * hd + jj, hi = lo + half;
        const float c = cs[2 * jj], s = cs[2 * jj + 1];
        const float q0 = row[lo] + b[lo], q1 = row[hi] + b[hi];
        const float k0 = row[D + lo] + b[D + lo], k1 = row[D + hi] + b[D + hi];
        const u64 o = ((u64)h * N + n) * hd + j;
        H(2)[o] = f32_to_f16(j < half ? q0 * c - q1 * s : q1 * c + q0 * s);
        H(3)[o] = f32_to_f16(j < half ? k0 * c - k1 * s : k1 * c + k0 * s);
        H(4)[((u64)h * hd + j) * Np + n] = f32_to_f16(row[2 * D + i] + b[2 * D + i]);
    }
}

/// Row softmax into f16: p[t][n] = softmax over n < N of scale * s[t][n]; columns N..Np are written 0.
/// The probabilities are stored times [5] (a power of two: over thousands of keys most of them are below f16's
/// smallest normal number, 6e-5, where a half keeps few bits); the consumer of the product divides it out.
/// info: [0] s (T x Np), [1] p16 (T x Np), [2] N, [3] Np, [4] scale bits, [5] gain bits. Grid T, 256 threads.
extern "C" __global__ void qxv_softmax16(const u64* info) {
    __shared__ float sc[32];
    const u32 t = blockIdx.x, N = U(2), Np = U(3);
    const float scale = __uint_as_float(U(4));
    const float* s = CF(0) + (u64)t * Np;
    float m = -3.402823466e38f;
    for (u32 i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, s[i]);
    m = block_max(m, sc);
    __syncthreads();
    float sum = 0.f;
    for (u32 i = threadIdx.x; i < N; i += blockDim.x) sum += expf((s[i] - m) * scale);
    sum = block_sum(sum, sc);
    const float inv = __uint_as_float(U(5)) / sum;
    u16* p = H(1) + (u64)t * Np;
    for (u32 i = threadIdx.x; i < Np; i += blockDim.x) p[i] = i < N ? f32_to_f16(expf((s[i] - m) * scale) * inv) : (u16)0;
}

/// x[n][d] += y[n][d] + b[d]. info: [0] x (N x D), [1] y (N x D; 0: none), [2] b (D; 0: none), [3] N, [4] D.
/// Grid ceil(N * D / 256).
extern "C" __global__ void qxv_add(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, D = info[4];
    if (i >= info[3] * D) return;
    float v = F(0)[i];
    if (info[1]) v += CF(1)[i];
    if (info[2]) v += CF(2)[i % D];
    F(0)[i] = v;
}

/// f32 to f16, with a factor, a bias and an activation on the way: out[i] = f16(act(x[i] * m + b[i % K])).
/// info: [0] x (T x K), [1] out16 (T x K), [2] T, [3] K, [4] b (K; 0: none), [5] act (0: none, 1: the erf GELU),
/// [6] m bits.
/// Grid ceil(T * K / 256).
extern "C" __global__ void qxv_f16(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, K = info[3];
    if (i >= info[2] * K) return;
    float v = CF(0)[i] * __uint_as_float(U(6));
    if (info[4]) v += CF(4)[i % K];
    if (info[5] == 1) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    H(1)[i] = f32_to_f16(v);
}

/// The SwiGLU's middle: out[i] = f16(silu(g[i]) * u[i]). info: [0] g, [1] u, [2] out16, [3] elements.
/// Grid ceil(elements / 256).
extern "C" __global__ void qxv_swiglu16(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= info[3]) return;
    const float g = CF(0)[i];
    H(2)[i] = f32_to_f16(g / (1.f + expf(-g)) * CF(1)[i]);
}

/// The aligner's patch merge (torch's F.unfold over the zero-padded grid, feature-major):
/// cells[c][f * r*r + ky * r + kx] = h[(gy*r + ky) * nw + gx*r + kx][f] for the cell c = gy * gw + gx, 0 outside
/// the grid. info: [0] h16 (nh*nw x D), [1] cells16 (C x D*r*r), [2] D, [3] r, [4] nh, [5] nw, [6] gw, [7] C.
/// Grid ceil(C * D * r*r / 256).
extern "C" __global__ void qxv_unfold16(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u32 D = U(2), r = U(3), nh = U(4), nw = U(5), gw = U(6);
    const u64 per = (u64)D * r * r;
    if (i >= info[7] * per) return;
    const u32 c = (u32)(i / per), e = (u32)(i % per), f = e / (r * r), ky = (e % (r * r)) / r, kx = e % r;
    const u32 y = (c / gw) * r + ky, x = (c % gw) * r + kx;
    H(1)[i] = y < nh && x < nw ? reinterpret_cast<const u16*>(info[0])[((u64)y * nw + x) * D + f] : (u16)0;
}
