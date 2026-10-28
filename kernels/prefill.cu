// Prefill: the per-token operations of the forward pass for a chunk of T consecutive tokens of one sequence,
// positions p0 .. p0 + T - 1. Activations are [T][...] f32; token t's row is at t * stride. The chunk's own
// position comes from the step state's pos word (p0) plus t.
#include "qx_quant.cuh"
#include "qx_block.cuh"
#include "qx_math.cuh"
#include "qx_kv.cuh"

QX_DEV u64 ld_acquire_sys_ple(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

#define F(i) reinterpret_cast<float*>(info[i])
#define CF(i) reinterpret_cast<const float*>(info[i])
#define U(i) ((u32)info[i])
#define P0 (*reinterpret_cast<const u32*>(info[1]))

/// RMS norm per (token, group): out[t][g] = x[t][g] / rms * w[g] (w stride per group, 0 shared).
/// info: [0] x, [1] unused, [2] w, [3] out, [4] dim, [5] w stride, [6] eps bits, [7] x row stride, [8] out row stride.
/// Grid (groups, T), 1024 threads.
extern "C" __global__ void qxp_rms(const u64* info) {
    __shared__ float sc[32];
    const u32 g = blockIdx.x, t = blockIdx.y, d = U(4);
    const float* x = CF(0) + (u64)t * info[7] + (u64)g * d;
    const float* w = CF(2) + (u64)g * U(5);
    float* o = F(3) + (u64)t * info[8] + (u64)g * d;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)d + __uint_as_float(U(6)));
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) o[i] = x[i] * r * w[i];
}

/// x[t][i] = silu(x[t][i] * s) over a T x n block. info: [0] x, [1] count (T * n), [2] s bits.
extern "C" __global__ void qxp_scale_silu(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < info[1]) F(0)[i] = siluf_(F(0)[i] * __uint_as_float(U(2)));
}

/// HC mix output per token. info: [0] xn (T x HC*E), [1] gate (T x HC*E), [2] x out (T x E), [3] E, [4] HC, [5] T.
extern "C" __global__ void qxp_hc_mix_out(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u32 n = U(3), hc = U(4);
    if (i >= (u64)U(5) * n) return;
    const u64 t = i / n, e = i % n;
    float s = 0.f;
    for (u32 k = 0; k < hc; ++k) s += CF(0)[t * hc * n + k * n + e] * sigmoidf_(CF(1)[t * hc * n + k * n + e]);
    F(2)[i] = s / (float)hc;
}

/// HC combine with the next mix's norm, per token. info: [0] R (T x HC*E), [1] y (T x E), [2] inject (T x HC),
/// [3] next norm w, [4] xn out, [5] E, [6] eps bits, [7] HC. Grid (HC, T), 1024 threads.
extern "C" __global__ void qxp_hc_combine_norm(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, t = blockIdx.y, n = U(5), hc = U(7);
    float* R = F(0) + ((u64)t * hc + s) * n;
    const float* y = CF(1) + (u64)t * n;
    const float w = 2.f * sigmoidf_(CF(2)[(u64)t * hc + s] / (float)hc);
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = R[i] + y[i] * w;
        R[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)n + __uint_as_float(U(6)));
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) F(4)[((u64)t * hc + s) * n + i] = R[i] * r * CF(3)[(u64)s * n + i];
}

/// R[t][s] = e[t] for every stream. info: [0] R, [1] e (T x E), [2] E, [3] HC, [4] T.
extern "C" __global__ void qxp_hc_init(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, n = U(2), hc = U(3);
    if (i >= (u64)U(4) * hc * n) return;
    const u64 t = i / (hc * n);
    F(0)[i] = CF(1)[t * n + i % n];
}

/// GDN causal conv over the chunk, fused with the q/k L2 norms: blocks (head chunk of 128 channels, t).
/// The conv state (3 x ch, oldest first) is read before the chunk and written once with its last 3 inputs, by
/// a separate pass (qxp_gdn_conv_state) after every block has read it.
/// info: [0] qkv (T x ch), [1] unused, [2] w (ch x 4), [3] out (T x ch), [4] ch, [5] dim, [6] key heads,
/// [7] eps bits, [8] state, [9] T.
extern "C" __global__ void qxp_gdn_conv(const u64* info) {
    __shared__ float sc[32];
    const u32 ch = U(4), d = U(5), t = blockIdx.y, i = blockIdx.x * d + threadIdx.x;
    const float* qkv = CF(0);
    const float* st = CF(8);
    auto in = [&](int tt) -> float { return tt >= 0 ? qkv[(u64)tt * ch + i] : st[(u64)(3 + tt) * ch + i]; };
    const float* w = CF(2) + 4ull * i;
    float c = siluf_(w[3] * in((int)t) + w[0] * in((int)t - 3) + w[1] * in((int)t - 2) + w[2] * in((int)t - 1));
    if (blockIdx.x < 2 * U(6)) {
        const float ssum = block_sum(c * c, sc);
        c *= rsqrtf(ssum + __uint_as_float(U(7)));
    }
    F(3)[(u64)t * ch + i] = c;
}

/// The conv state after the chunk: its last 3 inputs (or older state when T < 3). info: [0] qkv, [1] state,
/// [2] ch, [3] T.
extern "C" __global__ void qxp_gdn_conv_state(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, ch = U(2);
    const int T = (int)U(3);
    if (i >= ch) return;
    float* st = F(1);
    float s[3] = {st[i], st[ch + i], st[2 * ch + i]};
    float o[3];
    for (int j = 0; j < 3; ++j) {
        const int tt = T - 3 + j;
        o[j] = tt >= 0 ? CF(0)[(u64)tt * ch + i] : s[3 + tt];
    }
    for (int j = 0; j < 3; ++j) st[(u64)j * ch + i] = o[j];
}

/// The gated delta rule over the whole chunk: block (head h, column chunk c) keeps its 128 x 32 part of S in
/// registers for all T tokens. info: [0] conv out (T x ch), [1] S, [2] o out (T x vh x d), [3] beta raw (T x vh),
/// [4] alpha raw (T x vh), [5] dt bias, [6] a, [7] key heads, [8] dim, [9] T, [10] ch, [11] value heads.
/// Grid (value heads, 4), 128 threads.
extern "C" __global__ void __launch_bounds__(128) qxp_gdn_recur(const u64* info) {
    __shared__ float qs[128], ks[128];
    __shared__ float part[4][32];
    __shared__ float dl[32];
    const u32 h = blockIdx.x, jc = blockIdx.y, d = U(8), kh = U(7), T = U(9), ch = U(10), vh = U(11);
    const u32 jj = threadIdx.x % 32, ic = threadIdx.x / 32, j = jc * 32 + jj;
    const u32 khh = h % kh;
    float* S = F(1) + (u64)h * d * d;
    float col[32];
#pragma unroll
    for (u32 t2 = 0; t2 < 32; ++t2) col[t2] = S[(ic * 32 + t2) * d + j];
    const float dtb = CF(5)[h], av = CF(6)[h];
    for (u32 t = 0; t < T; ++t) {
        const float* cv = CF(0) + (u64)t * ch;
        __syncthreads();
        qs[threadIdx.x] = cv[khh * d + threadIdx.x];
        ks[threadIdx.x] = cv[(u64)kh * d + khh * d + threadIdx.x];
        __syncthreads();
        const float beta = sigmoidf_(CF(3)[(u64)t * vh + h]);
        const float decay = __expf(softplusf_(CF(4)[(u64)t * vh + h] + dtb) * av);
        float sk = 0.f;
#pragma unroll
        for (u32 t2 = 0; t2 < 32; ++t2) {
            col[t2] *= decay;
            sk += col[t2] * ks[ic * 32 + t2];
        }
        part[ic][jj] = sk;
        __syncthreads();
        if (ic == 0) dl[jj] = (cv[2ull * kh * d + (u64)h * d + j] - (part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj])) * beta;
        __syncthreads();
        const float delta = dl[jj];
        float o = 0.f;
#pragma unroll
        for (u32 t2 = 0; t2 < 32; ++t2) {
            col[t2] += ks[ic * 32 + t2] * delta;
            o += col[t2] * qs[ic * 32 + t2];
        }
        __syncthreads();
        part[ic][jj] = o;
        __syncthreads();
        if (ic == 0) F(2)[((u64)t * vh + h) * d + j] = (part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj]) * rsqrtf((float)d);
    }
#pragma unroll
    for (u32 t2 = 0; t2 < 32; ++t2) S[(ic * 32 + t2) * d + j] = col[t2];
}

/// qxp_gdn_recur in chunks of 32 tokens (the chunked form of the gated delta rule): with g_t the chunk's running
/// sum of log decays and S0 the state before the chunk,
///   delta = (I + A)^-1 (beta * (v - e^g K S0)),  A[t][s] = beta_t e^(g_t - g_s) k_t.k_s  (s < t)
///   o_t = (e^g_t q_t S0 + sum_{s<=t} e^(g_t - g_s) (q_t.k_s) delta_s) / sqrt(d)
///   S = e^g_n S0 + sum_t e^(g_n - g_t) k_t delta_t^T
/// which is the token-by-token recurrence regrouped (not its bits). Same info and grid as qxp_gdn_recur; 256
/// threads: thread (j, ig) of a warp holds S[16 ig .. 16 ig + 15][j] for one of the block's 32 value columns.
#define GC 32
#define GKP 136  // padded key row: dim 128 plus one float per 16 (the 8 i-groups of a warp read distinct banks)
/// a . b over 128 dims of two padded key rows, four independent sums
QX_DEV float dot128(const float* a, const float* b) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (u32 g = 0; g < 8; ++g)
#pragma unroll
        for (u32 i = 0; i < 16; i += 4) {
            const u32 o = g * 17 + i;
            s0 = fmaf(a[o], b[o], s0);
            s1 = fmaf(a[o + 1], b[o + 1], s1);
            s2 = fmaf(a[o + 2], b[o + 2], s2);
            s3 = fmaf(a[o + 3], b[o + 3], s3);
        }
    return (s0 + s1) + (s2 + s3);
}
extern "C" __global__ void __launch_bounds__(256) qxp_gdn_chunk(const u64* info) {
    __shared__ float ks[GC][GKP], qs[GC][GKP];
    __shared__ float vs[GC][33], ds[GC][33];
    __shared__ float am[GC][GC + 1];  // A for the solve, then P for the outputs
    __shared__ float gl[GC], bt[GC];
    const u32 h = blockIdx.x, jc = blockIdx.y, d = U(8), kh = U(7), T = U(9), ch = U(10), vh = U(11);
    const u32 tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const u32 ig = lane % 8, jl = warp * 4 + lane / 8, j = jc * 32 + jl;
    const u32 khh = h % kh;
    auto pad = [](u32 i) { return i + i / 16; };
    float* S = F(1) + (u64)h * d * d;
    float sreg[16];
#pragma unroll
    for (u32 ii = 0; ii < 16; ++ii) sreg[ii] = S[(ig * 16 + ii) * d + j];
    const float dtb = CF(5)[h], av = CF(6)[h], rs = rsqrtf((float)d);
    // the next chunk's inputs, loaded into registers while the current one computes
    float pk[16], pq[16], pv[4], pa = 0.f, pb = 0.f;
    auto fetch = [&](u32 c0) {
        const u32 n = c0 < T ? min((u32)GC, T - c0) : 0;
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256, t = e / 128, i = e % 128;
            const float* cv = CF(0) + (u64)(c0 + t) * ch;
            pk[r] = t < n ? cv[(u64)kh * d + khh * d + i] : 0.f;
            pq[r] = t < n ? cv[khh * d + i] : 0.f;
        }
#pragma unroll
        for (u32 r = 0; r < 4; ++r) {
            const u32 e = tid + r * 256, t = e / 32, jj = e % 32;
            pv[r] = t < n ? CF(0)[(u64)(c0 + t) * ch + 2ull * kh * d + (u64)h * d + jc * 32 + jj] : 0.f;
        }
        if (warp == 0) {
            pa = lane < n ? CF(4)[(u64)(c0 + lane) * vh + h] : 0.f;
            pb = lane < n ? CF(3)[(u64)(c0 + lane) * vh + h] : 0.f;
        }
    };
    fetch(0);
    for (u32 c0 = 0; c0 < T; c0 += GC) {
        const u32 n = min((u32)GC, T - c0);
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256, t = e / 128, i = e % 128;
            ks[t][pad(i)] = pk[r];
            qs[t][pad(i)] = pq[r];
        }
#pragma unroll
        for (u32 r = 0; r < 4; ++r) {
            const u32 e = tid + r * 256;
            vs[e / 32][e % 32] = pv[r];
        }
        const float ra = pa, rb = pb;
        fetch(c0 + GC);
        if (warp == 0) {
            const u32 t = lane;
            float lg = t < n ? softplusf_(ra + dtb) * av : 0.f;
            bt[t] = t < n ? sigmoidf_(rb) : 0.f;
#pragma unroll
            for (u32 o = 1; o < 32; o <<= 1) {
                const float u = __shfl_up_sync(0xffffffffu, lg, o);
                if (lane >= o) lg += u;
            }
            gl[t] = lg;
        }
        __syncthreads();
        // A (s < t): one (t, s) pair per thread per round
        for (u32 e = tid; e < GC * GC; e += 256) {
            const u32 t = e / GC, s = e % GC;
            float a = 0.f;
            if (s < t && t < n) {
                a = bt[t] * __expf(gl[t] - gl[s]) * dot128(ks[t], ks[s]);
            }
            am[t][s] = a;
        }
        // u = K S0 and w = Q S0 for this thread's column, reduced over the 8 i-groups of the warp
#pragma unroll 4
        for (u32 t = 0; t < GC; ++t) {
            float u = 0.f, w = 0.f;
#pragma unroll
            for (u32 ii = 0; ii < 16; ++ii) {
                u = fmaf(ks[t][ig * 17 + ii], sreg[ii], u);
                w = fmaf(qs[t][ig * 17 + ii], sreg[ii], w);
            }
#pragma unroll
            for (u32 o = 1; o < 8; o <<= 1) {
                u += __shfl_xor_sync(0xffffffffu, u, o);
                w += __shfl_xor_sync(0xffffffffu, w, o);
            }
            if (ig == 0) {
                const float eg = __expf(gl[t]);
                ds[t][jl] = bt[t] * (vs[t][jl] - eg * u);  // the right-hand side, solved in place below
                vs[t][jl] = eg * w;                        // v is spent: keep e^g q S0
            }
        }
        __syncthreads();
        // forward substitution, one thread per column
        if (tid < 32) {
            for (u32 t = 1; t < n; ++t) {
                float x0 = ds[t][tid], x1 = 0.f, x2 = 0.f, x3 = 0.f;
                u32 s = 0;
                for (; s + 4 <= t; s += 4) {
                    x0 = fmaf(-am[t][s], ds[s][tid], x0);
                    x1 = fmaf(-am[t][s + 1], ds[s + 1][tid], x1);
                    x2 = fmaf(-am[t][s + 2], ds[s + 2][tid], x2);
                    x3 = fmaf(-am[t][s + 3], ds[s + 3][tid], x3);
                }
                for (; s < t; ++s) x0 = fmaf(-am[t][s], ds[s][tid], x0);
                ds[t][tid] = (x0 + x1) + (x2 + x3);
            }
        }
        __syncthreads();
        // P (s <= t), over A
        for (u32 e = tid; e < GC * GC; e += 256) {
            const u32 t = e / GC, s = e % GC;
            float p = 0.f;
            if (s <= t && t < n) {
                p = __expf(gl[t] - gl[s]) * dot128(qs[t], ks[s]);
            }
            am[t][s] = p;
        }
        __syncthreads();
        // outputs: thread (t, column) pairs
        for (u32 e = tid; e < GC * 32; e += 256) {
            const u32 t = e / 32, jj = e % 32;
            if (t >= n) continue;
            float o0 = vs[t][jj], o1 = 0.f, o2 = 0.f, o3 = 0.f;
            u32 s = 0;
            for (; s + 4 <= t + 1; s += 4) {
                o0 = fmaf(am[t][s], ds[s][jj], o0);
                o1 = fmaf(am[t][s + 1], ds[s + 1][jj], o1);
                o2 = fmaf(am[t][s + 2], ds[s + 2][jj], o2);
                o3 = fmaf(am[t][s + 3], ds[s + 3][jj], o3);
            }
            for (; s <= t; ++s) o0 = fmaf(am[t][s], ds[s][jj], o0);
            const float o = (o0 + o1) + (o2 + o3);
            F(2)[((u64)(c0 + t) * vh + h) * d + jc * 32 + jj] = o * rs;
        }
        // the state after the chunk
        const float gn = gl[n - 1];
        const float en = __expf(gn);
#pragma unroll
        for (u32 ii = 0; ii < 16; ++ii) sreg[ii] *= en;
        for (u32 t = 0; t < n; ++t) {
            const float cf = __expf(gn - gl[t]) * ds[t][jl];
#pragma unroll
            for (u32 ii = 0; ii < 16; ++ii) sreg[ii] = fmaf(ks[t][ig * 17 + ii], cf, sreg[ii]);
        }
    }
#pragma unroll
    for (u32 ii = 0; ii < 16; ++ii) S[(ig * 16 + ii) * d + j] = sreg[ii];
}

/// The chunked gated delta rule in two passes. With T = (I + A)^-1 per 32-token chunk (A as in qxp_gdn_chunk),
/// delta = U - W S0 where U = T (beta v) and W = T (beta e^g k) depend on the chunk alone; so does
/// P[t][s] = e^(g_t - g_s) q_t.k_s. qxp_gdn_prep computes U, W, P and g for every chunk and head at once;
/// qxp_gdn_scan then walks the chunks in order with plain products against the state.
/// info: [0] conv out (T x ch), [1] U out (T x vh x d), [2] W out (T x vh x d), [3] P out (chunks x vh x 32 x 32),
/// [4] g out (T x vh), [5] beta raw (T x vh), [6] alpha raw (T x vh), [7] dt bias, [8] a, [9] key heads, [10] dim,
/// [11] T, [12] ch, [13] value heads. Grid (ceil(T / 32), value heads), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qxp_gdn_prep(const u64* info) {
    __shared__ float ks[GC][GKP], qs[GC][GKP];
    __shared__ float am[GC][GC + 1];
    __shared__ float gl[GC], bt[GC];
    const u32 c = blockIdx.x, h = blockIdx.y, d = U(10), kh = U(9), T = U(11), ch = U(12), vh = U(13);
    const u32 tid = threadIdx.x, warp = tid / 32, lane = tid % 32, khh = h % kh;
    const u32 c0 = c * GC, n = min((u32)GC, T - c0);
    auto pad = [](u32 i) { return i + i / 16; };
    for (u32 e = tid; e < GC * 128; e += 256) {
        const u32 t = e / 128, i = e % 128;
        const float* cv = CF(0) + (u64)(c0 + t) * ch;
        ks[t][pad(i)] = t < n ? cv[(u64)kh * d + khh * d + i] : 0.f;
        qs[t][pad(i)] = t < n ? cv[khh * d + i] : 0.f;
    }
    if (warp == 0) {
        const u32 t = lane;
        float lg = t < n ? softplusf_(CF(6)[(u64)(c0 + t) * vh + h] + CF(7)[h]) * CF(8)[h] : 0.f;
        bt[t] = t < n ? sigmoidf_(CF(5)[(u64)(c0 + t) * vh + h]) : 0.f;
#pragma unroll
        for (u32 o = 1; o < 32; o <<= 1) {
            const float u = __shfl_up_sync(0xffffffffu, lg, o);
            if (lane >= o) lg += u;
        }
        gl[t] = lg;
        if (t < n) F(4)[(u64)(c0 + t) * vh + h] = lg;
    }
    __syncthreads();
    float* P = F(3) + ((u64)c * vh + h) * GC * GC;
    for (u32 e = tid; e < GC * GC; e += 256) {
        const u32 t = e / GC, s = e % GC;
        float a = 0.f, p = 0.f;
        if (s <= t && t < n) {
            const float r = __expf(gl[t] - gl[s]);
            if (s < t) a = bt[t] * r * dot128(ks[t], ks[s]);
            p = r * dot128(qs[t], ks[s]);
        }
        am[t][s] = a;
        P[e] = p;
    }
    __syncthreads();
    // (I + A) x = r by forward substitution, one column per thread: U's 128 value columns, then W's 128 key dims
    float x[GC];
#pragma unroll
    for (u32 t = 0; t < GC; ++t) {
        float r = 0.f;
        if (t < n) r = tid < 128 ? bt[t] * CF(0)[(u64)(c0 + t) * ch + 2ull * kh * d + (u64)h * d + tid] : bt[t] * __expf(gl[t]) * ks[t][pad(tid - 128)];
#pragma unroll
        for (u32 s = 0; s < t; ++s) r = fmaf(-am[t][s], x[s], r);
        x[t] = r;
    }
    float* out = tid < 128 ? F(1) : F(2);
    const u32 col = tid % 128;
#pragma unroll
    for (u32 t = 0; t < GC; ++t)
        if (t < n) out[((u64)(c0 + t) * vh + h) * d + col] = x[t];
}

/// The serial pass: block (head h, column chunk jc) keeps S[:, CW jc .. CW jc + CW - 1] in registers, thread
/// (column, i-group) holds 128 / IG rows of its column, and per chunk computes delta = U - W S0,
/// o = (e^g q S0 + P delta) / sqrt(d) and S = e^g_n S0 + sum_t e^(g_n - g_t) k_t delta_t.
/// info: [0] conv out, [1] S, [2] o out (T x vh x d), [3] U, [4] W, [5] P, [6] g, [7] key heads, [8] dim, [9] T,
/// [10] ch, [11] value heads. Grid (value heads, 128 / CW), 256 threads.
template <int CW>
QX_DEV void gdn_scan(const u64* info) {
    constexpr u32 IG = 256 / CW, IPT = 128 / IG, CJ = GC * CW / 256;
    // W, then q, then the decayed k of the chunk; rows padded by 4 floats per 16 (a thread's 16 values are
    // 16-byte aligned and the 8 i-groups of a warp hit distinct banks)
    __shared__ __align__(16) float buf[GC][160];
    __shared__ float us[GC][CW + 1], ds[GC][CW + 1], pm[GC][GC + 1];
    __shared__ float gl[GC];
    const u32 h = blockIdx.x, jc = blockIdx.y, d = U(8), kh = U(7), T = U(9), ch = U(10), vh = U(11);
    const u32 tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const u32 ig = lane % IG, jl = warp * (32 / IG) + lane / IG, j = jc * CW + jl;
    const u32 khh = h % kh;
    auto pad = [](u32 i) { return i + 4 * (i / 16); };
    const u32 rb = pad(ig * IPT);  // this thread's rows sit at rb .. rb + IPT - 1 of a padded row
    float* S = F(1) + (u64)h * d * d;
    float sreg[IPT];
#pragma unroll
    for (u32 ii = 0; ii < IPT; ++ii) sreg[ii] = S[(ig * IPT + ii) * d + j];
    const float rs = rsqrtf((float)d);
    const u32 nch = (T + GC - 1) / GC;
    // one chunk ahead in registers: W, q, k (16 each), U (CJ) and P (4), g (warp 0)
    float rw[16], rq[16], rk[16], ru[CJ], rp[4], rg = 0.f;
    auto rows = [&](u32 c) { return c < nch ? min((u32)GC, T - c * GC) : 0u; };
    auto fw = [&](u32 c) {
        const u32 c0 = c * GC, n = rows(c);
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256, t = e / 128, i = e % 128;
            rw[r] = t < n ? CF(4)[((u64)(c0 + t) * vh + h) * d + i] : 0.f;
        }
#pragma unroll
        for (u32 r = 0; r < CJ; ++r) {
            const u32 e = tid + r * 256, t = e / CW, jj = e % CW;
            ru[r] = t < n ? CF(3)[((u64)(c0 + t) * vh + h) * d + jc * CW + jj] : 0.f;
        }
#pragma unroll
        for (u32 r = 0; r < 4; ++r) rp[r] = n ? CF(5)[((u64)c * vh + h) * GC * GC + tid + r * 256] : 0.f;
        if (warp == 0) rg = lane < n ? CF(6)[(u64)(c0 + lane) * vh + h] : 0.f;
    };
    auto fqk = [&](float* reg, u32 c, u64 off) {
        const u32 c0 = c * GC, n = rows(c);
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256, t = e / 128, i = e % 128;
            reg[r] = t < n ? CF(0)[(u64)(c0 + t) * ch + off + i] : 0.f;
        }
    };
    const u64 qoff = khh * d, koff = (u64)kh * d + khh * d;
    fw(0);
    fqk(rq, 0, qoff);
    fqk(rk, 0, koff);
    // S0 times the rows of buf, summed over the column's i-groups
    auto rowdot = [&](u32 t) {
        float u = 0.f;
#pragma unroll
        for (u32 ii = 0; ii < IPT; ii += 4) {
            const float4 v = *reinterpret_cast<const float4*>(&buf[t][rb + ii]);
            u = fmaf(v.x, sreg[ii], u);
            u = fmaf(v.y, sreg[ii + 1], u);
            u = fmaf(v.z, sreg[ii + 2], u);
            u = fmaf(v.w, sreg[ii + 3], u);
        }
#pragma unroll
        for (u32 o = 1; o < IG; o <<= 1) u += __shfl_xor_sync(0xffffffffu, u, o);
        return u;
    };
    for (u32 c = 0; c < nch; ++c) {
        const u32 c0 = c * GC, n = rows(c);
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256;
            buf[e / 128][pad(e % 128)] = rw[r];
        }
#pragma unroll
        for (u32 r = 0; r < CJ; ++r) {
            const u32 e = tid + r * 256;
            us[e / CW][e % CW] = ru[r];
        }
#pragma unroll
        for (u32 r = 0; r < 4; ++r) {
            const u32 e = tid + r * 256;
            pm[e / GC][e % GC] = rp[r];
        }
        if (warp == 0) gl[lane] = rg;
        fw(c + 1);
        __syncthreads();
        // delta = U - W S0
#pragma unroll 4
        for (u32 t = 0; t < GC; ++t) {
            const float u = rowdot(t);
            if (ig == 0) ds[t][jl] = us[t][jl] - u;
        }
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256;
            buf[e / 128][pad(e % 128)] = rq[r];
        }
        fqk(rq, c + 1, qoff);
        __syncthreads();
        // e^g q S0, over U (spent)
#pragma unroll 4
        for (u32 t = 0; t < GC; ++t) {
            const float w = rowdot(t);
            if (ig == 0) us[t][jl] = __expf(gl[t]) * w;
        }
        __syncthreads();
        const float gn = gl[n - 1];
#pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            const u32 e = tid + r * 256, t = e / 128;
            buf[t][pad(e % 128)] = rk[r] * __expf(gn - gl[t]);
        }
        fqk(rk, c + 1, koff);
        for (u32 e = tid; e < GC * CW; e += 256) {
            const u32 t = e / CW, jj = e % CW;
            if (t >= n) continue;
            float o0 = us[t][jj], o1 = 0.f, o2 = 0.f, o3 = 0.f;
            u32 s = 0;
            for (; s + 4 <= t + 1; s += 4) {
                o0 = fmaf(pm[t][s], ds[s][jj], o0);
                o1 = fmaf(pm[t][s + 1], ds[s + 1][jj], o1);
                o2 = fmaf(pm[t][s + 2], ds[s + 2][jj], o2);
                o3 = fmaf(pm[t][s + 3], ds[s + 3][jj], o3);
            }
            for (; s <= t; ++s) o0 = fmaf(pm[t][s], ds[s][jj], o0);
            F(2)[((u64)(c0 + t) * vh + h) * d + jc * CW + jj] = ((o0 + o1) + (o2 + o3)) * rs;
        }
        __syncthreads();
        const float en = __expf(gn);
#pragma unroll
        for (u32 ii = 0; ii < IPT; ++ii) sreg[ii] *= en;
        for (u32 t = 0; t < n; ++t) {
            const float dv = ds[t][jl];
#pragma unroll
            for (u32 ii = 0; ii < IPT; ii += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&buf[t][rb + ii]);
                sreg[ii] = fmaf(v.x, dv, sreg[ii]);
                sreg[ii + 1] = fmaf(v.y, dv, sreg[ii + 1]);
                sreg[ii + 2] = fmaf(v.z, dv, sreg[ii + 2]);
                sreg[ii + 3] = fmaf(v.w, dv, sreg[ii + 3]);
            }
        }
    }
#pragma unroll
    for (u32 ii = 0; ii < IPT; ++ii) S[(ig * IPT + ii) * d + j] = sreg[ii];
}

extern "C" __global__ void __launch_bounds__(256) qxp_gdn_scan(const u64* info) { gdn_scan<32>(info); }

/// GDN output per (head, token): on = rms(o) * w * sigmoid(z). info: [0] o, [1] w, [2] z, [3] out, [4] dim,
/// [5] eps bits, [6] value heads. Grid (vh, T), 128 threads.
extern "C" __global__ void qxp_gdn_out(const u64* info) {
    __shared__ float sc[32];
    const u32 h = blockIdx.x, t = blockIdx.y, d = U(4), vh = U(6);
    const u64 base = ((u64)t * vh + h) * d;
    const float* o = CF(0) + base;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) ss += o[i] * o[i];
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)d + __uint_as_float(U(5)));
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) F(3)[base + i] = o[i] * r * CF(1)[i] * sigmoidf_(CF(2)[base + i]);
}

/// Per (head, token) RMS norm and RoPE at position p0 + t. info: [0] x, [1] p0 ptr, [2] w, [3] out, [4] x stride
/// per head, [5] dim, [6] n_rot, [7] x row stride, [8] base bits, [9] eps bits, [10] out stride per head,
/// [11] out row stride. Grid (heads, T), dim threads.
extern "C" __global__ void qxp_norm_rope(const u64* info) {
    __shared__ float sc[32];
    __shared__ float buf[256];
    const u32 h = blockIdx.x, t = blockIdx.y, d = U(5), nr = U(6), i = threadIdx.x;
    const float* x = CF(0) + (u64)t * info[7] + (u64)h * U(4);
    const float ss = block_sum(i < d ? x[i] * x[i] : 0.f, sc);
    const float r = rsqrtf(ss / (float)d + __uint_as_float(U(9)));
    if (i < d) buf[i] = x[i] * r * CF(2)[i];
    __syncthreads();
    if (i < d) {
        float v = buf[i];
        if (i < nr) {
            const u32 p = i % (nr / 2);
            const float theta = (float)(P0 + t) * powf(__uint_as_float(U(8)), -2.f * (float)p / (float)nr);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = buf[p], x1 = buf[p + nr / 2];
            v = i < nr / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        F(3)[(u64)t * info[11] + (u64)h * U(10) + i] = v;
    }
}

/// The second half of each 2*dim chunk, per token. info: [0] x (T x 2*heads*dim), [1] out (T x heads*dim),
/// [2] dim, [3] heads, [4] T.
extern "C" __global__ void qxp_take_halves(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, d = U(2), n = (u64)U(3) * d;
    if (i >= n * U(4)) return;
    const u64 t = i / n, j = i % n;
    F(1)[i] = CF(0)[t * 2 * n + (j / d) * 2 * d + d + j % d];
}

/// FP8 KV append for every token of the chunk: grid (kv, 2, T), dim/2 threads; as qx_kv_append_fp8 with the
/// position p0 + t. info: [0] k (T x kv x dim), [1] v, [2] this layer's table entry for the slot, [10] p0 ptr,
/// [11] window W, [12] kv, [13] dim.
extern "C" __global__ void qxp_kv_append_fp8(const u64* info) {
    const u32 g = blockIdx.x, which = blockIdx.y, t = blockIdx.z, nkv = U(12), d = U(13), i = threadIdx.x;
    const u32 pos = *reinterpret_cast<const u32*>(info[10]) + t, W = U(11);
    const float* src = CF(which) + ((u64)t * nkv + g) * d;
    const float a = src[2 * i], b = src[2 * i + 1];
    // one scale per 32-dim group (16 threads)
    const float amax = group16_max(fmaxf(fabsf(a), fabsf(b)));
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const u16 q = f32x2_to_e4m3x2(a / scale, b / scale);
    const u64 rcell = (u64)(pos % W) * nkv + g;
    const u64* tab = reinterpret_cast<const u64*>(info[2]);
    float* hs;
    u8* hrow = kv_page_row(tab, which == 1, pos, g, nkv, d, hs);
    reinterpret_cast<u16*>(tab[which])[rcell * (d / 2) + i] = q;
    reinterpret_cast<u16*>(hrow)[i] = q;
    if (i % 16 == 0) {
        reinterpret_cast<float*>(tab[2 + which])[rcell * QX_KV_SG + i / 16] = scale;
        hs[i / 16] = scale;
    }
}

/// Indexer pooling for every block the chunk completes: block b (cells 4b .. 4b+3, the last in the chunk) is
/// pooled from the chunk's raw keys and, for cells before p0, the ring of the previous block's raw keys.
/// info: [0] chunk raw keys (T x dim), [1] p0 ptr, [2] w, [3] pooled keys, [4] dim, [5] n_rot, [6] base bits,
/// [7] eps bits, [8] block size, [9] ring (ring slots x dim, cell c at c % slots), [10] T, [11] ring slots.
/// Grid: ceil(T / block) + 1 blocks of dim threads.
extern "C" __global__ void qxp_idx_pool(const u64* info) {
    __shared__ float sc[32];
    __shared__ float buf[256];
    const u32 d = U(4), nr = U(5), kp = U(8), T = U(10), i = threadIdx.x, p0 = P0;
    const u32 b = p0 / kp + blockIdx.x;  // candidate block
    const u32 last = b * kp + kp - 1;
    if (last < p0 || last >= p0 + T) return;
    float m = 0.f;
    if (i < d) {
        for (u32 c = b * kp; c <= last; ++c) m += c >= p0 ? CF(0)[(u64)(c - p0) * d + i] : CF(9)[(u64)(c % U(11)) * d + i];
        m /= (float)kp;
    }
    const float ss = block_sum(i < d ? m * m : 0.f, sc);
    const float r = rsqrtf(ss / (float)d + __uint_as_float(U(7)));
    if (i < d) buf[i] = m * r * CF(2)[i];
    __syncthreads();
    if (i < d) {
        float v = buf[i];
        if (i < nr) {
            const u32 p = i % (nr / 2);
            const float theta = (float)(b * kp) * powf(__uint_as_float(U(6)), -2.f * (float)p / (float)nr);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = buf[p], x1 = buf[p + nr / 2];
            v = i < nr / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        F(3)[(u64)b * d + i] = v;
    }
}

/// The raw-key ring after the chunk: slot c % slots gets cell c for the chunk's last cells. info: [0] chunk raw
/// keys, [1] p0 ptr, [2] ring, [3] dim, [4] ring slots, [5] T. One block of dim threads.
extern "C" __global__ void qxp_idx_ring(const u64* info) {
    const u32 d = U(3), rs = U(4), T = U(5), p0 = P0, i = threadIdx.x;
    if (i >= d) return;
    for (u32 j = (T > rs ? T - rs : 0); j < T; ++j) F(2)[(u64)((p0 + j) % rs) * d + i] = CF(0)[(u64)j * d + i];
}

/// The selection scores for the whole chunk as a tiled product: block (x, y) scores blocks 64x .. 64x + 63 for
/// queries 64y .. 64y + 63, each thread a 4 x 4 tile from 32-dim slices of Q and the pooled keys in shared memory.
/// Each score is the same chain of operations as `qxp_qsa_select`'s own loop (per head an fma over the dims in
/// order, relu, summed over heads in order, times 1/sqrt(dim)), so the scores are the same bits.
/// info: [0] query indexer heads (T x heads x dim), [1] p0 ptr, [2] pooled keys, [3] scores (T x max blocks),
/// [4] heads, [5] dim, [6] block, [7] top, [8] max blocks, [9] T. Grid (ceil(blocks / 64), ceil(T / 64)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qxp_idx_score(const u64* info) {
    __shared__ __align__(16) float qs[32][68];
    __shared__ __align__(16) float ks[32][68];
    const u32 heads = U(4), d = U(5), kp = U(6), top = U(7), T = U(9), p0 = P0;
    const u32 tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const u32 t0 = blockIdx.y * 64, b0 = blockIdx.x * 64;
    const u32 t_hi = min(t0 + 64, T) - 1;
    const u32 nv_hi = (p0 + t_hi + 1) / kp;
    if (nv_hi <= top || b0 >= nv_hi) return;  // every query here takes all its blocks, or no block is visible
    float tot[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[a][c] = 0.f;
    for (u32 h = 0; h < heads; ++h) {
        float s[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int c = 0; c < 4; ++c) s[a][c] = 0.f;
        for (u32 i0 = 0; i0 < d; i0 += 32) {
            __syncthreads();
            for (u32 e = tid; e < 64 * 32; e += 256) {
                const u32 r = e / 32, i = e % 32;
                const u32 t = t0 + r, b = b0 + r;
                qs[i][r] = t < T ? CF(0)[((u64)t * heads + h) * d + i0 + i] : 0.f;
                ks[i][r] = b < nv_hi ? CF(2)[(u64)b * d + i0 + i] : 0.f;
            }
            __syncthreads();
#pragma unroll 8
            for (u32 i = 0; i < 32; ++i) {
                const float4 q = *reinterpret_cast<const float4*>(&qs[i][ty * 4]);
                const float4 k = *reinterpret_cast<const float4*>(&ks[i][tx * 4]);
                const float qv[4] = {q.x, q.y, q.z, q.w}, kv[4] = {k.x, k.y, k.z, k.w};
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int c = 0; c < 4; ++c) s[a][c] = fmaf(qv[a], kv[c], s[a][c]);
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int c = 0; c < 4; ++c) tot[a][c] += fmaxf(s[a][c], 0.f);
    }
    const float inv = rsqrtf((float)d);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const u32 t = t0 + ty * 4 + a;
        if (t >= T) continue;
        const u32 nv = (p0 + t + 1) / kp;
        float* sc = F(3) + (u64)t * info[8];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const u32 b = b0 + tx * 4 + c;
            if (b < nv) sc[b] = tot[a][c] * inv;
        }
    }
}

/// Per-query QSA selection for the chunk: query t at position r = p0 + t keeps every cell of its top `top`
/// visible blocks (ties to the lower block), or of all visible blocks when there are no more, then the tail.
/// info: [0] query indexer heads (T x heads x dim), [1] p0 ptr, [2] pooled keys, [3] cells out (T x stride),
/// [4] counts out (T), [5] heads, [6] dim, [7] block, [8] top, [9] scores scratch (T x max blocks),
/// [10] max blocks, [11] cells stride, [12] 1: the scores are already in [9] (`qxp_idx_score`). Grid T blocks of
/// 1024 threads.
extern "C" __global__ void qxp_qsa_select(const u64* info) {
    __shared__ u32 hist[256];
    __shared__ u32 s_prefix, s_need, s_count;
    __shared__ u32 scan[1024];
    __shared__ float qi[1024];
    const u32 t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const u32 r = P0 + t, heads = U(5), d = U(6), kp = U(7), top = U(8);
    const u32 nv = (r + 1) / kp, n_tail = (r + 1) % kp;
    u32* cells = reinterpret_cast<u32*>(info[3]) + (u64)t * info[11];
    if (nv <= top) {
        for (u32 c = tid; c < nv * kp; c += nt) cells[c] = c;
        if (tid < n_tail) cells[nv * kp + tid] = r - tid;
        if (tid == 0) reinterpret_cast<u32*>(info[4])[t] = nv * kp + n_tail;
        return;
    }
    for (u32 i = tid; i < heads * d; i += nt) qi[i] = CF(0)[(u64)t * heads * d + i];
    __syncthreads();
    float* sc = F(9) + (u64)t * info[10];
    const float inv = rsqrtf((float)d);
    for (u32 b = U(12) ? nv : tid; b < nv; b += nt) {
        const float* kb = CF(2) + (u64)b * d;
        float tot = 0.f;
        for (u32 h = 0; h < heads; ++h) {
            float s = 0.f;
            for (u32 i = 0; i < d; ++i) s += qi[h * d + i] * kb[i];
            tot += fmaxf(s, 0.f);
        }
        sc[b] = tot * inv;
    }
    __syncthreads();
    auto key = [&](u32 b) { const u32 u = __float_as_uint(sc[b]); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
    u32 prefix = 0, need = top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (u32 i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const u32 hmask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (u32 b = tid; b < nv; b += nt) {
            const u32 k = key(b);
            if ((k & hmask) == (prefix & hmask)) atomicAdd(&hist[(k >> shift) & 0xff], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            u32 acc = 0, dd = 255;
            for (;; --dd) {
                if (acc + hist[dd] >= need) break;
                acc += hist[dd];
                if (dd == 0) break;
            }
            s_prefix = prefix | (dd << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        __syncthreads();
    }
    if (tid == 0) s_count = 0;
    __syncthreads();
    u32 taken_eq = 0;
    for (u32 base = 0; base < nv; base += nt) {
        const u32 b = base + tid;
        const u32 k = b < nv ? key(b) : 0;
        const u32 eq = (b < nv && k == prefix) ? 1u : 0u;
        scan[tid] = eq;
        __syncthreads();
        for (u32 off = 1; off < nt; off <<= 1) {
            const u32 v = tid >= off ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const u32 eq_before = scan[tid] - eq;
        const u32 eq_total = scan[nt - 1];
        const u32 take = (b < nv && (k > prefix || (eq && taken_eq + eq_before < need))) ? 1u : 0u;
        __syncthreads();
        // slots in block order (a prefix sum, not an atomic counter): the attention sums cells in list order,
        // so an order that varied run to run made the chunk's logits vary too
        scan[tid] = take;
        __syncthreads();
        for (u32 off = 1; off < nt; off <<= 1) {
            const u32 v = tid >= off ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        if (take) {
            const u32 slot = s_count + scan[tid] - 1;
            for (u32 c = 0; c < kp; ++c) cells[slot * kp + c] = b * kp + c;
        }
        __syncthreads();
        if (tid == 0) s_count += scan[nt - 1];
        taken_eq += eq_total;
        __syncthreads();
    }
    if (tid < n_tail) cells[top * kp + tid] = r - tid;
    if (tid == 0) reinterpret_cast<u32*>(info[4])[t] = top * kp + n_tail;
}

// ---- MoE for a chunk ----

/// Routing per token: as qx_moe_route on row t. info: [0] logits (T x n), [1] n, [2] k, [3] ids (T x k),
/// [4] weights (T x k). Grid T blocks of 32 threads.
extern "C" __global__ void qxp_route(const u64* info) {
    const u32 n = U(1), k = U(2), l = threadIdx.x, t = blockIdx.x;
    const float* lg = CF(0) + (u64)t * n;
    float v[32];
    const u32 per = (n + 31) / 32;
    float m = -3.402823466e38f;
    for (u32 i = 0; i < per; ++i) {
        const u32 e = i * 32 + l;
        v[i] = e < n ? lg[e] : -3.402823466e38f;
        m = fmaxf(m, v[i]);
    }
    m = warp_max(m);
    float z = 0.f;
    for (u32 i = 0; i < per; ++i) {
        const u32 e = i * 32 + l;
        v[i] = e < n ? expf(v[i] - m) : -1.f;
        if (e < n) z += v[i];
    }
    z = warp_sum(z);
    float sel = 0.f, tot = 0.f;
    i32 myid = 0;
    for (u32 r = 0; r < k; ++r) {
        float bv = -1.f;
        i32 bi = 0x7fffffff;
        for (u32 i = 0; i < per; ++i) {
            const i32 e = (i32)(i * 32 + l);
            if (v[i] > bv || (v[i] == bv && e < bi)) { bv = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const i32 oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const float p = bv / z;
        tot += p;
        if (l == r) { sel = p; myid = bi; }
        // bi stays at the sentinel when no lane had a candidate (NaN logits): never index with it
        if (bi != 0x7fffffff && (u32)(bi % 32) == l) v[bi / 32] = -2.f;
    }
    tot = fmaxf(tot, 6.103515625e-5f);
    if (l < k) {
        reinterpret_cast<i32*>(info[3])[(u64)t * k + l] = myid;
        F(4)[(u64)t * k + l] = sel / tot;
    }
}

/// Groups the chunk's assignments (a = t * k + j) by expert and cuts each expert's rows into tiles.
/// Also sets each used expert's weight pointer: its VRAM slot, or its record in device-mapped host memory.
/// info: [0] ids (A), [1] A, [2] n_expert, [3] sorted out (A), [4] expert row ends out (n_expert),
/// [5] tile expert out, [6] tile first row out, [7] tile count out, [8] ptrs out (n_expert), [9] table (this
/// layer's n_expert entries), [10] host record 0, [11] host record stride, [12] rows per tile, [13] this
/// layer's routing histogram (u32 per expert, counts added; 0 for none). One block of 1024 threads.
extern "C" __global__ void qxp_moe_group(const u64* info) {
    __shared__ u32 cnt[1024], off[1024], cur[1024];
    const u32 A = U(1), ne = U(2), tid = threadIdx.x;
    const i32* ids = reinterpret_cast<const i32*>(info[0]);
    for (u32 e = tid; e < ne; e += blockDim.x) { cnt[e] = 0; cur[e] = 0; }
    __syncthreads();
    for (u32 a = tid; a < A; a += blockDim.x) atomicAdd(&cnt[ids[a]], 1u);
    __syncthreads();
    if (tid == 0) {
        u32 acc = 0, tiles = 0;
        u32* tile_e = reinterpret_cast<u32*>(info[5]);
        u32* tile_r = reinterpret_cast<u32*>(info[6]);
        for (u32 e = 0; e < ne; ++e) {
            off[e] = acc;
            for (u32 r = 0; r < cnt[e]; r += U(12)) { tile_e[tiles] = e; tile_r[tiles] = acc + r; ++tiles; }
            acc += cnt[e];
            reinterpret_cast<u32*>(info[4])[e] = acc;
        }
        *reinterpret_cast<u32*>(info[7]) = tiles;
    }
    __syncthreads();
    for (u32 a = tid; a < A; a += blockDim.x) {
        const u32 e = (u32)ids[a];
        reinterpret_cast<u32*>(info[3])[off[e] + atomicAdd(&cur[e], 1u)] = a;
    }
    const u64* table = reinterpret_cast<const u64*>(info[9]);
    for (u32 e = tid; e < ne; e += blockDim.x) {
        const u64 slot = table[e];
        reinterpret_cast<u64*>(info[8])[e] = slot ? slot : info[10] + (u64)e * info[11];
        if (info[13]) reinterpret_cast<u32*>(info[13])[e] += cnt[e];
    }
}

/// h[r][i] = silu(gu[r][i]) * gu[r][ff + i] for the A sorted rows. info: [0] gu (A x 2ff), [1] h (A x ff), [2] A, [3] ff.
extern "C" __global__ void qxp_silu_mul(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, ff = U(3);
    if (i >= (u64)U(2) * ff) return;
    const u64 r = i / ff, j = i % ff;
    F(1)[i] = qx_silu_mul(CF(0)[r * 2 * ff + j], CF(0)[r * 2 * ff + ff + j]);
}

/// y[t][i] = sum_j w[t][j] v[t*k + j][i] + sigmoid(sg[t]) sh[t][i]. info: [0] v (A x E), [1] w (T x k), [2] y (T x E),
/// [3] E, [4] k, [5] shared out (T x E), [6] shared gate (T), [7] T.
extern "C" __global__ void qxp_moe_combine(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, n = U(3), k = U(4);
    if (i >= (u64)U(7) * n) return;
    const u64 t = i / n, e = i % n;
    float s = 0.f;
    for (u64 j = 0; j < k; ++j) s = __fadd_rn(s, __fmul_rn(CF(1)[t * k + j], CF(0)[(t * k + j) * n + e]));
    F(2)[i] = s + sigmoidf_(CF(6)[t]) * CF(5)[i];
}

/// silu(a) * b elementwise over T x n. info: [0] a, [1] b, [2] out, [3] count.
extern "C" __global__ void qxp_silu_mul2(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < info[3]) F(2)[i] = siluf_(CF(0)[i]) * CF(1)[i];
}

/// a[i] *= sigmoid(b[i]) over count elements. info: [0] a, [1] b, [2] count.
extern "C" __global__ void qxp_mul_sigmoid(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < info[2]) F(0)[i] *= sigmoidf_(CF(1)[i]);
}

// ---- PLE for a chunk ----

/// Waits for the PLE embeddings of positions p0 .. p0 + T - 1 in the mapped ring (slot pos % cap), copies them
/// out and marks them consumed. info: [0] PLE mailbox, [1] p0 ptr, [2] emb out (T x E), [3] E, [4] T, [5] cap.
/// One block.
extern "C" __global__ void qxp_ple_wait(const u64* info) {
    const u8* mb = reinterpret_cast<const u8*>(info[0]);
    const u32 p0 = P0, T = U(4), E = U(3), cap = U(5);
    if (threadIdx.x == 0)
        while (ld_acquire_sys_ple(reinterpret_cast<const u64*>(mb + 64)) < (u64)p0 + T) {
        }
    __syncthreads();
    const volatile float* ring = reinterpret_cast<const volatile float*>(mb + 128);
    for (u64 i = threadIdx.x; i < (u64)T * E; i += blockDim.x) {
        const u64 t = i / E;
        F(2)[i] = ring[((p0 + t) % cap) * E + i % E];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(const_cast<u8*>(mb)), "l"((u64)p0 + T) : "memory");
    }
}

/// PLE gates per (stream s, token t), as qx_ple_gate on row t. info: [0] kn (T x HC*E), [1] qn (T x HC*E),
/// [2] g out (T x HC), [3] E. Grid (HC, T).
extern "C" __global__ void qxp_ple_gate(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, t = blockIdx.y, n = U(3), hc = gridDim.x;
    const u64 base = ((u64)t * hc + s) * n;
    float v = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) v += CF(0)[base + i] * CF(1)[base + i];
    v = block_sum(v, sc) * rsqrtf((float)n);
    if (threadIdx.x == 0) {
        const float sg = v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f);
        F(2)[(u64)t * hc + s] = sigmoidf_(sg * sqrtf(fmaxf(fabsf(v), 1e-6f)));
    }
}

/// Sum of squares of the gated values per (stream, token): ss[t][s] = sum_e (value[t][e] g[t][s])^2.
/// info: [0] value (T x E), [1] g (T x HC), [2] ss out (T x HC), [3] E. Grid (HC, T).
extern "C" __global__ void qxp_ple_gss(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, t = blockIdx.y, n = U(3), hc = gridDim.x;
    const float g = CF(1)[(u64)t * hc + s];
    float v = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = CF(0)[(u64)t * n + i] * g;
        v += x * x;
    }
    v = block_sum(v, sc);
    if (threadIdx.x == 0) F(2)[(u64)t * hc + s] = v;
}

/// PLE apply over the chunk, one thread per channel c (stream c / E) walking the tokens in order: gated = value * g,
/// cn = rms(gated) * w_conv, conv over the last 9 cn (dilation 3, kernel 4), R += gated + silu(conv); the history
/// holds the 9 latest cn afterwards. info: [0] value (T x E), [1] g (T x HC), [2] ss (T x HC), [3] w_conv norm,
/// [4] conv w (HCD x 4), [5] hist (9 x HCD), [6] R (T x HCD), [7] E, [8] eps bits, [9] T, [10] HC.
extern "C" __global__ void qxp_ple_apply(const u64* info) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x, n = U(7), hc = U(10), hcd = n * hc, T = U(9);
    if (c >= hcd) return;
    const u32 s = c / n, e = c % n;
    float hist[9];
    for (int i = 0; i < 9; ++i) hist[i] = CF(5)[(u64)i * hcd + c];
    const float* w = CF(4) + 4ull * c;
    const float wn = CF(3)[c], eps = __uint_as_float(U(8));
    for (u32 t = 0; t < T; ++t) {
        const float g = CF(1)[(u64)t * hc + s];
        const float gated = CF(0)[(u64)t * n + e] * g;
        const float r = rsqrtf(CF(2)[(u64)t * hc + s] / (float)n + eps);
        const float cn = gated * r * wn;
        const float acc = w[0] * hist[0] + w[1] * hist[3] + w[2] * hist[6] + w[3] * cn;
        for (int i = 0; i < 8; ++i) hist[i] = hist[i + 1];
        hist[8] = cn;
        F(6)[(u64)t * hcd + c] += gated + siluf_(acc);
    }
    for (int i = 0; i < 9; ++i) F(5)[(u64)i * hcd + c] = hist[i];
}

/// h = silu(g) * u for the sorted expert rows, with the deterministic silu. info: [0] g, [1] u, [2] h, [3] count.
extern "C" __global__ void qxp_silu_mul2_rows(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < info[3]) F(2)[i] = qx_silu_mul(CF(0)[i], CF(1)[i]);
}

/// Token embeddings of the chunk: e[t] = Q6_K row ctx[p0 + t]. One warp per token.
/// info: [0] table, [1] ctx tokens, [2] p0 ptr, [3] e out (T x E), [4] T, [5] row bytes.
extern "C" __global__ void qxp_embed_q6k(const u64* info) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= U(4)) return;
    const u32 tok = reinterpret_cast<const u32*>(info[1])[*reinterpret_cast<const u32*>(info[2]) + warp];
    const u8* row = reinterpret_cast<const u8*>(info[0]) + (u64)tok * info[5];
    float* o = F(3) + (u64)warp * 2560;
    for (u32 cidx = 0; cidx < 10; ++cidx) {
        float w[8];
        decode8<QX_Q6_K>(row, cidx, lane, w);
        for (int i = 0; i < 8; ++i) o[cidx * 256 + 8 * lane + i] = w[i];
    }
}

// ---- expert staging for a chunk: the copy stream fills a VRAM staging half, the chunk's MoE consumes it ----

/// Spins until the u32 at info[0] is >= info[1] (a counter the copy stream writes). One thread.
extern "C" __global__ void qxp_wait_ge(const u64* info) {
    const volatile u32* p = reinterpret_cast<const volatile u32*>(info[0]);
    const u32 v = (u32)info[1];
    while ((int)(*p - v) < 0) {
    }
    __threadfence();
}

/// Publishes info[1] to the u64 at info[0] (mapped host memory the copy thread polls) once every earlier
/// kernel in the stream is done. One thread.
extern "C" __global__ void qxp_signal(const u64* info) {
    __threadfence_system();
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(info[0]), "l"(info[1]) : "memory");
}

/// h = silu(g) * u for the sorted rows of experts [e_lo, e_hi). info: [0] g, [1] u, [2] h, [3] expert row ends,
/// [4] e_lo, [5] e_hi, [6] ff, [7] A.
extern "C" __global__ void qxp_silu_mul2_range(const u64* info) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x, ff = info[6];
    if (i >= info[7] * ff) return;
    const u32* ends = reinterpret_cast<const u32*>(info[3]);
    const u32 lo = U(4), hi = U(5);
    const u64 r = i / ff, r0 = lo ? ends[lo - 1] : 0, r1 = ends[hi - 1];
    if (r >= r0 && r < r1) F(2)[i] = qx_silu_mul(CF(0)[i], CF(1)[i]);
}
