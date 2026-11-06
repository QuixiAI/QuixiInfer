// DeepSeek-V4.1-Flash decode kernels: the model-specific pieces around the shared MGemv, MoE and step kernels.
// The arithmetic follows crates/qx-ref/src/dsv41.rs (the CPU reference) op for op; where the reference has a
// choice of order, the kernels take the same one, so their results are compared tightly against it.
//
// Layouts (floats unless said): the residual R is [hc][E] per token; the hyper-connection mixes are 24 per token
// ([0..4) pre, [4..8) post, [8..24) comb raw, comb index src*4 + dst); the latent kv rows are 512; a compressed
// row is [latent (512) | index key (128)]; a window of T tokens has token t's rows t * stride apart.
#include "qx_common.cuh"
#include "qx_block.cuh"
#include "qx_math.cuh"
#include "qx_sys.cuh"

#define F(i) reinterpret_cast<float*>(info[i])
#define CF(i) reinterpret_cast<const float*>(info[i])
#define U(i) ((u32)info[i])
#define QX_NINF __int_as_float(0xff800000)
#define I(i) ((i32)info[i])

QX_DEV float sigmoid_rn(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

// ---- hyper-connections ------------------------------------------------------------------------------------

/// The hyper-connection mixes: m[r] = fn[r] . rn for the 24 rows of fn (F32, rows x n) and the RMS-normed flat
/// residual rn (n = hc*E). One block of 32 * rows threads per token: warp r takes row r.
/// info: [0] fn, [1] rn (T x n), [2] mixes out (T x rows), [3] n, [4] rows. Grid T.
extern "C" __global__ void qx_ds_hc_mixes(const u64* info) {
    // split over gridDim.y blocks per output: block s covers [s * n / S, (s + 1) * n / S); partial s of output r of
    // token t goes to out[(t * R + r) * S + s], summed in order by qx_ds_hc_terms
    __shared__ float red[8];
    const u32 r = blockIdx.x, s = blockIdx.y, S = gridDim.y, t = blockIdx.z, n = U(3), R = U(4);
    const u32 w_ = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* w = CF(0) + (u64)r * n;
    const float* x = CF(1) + (u64)t * n;
    const u32 i0 = s * n / S, i1 = (s + 1) * n / S;
    float acc = 0.f;
    for (u32 i = i0 + threadIdx.x; i < i1; i += blockDim.x) acc += w[i] * x[i];
    acc = warp_sum(acc);
    if (lane == 0) red[w_] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (u32 i = 0; i < blockDim.x / 32; ++i) sum += red[i];
        F(2)[((u64)t * R + r) * S + s] = sum;
    }
}

/// From a token's 24 mixes: pre[c] = sigmoid(m[c] * scale[0] + base[c]) + eps, post[c] = 2 * sigmoid(m[4+c] *
/// scale[1] + base[4+c]), comb = Sinkhorn(softmax over dst of m[8+s*4+d] * scale[2] + base[8+s*4+d]) with `iters`
/// column/row normalizations as the reference does them. info: [0] mixes (T x 24), [1] scale (3), [2] base (24),
/// [3] pre out (T x 4), [4] post out (T x 4), [5] comb out (T x 16), [6] eps bits, [7] iters. Grid T, 32 threads.
extern "C" __global__ void qx_ds_hc_terms(const u64* info) {
    __shared__ float c[16];
    __shared__ float sum[4];
    __shared__ float m[24];
    const u32 t = blockIdx.x, l = threadIdx.x, S = U(8);
    // the mixes arrive as S partials per value (qx_ds_hc_mixes), summed here in a fixed order
    if (l < 24) {
        const float* mp = CF(0) + ((u64)t * 24 + l) * S;
        float a = 0.f;
        for (u32 s = 0; s < S; ++s) a += mp[s];
        m[l] = a;
        F(9)[(u64)t * 24 + l] = a;
    }
    __syncthreads();
    const float* sc = CF(1);
    const float* base = CF(2);
    const float eps = __uint_as_float(U(6));
    if (l < 4) {
        F(3)[(u64)t * 4 + l] = sigmoid_rn(m[l] * sc[0] + base[l]) + eps;
        F(4)[(u64)t * 4 + l] = 2.f * sigmoid_rn(m[4 + l] * sc[1] + base[4 + l]);
    }
    if (l < 16) c[l] = m[8 + l] * sc[2] + base[8 + l];
    __syncthreads();
    // softmax over dst (4 consecutive entries) for each src row s
    if (l < 4) {
        const u32 s = l;
        float mx = fmaxf(fmaxf(c[4 * s], c[4 * s + 1]), fmaxf(c[4 * s + 2], c[4 * s + 3]));
        float e[4], z = 0.f;
        for (int d = 0; d < 4; ++d) { e[d] = expf(c[4 * s + d] - mx); z += e[d]; }
        for (int d = 0; d < 4; ++d) c[4 * s + d] = e[d] / z + eps;
    }
    __syncthreads();
    const u32 iters = U(7);
    // norm_cols: each dst column over src; norm_rows: each src row over dst
    for (u32 it = 0; it < iters; ++it) {
        if (it > 0) {
            if (l < 4) sum[l] = c[4 * l] + c[4 * l + 1] + c[4 * l + 2] + c[4 * l + 3] + eps;
            __syncthreads();
            if (l < 16) c[l] /= sum[l / 4];
            __syncthreads();
        }
        if (l < 4) sum[l] = c[l] + c[4 + l] + c[8 + l] + c[12 + l] + eps;
        __syncthreads();
        if (l < 16) c[l] /= sum[l % 4];
        __syncthreads();
    }
    if (l < 16) F(5)[(u64)t * 16 + l] = c[l];
}

/// x[e] = sum_c R[c][e] * pre[c]. info: [0] R (T x hc*E), [1] pre (T x hc), [2] out (T x E), [3] E, [4] hc.
/// Grid (ceil(E / 256), T).
extern "C" __global__ void qx_ds_hc_pre(const u64* info) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, E = U(3), hc = U(4);
    if (e >= E) return;
    const float* r = CF(0) + (u64)t * hc * E;
    const float* pre = CF(1) + (u64)t * hc;
    float v = 0.f;
    for (u32 c = 0; c < hc; ++c) v += r[c * E + e] * pre[c];
    F(2)[(u64)t * E + e] = v;
}

/// R[d][e] = y[e] * post[d] + sum_s R[s][e] * comb[s*hc + d], in place (each thread owns one e for every d).
/// info: [0] y (T x E), [1] R (T x hc*E), [2] post (T x hc), [3] comb (T x hc*hc), [4] E, [5] hc. Grid (ceil(E/256), T).
extern "C" __global__ void qx_ds_hc_post(const u64* info) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, E = U(4), hc = U(5);
    if (e >= E) return;
    float* r = F(1) + (u64)t * hc * E;
    const float y = CF(0)[(u64)t * E + e];
    const float* post = CF(2) + (u64)t * hc;
    const float* comb = CF(3) + (u64)t * hc * hc;
    float in[4];
    for (u32 s = 0; s < hc; ++s) in[s] = r[s * E + e];
    for (u32 d = 0; d < hc; ++d) {
        float v = y * post[d];
        for (u32 s = 0; s < hc; ++s) v += in[s] * comb[s * hc + d];
        r[d * E + e] = v;
    }
}

/// hc_init for one token: R[c][e] = e[e] for every stream. info: [0] R (T x hc*E), [1] x (T x E), [2] E, [3] hc.
/// Grid (ceil(hc*E / 256), T).
extern "C" __global__ void qx_ds_hc_init(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, E = U(2), hc = U(3);
    if (i >= hc * E) return;
    F(0)[(u64)t * hc * E + i] = CF(1)[(u64)t * E + i % E];
}

// ---- RoPE (ggml NORM mode: consecutive pairs) with YaRN, as the reference's `Rope::apply` -------------------

/// Rotates dims [off, off + n_rot) of `heads` vectors of `dim` per token, in place. Pair i of the rotary dims turns
/// by theta_i: theta_extrap = pos * base^(-2i/n_rot) (the reference multiplies theta_scale up iteratively; so does
/// this), theta_interp = freq_scale * theta_extrap, mixed by the YaRN ramp between corr[0] and corr[1] when ext
/// is set. `inverse` turns the other way (the output un-RoPE).
/// info: [0] x, [1] heads, [2] dim, [3] n_rot, [4] off, [5] positions (u32 x T), [6] position delta (i32, added),
/// [7] base bits, [8] freq_scale bits, [9] corr0 bits, [10] corr1 bits, [11] ext (0/1), [12] inverse (0/1),
/// [13] x stride per token (floats), [14] head stride (floats). Grid (heads, T), n_rot / 2 threads.
extern "C" __global__ void qx_ds_rope(const u64* info) {
    const u32 h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, nr = U(3);
    if (2 * i >= nr) return;
    float* x = F(0) + (u64)t * info[13] + (u64)h * info[14] + U(4);
    const i32 pos_i = (i32)reinterpret_cast<const u32*>(info[5])[t] + I(6);
    const float pos = (float)pos_i;
    const float base = __uint_as_float(U(7)), freq_scale = __uint_as_float(U(8));
    const float theta_scale = powf(base, -2.f / (float)nr);
    float theta_base = pos;
    for (u32 j = 0; j < i; ++j) theta_base *= theta_scale;
    const float theta_extrap = theta_base;
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (U(11)) {
        const float lo = __uint_as_float(U(9)), hi = __uint_as_float(U(10));
        const float y = ((float)i - lo) / fmaxf(hi - lo, 0.001f);
        const float ramp = 1.f - fminf(fmaxf(y, 0.f), 1.f);
        theta = theta_interp * (1.f - ramp) + theta_extrap * ramp;
    }
    float s, c;
    sincosf(theta, &s, &c);
    if (U(12)) s = -s;
    const float x0 = x[2 * i], x1 = x[2 * i + 1];
    x[2 * i] = x0 * c - x1 * s;
    x[2 * i + 1] = x0 * s + x1 * c;
}

// ---- elementwise -----------------------------------------------------------------------------------------

/// out[i] = silu(min(a, c)) * clamp(b, -c, c), the pool's arithmetic (qx_silu_mul). info: [0] a, [1] b, [2] out,
/// [3] n, [4] clamp bits (0: plain silu(a) * b).
extern "C" __global__ void qx_ds_silu_mul_clamp(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= U(3)) return;
    const float c = __uint_as_float(U(4));
    const float a = CF(0)[i], b = CF(1)[i];
    F(2)[i] = c > 0.f ? qx_silu_mul(fminf(a, c), fminf(fmaxf(b, -c), c)) : qx_silu_mul(a, b);
}

// ---- Engram ----------------------------------------------------------------------------------------------

/// The Engram gate per stream and its application: dot_c = sum_i rms(R_c)_i * rms(key_c)_i * g_c[i] / sqrt(E),
/// gate_c = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), R_c += value * gate_c. The keys and the value come
/// from `engram_kv`'s output: [hc keys of E | value of E] per token. info: [0] R (T x hc*E), [1] kv (T x (hc+1)*E),
/// [2] gate weights (hc x E), [3] E, [4] eps bits, [5] gates out (T x hc; 0: none). Grid (hc, T), 256 threads.
extern "C" __global__ void qx_ds_engram_gate(const u64* info) {
    __shared__ float sc[32];
    const u32 c = blockIdx.x, t = blockIdx.y, E = U(3), hc = gridDim.x;
    float* r = F(0) + (u64)t * hc * E + (u64)c * E;
    const float* key = CF(1) + (u64)t * (hc + 1) * E + (u64)c * E;
    const float* value = CF(1) + (u64)t * (hc + 1) * E + (u64)hc * E;
    const float* g = CF(2) + (u64)c * E;
    const float eps = __uint_as_float(U(4));
    float sx = 0.f, sk = 0.f;
    for (u32 i = threadIdx.x; i < E; i += blockDim.x) { sx += r[i] * r[i]; sk += key[i] * key[i]; }
    sx = block_sum(sx, sc);
    __syncthreads();
    sk = block_sum(sk, sc);
    const float rx = 1.f / sqrtf(sx / (float)E + eps), rk = 1.f / sqrtf(sk / (float)E + eps);
    float dot = 0.f;
    for (u32 i = threadIdx.x; i < E; i += blockDim.x) dot += (r[i] * rx) * (key[i] * rk) * g[i];
    __syncthreads();
    dot = block_sum(dot, sc) / sqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gate = sigmoid_rn(dot < 0.f ? -mag : mag);
    if (threadIdx.x == 0 && info[5]) F(5)[(u64)t * hc + c] = gate;
    for (u32 i = threadIdx.x; i < E; i += blockDim.x) r[i] += value[i] * gate;
}

// ---- MoE routing and combine -----------------------------------------------------------------------------

/// DeepSeek routing: probs = sqrt(softplus(logits)); the top k by probs + bias (descending, ties to the lower id);
/// weights = probs / (sum of the k + 1e-20) * scale. Lane l holds experts l, l+32, ... One block of 32 threads per
/// token. info: [0] logits (T x n), [1] n (<= 1024), [2] k (<= 32), [3] bias (n), [4] ids out (i32 T x k),
/// [5] weights out (f32 T x k), [6] scale bits.
extern "C" __global__ void qx_ds_route(const u64* info) {
    const u32 n = U(1), k = U(2), l = threadIdx.x, t = blockIdx.x;
    const float* lg = CF(0) + (u64)t * n;
    const float* bias = CF(3);
    const u32 per = (n + 31) / 32;
    float p[32], s[32];
    for (u32 i = 0; i < per; ++i) {
        const u32 e = i * 32 + l;
        if (e < n) {
            const float x = lg[e];
            p[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
            s[i] = p[i] + bias[e];
        } else {
            p[i] = 0.f;
            s[i] = -3.402823466e38f;
        }
    }
    float mine = 0.f, tot = 0.f;
    i32 myid = 0;
    for (u32 r = 0; r < k; ++r) {
        float bv = -3.402823466e38f;
        i32 bi = 0x7fffffff;
        for (u32 i = 0; i < per; ++i) {
            const i32 e = (i32)(i * 32 + l);
            if (e < (i32)n && (s[i] > bv || (s[i] == bv && e < bi))) { bv = s[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const i32 oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float pv = 0.f;
        // bi stays at the sentinel only for NaN logits: never index with it
        if (bi != 0x7fffffff && (u32)(bi % 32) == l) {
            pv = p[bi / 32];
            s[bi / 32] = -3.402823466e38f;
        }
        pv = __shfl_sync(0xffffffffu, pv, bi != 0x7fffffff ? bi % 32 : 0);
        if (bi == 0x7fffffff) bi = 0;
        tot += pv;
        if (l == r) { mine = pv; myid = bi; }
    }
    tot = __fadd_rn(tot, 1e-20f);
    if (l < k) {
        reinterpret_cast<i32*>(info[4])[(u64)t * k + l] = myid;
        F(5)[(u64)t * k + l] = __fmul_rn(__fdiv_rn(mine, tot), __uint_as_float(U(6)));
    }
}

/// The grouped output projection's rows per token: dst[t][g * L + i] = src[g][t][i] (src rows TM apart per group).
/// info: [0] src, [1] dst, [2] G, [3] L, [4] TM. Grid (ceil(G * L / 256), T).
extern "C" __global__ void qx_ds_regroup(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, G = U(2), L = U(3), TM = U(4);
    if (i >= G * L) return;
    const u32 g = i / L, r = i % L;
    F(1)[(u64)t * G * L + i] = CF(0)[((u64)g * TM + t) * L + r];
}

/// Copies `n` floats from host-mapped memory to the device in stream order. info: [0] src (mapped), [1] dst, [2] n.
extern "C" __global__ void qx_ds_copy_f32(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(2)) F(1)[i] = reinterpret_cast<const volatile float*>(info[0])[i];
}

/// y[r] = sum over the k selections in order of w[a] * v_a[r] (GPU output when slots[a] is set, else the CPU's
/// mapped row), plus the shared expert's output (no gate). info: [0] GPU outputs (selections x E), [1] k, [2] CPU
/// outputs (mapped), [3] y out (T x E), [4] E, [5] slots, [6] weights, [7] shared expert out (T x E).
/// Grid (ceil(E / block), T).
extern "C" __global__ void qx_ds_moe_combine(const u64* info) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, n = U(4);
    if (r >= n) return;
    const u64 tk = blockIdx.y, a0 = tk * U(1);
    const u64* slots = reinterpret_cast<const u64*>(info[5]);
    const volatile float* cpu = reinterpret_cast<const volatile float*>(info[2]);
    float s = 0.f;
    for (u32 k = 0; k < U(1); ++k) {
        const u64 a = a0 + k;
        const float v = slots[a] ? CF(0)[a * n + r] : cpu[a * n + r];
        s = __fadd_rn(s, __fmul_rn(CF(6)[a], v));
    }
    F(3)[tk * n + r] = __fadd_rn(s, CF(7)[tk * n + r]);
}

// ---- the compressor (KV source layers) ---------------------------------------------------------------------

/// Pools a group of `ratio` latents into one compressed latent when the token completes its group, else only
/// records the token's pending (latent, score) pair. The pending pairs live per slot at [ratio][2][dim] (latent,
/// then score); the comp output (before RoPE) goes to `comp_tmp` row t for the index-key projection, and, when
/// the group is complete, RMS-normed into the latent part of the store row (pos + 1) / ratio - 1, which the
/// caller RoPEs afterwards. When the group is not complete, `comp_tmp` row t holds the pooled-so-far value, and
/// `flag[t]` is 0 (1 when complete).
/// info: [0] latent (T x dim, the compressor's kv projection), [1] score (T x dim; 0: ratio 1, scores 0),
/// [2] pending table (u64 per slot), [3] store table (u64 per slot: rows of `row_stride` floats), [4] norm w
/// (dim), [5] comp_tmp out (T x dim), [6] flag out (u32 x T), [7] positions (u32 x T), [8] slots (u32 x T),
/// [9] ratio, [10] dim, [11] row stride (floats), [12] eps bits, [13] T (0: 1). One block of 256 threads runs the
/// window's tokens in order (a group's pending pairs accumulate across them).
extern "C" __global__ void qx_ds_compress(const u64* info) {
    __shared__ float sc[32];
    const u32 ratio = U(9), dim = U(10), T = info[13] ? U(13) : 1;
    // the window's tokens in order within one block: a group's pending pairs accumulate across them
    for (u32 t = 0; t < T; ++t) {
        const u32 pos = reinterpret_cast<const u32*>(info[7])[t], slot = reinterpret_cast<const u32*>(info[8])[t];
        float* pend = reinterpret_cast<float*>(reinterpret_cast<const u64*>(info[2])[slot]);
        const float* lat = CF(0) + (u64)t * dim;
        const u32 at = pos % ratio;
        for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
            pend[(at * 2) * dim + i] = lat[i];
            pend[(at * 2 + 1) * dim + i] = info[1] ? CF(1)[(u64)t * dim + i] : 0.f;
        }
        __syncthreads();
        const bool complete = (pos + 1) % ratio == 0;
        // pooled[i] = sum_j v_j[i] * softmax_j(s_j[i]) over the group's j (the pending ones so far)
        const u32 n = at + 1;
        float ss = 0.f;
        float* tmp = F(5) + (u64)t * dim;
        for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
            float mx = -3.402823466e38f;
            for (u32 j = 0; j < n; ++j) mx = fmaxf(mx, pend[(2 * j + 1) * dim + i]);
            float z = 0.f;
            for (u32 j = 0; j < n; ++j) z += expf(pend[(2 * j + 1) * dim + i] - mx);
            // the reference: softmax first (e / z), then the weighted sum
            float v = 0.f;
            for (u32 j = 0; j < n; ++j) v += pend[2 * j * dim + i] * (expf(pend[(2 * j + 1) * dim + i] - mx) / z);
            tmp[i] = v;
            ss += v * v;
        }
        ss = block_sum(ss, sc);
        const float r = 1.f / sqrtf(ss / (float)dim + __uint_as_float(U(12)));
        const float* w = CF(4);
        for (u32 i = threadIdx.x; i < dim; i += blockDim.x) tmp[i] = tmp[i] * r * w[i];
        if (threadIdx.x == 0) reinterpret_cast<u32*>(info[6])[t] = complete ? 1u : 0u;
        if (complete) {
            float* row = reinterpret_cast<float*>(reinterpret_cast<const u64*>(info[3])[slot]) + (u64)((pos + 1) / ratio - 1) * info[11];
            for (u32 i = threadIdx.x; i < dim; i += blockDim.x) row[i] = tmp[i];
        }
        __syncthreads();
    }
}

/// Commits a complete group's index key: key = rms(ik) * w, RoPE'd by the caller afterwards, written at the store
/// row's key part (offset `koff`). Rows whose flag is 0 are skipped. info: [0] ik (T x kdim, the index-key
/// projection of comp_tmp), [1] store table (u64 per slot), [2] norm w (kdim), [3] flag (u32 x T), [4] positions,
/// [5] slots, [6] ratio, [7] kdim, [8] row stride, [9] koff, [10] eps bits. Grid T, 128 threads.
extern "C" __global__ void qx_ds_idx_key_commit(const u64* info) {
    __shared__ float sc[32];
    const u32 t = blockIdx.x;
    if (reinterpret_cast<const u32*>(info[3])[t] == 0) return;
    const u32 ratio = U(6), kdim = U(7);
    const u32 pos = reinterpret_cast<const u32*>(info[4])[t], slot = reinterpret_cast<const u32*>(info[5])[t];
    const float* ik = CF(0) + (u64)t * kdim;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < kdim; i += blockDim.x) ss += ik[i] * ik[i];
    ss = block_sum(ss, sc);
    const float r = 1.f / sqrtf(ss / (float)kdim + __uint_as_float(U(10)));
    float* row = reinterpret_cast<float*>(reinterpret_cast<const u64*>(info[1])[slot]) + (u64)((pos + 1) / ratio - 1) * info[8] + U(9);
    for (u32 i = threadIdx.x; i < kdim; i += blockDim.x) row[i] = ik[i] * r * CF(2)[i];
}

/// RoPE of a store row's parts for complete groups: like qx_ds_rope on the row at (pos + 1) / ratio - 1 of the
/// token's slot, at the group's first position pos + 1 - ratio, dims [off, off + n_rot) of one vector. Rows whose
/// flag is 0 are skipped. info: [0] store table, [1] flag, [2] positions, [3] slots, [4] ratio, [5] row stride,
/// [6] off, [7] n_rot, [8] base bits, [9] freq_scale bits, [10] corr0, [11] corr1, [12] ext. Grid T, n_rot/2 threads.
extern "C" __global__ void qx_ds_rope_row(const u64* info) {
    const u32 t = blockIdx.x, i = threadIdx.x, nr = U(7), ratio = U(4);
    if (2 * i >= nr || reinterpret_cast<const u32*>(info[1])[t] == 0) return;
    const u32 pos = reinterpret_cast<const u32*>(info[2])[t], slot = reinterpret_cast<const u32*>(info[3])[t];
    float* x = reinterpret_cast<float*>(reinterpret_cast<const u64*>(info[0])[slot]) + (u64)((pos + 1) / ratio - 1) * info[5] + U(6);
    const float p = (float)(pos + 1 - ratio);
    const float base = __uint_as_float(U(8)), freq_scale = __uint_as_float(U(9));
    const float theta_scale = powf(base, -2.f / (float)nr);
    float theta_base = p;
    for (u32 j = 0; j < i; ++j) theta_base *= theta_scale;
    float theta = freq_scale * theta_base;
    if (U(12)) {
        const float lo = __uint_as_float(U(10)), hi = __uint_as_float(U(11));
        const float y = ((float)i - lo) / fmaxf(hi - lo, 0.001f);
        const float ramp = 1.f - fminf(fmaxf(y, 0.f), 1.f);
        theta = theta * (1.f - ramp) + theta_base * ramp;
    }
    float s, c;
    sincosf(theta, &s, &c);
    const float x0 = x[2 * i], x1 = x[2 * i + 1];
    x[2 * i] = x0 * c - x1 * s;
    x[2 * i + 1] = x0 * s + x1 * c;
}

// ---- the lightning indexer ---------------------------------------------------------------------------------

/// score[r] = sum_h relu(iq_h . key_r) * iw_h for the visible rows r < (pos + 1) / ratio of the token's slot;
/// rows past that are left alone. One warp per row: lane h takes head h (heads = 32). info: [0] iq (T x 32 x kdim),
/// [1] iw (T x 32), [2] store table, [3] positions, [4] slots, [5] ratio, [6] kdim, [7] row stride, [8] koff,
/// [9] scores out (T x max_rows), [10] max_rows, [11] iw scale bits (1 / sqrt(heads * kdim)). Grid (ceil(max_rows / 8), T),
/// 256 threads.
extern "C" __global__ void qx_ds_idx_score(const u64* info) {
    const u32 t = blockIdx.y, warp = threadIdx.x / 32, h = threadIdx.x % 32, kdim = U(6), ratio = U(5);
    const u32 r = blockIdx.x * 8 + warp;
    const u32 pos = reinterpret_cast<const u32*>(info[3])[t], slot = reinterpret_cast<const u32*>(info[4])[t];
    const u32 nv = (pos + 1) / ratio;
    if (r >= nv) return;
    const float* key = reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[2])[slot]) + (u64)r * info[7] + U(8);
    const float* q = CF(0) + (u64)t * 32 * kdim + (u64)h * kdim;
    float d = 0.f;
    for (u32 i = 0; i < kdim; ++i) d += q[i] * key[i];
    d = fmaxf(d, 0.f) * (CF(1)[(u64)t * 32 + h] * __uint_as_float(U(11)));
    d = warp_sum(d);
    if (h == 0) F(9)[(u64)t * info[10] + r] = d;
}

/// `qx_ds_idx_score` with the same arithmetic, arranged for the memory system (2026-11-06): the token's 32 query
/// rows go to shared memory once per block, padded to 129 floats so lane h reading q[h][i] hits a distinct bank, and
/// the key is read as float4 broadcasts (all lanes the same address). Each (row, head) product is still the serial
/// 128-step chain d += q[i] * key[i] in order, and the head sum the same warp tree, so the scores are bit for bit
/// the old kernel's. Same info words and grid; 256 threads; 16.5 KB of shared memory.
extern "C" __global__ void qx_ds_idx_score2(const u64* info) {
    __shared__ float sq[32 * 129];
    const u32 t = blockIdx.y, warp = threadIdx.x / 32, h = threadIdx.x % 32, kdim = U(6), ratio = U(5);
    const u32 pos = reinterpret_cast<const u32*>(info[3])[t], slot = reinterpret_cast<const u32*>(info[4])[t];
    const u32 nv = (pos + 1) / ratio;
    if (blockIdx.x * 8 >= nv) return;  // whole block past the visible rows: nothing to load either
    const float* qt = CF(0) + (u64)t * 32 * kdim;
    for (u32 i = threadIdx.x; i < 32 * kdim; i += blockDim.x) sq[(i / kdim) * 129 + (i % kdim)] = qt[i];
    __syncthreads();
    const u32 r = blockIdx.x * 8 + warp;
    if (r >= nv) return;
    const float* key = reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[2])[slot]) + (u64)r * info[7] + U(8);
    const float* q = sq + h * 129;
    float d = 0.f;
    const u32 k4 = kdim / 4;
    for (u32 i = 0; i < k4; ++i) {
        const float4 kv = reinterpret_cast<const float4*>(key)[i];
        d += q[4 * i] * kv.x;
        d += q[4 * i + 1] * kv.y;
        d += q[4 * i + 2] * kv.z;
        d += q[4 * i + 3] * kv.w;
    }
    for (u32 i = k4 * 4; i < kdim; ++i) d += q[i] * key[i];
    d = fmaxf(d, 0.f) * (CF(1)[(u64)t * 32 + h] * __uint_as_float(U(11)));
    d = warp_sum(d);
    if (h == 0) F(9)[(u64)t * info[10] + r] = d;
}

/// Block maxima for the candidate selection: bmax[b] = max of scores[8b .. 8b+8) over the visible rows, plus 1e30 on
/// the block of the newest visible row. info: [0] scores (T x max_rows), [1] bmax out (T x max_blocks), [2]
/// positions, [3] ratio, [4] block size, [5] max_rows, [6] max_blocks. Grid (ceil(max_blocks / 256), T).
extern "C" __global__ void qx_ds_block_max(const u64* info) {
    const u32 b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, bs = U(4);
    const u32 pos = reinterpret_cast<const u32*>(info[2])[t];
    const u32 nv = (pos + 1) / U(3), nb = (nv + bs - 1) / bs;
    if (b >= nb || b >= U(6)) return;
    const float* s = CF(0) + (u64)t * info[5];
    float m = -3.402823466e38f;
    for (u32 r = b * bs; r < min(nv, (b + 1) * bs); ++r) m = fmaxf(m, s[r]);
    if (b == (nv - 1) / bs) m += 1e30f;
    F(1)[(u64)t * info[6] + b] = m;
}

/// Masks scores outside the candidate blocks: a bit per block in `inside` (set by qx_ds_mark_blocks), scores of
/// rows in other blocks become -inf. info: [0] scores (T x max_rows), [1] inside (u32 x T x words), [2] positions,
/// [3] ratio, [4] block size, [5] max_rows, [6] words per token. Grid (ceil(max_rows / 256), T).
extern "C" __global__ void qx_ds_cand_mask(const u64* info) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, bs = U(4);
    const u32 pos = reinterpret_cast<const u32*>(info[2])[t];
    const u32 nv = (pos + 1) / U(3);
    if (r >= nv) return;
    const u32 b = r / bs;
    const u32 bit = reinterpret_cast<const u32*>(info[1])[(u64)t * info[6] + b / 32] >> (b % 32) & 1u;
    if (!bit) F(0)[(u64)t * info[5] + r] = -3.402823466e38f;
}

/// Sets the bit of every selected block: inside[t][b] = 1 for b in sel[t][0..count[t]). The words are zeroed first by
/// the same launch (grid-stride over words, then a grid sync is avoided by zeroing in a separate launch: see
/// qx_ds_zero_u32). info: [0] sel (i32 x T x k), [1] count (u32 x T), [2] inside out, [3] k, [4] words per token.
/// Grid (ceil(k / 256), T).
extern "C" __global__ void qx_ds_mark_blocks(const u64* info) {
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= reinterpret_cast<const u32*>(info[1])[t]) return;
    const i32 b = reinterpret_cast<const i32*>(info[0])[(u64)t * U(3) + j];
    atomicOr(reinterpret_cast<u32*>(info[2]) + (u64)t * info[4] + b / 32, 1u << (b % 32));
}

/// Zeroes n u32 words. info: [0] p, [1] n.
extern "C" __global__ void qx_ds_zero_u32(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(1)) reinterpret_cast<u32*>(info[0])[i] = 0;
}

/// The top k of n scores per token: the k largest, ties to the lower index, as `top_k` in the reference; n may
/// be less than k (then all of them). Radix select over the monotone key of the float, four 8-bit passes with a
/// 256-bin histogram, then the indices above the threshold and the first (lowest-index) ones at it. The count
/// `n` comes from the token's visible rows: (pos + 1) / ratio, capped at max_rows. info: [0] scores (T x max_rows),
/// [1] k, [2] sel out (i32 x T x k), [3] count out (u32 x T), [4] positions, [5] ratio, [6] max_rows.
/// One block of 1024 threads per token.
extern "C" __global__ void qx_ds_topk(const u64* info) {
    __shared__ u32 hist[256];
    __shared__ u32 s_prefix, s_need, s_cnt, s_eq;
    __shared__ u32 wcount[64];  // per warp: the "above" count, then at [32 + warp] the "equal" count
    const u32 k = U(1), tid = threadIdx.x, nt = blockDim.x, t = blockIdx.x;
    const u32 pos = reinterpret_cast<const u32*>(info[4])[t];
    const u32 n = min((pos + 1) / U(5), U(6));
    const float* x = CF(0) + (u64)t * info[6];
    i32* sel = reinterpret_cast<i32*>(info[2]) + (u64)t * k;
    if (n <= k) {
        for (u32 i = tid; i < n; i += nt) sel[i] = (i32)i;
        if (tid == 0) reinterpret_cast<u32*>(info[3])[t] = n;
        return;
    }
    auto key = [&](u32 i) { const u32 u = __float_as_uint(x[i]); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
    u32 prefix = 0, need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (u32 i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const u32 hmask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (u32 i = tid; i < n; i += nt) {
            const u32 kk = key(i);
            if ((kk & hmask) == prefix) atomicAdd(&hist[(kk >> shift) & 0xff], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            // walk bins from the top until `need` entries are covered
            u32 acc = 0, b = 255;
            for (;; --b) {
                if (acc + hist[b] >= need) break;
                acc += hist[b];
                if (b == 0) break;
            }
            s_prefix = prefix | (b << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
    }
    // prefix is the k-th key: take every key above it, and the first `need` at it in index order
    if (tid == 0) { s_cnt = 0; s_eq = 0; }
    __syncthreads();
    const u32 nwarps = nt / 32, lane = tid % 32, warp = tid / 32;
    for (u32 base = 0; base < n; base += nt) {
        const u32 i = base + tid;
        const bool above = i < n && key(i) > prefix;
        const bool equal = i < n && key(i) == prefix;
        // block-wide ordered prefix counts via warp ballots
        const u32 ba = __ballot_sync(0xffffffffu, above), be = __ballot_sync(0xffffffffu, equal);
        if (lane == 0) { wcount[warp] = __popc(ba); wcount[32 + warp] = __popc(be); }
        __syncthreads();
        u32 pa = s_cnt, pe = s_eq;
        for (u32 w = 0; w < warp; ++w) { pa += wcount[w]; pe += wcount[32 + w]; }
        pa += __popc(ba & ((1u << lane) - 1));
        pe += __popc(be & ((1u << lane) - 1));
        if (above) sel[pa] = (i32)i;
        if (equal && pe < need) sel[k - need + pe] = (i32)i;  // the equal ones fill the tail of the k
        __syncthreads();
        if (tid == 0) {
            for (u32 w = 0; w < nwarps; ++w) { s_cnt += wcount[w]; s_eq += wcount[32 + w]; }
        }
        __syncthreads();
    }
    if (tid == 0) reinterpret_cast<u32*>(info[3])[t] = k;
}

// ---- attention over [window | selected compressed latents], K = V, per-head sinks ---------------------------

/// One block per (head, token). Keys: the window cells max(0, pos + 1 - n_swa) .. pos from the slot's latent
/// ring (cell c at ring slot c % W, rows of `dim`), then the selected rows sel[t][0..count[t]) of the slot's
/// compressed store (latent part). Scores s_j = q . k_j * scale; softmax with the head's sink in the denominator;
/// out = sum_j p_j k_j. Rounding as the reference: p_j = exp(s_j - m), z = sum p + exp(sink - m), out += p_j / z * k.
/// info: [0] q (T x heads x dim), [1] ring table (u64 per slot), [2] W, [3] positions, [4] slots, [5] sel (i32 x T x
/// ksel; 0: none), [6] count (u32 x T; 0: none), [7] store table, [8] row stride, [9] sinks (heads), [10] out
/// (T x heads x dim), [11] scale bits, [12] n_swa, [13] ksel, [14] dim. Grid (heads, T), 256 threads; dim <= 512,
/// n_swa + ksel <= 1024.
extern "C" __global__ void qx_ds_attn(const u64* info) {
    __shared__ float sc[1024];
    __shared__ float red[32];
    __shared__ float qs[512];
    const u32 h = blockIdx.x, t = blockIdx.y, dim = U(14), W = U(2), n_swa = U(12), ksel = U(13);
    const u32 pos = reinterpret_cast<const u32*>(info[3])[t], slot = reinterpret_cast<const u32*>(info[4])[t];
    const float* q = CF(0) + ((u64)t * gridDim.x + h) * dim;
    for (u32 i = threadIdx.x; i < dim; i += blockDim.x) qs[i] = q[i];
    const float* ring = reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[1])[slot]);
    const u32 first = pos + 1 > n_swa ? pos + 1 - n_swa : 0;
    const u32 nw = pos + 1 - first;
    const u32 ns = info[6] ? reinterpret_cast<const u32*>(info[6])[t] : 0;
    const i32* sel = info[5] ? reinterpret_cast<const i32*>(info[5]) + (u64)t * ksel : nullptr;
    const float* store = ns ? reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[7])[slot]) : nullptr;
    const float scale = __uint_as_float(U(11));
    auto key_row = [&](u32 j) -> const float* {
        return j < nw ? ring + (u64)((first + j) % W) * dim : store + (u64)sel[j - nw] * info[8];
    };
    __syncthreads();
    const u32 nk = nw + ns;
    // scores: one warp per key, the lanes over the dimensions (coalesced rows)
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarp = blockDim.x / 32;
    float m = CF(9)[h];
    for (u32 j = warp; j < nk; j += nwarp) {
        const float* kr = key_row(j);
        float d = 0.f;
        for (u32 i = lane; i < dim; i += 32) d += qs[i] * kr[i];
        d = warp_sum(d) * scale;
        if (lane == 0) sc[j] = d;
        m = fmaxf(m, d);
    }
    m = block_max(m, red);
    __syncthreads();
    float z = 0.f;
    for (u32 j = threadIdx.x; j < nk; j += blockDim.x) {
        sc[j] = expf(sc[j] - m);
        z += sc[j];
    }
    z = block_sum(z, red) + expf(CF(9)[h] - m);
    __syncthreads();
    for (u32 j = threadIdx.x; j < nk; j += blockDim.x) sc[j] /= z;
    __syncthreads();
    float* o = F(10) + ((u64)t * gridDim.x + h) * dim;
    for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.f;
        for (u32 j = 0; j < nk; ++j) acc += sc[j] * key_row(j)[i];
        o[i] = acc;
    }
}

/// Appends each token's latent to its slot's ring at cell pos % W. info: [0] kv (T x dim), [1] ring table, [2] W,
/// [3] positions, [4] slots, [5] dim. Grid T, 256 threads.
/// MLA decode attention split over key chunks (the 64 heads share one latent row as key and value, so a key chunk
/// is loaded once for a group of heads). Block (hg, c, t): heads [8 hg, 8 hg + 8) of token t against keys
/// [16 c, 16 c + 16) of its key list (the window's ring rows, then the selected compressed rows), writing the
/// chunk's (max, sum, o[dim]) per head to `part` at ((t * H + h) * C + c) * (dim + 2); empty chunks write
/// (-inf, 0, 0). info: [0..14] as qx_ds_attn, [15] partials out, [16] C (chunks per token). 256 threads.
extern "C" __global__ void qx_ds_attn_part(const u64* info) {
    constexpr u32 KC = 12;  // keys per chunk (8 heads x 512 q + KC x 513 keys fit the 48 KB of static shared)
    constexpr u32 KS = 513;  // key row stride in shared memory: consecutive rows on consecutive banks
    __shared__ float ks[KC * KS];
    __shared__ float qs[8 * 512];
    __shared__ float sc[8][KC];
    __shared__ float red[8][2];
    const u32 hg = blockIdx.x, c = blockIdx.y, t = blockIdx.z, dim = U(14), W = U(2), n_swa = U(12), ksel = U(13);
    const u32 H = gridDim.x * 8, C = U(16), tid = threadIdx.x;
    const u32 pos = reinterpret_cast<const u32*>(info[3])[t], slot = reinterpret_cast<const u32*>(info[4])[t];
    const float* ring = reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[1])[slot]);
    const u32 first = pos + 1 > n_swa ? pos + 1 - n_swa : 0;
    const u32 nw = pos + 1 - first;
    const u32 ns = info[6] ? reinterpret_cast<const u32*>(info[6])[t] : 0;
    const i32* sel = info[5] ? reinterpret_cast<const i32*>(info[5]) + (u64)t * ksel : nullptr;
    const float* store = ns ? reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[7])[slot]) : nullptr;
    const float scale = __uint_as_float(U(11));
    const u32 nk = nw + ns, j0 = c * KC;
    float* part = F(15) + (((u64)t * H + hg * 8) * C + c) * (dim + 2);
    const u64 pstride = (u64)C * (dim + 2);  // from one head's partial to the next
    if (j0 >= nk) {
        for (u32 h = tid / 32; h < 8; h += blockDim.x / 32) {
            float* p = part + h * pstride;
            if (tid % 32 == 0) { p[0] = QX_NINF; p[1] = 0.f; }
            for (u32 i = tid % 32; i < dim; i += 32) p[2 + i] = 0.f;
        }
        return;
    }
    // the group's queries and the chunk's key rows
    const float* q = CF(0) + ((u64)t * H + hg * 8) * dim;
    for (u32 i = tid; i < 8 * dim; i += blockDim.x) qs[i] = q[i];
    for (u32 j = 0; j < KC; ++j) {
        const u32 jj = j0 + j;
        if (jj < nk) {
            const float* kr = jj < nw ? ring + (u64)((first + jj) % W) * dim : store + (u64)sel[jj - nw] * info[8];
            for (u32 i = tid; i < dim; i += blockDim.x) ks[j * KS + i] = kr[i];
        } else {
            // zero, not garbage: the output loop multiplies every row by its (zero) probability
            for (u32 i = tid; i < dim; i += blockDim.x) ks[j * KS + i] = 0.f;
        }
    }
    __syncthreads();
    // scores: pair p = (head, key), two threads per pair over half the dimensions each
    {
        const u32 p = tid / 2, half = tid & 1, h = p / KC, j = p % KC;
        float d = 0.f;
        if (p < 8 * KC && j0 + j < nk) {
            const float* qq = qs + h * dim + half * (dim / 2);
            const float* kk = ks + j * KS + half * (dim / 2);
            for (u32 i = 0; i < dim / 2; ++i) d += qq[i] * kk[i];
        }
        d += __shfl_xor_sync(0xffffffffu, d, 1);
        if (half == 0 && p < 8 * KC) sc[h][j] = j0 + j < nk ? d * scale : QX_NINF;
    }
    __syncthreads();
    // per head: chunk max and sum, probabilities in place (warp h, lanes 0..15)
    {
        const u32 h = tid / 32, lane = tid % 32;
        float s = lane < KC ? sc[h][lane] : QX_NINF;
        float m = s;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float p = lane < KC && s != QX_NINF ? expf(s - m) : 0.f;
        float l = p;
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane < KC) sc[h][lane] = p;
        if (lane == 0) { red[h][0] = m; red[h][1] = l; }
    }
    __syncthreads();
    // partial outputs: thread (head h = tid / 32, dims lane + 32 r)
    {
        const u32 h = tid / 32, lane = tid % 32;
        float* p = part + h * pstride;
        if (lane == 0) { p[0] = red[h][0]; p[1] = red[h][1]; }
        for (u32 i = lane; i < dim; i += 32) {
            float acc = 0.f;
            for (u32 j = 0; j < KC; ++j) acc += sc[h][j] * ks[j * KS + i];
            p[2 + i] = acc;
        }
    }
}

/// Merges the chunk partials of qx_ds_attn_part with the head's sink: M = max(m_c, sink), L = sum l_c e^(m_c - M)
/// + e^(sink - M), o = sum o_c e^(m_c - M) / L. info: [0] partials, [1] sinks (H), [2] out (T x H x dim), [3] C,
/// [4] dim. Grid (H, T), 256 threads.
extern "C" __global__ void qx_ds_attn_merge(const u64* info) {
    __shared__ float w[64];
    __shared__ float s_m, s_l;
    const u32 h = blockIdx.x, t = blockIdx.y, H = gridDim.x, C = U(3), dim = U(4), tid = threadIdx.x;
    const float* part = CF(0) + ((u64)t * H + h) * C * (dim + 2);
    const float sink = CF(1)[h];
    if (tid == 0) {
        float M = sink;
        for (u32 c = 0; c < C; ++c) M = fmaxf(M, part[c * (dim + 2)]);
        float L = expf(sink - M);
        for (u32 c = 0; c < C; ++c) {
            const float m = part[c * (dim + 2)];
            w[c] = m == QX_NINF ? 0.f : expf(m - M);
            L += part[c * (dim + 2) + 1] * w[c];
        }
        s_m = M;
        s_l = L;
    }
    __syncthreads();
    float* o = F(2) + ((u64)t * H + h) * dim;
    for (u32 i = tid; i < dim; i += blockDim.x) {
        float acc = 0.f;
        for (u32 c = 0; c < C; ++c) acc += w[c] * part[c * (dim + 2) + 2 + i];
        o[i] = acc / s_l;
    }
}

extern "C" __global__ void qx_ds_kv_append(const u64* info) {
    const u32 t = blockIdx.x, dim = U(5), W = U(2);
    const u32 pos = reinterpret_cast<const u32*>(info[3])[t], slot = reinterpret_cast<const u32*>(info[4])[t];
    float* ring = reinterpret_cast<float*>(reinterpret_cast<const u64*>(info[1])[slot]) + (u64)(pos % W) * dim;
    const float* kv = CF(0) + (u64)t * dim;
    for (u32 i = threadIdx.x; i < dim; i += blockDim.x) ring[i] = kv[i];
}
