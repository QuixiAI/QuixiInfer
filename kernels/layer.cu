// The per-layer operations of the forward pass other than matrix products, for one token.
// Every kernel takes `const u64* info`; the layout is documented on each entry point.
#include "qx_block.cuh"
#include "qx_sys.cuh"

#define F(i) reinterpret_cast<float*>(info[i])
#define SLOTS(i) reinterpret_cast<const u32*>(info[i])

/// Token t's segment (the run of consecutive tokens of one slot, see step.cu): whether t starts one, and its length.
QX_DEV bool seg_start(const u32* slots, u32 t, u32 T, u32& len) {
    if (t > 0 && slots[t - 1] == slots[t]) return false;
    len = 1;
    while (t + len < T && slots[t + len] == slots[t]) ++len;
    return true;
}
#define CF(i) reinterpret_cast<const float*>(info[i])
#define U(i) ((u32)info[i])

/// RMS norm of `groups` groups of `dim`: out = x / sqrt(mean(x^2) + eps) * w.
/// info: [0] x, [1] w, [2] out, [3] dim, [4] w stride per group (0: one weight row shared), [5] eps bits,
/// [6] x stride per group (0: dim), [7] x stride per token, [8] out stride per token. Grid (groups, tokens).
extern "C" __global__ void qx_rms_groups(const u64* info) {
    __shared__ float sc[32];
    const u32 g = blockIdx.x, t = blockIdx.y, dim = U(3), xs = U(6) ? U(6) : dim;
    const float* x = CF(0) + (u64)t * info[7] + (u64)g * xs;
    const float* w = CF(1) + (u64)g * U(4);
    float* o = F(2) + (u64)t * info[8] + (u64)g * dim;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < dim; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)dim + __uint_as_float(U(5)));
    for (u32 i = threadIdx.x; i < dim; i += blockDim.x) o[i] = x[i] * r * w[i];
}

/// x[i] = silu(x[i] * s). info: [0] x, [1] n, [2] s bits.
extern "C" __global__ void qx_scale_silu(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(1)) F(0)[i] = siluf_(F(0)[i] * __uint_as_float(U(2)));
}

/// HC mix output: mixed[e] = mean_s xn[s][e] * sigmoid(gate[s][e]). info: [0] xn, [1] gate, [2] mixed, [3] E, [4] HC.
/// Grid (ceil(E / block), tokens); token t's rows are HC*E (xn, gate) and E (mixed) apart.
extern "C" __global__ void qx_hc_mix_out(const u64* info) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x, n = U(3), hc = U(4);
    if (e >= n) return;
    const u64 t = blockIdx.y;
    const float* xn = CF(0) + t * hc * n;
    const float* gt = CF(1) + t * hc * n;
    float s = 0.f;
    for (u32 k = 0; k < hc; ++k) s += xn[k * n + e] * sigmoidf_(gt[k * n + e]);
    F(2)[t * n + e] = s / (float)hc;
}

/// HC combine: R[s][e] += y[e] * 2 * sigmoid(inject[s] / HC). info: [0] R, [1] y, [2] inject, [3] E, [4] HC.
extern "C" __global__ void qx_hc_combine(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, n = U(3), hc = U(4);
    if (i >= n * hc) return;
    const u32 s = i / n, e = i % n;
    F(0)[i] += CF(1)[e] * 2.f * sigmoidf_(CF(2)[s] / (float)hc);
}

/// R[s][e] = x[e] for every stream. info: [0] R, [1] x, [2] E, [3] HC. Grid (ceil(HC*E / block), tokens).
extern "C" __global__ void qx_hc_init(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, n = U(2);
    const u64 t = blockIdx.y;
    if (i < n * U(3)) F(0)[t * n * U(3) + i] = CF(1)[t * n + i % n];
}

/// GDN causal conv, kernel 4: c[ch] = silu(w[ch][3] * qkv[ch] + sum_k<3 w[ch][k] * state[k][ch]), then shifts
/// qkv into the state. info: [0] qkv, [1] state (3 x ch, oldest first), [2] w (ch x 4), [3] out, [4] ch.
extern "C" __global__ void qx_gdn_conv(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, ch = U(4);
    if (i >= ch) return;
    const float* w = CF(2) + 4 * i;
    float* st = F(1);
    const float x = CF(0)[i];
    const float s0 = st[i], s1 = st[ch + i], s2 = st[2 * ch + i];
    F(3)[i] = siluf_(w[3] * x + w[0] * s0 + w[1] * s1 + w[2] * s2);
    st[i] = s1;
    st[ch + i] = s2;
    st[2 * ch + i] = x;
}

/// L2-normalise `groups` vectors of `dim` in place: x / sqrt(sum(x^2) + eps). info: [0] x, [1] dim, [2] eps bits.
extern "C" __global__ void qx_l2_groups(const u64* info) {
    __shared__ float sc[32];
    float* x = F(0) + (u64)blockIdx.x * U(1);
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < U(1); i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss + __uint_as_float(U(2)));
    for (u32 i = threadIdx.x; i < U(1); i += blockDim.x) x[i] *= r;
}

/// Gated delta rule for a window of T tokens in order. Block (head h, column chunk c) owns columns
/// [32c, 32c+32) of S; its 128 threads are 32 columns x 4 row chunks of 32, each holding 32 entries of S in
/// registers across the window. Row-chunk partial sums meet in shared memory. S is stored [head][i][j]
/// (i: key dim, j: value dim). Every token's arithmetic is the one-token sequence, so a window gives each token
/// the bits a token at a time would.
/// info: [0] conv out (T x (q | k | v)), [1] S, [2] out o (T x heads x dim), [3] beta raw (T x vh), [4] alpha
/// raw (T x vh), [5] dt bias, [6] a, [7] key heads, [8] dim (128), [9] T (0: 1), [10] conv channels (row
/// stride of [0]), [11] value heads, [12] S copy before the window (0: none), [13] the state to start from instead of
/// S (0: S), for a replay after a rollback, [14] slots (u32 x T), [15] S floats per slot. Grid (value heads, 4, T):
/// the block of a segment's first token runs the segment on its slot's S. 128 threads.
extern "C" __global__ void __launch_bounds__(128) qx_gdn_recur(const u64* info) {
    __shared__ float qs[128], ks[128];
    __shared__ float part[4][32];
    __shared__ float dl[32];
    const u32 h = blockIdx.x, jc = blockIdx.y, d = U(8), kh = U(7);
    const u32 T = U(9) ? U(9) : 1, ch = U(10), vh = U(11), t0 = blockIdx.z;
    u32 len;
    if (!seg_start(SLOTS(14), t0, T, len)) return;
    const u32 jj = threadIdx.x % 32, ic = threadIdx.x / 32, j = jc * 32 + jj;
    const u32 khh = h % kh;
    float* S = F(1) + (u64)SLOTS(14)[t0] * info[15] + (u64)h * d * d;
    const float* S_in = info[13] ? CF(13) + (u64)h * d * d : S;
    float col[32];
#pragma unroll
    for (u32 t2 = 0; t2 < 32; ++t2) col[t2] = S_in[(ic * 32 + t2) * d + j];
    if (info[12]) {
        float* S0 = F(12) + (u64)h * d * d;
#pragma unroll
        for (u32 t2 = 0; t2 < 32; ++t2) S0[(ic * 32 + t2) * d + j] = col[t2];
    }
    for (u32 tk = t0; tk < t0 + len; ++tk) {
        const float* cv = CF(0) + (u64)tk * ch;
        if (tk > t0) __syncthreads();
        qs[threadIdx.x] = cv[khh * d + threadIdx.x];
        ks[threadIdx.x] = cv[(u64)kh * d + khh * d + threadIdx.x];
        __syncthreads();
        const float beta = sigmoidf_(CF(3)[(u64)tk * vh + h]);
        const float decay = __expf(softplusf_(CF(4)[(u64)tk * vh + h] + CF(5)[h]) * CF(6)[h]);
        float sk = 0.f;
#pragma unroll
        for (u32 t2 = 0; t2 < 32; ++t2) {
            const u32 i = ic * 32 + t2;
            col[t2] = col[t2] * decay;
            sk += col[t2] * ks[i];
        }
        part[ic][jj] = sk;
        __syncthreads();
        if (ic == 0) {
            const float tot = part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj];
            const float v = cv[2ull * kh * d + (u64)h * d + j];
            dl[jj] = (v - tot) * beta;
        }
        __syncthreads();
        const float delta = dl[jj];
        float o = 0.f;
#pragma unroll
        for (u32 t2 = 0; t2 < 32; ++t2) {
            const u32 i = ic * 32 + t2;
            col[t2] += ks[i] * delta;
            o += col[t2] * qs[i];
        }
        __syncthreads();
        part[ic][jj] = o;
        __syncthreads();
        if (ic == 0) F(2)[((u64)tk * vh + h) * d + j] = (part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj]) * rsqrtf((float)d);
    }
#pragma unroll
    for (u32 t2 = 0; t2 < 32; ++t2) S[(ic * 32 + t2) * d + j] = col[t2];
}

/// GDN output: on[h][j] = rms(o[h]) * w[j] * sigmoid(z[h][j]). One block per head.
/// info: [0] o, [1] w (dim), [2] z, [3] out, [4] dim, [5] eps bits. Grid (heads, tokens): token t's rows are
/// heads x dim apart.
extern "C" __global__ void qx_gdn_out(const u64* info) {
    __shared__ float sc[32];
    const u32 h = blockIdx.x + blockIdx.y * gridDim.x, d = U(4);
    const float* o = CF(0) + (u64)h * d;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) ss += o[i] * o[i];
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)d + __uint_as_float(U(5)));
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) F(3)[(u64)h * d + i] = o[i] * r * CF(1)[i] * sigmoidf_(CF(2)[(u64)h * d + i]);
}

/// Per-head RMS norm then NEOX RoPE on the first n_rot dims, theta_i = pos * base^(-2i/n_rot).
/// info: [0] x, [1] x stride per head, [2] w (dim, shared), [3] out, [4] out stride per head, [5] dim,
/// [6] n_rot, [7] address of the u32 position, [8] base bits, [9] eps bits, [10] apply norm (1) or only
/// rope (0), [11] output floats per position (the output row moves with the position; 0: fixed), [12] x stride
/// per token, [13] out stride per token. [7] is the positions array (u32 x tokens). Grid (heads, tokens).
extern "C" __global__ void qx_norm_rope(const u64* info) {
    __shared__ float sc[32];
    __shared__ float buf[256];
    const u32 h = blockIdx.x, d = U(5), nr = U(6), tk = blockIdx.y;
    const u32 pos_u = reinterpret_cast<const u32*>(info[7])[tk];
    const float* x = CF(0) + (u64)tk * info[12] + (u64)h * U(1);
    float* o = F(3) + (u64)tk * info[13] + (u64)pos_u * info[11] + (u64)h * U(4);
    float r = 1.f;
    if (U(10)) {
        float ss = 0.f;
        for (u32 i = threadIdx.x; i < d; i += blockDim.x) ss += x[i] * x[i];
        ss = block_sum(ss, sc);
        r = rsqrtf(ss / (float)d + __uint_as_float(U(9)));
    }
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) buf[i] = U(10) ? x[i] * r * CF(2)[i] : x[i];
    __syncthreads();
    const float pos = (float)pos_u, base = __uint_as_float(U(8));
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) {
        float v = buf[i];
        if (i < nr) {
            const u32 p = i % (nr / 2);
            const float theta = pos * powf(base, -2.f * (float)p / (float)nr);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = buf[p], x1 = buf[p + nr / 2];
            v = i < nr / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        o[i] = v;
    }
}

/// Indexer block pooling once the position completes a block: b = pos / block, pooled[b] = RoPE at b*block
/// of RMS norm of the mean of the block's raw keys. Does nothing mid-block.
/// info: [0] raw keys, [1] address of the u32 position, [2] w, [3] pooled keys, [4] dim, [5] n_rot,
/// [6] base bits, [7] eps bits, [8] block size, [9] ring slots when the raw keys are a ring (cell at slot
/// cell % ring), 0 when they are indexed by cell. One block of `dim` threads.
extern "C" __global__ void qx_idx_pool(const u64* info) {
    __shared__ float sc[32];
    __shared__ float buf[256];
    const u32 d = U(4), nr = U(5), kp = U(8), i = threadIdx.x;
    const u32 pos = *reinterpret_cast<const u32*>(info[1]);
    if ((pos + 1) % kp != 0) return;
    const u32 b = pos / kp;
    const u32 ring = U(9);
    float m = 0.f;
    if (i < d) {
        for (u32 c = 0; c < kp; ++c) {
            const u32 cell = b * kp + c;
            m += CF(0)[(u64)(ring ? cell % ring : cell) * d + i];
        }
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

/// Indexer scores: score[b] = sum_h relu(q_h . K_b) / sqrt(dim) for the visible blocks b < (pos+1)/block.
/// Skipped entirely while every visible block is kept anyway. One warp per block b.
/// info: [0] q (heads x dim), [1] pooled keys, [2] scores, [3] address of the u32 position, [4] heads,
/// [5] dim, [6] block size, [7] blocks kept, [8] scores stride per token, [9] slots (u32 x tokens), [10] pooled
/// key floats per slot. [3] is the positions array. Grid (blocks, tokens): token t's query row is heads x dim
/// further, its keys its slot's.
extern "C" __global__ void qx_idx_score(const u64* info) {
    const u32 b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, l = threadIdx.x % 32, tk = blockIdx.y;
    const u32 nv = (reinterpret_cast<const u32*>(info[3])[tk] + 1) / U(6);
    if (nv <= U(7) || b >= nv) return;
    const u32 d = U(5);
    const float* kb = CF(1) + (u64)SLOTS(9)[tk] * info[10] + (u64)b * d;
    const float* q = CF(0) + (u64)tk * U(4) * d;
    float tot = 0.f;
    for (u32 h = 0; h < U(4); ++h) {
        float s = 0.f;
        for (u32 i = l; i < d; i += 32) s += q[h * d + i] * kb[i];
        s = warp_sum(s);
        tot += fmaxf(s, 0.f);
    }
    if (l == 0) F(2)[(u64)tk * info[8] + b] = tot * rsqrtf((float)d);
}

/// Decode attention for one query head per block over the cells `cells[0..*count]`.
/// K, V are f32 [cell][kv_head][dim]. info: [0] q (heads x dim), [1] K, [2] V, [3] cells (u32*),
/// [4] address of the u32 count, [5] out (heads x dim), [6] kv heads, [7] heads, [8] dim, [9] scale bits,
/// [10] scores scratch (heads x max cells), [11] max cells.
extern "C" __global__ void qx_attn_decode(const u64* info) {
    __shared__ float sc[32];
    __shared__ float qs[256];
    const u32 h = blockIdx.x, d = U(8), n = *reinterpret_cast<const u32*>(info[4]), nkv = U(6), kvh = h / (U(7) / nkv);
    const u32* cells = reinterpret_cast<const u32*>(info[3]);
    float* scores = F(10) + (u64)h * U(11);
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) qs[i] = CF(0)[(u64)h * d + i];
    __syncthreads();
    const float scale = __uint_as_float(U(9));
    const u32 w = threadIdx.x / 32, l = threadIdx.x % 32, nw = blockDim.x / 32;
    for (u32 c = w; c < n; c += nw) {
        const u32 cell = cells[c];
        const float* k = CF(1) + ((u64)cell * nkv + kvh) * d;
        float s = 0.f;
        for (u32 i = l; i < d; i += 32) s += qs[i] * k[i];
        s = warp_sum(s);
        if (l == 0) scores[c] = s * scale;
    }
    __syncthreads();
    float m = -3.402823466e38f;
    for (u32 c = threadIdx.x; c < n; c += blockDim.x) m = fmaxf(m, scores[c]);
    m = block_max(m, sc);
    float z = 0.f;
    for (u32 c = threadIdx.x; c < n; c += blockDim.x) {
        const float p = __expf(scores[c] - m);
        scores[c] = p;
        z += p;
    }
    z = block_sum(z, sc);
    __syncthreads();
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (u32 c = 0; c < n; ++c) {
            const u32 cell = cells[c];
            acc += scores[c] * CF(2)[((u64)cell * nkv + kvh) * d + i];
        }
        F(5)[(u64)h * d + i] = acc / z;
    }
}

/// x[i] *= sigmoid(g[i]). info: [0] x, [1] g, [2] n.
extern "C" __global__ void qx_mul_sigmoid(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(2)) F(0)[i] *= sigmoidf_(CF(1)[i]);
}

/// out[i] = silu(a[i]) * b[i]. info: [0] a, [1] b, [2] out, [3] n.
extern "C" __global__ void qx_silu_mul(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(3)) F(2)[i] = siluf_(CF(0)[i]) * CF(1)[i];
}

/// y[i] += sigmoid(*g) * s[i]. info: [0] y, [1] s, [2] g (one float), [3] n.
extern "C" __global__ void qx_add_gated(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(3)) F(0)[i] += sigmoidf_(CF(2)[0]) * CF(1)[i];
}

/// out[i] = a[i] + b[i]. info: [0] a, [1] b, [2] out, [3] n.
extern "C" __global__ void qx_add(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < U(3)) F(2)[i] = CF(0)[i] + CF(1)[i];
}

/// PLE gate per stream: g[s] = sigmoid(sign(t) * sqrt(max(|t|, 1e-6))), t = (kn[s] . qn[s]) / sqrt(E).
/// info: [0] kn, [1] qn, [2] g, [3] E. Grid (streams, tokens): token rows HC*E (kn, qn) and HC (g) apart.
extern "C" __global__ void qx_ple_gate(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x + blockIdx.y * gridDim.x, n = U(3);
    float t = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) t += CF(0)[(u64)s * n + i] * CF(1)[(u64)s * n + i];
    t = block_sum(t, sc) * rsqrtf((float)n);
    if (threadIdx.x == 0) {
        const float sg = t > 0.f ? 1.f : (t < 0.f ? -1.f : 0.f);
        F(2)[s] = sigmoidf_(sg * sqrtf(fmaxf(fabsf(t), 1e-6f)));
    }
}

/// PLE apply over a window of T tokens in order, one block per stream s: gated = value * g[s]; cn = rms(gated) *
/// w_conv[s]; conv[c] = silu(sum_k w[c][k] * x_k[c]) over the history (9 rows, oldest first) and cn at taps 9, 6,
/// 3, 0 back; R[s] += gated + conv; history shifts in cn.
/// info: [0] value (T x E), [1] g (T x HC), [2] w_conv norm (HC x E), [3] conv w (HC*E x 4), [4] hist (9 x HC*E),
/// [5] R (T x HC*E), [6] E, [7] eps bits, [8] unused, [9] T (0: 1), [10] history before the window (9 x HC*E;
/// 0: none), [11] each token's cn (T x HC*E; 0: none), [12] slots (u32 x T), [13] history floats per slot. The
/// two before let a rollback rebuild the history. Grid (HC, T): a segment's first token runs it on its slot's
/// history.
extern "C" __global__ void qx_ple_apply(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, n = U(6), hcd = gridDim.x * n, T = U(9) ? U(9) : 1, t0 = blockIdx.y;
    u32 len;
    if (!seg_start(SLOTS(12), t0, T, len)) return;
    float* hist = F(4) + (u64)SLOTS(12)[t0] * info[13];
    if (info[10])
        for (u32 i = threadIdx.x; i < n; i += blockDim.x)
            for (u32 r = 0; r < 9; ++r) F(10)[(u64)r * hcd + s * n + i] = hist[(u64)r * hcd + s * n + i];
    for (u32 tk = t0; tk < t0 + len; ++tk) {
        const float* value = CF(0) + (u64)tk * n;
        const float g = CF(1)[(u64)tk * gridDim.x + s];
        float* R = F(5) + (u64)tk * hcd;
        float ss = 0.f;
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = value[i] * g;
            ss += v * v;
        }
        ss = block_sum(ss, sc);
        const float r = rsqrtf(ss / (float)n + __uint_as_float(U(7)));
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
            const u32 c = s * n + i;
            const float gated = value[i] * g;
            const float cn = gated * r * CF(2)[c];
            const float* w = CF(3) + 4ull * c;
            // taps: k=0 -> 9 back (hist[0]), k=1 -> 6 back (hist[3]), k=2 -> 3 back (hist[6]), k=3 -> now
            const float acc = w[0] * hist[0ull * hcd + c] + w[1] * hist[3ull * hcd + c] + w[2] * hist[6ull * hcd + c] + w[3] * cn;
            for (u32 t = 0; t < 8; ++t) hist[(u64)t * hcd + c] = hist[(u64)(t + 1) * hcd + c];
            hist[8ull * hcd + c] = cn;
            if (info[11]) F(11)[(u64)tk * hcd + c] = cn;
            R[c] += gated + siluf_(acc);
        }
    }
}

/// The PLE history after the first `a` tokens of a window: the last 9 of [history before; cn of token 0 .. a-1].
/// info: [0] hist (9 x HCD), [1] history before, [2] cn (T x HCD), [3] HCD, [4] a.
extern "C" __global__ void qx_ple_hist_rollback(const u64* info) {
    const u64 c = (u64)blockIdx.x * blockDim.x + threadIdx.x, hcd = info[3];
    if (c >= hcd) return;
    const u32 a = (u32)info[4];
    for (u32 r = 0; r < 9; ++r) {
        const u32 j = r + a;
        F(0)[r * hcd + c] = j < 9 ? CF(1)[(u64)j * hcd + c] : CF(2)[(u64)(j - 9) * hcd + c];
    }
}

/// Index of the largest of n values; one block. info: [0] x, [1] n, [2] out (u32).
extern "C" __global__ void qx_argmax(const u64* info) {
    __shared__ float bv[1024];
    __shared__ u32 bi[1024];
    float v = -3.402823466e38f;
    u32 idx = 0;
    for (u32 i = threadIdx.x; i < U(1); i += blockDim.x) {
        const float x = CF(0)[i];
        if (x > v) { v = x; idx = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const u32 oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = o; bi[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) reinterpret_cast<u32*>(info[2])[0] = bi[0];
}

/// The second half of each `2*dim` chunk: out[h*dim + i] = x[h*2*dim + dim + i]. info: [0] x, [1] out, [2] dim, [3] n.
extern "C" __global__ void qx_take_halves(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, d = U(2);
    if (i < U(3)) F(1)[i] = CF(0)[(i / d) * 2 * d + d + i % d];
}

/// out[(pos % ring) * stride + i] = x[i]. info: [0] x, [1] out, [2] n, [3] pos ptr, [4] stride, [5] ring.
extern "C" __global__ void qx_copy_ring(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 off = (u64)(*reinterpret_cast<const u32*>(info[3]) % U(5)) * info[4];
    if (i < U(2)) F(1)[off + i] = CF(0)[i];
}

/// out[pos * stride + i] = x[i]. info: [0] x, [1] out, [2] n, [3] address of the u32 position (or 0), [4] stride.
extern "C" __global__ void qx_copy(const u64* info) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 off = info[3] ? (u64)*reinterpret_cast<const u32*>(info[3]) * info[4] : 0;
    if (i < U(2)) F(1)[off + i] = CF(0)[i];
}

/// QSA cell selection for the token at position `pos`: every cell of the top `top` visible blocks by score
/// (ties to the lower block, as the reference sorts), or of all visible blocks when there are no more than
/// `top`; then the tail, the `(pos+1) % block` most recent cells. Writes the list and its length.
/// info: [0] scores, [1] address of the u32 position, [2] block size, [3] top, [4] cells out (u32), [5] count out,
/// [6] scores stride per token, [7] cells stride per token. [1] is the positions array. Grid: one block of 1024
/// threads per token.
extern "C" __global__ void qx_qsa_select(const u64* info) {
    __shared__ u32 hist[256];
    __shared__ u32 s_prefix, s_need, s_count;
    __shared__ u32 scan[1024];
    const u32 tk = blockIdx.x;
    const float* sc = CF(0) + (u64)tk * info[6];
    const u32 pos = reinterpret_cast<const u32*>(info[1])[tk], kp = U(2), top = U(3);
    u32* cells = reinterpret_cast<u32*>(info[4]) + (u64)tk * info[7];
    const u32 nv = (pos + 1) / kp, n_tail = (pos + 1) % kp;
    const u32 tid = threadIdx.x, nt = blockDim.x;
    if (nv <= top) {
        for (u32 c = tid; c < nv * kp; c += nt) cells[c] = c;
        if (tid < n_tail) cells[nv * kp + tid] = pos - tid;
        if (tid == 0) reinterpret_cast<u32*>(info[5])[tk] = nv * kp + n_tail;
        return;
    }
    // Order-preserving keys: larger float, larger key. Radix-select the top-th largest key, 8 bits at a time.
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
            u32 acc = 0, d = 255;
            for (;; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
                if (d == 0) break;
            }
            s_prefix = prefix | (d << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        __syncthreads();
    }
    // prefix is the threshold key; take every block above it and the `need` lowest-index blocks equal to it.
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
        // slots in block order (prefix sum, not an atomic counter): attention sums cells in list order
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
    if (tid < n_tail) cells[top * kp + tid] = pos - tid;
    if (tid == 0) reinterpret_cast<u32*>(info[5])[tk] = top * kp + n_tail;
}

/// HC combine fused with the next HC mix's norm: R[s] += y * 2 * sigmoid(inject[s] / HC), then
/// xn[s] = rms(R[s]) * w[s]. One block per stream. info: [0] R, [1] y, [2] inject, [3] next norm w (HC x E),
/// [4] xn out, [5] E, [6] eps bits, [7] HC. Grid (HC, tokens): token rows HC*E (R, xn), E (y), HC (inject) apart.
extern "C" __global__ void qx_hc_combine_norm(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, n = U(5), hc = U(7);
    const u64 tk = blockIdx.y;
    float* R = F(0) + (tk * hc + s) * n;
    const float* y = CF(1) + tk * n;
    const float w = 2.f * sigmoidf_(CF(2)[tk * hc + s] / (float)hc);
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = R[i] + y[i] * w;
        R[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sc);
    const float r = rsqrtf(ss / (float)n + __uint_as_float(U(6)));
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) F(4)[(tk * hc + s) * n + i] = R[i] * r * CF(3)[(u64)s * n + i];
}

/// GDN causal conv fused with the q/k L2 norms, for a window of T tokens in order. Blocks 0 .. 2*kh-1 each own one
/// q or k head (128 channels) and normalise it after the conv; the rest own 128 v channels each.
/// info: [0] qkv (T x ch), [1] state (3 x ch, oldest first), [2] w (ch x 4), [3] out (T x ch), [4] ch, [5] dim,
/// [6] key heads, [7] eps bits, [8] state before the window (3 x ch; 0: none), [9] T (0: 1), [10] slots
/// (u32 x T), [11] state floats per slot. Grid (ch / 128, T) blocks of 128 threads: a segment's first token runs
/// the segment on its slot's state.
extern "C" __global__ void qx_gdn_conv_l2(const u64* info) {
    __shared__ float sc[32];
    const u32 ch = U(4), d = U(5), i = blockIdx.x * d + threadIdx.x, T = U(9) ? U(9) : 1, t0 = blockIdx.y;
    u32 len;
    if (!seg_start(SLOTS(10), t0, T, len)) return;
    const float* w = CF(2) + 4ull * i;
    float* st = F(1) + (u64)SLOTS(10)[t0] * info[11];
    float s0 = st[i], s1 = st[ch + i], s2 = st[2 * ch + i];
    if (info[8]) {
        F(8)[i] = s0;
        F(8)[ch + i] = s1;
        F(8)[2 * ch + i] = s2;
    }
    for (u32 tk = t0; tk < t0 + len; ++tk) {
        const float x = CF(0)[(u64)tk * ch + i];
        float c = siluf_(w[3] * x + w[0] * s0 + w[1] * s1 + w[2] * s2);
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (blockIdx.x < 2 * U(6)) {
            const float ssum = block_sum(c * c, sc);
            c *= rsqrtf(ssum + __uint_as_float(U(7)));
        }
        F(3)[(u64)tk * ch + i] = c;
    }
    st[i] = s0;
    st[ch + i] = s1;
    st[2 * ch + i] = s2;
}

/// The conv state after the first `a` tokens of a window: the last 3 of [state before; qkv of token 0 .. a-1].
/// info: [0] state (3 x ch), [1] state before, [2] qkv (T x ch), [3] ch, [4] a.
extern "C" __global__ void qx_conv_rollback(const u64* info) {
    const u64 c = (u64)blockIdx.x * blockDim.x + threadIdx.x, ch = info[3];
    if (c >= ch) return;
    const u32 a = (u32)info[4];
    for (u32 r = 0; r < 3; ++r) {
        const u32 j = r + a;
        F(0)[r * ch + c] = j < 3 ? CF(1)[(u64)j * ch + c] : CF(2)[(u64)(j - 3) * ch + c];
    }
}

/// The indexer for a window of T tokens, each segment in order: token t's raw key goes to its ring slot pos % ring,
/// and when pos completes a block, the block's pooled key is formed as qx_idx_pool does.
/// info: [0] raw keys (T x dim), [1] positions (u32 x T), [2] w, [3] pooled keys, [4] dim, [5] n_rot,
/// [6] base bits, [7] eps bits, [8] block size, [9] ring, [10] ring slots, [11] T, [12] ring floats per slot,
/// [13] pooled key floats per slot, [14] slots (u32 x T). Grid T blocks of `dim` threads: a segment's first
/// token runs the segment on its slot's ring and keys.
extern "C" __global__ void qx_idx_ring_pool(const u64* info) {
    __shared__ float sc[32];
    __shared__ float buf[256];
    const u32 d = U(4), nr = U(5), kp = U(8), rs = U(10), T = U(11), i = threadIdx.x, t0 = blockIdx.x;
    u32 len;
    if (!seg_start(SLOTS(14), t0, T, len)) return;
    const u32 slot = SLOTS(14)[t0];
    float* ring = F(9) + (u64)slot * info[12];
    float* pooled = F(3) + (u64)slot * info[13];
    for (u32 tk = t0; tk < t0 + len; ++tk) {
        const u32 pos = reinterpret_cast<const u32*>(info[1])[tk];
        if (i < d) ring[(u64)(pos % rs) * d + i] = CF(0)[(u64)tk * d + i];
        __syncthreads();
        if ((pos + 1) % kp != 0) continue;
        const u32 b = pos / kp;
        float m = 0.f;
        if (i < d) {
            for (u32 c = 0; c < kp; ++c) {
                const u32 cell = b * kp + c;
                m += ring[(u64)(cell % rs) * d + i];
            }
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
            pooled[(u64)b * d + i] = v;
        }
        __syncthreads();
    }
}

/// The MTP drafter's input rows: for token t and stream s, [rms(e_t) * enorm ; rms(h_t,s) * hnorm_s] (2E).
/// info: [0] e (T x E), [1] h (T x HC*E), [2] enorm (E), [3] hnorm (HC x E), [4] out (T x HC x 2E), [5] E,
/// [6] eps bits. Grid (HC, T), 256 threads.
extern "C" __global__ void qx_mtp_cat(const u64* info) {
    __shared__ float sc[32];
    const u32 s = blockIdx.x, n = U(5), hc = gridDim.x;
    const u64 tk = blockIdx.y;
    const float* e = CF(0) + tk * n;
    const float* h = CF(1) + (tk * hc + s) * n;
    float* o = F(4) + (tk * hc + s) * 2 * n;
    float se = 0.f, sh = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        se += e[i] * e[i];
        sh += h[i] * h[i];
    }
    se = block_sum(se, sc);
    sh = block_sum(sh, sc);
    const float re = rsqrtf(se / (float)n + __uint_as_float(U(6))), rh = rsqrtf(sh / (float)n + __uint_as_float(U(6)));
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        o[i] = e[i] * re * CF(2)[i];
        o[n + i] = h[i] * rh * CF(3)[(u64)s * n + i];
    }
}

/// Greedy pick and its probability per token: tok[t] = argmax of logits row t (lowest index on ties), p[t] =
/// softmax probability of it. info: [0] logits (T x n), [1] n, [2] tokens out (u32), [3] probabilities out.
/// One block of 1024 threads per token.
extern "C" __global__ void qx_argmax_p(const u64* info) {
    __shared__ float bv[1024];
    __shared__ u32 bi[1024];
    __shared__ float sc[32];
    const u32 n = U(1), tid = threadIdx.x, tk = blockIdx.x;
    const float* x = CF(0) + (u64)tk * n;
    float v = -3.402823466e38f;
    u32 idx = 0;
    for (u32 i = tid; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    bv[tid] = v;
    bi[tid] = idx;
    __syncthreads();
    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (bv[tid + s] > bv[tid] || (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]))) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
        __syncthreads();
    }
    const float m = bv[0];
    float z = 0.f;
    for (u32 i = tid; i < n; i += blockDim.x) z += __expf(x[i] - m);
    z = block_sum(z, sc);
    if (tid == 0) {
        reinterpret_cast<u32*>(info[2])[tk] = bi[0];
        F(3)[tk] = 1.f / z;
    }
}

/// *p = v. info: [0] p (u32), [1] v. One thread.
extern "C" __global__ void qx_set_u32(const u64* info) { *reinterpret_cast<u32*>(info[0]) = (u32)info[1]; }

/// *p = v. info: [0] p (u64), [1] v. One thread.
extern "C" __global__ void qx_set_u64(const u64* info) { *reinterpret_cast<u64*>(info[0]) = info[1]; }

/// The drafter's positions: dpos[t] = step pos + off + t for t < QX_WMAX (off signed). info: [0] the slot's step
/// row, [1] dpos (u32 x QX_WMAX), [2] off. One thread.
extern "C" __global__ void qx_draft_pos(const u64* info) {
    const i64 p = (i64)reinterpret_cast<const u32*>(info[0])[0] + (i64)info[2];
    for (u32 t = 0; t < QX_WMAX; ++t) reinterpret_cast<u32*>(info[1])[t] = (u32)(p + t);
}
