// Routed experts on the GPU for one token: routing, and the experts resident in the VRAM cache.
//
// A resident expert's slot holds [gate rows (ff x E) | up rows (ff x E) | down rows (E x ff)] in GGUF bytes.
// `slots[k]` is the slot of the k-th selected expert, snapshotted by qx_moe_post, or 0 when the CPU computes it.
#include "qx_quant.cuh"
#include "qx_block.cuh"
#include "qx_math.cuh"
#include "qx_sys.cuh"

#define F(i) reinterpret_cast<float*>(info[i])
#define CF(i) reinterpret_cast<const float*>(info[i])
#define U(i) ((u32)info[i])
#define QX_SLOT_MASK 0x0000ffffffffffffull

/// Softmax over n (<= 1024) logits, top k by probability (ties to the lower index), weights renormalised over
/// the k (sum clamped at 6.103515625e-5). One warp: each lane holds n/32 logits.
/// info: [0] logits, [1] n, [2] k, [3] ids (i32 x k), [4] weights (f32 x k). One block of 32 threads per token:
/// token t reads logits row t and writes ids and weights at t * k.
/// Cache-aware routing (opt-in, lossy) when [7] J > 0: the top J are kept, the other k - J picks prefer the experts
/// among the top M (<= 32) that the VRAM cache holds (table [5] at entry [6] + id, nonzero: resident), in rank
/// order, then the best of the rest; ids in rank order, weights renormalised over the picks.
extern "C" __global__ void qx_moe_route(const u64* info) {
    const u32 n = U(1), k = U(2), l = threadIdx.x, tk = blockIdx.x;
    const float* lg = CF(0) + (u64)tk * n;
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
    const u32 J = U(7), M = J ? min(max(U(8), k), 32u) : k;
    for (u32 r = 0; r < M; ++r) {
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
        if (r < k) tot += p;
        if (l == r) { sel = p; myid = bi; }
        // bi stays at the sentinel when no lane had a candidate (NaN logits): never index with it
        if (bi != 0x7fffffff && (u32)(bi % 32) == l) v[bi / 32] = -2.f;
    }
    if (J == 0) {
        tot = fmaxf(tot, 6.103515625e-5f);
        if (l < k) {
            reinterpret_cast<i32*>(info[3])[(u64)tk * k + l] = myid;
            F(4)[(u64)tk * k + l] = sel / tot;
        }
        return;
    }
    // lane r holds rank r's candidate: the first J, then resident ones in rank order, then the rest
    const u64* table = reinterpret_cast<const u64*>(info[5]) + info[6];
    const bool valid = l < M && myid != 0x7fffffff;
    const bool res = valid && *reinterpret_cast<const volatile u64*>(&table[myid]) != 0;
    const u32 want = valid && (l < J || res), lt = (1u << l) - 1;
    const u32 S = __ballot_sync(0xffffffffu, want), R = __ballot_sync(0xffffffffu, valid) & ~S;
    // the first k of S, then the first k - |S| of R
    auto first = [](u32 m, u32 n) { u32 o = 0; for (u32 i = 0; i < n && m; ++i) { const u32 b = m & (0u - m); o |= b; m ^= b; } return o; };
    const u32 ns = __popc(S);
    const u32 chosen = ns >= k ? first(S, k) : S | first(R, k - ns);
    const bool mine = (chosen >> l) & 1u;
    float sum = mine ? sel : 0.f;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    sum = fmaxf(sum, 6.103515625e-5f);
    if (mine) {
        const u32 at = __popc(chosen & lt);
        reinterpret_cast<i32*>(info[3])[(u64)tk * k + at] = myid;
        F(4)[(u64)tk * k + at] = sel / sum;
    }
}

/// Gate and up projections of the resident selected experts: gu[k][r] for r < 2*ff (gate rows, then up).
/// Grid: (ceil(2*ff / 8), selections). info: [0] slots, [1] experts per token (0: one token), [2] unused, [3] x
/// (token t's row at t * E), [4] gu out, [5] ff, [6] E (cols), [7] gate/up row stride bytes.
template <int T> QX_DEV void moe_gateup(const u64* info) {
    const u32 kk = blockIdx.y;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 ff = U(5), cols = U(6);
    if (warp >= 2 * ff) return;
    const u8* row = reinterpret_cast<const u8*>(slot) + (u64)warp * info[7];
    const float* x = CF(3) + (U(1) ? (u64)(kk / U(1)) * cols : 0);
    float acc = 0.f;
    for (u32 c = 0; c * 256 < cols; ++c) {
        float w[8];
        decode8<T>(row, c, lane, w);
        const u32 b = c * 256 + 8 * lane;
        const float4 x0 = *reinterpret_cast<const float4*>(x + b);
        const float4 x1 = *reinterpret_cast<const float4*>(x + b + 4);
        acc += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
    }
    acc = warp_sum(acc);
    if constexpr (T == QX_NVFP4) {
        // the per-tensor scales {gate, up, down} at the end of the record: info[8]
        acc *= reinterpret_cast<const float*>(slot + info[8])[warp < ff ? 0 : 1];
    }
    if (lane == 0) F(4)[(u64)kk * 2 * ff + warp] = acc;
}
extern "C" __global__ void qx_moe_gateup_iq2_s(const u64* info) { moe_gateup<QX_IQ2_S>(info); }
extern "C" __global__ void qx_moe_gateup_iq3_s(const u64* info) { moe_gateup<QX_IQ3_S>(info); }
extern "C" __global__ void qx_moe_gateup_nvfp4(const u64* info) { moe_gateup<QX_NVFP4>(info); }

/// f32 down projection of the resident selected experts (exact mode): yk[k][r] = down_e[r] . (silu(g) * u).
/// Grid: (ceil(E / 8), k). info: [0] slots, [1] unused, [2] unused, [3] unused, [4] gu, [5] yk out,
/// [6] ff, [7] E, [8] offset of the down rows in the slot (bytes), [9] down row stride bytes.
extern "C" __global__ void qx_moe_down_iq4_nl(const u64* info) {
    __shared__ float h[1024];
    const u32 kk = blockIdx.y, ff = U(6), n = U(7);
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (slot == 0) return;
    const float* gu = CF(4) + (u64)kk * 2 * ff;
    for (u32 i = threadIdx.x; i < ff; i += blockDim.x) h[i] = qx_silu_mul(gu[i], gu[ff + i]);
    __syncthreads();
    if (warp >= n) return;
    const u8* row = reinterpret_cast<const u8*>(slot) + info[8] + (u64)warp * info[9];
    float acc = 0.f;
    for (u32 c = 0; c * 256 < ff; ++c) {
        const u32 b = c * 256 + 8 * lane;
        if (b < ff) {
            float w[8];
            decode8<QX_IQ4_NL>(row, c, lane, w);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc += w[i] * h[b + i];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) F(5)[(u64)kk * n + warp] = acc;
}

/// y[r] = sum over k in order of w[k] * v_k[r], v_k from the GPU when slots[k] is set, else from the CPU's rows
/// in the mailbox; then y[r] += sigmoid(*sg) * sh[r], the shared expert. A fixed order and correctly rounded
/// operations make y independent of which tier ran which expert. info: [0] GPU outputs (selections x E), [1] k,
/// [2] CPU outputs (mapped, selections x E), [3] y, [4] E, [5] slots, [6] weights, [7] shared expert out, [8] its
/// gate. Grid (ceil(E / block), tokens): token t combines selections t*k .. t*k + k - 1.
extern "C" __global__ void qx_moe_combine(const u64* info) {
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
    F(3)[tk * n + r] = s + sigmoidf_(CF(8)[tk]) * CF(7)[tk * n + r];
}

// ---- quantized-activation experts, bit for bit the CPU pool's arithmetic ----

/// Q8_K quantization of x, as ggml and the CPU pool round it: per 256, the value of largest magnitude (first
/// on ties) sets iscale = -127 / max, q = min(127, rint(iscale * x)), d = 1 / iscale.
/// info: [0] x, [1] qs out (i8 x E), [2] d out (f32 x E/256). Grid E/256 blocks of 256.
extern "C" __global__ void qx_quant_q8k(const u64* info) {
    __shared__ float av[256];
    __shared__ u32 ai[256];
    const u32 b = blockIdx.x, t = threadIdx.x;
    const float v = CF(0)[b * 256 + t];
    av[t] = fabsf(v);
    ai[t] = t;
    __syncthreads();
    for (u32 s = 128; s > 0; s >>= 1) {
        if (t < s) {
            const float o = av[t + s];
            const u32 oi = ai[t + s];
            if (o > av[t] || (o == av[t] && oi < ai[t])) { av[t] = o; ai[t] = oi; }
        }
        __syncthreads();
    }
    const float amax = av[0];
    i8* qs = reinterpret_cast<i8*>(info[1]) + b * 256;
    if (amax == 0.f) {
        qs[t] = 0;
        if (t == 0) F(2)[b] = 0.f;
        return;
    }
    const float mx = CF(0)[b * 256 + ai[0]];
    const float iscale = __fdiv_rn(-127.f, mx);
    qs[t] = (i8)min(127, __float2int_rn(__fmul_rn(iscale, v)));
    if (t == 0) F(2)[b] = __fdiv_rn(1.f, iscale);
}

QX_DEV int pack4(int a, int b, int c, int d) { return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((u32)(d & 0xff) << 24); }

/// Gate and up rows of the resident selected experts against Q8_K activations, with integer dots:
/// gu[k][r] for r < 2*ff. Each warp does RPW consecutive rows with its lane's activations held in registers.
/// Grid: (ceil(2*ff / (8 * RPW)), selections), 256 threads. info: [0] slots, [1] x qs (token t at t * E), [2] x d
/// (token t at t * E / 256), [3] experts per token (0: one token), [4] gu out, [5] ff, [6] E, [7] gate row bytes.
#define RPW 1
template <int T> QX_DEV void gateup_q8k(const u64* info) {
    const u32 kk = blockIdx.y;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 ff = U(5), nb = U(6) / 256;
    const u32 tok = U(3) ? kk / U(3) : 0;
    const i8* xq = reinterpret_cast<const i8*>(info[1]) + (u64)tok * U(6);
    const float* xd = CF(2) + (u64)tok * nb;
    int2 a[10];
    float dq[10];
#pragma unroll
    for (u32 b = 0; b < 10; ++b) {
        if (b < nb) {
            a[b] = *reinterpret_cast<const int2*>(xq + b * 256 + 8 * lane);
            dq[b] = xd[b];
        }
    }
    for (u32 rr = 0; rr < RPW; ++rr) {
        const u32 r = warp * RPW + rr;
        if (r >= 2 * ff) return;
        const u8* row = reinterpret_cast<const u8*>(slot) + (u64)r * info[7];
        float acc = 0.f;
#pragma unroll
        for (u32 b = 0; b < 10; ++b) {
            if (b >= nb) break;
            int w0, w1, mult;
            float d;
            if (T == QX_IQ2_S) {
                const u8* blk = row + (u64)b * 82;
                d = ld_f16(blk);
                const u8 *qs = blk + 2, *signs = blk + 34, *qh = blk + 66, *scales = blk + 74;
                const int g = lane, sub = g / 2;
                mult = 2 * ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) + 1;
                const u64 grid = __ldg(&IQ2S_GRID[qs[g] | (((qh[g / 4] >> (2 * (g % 4))) & 3) << 8)]);
                const u8 sg = signs[g];
                int m[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int v = (int)((grid >> (8 * e)) & 0xff);
                    m[e] = (sg >> e) & 1 ? -v : v;
                }
                w0 = pack4(m[0], m[1], m[2], m[3]);
                w1 = pack4(m[4], m[5], m[6], m[7]);
            } else {
                const u8* blk = row + (u64)b * 110;
                d = ld_f16(blk);
                const u8 *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
                const int g = lane, sub = g / 4;
                mult = 2 * ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) + 1;
                const u8 sg = signs[g];
                int m[8];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int kq = 2 * g + h;
                    const u32 grid = __ldg(&IQ3S_GRID[qs[kq] | (((qh[kq / 8] >> (kq % 8)) & 1) << 8)]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int e = 4 * h + j;
                        const int v = (int)((grid >> (8 * j)) & 0xff);
                        m[e] = (sg >> e) & 1 ? -v : v;
                    }
                }
                w0 = pack4(m[0], m[1], m[2], m[3]);
                w1 = pack4(m[4], m[5], m[6], m[7]);
            }
            int isum = __dp4a(w1, a[b].y, __dp4a(w0, a[b].x, 0)) * mult;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) isum += __shfl_xor_sync(0xffffffffu, isum, o);
            float rv = __fmul_rn(__fmul_rn(d, dq[b]), (float)isum);
            if (T == QX_IQ2_S) rv = __fmul_rn(rv, 0.125f);
            acc = __fadd_rn(acc, rv);
        }
        if (lane == 0) F(4)[(u64)kk * 2 * ff + r] = acc;
    }
}
extern "C" __global__ void qx_moe_gateup_q8k_iq2_s(const u64* info) { gateup_q8k<QX_IQ2_S>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq3_s(const u64* info) { gateup_q8k<QX_IQ3_S>(info); }

// ---- the IQ and K-quant expert formats with Q8_K activations on both sides, bit for bit the CPU pool ----
//
// One 256-element block per step; lane l holds elements 8l .. 8l+8. `iq_terms<T>` gives the lane's integer
// contribution to the block's two sums from its 8 activations (packed int2 `a`): for the IQ formats
// `sum += mult * (signed magnitudes . a)`, for IQ1_S `sum += mult * (8 * grid . a +- a-sum)`, for Q2_K
// `sum += scale * (w . a)` and `msum += min * a-sum`. The block value is then `(d * dq) * sum * factor`
// (Q2_K: `dq * (d * sum - dmin * msum)`), blocks accumulated in order, as qx_cpu::dot's scalar versions do.

template <int T> QX_DEV void iq_terms(const u8* blk, int lane, int2 a, float& d, float& dmin, int& sum, int& msum) {
    int m[8];
    int mult = 0;
    const u32 hi = a.x, lo = a.y;
    if constexpr (T == QX_IQ2_S) {
        d = ld_f16(blk);
        const u8 *qs = blk + 2, *signs = blk + 34, *qh = blk + 66, *scales = blk + 74;
        const int g = lane, sub = g / 2;
        mult = 2 * ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) + 1;
        const u64 grid = __ldg(&IQ2S_GRID[qs[g] | (((qh[g / 4] >> (2 * (g % 4))) & 3) << 8)]);
        const u8 sg = signs[g];
#pragma unroll
        for (int e = 0; e < 8; ++e) { const int v = (int)((grid >> (8 * e)) & 0xff); m[e] = (sg >> e) & 1 ? -v : v; }
    } else if constexpr (T == QX_IQ3_S) {
        d = ld_f16(blk);
        const u8 *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        const int g = lane, sub = g / 4;
        mult = 2 * ((scales[sub / 2] >> (4 * (sub % 2))) & 0xf) + 1;
        const u8 sg = signs[g];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kq = 2 * g + h;
            const u32 grid = __ldg(&IQ3S_GRID[qs[kq] | (((qh[kq / 8] >> (kq % 8)) & 1) << 8)]);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int e = 4 * h + j; const int v = (int)((grid >> (8 * j)) & 0xff); m[e] = (sg >> e) & 1 ? -v : v; }
        }
    } else if constexpr (T == QX_IQ2_XXS) {
        d = ld_f16(blk);
        const int i = lane / 4, g = lane % 4;
        const u32 aw = ld_u32(blk + 2 + 8 * i), s = ld_u32(blk + 6 + 8 * i);
        mult = 2 * (int)(s >> 28) + 1;
        const u64 grid = __ldg(&IQ2XXS_GRID[(aw >> (8 * g)) & 0xff]);
        const u8 sg = __ldg(&KSIGNS[(s >> (7 * g)) & 0x7f]);
#pragma unroll
        for (int e = 0; e < 8; ++e) { const int v = (int)((grid >> (8 * e)) & 0xff); m[e] = (sg >> e) & 1 ? -v : v; }
    } else if constexpr (T == QX_IQ2_XS) {
        d = ld_f16(blk);
        const int g = lane, sub = g / 2;
        mult = 2 * ((blk[66 + sub / 2] >> (4 * (sub % 2))) & 0xf) + 1;
        const u32 q = ld_u16(blk + 2 + 2 * g);
        const u64 grid = __ldg(&IQ2XS_GRID[q & 511]);
        const u8 sg = __ldg(&KSIGNS[q >> 9]);
#pragma unroll
        for (int e = 0; e < 8; ++e) { const int v = (int)((grid >> (8 * e)) & 0xff); m[e] = (sg >> e) & 1 ? -v : v; }
    } else if constexpr (T == QX_IQ3_XXS) {
        d = ld_f16(blk);
        const int i = lane / 4, g = lane % 4;
        const u32 s = ld_u32(blk + 66 + 4 * i);
        mult = 2 * (int)(s >> 28) + 1;
        const u8 sg = __ldg(&KSIGNS[(s >> (7 * g)) & 0x7f]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const u32 grid = __ldg(&IQ3XXS_GRID[blk[2 + 8 * i + 2 * g + h]]);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int e = 4 * h + j; const int v = (int)((grid >> (8 * j)) & 0xff); m[e] = (sg >> e) & 1 ? -v : v; }
        }
    } else if constexpr (T == QX_IQ1_S) {
        d = ld_f16(blk);
        const int i = lane / 4, g = lane % 4;
        const u32 qh = ld_u16(blk + 34 + 2 * i);
        mult = 2 * (int)((qh >> 12) & 7) + 1;
        const u64 grid = __ldg(&IQ1S_GRID[blk[2 + 4 * i + g] | (((qh >> (3 * g)) & 7) << 8)]);
#pragma unroll
        for (int e = 0; e < 8; ++e) m[e] = (int)((grid >> (8 * e)) & 0xff);  // 0, 1, 2 for -1, 0, 1
        const int bsum = __dp4a(pack4(m[4], m[5], m[6], m[7]), (int)lo, __dp4a(pack4(m[0], m[1], m[2], m[3]), (int)hi, 0));
        const int qsum = __dp4a(0x01010101, (int)lo, __dp4a(0x01010101, (int)hi, 0));
        sum += mult * (8 * (bsum - qsum) + ((qh & 0x8000) ? -qsum : qsum));
        return;
    } else if constexpr (T == QX_Q2_K) {
        d = ld_f16(blk + 80);
        dmin = ld_f16(blk + 82);
        const int j = lane / 2;
        const u8* q = blk + 16 + 32 * (lane / 16) + 8 * (lane % 4);
        const int sh = 2 * ((lane % 16) / 4);
#pragma unroll
        for (int e = 0; e < 8; ++e) m[e] = (q[e] >> sh) & 3;
        const int isum = __dp4a(pack4(m[4], m[5], m[6], m[7]), (int)lo, __dp4a(pack4(m[0], m[1], m[2], m[3]), (int)hi, 0));
        const int qsum = __dp4a(0x01010101, (int)lo, __dp4a(0x01010101, (int)hi, 0));
        sum += (blk[j] & 0xf) * isum;
        msum += (blk[j] >> 4) * qsum;
        return;
    }
    sum += mult * __dp4a(pack4(m[4], m[5], m[6], m[7]), (int)lo, __dp4a(pack4(m[0], m[1], m[2], m[3]), (int)hi, 0));
}

/// The block value from the warp-summed integer terms, in the CPU pool's operation order.
template <int T> QX_DEV float iq_block_value(float d, float dmin, float dq, int sum, int msum) {
    if constexpr (T == QX_Q2_K) return __fmul_rn(dq, __fsub_rn(__fmul_rn(d, (float)sum), __fmul_rn(dmin, (float)msum)));
    float rv = __fmul_rn(__fmul_rn(d, dq), (float)sum);
    if constexpr (T == QX_IQ2_S || T == QX_IQ2_XXS || T == QX_IQ2_XS || T == QX_IQ1_S) rv = __fmul_rn(rv, 0.125f);
    if constexpr (T == QX_IQ3_XXS) rv = __fmul_rn(rv, 0.25f);
    return rv;
}

/// Gate and up rows of the resident selected experts against Q8_K activations, for an input of up to NB * 256
/// elements. Grid and info as gateup_q8k.
template <int T, int NB, bool SEL = false> QX_DEV void gateup_q8k_t(const u64* info) {
    // SEL (the chunk path's per-expert launches): blockIdx.y indexes a selection list [9] and [0] is the address
    // of one slot word the driver writes behind the record's copy (the launch waited on it)
    const u32 kk = SEL ? reinterpret_cast<const u32*>(info[9])[blockIdx.y] : blockIdx.y;
    // a chunk slot word carries the job's generation in its top 16 bits (no zeroing between jobs)
    const u64 slot = SEL ? (*reinterpret_cast<const volatile u64*>(info[0]) & QX_SLOT_MASK) : reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 ff = U(5), nb = U(6) / 256;
    const u32 tok = U(3) ? kk / U(3) : 0;
    const i8* xq = reinterpret_cast<const i8*>(info[1]) + (u64)tok * U(6);
    const float* xd = CF(2) + (u64)tok * nb;
    int2 a[NB];
    float dq[NB];
#pragma unroll
    for (u32 b = 0; b < NB; ++b) {
        if (b < nb) {
            a[b] = *reinterpret_cast<const int2*>(xq + b * 256 + 8 * lane);
            dq[b] = xd[b];
        }
    }
    const u32 r = warp;
    if (r >= 2 * ff) return;
    const u8* row = reinterpret_cast<const u8*>(slot) + (u64)r * info[7];
    float acc = 0.f;
#pragma unroll
    for (u32 b = 0; b < NB; ++b) {
        if (b >= nb) break;
        float d = 0.f, dmin = 0.f;
        int sum = 0, msum = 0;
        iq_terms<T>(row + (u64)b * Fmt<T>::BB, lane, a[b], d, dmin, sum, msum);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); msum += __shfl_xor_sync(0xffffffffu, msum, o); }
        acc = __fadd_rn(acc, iq_block_value<T>(d, dmin, dq[b], sum, msum));
    }
    if (lane == 0) F(4)[(u64)kk * 2 * ff + r] = acc;
}
extern "C" __global__ void qx_moe_gateup_q8k_iq2_s_w20(const u64* info) { gateup_q8k_t<QX_IQ2_S, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq2_xxs_w20(const u64* info) { gateup_q8k_t<QX_IQ2_XXS, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq2_xs_w20(const u64* info) { gateup_q8k_t<QX_IQ2_XS, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq3_xxs_w20(const u64* info) { gateup_q8k_t<QX_IQ3_XXS, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq1_s_w20(const u64* info) { gateup_q8k_t<QX_IQ1_S, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_q2_k_w20(const u64* info) { gateup_q8k_t<QX_Q2_K, 20>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq2_s_w20_sel(const u64* info) { gateup_q8k_t<QX_IQ2_S, 20, true>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq2_xxs_w20_sel(const u64* info) { gateup_q8k_t<QX_IQ2_XXS, 20, true>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq2_xs_w20_sel(const u64* info) { gateup_q8k_t<QX_IQ2_XS, 20, true>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq3_xxs_w20_sel(const u64* info) { gateup_q8k_t<QX_IQ3_XXS, 20, true>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_iq1_s_w20_sel(const u64* info) { gateup_q8k_t<QX_IQ1_S, 20, true>(info); }
extern "C" __global__ void qx_moe_gateup_q8k_q2_k_w20_sel(const u64* info) { gateup_q8k_t<QX_Q2_K, 20, true>(info); }

/// Down rows of the resident selected experts for formats with 256-element blocks: the expert's hidden
/// activations `h = silu(min(g, c)) * clamp(u, -c, c)` (c = 0: plain silu(g) * u) are quantized to Q8_K in shared
/// memory exactly as the CPU pool does (qx_quant_q8k's rule per 256), then each warp streams 2 rows with
/// integer dots, blocks summed in order. ff <= 2560. Grid: (ceil(E / (8 * DRPW)), k), 256 threads.
/// info: [0] slots, [1] gu, [2] clamp (f32 bits), [3] unused, [4] v out (k x E), [5] ff, [6] E, [7] offset of the
/// down rows in the slot, [8] down row bytes.
template <int T, bool SEL = false> QX_DEV void down_q8k_t(const u64* info) {
    __shared__ __align__(16) i8 hq[2560];
    __shared__ float hd[10];
    __shared__ float av[256];
    __shared__ u32 ai[256];
    const u32 kk = SEL ? reinterpret_cast<const u32*>(info[9])[blockIdx.y] : blockIdx.y, ff = U(5), n = U(6), nb = ff / 256;
    const u64 slot = SEL ? (*reinterpret_cast<const volatile u64*>(info[0]) & QX_SLOT_MASK) : reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const float* gu = CF(1) + (u64)kk * 2 * ff;
    const float c = __uint_as_float((u32)info[2]);
    const u32 t = threadIdx.x;
    for (u32 b = 0; b < nb; ++b) {
        const u32 i = b * 256 + t;
        const float g = gu[i], u = gu[ff + i];
        const float h = c > 0.f ? qx_silu_mul(fminf(g, c), fminf(fmaxf(u, -c), c)) : qx_silu_mul(g, u);
        av[t] = fabsf(h);
        ai[t] = t;
        __syncthreads();
        for (u32 s = 128; s > 0; s >>= 1) {
            if (t < s) {
                const float o = av[t + s];
                const u32 oi = ai[t + s];
                if (o > av[t] || (o == av[t] && oi < ai[t])) { av[t] = o; ai[t] = oi; }
            }
            __syncthreads();
        }
        if (av[0] == 0.f) {
            hq[i] = 0;
            if (t == 0) hd[b] = 0.f;
        } else {
            const u32 mi = b * 256 + ai[0];
            const float gm = gu[mi], um = gu[ff + mi];
            const float hm = c > 0.f ? qx_silu_mul(fminf(gm, c), fminf(fmaxf(um, -c), c)) : qx_silu_mul(gm, um);
            const float iscale = __fdiv_rn(-127.f, hm);
            hq[i] = (i8)min(127, __float2int_rn(__fmul_rn(iscale, h)));
            if (t == 0) hd[b] = __fdiv_rn(1.f, iscale);
        }
        __syncthreads();
    }
    const u32 warp = t / 32, lane = t % 32;
    for (u32 rr = 0; rr < 2; ++rr) {
        const u32 r = (blockIdx.x * (blockDim.x / 32) + warp) * 2 + rr;
        if (r >= n) return;
        const u8* row = reinterpret_cast<const u8*>(slot) + info[7] + (u64)r * info[8];
        float acc = 0.f;
        for (u32 b = 0; b < nb; ++b) {
            const int2 a = *reinterpret_cast<const int2*>(hq + b * 256 + 8 * lane);
            float d = 0.f, dmin = 0.f;
            int sum = 0, msum = 0;
            iq_terms<T>(row + (u64)b * Fmt<T>::BB, lane, a, d, dmin, sum, msum);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); msum += __shfl_xor_sync(0xffffffffu, msum, o); }
            acc = __fadd_rn(acc, iq_block_value<T>(d, dmin, hd[b], sum, msum));
        }
        if (lane == 0) F(4)[(u64)kk * n + r] = acc;
    }
}
extern "C" __global__ void qx_moe_down_q8k_iq2_s(const u64* info) { down_q8k_t<QX_IQ2_S>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq3_s(const u64* info) { down_q8k_t<QX_IQ3_S>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq2_xxs(const u64* info) { down_q8k_t<QX_IQ2_XXS>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq2_xs(const u64* info) { down_q8k_t<QX_IQ2_XS>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq3_xxs(const u64* info) { down_q8k_t<QX_IQ3_XXS>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq1_s(const u64* info) { down_q8k_t<QX_IQ1_S>(info); }
extern "C" __global__ void qx_moe_down_q8k_q2_k(const u64* info) { down_q8k_t<QX_Q2_K>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq2_s_sel(const u64* info) { down_q8k_t<QX_IQ2_S, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq3_s_sel(const u64* info) { down_q8k_t<QX_IQ3_S, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq2_xxs_sel(const u64* info) { down_q8k_t<QX_IQ2_XXS, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq2_xs_sel(const u64* info) { down_q8k_t<QX_IQ2_XS, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq3_xxs_sel(const u64* info) { down_q8k_t<QX_IQ3_XXS, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_iq1_s_sel(const u64* info) { down_q8k_t<QX_IQ1_S, true>(info); }
extern "C" __global__ void qx_moe_down_q8k_q2_k_sel(const u64* info) { down_q8k_t<QX_Q2_K, true>(info); }

/// NVFP4 gate and up rows . Q8_K x, bit-equal to the CPU pool (qx_cpu::dot::nvfp4_q8k): per 256-chunk, lane l
/// holds 8 codes of sub-block l / 2 (even lanes the low nibbles, odd the high); the integer sub-block sums pair
/// up, the 16 values d_s * isum_s are summed by a shuffle tree in `tree16`'s pairwise order, times the
/// activation scale, chunks in order; the row's per-tensor scale last. info as gateup_q8k plus [8] the record
/// offset of the scale tail {gate, up, down}.
extern "C" __global__ void qx_moe_gateup_q8k_nvfp4(const u64* info) {
    const u32 kk = blockIdx.y;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 ff = U(5), nb = U(6) / 256;
    const u32 tok = U(3) ? kk / U(3) : 0;
    const i8* xq = reinterpret_cast<const i8*>(info[1]) + (u64)tok * U(6);
    const float* xd = CF(2) + (u64)tok * nb;
    int2 a[10];
    float dq[10];
#pragma unroll
    for (u32 b = 0; b < 10; ++b) {
        if (b < nb) {
            a[b] = *reinterpret_cast<const int2*>(xq + b * 256 + 8 * lane);
            dq[b] = xd[b];
        }
    }
    const float* sc = reinterpret_cast<const float*>(slot + info[8]);
    const int sub = (lane % 8) / 2, sh = (lane % 2) * 4;
    for (u32 rr = 0; rr < RPW; ++rr) {
        const u32 r = warp * RPW + rr;
        if (r >= 2 * ff) return;
        const u8* row = reinterpret_cast<const u8*>(slot) + (u64)r * info[7];
        float acc = 0.f;
#pragma unroll
        for (u32 b = 0; b < 10; ++b) {
            if (b >= nb) break;
            const u8* blk = row + (u64)b * 144 + (lane / 8) * 36;
            const float d = ue4m3_half(blk[sub]);
            const u8* q = blk + 4 + 8 * sub;
            int m[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) m[i] = __ldg(&FP4_VALUES2[(q[i] >> sh) & 0xf]);
            int isum = __dp4a(pack4(m[4], m[5], m[6], m[7]), a[b].y, __dp4a(pack4(m[0], m[1], m[2], m[3]), a[b].x, 0));
            isum += __shfl_xor_sync(0xffffffffu, isum, 1);
            float v = __fmul_rn(d, (float)isum);
#pragma unroll
            for (int o = 2; o <= 16; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            acc = __fadd_rn(acc, __fmul_rn(dq[b], v));
        }
        if (lane == 0) F(4)[(u64)kk * 2 * ff + r] = __fmul_rn(acc, sc[r < ff ? 0 : 1]);
    }
}

// ---- resident experts grouped by slot: each expert's weights read once for all the window's tokens using it ----

#define QX_MG 32  // selections of one expert in a window at most (one per token, TMAX tokens)
#ifndef DRPW
#define DRPW 2
#endif

/// Groups a layer's selections by slot (resident ones only): out [0] group count, [1 + g] first member of group g
/// (into the member list), [1 + AMAX + g] its member count, [1 + 2 AMAX + i] the member list (selection indices).
/// Group g's leader is its lowest selection; members are listed in selection order. info: [0] slots, [1]
/// selections, [2] out. One block of 512 threads.
extern "C" __global__ void qx_moe_groups(const u64* info) {
    __shared__ u32 lead[512], gid[512], cnt[512], start[512];
    const u64* slots = reinterpret_cast<const u64*>(info[0]);
    const u32 A = U(1), t = threadIdx.x;
    u32* out = reinterpret_cast<u32*>(info[2]);
    constexpr u32 AM = 320;
    u32 l = 0xffffffffu;
    if (t < A && slots[t] != 0) {
        l = t;
        for (u32 j = 0; j < t; ++j)
            if (slots[j] == slots[t]) { l = j; break; }
    }
    lead[t] = l;
    __syncthreads();
    if (t == 0) {
        u32 g = 0;
        for (u32 i = 0; i < A; ++i)
            if (lead[i] == i) { gid[i] = g; cnt[g] = 0; ++g; }
        for (u32 i = 0; i < A; ++i)
            if (lead[i] != 0xffffffffu) cnt[gid[lead[i]]]++;
        u32 o = 0;
        for (u32 x = 0; x < g; ++x) { start[x] = o; out[1 + x] = o; out[1 + AM + x] = cnt[x]; o += cnt[x]; cnt[x] = 0; }
        for (u32 i = 0; i < A; ++i)
            if (lead[i] != 0xffffffffu) { const u32 x = gid[lead[i]]; out[1 + 2 * AM + start[x] + cnt[x]++] = i; }
        out[0] = g;
    }
}

/// `qx_moe_gateup_q8k_nvfp4` with the selections grouped by slot (`qx_moe_groups`): a warp decodes its row's
/// codes and scales once and computes every member's dot with exactly the per-selection arithmetic, so the
/// results are bit-equal to the ungrouped kernel's. Grid (ceil(2 ff / 8), AMAX), 256 threads.
/// info: [0] slots, [1] xq, [2] xd, [3] selections per token, [4] gu out, [5] ff, [6] E, [7] row bytes,
/// [8] scale-tail offset, [9] groups.
extern "C" __global__ void qx_moe_gateup_q8k_nvfp4_g(const u64* info) {
    const u32* grp = reinterpret_cast<const u32*>(info[9]);
    const u32 g = blockIdx.y;
    if (g >= grp[0]) return;
    const u32* mem = grp + 1 + 2 * 320 + grp[1 + g];
    const u32 nm = grp[1 + 320 + g];
    const u64 slot = reinterpret_cast<const u64*>(info[0])[mem[0]];
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const u32 ff = U(5), nb = U(6) / 256;
    const u32 r = warp;
    if (r >= 2 * ff) return;
    const float* sc = reinterpret_cast<const float*>(slot + info[8]);
    const int sub = (lane % 8) / 2, sh = (lane % 2) * 4;
    const u8* row = reinterpret_cast<const u8*>(slot) + (u64)r * info[7];
    float dw[10];
    int w0[10], w1[10];
#pragma unroll
    for (u32 b = 0; b < 10; ++b) {
        if (b < nb) {
            const u8* blk = row + (u64)b * 144 + (lane / 8) * 36;
            dw[b] = ue4m3_half(blk[sub]);
            const u8* q = blk + 4 + 8 * sub;
            int m[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) m[i] = __ldg(&FP4_VALUES2[(q[i] >> sh) & 0xf]);
            w0[b] = pack4(m[0], m[1], m[2], m[3]);
            w1[b] = pack4(m[4], m[5], m[6], m[7]);
        }
    }
    const float rs = sc[r < ff ? 0 : 1];
    for (u32 j = 0; j < nm; ++j) {
        const u32 kk = mem[j];
        const u32 tok = U(3) ? kk / U(3) : 0;
        const i8* xq = reinterpret_cast<const i8*>(info[1]) + (u64)tok * U(6);
        const float* xd = CF(2) + (u64)tok * nb;
        float acc = 0.f;
#pragma unroll
        for (u32 b = 0; b < 10; ++b) {
            if (b >= nb) break;
            const int2 a = *reinterpret_cast<const int2*>(xq + b * 256 + 8 * lane);
            int isum = __dp4a(w1[b], a.y, __dp4a(w0[b], a.x, 0));
            isum += __shfl_xor_sync(0xffffffffu, isum, 1);
            float v = __fmul_rn(dw[b], (float)isum);
#pragma unroll
            for (int o = 2; o <= 16; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            acc = __fadd_rn(acc, __fmul_rn(xd[b], v));
        }
        if (lane == 0) F(4)[(u64)kk * 2 * ff + r] = __fmul_rn(acc, rs);
    }
}

/// `qx_moe_down_q80_nvfp4` grouped by slot: the block forms every member's Q8_0 h (as the ungrouped kernel
/// does for its one), each lane decodes its row block once and computes every member's value with the same
/// arithmetic and the same in-order block sum. Grid (ceil(E / (8 DRPW)), AMAX), 256 threads.
/// info as qx_moe_down_q80_nvfp4 plus [10] groups.
extern "C" __global__ void qx_moe_down_q80_nvfp4_g(const u64* info) {
    __shared__ __align__(16) i8 hq[QX_MG][1024];
    __shared__ float hd[QX_MG][32];
    const u32* grp = reinterpret_cast<const u32*>(info[10]);
    const u32 g = blockIdx.y;
    if (g >= grp[0]) return;
    const u32* mem = grp + 1 + 2 * 320 + grp[1 + g];
    const u32 nm = min(grp[1 + 320 + g], (u32)QX_MG);
    const u32 ff = U(5), n = U(6), nb = ff / 32;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[mem[0]];
    for (u32 j = 0; j < nm; ++j) {
        const float* gu = CF(1) + (u64)mem[j] * 2 * ff;
        for (u32 i0 = 0; i0 < ff; i0 += blockDim.x) {
            const u32 i = i0 + threadIdx.x;
            const float h = i < ff ? qx_silu_mul(gu[i], gu[ff + i]) : 0.f;
            const float amax = warp_max(fabsf(h));
            const float d = __fdiv_rn(amax, 127.f);
            const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
            if (i < ff) hq[j][i] = (i8)roundf(__fmul_rn(h, id));
            if (i < ff && i % 32 == 0) hd[j][i / 32] = f16_to_f32(f32_to_f16(d));
        }
    }
    __syncthreads();
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const float sd = reinterpret_cast<const float*>(slot + info[9])[2];
    const int a0 = 2 * (lane % 2);
    for (u32 rr = 0; rr < DRPW; ++rr) {
        const u32 r = warp * DRPW + rr;
        if (r >= n) return;
        float da = 0.f, db = 0.f;
        int wa[4] = {0, 0, 0, 0}, wb[4] = {0, 0, 0, 0};
        if (lane < nb) {
            const u8* blk = reinterpret_cast<const u8*>(slot) + info[7] + (u64)r * info[8] + (lane / 2) * 36;
            da = ue4m3_half(blk[a0]);
            db = ue4m3_half(blk[a0 + 1]);
            const u8* c = blk + 4 + 8 * a0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const u8 b0 = c[4 * j], b1 = c[4 * j + 1], b2 = c[4 * j + 2], b3 = c[4 * j + 3];
                wa[j] = pack4(__ldg(&FP4_VALUES2[b0 & 0xf]), __ldg(&FP4_VALUES2[b1 & 0xf]), __ldg(&FP4_VALUES2[b2 & 0xf]), __ldg(&FP4_VALUES2[b3 & 0xf]));
                wa[2 + j] = pack4(__ldg(&FP4_VALUES2[b0 >> 4]), __ldg(&FP4_VALUES2[b1 >> 4]), __ldg(&FP4_VALUES2[b2 >> 4]), __ldg(&FP4_VALUES2[b3 >> 4]));
                const u8 c0 = c[8 + 4 * j], c1 = c[9 + 4 * j], c2 = c[10 + 4 * j], c3 = c[11 + 4 * j];
                wb[j] = pack4(__ldg(&FP4_VALUES2[c0 & 0xf]), __ldg(&FP4_VALUES2[c1 & 0xf]), __ldg(&FP4_VALUES2[c2 & 0xf]), __ldg(&FP4_VALUES2[c3 & 0xf]));
                wb[2 + j] = pack4(__ldg(&FP4_VALUES2[c0 >> 4]), __ldg(&FP4_VALUES2[c1 >> 4]), __ldg(&FP4_VALUES2[c2 >> 4]), __ldg(&FP4_VALUES2[c3 >> 4]));
            }
        }
        for (u32 j = 0; j < nm; ++j) {
            float v = 0.f;
            if (lane < nb) {
                const int* p = reinterpret_cast<const int*>(hq[j] + lane * 32);
                int ia = 0, ib = 0;
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    ia = __dp4a(wa[x], p[x], ia);
                    ia = __dp4a(wa[2 + x], p[2 + x], ia);
                    ib = __dp4a(wb[x], p[4 + x], ib);
                    ib = __dp4a(wb[2 + x], p[6 + x], ib);
                }
                v = __fmul_rn(hd[j][lane], __fadd_rn(__fmul_rn(da, (float)ia), __fmul_rn(db, (float)ib)));
            }
            float acc = 0.f;
            for (u32 b = 0; b < nb; ++b) acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, v, b));
            if (lane == 0) F(4)[(u64)mem[j] * n + r] = __fmul_rn(acc, sd);
        }
    }
}

/// h = silu(gate) * up for each resident selected expert, quantized to Q8_0 as the CPU pool does: per 32,
/// d = amax / 127, q = round(h / d), d stored f16-rounded. Grid: k blocks of ff threads.
/// info: [0] slots, [1] gu, [2] hq qs out (i8, k x ff), [3] hq d out (f32, k x ff/32), [4] ff.
extern "C" __global__ void qx_moe_hq(const u64* info) {
    const u32 kk = blockIdx.x, i = threadIdx.x, ff = U(4);
    if (reinterpret_cast<const u64*>(info[0])[kk] == 0 || i >= ff) return;
    const float* gu = CF(1) + (u64)kk * 2 * ff;
    const float h = qx_silu_mul(gu[i], gu[ff + i]);
    const float amax = warp_max(fabsf(h));
    const float d = __fdiv_rn(amax, 127.f);
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    reinterpret_cast<i8*>(info[2])[(u64)kk * ff + i] = (i8)roundf(__fmul_rn(h, id));
    if (i % 32 == 0) F(3)[(u64)kk * (ff / 32) + i / 32] = f16_to_f32(f32_to_f16(d));
}

/// Down rows of the resident selected experts, integer dots, blocks summed in order: v[k][r] (unweighted).
/// Each block first forms the expert's h = silu(gate) * up quantized to Q8_0 (as the CPU pool does: per 32,
/// d = amax / 127, q = round(h / d), d f16-rounded) in shared memory. Lane b owns block b of every row, so each
/// warp keeps its h block in registers and streams DRPW rows. Grid: (ceil(E / (8 * DRPW)), k), 256 threads.
/// info: [0] slots, [1] gu, [2] unused, [3] unused, [4] v out (k x E), [5] ff, [6] E, [7] offset of the down
/// rows in the slot, [8] down row bytes.
#define DRPW 2
extern "C" __global__ void qx_moe_down_q80_iq4_nl(const u64* info) {
    __shared__ __align__(16) i8 hq[1024];
    __shared__ float hd[32];
    const u32 kk = blockIdx.y, ff = U(5), n = U(6), nb = ff / 32;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const float* gu = CF(1) + (u64)kk * 2 * ff;
    for (u32 i0 = 0; i0 < ff; i0 += blockDim.x) {
        const u32 i = i0 + threadIdx.x;
        const float h = i < ff ? qx_silu_mul(gu[i], gu[ff + i]) : 0.f;
        const float amax = warp_max(fabsf(h));
        const float d = __fdiv_rn(amax, 127.f);
        const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
        if (i < ff) hq[i] = (i8)roundf(__fmul_rn(h, id));
        if (i < ff && i % 32 == 0) hd[i / 32] = f16_to_f32(f32_to_f16(d));
    }
    __syncthreads();
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    int aq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dq = 0.f;
    if (lane < nb) {
        const int* p = reinterpret_cast<const int*>(hq + lane * 32);
#pragma unroll
        for (int j = 0; j < 8; ++j) aq[j] = p[j];
        dq = hd[lane];
    }
    for (u32 rr = 0; rr < DRPW; ++rr) {
        const u32 r = warp * DRPW + rr;
        if (r >= n) return;
        float v = 0.f;
        if (lane < nb) {
            const u8* blk = reinterpret_cast<const u8*>(slot) + info[7] + (u64)r * info[8] + lane * 18;
            const float d = ld_f16(blk);
            int isum = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const u8 b0 = blk[2 + 4 * j], b1 = blk[3 + 4 * j], b2 = blk[4 + 4 * j], b3 = blk[5 + 4 * j];
                const int lo = pack4(__ldg(&IQ4NL_VALUES[b0 & 0xf]), __ldg(&IQ4NL_VALUES[b1 & 0xf]), __ldg(&IQ4NL_VALUES[b2 & 0xf]), __ldg(&IQ4NL_VALUES[b3 & 0xf]));
                const int hi = pack4(__ldg(&IQ4NL_VALUES[b0 >> 4]), __ldg(&IQ4NL_VALUES[b1 >> 4]), __ldg(&IQ4NL_VALUES[b2 >> 4]), __ldg(&IQ4NL_VALUES[b3 >> 4]));
                isum = __dp4a(lo, aq[j], isum);
                isum = __dp4a(hi, aq[4 + j], isum);
            }
            v = __fmul_rn(__fmul_rn(d, dq), (float)isum);
        }
        float acc = 0.f;
        for (u32 b = 0; b < nb; ++b) acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, v, b));
        if (lane == 0) F(4)[(u64)kk * n + r] = acc;
    }
}

/// NVFP4 down rows . the Q8_0 h of qx_moe_down_q80_iq4_nl, bit-equal to the CPU pool (qx_cpu::dot::nvfp4_q80):
/// lane b owns Q8_0 block b = sub-blocks 2b and 2b + 1, block value dq * (d_a * isum_a + d_b * isum_b), blocks
/// summed in order, the per-tensor down scale last. info as qx_moe_down_q80_iq4_nl plus [9] the record offset
/// of the scale tail.
extern "C" __global__ void qx_moe_down_q80_nvfp4(const u64* info) {
    __shared__ __align__(16) i8 hq[1024];
    __shared__ float hd[32];
    const u32 kk = blockIdx.y, ff = U(5), n = U(6), nb = ff / 32;
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    if (slot == 0) return;
    const float* gu = CF(1) + (u64)kk * 2 * ff;
    for (u32 i0 = 0; i0 < ff; i0 += blockDim.x) {
        const u32 i = i0 + threadIdx.x;
        const float h = i < ff ? qx_silu_mul(gu[i], gu[ff + i]) : 0.f;
        const float amax = warp_max(fabsf(h));
        const float d = __fdiv_rn(amax, 127.f);
        const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
        if (i < ff) hq[i] = (i8)roundf(__fmul_rn(h, id));
        if (i < ff && i % 32 == 0) hd[i / 32] = f16_to_f32(f32_to_f16(d));
    }
    __syncthreads();
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    int aq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dq = 0.f;
    if (lane < nb) {
        const int* p = reinterpret_cast<const int*>(hq + lane * 32);
#pragma unroll
        for (int j = 0; j < 8; ++j) aq[j] = p[j];
        dq = hd[lane];
    }
    const float sd = reinterpret_cast<const float*>(slot + info[9])[2];
    const int a0 = 2 * (lane % 2);
    for (u32 rr = 0; rr < DRPW; ++rr) {
        const u32 r = warp * DRPW + rr;
        if (r >= n) return;
        float v = 0.f;
        if (lane < nb) {
            const u8* blk = reinterpret_cast<const u8*>(slot) + info[7] + (u64)r * info[8] + (lane / 2) * 36;
            const float da = ue4m3_half(blk[a0]), db = ue4m3_half(blk[a0 + 1]);
            const u8* c = blk + 4 + 8 * a0;
            int ia = 0, ib = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const u8 b0 = c[4 * j], b1 = c[4 * j + 1], b2 = c[4 * j + 2], b3 = c[4 * j + 3];
                ia = __dp4a(pack4(__ldg(&FP4_VALUES2[b0 & 0xf]), __ldg(&FP4_VALUES2[b1 & 0xf]), __ldg(&FP4_VALUES2[b2 & 0xf]), __ldg(&FP4_VALUES2[b3 & 0xf])), aq[j], ia);
                ia = __dp4a(pack4(__ldg(&FP4_VALUES2[b0 >> 4]), __ldg(&FP4_VALUES2[b1 >> 4]), __ldg(&FP4_VALUES2[b2 >> 4]), __ldg(&FP4_VALUES2[b3 >> 4])), aq[2 + j], ia);
                const u8 c0 = c[8 + 4 * j], c1 = c[9 + 4 * j], c2 = c[10 + 4 * j], c3 = c[11 + 4 * j];
                ib = __dp4a(pack4(__ldg(&FP4_VALUES2[c0 & 0xf]), __ldg(&FP4_VALUES2[c1 & 0xf]), __ldg(&FP4_VALUES2[c2 & 0xf]), __ldg(&FP4_VALUES2[c3 & 0xf])), aq[4 + j], ib);
                ib = __dp4a(pack4(__ldg(&FP4_VALUES2[c0 >> 4]), __ldg(&FP4_VALUES2[c1 >> 4]), __ldg(&FP4_VALUES2[c2 >> 4]), __ldg(&FP4_VALUES2[c3 >> 4])), aq[6 + j], ib);
            }
            v = __fmul_rn(dq, __fadd_rn(__fmul_rn(da, (float)ia), __fmul_rn(db, (float)ib)));
        }
        float acc = 0.f;
        for (u32 b = 0; b < nb; ++b) acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, v, b));
        if (lane == 0) F(4)[(u64)kk * n + r] = __fmul_rn(acc, sd);
    }
}

// ---- the MTP drafter's experts: all resident, Q8_0, f32 activations ----

extern "C" __global__ void qx_moe_gateup_q8_0(const u64* info) { moe_gateup<QX_Q8_0>(info); }
extern "C" __global__ void qx_moe_gateup_iq4_nl(const u64* info) { moe_gateup<QX_IQ4_NL>(info); }

/// Down projection of resident experts in any decodable format with f32 activations: v[a][r] = down_e[r] .
/// (silu(g) * u). Grid: (ceil(E / 8), selections). info as qx_moe_down_iq4_nl.
template <int T> QX_DEV void moe_down(const u64* info) {
    __shared__ float h[1024];
    const u32 kk = blockIdx.y, ff = U(6), n = U(7);
    const u64 slot = reinterpret_cast<const u64*>(info[0])[kk];
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (slot == 0) return;
    const float* gu = CF(4) + (u64)kk * 2 * ff;
    for (u32 i = threadIdx.x; i < ff; i += blockDim.x) h[i] = qx_silu_mul(gu[i], gu[ff + i]);
    __syncthreads();
    if (warp >= n) return;
    const u8* row = reinterpret_cast<const u8*>(slot) + info[8] + (u64)warp * info[9];
    float acc = 0.f;
    for (u32 c = 0; c * 256 < ff; ++c) {
        const u32 b = c * 256 + 8 * lane;
        if (b < ff) {
            float w[8];
            decode8<T>(row, c, lane, w);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc += w[i] * h[b + i];
        }
    }
    acc = warp_sum(acc);
    if constexpr (T == QX_NVFP4) acc *= reinterpret_cast<const float*>(slot + info[10])[2];
    if (lane == 0) F(5)[(u64)kk * n + warp] = acc;
}
extern "C" __global__ void qx_moe_down_q8_0(const u64* info) { moe_down<QX_Q8_0>(info); }
extern "C" __global__ void qx_moe_down_nvfp4(const u64* info) { moe_down<QX_NVFP4>(info); }

/// Slot pointers when every expert is resident: slots[a] = base + ids[a] * stride. info: [0] ids, [1] slots out,
/// [2] base, [3] stride, [4] selections.
extern "C" __global__ void qx_moe_slots_all(const u64* info) {
    const u32 a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < U(4)) reinterpret_cast<u64*>(info[1])[a] = info[2] + (u64)reinterpret_cast<const i32*>(info[0])[a] * info[3];
}

// ---- the chunk path (prefill in chunks of many rows, experts streamed once per layer) ----

/// Spins until the slot word at [0] carries generation [1] in its top 16 bits (the driver writes `address |
/// gen << 48` behind the record's copy, or at once for a VRAM-resident record; no zeroing between jobs, so no
/// race between a reset and the copy streams' writes). One thread.
extern "C" __global__ void qx_moe_wait_word(const u64* info) {
    const u64* w = reinterpret_cast<const u64*>(info[0]);
    const u64 gen = info[1];
    while ((ld_acquire_sys(w) >> 48) != gen) {
    }
}

/// Stores [1] at the host-mapped word [0] with release semantics (the GPU's progress mark for the driver's slot
/// reuse). One thread.
extern "C" __global__ void qx_moe_signal(const u64* info) {
    st_release_sys(reinterpret_cast<u64*>(info[0]), info[1]);
}

/// Q8_K of one row of x per block, the post kernel's rule (per 256: the first index of the largest magnitude sets
/// the scale). info: [0] x (rows x E f32), [1] xq out (rows x E i8), [2] xd out (rows x E/256 f32), [3] E.
/// Grid (rows), 256 threads.
extern "C" __global__ void qx_moe_q8k_rows(const u64* info) {
    const u32 E = U(3), tk = blockIdx.x;
    const float* x = CF(0) + (u64)tk * E;
    const u32 w = threadIdx.x / 32, l = threadIdx.x % 32;
    for (u32 b = w; b < E / 256; b += blockDim.x / 32) {
        float v[8], am = 0.f;
        u32 ai = 0xffffffffu;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = x[b * 256 + 8 * l + i];
            const float a = fabsf(v[i]);
            if (a > am) { am = a; ai = 8 * l + i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float oa = __shfl_xor_sync(0xffffffffu, am, o);
            const u32 oi = __shfl_xor_sync(0xffffffffu, ai, o);
            if (oa > am || (oa == am && oi < ai)) { am = oa; ai = oi; }
        }
        i8* qs = reinterpret_cast<i8*>(info[1]) + (u64)tk * E + b * 256 + 8 * l;
        float* dd = reinterpret_cast<float*>(info[2]) + (u64)tk * (E / 256);
        if (am == 0.f) {
#pragma unroll
            for (int i = 0; i < 8; ++i) qs[i] = 0;
            if (l == 0) dd[b] = 0.f;
        } else {
            const float iscale = __fdiv_rn(-127.f, x[b * 256 + ai]);
#pragma unroll
            for (int i = 0; i < 8; ++i) qs[i] = (i8)min(127, __float2int_rn(__fmul_rn(iscale, v[i])));
            if (l == 0) dd[b] = __fdiv_rn(1.f, iscale);
        }
    }
}
