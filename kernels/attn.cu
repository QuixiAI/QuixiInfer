// Decode attention over the QSA-selected cells, flash-decoding style.
//
// Phase 1: block (split s, KV head g) takes cells [32 s, 32 s + 32) of the list and every query head of the
// group (heads g*G .. g*G+G, G = heads / kv heads), writing an unnormalised partial output with its running
// max and sum. Phase 2 merges the splits per head by log-sum-exp. The cell count comes from device memory,
// so a captured graph covers every position; splits past the end write nothing and are skipped by phase 2.
#include "qx_block.cuh"
#include "qx_sys.cuh"
#include "qx_kv.cuh"

#define F(i) reinterpret_cast<float*>(info[i])
#define CF(i) reinterpret_cast<const float*>(info[i])
#define U(i) ((u32)info[i])

#define SPLIT 32
#define MAXG 16

// ---- FP8 KV: E4M3 values, QX_KV_SG f32 scales per (cell, kv head) (one per 32 dims); the full cache in mapped host memory, the most
// recent W cells also in a VRAM ring at slot cell % W ----

// A layer's FP8 KV table: u64[slots][8] = ring K, ring V, ring K scales, ring V scales, host K, host V, host K
// scales, host V scales of each sequence slot.

/// Quantizes each token's new K and V and appends them at its position, to its slot's VRAM ring and host pages.
/// One block per kv head, tensor and token (grid kv x 2 x tokens: K then V), dim/2 threads, two elements each.
/// info: [0] k (tokens x kv x dim f32), [1] v, [2] KV table, [3] slots (u32 x tokens), [10]
/// positions (u32 x tokens), [11] window W, [12] kv, [13] dim.
extern "C" __global__ void qx_kv_append_fp8(const u64* info) {
    const u32 g = blockIdx.x, which = blockIdx.y, nkv = U(12), d = U(13), t = threadIdx.x;
    const u32 pos = reinterpret_cast<const u32*>(info[10])[blockIdx.z], W = U(11);
    const u64* tab = reinterpret_cast<const u64*>(info[2]) + 8ull * reinterpret_cast<const u32*>(info[3])[blockIdx.z];
    const float* src = CF(which) + ((u64)blockIdx.z * nkv + g) * d;
    const float a = src[2 * t], b = src[2 * t + 1];
    // thread t holds dims 2t, 2t+1: 16 threads per 32-dim group, each with its own scale
    const float amax = group16_max(fmaxf(fabsf(a), fabsf(b)));
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const u16 q = f32x2_to_e4m3x2(a / scale, b / scale);
    const u64 rcell = (u64)(pos % W) * nkv + g;
    float* hs;
    u8* hrow = kv_page_row(tab, which == 1, pos, g, nkv, d, hs);
    // Plain stores: the host copy is read only by later kernels, after this one has completed.
    reinterpret_cast<u16*>(tab[which])[rcell * (d / 2) + t] = q;
    reinterpret_cast<u16*>(hrow)[t] = q;
    if (t % 16 == 0) {
        reinterpret_cast<float*>(tab[2 + which])[rcell * QX_KV_SG + t / 16] = scale;
        hs[t / 16] = scale;
    }
}

/// Refills a slot's VRAM ring from its host pages for cells lo .. hi (after a restored prefix): grid (cells, 2) blocks
/// of dim/2 threads. info: [0] this layer's table entry for the slot, [1] lo, [2] hi, [3] window W, [4] kv, [5] dim.
extern "C" __global__ void qx_kv_ring_fill(const u64* info) {
    const u64* tab = reinterpret_cast<const u64*>(info[0]);
    const u32 cell = (u32)info[1] + blockIdx.x, which = blockIdx.y, W = U(3), nkv = U(4), d = U(5), t = threadIdx.x;
    if (cell >= (u32)info[2]) return;
    for (u32 g = 0; g < nkv; ++g) {
        float* hs;
        const u8* hrow = kv_page_row(tab, which == 1, cell, g, nkv, d, hs);
        const u64 rcell = (u64)(cell % W) * nkv + g;
        reinterpret_cast<u16*>(tab[which])[rcell * (d / 2) + t] = reinterpret_cast<const u16*>(hrow)[t];
        if (t < QX_KV_SG) reinterpret_cast<float*>(tab[2 + which])[rcell * QX_KV_SG + t] = hs[t];
    }
}

struct Fp8Kv {
    /// A layer's table entry for the slot (see above).
    const u64* tab;
    u32 pos, W, nkv, d;
    /// The cell's K (or V) bytes and its QX_KV_SG scales, from the VRAM ring when the cell is recent enough, else the
    /// host pages. The ring is trusted QX_WMAX cells short of its size: a speculative window may have written up
    /// to that many cells past the last kept one, over ring slots of older cells. Both copies hold the same bytes.
    QX_DEV const u8* row(bool v, u32 cell, u32 g, const float*& s) const {
        if (cell + W > pos + QX_WMAX) {
            const u64 i = (u64)(cell % W) * nkv + g;
            s = reinterpret_cast<const float*>(tab[2 + (v ? 1 : 0)]) + i * QX_KV_SG;
            return reinterpret_cast<const u8*>(tab[v ? 1 : 0]) + i * d;
        }
        float* hs;
        const u8* r = kv_page_row(tab, v, cell, g, nkv, d, hs);
        s = hs;
        return r;
    }
};

/// Copies cells 0 .. n-1 of a slot's layer KV (VRAM ring for recent cells, host pages for the rest) into a VRAM
/// mirror laid out as a ring of `W2` >= n + QX_WMAX + 1 cells, so every cell index maps to itself, and writes the
/// mirror's table entry (the slot's page table words copied). The chunk's attention then reads only VRAM: before,
/// a 1024-query chunk read each selected cell from the host pages once per query, ~2 GB per layer over PCIe.
/// info: [0] the slot's table entry, [1] mirror table entry out (8 u64), [2] mirror K, [3] V, [4] K scales,
/// [5] V scales, [6] n cells, [7] latest cell, [8] ring window W, [9] kv heads, [10] dim. Grid (n, 2), dim/2 threads.
extern "C" __global__ void qxp_kv_mirror(const u64* info) {
    const u64* tab = reinterpret_cast<const u64*>(info[0]);
    const u32 cell = blockIdx.x, which = blockIdx.y, nkv = U(9), d = U(10), t = threadIdx.x;
    if (cell == 0 && which == 0 && t < 8) {
        u64* out = reinterpret_cast<u64*>(info[1]);
        out[t] = t < 4 ? info[2 + t] : tab[t];
    }
    if (cell >= U(6)) return;
    const Fp8Kv kv{tab, U(7), U(8), nkv, d};
    for (u32 g = 0; g < nkv; ++g) {
        const float* s;
        const u8* row = kv.row(which == 1, cell, g, s);
        const u64 i = (u64)cell * nkv + g;
        reinterpret_cast<u16*>(info[2 + which])[i * (d / 2) + t] = reinterpret_cast<const u16*>(row)[t];
        if (t < QX_KV_SG) reinterpret_cast<float*>(info[4 + which])[i * QX_KV_SG + t] = s[t];
    }
}

/// Eight E4M3 values (dims 8j .. 8j+7 of a row, within one 32-dim group) times that group's scale `s`.
QX_DEV void e4m3x8(u64 b, float s, float* o) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        e4m3x2_to_f32x2((u16)((b >> (16 * i)) & 0xffff), lo, hi);
        o[2 * i] = lo * s;
        o[2 * i + 1] = hi * s;
    }
}

/// info: [0] q (heads x dim), [1] K, [2] V (f32 [cell][kv][dim]), [3] cells, [4] count ptr, [5] partial out
/// (splits x heads x dim), [6] partial max (splits x heads), [7] partial sum, [8] kv heads, [9] heads,
/// [10] dim (256), [11] scale bits. Grid (max splits, kv heads), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qx_attn_split(const u64* info) {
    __shared__ float qs[MAXG][256];
    __shared__ float sc[MAXG][SPLIT];
    __shared__ float pm[MAXG], ps[MAXG];
    const u32 s = blockIdx.x, g = blockIdx.y, n = *reinterpret_cast<const u32*>(info[4]);
    const u32 c0 = s * SPLIT;
    if (c0 >= n) return;
    const u32 nc = min((u32)SPLIT, n - c0);
    const u32 nkv = U(8), nh = U(9), d = U(10), G = nh / nkv;
    const u32* cells = reinterpret_cast<const u32*>(info[3]);
    const float scale = __uint_as_float(U(11));
    for (u32 i = threadIdx.x; i < G * d; i += blockDim.x) qs[i / d][i % d] = CF(0)[(u64)(g * G + i / d) * d + i % d];
    __syncthreads();
    // scores: warp w takes cells w, w+8, ...; each lane holds 8 of the 256 key elements
    const u32 w = threadIdx.x / 32, l = threadIdx.x % 32;
    for (u32 c = w; c < nc; c += 8) {
        const float* k = CF(1) + ((u64)cells[c0 + c] * nkv + g) * d + 8 * l;
        const float4 k0 = *reinterpret_cast<const float4*>(k), k1 = *reinterpret_cast<const float4*>(k + 4);
        for (u32 h = 0; h < G; ++h) {
            const float* q = &qs[h][8 * l];
            float t = q[0] * k0.x + q[1] * k0.y + q[2] * k0.z + q[3] * k0.w + q[4] * k1.x + q[5] * k1.y + q[6] * k1.z + q[7] * k1.w;
            t = warp_sum(t);
            if (l == 0) sc[h][c] = t * scale;
        }
    }
    __syncthreads();
    // per-head max and exponentials over this split; one warp per head
    for (u32 h = w; h < G; h += 8) {
        float m = l < nc ? sc[h][l] : -3.402823466e38f;
        m = warp_max(m);
        const float e = l < nc ? __expf(sc[h][l] - m) : 0.f;
        if (l < nc) sc[h][l] = e;
        const float z = warp_sum(e);
        if (l == 0) { pm[h] = m; ps[h] = z; }
    }
    __syncthreads();
    // values: thread t owns dim t for every head of the group
    const u32 t = threadIdx.x;
    float acc[MAXG];
    for (u32 h = 0; h < G; ++h) acc[h] = 0.f;
    for (u32 c = 0; c < nc; ++c) {
        const float v = CF(2)[((u64)cells[c0 + c] * nkv + g) * d + t];
        for (u32 h = 0; h < G; ++h) acc[h] += sc[h][c] * v;
    }
    for (u32 h = 0; h < G; ++h) {
        const u32 hh = g * G + h;
        F(5)[((u64)s * nh + hh) * d + t] = acc[h];
        if (t == 0) {
            F(6)[(u64)s * nh + hh] = pm[h];
            F(7)[(u64)s * nh + hh] = ps[h];
        }
    }
}

/// Merges the splits: out[h] = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) z_s. info: [0] partial out,
/// [1] partial max, [2] partial sum, [3] count ptr, [4] out (heads x dim), [5] heads, [6] dim.
/// Grid: heads blocks of dim threads.
extern "C" __global__ void qx_attn_merge(const u64* info) {
    const u32 h = blockIdx.x, t = threadIdx.x, nh = U(5), d = U(6);
    const u32 n = *reinterpret_cast<const u32*>(info[3]);
    const u32 ns = (n + SPLIT - 1) / SPLIT;
    float M = -3.402823466e38f;
    for (u32 s = 0; s < ns; ++s) M = fmaxf(M, CF(1)[(u64)s * nh + h]);
    float num = 0.f, den = 0.f;
    for (u32 s = 0; s < ns; ++s) {
        const float f = __expf(CF(1)[(u64)s * nh + h] - M);
        num += f * CF(0)[((u64)s * nh + h) * d + t];
        den += f * CF(2)[(u64)s * nh + h];
    }
    F(4)[(u64)h * d + t] = num / den;
}

/// qx_attn_split_fp8 for every query of a window: grid (splits, kv, T). Query t reads q row t, its own cell list
/// (row t of cells, stride info[22]) and count, and its slot's KV; partial outputs gain a token dimension.
/// info as qx_attn_split_fp8 with [4] counts (T), [12] KV table, [13] slots (u32 x T), [20]
/// positions (u32 x T), [22] cells stride. A query trusts its slot's ring QX_WMAX cells short of its own
/// position: the segment's later tokens are at most that far ahead.
extern "C" __global__ void __launch_bounds__(256) qxp_attn_split_fp8(const u64* info) {
    __shared__ float qs[MAXG][256];
    __shared__ float sc[MAXG][SPLIT];
    __shared__ float pm[MAXG], ps[MAXG];
    __shared__ float vs[SPLIT][QX_KV_SG];
    __shared__ const u8* vrow[SPLIT];
    const u32 s = blockIdx.x, g = blockIdx.y, tq = blockIdx.z;
    const u32 n = reinterpret_cast<const u32*>(info[4])[tq];
    const u32 c0 = s * SPLIT;
    if (c0 >= n) return;
    const u32 nc = min((u32)SPLIT, n - c0);
    const u32 nkv = U(8), nh = U(9), d = U(10), G = nh / nkv;
    const u32* cells = reinterpret_cast<const u32*>(info[3]) + (u64)tq * info[22];
    const float scale = __uint_as_float(U(11));
    const u32 latest = reinterpret_cast<const u32*>(info[20])[tq];
    const u64* tab = reinterpret_cast<const u64*>(info[12]) + 8ull * reinterpret_cast<const u32*>(info[13])[tq];
    const Fp8Kv kv{tab, latest, U(21), nkv, d};
    const float* qrow = CF(0) + (u64)tq * nh * d;
    for (u32 i = threadIdx.x; i < G * d; i += blockDim.x) qs[i / d][i % d] = qrow[(u64)(g * G + i / d) * d + i % d];
    __syncthreads();
    const u32 w = threadIdx.x / 32, l = threadIdx.x % 32;
    for (u32 c = w; c < nc; c += 8) {
        const float* ks;
        const u8* kr = kv.row(false, cells[c0 + c], g, ks);
        float k8[8];
        e4m3x8(*reinterpret_cast<const u64*>(kr + 8 * l), ks[l / 4], k8);
        for (u32 h = 0; h < G; ++h) {
            const float* q = &qs[h][8 * l];
            float t = q[0] * k8[0] + q[1] * k8[1] + q[2] * k8[2] + q[3] * k8[3] + q[4] * k8[4] + q[5] * k8[5] + q[6] * k8[6] + q[7] * k8[7];
            t = warp_sum(t);
            if (l == 0) sc[h][c] = t * scale;
        }
        const float* vsc;
        const u8* vr = kv.row(true, cells[c0 + c], g, vsc);
        if (l == 0) vrow[c] = vr;
        if (l < QX_KV_SG) vs[c][l] = vsc[l];
    }
    __syncthreads();
    for (u32 h = w; h < G; h += 8) {
        float m = l < nc ? sc[h][l] : -3.402823466e38f;
        m = warp_max(m);
        const float e = l < nc ? __expf(sc[h][l] - m) : 0.f;
        if (l < nc) sc[h][l] = e;
        const float z = warp_sum(e);
        if (l == 0) { pm[h] = m; ps[h] = z; }
    }
    __syncthreads();
    const u32 t = threadIdx.x, half = t / 128, tt = t % 128;
    float acc0[MAXG], acc1[MAXG];
    for (u32 h = 0; h < G; ++h) acc0[h] = acc1[h] = 0.f;
    if (tt < d / 2) {
        for (u32 c = half; c < nc; c += 2) {
            float v0, v1;
            e4m3x2_to_f32x2(*reinterpret_cast<const u16*>(vrow[c] + 2 * tt), v0, v1);
            const float vsg = vs[c][tt / 16];
            v0 *= vsg;
            v1 *= vsg;
            for (u32 h = 0; h < G; ++h) {
                acc0[h] += sc[h][c] * v0;
                acc1[h] += sc[h][c] * v1;
            }
        }
    }
    __syncthreads();
    float* red = &qs[0][0];
    if (half == 1 && tt < d / 2)
        for (u32 h = 0; h < G; ++h) {
            red[(h * 128 + tt) * 2] = acc0[h];
            red[(h * 128 + tt) * 2 + 1] = acc1[h];
        }
    __syncthreads();
    const u64 ms = gridDim.x;  // splits
    if (half == 0 && tt < d / 2)
        for (u32 h = 0; h < G; ++h) {
            const u64 hh = g * G + h;
            const u64 base = (((u64)tq * ms + s) * nh + hh) * d;
            F(5)[base + 2 * tt] = acc0[h] + red[(h * 128 + tt) * 2];
            F(5)[base + 2 * tt + 1] = acc1[h] + red[(h * 128 + tt) * 2 + 1];
        }
    if (t < G) {
        F(6)[((u64)tq * ms + s) * nh + g * G + t] = pm[t];
        F(7)[((u64)tq * ms + s) * nh + g * G + t] = ps[t];
    }
}

/// Merges each query's splits. info: [0] partial out, [1] partial max, [2] partial sum, [3] counts (T),
/// [4] out (T x heads x dim), [5] heads, [6] dim, [7] splits per query. Grid (heads, T), dim threads.
extern "C" __global__ void qxp_attn_merge(const u64* info) {
    const u32 h = blockIdx.x, tq = blockIdx.y, t = threadIdx.x, nh = U(5), d = U(6), ms = U(7);
    const u32 n = reinterpret_cast<const u32*>(info[3])[tq];
    const u32 ns = (n + SPLIT - 1) / SPLIT;
    const u64 qb = (u64)tq * ms;
    float M = -3.402823466e38f;
    for (u32 s = 0; s < ns; ++s) M = fmaxf(M, CF(1)[(qb + s) * nh + h]);
    float num = 0.f, den = 0.f;
    for (u32 s = 0; s < ns; ++s) {
        const float f = __expf(CF(1)[(qb + s) * nh + h] - M);
        num += f * CF(0)[((qb + s) * nh + h) * d + t];
        den += f * CF(2)[(qb + s) * nh + h];
    }
    F(4)[((u64)tq * nh + h) * d + t] = num / den;
}

/// Prefill attention over the FP8 cache without splits: block (query t, KV head g) runs an online softmax over
/// all of query t's selected cells, 256 at a time, for the G query heads of the group. Scores take one cell
/// per thread (no reductions), values one pair of dims per thread for half of the heads.
/// info: [0] q (T x heads x dim), [1] cells (T x stride), [2] counts (T), [3] out (T x heads x dim),
/// [4] kv heads, [5] heads, [6] dim (256), [7] scale bits, [8] this layer's table entry for the slot, [16] p0 ptr,
/// [17] window W, [18] cells stride, [19] T - 1. Grid (T, kv heads), 256 threads.
template <int G>
QX_DEV void attn_fp8_full(const u64* info) {
    __shared__ __align__(16) float qs[G][256];
    __shared__ float sc[G][256];
    __shared__ const u8* vrow[256];
    __shared__ float vsc[256][QX_KV_SG];
    __shared__ float mrun[G], lrun[G], corr[G];
    const u32 tq = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const u32 nkv = U(4), nh = U(5), d = U(6);
    const float scale = __uint_as_float(U(7));
    const u32 n = reinterpret_cast<const u32*>(info[2])[tq];
    const u32* cells = reinterpret_cast<const u32*>(info[1]) + (u64)tq * info[18];
    const u32 latest = *reinterpret_cast<const u32*>(info[16]) + U(19);
    const Fp8Kv kv{reinterpret_cast<const u64*>(info[8]), latest, U(17), nkv, d};
    const float* qrow = CF(0) + ((u64)tq * nh + (u64)g * G) * d;
    for (u32 i = tid; i < G * 256; i += 256) qs[i / 256][i % 256] = qrow[i] * scale;
    if (tid < G) { mrun[tid] = -3.402823466e38f; lrun[tid] = 0.f; }
    const u32 half = tid / 128, tt = tid % 128;
    constexpr int GH = G / 2;
    float acc0[GH], acc1[GH];
#pragma unroll
    for (int j = 0; j < GH; ++j) acc0[j] = acc1[j] = 0.f;
    __syncthreads();
    for (u32 c0 = 0; c0 < n; c0 += 256) {
        const u32 nc = min(256u, n - c0);
        if (tid < nc) {
            const u32 cell = cells[c0 + tid];
            const float* ks;
            const u8* kr = kv.row(false, cell, g, ks);
            float s[G];
#pragma unroll
            for (int h = 0; h < G; ++h) s[h] = 0.f;
            for (u32 dd = 0; dd < 256; dd += 8) {
                float k8[8];
                e4m3x8(*reinterpret_cast<const u64*>(kr + dd), ks[dd / 32], k8);
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 a = *reinterpret_cast<const float4*>(&qs[h][dd]);
                    const float4 b = *reinterpret_cast<const float4*>(&qs[h][dd + 4]);
                    s[h] += a.x * k8[0] + a.y * k8[1] + a.z * k8[2] + a.w * k8[3] + b.x * k8[4] + b.y * k8[5] + b.z * k8[6] + b.w * k8[7];
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) sc[h][tid] = s[h];
            const float* vs_;
            vrow[tid] = kv.row(true, cell, g, vs_);
#pragma unroll
            for (int j = 0; j < QX_KV_SG; ++j) vsc[tid][j] = vs_[j];
        }
        __syncthreads();
        for (u32 h = warp; h < G; h += 8) {
            float m = -3.402823466e38f;
            for (u32 c = lane; c < nc; c += 32) m = fmaxf(m, sc[h][c]);
            m = warp_max(m);
            const float mo = mrun[h], mn = fmaxf(mo, m);
            float l = 0.f;
            for (u32 c = lane; c < nc; c += 32) {
                const float e = __expf(sc[h][c] - mn);
                sc[h][c] = e;
                l += e;
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float cf = __expf(mo - mn);
                corr[h] = cf;
                mrun[h] = mn;
                lrun[h] = lrun[h] * cf + l;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < GH; ++j) {
            const float cf = corr[half + 2 * j];
            acc0[j] *= cf;
            acc1[j] *= cf;
        }
        for (u32 c = 0; c < nc; ++c) {
            float v0, v1;
            e4m3x2_to_f32x2(*reinterpret_cast<const u16*>(vrow[c] + 2 * tt), v0, v1);
            const float vsg = vsc[c][tt / 16];
            v0 *= vsg;
            v1 *= vsg;
#pragma unroll
            for (int j = 0; j < GH; ++j) {
                const float p = sc[half + 2 * j][c];
                acc0[j] += p * v0;
                acc1[j] += p * v1;
            }
        }
        __syncthreads();
    }
    float* o = F(3) + ((u64)tq * nh + (u64)g * G) * d;
#pragma unroll
    for (int j = 0; j < GH; ++j) {
        const u32 h = half + 2 * j;
        const float inv = 1.f / lrun[h];
        o[(u64)h * d + 2 * tt] = acc0[j] * inv;
        o[(u64)h * d + 2 * tt + 1] = acc1[j] * inv;
    }
}

// One entry point per heads-per-kv-head count: a kernel's static shared memory is the sum over the functions it
// instantiates, and the three together (with the V scales in shared memory) exceed the 99 KiB limit.
extern "C" __global__ void __launch_bounds__(256) qxp_attn_fp8_g8(const u64* info) {
    if (U(6) != 256 || U(5) / U(4) != 8) __trap();
    attn_fp8_full<8>(info);
}
extern "C" __global__ void __launch_bounds__(256) qxp_attn_fp8_g12(const u64* info) {
    if (U(6) != 256 || U(5) / U(4) != 12) __trap();
    attn_fp8_full<12>(info);
}
extern "C" __global__ void __launch_bounds__(256) qxp_attn_fp8_g16(const u64* info) {
    if (U(6) != 256 || U(5) / U(4) != 16) __trap();
    attn_fp8_full<16>(info);
}
