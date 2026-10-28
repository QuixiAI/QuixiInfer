// Matrix products for many tokens at once (prefill): Y[t][n] = sum_k X[t][k] * W[n][k], f32 accumulation on
// bf16 tensor cores (mma m16n8k16). W rows are any decodable format; they are decoded 8 elements at a time into
// bf16 shared-memory tiles, X is rounded to bf16 as it is staged.
//
// Tile: 64 tokens x 64 output rows per block, k steps of 32; 4 warps, each 32 tokens x 32 rows (2 x 4 mma tiles).
// info: [0] X (f32, T x K, row stride = K), [1] W, [2] Y (f32, T x N, row stride info[8]), [3] T, [4] N, [5] K,
// [6] W type (QxType), [7] W row stride in bytes, [8] Y row stride (floats), [9] mode (0 store, 1 add).
#include "qx_quant.cuh"
#include "qx_block.cuh"

#define BT 64
#define BN 64
#define BK 32
// rows of one expert per grouped tile (qxp_moe_group cuts tiles of this size)
#define GROUP_TILE 64

QX_DEV void dec_any(u32 ty, const u8* row, u32 cols, u32 base, float* w) {
    const u32 c = base / 256, l = (base % 256) / 8;
    switch (ty) {
        case QX_F32: decode8<QX_F32>(row, c, l, w); break;
        case QX_BF16: decode8<QX_BF16>(row, c, l, w); break;
        case QX_Q8_0: decode8<QX_Q8_0>(row, c, l, w); break;
        case QX_Q6_K: decode8<QX_Q6_K>(row, c, l, w); break;
        case QX_IQ4_NL: decode8<QX_IQ4_NL>(row, c, l, w); break;
        case QX_IQ3_S: decode8<QX_IQ3_S>(row, c, l, w); break;
        case QX_IQ2_S: decode8<QX_IQ2_S>(row, c, l, w); break;
        case QX_NVFP4: decode8<QX_NVFP4>(row, c, l, w); break;
        case QX_Q6_K_P: decode8_q6k_p(row, cols / 256, c, l, w); break;
        case QX_Q8_0_P: decode8_q80_p(row, cols / 32, c, l, w); break;
        default: for (int i = 0; i < 8; ++i) w[i] = 0.f;
    }
}

/// The decoder for one compile-time weight type (dec_any's per-call switch keeps every decoder live in the
/// inner loop).
template <int TY>
QX_DEV void dec_t(const u8* row, u32 cols, u32 base, float* w) {
    const u32 c = base / 256, l = (base % 256) / 8;
    if constexpr (TY == QX_Q6_K_P) decode8_q6k_p(row, cols / 256, c, l, w);
    else if constexpr (TY == QX_Q8_0_P) decode8_q80_p(row, cols / 32, c, l, w);
    else decode8<(QxType)TY>(row, c, l, w);
}

#define QX_TYPE_CASES(X) X(QX_F32) X(QX_BF16) X(QX_Q8_0) X(QX_Q6_K) X(QX_IQ4_NL) X(QX_IQ3_S) X(QX_IQ2_S) X(QX_Q6_K_P) X(QX_Q8_0_P) X(QX_NVFP4)

QX_DEV u32 pack_bf16(float a, float b) { return (u32)f32_to_bf16(a) | ((u32)f32_to_bf16(b) << 16); }

QX_DEV void mma_bf16(float* c, const u32* a, const u32* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/// The tile product for rows of W at `wrow(n)` (n < N): a BTxBN output tile, 4 warps in a WMxWN grid, each warp
/// MI x 16 tokens by NI x 8 rows. Shared by the dense and the grouped-expert entries.
template <int TY, int BTT, int WM, int WN, int MI, int NI, typename WRow, typename XRow, typename Out>
QX_DEV void gemm_tile(u32* smem, u32 t0, u32 n0, u32 T, u32 N, u32 K, WRow wrow, XRow xrow, Out out) {
    constexpr int BNN = WN * NI * 8;
    static_assert(WM * MI * 16 == BTT, "tile rows");
    auto xs = reinterpret_cast<u32(*)[BK / 2 + 4]>(smem);
    auto ws = reinterpret_cast<u32(*)[BK / 2 + 4]>(smem + BTT * (BK / 2 + 4));
    const u32 tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const u32 wt = (warp / WN) * MI * 16, wn = (warp % WN) * NI * 8;
    float acc[MI][NI][4] = {};
    for (u32 k0 = 0; k0 < K; k0 += BK) {
        for (u32 i = tid; i < BTT * BK / 8; i += blockDim.x) {
            const u32 r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
            const u32 t = t0 + r;
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (t < T && k0 + c8 < K) {
                const float* xp = xrow(t) + k0 + c8;
#pragma unroll
                for (int j = 0; j < 8; ++j) v[j] = xp[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) xs[r][c8 / 2 + j] = pack_bf16(v[2 * j], v[2 * j + 1]);
        }
        for (u32 i = tid; i < BNN * BK / 8; i += blockDim.x) {
            const u32 r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
            const u32 n = n0 + r;
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (n < N && k0 + c8 < K) dec_t<TY>(wrow(n), K, k0 + c8, v);
#pragma unroll
            for (int j = 0; j < 4; ++j) ws[r][c8 / 2 + j] = pack_bf16(v[2 * j], v[2 * j + 1]);
        }
        __syncthreads();
#pragma unroll
        for (u32 kk = 0; kk < BK; kk += 16) {
            const u32 g = lane / 4, q = lane % 4;
            u32 a[MI][4], b[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const u32 r = wt + mi * 16 + g;
                a[mi][0] = xs[r][(kk + 2 * q) / 2];
                a[mi][1] = xs[r + 8][(kk + 2 * q) / 2];
                a[mi][2] = xs[r][(kk + 8 + 2 * q) / 2];
                a[mi][3] = xs[r + 8][(kk + 8 + 2 * q) / 2];
            }
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                const u32 c = wn + ni * 8 + g;
                b[ni][0] = ws[c][(kk + 2 * q) / 2];
                b[ni][1] = ws[c][(kk + 8 + 2 * q) / 2];
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], a[mi], b[ni]);
        }
        __syncthreads();
    }
    const u32 g = lane / 4, q = lane % 4;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const u32 t = t0 + wt + mi * 16 + g + (e >= 2 ? 8 : 0);
                const u32 n = n0 + wn + ni * 8 + 2 * q + (e & 1);
                if (t < T && n < N) out(t, n, acc[mi][ni][e]);
            }
}

QX_DEV void ldsm_x4(u32* r, const void* p) {
    const u32 a = (u32)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/// gemm_tile with two shared-memory stages (the next k step's X and decoded W are loaded and stored while the
/// current one's mmas run, one barrier per step) and ldmatrix fragment loads. The same bf16 operands meet in the
/// same mma order, so the results are gemm_tile's bits. NT: threads per block.
template <int TY, int BTT, int WM, int WN, int MI, int NI, int NT, typename WRow, typename XRow, typename Out>
QX_DEV void gemm_tile2(u32* smem, u32 t0, u32 n0, u32 T, u32 N, u32 K, WRow wrow, XRow xrow, Out out) {
    constexpr int BNN = WN * NI * 8, LD = BK / 2 + 4;
    constexpr int XI = BTT * BK / 8 / NT, WI = BNN * BK / 8 / NT;
    static_assert(WM * MI * 16 == BTT && XI * NT == BTT * BK / 8 && WI * NT == BNN * BK / 8 && NI % 2 == 0, "tile shape");
    auto xs = [&](int st) { return reinterpret_cast<u32(*)[LD]>(smem + st * (BTT + BNN) * LD); };
    auto ws = [&](int st) { return reinterpret_cast<u32(*)[LD]>(smem + st * (BTT + BNN) * LD + BTT * LD); };
    const u32 tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const u32 wt = (warp / WN) * MI * 16, wn = (warp % WN) * NI * 8;
    float acc[MI][NI][4] = {};
    u32 xr[XI][4], wr[WI][4];
    auto load = [&](u32 k0) {
#pragma unroll
        for (int j = 0; j < XI; ++j) {
            const u32 i = tid + j * NT, r = i / (BK / 8), c8 = (i % (BK / 8)) * 8, t = t0 + r;
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (t < T && k0 + c8 < K) {
                const float4* xp = reinterpret_cast<const float4*>(xrow(t) + k0 + c8);
                const float4 a = xp[0], b = xp[1];
                v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) xr[j][q] = pack_bf16(v[2 * q], v[2 * q + 1]);
        }
#pragma unroll
        for (int j = 0; j < WI; ++j) {
            const u32 i = tid + j * NT, r = i / (BK / 8), c8 = (i % (BK / 8)) * 8, n = n0 + r;
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (n < N && k0 + c8 < K) dec_t<TY>(wrow(n), K, k0 + c8, v);
#pragma unroll
            for (int q = 0; q < 4; ++q) wr[j][q] = pack_bf16(v[2 * q], v[2 * q + 1]);
        }
    };
    auto store = [&](int st) {
#pragma unroll
        for (int j = 0; j < XI; ++j) {
            const u32 i = tid + j * NT, r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
            *reinterpret_cast<uint4*>(&xs(st)[r][c8 / 2]) = make_uint4(xr[j][0], xr[j][1], xr[j][2], xr[j][3]);
        }
#pragma unroll
        for (int j = 0; j < WI; ++j) {
            const u32 i = tid + j * NT, r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
            *reinterpret_cast<uint4*>(&ws(st)[r][c8 / 2]) = make_uint4(wr[j][0], wr[j][1], wr[j][2], wr[j][3]);
        }
    };
    load(0);
    store(0);
    __syncthreads();
    int st = 0;
    for (u32 k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) load(k0 + BK);
        auto x = xs(st);
        auto w = ws(st);
#pragma unroll
        for (u32 kk = 0; kk < BK; kk += 16) {
            u32 a[MI][4], b[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) ldsm_x4(a[mi], &x[wt + mi * 16 + lane % 16][(kk + (lane / 16) * 8) / 2]);
#pragma unroll
            for (int ni = 0; ni < NI; ni += 2) {
                u32 r4[4];
                ldsm_x4(r4, &w[wn + ni * 8 + (lane / 16) * 8 + lane % 8][(kk + ((lane / 8) % 2) * 8) / 2]);
                b[ni][0] = r4[0]; b[ni][1] = r4[1]; b[ni + 1][0] = r4[2]; b[ni + 1][1] = r4[3];
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], a[mi], b[ni]);
        }
        if (more) store(st ^ 1);
        __syncthreads();
        st ^= 1;
    }
    const u32 g = lane / 4, q = lane % 4;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const u32 t = t0 + wt + mi * 16 + g + (e >= 2 ? 8 : 0);
                const u32 n = n0 + wn + ni * 8 + 2 * q + (e & 1);
                if (t < T && n < N) out(t, n, acc[mi][ni][e]);
            }
}

template <int TY>
QX_DEV void gemm_dense(u32* smem, const u64* info) {
    const u32 T = (u32)info[3], N = (u32)info[4], K = (u32)info[5];
    const u64 wstride = info[7], ystride = info[8];
    const u8* W = reinterpret_cast<const u8*>(info[1]);
    const float* X = reinterpret_cast<const float*>(info[0]);
    float* Y = reinterpret_cast<float*>(info[2]);
    const bool add = info[9] == 1;
    gemm_tile<TY, 128, 4, 2, 2, 4>(smem, blockIdx.y * 128, blockIdx.x * 64, T, N, K,
              [&](u32 n) { return W + (u64)n * wstride; },
              [&](u32 t) { return X + (u64)t * K; },
              [&](u32 t, u32 n, float v) {
                  float* p = Y + (u64)t * ystride + n;
                  *p = add ? *p + v : v;
              });
}

template <int TY>
QX_DEV void gemm_dense2(u32* smem, const u64* info) {
    const u32 T = (u32)info[3], N = (u32)info[4], K = (u32)info[5];
    const u64 wstride = info[7], ystride = info[8];
    const u8* W = reinterpret_cast<const u8*>(info[1]);
    const float* X = reinterpret_cast<const float*>(info[0]);
    float* Y = reinterpret_cast<float*>(info[2]);
    const bool add = info[9] == 1;
    gemm_tile2<TY, 128, 4, 2, 2, 4, 256>(smem, blockIdx.y * 128, blockIdx.x * 64, T, N, K,
              [&](u32 n) { return W + (u64)n * wstride; },
              [&](u32 t) { return X + (u64)t * K; },
              [&](u32 t, u32 n, float v) {
                  float* p = Y + (u64)t * ystride + n;
                  *p = add ? *p + v : v;
              });
}

/// qx_gemm on gemm_tile2 (same info, grid and bits).
extern "C" __global__ void __launch_bounds__(256) qx_gemm2(const u64* info) {
    __shared__ __align__(16) u32 smem[2 * (128 + 64) * (BK / 2 + 4)];
    switch ((u32)info[6]) {
#define X(t) case t: gemm_dense2<t>(smem, info); break;
        QX_TYPE_CASES(X)
#undef X
    }
}

/// Dense tile product: 128 tokens x 64 rows per block, 8 warps. Grid (ceil(N / 64), ceil(T / 128)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qx_gemm(const u64* info) {
    __shared__ u32 smem[(128 + 64) * (BK / 2 + 4)];
    switch ((u32)info[6]) {
#define X(t) case t: gemm_dense<t>(smem, info); break;
        QX_TYPE_CASES(X)
#undef X
    }
}

template <int TY>
QX_DEV void gemm_grouped(u32* smem, const u64* info) {
    const u32 tile = blockIdx.x;
    if (tile >= *reinterpret_cast<const u32*>(info[6])) return;
    const u32* sorted = reinterpret_cast<const u32*>(info[3]);
    const u32 e = reinterpret_cast<const u32*>(info[4])[tile], r0 = reinterpret_cast<const u32*>(info[5])[tile];
    if (e < (u32)info[16] || e >= (u32)info[17]) return;
    const u32 rend = reinterpret_cast<const u32*>(info[7])[e];
    const u32 N = (u32)info[8], K = (u32)info[9], kpt = (u32)info[15];
    const u8* W = reinterpret_cast<const u8*>(reinterpret_cast<const u64*>(info[1])[e]) + info[12];
    const u64 wrb = info[11];
    const float* X = reinterpret_cast<const float*>(info[0]);
    float* Y = reinterpret_cast<float*>(info[2]);
    const bool gather = info[13] == 1, scatter = info[14] == 1;
    const u32 nrows = min((u32)GROUP_TILE, rend - r0);
    // NVFP4 records end in the expert's per-tensor scales {gate, up, down}: info[18] their offset (0: none),
    // info[19] which one this product gets
    const float sc = info[18] ? reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[1])[e] + info[18])[info[19]] : 1.f;
    gemm_tile<TY, GROUP_TILE, 2, 2, 2, 4>(smem, 0, blockIdx.y * 64, nrows, N, K,
              [&](u32 n) { return W + (u64)n * wrb; },
              [&](u32 t) { const u32 r = r0 + t; return X + (u64)(gather ? sorted[r] / kpt : r) * K; },
              [&](u32 t, u32 n, float v) { const u32 r = r0 + t; Y[(u64)(scatter ? sorted[r] : r) * N + n] = sc * v; });
}

template <int TY>
QX_DEV void gemm_grouped2(u32* smem, const u64* info) {
    const u32 tile = blockIdx.x;
    if (tile >= *reinterpret_cast<const u32*>(info[6])) return;
    const u32* sorted = reinterpret_cast<const u32*>(info[3]);
    const u32 e = reinterpret_cast<const u32*>(info[4])[tile], r0 = reinterpret_cast<const u32*>(info[5])[tile];
    if (e < (u32)info[16] || e >= (u32)info[17]) return;
    const u32 rend = reinterpret_cast<const u32*>(info[7])[e];
    const u32 N = (u32)info[8], K = (u32)info[9], kpt = (u32)info[15];
    const u8* W = reinterpret_cast<const u8*>(reinterpret_cast<const u64*>(info[1])[e]) + info[12];
    const u64 wrb = info[11];
    const float* X = reinterpret_cast<const float*>(info[0]);
    float* Y = reinterpret_cast<float*>(info[2]);
    const bool gather = info[13] == 1, scatter = info[14] == 1;
    const u32 nrows = min((u32)GROUP_TILE, rend - r0);
    // NVFP4 records end in the expert's per-tensor scales {gate, up, down}: info[18] their offset (0: none),
    // info[19] which one this product gets
    const float sc = info[18] ? reinterpret_cast<const float*>(reinterpret_cast<const u64*>(info[1])[e] + info[18])[info[19]] : 1.f;
    gemm_tile2<TY, GROUP_TILE, 2, 2, 2, 4, 128>(smem, 0, blockIdx.y * 64, nrows, N, K,
              [&](u32 n) { return W + (u64)n * wrb; },
              [&](u32 t) { const u32 r = r0 + t; return X + (u64)(gather ? sorted[r] / kpt : r) * K; },
              [&](u32 t, u32 n, float v) { const u32 r = r0 + t; Y[(u64)(scatter ? sorted[r] : r) * N + n] = sc * v; });
}

/// Grouped expert GEMM for a prefill chunk. Assignments (token t, slot k) are sorted by expert; tile i covers
/// sorted rows [r0, r0 + GROUP_TILE) of expert e (clipped to the expert's rows). Expert e's weights start at
/// ptrs[e]; its rows used here start `w_off` bytes in.
/// X row of sorted row r: `xsrc + (gather ? token(r) : r) * K` where token(r) = sorted[r] / k_per_token.
/// Output: `y + (scatter ? sorted[r] : r) * N + n`.
/// info: [0] xsrc, [1] ptrs (u64 per expert), [2] y, [3] sorted assignments (u32), [4] tile expert (u32),
/// [5] tile first row (u32), [6] tile count ptr (u32), [7] expert row ends (u32, exclusive, per expert),
/// [8] N, [9] K, [10] W type, [11] W row bytes, [12] w_off, [13] gather (1) or contiguous rows (0),
/// [14] scatter (1) or contiguous (0), [15] k per token, [16] e_lo, [17] e_hi: only tiles of experts in
/// [e_lo, e_hi) run, [18] offset of the per-tensor scale tail in the expert's record (0: none), [19] index of
/// the scale (0 gate, 1 up, 2 down). Grid (max tiles, ceil(N / 64)), 128 threads.
extern "C" __global__ void __launch_bounds__(128) qx_gemm_grouped(const u64* info) {
    __shared__ u32 smem[(GROUP_TILE + 64) * (BK / 2 + 4)];
    switch ((u32)info[10]) {
        case QX_IQ2_S: gemm_grouped<QX_IQ2_S>(smem, info); break;
        case QX_IQ3_S: gemm_grouped<QX_IQ3_S>(smem, info); break;
        case QX_IQ4_NL: gemm_grouped<QX_IQ4_NL>(smem, info); break;
        case QX_Q8_0: gemm_grouped<QX_Q8_0>(smem, info); break;
        case QX_Q6_K: gemm_grouped<QX_Q6_K>(smem, info); break;
        case QX_NVFP4: gemm_grouped<QX_NVFP4>(smem, info); break;
    }
}

/// qx_gemm_grouped on gemm_tile2 (same info, grid and bits).
extern "C" __global__ void __launch_bounds__(128) qx_gemm_grouped2(const u64* info) {
    __shared__ __align__(16) u32 smem[2 * (GROUP_TILE + 64) * (BK / 2 + 4)];
    switch ((u32)info[10]) {
        case QX_IQ2_S: gemm_grouped2<QX_IQ2_S>(smem, info); break;
        case QX_IQ3_S: gemm_grouped2<QX_IQ3_S>(smem, info); break;
        case QX_IQ4_NL: gemm_grouped2<QX_IQ4_NL>(smem, info); break;
        case QX_Q8_0: gemm_grouped2<QX_Q8_0>(smem, info); break;
        case QX_Q6_K: gemm_grouped2<QX_Q6_K>(smem, info); break;
        case QX_NVFP4: gemm_grouped2<QX_NVFP4>(smem, info); break;
    }
}

/// f32 matrix product for F32 weights (router, HC inject, GDN gates, shared-expert gate): Y[t][n] = X[t] . W[n].
/// info: [0] X (T x K), [1] W (N x K f32), [2] Y, [3] T, [4] N, [5] K, [6] Y row stride. Block tile 64 tokens x
/// 64 rows, k steps of 16; each thread 4 x 4 outputs. Grid (ceil(N/64), ceil(T/64)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qx_gemm_f32(const u64* info) {
    __shared__ __align__(16) float xs[16][64 + 4], ws[16][64 + 4];
    const u32 T = (u32)info[3], N = (u32)info[4], K = (u32)info[5];
    const float* X = reinterpret_cast<const float*>(info[0]);
    const float* Wt = reinterpret_cast<const float*>(info[1]);
    float* Y = reinterpret_cast<float*>(info[2]);
    const u32 t0 = blockIdx.y * 64, n0 = blockIdx.x * 64, tid = threadIdx.x;
    const u32 ty = tid / 16, tx = tid % 16;
    // loads: 64 rows x 16 k = 1024 floats per operand, 4 per thread (row tid / 4, k 4 * (tid % 4))
    const u32 lr = tid / 4, lk = (tid % 4) * 4;
    float acc[4][4] = {};
    for (u32 k0 = 0; k0 < K; k0 += 16) {
        {
            const u32 t = t0 + lr, n = n0 + lr;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const u32 kk = k0 + lk + j;
                xs[lk + j][lr] = (t < T && kk < K) ? X[(u64)t * K + kk] : 0.f;
                ws[lk + j][lr] = (n < N && kk < K) ? Wt[(u64)n * K + kk] : 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (u32 kk = 0; kk < 16; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const u32 t = t0 + ty * 4 + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const u32 n = n0 + tx * 4 + j;
            if (n < N) Y[(u64)t * info[6] + n] = acc[i][j];
        }
    }
}

/// qx_gemm_f32 for few output rows (N < 128): one warp per (token, row) dot product. info as qx_gemm_f32.
/// Grid (ceil(T / 8), N), 256 threads.
extern "C" __global__ void __launch_bounds__(256) qx_gemm_f32_narrow(const u64* info) {
    const u32 T = (u32)info[3], K = (u32)info[5], warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const u32 t = blockIdx.x * 8 + warp, n = blockIdx.y;
    if (t >= T) return;
    const float4* x = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(info[0]) + (u64)t * K);
    const float4* w = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(info[1]) + (u64)n * K);
    float s = 0.f;
    for (u32 i = lane; i < K / 4; i += 32) {
        const float4 a = x[i], b = w[i];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) reinterpret_cast<float*>(info[2])[(u64)t * info[6] + n] = s;
}

// ---- Tensor-core matrix-vector products for decode windows (up to 16 tokens) ----
//
// Several matrices sharing one input, as qx_mgemv. A segment is 12 u64 words: [0] W, [1] y, [2] rows, [3] cols,
// [4] type, [5] blocks per 16-row tile (k split), [6] first block, [7] row stride in bytes (0: dense), [8] partials
// (k split x 16 x 16 floats per tile), [9] tile counters (u32 per tile, zero between launches). The ks blocks of a
// tile and their 8 warps split K: warp w of block b takes every (8 ks)-th 32-column step from 8 b + w. Warps sum in
// warp order, then blocks in block order (the last block of a tile to finish adds the others' partials and resets
// the counter), so the result is the same every launch. W is the mma A
// operand (16 rows x 16 k), the tokens are N (n8 tiles, as many as the token count needs, up to 4). The k slots of
// an mma are a fixed permutation of 16 of the step's columns, so each lane decodes 8 consecutive columns of its two
// rows straight into its A fragments: lane (g, q) holds columns 8q .. 8q + 7 of rows g and g + 8; slots 2q, 2q + 1,
// 2q + 8, 2q + 9 of the first mma take columns 8q .. 8q + 3, the second mma's take 8q + 4 .. 8q + 7.
// A token's result depends on its own x only (no arithmetic crosses mma columns), and on nothing about the other
// tokens: any window gives each token the bits it gets alone.
// info: [0] x, [1] segments, [2] segment count, [3] x stride per token (floats), [4] x argument (as qx_mgemv),
// [5] tokens (1 to 16).

// Each warp stages a 16-row x 128-column chunk of W as bf16 in shared memory: half-warp h decodes row 2i + h, lanes
// 8-column pieces of it, so every load reads one row's contiguous bytes. The fragments then come out of the tile
// (row stride TMV_LD words: conflict-free) in the plain mma layout.
#define TMV_LD 68
#ifndef TMV_Q8_1_BLOCKS
#define TMV_Q8_1_BLOCKS 2
#endif
#ifndef TMV_Q8_BLOCKS
#define TMV_Q8_BLOCKS 2
#endif
#ifndef TMV_GRP
#define TMV_GRP 4
#endif

/// The end of a tile's product: the ws warps of the tile, then the ks blocks (the last to finish sums them in
/// block order), then y[n][r] for the tile's rows and tokens.
QX_DEV void tmv_finish(const float (&acc)[4][4], u32 T, u32 nt, u32 rows, u32 r0, float (*part)[4][4][32], float* y, u32 kb, u32 ks, u32 ws, float* partials, u32* counter) {
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4, wsub = warp % ws;
    float out[4][4] = {};
    if (ws > 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[warp][j][e][lane] = acc[j][e];
        __syncthreads();
        if (wsub != 0) return;
        for (u32 j = 0; j < nt; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                for (u32 w = 0; w < ws; ++w) out[j][e] += part[warp + w][j][e][lane];
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) out[j][e] = acc[j][e];
    }
    if (ks > 1) {
        // this block's partial; the last block of the tile to finish sums them all in block order
        __shared__ u32 s_last[8];
        for (u32 j = 0; j < nt; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) partials[((u64)kb * 16 + j * 4 + e) * 32 + lane] = out[j][e];
        __threadfence();
        __syncwarp();
        if (lane == 0) s_last[warp] = atomicAdd(counter, 1u) == ks - 1;
        __syncwarp();
        if (!s_last[warp]) return;
        __threadfence();
        for (u32 j = 0; j < nt; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = 0.f;
                for (u32 b = 0; b < ks; ++b) v += reinterpret_cast<volatile float*>(partials)[((u64)b * 16 + j * 4 + e) * 32 + lane];
                out[j][e] = v;
            }
        if (lane == 0) *counter = 0;
    }
    for (u32 j = 0; j < nt; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const u32 r = r0 + g + (e >= 2 ? 8 : 0), n = 8 * j + 2 * q + (e & 1);
            if (r < rows && n < T) y[(u64)n * rows + r] = out[j][e];
        }
}

template <int TY>
QX_DEV void tmv_rows(const u8* W, u64 stride, u32 rows, u32 K, u32 r0, const u32* xb, u32 T, float (*part)[4][4][32], u32 (*tile)[TMV_LD], float* y, u32 kb, u32 ks, u32 ws, float* partials, u32* counter) {
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4, h = lane / 16, l16 = lane % 16;
    const u32 nt = (T + 7) / 8;  // token tiles of 8 (up to 4: 32 tokens)
    float acc[4][4] = {};
    // this warp's share of K: the ws warps of its tile in each of the tile's ks blocks
    const u32 wsub = warp % ws, wi = kb * ws + wsub, nw = ws * ks, chunks = K / 128;
    // the k16 blocks of one staged chunk (n16 of them): each token tile's x pairs, then the mmas
    auto mmas = [&](u32 c0, u32 n16) {
        __syncwarp();
#pragma unroll 8
        for (u32 k16 = 0; k16 < n16; ++k16) {
            const u32 a[4] = {tile[g][k16 * 8 + q], tile[g + 8][k16 * 8 + q], tile[g][k16 * 8 + 4 + q], tile[g + 8][k16 * 8 + 4 + q]};
            for (u32 j = 0; j < nt; ++j) {
                // x as prepared by qx_tmv_prep: bf16 pairs, token n's row K / 2 words, zero past T
                const u32* xr = xb + (u64)(8 * j + g) * (K / 2) + (c0 + k16 * 16 + 2 * q) / 2;
                const u32 b[2] = {xr[0], xr[4]};
                mma_bf16(acc[j], a, b);
            }
        }
        __syncwarp();
    };
    for (u32 c = wi; c < chunks; c += nw) {
        // groups of TMV_GRP row pieces, each group's loads all in flight before any is stored
#pragma unroll
        for (int i0 = 0; i0 < 8; i0 += TMV_GRP) {
            float w[TMV_GRP][8];
#pragma unroll
            for (int i = 0; i < TMV_GRP; ++i) {
                const u32 r = r0 + 2 * (i0 + i) + h;
#pragma unroll
                for (int e = 0; e < 8; ++e) w[i][e] = 0.f;
                if (r < rows) dec_t<TY>(W + (u64)r * stride, K, c * 128 + 8 * l16, w[i]);
            }
#pragma unroll
            for (int i = 0; i < TMV_GRP; ++i)
                *reinterpret_cast<uint4*>(&tile[2 * (i0 + i) + h][4 * l16]) = make_uint4(pack_bf16(w[i][0], w[i][1]), pack_bf16(w[i][2], w[i][3]), pack_bf16(w[i][4], w[i][5]), pack_bf16(w[i][6], w[i][7]));
        }
        mmas(c * 128, 8);
    }
    // the last K % 128 columns (a multiple of 32): one more chunk, short, for the warp next in turn
    const u32 tail = K % 128;
    if (tail && wi == chunks % nw) {
        const u32 c0 = chunks * 128;
        for (u32 i = 0; i < 8; ++i) {
            const u32 r = r0 + 2 * i + h;
            float w8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (r < rows && 8 * l16 < tail) dec_t<TY>(W + (u64)r * stride, K, c0 + 8 * l16, w8);
            u32* t = &tile[2 * i + h][4 * l16];
            t[0] = pack_bf16(w8[0], w8[1]); t[1] = pack_bf16(w8[2], w8[3]); t[2] = pack_bf16(w8[4], w8[5]); t[3] = pack_bf16(w8[6], w8[7]);
        }
        mmas(c0, tail / 16);
    }
    tmv_finish(acc, T, nt, rows, r0, part, y, kb, ks, ws, partials, counter);
}

QX_DEV void mma_s8(int* c, const u32* a, const u32* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/// `tmv_rows` for planar Q8_0 rows against int8 inputs (`qx_mgemv_q8x`'s blocks of 32, scales transposed: block b's
/// at xd + 32 b): one int8 mma per Q8_0 block, its quants loaded straight into the fragments, then the block's two
/// scales. The mma's k order is permuted the same way on both sides (lane q holds bytes 8q .. 8q + 7 of a block),
/// which leaves each integer dot product exact, and every load is 8 contiguous bytes of one row. NT token tiles of 8.
template <int NT>
QX_DEV void tmv_rows_q8(const u8* W, u64 stride, u32 rows, u32 K, u32 r0, const i8* __restrict__ xq, const float* __restrict__ xd, u32 T, float (*part)[4][4][32], float* y, u32 kb, u32 ks, u32 ws, float* partials, u32* counter) {
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const u32 nb = K / 32;
    float acc[4][4] = {};
    const u32 wsub = warp % ws, wi = kb * ws + wsub, nw = ws * ks, chunks = K / 128;
    const u8* wa = W + (u64)min(r0 + g, rows - 1) * stride;
    const u8* wb = W + (u64)min(r0 + g + 8, rows - 1) * stride;
    const u16* da = reinterpret_cast<const u16*>(wa + (u64)nb * 32);
    const u16* db = reinterpret_cast<const u16*>(wb + (u64)nb * 32);
    // one chunk's weights (4 blocks of 32 for rows g and g + 8), loaded a chunk ahead of its mmas and streamed past
    // L1, so the inputs every warp of the block reads stay there
    struct Wc {
        int2 qa[4], qb[4];
        uint2 sa, sb;  // the 4 blocks' f16 scales of each row
    };
    auto load = [&](u32 b0, u32 n, Wc& w) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (i < (int)n) {
                w.qa[i] = __ldcs(reinterpret_cast<const int2*>(wa + (b0 + i) * 32 + 8 * q));
                w.qb[i] = __ldcs(reinterpret_cast<const int2*>(wb + (b0 + i) * 32 + 8 * q));
            }
        if (n == 4) {
            w.sa = __ldcs(reinterpret_cast<const uint2*>(da + b0));
            w.sb = __ldcs(reinterpret_cast<const uint2*>(db + b0));
        } else {
            u16 a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
            for (u32 i = 0; i < n; ++i) a[i] = da[b0 + i], b[i] = db[b0 + i];
            w.sa = make_uint2(a[0] | (u32)a[1] << 16, a[2] | (u32)a[3] << 16);
            w.sb = make_uint2(b[0] | (u32)b[1] << 16, b[2] | (u32)b[3] << 16);
        }
    };
    auto half = [](uint2 v, int i) { return (u16)((i < 2 ? v.x : v.y) >> (16 * (i & 1))); };
    // a chunk's inputs: block i's int8 pieces and scale pairs for each token tile
    struct Xc {
        int2 v[4][NT];
        float2 s[4][NT];
    };
    auto xload = [&](u32 b0, u32 n, Xc& x) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
                if (i < (int)n) {
                    x.v[i][j] = __ldg(reinterpret_cast<const int2*>(xq + (u64)(8 * j + g) * K + (b0 + i) * 32 + 8 * q));
                    x.s[i][j] = __ldg(reinterpret_cast<const float2*>(xd + (u64)(b0 + i) * 32 + 8 * j + 2 * q));
                }
    };
    auto mmas = [&](u32 n, const Wc& w, const Xc& x) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i >= (int)n) break;
            const u32 a[4] = {(u32)w.qa[i].x, (u32)w.qb[i].x, (u32)w.qa[i].y, (u32)w.qb[i].y};
            const float sa = f16_to_f32(half(w.sa, i)), sb = f16_to_f32(half(w.sb, i));
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const u32 b[2] = {(u32)x.v[i][j].x, (u32)x.v[i][j].y};
                int c[4] = {0, 0, 0, 0};
                mma_s8(c, a, b);
                acc[j][0] += sa * x.s[i][j].x * (float)c[0];
                acc[j][1] += sa * x.s[i][j].y * (float)c[1];
                acc[j][2] += sb * x.s[i][j].x * (float)c[2];
                acc[j][3] += sb * x.s[i][j].y * (float)c[3];
            }
        }
    };
    // up to 2 token tiles the next chunk's inputs come with its weights; wider ones load theirs at the chunk
    constexpr bool XPRE = NT <= 2;
    if (wi < chunks) {
        Wc cur;
        Xc xc;
        load(wi * 4, 4, cur);
        if constexpr (XPRE) xload(wi * 4, 4, xc);
        for (u32 c = wi; c < chunks; c += nw) {
            Wc nxt;
            Xc xn;
            if (c + nw < chunks) {
                load((c + nw) * 4, 4, nxt);
                if constexpr (XPRE) xload((c + nw) * 4, 4, xn);
            }
            if constexpr (!XPRE) xload(c * 4, 4, xc);
            mmas(4, cur, xc);
            cur = nxt;
            if constexpr (XPRE) xc = xn;
        }
    }
    // the last K % 128 columns (a multiple of 32): one more chunk, short, for the warp next in turn
    const u32 tail = K % 128;
    if (tail && wi == chunks % nw) {
        Wc w;
        Xc x;
        load(chunks * 4, tail / 32, w);
        xload(chunks * 4, tail / 32, x);
        mmas(tail / 32, w, x);
    }
    tmv_finish(acc, T, NT, rows, r0, part, y, kb, ks, ws, partials, counter);
}

template <int Q8> QX_DEV void tmv(const u64* info) {
    __shared__ float part[8][4][4][32];
    // the int8 entry stages nothing: without the bf16 tiles more blocks fit on an SM
    __shared__ u32 tiles[Q8 > 0 ? 1 : 8][16][TMV_LD];
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 12 + 6]) ++s;
    const u64* d = segs + s * 12;
    const u32 rows = (u32)d[2], K = (u32)d[3], ty = (u32)d[4], ks = (u32)d[5], ws = (u32)d[10];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, K);
    // block bi: tiles (bi / ks) * (8 / ws) .. + 8 / ws, its warps ws to a tile; k part bi % ks
    const u32 bi = blockIdx.x - (u32)d[6], kb = bi % ks, tile = (bi / ks) * (8 / ws) + threadIdx.x / 32 / ws;
    const u32 r0 = tile * 16;
    float* partials = reinterpret_cast<float*>(d[8]) + (u64)tile * ks * 512;
    u32* counter = reinterpret_cast<u32*>(d[9]) + tile;
    const u8* W = reinterpret_cast<const u8*>(d[0]);
    float* y = reinterpret_cast<float*>(d[1]);
    const u32* xb = reinterpret_cast<const u32*>(info[0]);
    if constexpr (Q8 > 0) {
        tmv_rows_q8<Q8>(W, stride, rows, K, r0, reinterpret_cast<const i8*>(info[6]), reinterpret_cast<const float*>(info[7]), (u32)info[5], part, y, kb, ks, ws, partials, counter);
        return;
    }
    if constexpr (Q8 == 0) {
        switch (ty) {
#define C(t) case t: tmv_rows<t>(W, stride, rows, K, r0, xb, (u32)info[5], part, tiles[threadIdx.x / 32], y, kb, ks, ws, partials, counter); return;
            C(QX_F32) C(QX_BF16) C(QX_Q8_0) C(QX_Q6_K) C(QX_Q6_K_P) C(QX_Q8_0_P)
#undef C
        }
    }
}

/// info: [0] x prepared by qx_tmv_prep (bf16, 32 token rows of K), [1] segments, [2] segment count, [5] tokens
/// (1 to 32), [6] the int8 inputs (`qx_mgemv_q8x`, token n at + n * K) or 0, [7] their scales (transposed: token n
/// of block b at + 32 b + n):
/// with them, planar Q8_0 segments take integer products.
extern "C" __global__ void __launch_bounds__(256, 2) qx_tmv(const u64* info) { tmv<0>(info); }
/// `qx_tmv` for planar Q8_0 segments with the int8 inputs (info [6], [7]); one entry per count of 8-token tiles, so
/// each has its own registers.
extern "C" __global__ void __launch_bounds__(256, TMV_Q8_1_BLOCKS) qx_tmv_q8_1(const u64* info) { tmv<1>(info); }
extern "C" __global__ void __launch_bounds__(256, TMV_Q8_BLOCKS) qx_tmv_q8_2(const u64* info) { tmv<2>(info); }
extern "C" __global__ void __launch_bounds__(256, TMV_Q8_BLOCKS) qx_tmv_q8_3(const u64* info) { tmv<3>(info); }
extern "C" __global__ void __launch_bounds__(256, TMV_Q8_BLOCKS) qx_tmv_q8_4(const u64* info) { tmv<4>(info); }

/// The products' input for qx_tmv: xb[n][k] = bf16 of x[n][k] transformed as qx_mgemv's modes do, for n < 32 (zero
/// from T on). info: [0] x, [1] x stride per token, [2] argument, [3] mode (0 plain, 1 silu(x * s), 2 silu(x) * x2),
/// [4] T, [5] K, [6] out (u32 pairs). Grid ceil(32 * K / 2 / 256).
extern "C" __global__ void qx_tmv_prep(const u64* info) {
    const u32 K = (u32)info[5], T = (u32)info[4], mode = (u32)info[3];
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 32ull * K / 2) return;
    const u32 n = (u32)(i / (K / 2)), k = (u32)(i % (K / 2)) * 2;
    float v0 = 0.f, v1 = 0.f;
    if (n < T) {
        const float* x = reinterpret_cast<const float*>(info[0]) + (u64)n * info[1];
        v0 = x[k];
        v1 = x[k + 1];
        if (mode == 1) {
            const float sc = __uint_as_float((u32)info[2]);
            v0 = siluf_(v0 * sc);
            v1 = siluf_(v1 * sc);
        } else if (mode == 2) {
            const float* x2 = reinterpret_cast<const float*>(info[2]) + (u64)n * info[1];
            v0 = siluf_(v0) * x2[k];
            v1 = siluf_(v1) * x2[k + 1];
        }
    }
    reinterpret_cast<u32*>(info[6])[i] = pack_bf16(v0, v1);
}
