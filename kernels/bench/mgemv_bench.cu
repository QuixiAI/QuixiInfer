// The dense one-launch GEMV's micro-benchmark: one planar-Q8_0 segment (rows x cols), a 128-token int8 slab,
// the engine's table and info words, each entry timed and its output compared bit for bit against the current
// one (qx_mgemv_t8z). Build and run (seconds, no model):
//   /usr/local/cuda-13.3/bin/nvcc -arch=sm_120a -std=c++17 -O3 -lineinfo -I kernels/include -o ~/qx-bins/mgemv_bench kernels/bench/mgemv_bench.cu
//   ~/qx-bins/mgemv_bench [rows=5120] [cols=4096] [tokens=128]
#define QX_MGEMV_BENCH
#include "../mgemv.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>


/// One row per lane, the 32 lane positions of the reference's split-K mapping walked inside the thread, the
/// warp_sum tree folded with a stack in the tree's leaf order (lane 0's result: leaves in bit-reversed lane order,
/// a complete binary tree over them), the 8 warps = the reference's 8 step classes summed in order at the end.
/// Grid (ceil(rows / 32) per segment, groups), 256 threads. Planar Q8_0 split-K segments only (the bench).
QX_DEV u32 brev5(u32 i) { return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4); }
template <int NT> QX_DEV void mgemv_zs(const u64* info) {
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 8 + 6]) ++s;
    const u64* d = segs + s * 8;
    const u32 rows = (u32)d[2], cols = (u32)d[3], ty = (u32)d[4];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, cols);
    const u32 blk = blockIdx.x - (u32)d[6];
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* y = reinterpret_cast<float*>(d[1]);
    const u32 z = blockIdx.y;
    const u64 z0 = (u64)z * NT;
    const u32 r = blk * 32 + lane;
    const bool valid = r < rows;
    const u32 nb = cols / 32;
    const u8* row = reinterpret_cast<const u8*>(d[0]) + (u64)(valid ? r : rows - 1) * stride;
    const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
    const i8* xq[NT];
    const float* xd[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        xq[n] = reinterpret_cast<const i8*>(info[7]) + (z0 + n) * info[9];
        xd[n] = reinterpret_cast<const float*>(info[8]) + (z0 + n) * (info[9] / 32);
    }
    __shared__ float part[8][32][NT];
    float pend[6][NT];
#pragma unroll
    for (u32 i = 0; i < 32; ++i) {
        const u32 L = brev5(i);
        // the lane-L chain over this warp's steps: step0 = warp, dstep = 8, lanes = 32 (row_partial_q8's loop)
        float a0[NT], a1[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) a0[n] = a1[n] = 0.f;
        u32 st = warp;
        for (; (st + 8) * 256 + 8 * L < cols; st += 16) {
            const u32 b0 = st * 256 + 8 * L, b1 = b0 + 8 * 256;
            const int2 w0 = *reinterpret_cast<const int2*>(row + b0), w1 = *reinterpret_cast<const int2*>(row + b1);
            const float d0 = f16_to_f32(dw[b0 / 32]), d1 = f16_to_f32(dw[b1 / 32]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int2 q0 = *reinterpret_cast<const int2*>(xq[n] + b0), q1 = *reinterpret_cast<const int2*>(xq[n] + b1);
                a0[n] += d0 * xd[n][b0 / 32] * (float)__dp4a(w0.y, q0.y, __dp4a(w0.x, q0.x, 0));
                a1[n] += d1 * xd[n][b1 / 32] * (float)__dp4a(w1.y, q1.y, __dp4a(w1.x, q1.x, 0));
            }
        }
        const u32 b = st * 256 + 8 * L;
        if (b < cols) {
            const int2 wv = *reinterpret_cast<const int2*>(row + b);
            const float dd = f16_to_f32(dw[b / 32]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int2 q = *reinterpret_cast<const int2*>(xq[n] + b);
                a0[n] += dd * xd[n][b / 32] * (float)__dp4a(wv.y, q.y, __dp4a(wv.x, q.x, 0));
            }
        }
        // fold the leaf into the tree: pend[k] is the pending partial sum of level k (present iff bit k of the
        // leaves seen so far is set); the leaf merges upward through every trailing one of i, compile-time per i
        float cur[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) cur[n] = a0[n] + a1[n];
#pragma unroll
        for (u32 k = 0; k < 5; ++k) {
            if ((i >> k) & 1) {
#pragma unroll
                for (int n = 0; n < NT; ++n) cur[n] = pend[k][n] + cur[n];
            } else {
#pragma unroll
                for (int n = 0; n < NT; ++n) pend[k][n] = cur[n];
                break;
            }
        }
        if (i == 31) {
#pragma unroll
            for (int n = 0; n < NT; ++n) pend[5][n] = cur[n];
        }
    }
    // after 32 leaves the root sits at level 5
#pragma unroll
    for (int n = 0; n < NT; ++n) part[warp][lane][n] = pend[5][n];
    __syncthreads();
    for (u32 i = threadIdx.x; i < 32 * NT; i += blockDim.x) {
        const u32 rr = i / NT, n = i % NT;
        float t = 0.f;
        for (int w = 0; w < 8; ++w) t += part[w][rr][n];
        if (blk * 32 + rr < rows) y[(info[6] + z0 + n) * rows + blk * 32 + rr] = t;
    }
}

extern "C" __global__ void __launch_bounds__(256) b_t8zs(const u64* info) { mgemv_zs<8>(info); }

/// `mgemv_zt`'s dots (4 rows per block sharing the activation fragments) with the warp_sum trees folded after a
/// transpose through shared memory: lane p takes the 32 lane partials of pair p (row p / 8, token p % 8) and folds
/// them in the tree's leaf order (31 adds, no shuffles); the 8 warps are summed per group in order (barriers measured
/// free). Grid (ceil(rows / 4) per segment), 256 threads. Planar Q8_0 split-K segments only (the bench).
template <int NT, int RB> QX_DEV void mgemv_zx(const u64* info) {
    static_assert(RB * NT == 32, "32 pairs per warp: one per lane in the fold");
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 8 + 6]) ++s;
    const u64* d = segs + s * 8;
    const u32 rows = (u32)d[2], cols = (u32)d[3], ty = (u32)d[4];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, cols);
    const u32 blk = blockIdx.x - (u32)d[6];
    const u32 warp = threadIdx.x / 32, wl = threadIdx.x % 32;
    float* y = reinterpret_cast<float*>(d[1]);
    const u32 nz = (u32)(info[5] >> 32) ? (u32)(info[5] >> 32) : 1;
    __shared__ float tr[8][32][33];  // per warp: [lane][pair], padded
    __shared__ float part[8][32];
    const u32 r0 = blk * RB;
    const u32 nr = min((u32)RB, rows - r0);
    const u32 nb = cols / 32;
    const u8* w = reinterpret_cast<const u8*>(d[0]) + (u64)r0 * stride;
    for (u32 z = 0; z < nz; ++z) {
        const u64 z0 = (u64)z * NT;
        const i8* xq[NT];
        const float* xd[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            xq[n] = reinterpret_cast<const i8*>(info[7]) + (z0 + n) * info[9];
            xd[n] = reinterpret_cast<const float*>(info[8]) + (z0 + n) * (info[9] / 32);
        }
        float a0[RB][NT], a1[RB][NT];
#pragma unroll
        for (int r = 0; r < RB; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n) a0[r][n] = a1[r][n] = 0.f;
        u32 st = warp;
        for (; (st + 8) * 256 + 8 * wl < cols; st += 16) {
            const u32 b0 = st * 256 + 8 * wl, b1 = b0 + 8 * 256;
            int2 q0[NT], q1[NT];
            float s0[NT], s1[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                q0[n] = *reinterpret_cast<const int2*>(xq[n] + b0);
                q1[n] = *reinterpret_cast<const int2*>(xq[n] + b1);
                s0[n] = xd[n][b0 / 32];
                s1[n] = xd[n][b1 / 32];
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (r < nr) {
                    const u8* row = w + (u64)r * stride;
                    const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                    const int2 w0 = *reinterpret_cast<const int2*>(row + b0), w1 = *reinterpret_cast<const int2*>(row + b1);
                    const float d0 = f16_to_f32(dw[b0 / 32]), d1 = f16_to_f32(dw[b1 / 32]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        a0[r][n] += d0 * s0[n] * (float)__dp4a(w0.y, q0[n].y, __dp4a(w0.x, q0[n].x, 0));
                        a1[r][n] += d1 * s1[n] * (float)__dp4a(w1.y, q1[n].y, __dp4a(w1.x, q1[n].x, 0));
                    }
                }
            }
        }
        const u32 b = st * 256 + 8 * wl;
        if (b < cols) {
            int2 q[NT];
            float sc[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                q[n] = *reinterpret_cast<const int2*>(xq[n] + b);
                sc[n] = xd[n][b / 32];
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (r < nr) {
                    const u8* row = w + (u64)r * stride;
                    const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                    const int2 wv = *reinterpret_cast<const int2*>(row + b);
                    const float dd = f16_to_f32(dw[b / 32]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) a0[r][n] += dd * sc[n] * (float)__dp4a(wv.y, q[n].y, __dp4a(wv.x, q[n].x, 0));
                }
            }
        }
        // transpose: this lane's 32 pair partials to tr[warp][lane][pair]
#pragma unroll
        for (int r = 0; r < RB; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n) tr[warp][wl][r * NT + n] = a0[r][n] + a1[r][n];
        __syncwarp();
        // fold pair wl's 32 lane values in the tree's leaf order (bit-reversed lanes), a complete binary tree
        float pend[6];
#pragma unroll
        for (u32 i = 0; i < 32; ++i) {
            float cur = tr[warp][brev5(i)][wl];
#pragma unroll
            for (u32 k = 0; k < 5; ++k) {
                if ((i >> k) & 1) {
                    cur = pend[k] + cur;
                } else {
                    pend[k] = cur;
                    break;
                }
            }
            if (i == 31) pend[5] = cur;
        }
        part[warp][wl] = pend[5];
        __syncthreads();
        if (threadIdx.x < 32) {
            const u32 r = threadIdx.x / NT, n = threadIdx.x % NT;
            float t = 0.f;
            for (int k = 0; k < 8; ++k) t += part[k][threadIdx.x];
            if (r < nr) y[(info[6] + z0 + n) * rows + r0 + r] = t;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256) b_t8zx(const u64* info) { mgemv_zx<8, 4>(info); }
template <int XM, int NT, int RB, int DIAG> QX_DEV void mgemv_zt_diag(const u64* info) {
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 8 + 6]) ++s;
    const u64* d = segs + s * 8;
    const u32 rows = (u32)d[2], cols = (u32)d[3], ty = (u32)d[4], lanes = (u32)d[5];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, cols);
    const u32 blk = blockIdx.x - (u32)d[6];
    const u32 warp = threadIdx.x / 32, wl = threadIdx.x % 32;
    float* y = reinterpret_cast<float*>(d[1]);
    const u32 nz = (u32)(info[5] >> 32) ? (u32)(info[5] >> 32) : 1;
    __shared__ float part[8][RB * 128];  // (warp, row, token) over all the groups: NT * nz <= 128 tokens a launch
    if (lanes == 0 && ty == QX_Q8_0_P && info[7] != 0) {
        const u32 r0 = blk * RB;
        const u32 nr = min((u32)RB, rows - r0);
        const u32 nb = cols / 32;
        const u8* w = reinterpret_cast<const u8*>(d[0]) + (u64)r0 * stride;
        for (u32 z = 0; z < nz; ++z) {
            const u64 z0 = (u64)z * NT;
            const i8* xq[NT];
            const float* xd[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                xq[n] = reinterpret_cast<const i8*>(info[7]) + (z0 + n) * info[9];
                xd[n] = reinterpret_cast<const float*>(info[8]) + (z0 + n) * (info[9] / 32);
            }
            float a0[RB][NT], a1[RB][NT];
#pragma unroll
            for (int r = 0; r < RB; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n) a0[r][n] = a1[r][n] = 0.f;
            u32 st = warp;
            for (; (st + 8) * 256 + 8 * wl < cols; st += 16) {
                const u32 b0 = st * 256 + 8 * wl, b1 = b0 + 8 * 256;
                int2 q0[NT], q1[NT];
                float s0[NT], s1[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    q0[n] = *reinterpret_cast<const int2*>(xq[n] + b0);
                    q1[n] = *reinterpret_cast<const int2*>(xq[n] + b1);
                    s0[n] = xd[n][b0 / 32];
                    s1[n] = xd[n][b1 / 32];
                }
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    if (r < nr) {
                        const u8* row = w + (u64)r * stride;
                        const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                        const int2 w0 = *reinterpret_cast<const int2*>(row + b0), w1 = *reinterpret_cast<const int2*>(row + b1);
                        const float d0 = f16_to_f32(dw[b0 / 32]), d1 = f16_to_f32(dw[b1 / 32]);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            if (DIAG == 2) { a0[r][n] += (float)(q0[n].x ^ w0.x) + s0[n] + d0; a1[r][n] += (float)(q1[n].x ^ w1.x) + s1[n] + d1; } else {
                            a0[r][n] += d0 * s0[n] * (float)__dp4a(w0.y, q0[n].y, __dp4a(w0.x, q0[n].x, 0));
                            a1[r][n] += d1 * s1[n] * (float)__dp4a(w1.y, q1[n].y, __dp4a(w1.x, q1[n].x, 0)); }
                        }
                    }
                }
            }
            const u32 b = st * 256 + 8 * wl;
            if (b < cols) {
                int2 q[NT];
                float sc[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    q[n] = *reinterpret_cast<const int2*>(xq[n] + b);
                    sc[n] = xd[n][b / 32];
                }
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    if (r < nr) {
                        const u8* row = w + (u64)r * stride;
                        const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                        const int2 wv = *reinterpret_cast<const int2*>(row + b);
                        const float dd = f16_to_f32(dw[b / 32]);
#pragma unroll
                        for (int n = 0; n < NT; ++n) a0[r][n] += dd * sc[n] * (float)__dp4a(wv.y, q[n].y, __dp4a(wv.x, q[n].x, 0));
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float v = DIAG == 1 ? a0[r][n] + a1[r][n] : warp_sum(a0[r][n] + a1[r][n]);
                    if (wl == 0) part[warp][r * 128 + z0 + n] = v;
                }
        }
        __syncthreads();
        const u32 nt = nz * NT;
        for (u32 i = threadIdx.x; i < RB * nt; i += blockDim.x) {
            const u32 r = i / nt, n = i % nt;
            if (r < nr) {
                float t = 0.f;
                for (int k = 0; k < 8; ++k) t += part[k][r * 128 + n];
                y[(info[6] + n) * rows + r0 + r] = t;
            }
        }
        return;
    }
    // every other segment: the one-row code with one barrier per block
    for (u32 z = 0; z < nz; ++z) {
        const u64 z0 = (u64)z * NT;
        XIn<XM> x[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            x[n].x = reinterpret_cast<const float*>(info[0]) + (z0 + n) * info[3];
            x[n].arg = XM == 2 ? info[4] + 4ull * (z0 + n) * info[3] : info[4];
            x[n].xq = info[7] ? reinterpret_cast<const i8*>(info[7]) + (z0 + n) * info[9] : nullptr;
            x[n].xd = info[7] ? reinterpret_cast<const float*>(info[8]) + (z0 + n) * (info[9] / 32) : nullptr;
        }
        float a[NT];
        if (lanes == 0) {
            const u32 r = blk;
            partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, wl, 32, warp, 8, a);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const float v = warp_sum(a[n]);
                if (wl == 0) part[warp][z0 + n] = v;
            }
            continue;
        }
        const u32 rpw = 32 / lanes;  // rows per warp
        const u32 r = (blk * 8 + warp) * rpw + wl / lanes;
        const u32 lane = wl % lanes;
        if (r < rows) partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, lane, lanes, 0, 1, a);
        else
#pragma unroll
            for (int n = 0; n < NT; ++n) a[n] = 0.f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            float v = a[n];
            for (u32 o = lanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (r < rows && lane == 0) y[(info[6] + z0 + n) * rows + r] = v;
        }
    }
    if (lanes == 0) {
        __syncthreads();
        const u32 nt = nz * NT;
        for (u32 n = threadIdx.x; n < nt; n += blockDim.x) {
            float t = 0.f;
            for (int i = 0; i < 8; ++i) t += part[i][n];
            y[(info[6] + n) * rows + blk] = t;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) b_t8zt8(const u64* info) { mgemv_zt<0, 8, 8>(info); }
extern "C" __global__ void __launch_bounds__(256) b_zt_notree(const u64* info) { mgemv_zt_diag<0, 8, 4, 1>(info); }
extern "C" __global__ void __launch_bounds__(256) b_zt_nodots(const u64* info) { mgemv_zt_diag<0, 8, 4, 2>(info); }
template <int XM, int NT, int RB> QX_DEV void mgemv_zt2(const u64* info) {
    const u64* segs = reinterpret_cast<const u64*>(info[1]);
    const u32 nseg = (u32)info[2];
    u32 s = 0;
    while (s + 1 < nseg && blockIdx.x >= (u32)segs[(s + 1) * 8 + 6]) ++s;
    const u64* d = segs + s * 8;
    const u32 rows = (u32)d[2], cols = (u32)d[3], ty = (u32)d[4], lanes = (u32)d[5];
    const u64 stride = d[7] ? d[7] : type_row_bytes(ty, cols);
    const u32 blk = blockIdx.x - (u32)d[6];
    const u32 warp = threadIdx.x / 32, wl = threadIdx.x % 32;
    float* y = reinterpret_cast<float*>(d[1]);
    const u32 nz = (u32)(info[5] >> 32) ? (u32)(info[5] >> 32) : 1;
    __shared__ float part[8][RB * 128];  // (warp, row, token) over all the groups: NT * nz <= 128 tokens a launch
    if (lanes == 0 && ty == QX_Q8_0_P && info[7] != 0) {
        const u32 r0 = blk * RB;
        const u32 nr = min((u32)RB, rows - r0);
        const u32 nb = cols / 32;
        const u8* w = reinterpret_cast<const u8*>(d[0]) + (u64)r0 * stride;
        for (u32 z = 0; z < nz; ++z) {
            const u64 z0 = (u64)z * NT;
            const u32 xs = (u32)info[9], ds = xs / 32;
            const i8* xq0 = reinterpret_cast<const i8*>(info[7]) + z0 * xs;
            const float* xd0 = reinterpret_cast<const float*>(info[8]) + z0 * ds;
            float a0[RB][NT], a1[RB][NT];
#pragma unroll
            for (int r = 0; r < RB; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n) a0[r][n] = a1[r][n] = 0.f;
            u32 st = warp;
            for (; (st + 8) * 256 + 8 * wl < cols; st += 16) {
                const u32 b0 = st * 256 + 8 * wl, b1 = b0 + 8 * 256;
                int2 q0[NT], q1[NT];
                float s0[NT], s1[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    q0[n] = *reinterpret_cast<const int2*>(xq0 + n * xs + b0);
                    q1[n] = *reinterpret_cast<const int2*>(xq0 + n * xs + b1);
                    s0[n] = xd0[n * ds + b0 / 32];
                    s1[n] = xd0[n * ds + b1 / 32];
                }
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    if (r < nr) {
                        const u8* row = w + (u64)r * stride;
                        const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                        const int2 w0 = *reinterpret_cast<const int2*>(row + b0), w1 = *reinterpret_cast<const int2*>(row + b1);
                        const float d0 = f16_to_f32(dw[b0 / 32]), d1 = f16_to_f32(dw[b1 / 32]);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            a0[r][n] += d0 * s0[n] * (float)__dp4a(w0.y, q0[n].y, __dp4a(w0.x, q0[n].x, 0));
                            a1[r][n] += d1 * s1[n] * (float)__dp4a(w1.y, q1[n].y, __dp4a(w1.x, q1[n].x, 0));
                        }
                    }
                }
            }
            const u32 b = st * 256 + 8 * wl;
            if (b < cols) {
                int2 q[NT];
                float sc[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    q[n] = *reinterpret_cast<const int2*>(xq0 + n * xs + b);
                    sc[n] = xd0[n * ds + b / 32];
                }
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    if (r < nr) {
                        const u8* row = w + (u64)r * stride;
                        const u16* dw = reinterpret_cast<const u16*>(row + (u64)nb * 32);
                        const int2 wv = *reinterpret_cast<const int2*>(row + b);
                        const float dd = f16_to_f32(dw[b / 32]);
#pragma unroll
                        for (int n = 0; n < NT; ++n) a0[r][n] += dd * sc[n] * (float)__dp4a(wv.y, q[n].y, __dp4a(wv.x, q[n].x, 0));
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float v = warp_sum(a0[r][n] + a1[r][n]);
                    if (wl == 0) part[warp][r * 128 + z0 + n] = v;
                }
        }
        __syncthreads();
        const u32 nt = nz * NT;
        for (u32 i = threadIdx.x; i < RB * nt; i += blockDim.x) {
            const u32 r = i / nt, n = i % nt;
            if (r < nr) {
                float t = 0.f;
                for (int k = 0; k < 8; ++k) t += part[k][r * 128 + n];
                y[(info[6] + n) * rows + r0 + r] = t;
            }
        }
        return;
    }
    // every other segment: the one-row code with one barrier per block
    for (u32 z = 0; z < nz; ++z) {
        const u64 z0 = (u64)z * NT;
        XIn<XM> x[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            x[n].x = reinterpret_cast<const float*>(info[0]) + (z0 + n) * info[3];
            x[n].arg = XM == 2 ? info[4] + 4ull * (z0 + n) * info[3] : info[4];
            x[n].xq = info[7] ? reinterpret_cast<const i8*>(info[7]) + (z0 + n) * info[9] : nullptr;
            x[n].xd = info[7] ? reinterpret_cast<const float*>(info[8]) + (z0 + n) * (info[9] / 32) : nullptr;
        }
        float a[NT];
        if (lanes == 0) {
            const u32 r = blk;
            partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, wl, 32, warp, 8, a);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const float v = warp_sum(a[n]);
                if (wl == 0) part[warp][z0 + n] = v;
            }
            continue;
        }
        const u32 rpw = 32 / lanes;  // rows per warp
        const u32 r = (blk * 8 + warp) * rpw + wl / lanes;
        const u32 lane = wl % lanes;
        if (r < rows) partial<XM, NT>(ty, reinterpret_cast<const u8*>(d[0]) + (u64)r * stride, x, cols, lane, lanes, 0, 1, a);
        else
#pragma unroll
            for (int n = 0; n < NT; ++n) a[n] = 0.f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            float v = a[n];
            for (u32 o = lanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (r < rows && lane == 0) y[(info[6] + z0 + n) * rows + r] = v;
        }
    }
    if (lanes == 0) {
        __syncthreads();
        const u32 nt = nz * NT;
        for (u32 n = threadIdx.x; n < nt; n += blockDim.x) {
            float t = 0.f;
            for (int i = 0; i < 8; ++i) t += part[i][n];
            y[(info[6] + n) * rows + blk] = t;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) b_t8zt2(const u64* info) { mgemv_zt2<0, 8, 4>(info); }
extern "C" __global__ void __launch_bounds__(256, 2) b_t8zt2o(const u64* info) { mgemv_zt2<0, 8, 4>(info); }
extern "C" __global__ void __launch_bounds__(256) b_t8z(const u64* info) { mgemv_z<0, 8>(info); }
extern "C" __global__ void __launch_bounds__(256) b_t8zb(const u64* info) { mgemv_zb<0, 8>(info); }
extern "C" __global__ void __launch_bounds__(256) b_t8zt(const u64* info) { mgemv_zt<0, 8, 4>(info); }

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

static u16 f32_to_f16_bits(float f) {
    // round-to-nearest-even, normal range only (the scales are around 0.01-0.1)
    u32 u; std::memcpy(&u, &f, 4);
    const u32 sign = (u >> 16) & 0x8000; int exp = (int)((u >> 23) & 0xff) - 127 + 15; u32 mant = u & 0x7fffff;
    if (exp <= 0) return (u16)sign; if (exp >= 31) return (u16)(sign | 0x7c00);
    u32 h = sign | ((u32)exp << 10) | (mant >> 13); if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) ++h; return (u16)h;
}

int main(int argc, char** argv) {
    const u32 rows = argc > 1 ? atoi(argv[1]) : 5120, cols = argc > 2 ? atoi(argv[2]) : 4096, T = argc > 3 ? atoi(argv[3]) : 128;
    const u32 nb = cols / 32;
    const u64 stride = ((u64)nb * 32 + nb * 2 + 15) / 16 * 16;
    std::vector<u8> w((u64)rows * stride);
    srand(7);
    for (u32 r = 0; r < rows; ++r) {
        u8* row = w.data() + (u64)r * stride;
        for (u32 i = 0; i < nb * 32; ++i) row[i] = (u8)(i8)((rand() % 255) - 127);
        u16* d = reinterpret_cast<u16*>(row + (u64)nb * 32);
        for (u32 b = 0; b < nb; ++b) d[b] = f32_to_f16_bits(0.005f + (rand() % 1000) * 1e-5f);
    }
    std::vector<i8> xq((u64)T * cols);
    std::vector<float> xd((u64)T * nb), xf((u64)T * cols, 0.f);
    for (auto& v : xq) v = (i8)((rand() % 255) - 127);
    for (auto& v : xd) v = 0.01f + (rand() % 1000) * 1e-5f;
    u8* dw; i8* dxq; float* dxd; float* dxf; float* dy; u64* dtab; u64* dinfo;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dxq, xq.size())); CK(cudaMalloc(&dxd, xd.size() * 4)); CK(cudaMalloc(&dxf, xf.size() * 4));
    CK(cudaMalloc(&dy, (u64)T * rows * 4)); CK(cudaMalloc(&dtab, 8 * 8)); CK(cudaMalloc(&dinfo, 16 * 8));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, xd.data(), xd.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
    // the engine's segment words: w, y, rows, cols, type, lanes (0: split-K), first block, stride
    auto table = [&](u64 blocks_of_seg) { u64 t[8] = {(u64)dw, (u64)dy, rows, cols, QX_Q8_0_P, 0, 0, stride}; (void)blocks_of_seg; CK(cudaMemcpy(dtab, t, 64, cudaMemcpyHostToDevice)); };
    table(rows);
    u64 info[16] = {(u64)dxf, (u64)dtab, 1, cols, 0, 8ull | ((u64)(T / 8) << 32), 0, (u64)dxq, (u64)dxd, cols, 0, 0, 0, 0, 0, 0};
    CK(cudaMemcpy(dinfo, info, 128, cudaMemcpyHostToDevice));
    std::vector<float> ref((u64)T * rows), out((u64)T * rows);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    struct V { const char* name; void (*k)(const u64*); u32 blocks; };
    const V vs[] = {{"t8z (current)", b_t8z, rows}, {"t8zb (one barrier)", b_t8zb, rows}, {"t8zt (one barrier, 4 rows)", b_t8zt, (rows + 3) / 4}, {"t8zs (row per lane, stack)", b_t8zs, 0}, {"t8zx (4 rows, transpose fold)", b_t8zx, (rows + 3) / 4}, {"t8zt2 (4 rows, base+stride)", b_t8zt2, (rows + 3) / 4}, {"t8zt2o (same, 2 blocks/SM)", b_t8zt2o, (rows + 3) / 4}, {"t8zt8 (8 rows)", b_t8zt8, (rows + 7) / 8}, {"zt DIAG no tree (inexact)", b_zt_notree, (rows + 3) / 4}, {"zt DIAG no dots (inexact)", b_zt_nodots, (rows + 3) / 4}};
    std::printf("rows %u cols %u tokens %u: weights %.1f MB, x int8 %.1f MB\n", rows, cols, T, w.size() / 1e6, xq.size() / 1e6);
    for (size_t vi = 0; vi < sizeof(vs) / sizeof(vs[0]); ++vi) {
        const V& v = vs[vi];
        CK(cudaMemset(dy, 0, (u64)T * rows * 4));
        const dim3 grid = v.blocks ? dim3(v.blocks, 1, 1) : dim3((rows + 31) / 32, T / 8, 1);
        v.k<<<grid, 256>>>(dinfo); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        const int iters = 20;
        for (int i = 0; i < iters; ++i) v.k<<<grid, 256>>>(dinfo);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
        CK(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost));
        const double macs = (double)rows * cols * T;
        if (vi == 0) ref = out;
        const bool same = std::memcmp(ref.data(), out.data(), out.size() * 4) == 0;
        std::printf("%-30s %8.1f us  %6.1f TMAC/s  %s\n", v.name, ms * 1e3, macs / (ms * 1e-3) / 1e12, same ? "bit-identical" : "DIFFERS");
    }
    return 0;
}
