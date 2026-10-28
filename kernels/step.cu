// Step control and GPU <-> CPU handoffs, so a whole token step runs with no host involvement.
//
// Device step state (u32 words): [0] pos, [1] tok, [2] sampled token, [3] forced tokens (positions below
// this take their token from the context array), [4..6) MoE job counter (u64), [6] misses of the current job.
#include "qx_sys.cuh"

#define ST(i) reinterpret_cast<u32*>(info[0])[i]

// Lanes: a window's tokens each belong to a sequence slot (u32 per token, written by the host). Consecutive
// tokens of one slot form a segment at consecutive positions: a speculative window is one segment, a batched
// decode step one segment of one token per sequence. Each slot has its own row of step words (QX_STEP_ROW u32
// at step + slot * QX_STEP_ROW), context array and output ring; the MoE job words live in row 0
// (the second half of a split window has its own words and mailbox).

/// Token t's index within its segment.
QX_DEV u32 seg_index(const u32* slots, u32 t) {
    u32 k = 0;
    while (k < t && slots[t - k - 1] == slots[t]) ++k;
    return k;
}

/// Takes the window's lanes: the host writes each window's lanes to ring entry n % R before queueing window n, and
/// this kernel, first in the window, copies entry n to the device lanes array and counts n up, publishing it to the
/// host (which keeps fewer than R windows ahead). info: [0] mapped ring (u32[R][TMAX]), [1] lanes out (u32 x TMAX),
/// [2] R, [3] TMAX, [4] device counter (u32), [5] mapped published counter (u64). One thread.
extern "C" __global__ void qx_lanes_take(const u64* info) {
    u32* ctr = reinterpret_cast<u32*>(info[4]);
    const u32 n = *ctr, R = (u32)info[2], T = (u32)info[3];
    const volatile u32* src = reinterpret_cast<const volatile u32*>(info[0]) + (u64)(n % R) * T;
    u32* dst = reinterpret_cast<u32*>(info[1]);
    for (u32 t = 0; t < T; ++t) dst[t] = src[t];
    *ctr = n + 1;
    __threadfence_system();
    st_release_sys(reinterpret_cast<u64*>(info[5]), (u64)n + 1);
}

/// The second half of a split window takes its lanes from the first's: dst[i] = src[min(off + i, T - 1)] for
/// i < T. info: [0] src lanes, [1] dst lanes, [2] off, [3] T. One thread.
extern "C" __global__ void qx_lanes_split(const u64* info) {
    const u32* src = reinterpret_cast<const u32*>(info[0]);
    u32* dst = reinterpret_cast<u32*>(info[1]);
    const u32 off = (u32)info[2], T = (u32)info[3];
    for (u32 i = 0; i < T; ++i) dst[i] = src[min(off + i, T - 1)];
}

/// Starts a window: token t (k-th of its segment, slot s) is at position pos_s + k; its token is ctx_s[pos_s + k]
/// when that position is forced or k > 0, else the token slot s sampled last. Writes the tokens and positions.
/// info: [0] step state, [1] context tokens (slot s at s * [6]), [2] tokens out (u32 x T), [3] positions out
/// (u32 x T), [4] T, [5] slots (u32 x T), [6] context stride (u32 per slot). One thread.
extern "C" __global__ void qx_win_begin(const u64* info) {
    const u32 T = (u32)info[4];
    const u32* slots = reinterpret_cast<const u32*>(info[5]);
    for (u32 t = 0; t < T; ++t) {
        const u32 s = slots[t], k = seg_index(slots, t);
        u32* st = reinterpret_cast<u32*>(info[0]) + (u64)s * QX_STEP_ROW;
        const u32* ctx = reinterpret_cast<const u32*>(info[1]) + (u64)s * info[6];
        const u32 p = st[0] + k;
        const u32 tok = p < st[3] || k > 0 ? ctx[p] : st[2];
        reinterpret_cast<u32*>(info[2])[t] = tok;
        reinterpret_cast<u32*>(info[3])[t] = p;
        if (k == 0) st[1] = tok;
    }
}

/// Ends a window: per segment of slot s at pos_s, token k's sample goes to s's output ring at pos_s + k (and into
/// its context past the forced tokens), the last one becomes s's sampled token, pos_s advances by the segment's
/// length, then s's step count is published. info: [0] step state, [1] context tokens, [2] mapped output (slot s's
/// ring of u32 at s * [8] bytes, its u64 step count [3] bytes further), [3] count offset, [4] samples (u32 x T),
/// [5] T, [6] slots, [7] context stride (u32), [8] output stride (bytes). One thread.
extern "C" __global__ void qx_win_end(const u64* info) {
    const u32 T = (u32)info[5];
    const u32* slots = reinterpret_cast<const u32*>(info[6]);
    const u32* smp = reinterpret_cast<const u32*>(info[4]);
    u8* out = reinterpret_cast<u8*>(info[2]);
    for (u32 t = 0; t < T; ++t) {
        const u32 s = slots[t], k = seg_index(slots, t);
        u32* st = reinterpret_cast<u32*>(info[0]) + (u64)s * QX_STEP_ROW;
        u32* ctx = reinterpret_cast<u32*>(info[1]) + (u64)s * info[7];
        const u32 p = st[0] + k, next = p + 1;
        if (next >= st[3]) ctx[next] = smp[t];
        reinterpret_cast<volatile u32*>(out + (u64)s * info[8])[p] = smp[t];
        if (t + 1 == T || slots[t + 1] != s) {
            st[2] = smp[t];
            st[0] = next;
        }
    }
    __threadfence_system();
    for (u32 t = 0; t < T; ++t)
        if (t + 1 == T || slots[t + 1] != slots[t]) {
            const u32 s = slots[t];
            st_release_sys(reinterpret_cast<u64*>(out + (u64)s * info[8] + info[3]), (u64)reinterpret_cast<const u32*>(info[0])[(u64)s * QX_STEP_ROW]);
        }
}

/// Greedy sampling into the step state. info: [0] logits, [1] n, [2] step state. One block of 1024.
extern "C" __global__ void qx_sample_greedy(const u64* info) {
    __shared__ float bv[1024];
    __shared__ u32 bi[1024];
    const float* x = reinterpret_cast<const float*>(info[0]);
    float v = -3.402823466e38f;
    u32 idx = 0;
    for (u32 i = threadIdx.x; i < (u32)info[1]; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
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
    if (threadIdx.x == 0) reinterpret_cast<u32*>(info[2])[2] = bi[0];
}

/// Posts a MoE job: the selected experts the VRAM table lacks, their selection indices, x and every selected id, then
/// the job number. Also snapshots the selected experts' slots, so the GPU kernels of this layer use exactly
/// the residency the CPU was told about even if the table changes meanwhile.
/// Also quantizes x to Q8_K for the GPU expert kernels, exactly as qx_quant_q8k.
/// info: [0] mailbox (mapped), [1] table, [2] layer * n_expert, [3] ids, [4] weights, [5] x, [6] step state,
/// [7] layer, [8] k, [9] E, [10] slots out (u64 x selections), [11] x qs out (i8 x E per token), [12] x d out
/// (f32 x E/256 per token), [13] tokens in the window (0: 1): selection a = t * k + j, token t's x at t * E.
/// Grid: one block of 256 threads per token. Block 0 scans the selections and posts the job; every block copies
/// (when the CPU has work) and quantizes its token's x; block 0 publishes the job once all have. Step words [7]
/// (the job whose scan is out) and [8] (blocks done with it) order them; all blocks of a window are resident
/// together, so the waits are safe.
extern "C" __global__ void qx_moe_post(const u64* info) {
    u8* mb = reinterpret_cast<u8*>(info[0]);
    const u32 E = (u32)info[9], T = info[13] ? (u32)info[13] : 1, A = T * (u32)info[8], tk = blockIdx.x;
    u32* st = reinterpret_cast<u32*>(info[6]);
    volatile u32* vst = st;
    const u64 prev = *reinterpret_cast<const u64*>(st + 4);
    __shared__ u32 s_n;
    if (threadIdx.x == 0) {
        if (tk == 0) {
            // The mailbox has one slot: wait until the CPU has answered the previous job before overwriting it. A job
            // with no misses is not waited for when posted, so this is where its answer is collected.
            while (ld_acquire_sys(reinterpret_cast<const u64*>(mb + MB_DONE)) < prev) {
            }
            const u64* table = reinterpret_cast<const u64*>(info[1]) + info[2];
            const i32* ids = reinterpret_cast<const i32*>(info[3]);
            u32 n = 0;
            u64* slots = reinterpret_cast<u64*>(info[10]);
            i32* all = reinterpret_cast<i32*>(mb + MB_ALL(E));
            for (u32 j = 0; j < A; ++j) {
                const u64 slot = *reinterpret_cast<const volatile u64*>(&table[ids[j]]);
                slots[j] = slot;
                all[j] = ids[j];
                if (slot == 0) {
                    reinterpret_cast<i32*>(mb + MB_IDS)[n] = ids[j];
                    reinterpret_cast<i32*>(mb + MB_KIDX)[n] = (i32)j;
                    ++n;
                }
            }
            *reinterpret_cast<u32*>(mb + MB_LAYER) = (u32)info[7];
            *reinterpret_cast<u32*>(mb + MB_NMISS) = n;
            *reinterpret_cast<u32*>(mb + MB_NTOK) = T;
            *reinterpret_cast<u32*>(mb + MB_KPT) = (u32)info[8];
            vst[6] = n;
            __threadfence();
            vst[7] = (u32)(prev + 1);
        } else {
            while (vst[7] != (u32)(prev + 1)) {
            }
            __threadfence();
        }
        s_n = vst[6];
    }
    __syncthreads();
    const float* x = reinterpret_cast<const float*>(info[5]) + (u64)tk * E;
    // x crosses PCIe only when the CPU has experts to compute
    if (s_n > 0) {
        float* mx = reinterpret_cast<float*>(mb + MB_X) + (u64)tk * E;
        for (u32 i = threadIdx.x; i < E; i += blockDim.x) mx[i] = x[i];
    }
    // Q8_K of x: per 256, the first index of the largest magnitude sets the scale. One warp per 256-block.
    {
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
            i8* qs = reinterpret_cast<i8*>(info[11]) + (u64)tk * E + b * 256 + 8 * l;
            float* dd = reinterpret_cast<float*>(info[12]) + (u64)tk * (E / 256);
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
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd(st + 8, 1u);
        if (tk == 0) {
            while (vst[8] < T) {
            }
            vst[8] = 0;
            u64* ctr = reinterpret_cast<u64*>(st + 4);
            const u64 seq = prev + 1;
            *ctr = seq;
            __threadfence_system();
            st_release_sys(reinterpret_cast<u64*>(mb + MB_SEQ), seq);
        }
    }
}

/// Waits until the CPU has answered the current MoE job, unless the job has no misses. info: [0] mailbox,
/// [1] step state. One thread.
extern "C" __global__ void qx_moe_wait(const u64* info) {
    if (reinterpret_cast<const u32*>(info[1])[6] == 0) return;
    const u64 seq = *reinterpret_cast<const u64*>(reinterpret_cast<const u32*>(info[1]) + 4);
    const u64* done = reinterpret_cast<const u64*>(reinterpret_cast<const u8*>(info[0]) + MB_DONE);
    while (ld_acquire_sys(done) < seq) {
    }
}

/// Waits until the CPU has posted each token's PLE embedding, then copies them to `emb` (token t at t * E): token
/// t of slot s at position p reads s's ring (at s * [7] bytes), slot p % cap. Each slot's consumed mark then moves
/// past its last token. info: [0] PLE mailboxes, [1] positions (u32 x T), [2] emb out, [3] E, [4] ring slots,
/// [5] T, [6] slots (u32 x T), [7] mailbox stride (bytes). One block.
extern "C" __global__ void qx_ple_wait(const u64* info) {
    const u32 T = (u32)info[5], E = (u32)info[3];
    const u32* pos = reinterpret_cast<const u32*>(info[1]);
    const u32* slots = reinterpret_cast<const u32*>(info[6]);
    for (u32 tk = 0; tk < T; ++tk) {
        const u8* mb = reinterpret_cast<const u8*>(info[0]) + (u64)slots[tk] * info[7];
        if (threadIdx.x == 0) {
            while (ld_acquire_sys(reinterpret_cast<const u64*>(mb + PLE_READY)) < (u64)pos[tk] + 1) {
            }
        }
        __syncthreads();
        const volatile float* src = reinterpret_cast<const volatile float*>(mb + PLE_EMB) + (u64)(pos[tk] % (u32)info[4]) * E;
        for (u32 i = threadIdx.x; i < E; i += blockDim.x) reinterpret_cast<float*>(info[2])[(u64)tk * E + i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (u32 tk = 0; tk < T; ++tk)
            if (tk + 1 == T || slots[tk + 1] != slots[tk])
                st_release_sys(reinterpret_cast<u64*>(reinterpret_cast<u8*>(info[0]) + (u64)slots[tk] * info[7] + PLE_CONSUMED), (u64)pos[tk] + 1);
    }
}

QX_DEV u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/// Samples the next token into the step state: temperature, top-k (radix select of the k-th largest logit),
/// top-p over the k, min-p, then a draw from a counter-based RNG keyed by (seed, position). Temperature 0 is
/// argmax. Parameters (f32 / u32 words, 8 per slot): [0] temperature, [1] top_p, [2] min_p, [3] top_k (<= 1024),
/// [4..6) seed. info: [0] logits, [1] n, [2] unused, [3] parameters, [4] samples out (u32 per token), [5] positions
/// (u32 x T), [6] slots (u32 x T), [7] token masks (mapped: u32 per token, 1 = masked, then ceil(n / 32) words
/// per token with bit i allowing token i; 0: none), [8] the window's first lane among the masks' (the second half
/// of a split window). One block of 1024 threads per token t: logits row t, the
/// parameters of its slot, drawn at its position; a masked token's disallowed logits are set to -inf first.
extern "C" __global__ void qx_sample(const u64* info) {
    __shared__ u32 hist[256];
    __shared__ u32 s_prefix, s_need, s_cnt;
    __shared__ float cv[1024];
    __shared__ u32 ci[1024];
    __shared__ float bv[1024];
    __shared__ u32 bi[1024];
    const u32 n = (u32)info[1], tid = threadIdx.x, nt = blockDim.x, tk = blockIdx.x;
    const float* x = reinterpret_cast<const float*>(info[0]) + (u64)tk * n;
    const u32 pos = reinterpret_cast<const u32*>(info[5])[tk];
    const u32* prm = reinterpret_cast<const u32*>(info[3]) + 8ull * reinterpret_cast<const u32*>(info[6])[tk];
    auto put = [&](u32 pick) { reinterpret_cast<u32*>(info[4])[tk] = pick; };
    const u32 lane = tk + (u32)info[8];
    if (info[7] && reinterpret_cast<const volatile u32*>(info[7])[lane]) {
        const u32 words = (n + 31) / 32;
        const volatile u32* m = reinterpret_cast<const volatile u32*>(info[7]) + QX_TMAX + (u64)lane * words;
        float* xw = reinterpret_cast<float*>(info[0]) + (u64)tk * n;
        for (u32 w = tid; w < words; w += nt) {
            const u32 bits = m[w];
            if (bits == 0xffffffffu) continue;
            for (u32 b = 0; b < 32; ++b) {
                const u32 i = 32 * w + b;
                if (i < n && !((bits >> b) & 1u)) xw[i] = -3.402823466e38f;
            }
        }
        __syncthreads();
    }
    const float temp = __uint_as_float(prm[0]), top_p = __uint_as_float(prm[1]), min_p = __uint_as_float(prm[2]);
    const u32 top_k = min(max(prm[3], 1u), 1024u);
    const u64 seed = (u64)prm[4] | ((u64)prm[5] << 32);
    if (temp <= 0.f || top_k == 1) {
        float v = -3.402823466e38f;
        u32 idx = 0;
        for (u32 i = tid; i < n; i += nt)
            if (x[i] > v) { v = x[i]; idx = i; }
        bv[tid] = v;
        bi[tid] = idx;
        __syncthreads();
        for (u32 s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && (bv[tid + s] > bv[tid] || (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]))) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) put(bi[0]);
        return;
    }
    auto key = [&](u32 i) { const u32 u = __float_as_uint(x[i]); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
    u32 prefix = 0, need = top_k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (u32 i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const u32 hmask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (u32 i = tid; i < n; i += nt) {
            const u32 k = key(i);
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
    // candidates: every logit at or above the threshold key (ties may exceed k; capped at 1024)
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    for (u32 i = tid; i < n; i += nt) {
        if (key(i) >= prefix) {
            const u32 slot = atomicAdd(&s_cnt, 1u);
            if (slot < 1024) { cv[slot] = x[i]; ci[slot] = i; }
        }
    }
    __syncthreads();
    const u32 m = min(s_cnt, 1024u);
    // bitonic sort of the candidates, largest first (padding sorts last)
    for (u32 i = tid; i < 1024; i += nt) if (i >= m) { cv[i] = -3.402823466e38f; ci[i] = 0xffffffffu; }
    __syncthreads();
    for (u32 k = 2; k <= 1024; k <<= 1)
        for (u32 j = k >> 1; j > 0; j >>= 1) {
            for (u32 i = tid; i < 1024; i += nt) {
                const u32 p = i ^ j;
                if (p > i) {
                    const bool desc = (i & k) == 0;
                    const bool swap = desc ? (cv[i] < cv[p] || (cv[i] == cv[p] && ci[i] > ci[p])) : (cv[i] > cv[p] || (cv[i] == cv[p] && ci[i] < ci[p]));
                    if (swap) {
                        const float tv = cv[i]; cv[i] = cv[p]; cv[p] = tv;
                        const u32 ti = ci[i]; ci[i] = ci[p]; ci[p] = ti;
                    }
                }
            }
            __syncthreads();
        }
    if (tid == 0) {
        const u32 kk = min(top_k, m);
        const float mx = cv[0];
        float z = 0.f;
        for (u32 i = 0; i < kk; ++i) { bv[i] = expf((cv[i] - mx) / temp); z += bv[i]; }
        // top-p: the smallest prefix reaching top_p; min-p: drop below min_p * the largest probability
        u32 keep = kk;
        float cum = 0.f;
        for (u32 i = 0; i < kk; ++i) {
            cum += bv[i] / z;
            if (cum >= top_p) { keep = i + 1; break; }
        }
        float z2 = 0.f;
        u32 keep2 = 0;
        for (u32 i = 0; i < keep; ++i) {
            if (bv[i] / z < min_p * (bv[0] / z)) break;
            z2 += bv[i];
            keep2 = i + 1;
        }
        const u64 r = splitmix64(seed ^ (0x51ED27u + (u64)pos * 0x2545F4914F6CDD1Dull));
        const float u = (float)(r >> 40) * (1.0f / 16777216.0f) * z2;
        float acc = 0.f;
        u32 pick = ci[0];
        for (u32 i = 0; i < keep2; ++i) {
            acc += bv[i];
            if (u < acc) { pick = ci[i]; break; }
            pick = ci[i];
        }
        put(pick);
    }
}
