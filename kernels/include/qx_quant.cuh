// Block decoders for the GGUF formats QuixiInfer reads on the GPU.
//
// Every decoder produces the 8 consecutive elements [8*lane, 8*lane+8) of one 256-element chunk of a row,
// so one warp covers one chunk per step whatever the format. For 32-element formats a chunk is 8 blocks.
// `row` points at the row's first byte; `chunk` counts 256-element chunks from the row start.
#pragma once
#include "qx_common.cuh"
#include "qx_iq_tables.cuh"

enum QxType { QX_F32 = 0, QX_BF16 = 1, QX_Q8_0 = 2, QX_Q6_K = 3, QX_IQ4_NL = 4, QX_IQ3_S = 5, QX_IQ2_S = 6, QX_F16 = 7, QX_Q6_K_P = 8, QX_Q8_0_P = 9, QX_NVFP4 = 10 };

/// Unsigned E4M3 as ggml decodes NVFP4's sub-block scales: the value halved, so the doubled FP4 codes multiply back
/// to the real product. 0 and 0x7f decode to 0.
QX_DEV float ue4m3_half(u8 x) {
    if (x == 0 || x == 0x7f) return 0.f;
    const int e = (x >> 3) & 0xf, m = x & 7;
    const float raw = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.f + (float)m / 8.f, e - 7);
    return raw * 0.5f;
}

QX_DEV float ld_f16(const u8* p) { return f16_to_f32((u16)p[0] | ((u16)p[1] << 8)); }

template <int T> struct Fmt;
template <> struct Fmt<QX_F32> { static constexpr int BE = 1, BB = 4; };
template <> struct Fmt<QX_F16> { static constexpr int BE = 1, BB = 2; };
template <> struct Fmt<QX_BF16> { static constexpr int BE = 1, BB = 2; };
template <> struct Fmt<QX_Q8_0> { static constexpr int BE = 32, BB = 34; };
template <> struct Fmt<QX_Q6_K> { static constexpr int BE = 256, BB = 210; };
template <> struct Fmt<QX_IQ4_NL> { static constexpr int BE = 32, BB = 18; };
template <> struct Fmt<QX_IQ3_S> { static constexpr int BE = 256, BB = 110; };
template <> struct Fmt<QX_IQ2_S> { static constexpr int BE = 256, BB = 82; };
template <> struct Fmt<QX_NVFP4> { static constexpr int BE = 64, BB = 36; };

/// Bytes of a row of `cols` elements.
template <int T> QX_DEV u64 row_bytes(u32 cols) { return (u64)cols / Fmt<T>::BE * Fmt<T>::BB; }

template <int T> QX_DEV void decode8(const u8* row, u32 chunk, int lane, float* o);

template <> QX_DEV void decode8<QX_F32>(const u8* row, u32 chunk, int lane, float* o) {
    const float* p = reinterpret_cast<const float*>(row) + chunk * 256 + 8 * lane;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = p[i];
}
template <> QX_DEV void decode8<QX_BF16>(const u8* row, u32 chunk, int lane, float* o) {
    const u16* p = reinterpret_cast<const u16*>(row) + chunk * 256 + 8 * lane;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = bf16_to_f32(p[i]);
}
template <> QX_DEV void decode8<QX_F16>(const u8* row, u32 chunk, int lane, float* o) {
    const u16* p = reinterpret_cast<const u16*>(row) + chunk * 256 + 8 * lane;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = f16_to_f32(p[i]);
}
template <> QX_DEV void decode8<QX_Q8_0>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)(chunk * 8 + lane / 4) * 34;
    const float d = ld_f16(b);
    const i8* q = reinterpret_cast<const i8*>(b + 2 + (lane % 4) * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = d * (float)q[i];
}
template <> QX_DEV void decode8<QX_IQ4_NL>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)(chunk * 8 + lane / 4) * 18;
    const float d = ld_f16(b);
    const int s = (lane % 4) * 8;  // element offset in the block: 0, 8 low nibbles; 16, 24 high nibbles
    const u8* q = b + 2 + (s % 16);
    const int sh = s >= 16 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = d * (float)__ldg(&IQ4NL_VALUES[(q[i] >> sh) & 0xf]);
}
/// NVFP4: a 256-chunk is 4 blocks of 64 (36 bytes: 4 UE4M3 scales, 32 bytes of codes); lane l's 8 elements are
/// one half of sub-block (l % 8) / 2 of block l / 8: the low (even lane) or high (odd lane) nibbles of its 8 bytes.
template <> QX_DEV void decode8<QX_NVFP4>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)(chunk * 4 + lane / 8) * 36;
    const int sub = (lane % 8) / 2, sh = (lane % 2) * 4;
    const float d = ue4m3_half(b[sub]);
    const u8* q = b + 4 + 8 * sub;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = d * (float)__ldg(&FP4_VALUES2[(q[i] >> sh) & 0xf]);
}
template <> QX_DEV void decode8<QX_Q6_K>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)chunk * 210;
    const int n = 8 * lane, half = n / 128, m = n % 128, qd = m / 32, l0 = m % 32;
    const u8* ql = b + 64 * half + l0 + (qd % 2) * 32;
    const u8* qh = b + 128 + 32 * half + l0;
    const i8* sc = reinterpret_cast<const i8*>(b + 192 + 8 * half);
    const float d = ld_f16(b + 208) * (float)sc[l0 / 16 + 2 * qd];
    const int lsh = qd >= 2 ? 4 : 0, hsh = 2 * qd;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = (((ql[i] >> lsh) & 0xf) | (((qh[i] >> hsh) & 3) << 4)) - 32;
        o[i] = d * (float)q;
    }
}
template <> QX_DEV void decode8<QX_IQ3_S>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)chunk * 110;
    const float d = ld_f16(b);
    const u8* qs = b + 2; const u8* qh = b + 66; const u8* signs = b + 74; const u8* scales = b + 106;
    const int g = lane, sub = g / 4;
    const int sc = (scales[sub / 2] >> (4 * (sub % 2))) & 0xf;
    const float db = d * (float)(1 + 2 * sc);
    const u8 sg = signs[g];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int k = 2 * g + half;
        const u32 grid = __ldg(&IQ3S_GRID[qs[k] | (((qh[k / 8] >> (k % 8)) & 1) << 8)]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = 4 * half + j;
            const float m = (float)((grid >> (8 * j)) & 0xff);
            o[e] = (sg >> e) & 1 ? -db * m : db * m;
        }
    }
}
template <> QX_DEV void decode8<QX_IQ2_S>(const u8* row, u32 chunk, int lane, float* o) {
    const u8* b = row + (u64)chunk * 82;
    const float d = ld_f16(b);
    const u8* qs = b + 2; const u8* signs = b + 34; const u8* qh = b + 66; const u8* scales = b + 74;
    const int g = lane, sub = g / 2;
    const int sc = (scales[sub / 2] >> (4 * (sub % 2))) & 0xf;
    const float db = d * (0.5f + (float)sc) * 0.25f;
    const u64 grid = __ldg(&IQ2S_GRID[qs[g] | (((qh[g / 4] >> (2 * (g % 4))) & 3) << 8)]);
    const u8 sg = signs[g];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float m = (float)((grid >> (8 * e)) & 0xff);
        o[e] = (sg >> e) & 1 ? -db * m : db * m;
    }
}

// ---- planar layouts: the same bytes regrouped per row so lanes load 8 aligned bytes at a time ----
//
// Q6_K planar row of nb blocks: [ql: nb*128][qh: nb*64][scales: nb*16][d: nb*2], rows padded to 16 bytes.
// Q8_0 planar row of nb blocks: [qs: nb*32][d: nb*2], rows padded to 16 bytes.

QX_DEV void decode8_q6k_p(const u8* row, u32 nb, u32 chunk, int lane, float* o) {
    const int n = 8 * lane, half = n / 128, m = n % 128, qd = m / 32, l0 = m % 32;
    const u64 ql = *reinterpret_cast<const u64*>(row + chunk * 128 + 64 * half + l0 + (qd % 2) * 32);
    const u64 qh = *reinterpret_cast<const u64*>(row + (u64)nb * 128 + chunk * 64 + 32 * half + l0);
    const i8 sc = reinterpret_cast<const i8*>(row + (u64)nb * 192 + chunk * 16)[8 * half + l0 / 16 + 2 * qd];
    const float d = f16_to_f32(reinterpret_cast<const u16*>(row + (u64)nb * 208)[chunk]) * (float)sc;
    const int lsh = qd >= 2 ? 4 : 0, hsh = 2 * qd;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int lb = (int)((ql >> (8 * i)) & 0xff), hb = (int)((qh >> (8 * i)) & 0xff);
        o[i] = d * (float)((((lb >> lsh) & 0xf) | (((hb >> hsh) & 3) << 4)) - 32);
    }
}

QX_DEV void decode8_q80_p(const u8* row, u32 nb, u32 chunk, int lane, float* o) {
    const u32 blk = chunk * 8 + lane / 4;
    const u64 q = *reinterpret_cast<const u64*>(row + blk * 32 + (lane % 4) * 8);
    const float d = f16_to_f32(reinterpret_cast<const u16*>(row + (u64)nb * 32)[blk]);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = d * (float)(i8)((q >> (8 * i)) & 0xff);
}

/// Bytes of one dense row of `cols` elements of type `ty` (0 for the planar layouts, which carry their stride).
QX_DEV u64 type_row_bytes(u32 ty, u32 cols) {
    switch (ty) {
        case QX_F32: return 4ull * cols;
        case QX_BF16: case QX_F16: return 2ull * cols;
        case QX_Q8_0: return (u64)cols / 32 * 34;
        case QX_Q6_K: return (u64)cols / 256 * 210;
        case QX_IQ4_NL: return (u64)cols / 32 * 18;
        case QX_IQ3_S: return (u64)cols / 256 * 110;
        case QX_IQ2_S: return (u64)cols / 256 * 82;
        case QX_NVFP4: return (u64)cols / 64 * 36;
        default: return 0;
    }
}
