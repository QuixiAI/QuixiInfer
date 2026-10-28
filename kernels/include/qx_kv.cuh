// The FP8 KV cache's host copy, paged (mirrored in crates/qx-engine/src/kvpages.rs).
#pragma once
#include "qx_common.cuh"

// The host copy is paged: a page holds QX_KV_PAGE cells of every attention layer, [K | V | K scales | V scales]
// per layer, and a sequence slot's page table (u64 page addresses by cell / QX_KV_PAGE) lives in device memory.
// A layer's table entry for a slot is 8 u64: [0] ring K, [1] ring V, [2] ring K scales, [3] ring V scales,
// [4] the slot's page table, [5] the layer's K offset in a page, [6] its V offset, [7] its K scales offset (the V
// scales follow, QX_KV_PAGE * kv * QX_KV_SG floats on).
// A cell's K (and V) of one kv head has QX_KV_SG f32 scales, one per group of 32 of its 256 dims: with one scale
// per head, dims far below the head's largest fell into E4M3's subnormals (a 3-5% error on the attention output).
#define QX_KV_PAGE 256
#define QX_KV_SG 8

/// The host copy's K or V bytes and scales of a cell.
QX_DEV u8* kv_page_row(const u64* tab, bool v, u32 cell, u32 g, u32 nkv, u32 d, float*& scale) {
    const u64 base = reinterpret_cast<const u64*>(tab[4])[cell / QX_KV_PAGE];
    const u64 i = (u64)(cell % QX_KV_PAGE) * nkv + g;
    scale = reinterpret_cast<float*>(base + tab[7] + (v ? 4ull * QX_KV_PAGE * nkv * QX_KV_SG : 0) + i * 4 * QX_KV_SG);
    return reinterpret_cast<u8*>(base + (v ? tab[6] : tab[5]) + i * d);
}

/// Max over the 16 lanes of a half-warp (one 32-dim group when a lane holds two dims).
QX_DEV float group16_max(float m) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    return m;
}
