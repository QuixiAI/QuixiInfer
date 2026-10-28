// System-scope memory ordering for the mailboxes in mapped host memory, and their layout.
#pragma once
#include "qx_common.cuh"

QX_DEV u64 ld_acquire_sys(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
QX_DEV void st_release_sys(u64* p, u64 v) { asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory"); }

// MoE mailbox (bytes): the GPU posts a layer's missing experts for a window of up to QX_TMAX tokens, the CPU
// pool answers with their outputs. Selection a = t * k + j is token t's j-th expert. Mirrored in
// crates/qx-engine/src/mailbox.rs.
#define QX_WMAX 8     // tokens per speculative window (one sequence)
#define QX_TMAX 32    // tokens per window (all sequences)
#define QX_AMAX 320   // selections per window (QX_TMAX * k)
#define QX_STEP_ROW 16  // u32 step words per sequence slot
#define MB_SEQ 0      // u64, GPU -> CPU: job number, written last
#define MB_DONE 64    // u64, CPU -> GPU: job number answered, written last
#define MB_LAYER 128  // u32
#define MB_NMISS 132  // u32
#define MB_NTOK 136   // u32: tokens in the window
#define MB_KPT 140    // u32: experts per token (k)
#define MB_IDS 256    // i32[QX_AMAX]: the missing experts
#define MB_KIDX 1536  // i32[QX_AMAX]: their selection index a
#define MB_ALL_ 2816  // i32[QX_AMAX]: every selected expert, for the driver's cache scoring
#define MB_ALL(E) MB_ALL_
#define MB_X 4096     // f32[QX_TMAX][E]: the window's MoE inputs, token t at t * E
#define MB_Y(E) (MB_X + 4 * QX_TMAX * (E))  // f32[QX_AMAX][E]: unweighted output of missing selection a at row a
#define MB_BYTES(E) (MB_Y(E) + 4 * QX_AMAX * (E) + 64)

// PLE mailbox: the CPU posts the PLE embedding of each position into a ring, slot pos % cap, and posts position p
// only once p < consumed + cap.
#define PLE_CONSUMED 0  // u64, GPU -> CPU: 1 + the last position copied out
#define PLE_READY 64    // u64, CPU -> GPU: 1 + the last position whose embedding is in place
#define PLE_EMB 128     // f32[cap][E]
