# Established Findings: Do Not Re-Derive

These are established findings, distilled from `docs/progress.md` (DeepSeek entries, 2026-11-03 to 2026-11-05)
and from `optimization_status.md`. Treat each as current truth until it is re-measured. Every entry names its
date and source, so new data can challenge it.

## Environment anchor

The host is an RTX 5090 (32 GiB, driver 595.99.02, 575 W), a Ryzen 9 9950X (16 cores) and 120 GiB of DDR5, with
the pack on the root NVMe. The fingerprint is `rtx-5090_ryzen-9-9950x_121g`.

All DeepSeek numbers below use the forced benchmark on `golden/data/route1k.tokens` unless stated. Tier settings
were 20-21 GiB VRAM and 88-100 GiB RAM. The settings before 2026-11-04 differ from the standard ones in
`perf.md`, so read them as history, not as a baseline.

## Where the time goes (current truth)

| fact | numbers | date / source |
|---|---|---|
| Decode is pool-bound | 31 ms per token = GPU-only 13 + resident experts about 3 + pool about 20 (about 3 of it NVMe wait), partly overlapped | 2026-11-04, progress "context growth was all attention" |
| The pool is at the RAM's bandwidth | 0.22 ms per 9.4 MiB expert at any batch size, 44-54 GB/s on 16 cores; no CPU-side lever except fewer CPU experts | 2026-11-04, `qx-poolbw` |
| Wide prefill windows make the pool compute-bound | 128 dot products per expert row; 16 AVX-512 cores saturate at about 4 TMAC/s (0.45 ms per expert for 128 tokens) | 2026-11-04, windows entry |
| The GPU-only step is flat in context up to 1k | 13.1 ms at 1k after flash-decoding attention (was 24); 24 ms at 16k for a single decode step | 2026-11-04 |
| Graph replay costs nothing; capture costs about 0.9 s per new width | replay 0.01 ms; capture fills CubeCL's info cache | 2026-11-04 |
| A distinct expert set per window grows sublinearly with width | about 100 per layer at 32 tokens, all 384 past about 500 tokens | 2026-11-04 |
| Run-to-run noise | same commit: decode 31.2-33.1 ms, prefill64 11.7-12.4 ms (spread up to 6%) | 2026-11-05, this scaffolding |

## Wins

| finding | effect | date / source |
|---|---|---|
| Asynchronous driver: mailbox-driven pool on its own thread, no per-layer host round trip | 9.2 to 12.0 tok/s at the time; the base of everything after | 2026-11-03 |
| Promotions off the critical path: event-ordered VRAM copies, async io_uring RAM promotions | steady token 64-68 to 52-57 ms | 2026-11-03 |
| Pool computes RAM experts while NVMe reads are in flight (`fetch_submit` / `fetch_wait`) | NVMe wait overlapped; part of 52 to 42 ms | 2026-11-03 |
| `hc_mixes` a block per output, then split 16 ways; `qx_ds_attn` a warp per key | GPU-only token 20 to 16.5 to 13.7 ms | 2026-11-03 / 04 |
| Engram rows in one io_uring batch per token | 4.1 to 0.1-0.3 ms per token (cold-start cost mostly) | 2026-11-03 |
| Flash-decoding MLA attention (`qx_ds_attn_part` / `merge`, 8-head groups x 12-key chunks) | GPU-only 24 to 13.1 ms at 1k; decode 43.3 to 31.9 ms per token | 2026-11-04 |
| Windows of up to 128 tokens, pool reads each expert once per window | prefill 34 to 21.8 ms per token at windows of 64 | 2026-11-04 |
| DMA-staged GPU share of RAM misses (6/8 windows, 3/8 decode, 96-record area, 4 streams, deduplicated) | prefill64 21.8 to 11.8 ms per token (1.85x), decode 31.4; bit-identical | 2026-11-04 |
| Fixed window widths (`WIDTHS`) captured at server warm-up | removes 0.9 s capture stalls from requests | 2026-11-04 |

## Correctness rules learned the hard way

- **Never upload with `DeviceBuf::upload` into memory a queued kernel reads.** It runs on the legacy stream,
  unordered with CubeCL's stream. Use a mapped buffer plus a copy kernel, or sync first. This caused the
  Engram race in 2026-11-03.
- **Never let a kernel compile or load for the first time while an expert job is posted.** A module load
  deadlocks against a spinning wait kernel. Each new window width runs a dry pass first (2026-11-04, the
  intermittent ds-checks hang).
- **Per-token buffers must be sized by `DS_TMAX`.** One-token-wide top-k buffers silently corrupted neighbouring
  GEMV tables in windows. compute-sanitizer found it (2026-11-04).
- **Grids inside captured graphs must not depend on position.** Size them from the maximum and bound each kernel
  by the true position.

## Rejected: with the reason, so they are not retried

- **GPU reads of RAM-resident experts in place over PCIe (`QX_DS_GPU_SHARE`) were rejected on 2026-11-04.**
  Mapped reads ran at about 7 GB/s, 1.3 ms per expert against the pool's 0.22. Decode went from 32 to 135-175 ms
  per token at shares of 4-6 eighths, and the VRAM tier stopped adapting. The knob stays at 0. DMA staging
  replaced it.
- **The bf16 tensor-core GEMV for DeepSeek was rejected on 2026-11-04.** It is faster in places: windows of 32 ran
  22.6 ms per token against 24.0 for f32. Windows then stop matching single steps bit for bit. It is opt-in with
  `QX_TMV=1`, capped at 32 tokens.
- **Token graph capture as a speed lever brought no gain at the time (2026-11-03).** The GPU-only token stayed at
  20 ms because the cost was kernel time, not launches. Graphs are kept because replay is free and they are
  needed for windows, but more launch-count work will not pay.
- **Prefill windows of 128 lose to 64 with staging (2026-11-04).** They ran 14.6 against 11.8 ms per token,
  because a window's distinct experts overflow the 96-record staging area and the rest fall back to the pool.
  64 stays the prompt chunk.
- **A staged share above 6/8 for windows makes no difference (2026-11-04).** 7/8 measured 11.8 and 8/8 11.6,
  against 11.8 at 6/8. That is inside the noise, and the limit moves to the copies and the GPU expert kernels.
- **A decode staged share of 4/8 makes no difference (2026-11-04).** It measured 31.2 against 31.4 ms at 3/8,
  inside the noise. A staged expert's DMA costs about 0.7 ms against the pool's 0.22, so the balance point stays
  small.
- **Pinning the engine thread off the pool's cores (`QX_DS_CPUS`, `QX_DS_MAIN_CPU`) changed nothing (2026-11-03).**
- **SMT for the pool is slower:** 50.5 against 54 GB/s (2026-10, `qx-rambw`).
- **Speculative decoding to amortize the pool was rejected on analysis (2026-11-04).** Different tokens pick
  different experts, so decode windows do not reduce pool bytes per token. It is not measured. Re-open only with
  a routing-overlap measurement.
