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
| **The box's DRAM read ceiling is 52-59 GB/s** (two DDR5-5600 DIMMs; 45 GB/s from one thread, 59 from 16, a plain AVX2 sum over 512 MiB per thread). The pool's 44-54 is 75-92% of it. **Decode at the standard settings is at this design's ceiling:** every RAM-resident expert a token needs crosses the memory bus once, to the cores or to the GPU by DMA, so only VRAM residency or fewer bytes per token can move the 15.6 ms of pool per token. | `rambw.c`, 2026-11-06 | 2026-11-06 |
| **A 256k decode step (46 ms on 0ced38e, golden profile) is 28 ms of waiting for experts outside VRAM and 21 ms of kernels; the position cost is 6.6 ms.** nsys per step: `qx_moe_wait_staged` 17.0, `qx_moe_wait` 10.8, `qx_mgemv` width 1 6.5 (511 launches), `qx_ds_topk` 2.5 (one block, 4 radix passes + a barrier per 1,024 rows), `qx_ds_idx_score2` 2.2, attention 1.9, expert kernels ~4. Per token about 241 selections: 148 from VRAM (15.4 GiB tier), 32 DMA'd, 53 pool-computed, 8.6 from NVMe; a record is ~9.5 MB, so ~880 MB of expert bytes cross the memory bus per token (19 GB/s effective against the 52-59 ceiling: the per-layer round trip of ~2.4 records is latency, not bandwidth). The staging share is balanced at the default 3/8 (0/8 47.4, 3/8 45.4, 5/8 46.0, 8/8 46.1: DMA and the pool cost the same per record). Previous-token prediction covers 23% of the host-served selections (13% at 1k): no cheap prefetch. What is left on decode256k is kernels (top-k 5%, GEMV launches 3-5%) and VRAM residency (latent rows to host memory would free ~8 GiB for experts). | `nsys-dec256k2`, `stg-*`, `spec-*` logs; notebook 2026-11-07 | 2026-11-07 |
| Wide prefill windows make the pool compute-bound | 128 dot products per expert row; 16 AVX-512 cores saturate at about 4 TMAC/s (0.45 ms per expert for 128 tokens) | 2026-11-04, windows entry |
| The GPU-only step is flat in context up to 1k | 13.1 ms at 1k after flash-decoding attention (was 24); 24 ms at 16k for a single decode step | 2026-11-04 |
| **The long-context cost was the tier's memory, not the position (fixed 2026-11-06).** The score decay ran once per adapt call, so in 128-token windows the half-life of 256 meant 32,768 tokens and the tier could not follow a text's topic drift; the forced 64k pass showed the VRAM share collapse from 56% to 17% at a topic shift and take 30k tokens to recover (14 to 57 ms per token). Decay per token: long16k decode 38.5 to 33.8 (equal to decode at 1k), 16k prefill 18.4 to 14.4, pool 12.2 to 7.6 ms per token. | `decay-per-step` against `decay-per-token`, 3 repeats | 2026-11-06 |
| **End-to-end decode at 16k is 35-37 ms per token, not 53.** The 2026-11-04 figure averaged the width-1 graph capture (0.85-1.5 s, first tail step) over 64 steps. Steady steps at 16k: device wait 30-37 ms, pool 14.4 ms per token over the run. Prefill averaged over 0-16k is 19.5 ms per token against 12.3 at 1k. | harness `long16k`, three repeats: last-32 35.3-37.5, tail means 49.1-60.4 | 2026-11-06 |
| **A short prompt is bounded by the disk, not by kernels or the link (2026-11-12, revised the same evening).** Warmed 2,040-row full-stack chunk on chunk110, 3.84 s: the driver thread is busy 3.6 s of it, 1.99 s blocked in the io_uring submits and 1.49 s waiting for the reads (3,839 NVMe records, 36 GB, ~10.4 GB/s; the drive's largest request is 128 KiB, so a batch of 32 records is ~2,400 requests against 1,023 tags); the copy calls take 56 ms (the earlier "2.06 s of copies at the link's speed" was a subtraction, and wrong). `fio` on the pack alone: 11.1-11.8 GB/s. Only 2,356 of those reads were of experts the chunk routes to (the prefetch picks before the routing). With picks skipped once the routing is known, batches of 8 and eight event sets (chunk113): 3.24 s, 3,048 NVMe records, driver busy 3.0 s; 264 tokens 3.63 -> 1.84 s, a forced 32-row chunk in `qx-ds-check` 4.50 -> 1.88 s (the server chunks only stretches of 256 tokens or more). Left: ~700 blind reads and 2,356 used ones (2.0 s of disk) per 2k chunk. Predicting use from the previous chunk is weak (68% against a 50% base rate); predicting it from the next layer's router on this layer's FFN input is not (2026-11-14 row in Wins: a chunk's time follows the records it stages, 7,562 staged for 5,644 used at 256 rows before, 6,331 after). 16k prompt (chunk110): driver busy 2.2 of 4.8 s (KV-only chunk) and 4.4 of 7.3 s, ring-slot waits ~0.85 s each. The serve256k1 spread is a tier-state mode: a level-1 request is slow (10.2-10.9 s per 256 tokens) when it finds 31,600-35,600 of its selections in VRAM and reads ~1,150-1,480 records from NVMe, fast (9.0-9.7 s) at 36,400-39,900 and 685-955; the six warm-up requests decide which | 2026-11-12, notebook "Kernel modules loaded once per source..." and "A prefetch job skips the picks..." |
| Graph replay costs nothing; capture costs about 0.9 s per new width | replay 0.01 ms; capture fills CubeCL's info cache | 2026-11-04 |
| A distinct expert set per window grows sublinearly with width | about 100 per layer at 32 tokens, all 384 past about 500 tokens | 2026-11-04 |
| Run-to-run noise | same commit: decode 31.2-33.1 ms, prefill64 11.7-12.4 ms (spread up to 6%) | 2026-11-05, this scaffolding |
| **Narrow windows cost per position:** T = 1: 32.7 ms, T = 2: 23.6, T = 4: 19.4, T = 8: 15.7, T = 64: 12.3. A window of T positions costs T x w_T, so a draft-and-verify round breaks even at 41-46% acceptance before the drafter's cost and gains 6% at 75% acceptance (T = 4), 19% at 85%. | harness `narrow-windows`, 3 repeats, spreads 0.8-3.1% | 2026-11-06 |

## The chunk path's shape (2026-11-07, measured)

Per row at 4096-row chunks on the 16k prompt: the dense slab passes about 2.6 ms full stack / 1.2 KV-only, GPU-
bound (the dense GEMV ~40% of the chunk's GPU time even with a slab's groups in one block; attention ~16%; the
rest small); the expert phase about 1.7 / 0.85 ms, bound by dequantizing each expert once per (expert, 64-row
pass) and so halving with the chunk size; the chunk buffers 0.9 / 1.8 / 3.5 GB of VRAM at 2048 / 4096 / 8192
rows, taken from the expert tier (decode after a prompt +1% / +6%). Host launch count is not a bottleneck (the
z-group change halved it with no gain). Batching expert launches is slower (L2 and in-kernel waits). Prefetching
experts during the slab passes is neutral (the phase is not transfer-bound). A changed `mgemv.cu` costs ~9 min of
driver JIT at first use.

## Wins

| finding | effect | date / source |
|---|---|---|
| Prompt stretches of 64 tokens or more run in chunks (`QX_DS_CHUNK_MIN`, was 256): a full-stack window fetches its misses per window (the first request reads 5,951 NVMe records in windows, 2,823 in a chunk), so windows were the slow path for 64-255-token prompts. Chunks under 256 rows keep the exact kernels (`QX_DS_CHUNK_F16_MIN`: bit-identical to the windows) and post no blind prefetch job (`QX_DS_CHUNK_PREFETCH_MIN`: their rows route to a minority of the experts) | fresh three-slot 256k server, prompt time: 79 tokens 3.58 -> 1.54 s, 128: 7.02 -> 1.78, 186: 8.52 -> 2.22, 232: 9.35 -> 2.52 (of which no prefetch: 1.80 -> 1.54, 1.89 -> 1.78, 2.31 -> 2.22, 2.60 -> 2.52); warm repeats 79: 1.55 -> 1.45, 128: 2.32 -> 1.74; three-turn loadgen at three clients: wall 61.3 -> 55.9 s, TTFT median 2.95 -> 1.45 s, per-request decode 9.31 -> 8.43 tok/s (decoders wait behind chunks). Replies identical to the windows' in default mode (316/316 rows). At 32 the 42-46-token loadgen prompts lose (TTFT 0.92 -> 1.22 s) | 2026-11-13 morning, `setd-promo25-074000` |
| A prompt chunk's prefetch job follows the next layer's predicted routing (layer l + 1's router on layer l's FFN input, top 12 per row, summed to predicted rows per expert): picks in that order, and the ones predicted with fewer than 3 rows held back until the routing is known (`QX_DS_CHUNK_PRED=0` restores the old order; `QX_DS_CHUNK_PRED_T`, `QX_DS_CHUNK_PRED_MAX`). Chunks of 256 to 2,048 rows (768 with the exact kernels) | three-slot 256k server, fresh prompts: 298 tokens 2.46 -> 2.07 s, 413: 2.31 -> 2.06, 582: 2.61 -> 2.39, 872: 2.71 -> 2.50, 960: 2.94 -> 2.76, 1,638: 3.17 -> 3.14; repeats 2.19 -> 1.87, 2.57 -> 2.27; warmed single chunk 256 / 592 / 1,024 / 2,040 rows 1.84 / 2.30 / 2.65 / 3.24 -> 1.49 / 1.98 / 2.38 / 3.13-3.17 s; replies and logits byte-identical; set D 19.167 against 19.166 | 2026-11-14, `setd-promo28-013028`, chunk122 |
| Prompt stretches of 256 tokens or more run in chunks while other requests decode (the decoders get 4 steps between two chunks); before, such a prompt shared 128-wide full-stack windows with the decoders at 2.5 s per window (`worker_ds.rs`; `QX_DS_CHUNK_BESIDE=0`, `QX_DS_CHUNK_GAP`) | three-slot 256k server: a 10,031-token prompt arriving mid-reply 196.6 -> 9.0 s, that reply over 197 s (cut at 93 of 176 tokens by the client's 200 s limit) -> 18.7 s (alone 8.0); two long prompts at once: the second 72.0 -> 15.4 s, the first's reply 64.9 -> 9.1 s; a 413-token prompt mid-reply 23.9 -> 2.4 s. A decoder waits one chunk at a time (4-6 s; 9-16 s at 250k positions by prefill256k's chunks in `qx-ds-check`, not measured beside a decoder). Exact-knob and default row hashes equal alone vs beside; set D unchanged (no such prompt in it) | 2026-11-13, `setd-promo17-032618` |
| A prefetch chunk job skips the picks the routing leaves without rows (the engine publishes the routing as soon as it has the lists); NVMe batches of 8 records instead of 32 (a batch is read blind once submitted); eight event sets over the 64 landing places instead of two. `QX_DS_CHUNK_SKIP=0`, `QX_DS_CHUNK_READS=32`, `QX_DS_CHUNK_SETS=2` restore the old form | warmed prompt chunk: a forced 32-row chunk in `qx-ds-check` 4.50 -> 1.88 s (the server chunks only stretches of 256 tokens or more), 264 tokens 3.63 -> 1.84 s, 2,048 tokens 3.81-3.84 -> 3.22-3.26 s, 16k's mixed chunk 7.28 -> 6.96 s; three-slot 256k server 1,638 tokens 4.53 -> 3.37 s, 3,293 tokens 4.95 -> 4.47 s; set D within noise (19.213 against 19.166); logits byte-identical | 2026-11-12 (evening), `setd-promo15-215438`, chunk113 |
| One loaded module per kernel source (CubeCL compiled and loaded a source once per kernel id: 44 compiles, `mgemv.cu` eight times), shared with the raw launcher; PTX on disk in `~/.cache/quixiinfer/ptx`; NVRTC programs freed | fresh process, 16k prompt 710-739 -> 1,132-1,158 tok/s (Strata 841-880); server warm-up 24.5-25.0 -> 12.0-12.1 s; free device memory after a three-slot 256k warm-up 0.45 -> 1.21 GiB; ~1,440 ms of prefill256k's scored chunk (two compiles sat inside it) | 2026-11-12 (24090ba) |
| A chunk job's NVMe batches read while its RAM records copy (the disk and the link took turns before) | warmed 2,048-token prompt 4.81 -> 3.83 s; 16k 12.64 -> 12.02 s; three-slot 256k server 13,283 tokens 12.92 -> 11.34 s, 3,293 tokens 5.96 -> 4.95 s; bit-identical | 2026-11-12 (24090ba) |
| Asynchronous driver: mailbox-driven pool on its own thread, no per-layer host round trip | 9.2 to 12.0 tok/s at the time; the base of everything after | 2026-11-03 |
| Promotions off the critical path: event-ordered VRAM copies, async io_uring RAM promotions | steady token 64-68 to 52-57 ms | 2026-11-03 |
| Pool computes RAM experts while NVMe reads are in flight (`fetch_submit` / `fetch_wait`) | NVMe wait overlapped; part of 52 to 42 ms | 2026-11-03 |
| `hc_mixes` a block per output, then split 16 ways; `qx_ds_attn` a warp per key | GPU-only token 20 to 16.5 to 13.7 ms | 2026-11-03 / 04 |
| Engram rows in one io_uring batch per token | 4.1 to 0.1-0.3 ms per token (cold-start cost mostly) | 2026-11-03 |
| Flash-decoding MLA attention (`qx_ds_attn_part` / `merge`, 8-head groups x 12-key chunks) | GPU-only 24 to 13.1 ms at 1k; decode 43.3 to 31.9 ms per token | 2026-11-04 |
| Windows of up to 128 tokens, pool reads each expert once per window | prefill 34 to 21.8 ms per token at windows of 64 | 2026-11-04 |
| DMA-staged GPU share of RAM misses (6/8 windows, 3/8 decode, 96-record area, 4 streams, deduplicated) | prefill64 21.8 to 11.8 ms per token (1.85x), decode 31.4; bit-identical | 2026-11-04 |
| Fixed window widths (`WIDTHS`) captured at server warm-up | removes 0.9 s capture stalls from requests | 2026-11-04 |
| The window staged share (6/8) from width 2 instead of 8 (`QX_DS_STAGE_WIDE_MIN`) | -4.7% per position at width 2, -8.2% at width 4 (forced, outside noise); serve3 -2.9%, within noise: KEPT, not a scoreboard win | 2026-11-06 |
| **The prefill ceiling of this design:** windows of 128 with the full staged share (8/8) and a 256-record area: 10.8 ms per token against 12.3 at the standard 64-chunk (-12.6%), a 16k prompt at 18.2 against 21.8 (-16.6%); the pool idles and the copies plus the GPU expert kernels set the time. Opt-in only: the 2.4 GiB area costs serve3 +10% at concurrency 3 | notebook, "128-chunk bundle" | 2026-11-06 |

- **The 256k indexer scorer (2026-11-06):** `qx_ds_idx_score2` keeps the 32 index queries in shared memory
  instead of re-reading 32 cache lines per key row: decode256k 56.6 → 43.6 ms per token (-23%), prefill256k
  34.5 → 23.2 (-33%), bit-identical. The position cost of 256k was one kernel's memory pattern, not the top-k.
| Engine thread on its own core, pool on the other 15; the server's own threads kept off the engine's CPU | decode256k 44.1 -> 42.0; TTFT at concurrency 3 back to 1.8 s (was 2.8-3.3 in 4 of 6 with the full sweep) | 2026-11-08 / 2026-11-10, notebook |
| Widths 2-3 (a 3-slot server's concurrent decode steps) on a 4/8 staged share, window share from width 4 (`QX_DS_STAGE_SHARE_NARROW`, `QX_DS_STAGE_WIDE_MIN`) | serve256k3 106.3 -> 95.3 (-10.4%), concurrency 2 74 -> 67, serve3 96.5 -> 87.9: on 8/8 the pool idled | 2026-11-10 |
| Exclusive tiers (`QX_DS_RAM_DEDUP=2`): a VRAM victim is copied back from its slot into the RAM place of a duplicate; RAM promotions evict duplicates first | decode256k 42.0 -> 38.3 (-8.8%, same-binary interleaved), serve256k3 95.06 -> 90.01; NVMe records per 64 tokens 556 -> 285. Over half the NVMe fetches were promotion churn | 2026-11-10 |
| Set D on the new host state (driver 620.70.06, VM off, compaction_proactiveness 0) | score 30.047 -> 26.925: decode256k 38.25, prefill256k 5.70, serve256k3 89.53 | 2026-11-10, `setd-promo-052925` |
| Chunk-path experts on 8 side streams (`QX_DS_CHUNK_STREAMS`, `CUDA_DEVICE_MAX_CONNECTIONS=32`): N experts overlap on the card | 16k prompt 361 -> 461 tok/s, prefill256k 5.7 -> 5.2, bit-identical | 2026-11-10 |
| Prompt chunks' dense products as f16 x f16 -> f32 tensor-core GEMMs by the own kernel `qx_gemm16`, weights decoded to f16 once per layer and chunk (KL-bounded; `QX_DS_PREFILL_EXACT=1` exact) | prefill256k 5.2 -> 4.7; three-slot 256k server, 13.3k prompt: second chunk 16.2 -> 14.5 s; KL 0.005-0.009, top-1 97.5-100% | 2026-11-10 |
| Set D with both | score 26.925 -> 25.447: decode256k 38.5, prefill256k 4.7, serve256k3 91.07 | 2026-11-10, `setd-promo6-211440` |
| Prompt-chunk attention on f16 tensor cores (`qx_ds_attn16`, 16 heads x 16 keys per step, running softmax; `QX_DS_ATTN16=0` exact) | attention kernels 9.4 -> 1.8 s of GPU time per 16k prompt; KL against exact 0.0085-0.0091, top-1 95.1-98.4% | 2026-11-11 |
| `qx_gemm16` planes: the 8 output-projection groups in one launch, products of few output rows K-sliced to ~320 blocks, partials summed in order (`QX_DS_GEMM16_BLOCKS`) | `qx_gemm16` 8.25 -> 1.6 s per 16k prompt: a block's time is its K loop (44 us per 1,024 of K) however few blocks run | 2026-11-11 |
| Mixed chunks (`forward_chunk_keep`): the full stack on the last `prompt_keep` (2,796) rows only, the rows before them stopping at the last KV source in the same chunk (`QX_DS_CHUNK_MIXED=0`: two chunks) | the last 8,192 rows at 256k 38.4 -> 18.4 s; server, 3,293-token prompt 5.94 s against 6.94 s without the skip; bit-identical to the split plan and, with the exact knob, to no skip | 2026-11-11 |
| A chunk's Engram rows on a background thread (direct page reads; each engram layer waits for its rows), the thread on every CPU | 0.4-1.0 s per 4,096-row chunk with the card idle -> 0.06-0.2 s waited. The reads themselves do not get faster however they are issued | 2026-11-11 |
| Set D with these | score 25.447 -> 19.770: decode256k 38.6, prefill256k 2.2 (the mean over the last 8,192 rows), serve256k3 90.99; three-slot 256k server, 13.3k prompt 26.8 -> 13.0 s (495 -> 1,024 tok/s) | 2026-11-11, `setd-promo8-032500` |
| Evictions at a prompt chunk's tick capped at the per-token budget (`QX_DS_EVICT_CAP`): the chunk's tick evicted hundreds of VRAM residents and promoted none (the flex tail holds the free slots) | three-slot 256k server: the 64 tokens after a 13k prompt 3.76 -> 3.55 s, after a second one 3.33 -> 2.72 s; scored cases unchanged by construction. Cost: TTFT of the short requests after it +0.1 s | 2026-11-11 |


## Strata on this box (2026-11-07, same files and ids; details in the notebook)

Decode even (27.5 against our 29.2 warm / 23.8 cold-ish); DSpark gains nothing here (26.9); prefill 1k 148 against
our 91; prefill 16k 250 / 696 / 935 tok/s at their chunk 1024 / 4096 / 8192 against our 217 (exact chunks + skip).
Their prefill is FP16 tensor-core and not bit-identical to their decode; ours is bit-identical. The 4x at long
prompts is structural (chunk size, tensor cores, launch count), not tuning.

## Prefill exactness, priced (2026-11-07, notebook "What a non-exact (FP16 tile) prefill costs")

- Teacher-forced on the same ids: Strata's FP16-tile prefix changes nothing against its own no-tile path (16k text, last 4,095 positions: KL 0.026-0.029 against our exact path in all three arms, perplexity 2.53-2.54 against our 2.49). Its distance from us is engine-wide numerics (FP16 K/V, activations), not the prefill.
- Against the f32 reference on 1k: ours (int8 GEMV inputs and expert kernels) KL 0.011-0.020, top-1 96%, perplexity +0.1%; Strata 0.025-0.046, top-1 93-94%, perplexity within ±2%. The ds-checks' 0.13 is 59 positions of one prompt.
- So a bf16-tile dense path for prompts would cost at most ~0.03 KL against the exact path, likely under 0.01 with f32 accumulation and the exact K/V and expert paths; what it spends is the bit-identity contract (windows/chunks identity checks, prefix reuse equal to single steps), which would become KL-bounded checks.

## Correctness rules learned the hard way
- 2026-11-11: a chunk's dry passes of the slab widths write the ring and store cells of the positions in the
  window arrays, so the arrays must hold the chunk's first rows for the WIDEST slab: with a first slab narrower
  than a later one (a mixed chunk with 5 KV-only rows) the wider passes ran on stale positions and overwrote
  other positions' stored keys (0 of 64 rows; the breaker's test B).
- 2026-11-11: a thread spawned by the engine thread inherits its one-CPU mask: helper threads set their own
  affinity. And a helper's panic must stop the engine: a wait on its progress checks that it is alive.
- 2026-11-11: `prompt_keep` is measured from the prompt's END: anything compared bit for bit before the end (a
  tool's tail of single steps) must add its length to the keep.

- **Dry first-use windows must preserve incoming compressor pending pairs (2026-11-09).** At ratio 2, a dry
  window spanning positions 1,2 replaces pending position 0 with dry position 2; real position 1 then pools the
  wrong pair. The all-row mixed-sequence checker gives B 1/834 exact before the fix, 834/834 after saving and
  restoring pending buffers around the dry pass, and 834/834 with startup warm-up on the unfixed engine. The
  old checker inspected only one decode row per window and missed this. Rings have enough spare cells and
  committed rows are recomputed before use; the circular pending history is the exception. Server identity is
  still an open issue because its scheduler widths are already warmed at startup.
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
- **Every kernel the server will ever launch must be compiled and loaded in the warm-up, and the kernel set is
  sealed after it.** CubeCL reports a failed compile or module load only on the buffers a launch binds, and these
  launches bind none: a kernel that cannot load is retried on every launch and the window silently runs without
  it (2026-11-06, `qx_mgemv_t6` at width 6, out of device memory after the graphs). `qx_gpu::kernel::seal()`
  makes a later compile fatal; `QX_DS_VERBOSE` logs every compile; the VRAM headroom is `QX_DS_VRAM_RESERVE_MIB`
  (2.5 GiB) and the warm-up prints what is left.
- **A replay of n_swa tokens does not rebuild the rings (2026-11-06).** A layer's keys at a position come from the
  layer below's output there, which needs that layer's own 128-wide window: each layer adds 127 positions of
  warm-up, and the replayed positions rewrite correct store rows with wrong ones. `replay_to` now covers
  `replay_span()` = 128 + 40 × 127 = 5,208 tokens with the stores' rows saved and put back, and `Parked` carries
  the rings so an unpark at the parked position replays nothing. Every earlier replay check had replayed from 0.
- **No `cuEventSynchronize` or `cuEventQuery` from the driver thread while the engine thread records a graph.**
  CubeCL captures in global mode, under which those calls from any thread fail with
  `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` and invalidate the capture. The forced bench captures one width per run
  and never hit it; the server's warm-up captures ten widths back to back while the previous window's promotion
  copies are in flight, and panicked in the first `serve` case of the harness (2026-11-06). The fix is
  `Shared::capture`, a mutex the engine holds across start/stop capture and the driver takes around each event
  sync, plus a device drain between the eager run and the recording.

## Operating envelope: host memory (2026-11-06)

Measured from `/proc/PID/status` while serving (`quixiinfer serve --slots 3 --ctx 8192`, 100 GiB RAM tier):

| | RssShmem (pinned tier) | RssAnon | RssFile | total |
|---|--:|--:|--:|--:|
| idle after warm-up | 102.5 GB | 1.9 GB | 6.7 GB (the pack's mmap: page cache from the tier fill, reclaimable) | 111.1 GB |
| under load, no memory pressure (112G cap, five probes) | 102.5 GB | 1.9 GB | 6.7 GB | 111.1 GB |
| under load at a 110G cap (1.5 GB headroom) | 102.5 GB | 1.9 → 9.3 GB | 6.7 → 0.1 GB | at the cap, then killed |
| under load at a 118G cap (global pressure) | 105 GB | 14.7 GB | 0.06 GB | global OOM kill |

The forced bench (`qx-ds-check`, one slot) stays at 0.8 GB anonymous over 1,024 steps; `quixiinfer run` with 1 or
3 slots at 1.6-1.8 GB over 400 tokens. **The growth was NVRTC, retrying every launch of a kernel whose module
could not be loaded** (`cuModuleLoadData: CUDA_ERROR_OUT_OF_MEMORY`: the VRAM tier left 1.5 GiB, the warm-up's
graphs took 1.2 of it, and width 6 was not warmed), 63 MiB of host memory per attempt, and the window ran without
that GEMV. The first diagnosis, glibc arena fragmentation, was wrong: pinning malloc's thresholds removed the
64 MiB heaps but not the growth. Fixed by warming every width, sealing the kernel set (a later compile aborts),
and a 2.5 GiB headroom (notebook, "a kernel that could not load"). The rules that follow:

- A 100 GiB RAM tier is the forced bench's setting, where the process is 104-110 GB. **The server runs at 95 GiB**
  in the harness (about 106 GB RSS): at 100 it would be 111 GB, the wrong side of safe on a box that also runs the
  agent. 95 against 90 measured serve3 -5.4% and serve1 -5.0% (2026-11-06): residency is the serving lever, so
  the tier is worth its last safe gigabytes.
- **`MemoryMax=112G`, never 118G, for anything that pins the tier.** At 118G the box's global OOM killer fires
  before the cgroup's, and that is the case that takes the agent harness down. At 112G the cgroup kills only the
  offender; `OOMPolicy=continue` keeps the surrounding script alive to record it.
- Never let the session's background-task memory guard see a pinned tier: launch heavy runs detached (`nohup
  setsid`), as `bench.sh` is meant to be run.

## The neighbours' mechanics, from their code (2026-11-06)

Read for the "bridge the gap" plan (families 11-13). Numbers are theirs, on their boxes.

- **Strata prefill** (`~/git/Strata/src/dsv41/ds4_model.cu`): chunks of 8192 rows by default, raised to 16-32K on
  one GPU when the activations fit (`chunk_default` 3993); layer-major across chunks with the per-chunk state
  parked in pinned RAM; every expert a chunk touches is streamed **once per (chunk, layer)** through a 0.4-1.2 GB
  VRAM ring (`flow_pump` 4756) into `set_tc_kernel` (FP16 WMMA tiles, F32 accumulate, `ds4_iq.cu:821`), no CPU
  expert compute in prefill; dense GEMMs through cuBLAS FP16; the head only for the last 16 rows; and the
  **prompt skip**: layers 20-39 only for the last 2,668 rows (12915). Their chunk table on one R9700: 1024: 200,
  2048: 277, 4096: 353, 8192: 431 tok/s. Prefill is **not** bit-identical to their decode ("the two paths round
  differently", 12445; KL 0.1685 vs 0.1686).
- **DSpark** (DeepInfer `mtp/dflash.py`, Strata `ds4_model.cu:10748+`, colibri `deepseek_v4_dspark.inc`): a
  3-block draft model shipped separately (Unsloth `dspark-*-Q8_0.gguf`, `dflash`: experts 128 × 3 used in MXFP4,
  dense Q8_0, `fc` 5120 → 3×5120 from the trunk's layer 37-39 attention inputs, Markov bias head 2 × [129280, 256]
  bf16, confidence head). One non-causal pass drafts 5 tokens; the trunk verifies them as one (k+1)-row window;
  greedy acceptance = longest matching prefix plus the bonus token. Rollback must cover the trunk's SWA rings, the
  compressor's pending pair and the published index-key slot (colibri 4216-4268); the drafter's own 3 rings need
  none. Acceptance: Strata 31-32% per draft (1.6 per round; 4×R9700 31 → 55-69 tok/s, 3060 +2-13%, the 4090-48 GB
  user 16-22 → 40), DeepInfer 3.2 of 5 in production; `--dspark auto` is a *load* decision (off when the routed
  experts exceed 1.5× free VRAM). colibri's 60% guard over 10 proposals pauses drafting for 64 tokens.
- **Our own shape** (`crates/qx-engine/src/dsv41/mod.rs`, `kernels/moe.cu`): 384 experts, 6 used; the GPU expert
  kernels are a GEMV per (token, expert) selection with the token's activations in registers (gate/up one warp per
  weight row, grid `(2ff/8, k·t)`; down re-quantizes h per block), so an expert's bytes are read once per selection
  and the per-token HBM traffic (~2.3 GB) does not shrink with width; the CPU pool is per expert over tokens;
  bit-identity between windows and single steps rests on the per-(row, selection) integer block sums and the
  in-order float chain `acc += d·dq·isum` (`moe.cu:383-389`), which a grouped kernel can keep.

## Big-MoE-Small-Box (2026-11-08, code read; backlog family 15)

- exllamav3-based, 3090 + 64 GB + SN850X. DeepSeek exact: 10.6-11.8 tok/s decode, 248 tok/s prefill at 8k with
  fp16 tensor cores and an NVMe floor of ~15 s per 8192-token chunk (half its experts come from SSD each chunk);
  its 680-725 tok/s prefill is the lossy warm prefill (KL 0.023 vs its exact prefill). Not bit-identical between
  prefill and decode even in its exact mode (DSpark verify forwards round in other GEMM shapes).
- Its README's layer-ahead prefetch and staging pool are off for DeepSeek in its own launch spec; measured on GLM:
  ~30% useful stagings, 54% late, ~5% precision past rank 4. Consistent with our 23% previous-token figure.
- Its CPU lane is near-exact (KL 0.001) and an opt-in mode emulates the GPU's fp16 MMA rounding on the CPU at 5.5x
  the cost; its own doc names integer products + exact accumulation + one rounding (our design) as the way out.
- Mechanisms worth our time: elastic VRAM for prefill buffers (8192 chunks without the decode cost), the per-record
  PCIe landing latency (0.36 ms per 8-9 MB record there vs our 0.7 ms per staged record), DSpark acceptance data
  (75% at confidence 0.5), serving hardening and a disk tier for parked states.

## Rejected: with the reason, so they are not retried

- **A larger staging ring for long prompts** (2026-11-13): `QX_DS_STAGE_CAP=512` on a 16k prompt takes the ring-slot waits from 961 + 944 ms to 56 + 0 and leaves the chunks at 4,779 + 7,070 ms against 4,794 + 6,972: the waits are back-pressure from the GPU, not lost time.
- **Routing prediction for a chunk's prefetch, the variants that lost** (2026-11-14): a hold-back threshold calibrated on the hit rate per score class (70 / 85 / 95%: 2,040 rows 3.26 / 3.89 / 4.30 s against 3.13 at a fixed 3: the long chunks need recall); the predictor on chunks of 4,096 rows or more (+1.5% at 4,096, +3.8 to +5.6% at 8,192: its exact router products; on every 16th slab still +0.4 to +2.0%); a prediction for chunks under 256 rows (no prefetch job there: 1.54-2.55 s either way); the sharper later guess (the layer's own router before its attention: NVMe recall 91% against 84% at 2,040 rows) needs a sync per layer. The ceiling with a perfect prediction is 1.43 / 1.86 / 2.97 s at 256 / 592 / 2,040 rows against 1.49 / 1.98 / 3.13 now: at most 4-6% is left in the predictor.
- **A blind prefetch for short chunks** (2026-11-13): under ~256 rows the prefetch job reads experts the rows never route to (79 rows: 1,731 NVMe records against 1,050 without it; 1,805 against 1,523 ms). Above ~400 rows it pays (1,638 rows: 3,406 against 4,040 ms without).
- **Chunks for prompts under 64 tokens** (2026-11-13): `QX_DS_CHUNK_MIN=32` puts the 42-46-token loadgen prompts through a ~1.2 s chunk floor: first token 0.92 / 0.93 -> 1.22 / 1.22 s, three at once 13.35 / 12.55 -> 10.24 / 10.35 tok/s per request, TTFT p90 2.23 / 2.37 -> 3.93 / 3.96 s (two sessions each).
- **f16 kernels for chunks under 256 rows** (2026-11-13): 0.03-0.27 s faster per chunk (1.52 / 2.06 / 2.27 / 1.67 s against 1.54 / 2.22 / 2.52 / 1.78 exact at 79 / 186 / 232 / 128 rows) for KL 0.019 and top-1 95.7% against exact over 513 rows of one text (bound 95%: cleared by three rows; the last 129 rows 90.7%). Kept behind `QX_DS_CHUNK_F16_MIN=0`.
- 2026-11-12 (evening), on the prompt chunk's staging: registered io_uring buffers for the landing places (ENOMEM
  under the 8 MiB memlock limit, and the submits were blocking on the queue's tags, not on page pinning: 0.43 s
  for 2,965 records in batches of 8); a cap on the RAM records of a prefetch (`QX_DS_CHUNK_PRE_RAM`: 3.87-4.00 s
  against 3.89-3.93); read-ahead by one batch with batches of 8 (3.29-3.33 against 3.24-3.26 s); ordering the
  NVMe picks by "used in the previous chunk" (68% precise against a 50% base rate).
- 2026-11-12: three more ways to feed a chunk layer's experts sooner, all flat on a warmed 2,048-token prompt
  (3.84 s): the next layer's prefetch posted behind the current layer's experts job (`chunk-early.patch`: 3.81-3.90
  s; the ring's 256 slots are held until the current layer's experts finish); the next NVMe batch submitted before
  the current batch's copies are enqueued (`QX_DS_CHUNK_AHEAD` in `~/qx-wt/exp`: 3.87-3.90 s, NVMe wait unchanged at
  1.48 s); 16 side streams (3.83 s); and a 512-slot ring is worse (4.09 s).
- 2026-11-11: learning the tiers from the prompt, three forms, all flat in the three-slot 256k server: the
  chunk's selection counts as a per-token rate in the scores (`chunk-scores-fill.patch`), the layer's hottest
  NVMe-only records promoted to RAM during the chunk (NVMe selections on the first decode step 36 -> 38 -> 37 with
  12 / 32 records per layer promoted), the tail's residents without a RAM copy saved into RAM
  (`evict-cap+flex-save.patch`). The reply to a chat prompt routes differently from the prompt's text: the
  decode's first misses are not on the records the prompt used most.
- 2026-11-11: an eviction cap on every tier tick (1ac3e01): prefill128 10.5 -> 15.8, decode256k 38.6 -> 39.4.
  Window and decode ticks evict in bursts that the next ticks' fill budgets refill; only a prompt chunk's tick
  cannot be refilled (landed in that form).
- 2026-11-11: filling the flex tail's empty slots when a prompt ends with the hottest RAM records
  (`~/qx-chains/flex-fill.patch`): empty slots 102 -> 0 in a three-slot server, and nothing moved (64 tokens after
  a 13.3k prompt 3.70 s both ways, loadgen at 1 and 3 unchanged). The slow tokens after a long prompt are not the
  empty slots.
- 2026-11-11: faster Engram row reads for a chunk (six buffered rings; twelve threads of positioned reads; direct
  page reads on four rings): 0.4-1.0 s per 4,096 rows every way (positioned reads on threads: slower). Overlapping
  them with the card's work is what paid.
- 2026-11-10: cuBLAS for the chunk path's GEMMs (`QX_DS_GEMM16=cublas`, kept as a knob): the fastest arithmetic
  measured (16k prompt chunks 4.8 / 4.6 / 7.5 s against the own kernel's 6.3 / 6.0 / 10.0; prefill256k 4.0 in
  `qx-ds-check`), but its handle and kernel libraries (~1.4 GiB) fit no server's headroom: a panic on the first
  tree, then the exact fallback in a three-slot and a one-slot 256k server. Also: it loads a kernel library at a
  shape's first use (a hang with an expert job posted; needs a dry pass). Re-open only with VRAM budgeted for it.
- 2026-11-10: a slab's attention slices on side streams: -1% (the attention kernels fill the card). Whole-slab
  pipelining: blocked by `ring_w = n_swa + DS_TMAX` and the saved states that carry it.
- 2026-11-10: `QX_DS_ATTN=tile` in the chunk path: -4% on prefill, doubles the decode steps after the prompt.
- Validation lesson (2026-11-10): short-prompt identity sessions never run a chunk (`chunk steps 0`); a change to
  prompt processing needs a long prompt through the server (it crashed the first FP16 tree).
- 2026-11-10: draft-and-verify at 256k (long shot): windows of 2/4/8 of one stream cost 33.5 / 33.2 / 31.3 ms per
  position against 38.3 for single steps; break-even acceptance 66-79% of 7 drafts. Host expert bytes scale with
  positions. Re-open under ~12 ms per position at width 4-8 (bigger VRAM tier, second card).
- 2026-11-10: prefetch at width 3 in the server (`mix` K=6): 95.6-96.1 against 97.2-97.4 with one copy, 99.7 with
  three: the copies are not free in the gap. Re-open: a predictor that costs nothing (parallel to the wait kernel).
- 2026-11-10: NVMe write-back of VRAM victims (mode 1): halves the NVMe-served records but not the fetch wait
  (25 promotion reads per token contend with demand fetches); the device-to-host copy (mode 2) replaced it.
- 2026-11-10: `QX_DS_VRAM_PROMOS=4` / 0: decode256k 44.0 / 49.0 against 42.0: the promotions buy hit rate.
- 2026-11-09: latent rows to host for decode256k: VRAM slope 15.4/13.0/11.0 GiB -> 41.8/43.6/44.4, net ~0.
- 2026-11-09: decode staged share 5/8: 42.6 against 41.8 at 3/8 (2/8 equal).
- Candidate, off: prefetch `mix` K=6 one copy, decode256k -3.0% (before the exclusive tiers; re-measure on them).

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
  64 stays the prompt chunk. **Confirmed and bounded on 2026-11-06:** with a 192-record area, 128 ties 64
  (12.1 against 12.3-12.4, within noise) at a cost of 0.9 GiB of VRAM; neither width nor capacity beats 64's
  12.3. Prefill at 64 is bound by the copies and the GPU expert kernels, with the pool at 11 ms per token
  overlapped. Backlog family 5 closed.
- **A staged share above 6/8 for windows makes no difference (2026-11-04).** 7/8 measured 11.8 and 8/8 11.6,
  against 11.8 at 6/8. That is inside the noise, and the limit moves to the copies and the GPU expert kernels.
- **A decode staged share of 4/8 makes no difference (2026-11-04).** It measured 31.2 against 31.4 ms at 3/8,
  inside the noise. A staged expert's DMA costs about 0.7 ms against the pool's 0.22, so the balance point stays
  small.
- **Pinning the engine thread off the pool's cores (`QX_DS_CPUS`, `QX_DS_MAIN_CPU`) changed nothing (2026-11-03).**
- **SMT for the pool is slower:** 50.5 against 54 GB/s (2026-10, `qx-rambw`).
- **Speculative decoding to amortize the pool was rejected on analysis (2026-11-04).** Different tokens pick
  different experts, so decode windows do not reduce pool bytes per token. It is not measured. **Re-opened as a
  measurement on 2026-11-06** (backlog family 6): the analysis ignored that a window's staged GPU share and the
  pool's one-read-per-window make a 64-token window 2.7x cheaper per position than a single step, and Strata
  reports a user's 4090/128 GiB box going from 16-22 to 40 tok/s with DSpark on the same quant class. The
  narrow-window cost (`w_2`, `w_4`, `w_8`) decides it.

## Rejected elsewhere: measured by a neighbour on a similar problem, read before retrying here

These are not our measurements. They are cautions with a source.

- **`posix_fadvise(WILLNEED)` on the next step's experts while the current step computes** blocks against a
  drive that the demand reads already saturate: it cost 4 s of wall to save 1 s of disk (colibri,
  `docs/deepseek-v41.md`, "Two ways of hiding the reads that did not work"). Our io_uring batches per layer are
  the right shape; do not add readahead hints on top.
- **A pool of reader threads so each expert is multiplied as soon as it lands** did overlap the disk wait but
  slowed the matmuls 2.4x: a 5.9 MB `pread` is a kernel-side copy competing for the memory bandwidth the matmuls
  are bound by (colibri, same section). Our pool is at the RAM's bandwidth too; anything that adds a copy on the
  pool's cores costs what it saves.
- **Batched DMA submission** cut submission latency 83% and transfer time 1.3%, with no tokens-per-second gain
  found (Strata, `docs/BATCHED_DMA.md`).
- **Fewer, larger adaptive-tier rounds** cost 15-40% on cold and topic-change requests against a round every
  window; round size and decay alone were within noise (basalt, `docs/configuration/adaptive.md`). Cadence is
  the lever, and ours is already per token.

## Open: windows of 5 are not bit-identical to single steps (2026-11-06)

`QX_DS_WINDOW=5` on route1k 128 gave 125 / 128 bit-identical rows against single steps, while widths 6 and 7 gave
128 / 128 (and 8 and 64 do in `tools/ds-checks.sh`). Width 5 is not in `WIDTHS`, so the server never uses it, and
`run_t` splits wider windows into launches of 8 plus a remainder (14 = 8 + 6, 13 = 8 + 5), so a remainder of 5
could reach `qx_mgemv_t5` from widths 13, 21, 29... none of which are in `WIDTHS` either. Not urgent, but a
kernel that is wrong for one width is wrong, and the check is cheap: `tools/cmp_logits.py` on the dumps.
