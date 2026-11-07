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

## Strata on this box (2026-11-07, same files and ids; details in the notebook)

Decode even (27.5 against our 29.2 warm / 23.8 cold-ish); DSpark gains nothing here (26.9); prefill 1k 148 against
our 91; prefill 16k 250 / 696 / 935 tok/s at their chunk 1024 / 4096 / 8192 against our 217 (exact chunks + skip).
Their prefill is FP16 tensor-core and not bit-identical to their decode; ours is bit-identical. The 4x at long
prompts is structural (chunk size, tensor cores, launch count), not tuning.

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
