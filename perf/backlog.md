# Optimization Backlog

The beam holds three to five active idea families, best first. Pick from the top, and update the beam after every
concluded experiment.

Kill criteria are binding. When one fires, record the kill in `findings.md` with its numbers, then remove the
family. Each family's target comes from recorded data: a counter, a measured rate or a named baseline, never an
invented one. The aggregate score lives in `scoreboard.md`.

## Beam

### 1. Harness coverage: long context and serving (prerequisite)

- **Parent result:** the standard benchmark covers forced decode and prefill at 1k only. Recorded costs grow with
  context: prefill 23 ms per token at 1k against 46 at 12k, and decode 32 ms against 53 at 16k. Serving per-request
  decode is 14-17 tok/s warm, against the forced bench's 31 (2026-11-04). Neither is measured in a way the
  harness can diff.
- **Hypothesis:** a `long16k` case and a `serve` case make families 2 and 4 decidable. `long16k` runs
  `route16k.tokens` in windows of 64 with a 64-step decode tail. `serve` starts `quixiinfer serve --slots 3`, warms
  it, and drives `tools/loadgen.py` at concurrency 1 and 3.
- **Next action:**
  1. Add both cases to `bench.sh` and parsers to `perf_diff.py parse`.
  2. Vendor or document the token files, which are git-ignored (`perf.md` Open Item 1).
  3. Take a baseline.
- **Kill criteria:** none, since this is prerequisite work. It concludes when both cases run three repeats with a
  spread under 10%.
- **Concluded 2026-11-06.** Both cases run 3/3: long16k spread 0.5% (after moving its metric to the last 32
  steps), serve3 0.9%. Set B baseline recorded at 34.327. The cases found two serving bugs on the way (notebook).
  Still open from this family: engine counters for the serve rows.

### 2. The serving gap: per-request decode at about half the forced bench

- **Parent result:**
  - Per-request decode is 14-17 tok/s warm and 5-12 cold in serving, against 31 tok/s in the forced bench
    (2026-11-04, progress "Serving").
  - Short answers live in the warm-up regime, in which the tiers adapt over the first hundred tokens of a new text.
  - With several slots decoding, a window of 2-3 tokens from different sequences touches more distinct experts.
- **Hypothesis:** most of the gap is residency, not compute. Two levers may close part of it without hurting the
  forced bench:
  - Seed the tiers from the live profile (`--profile-out`).
  - Raise promotions per token while a request's routing differs from the resident set.
- **Evidence so far:** a start from a live profile file gave 9x fewer NVMe misses on the first request
  (2026-11-04). It has not been measured with repeats.
- **Next action:** after family 1, measure the `serve` case's per-request decode against the forced bench at the
  same tier settings. Split the difference with the counters into NVMe, pool and GPU, and the windows' distinct
  experts per token.
- **Kill criteria:** the gap is still unexplained by counters after two measurement passes. A residency change
  that does not lift the `serve` per-request decode by 8% also fires the kill.

### 3. Decode: fewer CPU experts per token through residency

- **Parent result:** decode is pool-bound. The pool takes about 20 of the 31 ms, at 0.22 ms per expert from RAM
  plus NVMe waits. Steady routing late in route1k is 76% VRAM, 22% RAM and 1.8% NVMe at 21 GiB (2026-11-04).
- **Hypothesis:** each GiB of VRAM is about 100 more resident records, and the hottest misses go first. Two things
  free VRAM:
  - **Sizing the compressed stores to `max_ctx / ratio` rows per slot instead of `max_ctx` rows.** At 3 slots and
    an 8k context that is about 1.8 GiB, measured from the allocation code: `slots * max_ctx * 640 * 4` bytes per
    compressing layer.
  - **The staging area's 1 GiB in decode.**

  A fair estimate is +2-4 points of VRAM hits, worth 1-2 ms per token.
- **Evidence so far:** allocation arithmetic only.
- **Next action:**
  1. Size the stores by ratio, keeping parking and unparking in step: parking copies `pos / ratio` rows.
  2. Run `bench.sh` with `QX_DS_VRAM` at the freed amount and in the serving case.
- **Kill criteria:** decode improves by less than 3% at the freed VRAM, or the change breaks bit-identity.

### 4. Long-context step costs

- **Parent result:**
  - The GPU-only single step is 13 ms at 1k and 24 ms at 16k.
  - Prefill windows go from 3.1 to 3.8 ms per token GPU-only.
  - The pool carries most of the end-to-end rise, at 46 ms per token for prefill at 12k (2026-11-04, the 16k
    entry).
  - **Corrected 2026-11-06 by the harness:** steady decode at 16k is 35-37 ms per token (the 53 was
    capture-contaminated), so the decode rise from 1k is about 10%, not 65%. Prefill averaged over 0-16k is 19.5
    against 12.3 at 1k: the long-context cost is mostly in prefill, and that is where this family should look
    first (the per-window cost by position is in the `pos N:` lines of the long16k logs).
- **Hypothesis:** the GPU part is attention over 128 window rows plus 512 selected rows, and indexer scoring
  linear in the visible compressed rows; an nsys kernel table will name it. The pool part needs explaining first:
  the expert count per token should not depend on position.
- **Next action:** after family 1's `long16k` case exists, take an nsys kernel table of a decode step at 16k, and
  split the pool rise with the counters.
- **Kill criteria:** the rise turns out to be text-dependent routing rather than position, because route16k's
  later text is harder to keep resident. Then this family merges into family 3.

### 5. Prefill beyond 11.8 ms per token

- **Parent result:** with a 6/8 staged share, prefill64 is balanced between the copies, the GPU expert kernels and
  the pool. 8/8 gives 11.6, and windows of 128 lose because they overflow the 96-record staging area (2026-11-04).
- **Hypothesis:** a larger staging area, sized from VRAM freed by family 3, lets windows of 128 stage their
  distinct experts. Windows of 128 halve the pool's bytes per token against 64 for the RAM part. The target is
  prefill at or below 10 ms per token.
- **Next action:** sweep `QX_DS_STAGE_CAP` at 128, 160 and 192 with windows of 64 and 128, using
  `bench.sh --cases prefill64,prefill128 -- QX_DS_STAGE_CAP=N`, within the VRAM family 3 frees.
- **Kill criteria:** no width and capacity setting beats 11.8 by 8%. Prefill is then copy-bound at about 14-25 GB/s,
  and further gains need the NVMe tier changed.

### 6. Long shot: the economics of a verify window (DSpark-style drafting), measurement first

- **Parent result:** `findings.md` rejected speculative decoding on analysis (different tokens route to different
  experts, so a decode window does not cut pool bytes per token). Three neighbours measured otherwise:
  - Strata's user report of the same UD-IQ2_XXS-class model on an RTX 4090 48 GB with 128 GiB RAM: 16-22 tok/s
    with `--dspark auto` (which chose off) against **40.2 tok/s with `--dspark on --dspark-conf 0.65`**
    (`~/git/Strata/docs/DEEPSEEK41.md`, one report). Strata's own 3060 box: +2-13%.
  - colibri's CPU engine: 17% faster with drafting and an acceptance guard at 60%, slower with the guard off
    (`~/git/colibri/docs/deepseek-v41.md`, "DSpark"). Their lesson: stopping pays as much as starting.
  - DeepInfer ships `DEEPINFER_USE_NEXTN_DECODE=5` with a confidence head that schedules verification.
- **Why it might work here anyway:** our prefill64 runs 11.9 ms per token against decode's 32: windows are 2.7x
  cheaper per position because the staged GPU share and the pool's one-read-per-window amortize the experts. A
  verify window of 1 + 5 drafts is a window of 6; what matters is the ms per position of a *narrow* window, which
  was never measured (`WIDTHS` has 2, 3, 4, 6, 8, but the harness only ran 32-128).
- **Measurement (no engine change, about 15 minutes):** `bench.sh --cases prefill2,prefill4,prefill8` once
  those cases exist (one line each in `bench.sh`'s table). With `w_T` the ms per token of a window of T tokens, a
  round of T positions accepting `a` drafts on average costs `T * w_T` for `1 + a` tokens, so drafting pays when
  `T * w_T / (1 + a) < 32`. At T = 6 and w_6 = 15, break-even is a = 1.8 of 5 (36%); at w_6 = 25, a = 3.7 (74%).
  Record the break-even curve in `findings.md` either way.
- **Blockers if the curve says go:** the UD-IQ2_XXS GGUF carries no MTP tensors (all six shards scanned on
  2026-11-06: blocks 0-39 only). The drafter is a separate Unsloth DSpark Q8_0 GGUF of about 8 GB, not on the box
  (ask before downloading). Then: the head's three stages (window-only attention over the main stream's keys, 128
  own experts, 3 per token), rollback of the SWA ring, the compressor's group slots and the index-key slot for
  rejected drafts (colibri documents all three), greedy acceptance first, an acceptance guard like colibri's.
  That is weeks, not days; the measurement decides whether to start.
- **Kill criteria:** w_6 above 27 ms per token (break-even acceptance above 80%), or Strata's acceptance numbers
  for this quant turn out below the break-even.
- **Measured 2026-11-06 (notebook, "Narrow-window cost per position"): DEFERRED.** w_2 = 23.6, w_4 = 19.4,
  w_8 = 15.7 ms per position; break-even acceptance 41-46% before the drafter's cost; a 6% gain at 75% acceptance,
  19% at 85%. Not a kill, since 85% acceptance is plausible for MTP heads, but not worth weeks without a measured
  acceptance for this quant, and the drafter weights are not on the box. Re-open with: a measured per-position
  acceptance of 80% or more, or w_4 under 15 ms per position from window-side work (the staged share at widths
  2-4 is measured next).

### 7. Ideas from the neighbours, not yet ranked

Read on 2026-11-06 from `~/git/{colibri,Strata,basalt,DeepInfer}`. Each needs a translation to this box before it
becomes a family.

- **Wider prompt chunks on a layer-major path** (Strata: `--prefill-chunk 8192` gave 11-15% at 4k-32k prompts, a
  "layer-major" path for long prompts). Ours: windows of 128 lost to 64 only because of the 96-record staging
  area. Family 5's `STAGE_CAP` sweep should go to 192 and 384 with windows of 128, and beyond that a layer-major
  chunk of 256-1024 tokens would read each expert once per chunk from the pool. The pool is compute-bound past
  128 tokens per expert (4 TMAC/s), so the gain has a ceiling; compute it before building.
- **Grouped expert kernels with the window's activations staged in shared memory** (Strata EXPDEC, bit-identical,
  cut the per-extra-token cost of a resident expert by about 60% on a 2080 Ti). Prefill64 is partly bound by our
  GPU expert kernels over `slots`/`slots_stg`; an nsys table of a 64-token window names whether the multi-GEMV's
  per-token cost is the issue.
- **Batched DMA submission** (`cudaMemcpyBatchAsync`, Strata): submission latency -83%, transfer time -1.3%, no
  tokens-per-second gain established. Low value for us: 96 records over 4 streams is not submission-bound.
- **Adaptive tier cadence** (basalt): rounds every window matter far more than round size; 160 swaps against 96
  was within noise; a saved profile helps a cold start only slightly on one card. Ours promotes every token
  already (`QX_DS_PROMOS=32`). The transferable part is for family 2: a profile seeded from the live
  `--profile-out`, measured with `serve3` repeats.
- **Reads issued as one batch per step, depth 8, O_DIRECT** (colibri). Ours already batches io_uring reads per
  layer; the NVMe share at steady state is 1.8% of routed experts. Only the serving cold start is read-bound.
- **The first-touch floor** (colibri: a 16-token turn touches 3,371 distinct experts; no cache size removes the
  first read of each). For serving this is what the profile seed and parked prefixes address, nothing else.
- **Host memory** (ours, 2026-11-06, `findings.md` "Operating envelope"): the anonymous growth was NVRTC retrying
  a kernel whose module could not load for lack of VRAM, and is fixed. What remains: the 6.7 GB of file-backed resident pages at idle (the GGUF mmap kept
  warm after load, or CUDA's device mappings; `pmap -x` on the idle server says). If they are the GGUF's, dropping
  them after load (`madvise(DONTNEED)`, or reading the dense weights into VRAM without keeping the map) returns
  6.7 GB to the RAM tier, about 700 more resident experts. Also: the pool's `vec![0f32; ff]` per expert per token
  is allocator churn a per-thread scratch would remove.
- **SWA bounded replay** (DeepInfer `YOCO_PREFILL_TRUNCATION_MODE=compact`, keep 140 tokens): recompute only a
  bounded suffix when a decoder's state is rebuilt. Ours rebuilds the ring from the host latent history in
  `rewind`; the equivalent is already in place, but a measurement of `rewind`'s cost at 16k belongs to family 4.

## Parked
- **Parking idle sequences in host memory** (work in progress, 2026-11-05). This is a serving capability: prefix
  reuse across more conversations than slots. It is not a speed change to the standard benchmark. Its
  measurement belongs to the `serve` case: time to first token on a returning conversation.
