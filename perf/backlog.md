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

## Parked

- **Speculative decoding (DSpark / MTP) for throughput.** Rejected on analysis in `findings.md`. Re-open only with a
  measured routing overlap between consecutive tokens.
- **Parking idle sequences in host memory** (work in progress, 2026-11-05). This is a serving capability: prefix
  reuse across more conversations than slots. It is not a speed change to the standard benchmark. Its
  measurement belongs to the `serve` case: time to first token on a returning conversation.
