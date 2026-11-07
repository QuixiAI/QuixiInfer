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
- **Measured 2026-11-06** (notebook, "Where a serve3 step goes"): a serve3 step is 104 ms, device wait 97-103,
  pool 96 of which NVMe wait 31 and staging wait 13.6; host overhead under 1 ms. Shares VRAM 61 / staged 24 / RAM
  11 / NVMe 3.9%. Residency it is. Running: `QX_DS_PROMOS=64`, `QX_DS_STAGE_SHARE=4` on the serve case. Next
  code lever: family 3's store sizing (VRAM back to the tier), then per-layer NVMe prefetch if the wait is
  latency-bound. **Results:** PROMOS=64 flat, STAGE_SHARE=4 flat, RAM tier 95 GiB -5.4% (kept as the serve
  setting). Store sizing killed on arithmetic (family 3); NVMe reads are already at the drive's rate per record
  and the next layer's routing is unknown, so no prefetch lever. What remains for serving is structural:
  per-request hot sets (three conversations route wider than one), or more memory. Prefix reuse reporting 0 is by design: `worker_ds` reuses a shared prefix only from 64 tokens, and loadgen's prompts are 40-46 tokens; not a bug.
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
- **Store sizing killed on arithmetic, 2026-11-06.** Only the four KV source layers (2, 8, 14, 20) hold compressed
  stores, not every layer: 4 layers x 3 slots x 8,192 rows x 640 floats x 4 B = 240 MiB in all, so sizing by the
  ratio returns about 0.2 GiB, about 20 records, under half a percent of the tier. Not worth the code. The
  staging area (0.9 GiB) is shared with prompt windows, which need it. What does move serving residency, from
  the serve3 decomposition: the RAM tier (90 against the forced bench's 100: `serve-ram95` queued) and the
  promotion rate (`serve-promos64` running).

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
- **Closed 2026-11-06, solved:** the kill criterion fired in the best way. The rise was the tier's residency lag,
  caused by a per-step score decay (a 32,768-token memory in 128-token windows); decay per token landed:
  long16k 38.5 to 33.8 (decode at 16k now equals decode at 1k), 16k prefill 18.4 to 14.4 (notebook, "score
  decay per token"). What remains for long contexts is the host latent history (replaced by replay in the same
  patch) and the 256k runs in progress.

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
- **Killed 2026-11-06** (notebook, "Staging area 192 records"): cap 192 makes windows of 128 tie 64 (12.1 against
  12.3) and does nothing at 64. **Reopened and settled the same day:** share 8/8 with cap 256 at width 128 gives
  10.8 ms per token (-12.6%) and a 16k prompt -16.6%, but the 2.4 GiB area costs serve3 +10%, so it is an opt-in
  (README), not a default. What would give both is a staging area that borrows tier VRAM only while a wide
  window runs (family 7).

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
- **Re-opened 2026-11-06 (the user's direction: bridge the gap to Strata's 40 tok/s).** Facts from the code
  (`findings.md`, "The neighbours' mechanics"): DSpark is a real 3-block draft model in Unsloth's
  `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (8 GB; architecture `dflash`; 3 blocks of 128 experts, 3 used, in
  **MXFP4**, dense parts Q8_0, a Markov bias head of 2 × 66 MB and a confidence head), now on the box at
  `~/models/DeepSeek-V4.1-Flash-1017-GGUF/dspark/` (sha256 matches Strata's pin). It drafts 5 tokens in one
  non-causal pass fed by the trunk's layer 37-39 attention inputs; verification is one (k+1)-row window;
  Strata measures 31-32% per-draft acceptance (1.6 per round), DeepInfer 3.2 of 5. Our break-even is 41-46%.
  **Next action, measurement first:** pack the drafter (an MXFP4 expert kernel is needed; `pack_gguf.py:725-773`
  in Strata has the tensor map), run it beside normal greedy decode on code, agent and chat prompts and log the
  acceptance of its 5 drafts against the real next tokens; with the w_T table that gives the speedup before any
  verify path exists. Build the verify path only past +8%: the fast rewind covers k ≤ 16 but the compressor's
  pending pair must be saved per position (odd rewinds), plus DeepInfer-style confidence truncation.

### 8. Dense GEMV efficiency at decode (new 2026-11-06, from the nsys table)

- **Parent result:** in a decode step the dense GEMVs (`qx_mgemv`: attention projections, shared expert,
  router) take 6.7 ms across 530 launches, 13 per layer at 12.7 us each, and they are serial with the pool. A
  step reads 6.5 GB of dense weights (Q6_K/Q5_K: `attn_output_b` and `attn_q_b` 1.65 GB each, `attn_output_a`
  1.32, the shared expert 1.33), a 3.6 ms floor at 1.8 TB/s: the kernel runs at 54% of peak (notebook, "nsys
  kernel tables").
- **Hypothesis:** better memory access in `qx_mgemv` (vectorized row loads, the split-K choice per shape, fewer
  and larger launches per layer) takes the dense time from 6.7 toward 4 ms; that is up to -8% on decode and
  long16k, bit-identical because the sums do not change order within a row. `qx-kbench` has no DeepSeek shapes
  yet: add a group of the four big tensors first.
- **Per launch shape (the nsys sqlite, 192 decode steps):** grid 1024 x 8 launches per layer at 5.2 us each
  (1.66 ms per step, 23% of the dense time: the small projections, router and mixes, at launch-latency size);
  grid 5120 x 2 per layer at 21 us (1.70 ms: `attn_output_b` and the shared expert's down, split-K one row per
  block); grid 4992 x 1 at 28 us (1.12 ms); grid 4096 at 30 us (0.96 ms); the output head once per step at 356 us
  (540 MB: 1.5 TB/s, fine). The big matrices run at 1-1.4 TB/s; the eight small launches per layer are the
  cheap target: one multi-segment `MGemv` launch per group of independent projections (q_a and kv_a read the same
  normed input) would save most of their 1.66 ms, about -4% on decode.
- **Read 2026-11-06:** the eight grid-1024 launches are `attn_output_a` applied per head group (`o_groups` = 8,
  `wo_a_groups` in `dsv41/mod.rs`), each with its own input slice of the attention output, 1,024 rows x 4,096
  columns of Q6_K at 0.8 TB/s. One launch for all eight needs a per-segment input offset in `mgemv.cu` and
  `ops::Seg` (segments share one x today) and must keep the split-K one-row-per-block reduction (`lanes` 0) so
  every row's sum order stays what the exact-reference check pins. Expected gain: 17 us per layer of bandwidth
  plus seven launch gaps, 0.7-1.3 ms per step, 2-4% of decode: at or below the bar for a kernel change.
- **DEFERRED 2026-11-06:** not worth the kernel change alone. Reopen if another launch fusion in the same layer
  (the `g_x` group's small segments, the hc mixes) can ride along to clear 8%. **Blocked the same day:** the
  one-launch version built but its PTX took the driver's JIT more than 12 minutes at module load (notebook, "stalled
  in the driver's PTX JIT"); time `ptxas` offline before touching `mgemv.cu` again.
- **Mechanics, read 2026-11-07:** CubeCL 0.11 compiles the *whole* flattened module to PTX with NVRTC for every
  kernel id and loads it with `cuModuleLoadData`, so the driver runs ptxas over all of `mgemv.cu` (25 entries) or
  `moe.cu` (93) once per PTX hash and caches the result in `~/.nv/ComputeCache`; that cache is capped at 1 GiB by
  default and the engine's variants evicted each other (a binary swap cost 20 minutes before its first step;
  `bench.sh` now exports `CUDA_CACHE_MAXSIZE=4294967296`). A CubeCL-compatible fix for the first-use cost: have
  `CudaSource::source` return the module with every *other* `extern "C" __global__` body stripped (brace-matched),
  so each kernel id's PTX holds one instance and a run pays ptxas only for the entries it launches; the server's
  warm-up still pays for all of them. Tooling, not the 256k target: do it on a quiet day with a ds-checks run.
- **Kill criteria:** achieved bandwidth already above 70% of peak on the shapes that carry the time, or any
  change that breaks `tools/ds-checks.sh` line 3.

### 9. 256k decode: the position-proportional GPU time (new 2026-11-06, the set D target)

- **Parent result:** the first forced 256k pass (decay per token): prefill 14 ms per token at 24k rising to 28 at
  246k, decode 56.3 ms per token at 262k against 34 at 1k, with the tier shares and the pool's time flat along the
  curve: the growth is GPU-side and proportional to the visible context (notebook, "first forced 256k pass").
- **Hypothesis:** the lightning indexer: per token and per indexer layer (eight), `qx_ds_idx_score` scores every
  visible compressed row (one warp per row), `qx_ds_topk` selects the top 512 (one block of 1,024 threads, a
  four-pass radix select over up to 262k scores) and the candidate path adds a block scan, a block top-k and a
  mask; the arithmetic says 1-3 ms per token at 256k, the measurement says 22, so the kernels are far from their
  bandwidth (the one-block top-k is the first suspect: a single SM streams 1-4 MB per call, eight to sixteen calls
  per token).
- **Next action:** the nsys kernel table of a resumed 256k decode step (queued); then the kernel that carries the
  time: a multi-block top-k (per-block local selection, then a merge), or a threshold from the block maxima that
  skips most rows, keeping the reference's tie rule (lowest index) so line 3 of ds-checks stays exact.
- **Kill criteria:** the profile shows the time outside the indexer (then re-aim), or a change that breaks the
  bit-identity of the selection.
- **2026-11-06:** the nsys table named `qx_ds_idx_score` (8.8 ms per call, 8 per step, 33% of GPU time; the
  top-k is 0.3 ms). `qx_ds_idx_score2` (queries in shared memory, float4 key reads, same FMA order) is bit-identical
  at 262,144 and in ds-checks.
- **Concluded 2026-11-06: LANDED.** decode256k 56.6 → 43.6 ms per token (-23%), prefill256k 34.5 → 23.2 (-33%),
  3 repeats each way on one binary, spreads 2.8-9.1%, ranges disjoint. What remains of the position cost (43.6 at
  262k against 34 at 1k, part of it the cold resumed tier) re-opens as a smaller family once the states carry
  the tier ranking: the per-token `qx_ds_topk` (8 x 0.3 ms) and the candidate path are next.

### 10. The per-step uploads synchronize the whole context (new 2026-11-06)

- **Parent result:** the tail's phases at 256k: uploads 7 ms in the first single steps after the prefill, 1.4 ms
  on average, against 0.2 at 1k. The three per-step uploads (token ids, positions, slots: a few hundred bytes) go
  through `DeviceBuf::upload_at`, which does a synchronous copy and then `cuCtxSynchronize`, so each step waits
  for the driver's in-flight promotion copies on the copy streams.
- **Hypothesis:** writing the three small arrays into a device-mapped pinned buffer the kernels read directly (as
  the mailbox does) removes the synchronize: up to 7 ms a step while the tier adapts, about 1.4 ms (2.5%) steady at
  256k, and the same mechanism in serving after every prompt.
- **Next action:** a mapped-buffer path for `tok`, `pos` and `slot`; measure `decode256k` and `serve256k3`.
- **Kill criteria:** under 3% on both.
- **Concluded 2026-11-06: FLAT** (decode256k -1.5%, serve256k3 -0.3%, inside the spreads). Kept as the default
  (no cost; the stall it removes is in the first steps after a prompt, which no scored case samples). Removed
  from the beam; a TTFT case would be where it shows.

### 11. The long-prompt layer skip: KV-only windows (new 2026-11-06, from Strata's `prompt skip`)

- **Parent result:** `kv_source_layers` = 2, 8, 14, 20 and `compress_ratios` 0,0,2×18,1×20: layers 21-39 have no
  store of their own and see earlier positions only through layer 20's compressed rows and index keys plus their
  own 128-wide sliding windows. A prompt position's output above layer 20 feeds only its own logits and the next
  127 positions' windows, layer by layer up the stack. Strata runs layers 20-39 for the last 2,668 rows only
  (`ds4_model.cu:12915`, test `test_prompt_skip.py`).
- **Hypothesis:** windows with at least `prompt_keep()` = 2,796 prompt positions after them can stop after layer
  20's ring, store and key writes: about half the routed-expert bytes and GPU time of such a window, with every
  kept position bit for bit the full pass's. `prefill256k` 22 → ~12 ms per token, the 16k prompt 14.4 → ~8.
- **Next action:** in the tree (`forward_segments_opt`, `QX_DS_SKIP`); the route16k identity and timing checks
  and ds-checks are queued; then `prefill256k` and `long16k` with and without; then the serving policy (today
  only prompt-only windows qualify; a long prompt beside decoders gets no skip until windows alternate).
- **Kill criteria:** any kept-tail logit differing from the full pass, or under 25% on the 16k prompt.

### 12. Chunked expert-major prefill (new 2026-11-06, the Strata-class prefill win)

- **Parent result:** our GPU expert kernel is a GEMV per (token, expert) selection (`moe.cu:394-497`: grid
  `(2ff/8, k·t)`, an expert's weights re-read once per selection, ~2.3 GB of HBM reads per token at any width),
  the staging cap counts selections, `qx_moe_post` needs all T blocks co-resident (T < ~1000), the mailbox and the
  NVMe landing places (64) bound a window at 128. Strata streams each expert once per 8-32K-row chunk into
  tensor-core GEMMs and measures 200 → 431 tok/s from chunk 1024 → 8192 on one card.
- **Hypothesis:** a chunk of 4K+ rows processed layer-major: dense and attention in 128-row slabs with today's
  kernels (unchanged bits), the MoE of the whole chunk per layer expert-major (each touched expert streamed once
  through a VRAM staging ring, GPU only; a grouped kernel that keeps each (row, selection)'s integer block sums and
  float chain, so single steps and chunks stay bit-identical), the residual streams (80 KB per row) the only
  per-row state across layers. Bytes per 4K chunk ≈ 76 GB of RAM experts over PCIe + 56 GB from NVMe ≈ 6.5 s,
  so ~1.6 ms per token at 4K, ~0.9 with family 11; the 1k prompt ~6 ms per token (1.8x), long prompts 7-12x.
- **Next action:** after family 11 lands: the grouped expert kernel first (measurable alone at T = 128: it also
  cuts the per-window HBM traffic), then the per-layer slab graphs and the chunk-wide MoE phase, then the whole-
  layer NVMe streaming (the 64 landing places and QUEUE=64 are per window today).
- **Kill criteria:** the grouped kernel not bit-identical to the per-selection one, or under 1.5x on a 4K prompt.
- **2026-11-06, night:** the prototype is bit-identical at 1k and 16k in every arm; 16k prompt 17.9 → 7.0 (chunks
  of 2048, grouped kernels, prefetch) → **4.6 ms per token** with the layer skip inside chunks (217 tok/s). Grouped
  kernels -10% (landed in the path), prefetch neutral, batched launches slower (rejected). Measured remainder per
  2048-row chunk: host enqueue of the slab passes 5.8 s (~80 us per launch), expert kernels 6.5 s (warp
  reductions), dense GPU remainder 0.8 s. Next: a looped multi-token GEMV launch and per-slab graphs; a
  lane-per-row expert kernel. Not yet in the server (ds-check `QX_DS_CHUNK=N` only).
- **2026-11-07, morning:** exact int8 tensor-core expert kernels landed (expert phase 3.75 → 2.86 s per 2048 KV-only
  rows); z-group GEMV launches landed (exact, neutral: the slab passes are GPU-bound, `qx_mgemv_t8` 40% of GPU
  time, attention 16%); chunk 4096 in the harness cases (prefill256k 21.5 → 7.6), 2048 in the server (prompt-only
  steps). Remaining per row at 4096 KV-only: dense 1.4 ms, experts 0.86. Next: per-pass `gu`/`hq`/`hd`/`out`
  scratch (VRAM), a tiled exact dense GEMV, attention key sharing per slab, the chunk path beside decoders.

### 13. DSpark: see family 6 (re-opened 2026-11-06); measurement first, then the verify path.

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
  a kernel whose module could not load for lack of VRAM, and is fixed. The 7-8 GB of file-backed resident pages
  are the expert pack's mmap (`smaps` on a serving process: 7,955 MiB of `.qxp`, 97 GB of `/dev/zero` pinned
  tier, 2.1 GB anonymous): page cache left by the tier fill reading through the map, clean and reclaimable, so
  not a tier lever; an `madvise(MADV_DONTNEED)` on the map after the fill would make RSS honest. Low priority.
  Also: the pool's `vec![0f32; ff]` per expert per token is allocator churn a per-thread scratch would remove.
- **SWA bounded replay** (DeepInfer `YOCO_PREFILL_TRUNCATION_MODE=compact`, keep 140 tokens): recompute only a
  bounded suffix when a decoder's state is rebuilt. Ours rebuilds the ring from the host latent history in
  `rewind`; the equivalent is already in place, but a measurement of `rewind`'s cost at 16k belongs to family 4.

## Parked
- **Parking idle sequences in host memory** (work in progress, 2026-11-05). This is a serving capability: prefix
  reuse across more conversations than slots. It is not a speed change to the standard benchmark. Its
  measurement belongs to the `serve` case: time to first token on a returning conversation.
