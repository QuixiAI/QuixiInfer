# Optimization Backlog

The beam holds three to five active idea families, best first. Pick from the top, and update the beam after every
concluded experiment.

Kill criteria are binding. When one fires, record the kill in `findings.md` with its numbers, then remove the
family. Each family's target comes from recorded data: a counter, a measured rate or a named baseline, never an
invented one. The aggregate score lives in `scoreboard.md`.

## Beam

### Queue as of 2026-11-11 (after set D 19.770; the prompt path is bounded by the experts now)
1. **The tokens right after a long prompt** (partly done 2026-11-11: the chunk-tick eviction cap, 3.76 -> 3.55 s
   and 3.33 -> 2.72 s for the 64 tokens after a first and a second 13k prompt). What is left is the three-slot
   server's first steps (174, 102, 82, 79 ms): a tier that has not seen the reply's routing (first step: 18% of
   selections in VRAM, 16% on NVMe only). Learning from the prompt's counts does not help (three forms tried).
   Ideas not tried: a faster promotion budget for the first tokens of a reply; keeping a per-conversation set
   of the previous reply's experts; a larger RAM tier share for the server.
2. **Chunks while another slot decodes**: the policy "chunk path only when no slot decodes" makes a second long
   prompt wait in windows (10,031 tokens: 54.8 s; slot 0 beside it 21.8 s against 13.0 alone). Try: chunks of a
   bounded size between decode steps (a decoder waits one chunk), or the f16 kernels for prompt windows.
3. **The full-stack floor (~6 s per pass, however few rows)**: 40 layers x ~300 experts x 9.5 MB. Fewer bytes:
   route a short prompt's layers through the resident experts first; 16,384-row chunks for prompts over 8k (one
   pass less); the upper layers' experts prefetched while the lower layers run.
4. **Expert arithmetic in the chunk path**: 46-60 ms per layer of a 4,096-row chunk on int8 tensor cores (exact);
   an f16 tile kernel is allowed there (KL-bounded) if decode of the IQ formats to f16 is cheap enough per expert.
5. **One-shot cold start**: ~45 per-kernel NVRTC compiles and the dry passes put 4.5 s on a fresh process's
   first chunk (811 against Strata's 841-880 tok/s on 16k). A PTX cache on disk.
6. **The Engram projection in f16** (`qx_mgemv_t8z`, 1.3 s per 16k prompt): its f16 weights are 315 MB of scratch.
7. Carried: the concurrency-1 request spread (8.0-9.9 s, pool and NVMe time of the warm-up's tier state), the
   slow first token (~1 session in 8), `serve 200 calls` in the prefill256k log.

### Queue as of 2026-11-10 (after the conjecturer pass, `perf/ideation/2026-11-10-conjecturer.md`)
1. **Narrow staged share (widths 2-3 at 4/8)**: candidate default in 0bc58dc, serve256k3 -10.4%; set D, breaker,
   adjudicator pending.
2. **RAM dedup / exclusive tiers** (`QX_DS_RAM_DEDUP=1`, written, off): over half of decode256k's NVMe fetches are
   promotion churn (row 30). First arm queued (`qx-chain-dedup`). If it pays, the fuller form: a VRAM victim takes
   its replacement's RAM place (write-back), and flex-tail residents too.
3. **NVMe read-ahead from the `next` predictor** on a branch parallel to the wait kernel (E4): 7.0 ms per token of
   fetch wait on decode256k (halved if 2 lands). Two builds.
4. **Counter repair** (E6): the `serve`/gap counters are cumulative over 64 tokens incl. warm-up; print them for
   the scored window; per-kernel split of the gap. Also the dry pass's phantom jobs in them.
5. **Chunk "slab enqueue": host cost or GPU back-pressure?** (E8) one nsys run decides rows 10/21.
6. **First real use of a wide window costs ~2 s of launch/replay in a server** (width 96 formed by three prompts
   arriving together: TTFT 3.6-4.4 s instead of 1.2-2.1): see what the warm-up's dry pass leaves undone.
7. Width-1 GEMV fusion (family 8), prefetch as default (candidate -3.0% decode256k, flat on serve), pool cache
   pre-touch kill test (`qx-poolbw` warm vs cold), 32k-row chunks sweep.
8. Slow first repeat of decode256k after a different process shape (44.5 / 47.9 against 42-44): not kcompactd by
   the sampler; cause open. `vm.compaction_proactiveness=0` needs sudo (the user's call).
9. Breaker leftovers: Qwen caller of `qx_moe_post` passes 14 words where info[20] is read (pass explicit zeros;
   do not run Qwen); `QX_DS_STG_WAIT=words` unsafe; placement portability; bench.sh records only its own knobs.

### Correctness prerequisite (2026-11-09; promotion still blocked)

The expanded all-row `qx-ds-multi` exposes cold-width compressor corruption: B 1/834 exact becomes 834/834 when
incoming pending buffers are preserved around the dry pass. A/C remain 26/26; the full updated checks pass.
Startup-warmed control also passes without the fix, so this does not explain the intermittent server flip.
Fixed-binary server pair: default window 7/8 identical (French Revolution, reasoning char 400), window 1
8/8 identical. Repeat the pair, then build a teacher-forced reproducer from that actual prompt/continuation
and compare every row across widths and tiers. One clean control is inconclusive. `--window 1` still allows
a width-3 batch of concurrent decoders. See notebook and presupposition 26.

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

### 7b. Elastic VRAM for the chunk buffers: the design on our tier (2026-11-08, from the Big-MoE-Small-Box read)

- The VRAM expert tier is one `DeviceBuf` arena filled in profile order, hottest first (`dsv41/mod.rs` ~790-850), so
  its tail holds the coldest resident records: the natural flex region. Today the chunk buffers (~1.2 GiB at 4,096
  rows, ~2.4 at 8,192) are separate allocations taken before the tier, permanently.
- Design: allocate the arena with a tail of exactly the chunk buffers' size and make the chunk buffers views into
  that tail. At a prompt's first chunk (`chunk_enter`): every record whose address lies in the tail leaves the VRAM
  table (`table[g] = 0`, device table word written on the stream: the next window's wait serves it from RAM/NVMe as
  a miss), and the tail's slots leave the class free lists so promotions do not land in the buffers. After the
  prompt (`chunk_exit`, when no slot is in its prompt phase): the tail's slots return to the free lists empty and
  `adapt`'s promotions refill them with the current hot set. Stream order makes the overwrite safe: the chunk pass's
  kernels run after every earlier launch on the stream.
- Buys: 8,192-row chunks everywhere (prefill256k 6.8 -> 6.2, server TTFT 43.8 -> 37.3 s) with no decode cost in
  decode-only phases; under continuous prompts the behaviour equals today's. Exact. Cost per prompt: the tail's
  ~250 records re-promoted afterwards (~2.4 GB at the promotion rate, seconds, against a 40 s prompt).
- Implementation points: `ChunkBufs` as views + offsets into the tail; `chunk_enter/exit` in the engine with the
  worker deciding the exit (`worker_ds.rs`: no slot in prompt phase); the class free lists and `holder` map know
  the tail slots' sizes. Medium (a day), measurable with the server's serve256k TTFT and the serve256k3 decode.

### 15. Big-MoE-Small-Box (casper-hansen, exllamav3-based, 3090 + 64 GB) read 2026-11-08: what is worth porting

Source: ~/git/Big-MoE-Small-Box (code read, mechanisms with file:line in the session notes; the README's layer-ahead
prefetch and staging pool are OFF for DeepSeek in their own config). Their DeepSeek numbers: exact 10.6-11.8 tok/s
decode, 33 / 248 tok/s prefill at 1k / 8k (fp16 tensor cores, an NVMe floor of ~15 s per 8192 chunk); lossy "warm
prefill" (drop the cold experts' picks beyond the 64 heaviest per layer, rescale) 680-725 tok/s at KL 0.023. We are
2.5-3x ahead on exact decode on ~2x the hardware; our exact prefill lands where their 3090 does, which says our
exact prefill is far below the card (our profile agrees). Their prefill lead is the lossy mode. Ranked for us:

1. **Elastic VRAM for the chunk buffers** (`engine/tiers.py:454-492`, `wb_bulk`): evict the tier's last chunks at a
   chunk prefill's entry, give the memory to 8192-row buffers, re-enable the slots empty after. Buys our measured
   8192 gains (prefill256k 6.8 -> 6.2, server TTFT -22%) without the -3.9% three-stream decode that pinned the
   server at 4096. Exact. Medium. (Family 7 with a worked design to copy.)
2. **Per-record PCIe latency**: their 3090 lands an 8-9 MB record in 0.36 ms inside a gather kernel from mapped RAM
   (`nv2_dev.cu:369-491`, int4 loads, 48 x 256 threads) and 25 GB/s on the copy engine; our staged record costs 0.7
   ms and our 2026-11-04 mapped reads ran at 7 GB/s through the GEMV's access pattern. The 256k wait is per-layer
   latency on ~2.4 records, so halving a record's landing time is the lever on the 28 ms. Exact. MEASURED 2026-11-08
   (`kernels/bench/pcie_bench.cu`): the bus moves a 9.5 MB record in 0.19 ms (49 GB/s) by memcpy or by gather kernel
   alike, so the gather form has nothing to give here; the 0.7 ms is driver-side overhead around the copy. Next step:
   an nsys trace of a 256k decode step to split the 0.5 ms of overhead per record between host waits and GPU idle.
3. **Layer-ahead router prediction, measured offline first**: layer l+1's router on layer l's pre-MoE state
   (`nv2.py:506-523`). Their GLM numbers are sobering (top-4 ~30% useful, 54% late, ~5% precision past rank 4; no
   DeepSeek figure). A numpy pass over dumped routing + pre-MoE states gives our precision in a day; build nothing
   before it. Union with the previous-token signal (23%). Exact (residency only).
4. **DSpark re-costed**: their head with confidence 0.5 accepted 75% of drafted tokens (2.83 per verify) for +12.5%
   on an SSD-bound box; against our w_4 19.4 / w_8 15.7 ms per position that is past the 41-46% break-even. Measure
   acceptance on our prompts (family 6, as written). Also their rule: experts picked by one verify token are read
   zero-copy, never admitted (`nv2_host.cpp:1007-1010`), to stop VRAM churn.
5. **Exclusive VRAM/RAM tiers with victim write-back** (`nv2_host.cpp:635-654, 720-775`): +~1,600 records of host
   capacity if ours is inclusive (check); costs a 9.5 MB D2H per promotion. Uncertain; rank after the check.
6. **Piece-level NVMe streaming + strict demand-over-prefetch classes** (`pmask`, `Reader::pick`): <=2% at 256k
   forced decode, more in serving (NVMe wait 31 of 104 ms in serve3). Small-medium with io_uring CQEs per piece.
7. **Hygiene**: a startup bit-identity self-test (one token per layer with half the picks on the pool, compared bit
   for bit; lane off on mismatch, `nv2.py:653-707`) in the server's warm-up; routing-trace capture for offline tier
   simulation; MWAITX instead of PAUSE in the mailbox waits (a spinning core held their shared clock rail).
8. **Serving**: a disk tier behind parked sequences (re-prefill of a 256k state ~27 min vs ~1 s restore), the
   pre-stream 400 `context_length_exceeded` with a 1,024-token reserve, OOM retry with halved chunks, /health 503 on
   a dead engine thread.

**Second pass (2026-11-08, after the counters showed the 256k decode wait is the pool window: 22 of 28 ms per token,
~47 records through 16 threads at 0.37 ms each, 4.8 ms of NVMe fetch waits).** Their DeepSeek config runs the CPU lane,
NVMe->CPU streaming by landed pieces and early write-back; the layer-ahead prediction, the staging pool, the fast
publish kernel and X16 are OFF for DeepSeek (only GLM turns the first three on), so none of their DeepSeek numbers
include them. Their per-layer timeline at 32k: 4.5 ms = copy 1.2 + MoE 1.6 + CPU wait 1.0 + rest 0.6; CPU 508 us per
expert on 5 Zen 4 threads in RAM-only layers (0.39 experts/ms/thread against our 0.17 on the same H=5120/I=2304
geometry). Ranked for us:

9. **Dataflow CPU job, no stage barriers** (`ft_n135.h:688-979`): one unit queue prep -> gate/up bands -> act -> down
   bands -> out, workers take units from one atomic counter and spin only on the counter of the job they need; a
   unit is a contiguous 12 KB row band; late (NVMe) jobs' units are ordered by predicted landing. Our pool runs two
   pool-wide barriers per layer call and splits every matrix over all 16 threads (`pool.rs:133-200`), so a 1.2-record
   layer leaves threads idle at each barrier. Exact (integer sums, fixed output order). Measure first: one record on
   one pinned core with no barriers; if ~0.4 ms the gap to our 0.37-per-record-on-16-threads is barriers/idle, if
   ~0.9 ms it is the kernel or clocks. Expected 4-8 ms per 256k token if 20-40% of the window is idle. 2-4 days.
10. **MWAITX idle waits for the power budget** (`nv2_host.cpp:66-77, 1737-1748`): their controller sleeping on
   MONITORX/MWAITX instead of PAUSE spins lifted the lane cores from 4.34 to 4.8 GHz and the CPU cost per expert
   from 0.40 to 0.29 ms (-27%). Our driver thread spins 2M iterations before sleeping and the pool threads spin at
   barriers while the GPU runs attention. Check with turbostat during decode; then umwait/mwaitx (or a short spin
   budget then futex) in the driver loop and the pool barriers. Exact. Hours; 2-5 ms per token if the pool is
   compute-bound (25.7 GB/s says it is).
11. **Per-layer makespan planner** (`nv2_host.cpp:926-1062`, `Pol` costs): the host splits each layer's host-served
   picks between DMA and CPU to minimise max(cpu, gpu) from a cost model (DMA 0.36 ms per pick, CPU 0.42 + 0.12 per
   extra token, 0.145 fixed), colder picks to the CPU so DMA-staged ones are the promotion candidates, NVMe picks to
   whichever lane finishes first given the predicted landing. Ours is a fixed 3/8 share (the sweep said 3/8 at 1k
   and 256k alike). Exact. 1-2 days with the constants from our own timeline (item 13).
12. **NVMe reader classes and piece streaming into the pool** (`nv2_host.cpp:86-250, 1384-1406`): blocking O_DIRECT
   preads on 16 threads (6.7-7 GB/s, no io_uring), 512 KB pieces for decode, demand FIFO capped at 12 MB in flight so
   a layer's records land one after another, prefetch LIFO under a 6 MB cap, bulk behind; the CPU lane binds an NVMe
   pick when its read starts and each band unit waits only for its own pieces. NVMe wait 6.2 ms per token with it,
   25.3 in their exact (no CPU lane) mode. Ours: 7 NVMe records per token, 4.8 ms of fetch waits, read whole then
   staged. Compute NVMe-landed records on the pool as their gate/up pieces land; cap demand bytes in flight. Exact.
   1-2 ms per token. 2-3 days.
13. **Per-layer timeline instrumentation** (`nv2_dev.cu` globaltimer ring + host CLOCK_REALTIME stamps, `nv2.py:874-937`):
   publish, reply seen, copy start/end, CPU job start/end, last landing, combine start, per layer and split by layers
   with/without NVMe reads. We know the 28 ms total; this names the per-layer parts and feeds item 11's constants.
   1 day; do this before 9 and 11.
14. **PCIe-contention gating of background copies** (`HC_CPDONE`/`HC_MOEDONE` flags, `nv2_host.cpp:403-406`): promotions
   and write-backs only after the layer's critical copies finished. Ours issue promotions at the token boundary on
   the copy stream, which may overlap the next layers' staged copies. Hours; up to promos x 0.19 ms per token.
15. **Exclusive tiers with victim write-back by DMA** (`nv2_host.cpp:635-654, 720-775`): a VRAM victim parks in a 32-slot
   VRAM ring, the host copies it D2H on a side stream (0.83 ms mean) and flips its home to RAM; "SM stores to host
   memory run at ~4 GB/s vs 25 GB/s for the DMA engine". Ours evicts VRAM records that are still in RAM (inclusive at
   the VRAM/RAM boundary), so no write-back; the point stands only if we ever drop the RAM copy of a VRAM resident
   (+~1,700 RAM records). Low.
16. **Zero-copy no-admit policy** (`LN_RZC/LN_NZC`, `nv_noadmit`): NVMe-landed picks and single-token verify picks are
   read from RAM without taking a VRAM slot, so one-off picks cause no eviction churn. Hours; small.
17. **Hygiene worth copying:** startup self-tests that disable a lane on any mismatch (`nv2.py:652-753`), a `verify()`
   that byte-compares sampled VRAM/RAM slots against fresh O_DIRECT reads (`nv2.py:800-845`), per-record sha256 in
   the store with re-read verify, `posix_fadvise(DONTNEED)` on the checkpoint after the tier fill (our 7-8 GB of
   page cache from the pack mmap), the RAM tier sized from the cgroup's `memory.max - current - margin`, OOM retry
   with halved prefill chunks, bounded device waits that raise an error word a watchdog turns into SIGTERM.
18. Checked, not for us: X16 (CPU emulation of fp16 MMA rounding at 5.5x; `docs/x16.md`'s "would INT32 accumulation
   fix it" is our design), B70 remote tier, BCP/tail skip (lossy), the EXL3 VBMI trellis lane, their attention/KV
   (all exllamav3's; no paging, no indexer), the pinned bounce-slot loader (exllamav3-specific), hugepages/NUMA
   (absent). Their CPU-lane result is not bit-exact with their GPU path (fp32 partial, KL 0.001); only X16 is.

**Re-ranked 2026-11-08 (late) by the conjecturer pass (`perf/ideation/2026-11-08-conjecturer.md`), against the
measured facts:** item 2 (per-record driver overhead, "~45 ms") retired: the staged copies are not on the decode
critical path (ready-words test equal) and in serve they run at bus rate; item 9 (dataflow pool) downgraded to
<= 1.6 ms per 256k token: the pool's floor is DRAM at ~0.19 ms per record (one core does 2.55 ms, compute-bound;
16 cores 0.23), not barriers; item 10 (MWAITX) closed by the 5.3 GHz clock measurement; item 11 (per-layer
DMA/pool planner) rejected for the scored cases: both lanes draw on the same DRAM ceiling and the share sweep is
flat. Ranked in their place: (a) the layer-ahead router prediction, measured first (`QX_DS_PREDICT=pre|next`
counters, ec924e7; colibri measured 71.6% recall for layer L+1's router on layer L's state on GLM; our previous-token
signal is 13-23%): at recall r the decode256k host window (16 ms) shrinks by about r x 14 ms; (b) the 256-slot
staging area as an LRU cache of staged records (trace simulation: 25% of host-served selections single-stream at
256 slots, 37% under three streams at 1024; serve256k3 is within 10-20% of DRAM-bound, so bytes are its only lever:
-18 to -30% with a 512-1024-slot area lent by the flex tail); (c) per-layer timeline with GPU `%globaltimer`
stamps (is the `gap` counter GPU time, and what is the bus-bubble fraction in serve); (d) latent KV rows in host
memory re-costed: a serve256k3 lever (-20% at +1100 records over three slots), ~0 for decode256k by the trace
slope (1.2 hit-rate points per 100 records); (e) staggered slot streams in the server (fills the bus bubble, ceiling
from (c)); (f) NVMe read latency on the width-1 path (two reads per record, gate/up first): -4 to -6% decode256k;
(g) DSpark acceptance measured with the local Strata binary before any build (the 256k break-even is ~31%, lower
than at 1k).

Not transferable: warm prefill / BCP / tail skip (lossy, and they attack an NVMe floor our tier removes); the int16
VBMI trellis lane (EXL3-specific; our pool is DRAM-bound at decode); X16 (emulating fp16 tensor-core rounding on the
CPU at 5.5x the cost: the problem our integer kernels designed away); the dynamic per-layer lane planner (our share
sweep says DMA and pool cost the same per record here); their layer-ahead staging pool (they kept it off).

### 14. Prefill to Strata's number (new 2026-11-07 evening: the user's priority is matching Strata's 935 tok/s on a 16k prompt; ours 190-250)

- **Where the chunk's GPU time goes** (profile on chunk19, two 4096-row chunks, per slab-layer): the dense one-launch
  GEMV 42% (13.4 launches, 413 us each; latency-bound: 16 token groups per block in series, two barriers and a shared
  round trip each, about 3 us a group; 50x off the bandwidth and compute floors), attention 22% (`attn_part` 560 us
  + `merge` 157 us per slab-layer call), the exact int8 expert tiles 31%, the rest 5%.
- **Measured dead end:** 4 or 8 rows per block with per-group barriers (`mgemv_zr`): 12% slower, bit-identical.
- **In flight:** one barrier per block (`mgemv_zb`, QX_MGEMV_ZB=1: every warp's partial for every group to shared
  memory, one summing pass at the end in the same order); then the same with 4 rows per block sharing the activation
  fragments (`mgemv_zt`, QX_MGEMV_ZT=1, its own table). Target: the 817 us launch under 150 us, the dense passes
  under 10% of the chunk, prefill256k 6.2 -> about 4.5 ms per token.
- **Next:** the attention over the slab (one block per (head, token) over the window plus 512 selected latents; the
  same per-call latency pattern); the expert tiles' dequant per 16 elements. A bf16 tensor-core tile path for the
  dense passes behind an opt-in knob is the lossy option (measured: Strata's whole engine is at KL 0.03 from our exact
  path); the exact forms first.
- **Kill criteria:** a form that is not bit-identical on ds-checks line 3 and the 16k tails; a form inside the
  chunk timing's 1% spread.

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

- **Re-aimed 2026-11-07 (profile on 0ced38e, golden profile, decode256k 46 ms):** the indexer is 9% of the step (`idx_score2` 2.2 ms + `topk` 2.5); 57% is the two expert wait kernels (28 ms: ~93 host-served records of 9.5 MB per token, per-layer round trips), 13% `qx_mgemv` at width 1 (511 launches of 12.7 us). Measured dead ends: the staging share (3/8 is the optimum; all-DMA and all-pool both 46-47), previous-token prefetch (23% of host-served selections repeat). Live levers, each below the kernel bar alone, a bundle to clear it: (a) `qx_ds_topk` as one compaction pass (per-thread contiguous ranges, one block scan, warp-private histograms; order of `sel` must stay ascending-index with the equal keys at the tail, the reference's tie rule), 284 → ~100 us per call, ~1.6 ms per step; (b) family 8's GEMV fusions at width 1 (~2 ms); (c) structural: the latent rows of the compressed store to pinned host memory (read 512 rows per layer per token by gather), freeing ~8 GiB of VRAM for the expert tier; expected -7 ms of waits against +3 of gathers at 7 GB/s, measure the gather rate first.

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
