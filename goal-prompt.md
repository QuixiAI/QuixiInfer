Maximize the performance of the DeepSeek-V4.1-Flash UD-IQ2_XXS engine at 256k context (priority set by the user on
2026-11-06), through bold, well-measured experiments. Exactness is the floor, not the ceiling: be rigorous about
measurement and adventurous about ideas.

## The objective, in order
1. **Performance at 256k context**: the server at `--ctx 262144`. What counts, in ms per token: decode at a 256k
   position; prefill of the last stretch of a 256k prompt (the position curve, not just the mean); per-request
   decode and time to first token under concurrency at 256k. These are the standard cases ("set D") once the
   tooling below exists; the scoreboard follows them.
2. **No regression at short contexts**: decode and prefill128 at 1k, and serve3 at 8k, stay within noise of their
   set C baseline (34.621: decode 34.0, prefill128 11.0, long16k 38.8, serve3 99.0). They are guards, not
   targets.
3. **Capability**: 256k must work with at least one slot on this box (it does: 15.4 GiB tier at one slot, 12.2 at
   three) and must not grow host memory with context (it does not since `39c3bc0`).

## Tooling first (do this before any 256k optimization)
A 256k forced pass takes about an hour, so no experiment can afford three repeats of it. Build the state cache:
`qx-ds-check` saves a slot's state after a pass (`QX_DS_SAVE_STATE=path`: tokens and the used compressed-store
rows, what `park` produces, about 1.3 GB at 256k) and later runs resume from it (`QX_DS_LOAD_STATE=path`: upload
the rows, replay the last window, then run the remaining tokens as windows or single steps). Verify once at scale
that a resumed run is bit-identical to the straight pass over the same positions (as line 5 of ds-checks proves
at 128). Then add the set D cases to `perf/harness/bench.sh`: `decode256k` (64 single steps from the 262,144
state, last 32 scored), `prefill256k` (the last 16k tokens in windows of 128 from the 246k state, last 32 scored),
`serve256k` (the server at `--ctx 262144 --slots 3`, loadgen at concurrency 1 and 3 with a long-prompt arm), and
keep `decode`, `prefill128`, `serve3` as guards. Each case under 5 minutes a repeat. Take the set D baseline, note
the case-set change in the scoreboard, and only then optimize.

## Read first, every session
  perf/perf.md (rules), perf/findings.md (what is known and what was rejected, with the conditions for re-opening), perf/backlog.md, the last entries of perf/optimization_status.md, handoff.md (constraints, how to run things).

  ## Think wide before you pick
  Before each experiment, spend a real ideation pass, not a token one. The backlog is a starting beam, not a fence. Sources to mine, all local:
  - ~/git/colibri (pure-C streaming MoE, has its own DeepSeek V4.1 engine): docs/deepseek-v41.md (DSpark drafting and why accepting a draft is safe; streaming experts; "blocks of positions, not one position at a time"; the two read-hiding ideas that did NOT work, read those so you do not repeat them), docs/CACHE_ROUTE.md (cache-aware routing), docs/experiments/ (dsv41 expert-io and omp-team measurements).
  - ~/git/Strata (has strata-dsv41): docs/DEEPSEEK41.md, BATCHED_DMA.md (cudaMemcpyBatchAsync for expert uploads), EXPERTDEC.md (verify window's experts with activations in shared memory), KV_PREFETCH.md (overlap KV uploads with prefill), EXCHANGE_ROTATION.md (zero-copy expert exchange between RAM and VRAM), VRAM_ELASTIC.md, PROMPT_CACHE_TAIL.md, MESSAGE_BOUNDARY_CACHE.md, BATCHING.md, PREFILL_HEAD_ALIAS.md.
  - ~/git/basalt (Qwen3.8 MoE on Blackwell, experts in RAM): docs/unified-kernel.md (what was retired and the lessons), docs/configuration/adaptive.md (the adaptive VRAM expert tier, with measurements), docs/precision/two-stage-head.md (screen-then-score vocab head), docs/precision/routing.md, docs/probe.md.
  - ~/git/DeepInfer (DeepSeek's own V4.1-Flash framework): DSpark + confidence-scheduled verification, SWA bounded replay (YOCO prefill truncation), EPLB expert load balancing, batch-invariant kernels, the async zero-overhead scheduler. Read the model_executor layers for how they order MLA and expert work.
  Translate, don't transplant: ask what the technique buys on THIS box (one 5090, 16 AVX-512 cores at ~50 GB/s, NVMe), where the counters say our time goes (decode is pool-bound at ~20 ms of a 32 ms step; prefill64 is balanced across pool, DMA and GPU expert kernels), and what it would take to keep bit-identity.

  Out-of-the-box is welcome and expected. Examples of the kind of thinking wanted, not a list to execute: shift bytes between tiers instead of making a tier faster; change WHAT gets computed (draft/verify, bounded replay, screen-then-score) not just how fast; exploit routing statistics across tokens or layers (prefetch the next layer's likely experts while this layer runs, warm the staging area speculatively); batch or coalesce the DMA path; restructure the mailbox so the GPU never waits on the host for small things; measure things nobody has measured yet (routing overlap between consecutive tokens, which re-opens the speculative-decoding rejection; NVMe queue depth vs latency; per-layer expert reuse distance). A measurement that changes a belief in findings.md is a valid, recordable outcome even with no code change.

  Ideas that trade accuracy for speed (cache-aware routing, relaxed drafting, fewer experts) may be built, but only behind an opt-in knob that is OFF by default, never counted on the scoreboard, and recorded in a separate "lossy, opt-in" table in findings.md with their quality cost measured by tools/ds-checks.sh deltas. The default path stays exact.

  Budget: roughly one in four experiments should be a long shot (a bigger structural idea or a measurement of something unknown); the rest should be the best-expected-value items. Record long shots honestly as INCONCLUSIVE or REJECTED when they fail; that is a win for the next agent, not a loss.

  ## The loop — repeat until a stop condition
  1. Choose: write the hypothesis, the counter that will confirm or refute it, the expected gain, and why it is better than the next-best idea. Prefer experiments whose effect shows in prefill64 (spread <1%) over decode-only ones (spread ~6%); for decode-only effects use --repeats 5.
  2. Change ONE factor. Behind a knob when cheap.
  3. Build: cargo build --release -p qx-engine -p qx-serve --bin qx-ds-check --bin quixiinfer. Gate on cargo's exit status, never on a grep pipeline. step.cu/dsv41.cu builds can take ~10 min.
  4. Measure: perf/harness/bench.sh --check --label <slug>. Then python3 perf/harness/perf_diff.py gate <run-dir> against perf/baselines/rtx-5090_ryzen-9-9950x_121g.
  5. Decide by the bars: keep only if the gain beats 3% (low-risk) or 8% (complexity-adding) AND is outside the repeats' spread AND all five tools/ds-checks.sh lines are exact. Inside the noise is INCONCLUSIVE, not a win. A big idea that lands at 2% may be kept as CANDIDATE if you can name the follow-up that would unlock more; say so in the entry.
  6. Record every outcome: perf_diff.py entry, then perf_diff.py index. LANDED: perf_diff.py promote the run as the new baseline, perf_diff.py score --record --note "...", update perf/findings.md (Wins) and perf/backlog.md, commit and push to master. REJECTED/INCONCLUSIVE: revert the code, add the reason and the re-open condition to perf/findings.md, commit the docs only. New ideas discovered along the way go into perf/backlog.md with a one-line rationale, even if you do not pursue them now.
  7. Update handoff.md with a dated line when something lands, a belief changes, or a family closes.

  ## Standing constraints (never relax)
  - Never touch the Windows drive nvme1n1 (by-id nvme-CT4000T705SSD3_2529E9C6B4B9). Never start or destroy the VM.
  - DeepSeek only. Do not run, start or test Qwen or quixiinfer.service.
  - One heavy job at a time, always inside a memory cap: systemd-run --user --scope -p MemoryMax=118G -p MemorySwapMax=0 (bench.sh does this itself). An OOM kills the harness.
  - Wait for the GPU to be free before any CUDA job. Kill anything on the GPU that is not ours.
  - Never pkill -f / pgrep -f with a pattern that matches the tool shell; use pkill -x or kill by PID.
  - Never let a new kernel or window width load while an expert job is posted: new widths go through the dry pass.
  - Default GEMVs stay on the f32 path; windows must stay bit-identical to single steps.
  - tools/golden/golden-dsv41 stays modified and uncommitted. Exclude it from every commit.
- Default GEMVs stay on the f32 path; windows must stay bit-identical to single steps.
- tools/golden/golden-dsv41 stays modified and uncommitted. Exclude it from every commit.
- Commits are pre-authorized ONLY for landed wins and their records, and docs-only rejection/backlog records. One-line summary, body with the numbers, trailer "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>".

## Where to start
1. The state cache and set D (above). The forced 256k pass of 2026-11-06 (`perf/results` logs, `poscurve.py`) is
   the first position curve; the state it would have saved must be regenerated once with the new tooling.
2. Then the 256k levers, by expected value: what the indexer and the compressed-store scans cost per token at 256k
   (nsys on a resumed decode step; the two-level candidate scan over 32k blocks and the `scores`/`bmax` buffers);
   the tier's behaviour at 256k (the decay-per-token fix landed; measure the VRAM share along the curve and after
   topic shifts); the staging area and chunk at long positions (the 128-chunk bundle is the default; does cap 256
   still overflow at 256k?); serving at 256k with 3 slots (12.2 GiB tier: is per-request decode acceptable, and
   does prefix reuse at 256 tokens pay?).
3. Long shots still welcome at one in four: per-request expert hot sets; a staging area that borrows tier VRAM only
   during wide windows; DSpark drafting only with a measured acceptance.

## Stop and report (do not ask otherwise)
- Five consecutive REJECTED/INCONCLUSIVE experiments, excluding deliberate measurements.
- The next step would change the exactness contract, the standard settings or the case set (beyond item 1 above), or needs sudo beyond gdb attach, another disk, or more than the memory cap.
- Any ds-checks line stops matching and a revert does not restore it.
When stopping, give: current score vs 19.514, what landed, what was rejected and why, which beliefs changed, and the next three experiments ranked by expected value.

Three deliberate choices you may want to adjust:
- Lossy ideas are allowed but quarantined: built opt-in, never default, never on the scoreboard. If you'd rather I not spend cycles on them at all, delete that paragraph.
- One-in-four long shots and a stop rule of five non-wins (up from three) give room for exploration without letting the loop drift indefinitely.
- A measurement counts as an outcome. Without that, the loop would bias toward safe code tweaks over the experiments most likely to open a new direction.
