Drive perf/scoreboard.md down for the DeepSeek-V4.1-Flash UD-IQ2_XXS engine through bold, well-measured experiments. Exactness is the floor, not the ceiling: be rigorous about measurement and adventurous about ideas.

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
1. Short: finish harness coverage (long16k and serve cases, vendor the token files), re-baseline, record a scoreboard row noting the case-set change.
2. Then alternate between the backlog's best-EV items (store sizing by max_ctx/ratio, STAGE_CAP sweep, serving gap, long context) and long shots drawn from the sources above. The first long shot I'd suggest: measure routing overlap between consecutive decode tokens and across layers, because that single number decides whether DSpark-style drafting or cross-token expert prefetch can ever pay here.
3. The uncommitted parking work in dsv41/mod.rs is a serving feature; leave it unless the serving gap needs it.

## Stop and report (do not ask otherwise)
- Five consecutive REJECTED/INCONCLUSIVE experiments, excluding deliberate measurements.
- The next step would change the exactness contract, the standard settings or the case set (beyond item 1 above), or needs sudo beyond gdb attach, another disk, or more than the memory cap.
- Any ds-checks line stops matching and a revert does not restore it.
When stopping, give: current score vs 19.514, what landed, what was rejected and why, which beliefs changed, and the next three experiments ranked by expected value.

Three deliberate choices you may want to adjust:
- Lossy ideas are allowed but quarantined: built opt-in, never default, never on the scoreboard. If you'd rather I not spend cycles on them at all, delete that paragraph.
- One-in-four long shots and a stop rule of five non-wins (up from three) give room for exploration without letting the loop drift indefinitely.
- A measurement counts as an outcome. Without that, the loop would bias toward safe code tweaks over the experiments most likely to open a new direction.
