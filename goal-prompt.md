Maximize the performance of the DeepSeek-V4.1-Flash UD-IQ2_XXS engine at 256k context (priority set by the user on
2026-11-06), through bold, well-measured experiments. Exactness is the floor, not the ceiling: be rigorous about
measurement and adventurous about ideas.

## The objective, in order
1. **Performance at 256k context**: the server at `--ctx 262144`. What counts, in ms per token: decode at a 256k
   position; prefill of the last stretch of a 256k prompt (the position curve, not just the mean); per-request
   decode and time to first token under concurrency at 256k. These are the standard cases ("set D"); the scoreboard follows them. Time to first token on a long prompt is
   part of this objective but not yet a scored case: adding a `ttft256k` case (the 13.5k-token prompt at
   `--ctx 262144 --slots 3`, median of 3) is sanctioned work and a case-set change to note in the scoreboard.
2. **No regression at short contexts**: decode and prefill128 at 1k, and serve3 at 8k, stay within noise of the
   promoted baseline's values (`perf/scoreboard.md`, last row; `perf/baselines/.../summary.md`). They are guards,
   not targets.
3. **Capability**: 256k must work with at least one slot on this box (it does: 15.4 GiB tier at one slot, 12.2 at
   three) and must not grow host memory with context (it does not since `39c3bc0`).

## Priority order (stated, because it was only implied)
1. **Exactness bugs first.** A failing ds-checks line or a server identity mismatch stops all performance work until
   it is fixed and has a regression check. On an intermittent mismatch: record the run's schedule
   (`QX_DS_SCHED_TRACE`), replay it (`qx-ds-replay`, then `QX_DS_TRACE_LAYERS=1` against a frozen-tier replay) and
   read the lanes BEFORE running statistical arms (a morning of arms said nothing on 2026-11-09; the replay named
   the expert in one run).
2. **Then promote what has landed** (set D, breaker, adjudicator).
3. **Then new experiments.**

## Benchmark preflight (no scored run without it)
Scores are only comparable on a quiet box. Before any set D run, and recorded in the notebook entry: the win10 VM
is shut off (`virsh -c qemu:///system domstate win10`; never start it, never force it off: ask the user), swap is
mostly free (`free -g`), memory pressure is near zero (`/proc/pressure/memory` some avg10 < 1), no `kcompactd` /
`ksmd` / compile job is burning a core, tmpfs holds no stale scratch (`df -h /tmp`), no other engine is alive
(`pgrep -x qx-ds-check; pgrep -x quixiinfer`). If the box is not clean, do not run scored cases: fix what is ours,
ask the user for the rest. Relative A/B arms may run at a smaller RAM tier (same in every arm) meanwhile, labelled
as such, never gated against the baseline. (2026-11-09: with the VM up, the standard 100 GB tier overcommitted the
box; decode256k flipped between 42.0 and 44-45 ms on one binary and systemd-oomd killed four runs.)

## The standard set (done 2026-11-06/07; do not rebuild)
Set D exists: `perf/states/` holds the 262,144- and 245,760-token states, `perf/harness/bench.sh` runs `decode256k`
(64 single steps from the 262k state, last 32 scored), `prefill256k` (the last 16k tokens of a 256k prompt in
8192-row chunks, last 32 scored), `serve256k` (the server at `--ctx 262144 --slots 3`, loadgen at concurrency 1 and
3) and the guards `decode`, `prefill128`, `serve`. The scoreboard's last row is the promoted baseline; `perf_diff.py
gate` compares against `perf/baselines/rtx-5090_ryzen-9-9950x_121g/`. Measured spreads (2026-11-08): decode256k
0.2-3%, prefill256k 0-2%, serve256k3 0.2-1.7%, decode 1-4%: three repeats are enough; a knob flip needs no five.

## Read first, every session
  perf/perf.md (rules), perf/presuppositions.md (the beliefs the design rests on and how each would be killed), perf/findings.md (what is known and what was rejected, with the conditions for re-opening), perf/backlog.md, the last entries of perf/optimization_status.md, handoff.md (constraints, how to run things).

  ## Think wide before you pick
  Before each experiment, spend a real ideation pass, not a token one. The backlog is a starting beam, not a fence. Sources to mine, all local:
  - ~/git/colibri (pure-C streaming MoE, has its own DeepSeek V4.1 engine): docs/deepseek-v41.md (DSpark drafting and why accepting a draft is safe; streaming experts; "blocks of positions, not one position at a time"; the two read-hiding ideas that did NOT work, read those so you do not repeat them), docs/CACHE_ROUTE.md (cache-aware routing), docs/experiments/ (dsv41 expert-io and omp-team measurements).
  - ~/git/Strata (has strata-dsv41): docs/DEEPSEEK41.md, BATCHED_DMA.md (cudaMemcpyBatchAsync for expert uploads), EXPERTDEC.md (verify window's experts with activations in shared memory), KV_PREFETCH.md (overlap KV uploads with prefill), EXCHANGE_ROTATION.md (zero-copy expert exchange between RAM and VRAM), VRAM_ELASTIC.md, PROMPT_CACHE_TAIL.md, MESSAGE_BOUNDARY_CACHE.md, BATCHING.md, PREFILL_HEAD_ALIAS.md.
  - ~/git/basalt (Qwen3.8 MoE on Blackwell, experts in RAM): docs/unified-kernel.md (what was retired and the lessons), docs/configuration/adaptive.md (the adaptive VRAM expert tier, with measurements), docs/precision/two-stage-head.md (screen-then-score vocab head), docs/precision/routing.md, docs/probe.md.
  - ~/git/DeepInfer (DeepSeek's own V4.1-Flash framework): DSpark + confidence-scheduled verification, SWA bounded replay (YOCO prefill truncation), EPLB expert load balancing, batch-invariant kernels, the async zero-overhead scheduler. Read the model_executor layers for how they order MLA and expert work.
    - ~/git/Big-MoE-Small-Box (casper-hansen; exllamav3-based, 3090 + 64 GB): read twice (backlog family 15 lists what was taken, what was measured and closed, and what is left); five design issues were filed there (#1-#5) with predicted gains. When an idea belongs upstream rather than here, file it there.
  Translate, don't transplant: ask what the technique buys on THIS box (one 5090 on PCIe 5.0 at 0.19 ms per 9.5 MB expert record, 16 AVX-512 cores at ~50 GB/s, NVMe), where the counters say our time goes (2026-11-08: a 256k decode step is ~42 ms, ~28 ms of it waiting for the host, and the host's whole per-layer cost is the CPU pool (mailbox and copy issue are 7 us of a ~560 us call); the pool does 0.23 ms per record alone against 0.16 ideal; prefill is the chunk path, expert-major, 3.4-3.9 ms per row at 16k, launch- and kernel-bound), and what it would take to keep bit-identity. Closed for this box, do not re-mine: gather-from-mapped-RAM landing (the bus is the floor either way), MWAIT/idle-wait clock tricks (the pool cores run at 5.3 GHz while generating), the pool on SMT siblings (1.5x slower), slab attention under the raw launcher (a wash), staged-ready device words (equal), decode staging shares other than 3/8 (flat).

  Out-of-the-box is welcome and expected. Examples of the kind of thinking wanted, not a list to execute: shift bytes between tiers instead of making a tier faster; change WHAT gets computed (draft/verify, bounded replay, screen-then-score) not just how fast; exploit routing statistics across tokens or layers (prefetch the next layer's likely experts while this layer runs, warm the staging area speculatively); batch or coalesce the DMA path; restructure the mailbox so the GPU never waits on the host for small things; measure things nobody has measured yet (routing overlap between consecutive tokens, which re-opens the speculative-decoding rejection; NVMe queue depth vs latency; per-layer expert reuse distance). A measurement that changes a belief in findings.md is a valid, recordable outcome even with no code change.

  Ideas that trade accuracy for speed (cache-aware routing, relaxed drafting, fewer experts) may be built, but only behind an opt-in knob that is OFF by default, never counted on the scoreboard, and recorded in a separate "lossy, opt-in" table in findings.md with their quality cost measured by tools/ds-checks.sh deltas. The default path stays exact.

  ## Break presuppositions on purpose
  The wins that mattered this week were beliefs overturned by cheap measurements (a record lands in 0.19 ms, not
  0.7; the pool was not memory-bound, it was sharing cores with a spinning thread), and the folklore that blocked
  them was never written down. `perf/presuppositions.md` is the table of beliefs the design rests on, each with the
  measurement that would kill it and the date it was last tested. Rules:
  - A belief overturned is a LANDED-class outcome: update the row first, then the notebook and findings; list
    "beliefs tested" in every handoff update and in the stop report.
  - Every fourth experiment MUST attack an untested or stale row (or add and attack a new one), chosen by payoff x
    chance-it-is-false / cost. Count them in the handoff; the adjudicator flags four safe landings in a row.
  - Ideation is divergent before convergent: twenty "what if X is false" ideas with judgment deferred before any
    scoring, and the rejected list is read AFTER ideating, then checked against (the `perf-conjecturer` agent does
    this; the `perf-outsider` agent does it with no project context at all, from the physics of the box alone). Run
    the outsider at least once a day and whenever two experiments in a row are flat.
  - Vendor and folklore claims ("requires hardware support", "has to be on the GPU", "must be whole records",
    "cannot be predicted") are presuppositions until measured. QuixiCore ran NVFP4 on hardware that "could not":
    the math of any format can be implemented exactly anywhere; the question is only cost.
  - A long shot's cost is bounded by its cheapest killing measurement, not by the build: measure first, build only
    what survives. Before building, write its kill criterion and a budget (builds and GPU hours) in the notebook;
    at the budget, record the outcome and stop or re-justify (the prefetch took five builds to reach 2.6%).

  Budget: at least one in four experiments is a long shot (a bigger structural idea or a measurement of something unknown); the rest are the best-expected-value items. Record long shots honestly as INCONCLUSIVE or REJECTED when they fail; that is a win for the next agent, not a loss.

  ## The loop — repeat until a stop condition
  1. Choose: write the hypothesis, the counter that will confirm or refute it, the expected gain, and why it is better than the next-best idea. After every third experiment, or after two flat results in a row, run the `perf-conjecturer` agent (fresh context, given the facts and not your hypotheses) and re-rank against its list.
  2. Change ONE factor. Behind a knob when cheap.
  3. Build: cargo build --release -p qx-engine -p qx-serve --bin qx-ds-check --bin quixiinfer. Gate on cargo's exit status, never on a grep pipeline. step.cu/dsv41.cu builds can take ~10 min.
  4. Measure: perf/harness/bench.sh --check --label <slug>. Then python3 perf/harness/perf_diff.py gate <run-dir> against perf/baselines/rtx-5090_ryzen-9-9950x_121g.
  5. Decide by the bars, on the score OR on the targeted scored case (the geometric mean moves 2.4% when decode256k moves 6%, so name the case the change aims at before measuring): keep only if the gain beats 3% (low-risk: a default, a placement, a knob flip, a config) or 8% (complexity-adding) AND is outside the repeats' spread AND all five tools/ds-checks.sh lines are exact AND, for anything touching the server, the chunk buffers or the tiers, the server-level identity check passes THREE runs out of three (`tools/loadgen.py --dump` against the reference `perf/results/2026-11-07/srv2-chunk32/srv2-0.dump.jsonl`, `tools/compare_dumps.py` identical; one run is not evidence: the 2026-11-09 bug flipped a token in about one run in three), and `qx-ds-multi` passes with all rows checked (ds-checks never exercise the server's warm-up, scheduler or the driver's fallback paths). Inside the noise is INCONCLUSIVE, not a win. A big idea that lands at 2% may be kept as CANDIDATE if you can name the follow-up that would unlock more; say so in the entry.
     Before landing: run the `perf-breaker` agent on the diff and the claimed result and act on its tests; then the `perf-adjudicator` agent on the run directory. Nothing is LANDED or promoted without the adjudicator's verdict (PROMOTE) quoted in the notebook entry. Both run in fresh context and are never shown your reasoning.
  6. Record every outcome: perf_diff.py entry, then perf_diff.py index. LANDED: perf_diff.py promote the run as the new baseline, perf_diff.py score --record --note "...", update perf/findings.md (Wins) and perf/backlog.md, commit and push to master. REJECTED/INCONCLUSIVE: revert the code, add the reason and the re-open condition to perf/findings.md, commit the docs only. New ideas discovered along the way go into perf/backlog.md with a one-line rationale, even if you do not pursue them now.
  7. Update handoff.md with a dated line when something lands, a belief changes, or a family closes.

  ## Standing constraints (never relax)
  - Never touch the Windows drive nvme1n1 (by-id nvme-CT4000T705SSD3_2529E9C6B4B9). Never start or destroy the VM.
  - DeepSeek only. Do not run, start or test Qwen or quixiinfer.service.
  - One heavy job at a time, always inside a memory cap: `tools/capped.sh <GiB> <command>` or `systemd-run --user --scope -p MemoryMax=112G -p MemorySwapMax=0` (bench.sh does this itself at 112G; a QX_DS_RAM=100 run needs at least 116G). No cargo build beside a 100 GB engine run unless the build is capped too. An OOM kills the harness.
  - Wait for the GPU to be free before any CUDA job. Kill anything on the GPU that is not ours.
  - Never pkill -f / pgrep -f with a pattern that matches the tool shell; use pkill -x or kill by PID.
  - Never let a new kernel or window width load while an expert job is posted: new widths go through the dry pass.
  - Default GEMVs stay on the f32 path; windows must stay bit-identical to single steps.
  - tools/golden/golden-dsv41 stays modified and uncommitted. Exclude it from every commit.
- The chunk path's buffers are views into the VRAM arena's tail: nothing writes them without holding the flex tail (`assert_flex_held`); any new writer takes the tail first and the caller releases it.
- Thread placement: no spinning thread (engine thread, CUDA helpers) on a physical core a pool worker uses. Defaults come from `pool_cpus()` / `main_cpu()` (sysfs topology); `QX_DS_CPUS`, `QX_DS_MAIN_CPU` override. Confining the whole process with taskset deadlocks the engine; do not.
- Kernels must keep compiling for the RTX 30 and 40 series: `QX_NVCC_ARCHS=sm_86,sm_89,sm_120a` before landing a kernel change (mgemv takes ~20 min per target; run it in the background).
- Commits are pre-authorized for correctness fixes (with their validation), landed wins and their records, tools and knobs that are off by default, and docs-only rejection/backlog records. One-line summary, body with the numbers, ending with the attribution line the harness specifies for the session.
- Long jobs: scripts live in `~/qx-chains/` (on disk, never /tmp), are launched with `systemd-run --user --scope --unit qx-chain-NAME -- setsid nohup bash script.sh` (no MemoryMax on the chain scope: bench.sh caps itself), log one marker line per result to `~/qx-chains/log.out`, and are never edited while they run (write a new file).

## Where to start
State lives in `handoff.md`, not here: read its top section (the latest dated "Resume" / "START HERE"), the last ten
notebook entries and the marker lines in `~/qx-chains/log.out`. Then follow the priority order above. The ranked
experiment list is in the handoff and `perf/backlog.md`; the long-shot candidates are the untested rows of
`perf/presuppositions.md`.

## Stop and report (do not ask otherwise)
- Five consecutive REJECTED/INCONCLUSIVE experiments, excluding deliberate measurements.
- The next step would change the exactness contract, the standard settings or the case set (beyond item 1 above), or needs sudo beyond gdb attach, another disk, or more than the memory cap.
- Any ds-checks line stops matching and a revert does not restore it.
- The box cannot be made benchmark-clean without the user (the VM is up, another user's job holds memory or the GPU): report it with the evidence; keep relative arms going if they are meaningful.
When stopping, give: current score vs the promoted baseline (scoreboard last row), what landed, what was rejected and why, which presupposition rows were tested and how they changed, the long-shot count against the one-in-four rule, and the next three experiments ranked by expected value.

## Secondary objectives (sanctioned, the 256k objective still orders the GPU queue)
- Portability: the engine runs on the card it finds (RTX 30/40 series build-checked; nothing measured there). Keep it so; note what a 24 GB card would need when you learn it.
- Quality of the quantization: an EXL3-vs-equal-bytes-GGUF KL comparison is wanted when the GPU has a free slot (see handoff); the production format stays GGUF unless that measurement says otherwise.

Three deliberate choices you may want to adjust:
- Lossy ideas are allowed but quarantined: built opt-in, never default, never on the scoreboard. If you'd rather I not spend cycles on them at all, delete that paragraph.
- One-in-four long shots and a stop rule of five non-wins (up from three) give room for exploration without letting the loop drift indefinitely.
- A measurement counts as an outcome. Without that, the loop would bias toward safe code tweaks over the experiments most likely to open a new direction.
