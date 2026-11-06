# QuixiInfer Performance Handbook

This is the operating guide for measuring and optimizing QuixiInfer. Read it before changing anything for
speed. The first target is the DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS engine (`crates/qx-engine/src/dsv41/`,
`kernels/dsv41.cu`, `kernels/step.cu`, the CPU pool in `crates/qx-cpu/`, the tiers in
`crates/qx-engine/src/tiers.rs`, serving in `crates/qx-serve/src/worker_ds.rs`).

Do not treat this as a collection of tricks. Run a disciplined loop:

1. Name the bottleneck from counters.
2. Measure a clean baseline.
3. Change one factor.
4. Keep only verified wins that stay exact.
5. Record enough that the next pass starts from evidence instead of memory.

The files under `perf/`:

| file | what it is |
|---|---|
| `perf.md` (this file) | method: how to measure and decide |
| `optimization_status.md` | the results notebook: append-only dated entries, one per experiment or pass |
| `findings.md` | established truths and rejected ideas, distilled from the notebook; read before proposing anything |
| `backlog.md` | the beam: 3-5 active idea families, best first, each with a target and kill criteria |
| `scoreboard.md` | the one hill-climbable number, recorded at each landed win |
| `harness/bench.sh` | the standard end-to-end benchmark (build, guards, repeats, provenance, parsing) |
| `harness/perf_diff.py` | parse, summarize, validate, diff, gate, promote, score, notebook entry and index |
| `baselines/<host>/` | the committed baseline run per host fingerprint, which `gate` compares against |
| `results/` | raw run directories; git-ignored, so copy what matters into the notebook |

`docs/progress.md` is the project's narrative history, and it holds the measurements made before this scaffolding
existed. Cite its entries by date when a finding comes from there.

## Principles

**Exactness is the contract.** Every speed change must keep the engine's numerics. For DeepSeek that means the
regression set `tools/ds-checks.sh` reproduces these lines exactly:

```
1. exact f32 vs reference:      positions 6: top-1 agreement 6/6, KLD mean 0.000000
2. production vs reference:     positions 59: top-1 agreement 57/59, KLD mean 0.128643
3. windows of 8 vs single steps: bit-identical rows: 128 / 128
3. windows of 64 vs single steps: bit-identical rows: 128 / 128
4. tokenizer vs llama.cpp:      identical: 20 tokens decode round trip: exact
```

Line 2's 57/59 and 0.128643 are the production path's own fixed point, so any change in them is a numerics
change. The engine is deterministic and bit-reproducible across tier placements, window widths, staging shares
and slots. A change that alters numerics on purpose is not a speed change. It needs its own reference comparison
and the user's sign-off. The bf16 tensor-core GEMV is the standing example: faster, rejected for DeepSeek because
windows stop matching single steps.

**The end-to-end number decides; microbenchmarks explain.** A kernel that gets faster in isolation is a candidate
until the standard benchmark moves. The engine is a pipeline of the GPU, the CPU pool, the copy engine and the
NVMe drive working at once. Speeding up a stage that is not on the critical path changes nothing, and the
counters below show which stage is.

**Attack a named bottleneck**, stated as a counter value before the edit:

- **Pool-bound.** The CPU computes the experts that are not in VRAM. The pool runs at the RAM's read bandwidth, so
  pool ms per token tracks expert bytes read from RAM. Levers: fewer pool experts (more VRAM hits, a larger
  staged GPU share), fewer bytes per expert, better overlap.
- **Transfer-bound.** PCIe DMA of staged records, promotions or mapped reads. Judge against the measured copy
  rates below, not the 64 GB/s PCIe 5 spec.
- **Drive-bound.** NVMe misses: `fetch` inside pool time, and the early-token warm-up regime. Levers: residency
  (profile, promotions), batching, earlier submission.
- **GPU-bound.** The dense and attention kernels, measured with `QX_DS_NOEXPERTS=1` as the GPU-only step. Levers
  are kernel work, but only when the GPU is on the critical path.
- **Host or sync-bound.** Launch overhead, uploads and host syncs: the `phases per step` line. Graph capture
  already brings replay to about 0.01 ms, so this is rarely the bottleneck now.
- **Context-bound.** Costs that grow with position: attention, top-k selection, compressor work, pool time
  rising at long context. Measure at 16k as well as 1k.

## The Metrics

The **standard benchmark** (`perf/harness/bench.sh`) runs the engine on the vendored token files in
`perf/data/` (`route1k.tokens`, 1,024 tokens of a DeepSeek-tokenized mixed text, and `route16k.tokens`, its 16k
continuation), and the server under `tools/loadgen.py`. Every engine uses the fixed settings
`QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32 QX_DS_VERBOSE=1`.

| case | what runs | metric |
|---|---|---|
| `decode` | 1,024 single steps | **ms per token over the last 32** (steady state, tiers warm) |
| `prefill128` | 8 windows of 128 tokens (the server's prompt chunk) | **ms per token over the last 32** (the last window, at position 896-1023) |
| `prefill64` | 16 windows of 64 tokens | the previous chunk; optional since set C |
| `long16k` | 16,384 tokens in windows of 128, then 64 single steps | **ms per token over the last 32 single steps** (steady decode at 16k; the tail's first step pays the width-1 capture, 0.85-1.5 s, so the 64-step mean is recorded as `tail_ms_per_token` but not scored) |
| `long16k-prefill` | the same run | the windows' ms per token without the tail (prefill averaged over 0-16k); recorded, not scored |
| `serve3` | `quixiinfer serve --slots 3 --ctx 8192 --window 64`, 6 unmeasured requests, then `loadgen.py` at concurrency 1 and 3, 2 rounds of 256 greedy tokens | **ms per token of per-request decode at concurrency 3** (1000 / per-request tok/s) |
| `serve1` | the same session | the concurrency-1 number; recorded, not scored |
| `prefill2`, `prefill4`, `prefill8`, `prefill32`, `prefill128` | windows of other widths | optional, for width experiments; the narrow ones are the verify-window costs of backlog family 6 |

The **standard set** is `decode`, `prefill128`, `long16k` and `serve3` (`STANDARD_CASES` in `perf_diff.py`; set
C since 2026-11-06, when the 128-token chunk became the server's default: `prefill64` is optional now). A
scoreboard row or a promoted baseline must cover all four; `perf_diff.py` refuses otherwise. `bench.sh --quick`
runs `decode` and `prefill64` only (about 5 minutes) for screening: `gate` compares it case by case and reports
the score over the common cases, but it cannot be recorded or promoted. Up to 2026-11-05 the standard set was the
two quick cases; those rows are marked in the scoreboard and are not comparable with later ones.

- **Why the last 32.** The tiers adapt over the first few hundred tokens, and the first window of each width
  pays a one-time compile and capture cost. The mean over all 1,024 tokens is recorded too, and it is the right
  number for warm-up work.
- **Why these four.** `decode` and `prefill64` are the engine's two steady regimes at a short context. `long16k`
  holds the long-context cost that grows with position (backlog family 4), and `serve3` the serving gap between
  per-request decode and the forced bench (family 2). Before them, neither was decidable.
- **The score** is the geometric mean of the standard cases' median ms per token: the one hill-climbable number,
  recorded in `scoreboard.md`. Lower is better. It is comparable only at the same host fingerprint, settings and
  case set.
- **Repeats.** Use three per case by default. The median decides, and min, max and spread show the noise. A
  decode-only effect needs five repeats: decode's spread is about 6%.
- **The serve settings differ from the forced bench on purpose.** The server runs a 95 GiB RAM tier
  (`QX_PERF_SERVE_RAM`, recorded in `run.json` as `serve:QX_DS_RAM=95`), not 100: at 100 its RSS is 111 GB on
  the 120 GiB box next to the agent (see `findings.md`, "Operating envelope"); 95 against 90 measured serve3
  -5.4%. Its VRAM expert budget is what the context, the slots and the scratch
  leave of the 20 GiB, and its windows mix slots. That is the production shape; the forced bench is the engine's
  ceiling. The gap between `serve1` and `decode` is itself a finding.

**Secondary metrics.** Measure these when the change targets them, and record them in the notebook:

- **One long prompt** through the server: `tools/longprompt.py`.
- **Cold start and warm-up.** The mean over all 1,024 tokens, and the first request after a start with and
  without `--profile-out`.
- **Time to first token** in serving: the `ttft_med_s` and `ttft_p90_s` fields of the `serve*` rows.

## Host Reality

Regenerate this block rather than trusting it:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit,clocks.max.sm --format=csv
lscpu | grep -E "Model name|^CPU\(s\)|Thread|NUMA"; grep MemTotal /proc/meminfo
nvcc --version | tail -1; rustc --version; uname -r
lsblk -d -o NAME,MODEL,SIZE,ROTA      # the pack lives on the root NVMe; nvme1n1 is the Windows VM's disk: never touch it
```

**Current host (2026-11-05):**

- NVIDIA GeForce RTX 5090: SM 12.0, 32 GiB GDDR7, driver 595.99.02, 575 W limit.
- AMD Ryzen 9 9950X: 16 cores, 32 threads, two CCDs.
- 120 GiB DDR5, of which `MemTotal` is 126.5 GB.
- 4 TB NVMe root, which holds the pack.
- Kernel 7.0.0-38, CUDA 13.1 NVRTC and 13.3 nvcc, rustc 1.99.1.

The host fingerprint `perf_diff.py` derives is `rtx-5090_ryzen-9-9950x_121g`.

**Measured rates.** These are the rooflines to judge against, not spec sheets. Each row names the source entry in
`docs/progress.md`; re-measure with the named tool when a decision depends on one.

| path | measured | source |
|---|--:|---|
| RAM read, AVX-512, 16 cores | 54-56 GB/s (49-50 from one CCD) | `qx-rambw`, Phase 0, 2026-10 |
| CPU pool, DeepSeek expert records | 0.22 ms per 9.4 MiB expert, 44-54 GB/s | `qx-poolbw`, 2026-11-04 |
| PCIe host to device, pinned, large copies | 50 GB/s | Phase 0 probes |
| Staged expert DMA in the engine | 14-25 GB/s effective, about 0.7 ms per expert at 14 GB/s | staged share, 2026-11-04 |
| GPU in-place reads of mapped RAM | about 7 GB/s, 1.3 ms per expert | shelved, 2026-11-04 |
| NVMe pack reads, io_uring O_DIRECT | 17.7 GB/s at 48-record batches, 10.5 GB/s at 1-4 | tiers, 2026-11-03 |
| GPU DRAM read | 1.58 TB/s | Phase 0 probes |
| f32 multi-GEMV (`qx_mgemv`), one token | 1.2-1.6 TB/s | 2026-10 |

**Memory budget.** The RAM tier pins 100 GiB by default. About 11-13 GiB stay available with a server or
benchmark running, so a second heavy process will OOM the box. VRAM holds about 20-21 GiB of experts plus the
context, slots, staging area (about 1 GiB) and scratch. The engine caps the expert budget at free VRAM minus
1.5 GiB.

## Anatomy Of A DeepSeek Step

Read this before guessing where time goes. One step is a single token or a window of up to 128 tokens over the
slots. It is one CUDA graph replay per window width; there is no host involvement between launches. Per MoE
layer:

1. **GPU:** route, then `qx_moe_post` writes the layer's misses into the host-mapped mailbox. A share of the
   RAM-resident misses goes to a staged list instead: 6/8 for windows, 3/8 for decode. The post kernel then runs
   the VRAM-resident experts and the shared expert.
2. **Driver thread, `dsv41/driver.rs`:** sees the job and submits NVMe reads for misses that are on the drive
   only. It copies the staged records into the VRAM staging area over 4 copy streams, and runs the CPU pool over
   the RAM and NVMe misses. Then it stores `MB_STG_DONE`, then `MB_DONE`. It also performs promotions between
   tiers, which are asynchronous and event-ordered.
3. **GPU:** `qx_moe_wait_staged`, then the expert kernels over the staged slots, then `qx_moe_wait`, then the
   combine, which reads the pool's rows from the mailbox.

The decode budget at 31 ms per token (2026-11-04) breaks down like this:

| part | ms per token |
|---|--:|
| GPU-only step, measured with `QX_DS_NOEXPERTS=1`: flat in context at 1k, 24 ms at 16k | 13 |
| VRAM-resident experts | about 3 |
| CPU pool, of which about 3 ms is waiting on NVMe | about 20 |

The pool runs while the GPU waits, so decode is pool-bound. In prefill windows of 64, staging moves most of the
RAM experts to the GPU. Pool, copies and GPU expert kernels then end up close to balanced at about 12 ms per
token.

**Counters to read.** Every run's log has them, and `perf_diff.py parse` extracts them:

- **`pos N: ... vram A staged B pcie C ram D nvme E promos vram F ram G (H ms, pool I ms of which fetch J, stage
  wait K)`.** These are cumulative counts of expert evaluations per tier, promotions and their host time, and
  pool time. `fetch` is pool time spent waiting on NVMe, and `stage wait` is time spent waiting on staging
  copies. Divide by tokens for per-token numbers. Compare the start and end of the last 32 tokens for steady
  state.
- **`phases per step (last 32): uploads, launch/replay, device wait, history flush, logits download`.** These are
  host-side phases. `device wait` is everything asynchronous: GPU and driver together.
- **Diagnostic knobs:**
  - `QX_DS_NOEXPERTS=1` gives the GPU-only cost.
  - `QX_DS_SYNC=1` syncs after each phase.
  - `QX_DS_TRACE=1` names each launch.
  - `QX_GRAPH=0` disables graphs.
  - `QX_DS_VERBOSE=1` prints the counters, and adds a stuck-mailbox report after 15 s idle.
- **Microbenchmarks:**
  - `qx-poolbw`: the pool per expert.
  - `qx-rambw`: RAM read bandwidth.
  - `qx-membw`: pool plus concurrent PCIe reads, to see whether they share bandwidth.
  - `qx-kbench`: GPU kernels.
- **Profiles:** `nsys profile -t cuda,osrt --cuda-graph-trace=node -o <out> target/release/qx-ds-check ...` gives
  kernel times inside graphs. Do not commit profiler output. Record the path and a summary instead.

## Correctness Gate

```bash
QX_LLAMA_TOKENIZE=$HOME/git/llama.cpp-dsv41/build-tok/bin/llama-tokenize tools/ds-checks.sh
```

The gate takes about 6 minutes and prints six lines: the exact-f32 agreement (6/6, KLD 0.000000), the production
agreement (57/59, KLD 0.128643), windows of 8 and of 64 against single steps (128 / 128 each), the tokenizer
(identical), and since 2026-11-06 a rewind and a park + unpark against the straight pass (64 / 64 each: the
replay-based rewind). Run it in a capped unit, alone on the box. `bench.sh --check` runs it first and records
`pass` or `fail` in `run.json`. A run without a pass can be measured, but it cannot be landed, promoted
or put on the scoreboard. For changes to slots, serving, prefix reuse or parking, also run the server smoke test:
three concurrent requests, a repeat for prefix reuse, and a `json_object` request. An example is in
`handoff.md` and the 2026-11-04 entries of `docs/progress.md`.

## Measurement Harness

```bash
# a full measurement (3 repeats of decode, prefill64, long16k and serve; correctness first): about 75 minutes
perf/harness/bench.sh --label <what> --check
# a screening run (decode and prefill64 only): about 5 minutes, plus the check
perf/harness/bench.sh --label <what> --quick
# an experiment: knobs passed to every engine run, the server included (recorded in run.json)
perf/harness/bench.sh --label stage7 -- QX_DS_STAGE_SHARE=7
# compare, gate against the host's committed baseline, score
python3 perf/harness/perf_diff.py diff perf/results/<date>/<base> perf/results/<date>/<cand>
python3 perf/harness/perf_diff.py gate perf/results/<date>/<cand>
python3 perf/harness/perf_diff.py score perf/results/<date>/<cand> --record perf/scoreboard.md --note "<what changed>"
# after a landed win: make it the baseline, and write the notebook entry
python3 perf/harness/perf_diff.py promote perf/results/<date>/<cand>
python3 perf/harness/perf_diff.py entry perf/results/<date>/<cand>     # paste, complete, then:
python3 perf/harness/perf_diff.py index perf/optimization_status.md
```

`bench.sh` re-executes itself inside a `MemoryMax=118G` scope unless it is already capped. It waits for the GPU
to be free before every engine run, gates on cargo's exit status, and takes git provenance when the run starts:
the commit, the modified files, and the working-tree diff, saved as `working-tree.diff`. A full run takes about
75 minutes with `--check` (long16k is 10 minutes a repeat, serve about 5); `--quick` takes about 5 plus the check.
The serve case starts the server on port 8089 (`QX_PERF_PORT`), waits for its warm-up line, drives it with
`tools/loadgen.py`, then kills it by PID and waits for the GPU to be released. Its server log is
`logs/serve-N.server.log`. To keep a run off the agent's foreground, run it as a user service:

```bash
systemd-run --user --unit qx-perf-<label> -p MemoryMax=118G -p MemorySwapMax=0 --working-directory=$PWD \
  bash -c 'exec perf/harness/bench.sh --label <label> --check > <log> 2>&1'
```

## Timing Rules And Traps

Each of these traps has produced a wrong number in this project at least once.

- **One heavy job at a time.** The box has no memory to spare. Two engines pinning RAM, or a build next to a run,
  either OOM the box or contend for the RAM bandwidth the pool runs on. The production Qwen service
  `quixiinfer.service` must be stopped, and `bench.sh` refuses to start while the GPU is in use.
- **Teardown.** A process that pinned 100 GiB takes 10-30 s to release its VRAM after it exits. Starting the next
  run sooner OOMs the VRAM budget or measures a smaller tier. `bench.sh` waits for the GPU to be free.
- **Stale binary.** A service without `~/.cargo/bin` on `PATH` silently ran an old binary twice. Gate on cargo's
  exit status, not on a pipeline's: `cargo ... | grep` reports grep's status.
- **Warm-up and capture costs in averages.** A 16k "decode growth" turned out to be the 0.9 s first capture of
  width 1, averaged over a 64-step tail. Use the steady last-32 metric, and warm every width first. The server's
  warm-up captures the fixed width set, `WIDTHS` in `worker_ds.rs`.
- **The wrong token file.** `wiki1k` was tokenized with Qwen's tokenizer. Its ids above the vocabulary produced
  NaN, and the engine now asserts on them. Benchmarks use `route1k.tokens` and `route16k.tokens`.
- **Different settings are different experiments.** VRAM and RAM tier sizes, the promotion rate, the profile,
  slots and context all move the numbers. `diff` refuses runs whose settings differ.
- **Noise.** Run-to-run spread on this box is about 1-6%. On 2026-11-05 the same commit measured 33.1 ms per
  token for decode in one run and 31.2-31.3 in two others, and 12.4 against 11.7-11.8 ms for prefill. Do not
  read a single run, and do not accept a change inside the repeats' spread. Re-run when spread exceeds 10%.
- **Intermittent hangs are bugs, not noise.** `bench.sh` runs each engine under a 20-minute timeout. A timeout is
  a failed repeat and needs a diagnosis. Get thread stacks with
  `sudo -n gdb -p PID -batch -ex "thread apply all bt"`, because `ptrace_scope` is 1. CubeCL's server thread and
  ours are both named `DSD-0-0`.
- **Never kill with `pkill -f` or `pgrep -f`** using a pattern that appears in the calling shell's own command
  line. It kills the shell. Use `pkill -x qx-ds-check`, or a PID.

## Optimization Loop

1. **Inventory.** Read `findings.md`, the backlog entry and the code path. Know the current route, for example
   "decode: staged share 3/8, pool takes the rest".
2. **Baseline.** Use the committed baseline for this host. If the code or settings moved since, take a new one
   with `bench.sh --check`.
3. **Classify.** Name the bottleneck from the counters with a number, for example "pool 20 ms of the 31, 3 of it
   NVMe fetch". Work out the expected gain from the measured rates before editing. If the estimate cannot clear
   the bar, do not build it.
4. **Hypothesis before edit.** Write it in the notebook entry: what changes, which counter should move, and by
   how much.
5. **Experiment.** Change one factor at a time, small and reversible, and prefer a knob. Run the focused check
   first, then `bench.sh`.
6. **Decide** with the rules below.
7. **Record.** Add a notebook entry, whatever the verdict. Update `findings.md` for a durable truth or a
   rejection, and `backlog.md` for the beam's state. Put the score on the scoreboard for a landed win, and
   promote the baseline.
8. **Commit only when the user asks.** Gate the commit on cargo's exit status and a passing `ds-checks`.

## Decision Rules

**Land** a change when all of the following hold:

- `ds-checks` passes exactly.
- The score or the targeted case improves by at least 3% for a low-risk change, such as a knob default or a
  local kernel change. A change that adds complexity needs 8-10%: a new thread, protocol, buffer class or kernel
  family.
- The improvement is larger than the repeats' spread.
- No other standard case regresses by more than its spread.
- A counter explains the win, for example "pool ms per token fell from 20 to 14 because staged hits rose".

**Reject or defer** when the win is inside noise, or appears only at a non-standard setting or width without a
real-workload reason. Also reject or defer when the change adds substantial complexity for a marginal gain, or
breaks exactness. A rejected idea goes into `findings.md` with the reason and numbers, so it is not retried
blindly.

**Serving and long-context changes** use the same bars on their own metrics, measured the same way, with
repeats, spread and provenance.

## Recording Format

The notebook entries in `optimization_status.md` are append-only and ordered oldest first, under headings of the
form `## YYYY-MM-DD: <name> — VERDICT`. The verdict is one of:

- **LANDED:** kept, in the code.
- **KEPT:** kept behind a knob.
- **REJECTED**
- **CANDIDATE:** awaiting the full check.
- **DEFERRED**
- **INCONCLUSIVE**
- **MIXED**
- **RECORDED:** a measurement or baseline, not a change.

Each entry carries:

- the status;
- the bottleneck hypothesis, with its counter;
- the change, as one factor;
- the correctness result;
- the run label, commit, host and settings;
- a baseline-vs-candidate table, with the counters that explain it;
- the decision, with the bar it cleared or missed;
- rejected alternatives and open questions.

`perf_diff.py entry <run>` pre-fills the run-specific parts. Regenerate the index with `perf_diff.py index`
after appending. Raw run directories are git-ignored and do not survive a clone, so copy the numbers into the
entry rather than citing a path as the only evidence.

## Experiment Catalogue

These are the engine's levers, grouped by bottleneck. They are starting points for hypotheses, not rules.

**Expert residency (pool- and drive-bound).**
- The VRAM tier size against context and slots. Every GiB is about 100 more resident records.
- The promotion rate (`QX_DS_PROMOS`) and the scoring's decay and quarantine.
- The starting profile: `golden/profile-dsv41.txt` against a live `--profile-out` file.
- Per-layer budgets instead of a global ranking.
- Narrower store sizing that frees VRAM. Compressed stores are allocated at `max_ctx` rows per slot, although
  only `pos / ratio` rows are used.

**The GPU's share of misses (transfer-bound).**
- `QX_DS_STAGE_SHARE` and `QX_DS_STAGE_SHARE_DECODE`, `QX_DS_STAGE_CAP` and `QX_DS_STAGE_STREAMS`.
- Selecting staged experts by cost: records shared by several tokens of a window first, NVMe misses never.
- Overlapping the staging copies with earlier layers.

**The pool (pool-bound).**
- Core sets (`QX_DS_CPUS`) and the main thread's CPU (`QX_DS_MAIN_CPU`); SMT measured slower for the pool.
- Per-format kernels at the RAM's bandwidth.
- Splitting a layer's experts across CCDs, and NUMA-aware record placement.

**NVMe (drive-bound).** The io_uring depth and batch size, submitting the next layer's likely misses early, and
the pack's record layout. Never touch `nvme1n1`.

**GPU kernels (GPU-bound, only when the GPU is on the critical path).**
- The f32 multi-GEMV, attention (`qx_ds_attn_part` and `merge`), hc mixes, top-k and compression.
- Context growth at 16k: 13 to 24 ms GPU-only per decode step.
- `qx-kbench` and nsys for isolation.

**Windows and serving.** The prompt chunk width (64 is the best measured), `WIDTHS`, decode-and-prefill mixing in
continuous batching, slot count against VRAM, and prefix reuse and parking.

## Open Items

1. ~~The token files are not in git.~~ Vendored in `perf/data/` on 2026-11-06. `tools/ds-checks.sh` still reads
   the larger reference files from the git-ignored `golden/data/`, so the correctness gate runs only where those
   exist.
2. ~~Serving and long-context cases in the harness.~~ `long16k` and `serve` landed on 2026-11-06. Still open: the
   serve case has no engine counters (the server log holds the per-step prints, but `parse` does not read them
   yet), and `serve3`'s noise level is unknown until the first baseline's repeats.
3. **Re-measure the rooflines** in Host Reality on the current driver and kernel. They date from October, partly
   on the Qwen setup.
4. **An nsys recipe** for a 64-token window and a decode step, with the kernel table recorded in the notebook.
5. **The Qwen engine** is outside this harness. Its tools are `qx-bench`, `qx-ppl`, `tools/loadgen.py` and
   `quixiinfer.service`. Do not run Qwen work without the user's go-ahead. The shared mailbox limits
   `QX_TMAX`/`QX_AMAX` were raised for DeepSeek, so Qwen's VRAM budget needs a check before that service runs
   again.
6. **Record clocks and power state** in `run.json`: SM clock and power draw during the run.
