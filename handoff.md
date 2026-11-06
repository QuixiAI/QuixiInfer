# Handoff: DeepSeek-V4.1-Flash in QuixiInfer (updated 2026-11-05)

## Goal

Run [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) at
UD-IQ2_XXS in QuixiInfer on this box, as fast as possible. The results must stay exact against the CPU reference.
The box has an RTX 5090 (32 GB), a Ryzen 9950X (16 cores), 120 GiB of RAM and a 4 TB NVMe root. The experts live in
three tiers: VRAM, pinned RAM and the NVMe pack.

## Update 2026-11-06

- **The user set an autonomous optimization loop:** `goal-prompt.md` at the repo root is the directive (read it
  first). It runs experiments through `perf/`, keeps only measured wins, and records everything.
- **Harness set B.** `perf/harness/bench.sh` now runs `decode`, `prefill64`, `long16k` and `serve` by default
  (`--quick` for the first two); token files are vendored in `perf/data/`. `perf_diff.py` refuses to record or
  promote a partial case set. Set A and set B scores are not comparable (`perf/scoreboard.md`). **The set B
  baseline is recorded: 34.327** (decode 33.1, prefill64 12.1, long16k 36.4, serve3 95.2 ms per token), promoted
  to `perf/baselines/rtx-5090_ryzen-9-9950x_121g/`.
- **Measured today, no code change:** narrow windows cost 23.6 / 19.4 / 15.7 ms per position at widths 2 / 4 / 8,
  so DSpark-style drafting breaks even at 41-46% acceptance and gains 6% at 75%: backlog family 6 is DEFERRED.
  Steady decode at 16k is 36 ms per token, not 53 (the old figure averaged a graph capture). Windows of 5 are
  125/128 bit-identical to single steps (open, `findings.md`).
- **Kept:** the window staged share (6/8) applies from width 2 (`QX_DS_STAGE_WIDE_MIN`, was 8): -4.7% at width 2,
  -8.2% at width 4 on the forced bench; serve3 -2.9%, within noise. The serve case's RAM tier is 95 GiB (serve3
  -5.4% against 90); PROMOS=64, STAGE_SHARE=4 and STAGE_CAP=192 measured flat (notebook). The set B baseline was
  re-taken at the 95 GiB serve tier and promoted: **score 34.321** (decode 33.0, prefill64 12.4, long16k 37.3,
  serve3 90.9), `serve:QX_DS_RAM=95` in its `run.json`. Gate every later run against it.
- **Two bugs found by the serve case, both fixed in the working tree:**
  1. A capture race: the driver thread's `cuEventSynchronize`/`cuEventQuery` during a global-mode graph capture
     panicked the server's warm-up. Fix: `Shared::capture` mutex plus a device drain before recording
     (`dsv41/driver.rs`, `dsv41/mod.rs`; `docs/progress.md` 2026-11-06).
  2. Host memory: under serving load the server's anonymous memory grew at 330 MB/s until an OOM kill (at a
     118G cap the *global* OOM killer). Real cause (after a wrong first diagnosis of glibc fragmentation): the
     width-6 kernel `qx_mgemv_t6` was not in the warm-up, and when it first compiled mid-serving its module
     could not load (`CUDA_ERROR_OUT_OF_MEMORY`: the tier left 1.5 GiB and the graphs took 1.2); CubeCL retried
     on every launch (63 MiB of NVRTC host memory each) and the window ran without the GEMV. Fix: the server
     warms every width in `WIDTHS`, `qx_gpu::kernel::seal()` makes any later compile abort, the VRAM headroom is
     2.5 GiB (`QX_DS_VRAM_RESERVE_MIB`), and `QX_DS_VERBOSE` logs compiles. `tune_host_allocator()` (pinned malloc
     thresholds) stays as hygiene. Rules in `perf/findings.md` "Operating envelope": serve at 90 GiB,
     `MemoryMax=112G` with `OOMPolicy=continue`, launch heavy runs detached (`nohup setsid`), never as the
     session's tracked background tasks (its memory guard kills them when the pinned tier leaves little free RAM).
- The Qwen `quixiinfer.service` was found crash-looping every 10 s (18,742 restarts, `status=200/CHDIR`) and was
  stopped with `sudo systemctl stop`; it stays enabled, as before.
- Ideation from `~/git/{colibri,Strata,basalt,DeepInfer}` is in `perf/backlog.md` families 6 and 7 and in
  `perf/findings.md` "Rejected elsewhere".
- **Where the loop stopped (end of the 2026-11-06 session):** the standard-setting levers are measured to their
  ceilings. Decode is pool-bound at the box's DRAM read ceiling (52-59 GB/s measured; the pool runs 44-54); the
  dense GEMVs are 6.7 ms of a 33 ms step at 54% of GPU memory peak with a 2-4% fusion available (family 8,
  deferred); prefill at the 64-chunk is copies and GPU expert kernels, and the 128-chunk bundle (share 8/8, cap
  256) is -12.6% to -16.6% as an opt-in that costs serve3 +10%; serving is residency (the 95 GiB tier landed
  -5.4%). What remains needs a decision or a project: a staging area that borrows tier VRAM only while a wide
  window runs (would give the bundle's prefill without the serve3 loss), a change of the standard prompt chunk,
  per-request expert hot sets for serving, or DSpark drafting (needs the drafter weights and 80% acceptance). The first long shot is the narrow-window measurement for DSpark-style
  drafting (family 6); the UD-IQ2_XXS GGUF has no MTP tensors, so a drafter would need a separate download.

## Update 2026-11-05

- The verified work described below is now committed and pushed as `f359604`. The small slowdown was noise.
  Repeats of the same commit measured decode 31.2-33.2 ms and prefill64 11.7-12.4 ms per token.
- **There is a new performance scaffolding in `perf/`**, built at the user's request on the model of
  `~/git/QuixiCore/QuixiCore-CUDA/perf/perf.md`. It is committed and pushed along with this file. Read `perf/perf.md` before any speed
  work. Measure with `perf/harness/bench.sh --check`, then compare with `perf/harness/perf_diff.py gate`. The
  first baseline is promoted: decode 32.0 and prefill64 11.9 ms per token, score 19.514. The ranked next work is
  in `perf/backlog.md`.
- **Parking idle sequences is in progress and uncommitted.** It lives in `crates/qx-engine/src/dsv41/mod.rs` as
  `DsEngine::park` and `unpark` plus `Parked`, and it has not been built yet. Still to do:
  1. Add a `qx-ds-check` mode that parks slot 0 partway through, runs other tokens through the slot, unparks it,
     continues, and compares the logits with the plain run. They must be bit-identical.
  2. Add the worker's LRU of parked sequences with a byte budget, and pick parked prefixes at admission.
  3. Run the server test.
- Store sizing note for the backlog's family 3: the compressed stores allocate `max_ctx` rows per slot, but only
  `pos / ratio` rows are used.

## Ground rules from the user (all still in force)

- **Never touch the Windows drive `nvme1n1`** (by-id `nvme-CT4000T705SSD3_2529E9C6B4B9`). It is the win10 VM's
  disk. The VM is off. Stop it only gracefully, and never destroy it.
- **Work on DeepSeek, not Qwen.** The Qwen production service `quixiinfer.service` is stopped but still enabled at
  boot. Do not run Qwen checks unless the user says so.
- **Do not OOM the CPU RAM.** An OOM kills the agent harness. Run heavy jobs one at a time, as memory-capped
  systemd user units:
  `systemd-run --user --unit qx-NAME -p MemoryMax=118G -p MemorySwapMax=0 bash script.sh`.
  Inside the script, use `exec > log 2>&1` and `export PATH=$HOME/.cargo/bin:$PATH`.
- **The user owns the machine.** Anything on the GPU or system that is not ours may be killed. The Windows disk is
  still off limits.
- **Commit and push only when asked.** The user asked once for this workstream. Origin's default branch is
  `master`. `tools/golden/golden-dsv41` is a tracked binary that is modified and left uncommitted on purpose.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Where things stand

*(Historical: this section was written before `f359604`, which committed everything listed here.)*
The last commit was `c083124` on master. The working tree held verified, uncommitted work in
`README.md`, `crates/qx-engine/src/dsv41/{driver.rs,mod.rs}`, `crates/qx-engine/src/mailbox.rs`,
`crates/qx-gpu/src/lib.rs`, `docs/progress.md`, `kernels/include/qx_sys.cuh`, `kernels/step.cu` and
`tools/ds-checks.sh`. It contains:

1. **DMA-staged GPU share of the missing experts.** The post kernel sends a share, in eighths, of the RAM-resident
   misses to a staged list. The driver copies those records into a VRAM staging area over 4 copy streams. The GPU
   waits on `MB_STG_DONE`, then runs the same expert kernels over `slots_stg`. The knobs are `QX_DS_STAGE_SHARE`
   (default 6, for windows of 8 tokens or more), `QX_DS_STAGE_SHARE_DECODE` (default 3), `QX_DS_STAGE_CAP` (96)
   and `QX_DS_STAGE_STREAMS` (4).
2. **Fixed window widths and a warm-up.** Window widths come from the fixed set `WIDTHS` in
   `crates/qx-serve/src/worker_ds.rs`, and the server's warm-up captures their CUDA graphs. The engine also
   records per-step phase timing.
3. **A fix for an intermittent hang.** The full write-up is in `docs/progress.md` under "An intermittent hang,
   found with a debugger". CubeCL compiled and loaded a kernel, via `cuModuleLoadData` on its server thread named
   `DSD-0-0`, the first time an 8-token window launched it. That happened while `qx_moe_wait` spun on the GPU for
   the driver, and the module load and the driver's copies deadlocked. The fix is a dry compile pass for each
   window width before its first real window. Bit 32 of `qx_moe_post`'s width word, `info[13]`, marks a dry run.
   A dry run posts no job and zeroes the slot lists. See `warmed` and `warm` in `dsv41/mod.rs`. Two smaller
   changes stay as hygiene: `CUDA_MODULE_LOADING=EAGER` in `qx_gpu::init`, and a copy-path warm-up before
   `Driver::start` returns.
4. **GPU-free waits in `tools/ds-checks.sh`** before every engine run. A process that pinned the RAM tier takes
   10 to 30 s to release its VRAM after it exits.

### Verification of the uncommitted tree

`tools/ds-checks.sh` passed **three consecutive full runs**. Before the fix it hung about one run in three. Each
run gave these results:

```
1. exact f32 vs reference:      positions 6: top-1 agreement 6/6, KLD mean 0.000000
2. production vs reference:     positions 59: top-1 agreement 57/59, KLD mean 0.128643
3. windows of 8 vs single steps: bit-identical rows: 128 / 128
3. windows of 64 vs single steps: bit-identical rows: 128 / 128
4. tokenizer vs llama.cpp:      identical: 20 tokens decode round trip: exact
```

The speed run with the same tree used the forced benchmark on route1k with 20 GiB VRAM and 100 GiB RAM:

| Measurement | Before the hang fix | Now |
|---|---|---|
| Prefill, windows of 64, last 32 | 11.8 ms/token (85 tok/s) | 12.4 ms/token (81 tok/s) |
| Decode steady, last 32 | 31.4 ms/token | 33.1 ms/token |

The small slowdown is unexplained. The dry pass runs only once per width, so it should not affect steady state.
It may be run-to-run noise, so re-measure before acting on it.

The server smoke test passed with 3 slots. Warm-up took 8.4 s. Three concurrent requests answered correctly, a
`json_object` request returned `{"celsius": 100}`, and the second round was faster at 4.0 s wall against 12.9 s.

### Committing

Commit and push only when the user asks. Gate on cargo's exit status, not on the status of a grep pipeline: a
non-compiling commit, `b5a0818`, was pushed that way once.

## Open work, in the agreed order

1. **Park idle sequences and handle KV beyond `--ctx`.** Move idle slots' state out of VRAM and handle contexts
   longer than the ring.
2. **Qwen VRAM and load check.** Only with the user's go-ahead. `QX_TMAX` went from 32 to 128 and `QX_AMAX` to
   1280 in `qx_sys.cuh`. These constants are shared with the Qwen engine, so its VRAM budget must be re-checked
   before `quixiinfer.service` is used again.
3. **Investigate the small decode and prefill slowdown** noted above, if it reproduces.

## How to run things

- **Build:** `cargo build --release -p qx-engine -p qx-serve --bin qx-ds-check --bin quixiinfer`. Building
  `step.cu` or `dsv41.cu` can take about 10 minutes.
- **Model:** `~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf`.
  The pack is `QX_DS_PACK=.../UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp`.
- **Regression set:** run `tools/ds-checks.sh` with
  `QX_LLAMA_TOKENIZE=$HOME/git/llama.cpp-dsv41/build-tok/bin/llama-tokenize`. It takes about 5 minutes. Run it in a
  capped unit with `timeout 1500`.
- **Forced benchmark:**
  `QX_DS_PACK=... QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32 QX_DS_VERBOSE=1 target/release/qx-ds-check <model> golden/data/route1k.tokens 1024`.
  Add `QX_DS_WINDOW=64` to measure prefill windows instead of decode.
- **Server:**
  `QX_DS_PACK=... target/release/quixiinfer serve --model <model> --ctx 8192 --slots 3 --window 64 --ram-gib 100 --port 8089`.
  Run it under `MemoryMax=118G`. A service example is in `tools/quixiinfer-deepseek.service.example`.
- **Before each CUDA job, wait for the GPU to be free:**
  `until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done`.

## Hard-won lessons

- **Debugging a hang:** `ptrace_scope` is 1, so plain gdb cannot attach. Passwordless sudo works:
  `sudo -n gdb -p PID -batch -ex "set pagination off" -ex "thread apply all bt 16"`. The CUDA driver thread names
  are misleading. `DSD-0-0` is CubeCL's server thread as well as ours.
- **Never let a kernel compile or load for the first time while an expert job is posted.** Any new kernel or new
  window width must go through the dry pass.
- **Never run `pkill -f` or `pgrep -f`** with a pattern that appears in the tool shell's own command line. It kills
  the shell. Use `pkill -x qx-ds-check` or kill by PID.
- **Wait on specific PIDs.** A bare `wait` in a script that started a server waits for the server.
- **Never use `DeviceBuf::upload` into memory a queued kernel reads.** It goes on the legacy stream and races. Use
  a mapped buffer plus `qx_ds_copy_f32`.
- **Keep DeepSeek GEMVs on the f32 path.** The bf16 tensor-core path breaks bit-identity between windows and
  single steps. `QX_TMV=1` opts into it.
- **Graph capture:** prepare, eager run and record must happen in one call, and grids must not depend on position.
  A new width costs about 0.9 s to capture.

## Further reading

- `docs/progress.md`: dated entries for 2026-11-03 and 2026-11-04 with every measurement and fix.
- `docs/deepseek41-plan.md`: the plan and milestones.
- `README.md`: the DeepSeek section for users.
- Scratch scripts and logs from this session (`run_kNN.sh`, `kNN.log`) are in
  `/tmp/claude-1000/-home-eric-git-QuixiInfer/7b27a076-c316-432c-a7a6-fc085686af11/scratchpad/`. They may not
  survive a reboot.
