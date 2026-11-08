# Handoff: DeepSeek-V4.1-Flash in QuixiInfer (updated 2026-11-08, evening)

## Goal

Run [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) at
UD-IQ2_XXS in QuixiInfer on this box, as fast as possible. The results must stay exact against the CPU reference.
The box has an RTX 5090 (32 GB), a Ryzen 9950X (16 cores), 120 GiB of RAM and a 4 TB NVMe root. The experts live in
three tiers: VRAM, pinned RAM and the NVMe pack.

## Update 2026-11-07, evening: the priority is prefill (Strata's 935 tok/s at 16k against our 190-250)

- The user: "double our efforts to match Strata's prefill". The chunk profile (notebook, backlog family 14) says
  the dense one-launch GEMV is 42% of the chunk and latency-bound, attention 22%, the exact expert tiles 31%.
  Row-blocking with per-group barriers was 12% slower (rejected); the one-barrier form (`QX_MGEMV_ZB=1`, d88f99c)
  and the one-barrier + 4-rows form (`QX_MGEMV_ZT=1`, in the tree) are measuring (chains zb, zt in the scratchpad;
  `~/qx-bins/chunk21`, `chunk22`). Land whichever clears the bar bit-identically, then attention.
- The price of a non-exact prefill is measured (notebook, findings): Strata's FP16 tiles add nothing over its own
  decode path; its engine sits at KL ~0.03 from our exact path, ours at 0.01-0.02 from f32. A bf16 tile path is a
  lossy opt-in option by the goal's rules, not taken; exact forms first.
- Landed since noon: `qx_ds_topk2` (decode256k 45.4 -> 43.5-44.2, 8fae0e7), the server's 4096 chunk default
  (42c23e7), README on today's facts (53a86c8). Set D row 31.470 (d6cfb45).

## Update 2026-11-08, evening: the flex tail landed; the 256k wait is the CPU pool; portability to the 30/40 series begun

- **Elastic flex tail (bb9e370):** the chunk path's buffers are views into the VRAM arena's last `flex_bytes`;
  `forward_chunk_opt` asks the driver (`flex_check`) to evict every record any of whose bytes lie in the tail
  (a slot straddling the tail's start counted too: the first build tested starts only, overwrote one straddling
  record and the 4096 tail differed), `chunk_release` (worker, once no other prompt is pending; qx-ds-check after
  the prompt's last chunk) gives it back with the evicted records copied straight back into their slots. 16k tails
  bit-identical at 4096 and 8192; ds-checks pass. Server TTFT at 262k/3 slots on the 13.5k prompt: 8192 chunks
  35.5 s, 4096 42.3 s, both at the full 12.2 GiB expert tier. serve256k: 8192 arm 105.5 (baseline 109.65), **4096
  arm 76.8 / 27.4 (serve256k3 / serve256k1)**, which is 30% better than anything before and not explained yet: the
  tier counters show 69% VRAM hits against 53-55% with the same 1270 records; the chunk path never ran (88-token
  prompts). An A/B/B/A with text dumps (`chain-srv.sh`, `tools/loadgen.py --dump`) is queued to tell a real effect
  from a changed generation. Do not promote the 4096 number until that is understood.
- **Staged-ready words (knob `QX_DS_STG_WAIT=words`, off):** the GPU's staged wait no longer depends on the driver's
  done word, which was posted after the pool share. Equal in time (decode256k 44.7/44.6 vs 44.8/44.9): the pool
  share finishes no sooner than the copies. What the counters say about the 28 ms per 256k token: the pool window
  is 22 ms (~47 records at 0.37 ms on 16 threads, 4.8 ms of it NVMe fetch waits on 7 records). Tests queued:
  decode staging share 8/8 and 5/8 under the words path (`STGS`), the pool on all 32 hardware threads (`CPUS`).
- **PCIe landing time measured:** 0.19 ms per 9.5 MB record by memcpy or by gather kernel, the bus ceiling; the
  gather-from-mapped form Big-MoE-Small-Box uses buys nothing here (`kernels/bench/pcie_bench.cu`).
- **Big-MoE-Small-Box, second pass** (backlog family 15, items 9-18): dataflow CPU job without stage barriers,
  MWAITX idle waits for the power budget (their lane cores 4.34 -> 4.8 GHz, -27% per expert), per-layer makespan
  planner, NVMe piece streaming into the pool, per-layer timeline instrumentation. Measure first: one record on one
  core without barriers; turbostat during decode.
- **Portability (the user's ask, in progress):** `qx_gpu::mem::arch()`/`sm_count()`/`nvrtc_arch()` replace the
  hard-coded `sm_120a` in the raw launcher, the NVRTC validation and the tmv launch plan; software E4M3 conversions
  below sm_89 (`kernels/bench/e4m3_check.cu` checks every float against the instructions; queued on the GPU);
  `qx_tmv`'s bf16 form compiled out below sm_90 and never selected there (`use_tmv`); `QX_NVCC_ARCHS` for the
  build-time nvcc check. attn/gemm/prefill/smoke now compile for sm_86; the full sm_86/sm_89 matrix is running
  (mgemv's ptxas takes minutes per target). README has an "Other GPUs" paragraph. Untested on real 30/40-series
  cards: nothing here can be measured on this box beyond compiling.

## Update 2026-11-08, morning: set D 30.210; the GEMV kernels and the raw launcher are the defaults

- Set D on fd2473e with `QX_RAW_LAUNCH=1` (3113dd8 makes it the default): **31.470 → 30.210**. prefill256k 6.2 → 5.6
  ms/token (179 tok/s on the last 16k of a 256k prompt; 21.5 at the start of 2026-11-07), decode256k 44.9 (the
  tiled top-k), serve256k3 +1.2% (below the bar), guards within noise. Promoted.
- Defaults now: `QX_MGEMV_ZT` on (one kernel per segment kind: `_t8zt` Q8 4 rows, `_t8zf` f32 2 rows decode-once,
  `_t8z` rest), `QX_RAW_LAUNCH` on (chunk pass launches through the driver API on CubeCL's stream, pinned info ring;
  `qx_gpu::raw`, never inside a graph capture). Both `=0` restore the old paths.
- Queued on the GPU: the slab attention under the raw launcher (`QX_DS_ATTN=tile`, whole-window slice), the
  per-record PCIe latency bench (`kernels/bench/pcie_bench.cu`), the elastic flex tail (uncommitted in the tree:
  chunk buffers as views into the VRAM arena's tail, evicted for a prompt's duration by the driver's `flex_check`,
  `chunk_release` from the worker; chain-flex builds chunk28 and tests tails, the server's TTFT and serve256k at 8192).
- Read: Big-MoE-Small-Box (backlog family 15, findings): elastic VRAM (the above), per-record PCIe landing latency,
  layer-ahead router precision to measure offline, DSpark's 75% acceptance, serving hardening. LICENSE: MIT.

## Update 2026-11-08, small hours: the launch-path fix is written; everything is behind knobs and under test

- **Measured dead ends tonight** (all bit-identical, all in the notebook): rows per block with per-group barriers
  (+12%), one barrier per block (+3%), the slab attention at a 32-token slice (+5%), a 262k-entry CubeCL info cache
  (18.6 -> 46.6 s chunks, degrading: the pool does not scale), the 4-rows GEMV as one kernel (halves the Q8
  launches, 409 us; the union of branches makes the one-row fallback 1.85x slower on the Q6_K/Q5_K layers).
- **In the tree, off by default, compiled (chunk26), tests queued:** `QX_MGEMV_ZT=1` (one kernel per segment kind
  on sub-tables: `_t8zt` Q8 4 rows, `_t8zf` f32 2 rows decode-once, `_t8z` the rest), `QX_DS_ATTN=tile` (now the
  whole-window slice), and the big one: `QX_RAW_LAUNCH=1` (`crates/qx-gpu/src/raw.rs`): the chunk pass's launches
  through the driver API on CubeCL's stream, info words through a pinned ring, no channel, no pool. Chains:
  `chain-zt3.sh` (ZT=1: ds-checks, 16k tails, pairs), then `chain-raw.sh` (raw on/off tails and host breakdowns).
  Expected: the chunk's 8.7 s slab enqueue and 6.9 s expert enqueue mostly gone, then the kernel forms show.
- **Tooling that changed:** `kernels/bench/mgemv_bench.cu` (`~/qx-bins/mgemv_bench rows cols tokens [q6k]`): a
  kernel form is measured there in seconds and compared bit for bit before any engine run. `qx-gpu`'s NVRTC
  validation is PTX-only (a real target ran ptxas for 15 minutes per changed kernel file). `third_party/cubecl-cuda`
  is a path-patched copy (the info cache size from `QX_INFO_CACHE_ENTRIES`, the stream handles in `QX_STREAMS`).
- **Rules learned the hard way (memory notes):** no cargo work of any kind while an engine run is up (the reaper
  killed a `cargo check` at 100 GB pinned); a copied chain keeps bare binary numbers in its loops; two gpu_free
  waits race (an nsys run OOMed against a ds-checks run); every kernel-file change costs 20+ minutes of compiles
  before the first check line.
- **If the raw launcher lands:** re-measure ZT and the slab attention under it, then set D with
  `--check`, then the server (the chunk path is its prompt path: time to first token).

## Update 2026-11-07, night: the chunk path is launch-bound, not kernel-bound

- Every exact kernel form of the evening was bit-identical and read +5% in the engine's chunk timing while the
  standalone bench (`kernels/bench/mgemv_bench.cu`, `~/qx-bins/mgemv_bench`) said 2.3x per launch for the 4-rows
  GEMV. The chunk log explains it: of an 18.6 s chunk, "slab enqueue" (host) is 8.7 s and "expert enqueue" 6.9 s;
  the GPU is only 2.7 s ahead at the end of the slab passes. The chunk19 nsys trace: per slab-layer 55 dense
  launches and **93 host-to-device copies of 28 us** (CubeCL stages each launch's info words; its 4,096-entry
  cache of identical words, `cubecl-cuda/src/compute/stream.rs`, misses because the slab views change the
  pointers every slab, and every expert launch's words differ by expert). 2.6 of 11.5 GPU-ms per slab-layer and
  ~120 us of host time per launch. A faster kernel cannot shorten the chunk until the launch path does.
- Levers, by expected value: (1) identical info words across slabs (fixed per-slab staging buffers with a
  copy in/out, so a layer's 55 launches hit the cache from the second slab on; 2,200 distinct sets fit the 4,096
  cache); (2) the same for the expert jobs (per-layer fixed slot words); (3) fewer launches per slab-layer
  (fuse rms/rope/hc, the 8 output-A groups into one); (4) a patched cubecl-cuda with a larger metadata cache;
  (5) a raw driver-API launcher for the chunk path (kernel params by value, no staging). A host-side perf
  profile (chain-perf, `perf-chunk.txt`) is queued to see where the 120 us go.
- Kernel forms behind knobs, all exact, all measured in the bench and the engine: `QX_MGEMV_ZT=1` (4 rows per
  block on Q8 segments, 2 rows with decode-once on the f32 ones; bench 2.3x / 1.36x per launch), `QX_DS_ATTN=tile`
  (slab attention, merge in-block, 5% slower at a 32-token slice; the whole-window slice is in the tree, untested).
  Once the launch path is fixed, re-measure both; they should then show.

## Update 2026-11-07, noon: set D promoted at 34.447; the GEMV's register story; the 256k cases re-based

- **Scoreboard:** the chunk-path tree (6d8ce02) is the baseline, score 49.601 → **34.447** (prefill256k 21.5 →
  6.8 in 4096-row chunks, exact). Then the set D `--check` run on the final tree (0ced38e, `~/qx-bins/chunk17`,
  label `setd-split4`): **31.470**, promoted (d6cfb45): decode256k 46.4 (golden profile), prefill256k 6.2
  (8192-row chunks), serve256k3 108.3 (the GEMV split), every guard at or better than before. Two of the three
  scored wins are case settings, said so on the scoreboard.
- **The decode-side drift explained and mostly undone:** every decode case was +1 to +4.7% on the promoted tree.
  Cause: the in-block group loop in the shared `mgemv` template took the width 1-3 entries from 80 to 128
  registers (ptxas `--resource-usage` across four versions of `mgemv.cu`: notebook). Landed (0ced38e): two kernel
  bodies verbatim, `mgemv` (04b3fda's one-group code) for the width 1-8 entries, `mgemv_z` (6d8ce02's loop) for
  three new `_t8z` entries the one-launch path uses (`ZENTRIES` in `ops.rs`). Decode 34.2 → 33.4-33.7 (chunk9
  was 32.7; the rest is inside the case's 2-4% spread), the chunk's slab passes equal to before (alternating pairs),
  prefill128 11.1 against 10.9. Three intermediate rewrites of the loop (hoisted segment words, conditional
  barrier) were 3% slower on the chunk with identical register counts: when a kernel is "the same", measure it
  anyway, and prefer the measured body verbatim.
- **Case settings changed (bench.sh, 5869f86):** the 256k resume cases rank the tiers by the golden profile (the
  server's default), not the ranking qx-ds-check saves beside a state: decode256k 52.1 → **46.5**, prefill256k
  unchanged (6.80 = 6.80), same binary, 3 repeats each. prefill256k runs **8192-row chunks** (6.80 → 6.20; VRAM
  budget 13.1 GiB at 256k context); long16k keeps 4096 (8192 was 6% slower there). The server's chunk size at
  ctx ≥ 32k is still 2048: its own serve256k A/B (time to first token on a long prompt against serve256k3 decode)
  is the next settings question.
- **Also today:** the per-pass gate/up and hidden scratch landed (b14363b: bit-identical, time-neutral, 0.4-0.8
  GiB back to the tiers); a q8-only 16-token GEMV instance was rejected (exact, +4 to +7% per chunk, reverted);
  `bench.sh` exports `CUDA_CACHE_MAXSIZE=4294967296` because the driver's 1 GiB JIT cache evicted the engine's
  modules and a binary swap paid 20 minutes of ptxas (memory note `cuda-jit-cache`); the mechanics (CubeCL loads
  the whole module's PTX per kernel) and the per-entry PTX split idea are in the backlog.
- **Afternoon:** the server's default prompt chunk at ctx ≥ 32k is 4096 (42c23e7; time to first token on 13.5k
  tokens 47.7 → 43.8 s at 256k context with 3 slots, three-stream decode within spread; 8192 would be 37.3 s but
  costs serve256k3 3.9%, kept as `QX_DS_CHUNK=8192`). Decode pairs chunk9 against chunk17 in one session: 33.5/32.7
  against 32.8/32.9, prefill128 11.7 against 11.0: the split is level with the pre-groups kernel on decode and
  ahead on windows; the morning's "2% left" was the session. Decode numbers differ 2-3% between sessions: pair
  within a session. `~/qx-bins/chunk18` = the tree with the 4096 default (built after the pairs).
- **Next levers, in order:** (1) the dense slab passes (attention sharing a slab's keys; `qx_mgemv_t8z` at 40% of
  the chunk's GPU time wants a tiled exact kernel); (2) the chunk path beside decoders in the server (a prompt
  arriving while others decode still runs windows); (3) a chunk size from the VRAM headroom after the context and
  slots are placed, instead of a fixed default; (4) first-use JIT (per-entry PTX or shipped cubins).
- Operational: every chain script in the scratchpad installs binaries into `target/release` atomically and runs
  `bench.sh --no-build`; `~/qx-bins/chunk17` is the landed tree (0ced38e). Killing a chain by PID leaves its
  bench/engine child running: kill the child too. A chain renamed with `sed` kept bare binary numbers in a loop
  (`for arm in 16 12`), so one pair re-measured the old binary: check a copied chain's arms, not just its paths.

## Update 2026-11-07, 05:00: the chunk path landed in the server; the in-block GEMV; where it stands

- **Server:** prompt-only steps go through `forward_chunk` (2048-row chunks by default at `--ctx` >= 32k,
  `QX_DS_CHUNK` overrides, `0`/empty off). Time to first token on a 13.5k-token prompt: 191 s → **47 s**. Two
  rules learned: everything the chunk path can launch must be warmed before `seal()` (`eng.chunk_warm()` in the
  warm-up), and the chunk's remainder slab is cut into the warmed widths (never 5 or 7).
- **Harness:** `prefill256k` and `long16k` run `QX_DS_CHUNK=4096`: prefill256k 21.5 → **7.6 ms per token**;
  serve256k3 within spread; long16k's decode tail +5.7% (the 4096-row buffers off the VRAM tier). The set D run on
  the final tree (`~/qx-bins/chunk11`, label `setd-chunk`) is queued; promote it and record the row with the note
  (case change: chunks; projected score about 35 against 49.6).
- **Kernels since 0dad410:** the dense GEMV's token groups run inside one block (`info[5]` high bits; rows stay
  in L1; -10/-17% per chunk; a changed `mgemv.cu` costs ~9 minutes of driver JIT at first use on a box), the
  exact int8 tensor-core expert kernels (default), the remainder slabs. All bit-identical, ds-checks exact.
- **Measured truths for the chunk path (4096 rows, per row):** dense slab passes ~2.6 ms full / ~1.2 KV-only
  (GPU-bound: `qx_mgemv_t8` ~40% of GPU time, attention ~16%), experts ~1.7 full / ~0.85 KV-only
  (dequantization-bound; amortizes with the chunk), the chunk buffers 1.8 GB of VRAM. 16k prompt: 194 tok/s full
  stack, 249 KV-only; Strata 935 on the same box (chunk 8192, FP16 tiles, not bit-exact).
- **Next levers:** (1) the chunk buffers' VRAM (`gu`/`hq`/`hd` per pass; `out` per row) so 8192 fits; (2) the
  dense passes: attention sharing the window's keys per slab, the GEMV's L1-resident throughput; (3) the decode-tail
  tier ranking (decode256k 52.5 against 46.1 with the golden profile); (4) the chunk path beside decoders; (5) ship
  sm_120 cubins to kill the first-use JIT.
- Tooling notes: `QX_DS_BENCH=256` on a short file faults (unused); a server whose engine thread dies may linger
  holding its VRAM (kill -9 it; `gpu_free` in the chains checks memory, not processes).

## Update 2026-11-07, 03:00: the user said "make your own call"; the chunk path is the prompt path now

- Decision taken (and measured): stay exact. The expert kernels went to **int8 tensor cores while staying
  bit-identical** (`qx_moe_*_mma_*`: exact 32-wide integer dots, per-sub-block multipliers in int32, the float
  chain in block order; two masked mmas for the formats that scale per 16 elements). A tensor-core prefill that
  leaves the contract was not needed for the expert side; the dense side (`qx_mgemv_t8`, 40% of the chunk's GPU
  time) still wants a tiled exact kernel (weights decoded once per tile for all 128 tokens); 16/32-token
  instances of the current kernel do not compile in reasonable time (ptxas).
- Landed in order (all exact, all in the notebook): one-launch dense GEMV token groups (`blockIdx.z`; neutral in
  time, the slab passes are GPU-bound), the mma expert kernels, chunk size 4096 for the harness's long-prompt cases
  and 2048 by default in the server at >= 32k context, the server's chunked prompt path (prompt-only steps),
  the chunk kernels warmed before the server seals its kernel set. `prefill256k` 21.5 → **7.6 ms per token**;
  long16k's prompt 14.4 → 4.1; costs: the chunk buffers' VRAM (decode after a prompt +0.9% at 2048, +5.7% at 4096).
- In flight: the server TTFT test on a 13.5k-token prompt (windows: 191 s; the chunk arm reruns with the fixed
  binary), then a full set D `--check` run for the scoreboard row (expected about 35 against 49.6).
- Next levers, in order: (1) the per-selection chunk buffers (`gu`, `hq`, `hd`, `out`) as per-pass scratch so
  8192-row chunks fit without starving the tier; (2) a tiled exact dense GEMV for the slab passes; (3) attention
  sharing the window's keys across a slab's tokens (16% of the chunk's GPU time); (4) the decode-tail tier ranking
  (decode256k 52.5 against 46.1 with the golden profile); (5) the server's chunk path beside decoders.
- Strata head-to-head numbers stand (notebook 2026-11-07): their 16k prefill 935 tok/s at chunk 8192 against
  our 249 now (exact; 217 before tonight); decode even.

## Update 2026-11-07, 01:00: Strata measured here; stopped for a contract decision

- Strata ran on this box on our ids (notebook 2026-11-07 "Strata on this box"): decode 27.5 tok/s (ours 29.2
  warm), DSpark on 26.9 (no gain), prefill 1k 148 (ours 91), prefill 16k 250 / 696 / **935** tok/s at chunk
  1024 / 4096 / 8192 (ours **217**, exact). Set D baseline promoted (49.601). The chunk path is committed
  (04b3fda) with the records (93e2efe + this).
- **Open decision for the user:** keep prefill bit-identical to single steps (then the 4x gap closes slowly:
  f32 GEMM-class dense path, a much cheaper f32 expert kernel, graphs/fewer launches) or allow a tensor-core
  prefill path like Strata's (bf16/int8 tiles; `qx_tmv` exists; prompt exactness would become a KLD tolerance
  against the reference; decode stays exact). Both paths start with the same two items: the slab passes' launch
  count and the expert kernel's efficiency.
- Smaller items found tonight: the saved tier ranking is a worse decode-tail placement than the golden profile
  (decode256k 52.5 against 46.1; a decode-tail ranking or profile is a free ~12%); `QX_DS_BENCH=256` on a short
  file faults (status 700) — unused by the harness, check before use; the chunk path is ds-check only (the server
  still prefills in windows with the skip).

## Update 2026-11-06, night: the chunk path at 217 tok/s on 16k, Strata built for a head-to-head

- **Chunk path (family 12), measured at 16k, every arm bit-identical:** windows 17.9 → chunks of 2048 with the
  grouped expert kernels 7.0 ms per token → with the layer skip inside chunks (`QX_DS_SKIP=1` with `QX_DS_CHUNK`)
  **4.6 ms per token (217 tok/s)**. Rejected tonight: batched expert launches (slower on the GPU); neutral: the
  prefetch job (either tier first). What remains per 2048-row chunk: 5.8 s of host launch overhead for the slab
  passes (~70k CubeCL launches at ~80 us) and 6.5 s of grouped-kernel time (warp reductions). Next: a looped
  multi-token GEMV launch (16 → 1 per GEMV group per slab), per-slab graphs, a lane-per-row expert kernel; then
  the server. Knobs: `QX_DS_CHUNK=N`, `QX_DS_CHUNK_GROUPED=0`, `QX_DS_CHUNK_PREFETCH=0`, `QX_DS_CHUNK_BATCH=B`,
  `QX_DS_CHUNK_TRACE=1`. The working tree holds all of it, uncommitted until the queued ds-checks pass
  (`chain-dscheck2.sh` after the baseline; binaries `~/qx-bins/chunk5`). Three deadlocks were fixed on the way
  (notebook): never zero a word table the copy streams write (generation tags), warm every chunk kernel before
  the first job, and double-buffer the slot tables by layer parity.
- **Set D baseline:** running (`chain-baseline.sh`, label `baseline-setd`, committed binaries `~/qx-bins/chunk`
  = 642433a, the regenerated states with rings and tier ranking). Promote and record it (`perf_diff.py promote`,
  `score --record`), note the case-set change in the scoreboard.
- **Strata head-to-head (the user asked):** `~/git/Strata/build-dsv41/strata-dsv41` is built (sm_120). Its
  Engram reader rejects Unsloth's Q5_K table, so the supported recipe is Unsloth shards 1-5 + ISTA's IQ4_NL
  Engram (110 GB, downloading to `~/models/DeepSeek-V4.1-Flash-1017-GGUF/ista-engram/`). `chain-strata.sh` packs
  (`tools/dsv41/pack_gguf.py`, a map not a copy) and runs: decode 256 tokens after 64 prompt tokens (DSpark none
  and on), prefill 1k, prefill 16k at chunks 1024/4096/8192; `--ssd-cache 95` under `MemoryMax=110G`; the
  `prompt:` and `decode:` stderr lines carry tok/s. Our comparable numbers: decode 29 tok/s (1k), prefill 1k 91
  tok/s (windows; the chunk path at 1k not yet measured warm), 16k 217 tok/s (chunks + skip, cold tier).
  Caveats to state with the result: Strata's prefill is not bit-exact against its decode; Strata reads a 111 GB
  IQ4_NL Engram where we read the 135 GB Q5_K one; both engines on a cold start.

## Update 2026-11-06, evening: "bridge the gap" to Strata (prefill), and what the day found

Read `perf/backlog.md` families 9, 11, 12 and 6, `perf/findings.md` "The neighbours' mechanics", and the notebook's
2026-11-06 entries from "Nsight Compute" on. In short:

- **Decode at 256k:** `qx_ds_idx_score2` (queries in shared memory) is bit-identical to the old indexer scorer
  and measured decode256k **43.6 ms per token (3/3, spread 3.7%)** against 55-57 with the old kernel; its A/B
  twin (`QX_DS_IDX_KERNEL=1`, label `idx1`) was running when this was written. Default is the new kernel.
- **A real exactness bug, fixed in the tree:** `replay_to` rebuilt the rings from 128 tokens; each layer needs
  127 more positions than the one below, so every mid-stream resume/rewind (state cache, prefix reuse beyond 128
  tokens) was inexact and overwrote correct store rows. Now `Parked` carries the rings (`QXPK0002`, 20 MB per
  slot; an unpark at the parked position replays nothing) and a far rewind replays `replay_span()` = 5,208 tokens
  with the stores' rows saved and restored. Check queued: `QX_DS_REWIND=8192 QX_DS_PARK=8192` on route16k
  (`replay8k.log`). The perf states must be regenerated (queued, `chain-states2.sh`; also writes the tier ranking
  `-P.profile` beside each state, which `bench.sh` passes as `QX_DS_PROFILE`: the resumed cases had measured a
  cold tier, 35-44 ms per token against 22 steady). **No set D baseline is recorded yet**; take it after the
  states (`touch $S/baseline-go` releases the queued run) and only from a committed tree.
- **The long-prompt layer skip (family 11), in the tree, checks queued:** windows with `prompt_keep()` = 2,796
  prompt positions after them run layers 0-19 and layer 20's ring/store/key writes only (`forward_segments_opt(..,
  kv_only)`; ds-check `QX_DS_SKIP=1`; the server runs prompt-only windows KV-only, `QX_DS_SKIP=0` disables).
  Exact by construction for every kept position. Check: route16k tail-64 logits with and without must be
  identical (`skip-on-tail.bin` vs `skip-off-tail.bin`), and the prompt time both ways.
- **Chunked expert-major prefill (family 12), prototype M1 in the tree, built to `~/qx-bins/chunk`, checks queued
  (`chain-chunk.sh`):** `DsEngine::forward_chunk(slot, toks)` runs a prompt chunk layer-major: the window kernels
  over slabs of 128 rows on views of chunk-wide buffers (`ChunkBufs`, allocated at load by `QX_DS_CHUNK=N`), then
  the layer's experts over every row at once: the driver's `ChunkJob` streams each distinct expert once (VRAM
  residents' words at once, RAM records through the staging ring over the copy streams, NVMe records in batches
  of 48 through the landing places), the GPU's per-expert launches wait on the expert's slot word
  (`qx_moe_wait_word`), run the window kernels indexed by selection lists (`*_sel` variants), and mark progress
  (`qx_moe_signal`) for the ring's reuse. Eager launches, no graphs, no pool share. Expected to be bit-identical
  to windows (same kernels, same per-row math); the check compares the tail-64 logits after a 16k prompt in
  chunks of 1024 and 2048 against the window pass (`chunk1024-tail.bin` vs `skip-off-tail.bin`). If identical,
  the "chunk of N rows" lines in `chunk1024.log` give the time per row; the window pass is 11-14 ms per token.
  Not yet: the grouped kernel (reads each expert once per launch instead of per selection; needed past ~2K rows
  where HBM re-reads dominate), graphs per slab, the pool share, the server's use of it, VRAM borrowing for the
  chunk buffers (they come off the tier at load).
- **DSpark (family 6), measured facts, deprioritized:** the drafter GGUF is on the box
  (`~/models/DeepSeek-V4.1-Flash-1017-GGUF/dspark/`, 8 GB, `dflash`, experts in MXFP4). On this box decode is
  bandwidth-bound and a 6-row verify window reads ~2.3x the expert bytes of a step; at Strata's 31-32% acceptance
  on this quant class it does not pay. Measure acceptance before building anything.
- **Operational lessons of the day:** `/tmp` is tmpfs (RAM): build dirs there plus a concurrent build caused ten
  global OOM kills of bench runs; build to `~/qx-targets/<name>`, binaries to `~/qx-bins/<name>`, never build
  during a 100 GiB-tier run (ptxas alone peaks at 10 GB; `QX_SKIP_NVCC=1` skips it when the .cu files are
  unchanged or hand-checked with nvcc). `pkill -f` with a pattern that appears in the command text kills the
  shell (again). `sudo -E` drops the environment; ncu was abandoned.
- **Chains in flight (scratchpad `$S`, markers in `$S/chain-ab.out`):** `chain-ab2.sh` (idx A/B → skip checks →
  f10 A/B → `CHAIN-AB2-DONE`), `build-chunk.sh` (after `IDX-AB-DONE`), `chain-chunk.sh` (after both; `CHUNK-DONE`,
  then `touch states-go`), `chain-states2.sh` (states pass, `STATES2-DONE`; then waits for `baseline-go`).
  **Committed as 642433a (late evening)** after every check passed: the far rewind exact (256/256; the first
  fix, saving and restoring store rows, was not enough: the replay must not write the stores at all, so it runs
  in a rings-only, eager `replaying` mode), the skip identical at 16k (17.9 → 8.9 ms per token), the chunk path
  identical at 1k and 16k (8.4 / 7.6 ms per token at chunks of 1024 / 2048; three bugs on the way, see the
  notebook), ds-checks exact. In the working tree after that commit, untested: the prefetch job (the driver
  picks the layer's hottest non-resident experts and streams them during the slab passes; global ordinals and a
  persistent staging ring; `QX_DS_CHUNK_PREFETCH=0` disables) and KV-only chunks (`forward_chunk_opt(.., kv_only)`,
  ds-check `QX_DS_SKIP=1` with `QX_DS_CHUNK`). `chain-prefetch.sh` builds and tests them after the states pass and
  then releases the set D baseline (`baseline-go`), which runs on the committed binaries (`~/qx-bins/chunk`).

## Update 2026-11-06, late afternoon: the priority is now performance at 256k context

- **Uncommitted in the tree (an experiment in flight):** family 10, the per-step token/position/slot uploads
  through a host-mapped staging buffer and `qx_ds_copy_f32` instead of synchronous uploads that end in
  `cuCtxSynchronize` (`dsv41/mod.rs`, `in_host`; `QX_DS_SYNC_UPLOADS=1` restores the old path). Built into a
  separate target dir; its A/B (`f10-sync` against `f10-async` on `decode256k` and `serve256k`) is queued after
  the set D baseline. Land it only if the A/B clears 3% on either and `tools/ds-checks.sh` stays exact.

- The user reset the goal: maximize performance at 256k (`goal-prompt.md`, rewritten; read it first). Short-context
  numbers (set C baseline 34.621) become guards. The queued set C re-take was cancelled; the decay fix's win is
  recorded from its A/B (long16k 38.5 to 33.8).
- The tooling the goal asks for first: `qx-ds-check` saving a slot's state (`QX_DS_SAVE_STATE`, at `QX_DS_SAVE_AT`
  positions) and resuming from it (`QX_DS_LOAD_STATE`), so a 256k decode or late-prefill measurement takes minutes;
  then set D in `bench.sh` (`decode256k`, `prefill256k`, `serve256k` plus the guards) and its baseline. States live
  in `perf/states/` (git-ignored, about 1.3 GB each at 256k; regenerate with one full pass of
  `perf/data/wiki256k.tokens`).

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
- **Decision taken by the user (2026-11-06, late morning):** the server's default prompt chunk is 128 with the full
  staged share (`--window 128`, `QX_DS_STAGE_SHARE=8`, `QX_DS_STAGE_CAP=256` as defaults). The harness standard set
  is now set C (`decode`, `prefill128`, `long16k` in windows of 128, `serve3` at `--window 128`); committed as
  `b543bce`. The queue on the box at the end of the session: a forced 64k pass and two `--ctx 262144` server starts
  on the committed tree, then the set C baseline (`baseline-setc`), then the 256k patch (staged in the scratchpad
  as `ctx256k-patch.py` + `rewind-check-patch.py`: ratio-sized stores, replay-based rewind, decay per token)
  built in place with its bit-identity checks, `tools/ds-checks.sh`, 256k server starts and a forced 256k pass,
  then the decay A/B. The set C baseline landed: **34.621** (decode 34.0, prefill128 11.0, long16k 38.8, serve3 99.0),
  promoted, with `serve:window=128` in its `run.json`. If the tree has uncommitted engine changes, they are the 256k
  patch: verify them (`QX_DS_REWIND=64 QX_DS_PARK=64` on route1k 128 must print 64 / 64 twice; ds-checks exact)
  before committing.
- **Landed as `39c3bc0` (2026-11-06 afternoon):** the tier's score decay per token (long16k 38.5 to 33.8, the
  16k prefill 18.4 to 14.4; the long-context cost was this), the replay-based rewind (no host latent history; park
  and unpark keep tokens and store rows), stores sized by ratio, `qx-ds-check` `QX_DS_REWIND`/`QX_DS_PARK`, and
  regression line 5 (rewind and park, 64 / 64 twice) which the harness gate now requires. A full set C re-take
  (`baseline-setc-decay`) is queued after the forced 256k pass; promote it and record the row (expected about
  33.4) when it finishes clean.
- **256k context, facts so far (2026-11-06):** the model allows 1M with YaRN and the engine applies it; a
  `--ctx 262144` server starts and serves on the committed tree with 1 slot (14.5 GiB tier left) and with 3 slots
  (9.4 GiB); the compressed stores are 2.7 GB per slot at 256k (1.7 after the ratio sizing in the staged patch).
  The blockers are the host latent history (21.5 GB per slot; 5.7 GB measured at 64k) and the tier's residency
  lag after topic shifts (the forced 64k pass: 14 ms per token at 28k, 57 at 36k, decode tail 57 at 64k), which
  the per-step decay explains (notebook, "Forced 64k pass"). Both fixes are in the staged patch.
- **A kernel-file change invalidates the driver's PTX JIT cache:** the first run after rebuilding a `.cu` pays a
  JIT per module at load; the fused `mgemv.cu` took more than 12 minutes in `libnvidia-ptxjitcompiler` and looked
  like a hang (7 compiles printed, then nothing). Thread dumps tell the two apart. The fusion was reverted.
- **Where the loop stood before that decision:** the standard-setting levers are measured to their
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
