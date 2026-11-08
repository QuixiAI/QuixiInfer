# Conjecturer pass, 2026-11-08 (set D baseline 30.047; placement fix at decode256k 42.0 pending)

## 0. The facts I started from (read in the brief's order; numbers I derived myself are marked)

Post-placement decode256k at 42.0 ms (steady deltas over the last 8 tokens of `/home/eric/git/QuixiInfer/perf/results/2026-11-08/affin3-core15-162505/logs/decode256k-1.log`, derived): 240 selections per token = VRAM 179, staged 20.6, pool-RAM 34.6, NVMe 5.4 (61 host-served). Pool window 16.0 ms/token = NVMe fetch wait 5.1 + compute 10.9 over 40 records = **0.27 ms per record** (the bench's 0.23 is now close). Stage wait 2.5 ms/token. Serve call 467 us x 40 layers = 18.7 ms; "gap" 478 us x 40 = 19.1 ms; the remaining ~4 ms is between tokens (head, logits, sampling, and the driver's `adapt` at 1.75 ms/token). Promotions: 22 VRAM + 8 RAM per token, i.e. 4.2 ms of bus per token on top of the staged copies.

serve256k3 (baseline `serve256k-1.server.log`, steps 1200-1400, derived, prompt steps roughly subtracted): per 112-120 ms step about 450-530 staged, 6.6 pool, 42 NVMe, 456 VRAM (VRAM hit rate ~44% against 75% in decode256k); `pool_ns` 66 ms/step of which fetch 39; stage wait 56 ms/step; device wait is the whole step. DRAM bytes per step: 450-530 x 9.5 MB staged + 42 x 9.5 x 2 (NVMe write + pool read) + promotions = 5.2-6.0 GB in 112 ms = **46-54 GB/s against the box's 52-59 GB/s ceiling**. So serve256k3 is within 10-20% of DRAM-bound; bytes and bus bubbles are the only levers there, latency tricks are not.

Trace simulation I ran (script at `/tmp/claude-1000/-home-eric-git-QuixiInfer/7b27a076-c316-432c-a7a6-fc085686af11/scratchpad/sim.py`, on `/home/eric/git/QuixiInfer/golden/data/route-dsv41.trace`, 69k tokens, k=6, golden ranking from `golden/profile-dsv41.txt`): static golden top-N hit rate 1270: 41.3%, 1609: 46.2%, 2000: 51.1%, 2500: 56.6%, 3000: 61.6% (slope ~1.2 points per 100 records; the engine's 75% at 1609 is adaptation on in-distribution wikitext, so use the slope, not the level). Host-served records: previous-token repeat 20.5% (engine counter says 22.7%); **an LRU of 256 records over host-served selections hits 25.5%, 512: 38.2%, 1024: 52.5%**. Three interleaved streams at VRAM 1270: LRU 256 thrashes (2%), 512: 24%, 1024: 37%. NVMe share at RAM+VRAM 11714: 1.86%; +1000 RAM records: 0.99%.

Two facts from the neighbours that matter: colibri measured **71.6% recall for layer L+1's router applied to layer L's post-attention state** (GLM-5.2; 41.3% for previous-token) in `~/git/colibri/docs/tuning.md` "Router-lookahead prefetch"; BMSB's GLM figure is ~30% useful. Our row 5 has never measured the layer-ahead case. The DIMMs are two Crucial Pro DDR5-5600 at their rated speed (dmidecode), so there is no cheap BIOS headroom on the DRAM ceiling.

## 1. Twenty ideas, judgment deferred (marked after reading findings/backlog: R rejected, Q queued, N new)

1. **Row 5 false (next layer's routing is predictable):** run layer l's router on `hc_pre(R)` before layer l's attention, prefetch the predicted host-served records into the staging area during attention + dense (DRAM and bus are idle in that 0.48 ms per layer). Exact: the real routing still decides. [Q, family 15 item 3, never measured; colibri's 71.6% is the prior.]
2. **Row 14 false (serve slots need not move in lock step):** run the 3 slots as staggered per-layer streams so slot B's attention runs while slot A's experts are served; fills the bus bubble during GPU phases. Exact by the window-width contract. [N]
3. **Row 15 false (frequency ranking is not the right marginal policy):** make the 256-record staging area an LRU cache of records already staged (lookup before copy; no wasted copies). [N; distinct from the rejected previous-token *prefetch*, which wasted 77-86% of copies.]
4. **Row 8 false (latent rows need not live in VRAM):** gather the 512 + 128 rows per layer per token from pinned host; +~860 records in VRAM. [Q, family 9(c).]
5. **Row 6 false (records need not be whole):** two io_uring reads per NVMe record (gate/up 6.5 MB, then down 3 MB); the pool's phase 1 starts on the first. [Q, family 15 item 12.]
6. **Row 9 false (one token per step):** DSpark verify windows; at 256k the fixed GPU cost per step (~19 ms) is larger than at 1k, so break-even acceptance at width 2 is ~31%, not 41-46%. [Q, family 6/13 deferred; measurement via Strata's local binary needs no engine code.]
7. **Row 4 false in the other direction (barriers are not the 30%):** the 1-core number (2.55 ms per record = 3.7 GB/s per core) says the pool is compute-bound per core and 16x ideal is 0.16; the DRAM floor at 50 GB/s is 0.19. The payoff is 0.23 -> 0.19, not 0.17. [Q, family 15 item 9; re-rate.]
8. **Row 11 false (records compress):** entropy-code the IQ2_XXS codebook indices, decode in the staged expert kernels; the RAM tier stores compressed bytes for the DMA lane. [N; serve is byte-bound.]
9. **Row 12 stale (3/8 is the optimum):** with an LRU staging cache the staged lane becomes cheaper on average; re-sweep the decode share after idea 3. [Q, row says re-test due.]
10. **Row 10 false for 256k (the chunk is not launch-bound but wait-bound):** `prefill256k-1.log` shows fetch 4.7 s + stage wait 7.9 s of 82 s over the two 8192 chunks; in the chunk path every layer's expert set is known in advance, so NVMe reads and staged copies for layer l+1 can be issued during layer l deterministically. [R at 16k/4096 ("prefetch neutral, not transfer-bound"); the re-open condition is whether the GPU idles during those 12.6 s at 256k, unmeasured.]
11. **Row 7 false (exactness does not need f32 dense):** int8 IMMA tensor-core tiles compute exact int32 dot products; with the float scale chain in a fixed order (as the exact int8 expert tiles already do), the planar-Q8 dense segments of the chunk path can run on tensor cores bit-identically to a redefined CPU reference. [N-ish; family 14 only lists the lossy bf16 tile.]
12. **Row 2/3 residue (the engine's 0.27 vs bench 0.23):** the driver thread is pool worker 0 and does mailbox bookkeeping, promotion issue and victim scans between its pool shares; take it out of the pool (15 workers + a driver on the last core's sibling). [N, small.]
13. **Hidden belief: "the `gap` counter is GPU time."** `tl_gap` starts at `serve()` entry; the driver's `adapt` (1.75 ms/token) and `promote_one` victim scans can hold the job after the GPU posted it. A GPU `globaltimer` stamp at post time splits it. [Q, family 15 item 13 — pure measurement, never done.]
14. **Hidden belief: "promotions are free."** 22 VRAM promos per token = 4.2 ms of bus and DRAM reads per token that overlap the pool's DRAM reads (the 7% DMA contention measurement was for staging alone). Gate promotions into the GPU-phase windows where DRAM is idle. [Q, family 15 item 14.]
15. **Batched DMA submission** for the 13 staged copies per layer in serve. [R elsewhere: Strata measured no tok/s gain; our issue cost is 7 us per call.]
16. **Databases (2Q/ARC buffer pools):** the ranked tier as the frequency queue, the staging area as the probation/recency queue, admission to the ranked tier only from the recency queue. Idea 3's general form; simulate in `tools/cache_sim3.py` against the current LFU-with-decay before building. [N]
17. **Networking (cut-through vs store-and-forward):** per-piece ready words so a staged expert kernel starts on landed rows instead of after the whole 9.5 MB; saves up to one record time (0.19 ms) per layer where the copies are the pole (serve). [N]
18. **HPC (makespan planner, DMA vs pool split per layer):** [Q, family 15 item 11 — see rejection below: both lanes share the DRAM ceiling.]
19. **Hardware off-label (NVMe to GPU peer-to-peer DMA):** if the open kernel modules allow P2PDMA into the 5090's BAR1 on this kernel (7.0), NVMe records would cost zero DRAM passes (today two) and the drive's 10-14 GB/s becomes bandwidth additive to DRAM's 55. One-hour check (`cuFile` driver mode, `/sys/bus/pci/.../p2pmem`), very likely "no" on GeForce. [N]
20. **Between-token tail (~4 ms = 10% of decode256k):** start the next token's layer 0 before `adapt` finishes (move `adapt` to the GPU-phase window or to a helper thread on core 15's sibling). [N; depends on idea 13's measurement.]

## 2. Ranked experiments

### E1. Pure measurement: layer-ahead router precision (attacks row 5, "partly")
- **Hypothesis:** applying layer l's router to the residual entering layer l (before its attention) recalls more than 60% of the host-served experts at 256k, as it does for GLM in colibri (71.6%).
- **Mechanism:** a knob in the step that launches `qx_ds_route` on `hc_pre(R)` at the top of each layer and writes the top-6 into a spare mailbox region; `serve()` scores it against `all`/`miss` exactly as `spec_hits`/`spec_miss_hits` do today (`/home/eric/git/QuixiInfer/crates/qx-engine/src/dsv41/driver.rs:444-465`). Also score top-8 and top-12 (precision/recall trade for the bus budget: 0.48 ms of idle bus per layer = 2.5 records; host-served per layer is 1.5).
- **Counter:** two new cumulative counters on the stats line (`pred_hits/pred_total`, `pred_miss_hits/pred_miss_total`), decode256k and decode.
- **Expected gain if it holds:** the host window at decode256k is 16.0 ms; predicted hits become staged records already landed, so the per-layer wait becomes the unpredicted remainder: at recall r the saving is ~r x (16.0 - 2 ms of irreducible latency); r = 0.6 gives -8.4 ms (42.0 -> 33.6, -20%); r = 0.7 gives -10 ms (-24%). Serve256k3: zero or negative (DRAM-bound; wrong predictions add bytes), so the prefetch would be a width-1 mechanism. The NVMe fetch waits (5.1 ms/token) are covered by the same prefetch.
- **Exactness:** residency only; the routed selection is unchanged.
- **Cost:** 3-4 h, 2 GPU runs (10 min each). Build nothing until the counter says r >= 0.5.
- **What in findings would have to be wrong:** nothing; findings record only the previous-token case (13-23%). Family 2's sentence "the next layer's routing is unknown, so no prefetch lever" is the presupposition under test.
- **Row 5 status after:** "overturned (layer-ahead)" if r >= 0.5, payoff column "decode256k -8 to -10 ms"; "held" if r < 0.3, payoff "none; serve needs bytes".

### E2. The staging area as an LRU cache of staged records (attacks row 15 "held", re-opens row 12)
- **Hypothesis:** at least the 22.7% of host-served selections that repeat the previous token (the `spec_miss` counter) are already in a staging slot and need no copy; the trace says an LRU of 256 records catches 25.5% single-stream.
- **Mechanism:** in `serve()` (driver.rs:472-531) keep a map record -> staging slot across jobs; a staged id found in the map writes its slot address without a copy and refreshes recency; eviction is LRU over the 256 slots (the area's reuse-per-job assumption goes away: a slot is reused only after the GPU's `MB_DONE` of the job that last read it, which the sequence number already orders). The pool-served records that hit the cache should also be redirected to the staged lane (`ids` -> `sids`), which the post kernel's share logic has to allow; simplest first step: cache only the staged lane.
- **Counter:** split `staged_hits` into `staged_copied` and `staged_cached`; the `spec_miss` ratio is the lower bound to beat.
- **Expected gain:** decode256k: 61 host-served x 0.25 (cache hit rate) = 15 records x ~0.25 ms = -3.8 ms (-9%) if the pool lane is redirected, -1.3 ms if only the staged lane (20.6 x 0.25 x 0.25). Serve256k3: ~0 at 256 slots (thrash in the 3-stream sim); with the flex tail lending the area 512-1024 slots during decode-only phases, 24-37% of 450-530 staged records = 110-190 x 0.19 ms = -20 to -36 ms of bus per step (-18 to -30%) — this is the one byte-reducing lever besides VRAM capacity, and it beats giving the same VRAM to the ranked tier (3-stream sim: LRU 1024 catches 37% vs the ranked tier's +1024 records catching 23%).
- **Exactness:** the same bytes through the same kernels.
- **Cost:** 6-8 h, 3 GPU runs (decode256k, decode, serve256k3); then re-sweep `QX_DS_STAGE_SHARE_DECODE` (row 12).
- **What in findings would have to be wrong:** "the host-served tail has little temporal locality" (2026-11-07) — it was measured for one-token repeat only; the LRU-256 sim already contradicts the generalisation.
- **Row 15 after:** "partly" (frequency for the core, recency for the margin), payoff "decode -9%, serve up to -20% with a lent area".

### E3. Pure measurement: per-layer timeline with GPU stamps (attacks hidden belief 13 and feeds E1/E5)
- **Hypothesis:** part of the 478 us "gap" per layer is driver-side (adapt 1.75 ms/token, promotion victim scans) and in serve the bus idles 10-20% of the step during GPU phases.
- **Mechanism:** `qx_moe_post` writes `%globaltimer` into the mailbox at post; the driver stamps `CLOCK_REALTIME` at serve entry and at `MB_DONE`; a once-per-run offset calibration (write a host stamp, read a device stamp). Print per layer: post -> serve entry (driver latency), serve, done -> next post (true GPU time); plus the staged-copy end stamps from the ready words on the stage streams. Also a decode-only interval of the tier counters in the server (the `req:` lines mix prompt and decode).
- **Counter:** the new split itself; the serve-side byte count over a pure-decode interval (settles whether serve256k3 is at 46 or 54 GB/s).
- **Expected gain:** none directly; it decides E1's bus budget, E5's ceiling (bubble fraction), and idea 20 (if driver latency at layer 0 is > 1 ms per token, deferring `adapt` is a -2 to -4% win for free).
- **Cost:** 1 day, 2 GPU runs. BMSB family 15 item 13, never done; the brief's facts (558 us call, 7 us read+issue) do not cover the post-to-entry latency.
- **New presupposition rows to add:** "the `gap` counter is GPU time" (untested) and "serve256k3 is DRAM-bandwidth-bound" (derived, untested).

### E4. Long shot, structural: compressed-KV latent rows in pinned host memory (row 8, untested)
- **Hypothesis:** the ~4.6 GiB per 256k slot that cuts the expert tier (20.0 -> 15.4 GiB at 1 slot, 12.2 at 3) is mostly latent rows read sparsely (512 selected + 128 window per layer per token), so moving them out frees ~3-4 GiB per slot for experts.
- **Mechanism:** first the 1-hour measurement: itemise the slot's VRAM (latent stores of the 4 source layers, indexer keys of the 8 index layers, scores, masks, rings) from the allocation code; only the latent stores can leave (the indexer must scan all keys). Then a gather kernel over mapped pinned memory (int4 loads, as `pcie_bench.cu` showed 48 GB/s for contiguous records; the random-row rate is the unknown) feeding `attn_part` unchanged.
- **Counter:** `vram N` in the tiers line (records in VRAM) and the `vram/staged/ram/nvme` shares; the gather's time via the step's `device wait`.
- **Expected gain (trace slope 1.2 points per 100 records):** decode256k +400 records (1 slot, ~3.7 GiB) -> host-served -9% = -5.5 records x 0.27 = -1.5 ms, minus 40 x 0.03 ms of gathers = ~-0.3 ms net (-1%, probably noise). Serve256k3 +1100 records (3 slots) -> host-served -25% at VRAM 1270 -> bytes -25% -> -25 ms per step (-22%) minus 3 x 40 x 0.03 = -3.6 ms of gathers. So this is a serve lever, not a decode lever; the row's "-7 ms of waits" for decode256k does not survive the slope.
- **Exactness:** same rows, same kernels.
- **Cost:** 1 h to measure the slot's VRAM; 2-3 days to build; 3 GPU runs.
- **What would have to be wrong:** the 2026-11-04 "mapped reads 7 GB/s through the GEMV pattern" must not apply to a 1 KB-row gather (the record-level gather at 48 GB/s says the pattern, not the bus, was the problem).
- **Row 8 after:** payoff column "serve256k3 -20%, decode256k ~0".

### E5. Long shot, structural: staggered slot streams in the server (row 14, untested)
- **Hypothesis:** with 3 slots, the GPU phases of one slot can hide the host/bus phases of another; the ceiling is the bus-bubble fraction E3 measures (<= 20% of the step).
- **Mechanism:** the scheduler issues the 3 slots as 3 width-1 steps interleaved layer by layer on one stream (slot A layer l, slot B layer l, slot C layer l, ...), with three mailbox lanes so the driver serves A's job while B's attention runs. Dense weights are read 3x (3 x 6.5 ms, hidden under the 112 ms if the bus is the pole).
- **Counter:** the bus bubble (E3) before; `sched: forward` per step after.
- **Expected gain:** up to the bubble: 10-20% of serve256k3 if E3 shows it; 0 if serve is already at 54 GB/s. Decode256k: none.
- **Exactness:** by the existing contract (any width is bit-identical to single steps).
- **Cost:** 2-3 days; do only after E3.
- **Row 14 after:** "partly" (fillable in serve only), payoff = measured bubble.

### E6. NVMe fetch latency on the width-1 path (row 6, untested)
- **Hypothesis:** the 5.1 ms per token of exposed fetch waits (5.4 records at ~0.9 ms each) halves with two reads per record (gate/up first) and/or 8 parallel SQEs per record.
- **Mechanism:** a `fio`-free check with the engine's own io_uring path or a 30-line Rust bench: one 9.5 MB O_DIRECT read vs 8 x 1.2 MB vs 6.5 + 3 MB; then `fetch_submit` split, and `run_multi` for the NVMe list in two phases (phase 1 after the first CQE, phase 3 after the second).
- **Counter:** `fetch` ms in the pool line.
- **Expected gain:** decode256k -1.5 to -2.5 ms (-4 to -6%). Serve256k3: 0 (the 39 ms/step of fetch overlaps the staged copies; DRAM is the pole). Superseded by E1 if the predictor works (prefetch hides the fetch entirely).
- **Exactness:** unchanged bytes and order.
- **Cost:** 1 h measurement; 4-6 h build; 2 GPU runs.
- **What would have to be wrong:** family 2's "NVMe reads are already at the drive's rate per record" (a single 9.5 MB request's latency is what is at issue, not throughput).

### E7. Pure measurement: entropy of IQ2_XXS records (row 11, untested, likely true)
- **Hypothesis:** the codebook indices and sign/scale bytes of a record have less than 7.5 bits of entropy per byte, so an exact entropy coder would cut 10-15% of the bytes that bound serve256k3.
- **Mechanism:** a Python pass over 20 records from the pack (byte histograms per field; order-0 and order-1 entropy). If < 10%: close the row as held. If > 15%: cost a GPU-side rANS decode in the staged kernel (the pool lane would keep raw records; the RAM tier would hold both only for the hottest, so it is a knob).
- **Cost:** 10 minutes offline, 0 GPU runs.
- **Row 11 after:** "held" or "untested with a number".

### E8. Pure measurement: DSpark acceptance on this box with Strata's binary (row 9, untested)
- **Hypothesis:** per-draft acceptance on long wikitext-class text is above 31% (Strata's own) and the 256k cost structure (fixed GPU ~19 ms per step, host ~17 ms per token, width-2 step ~55 ms) makes width-2 verification pay above ~31% acceptance (ms per token = 55 / (1 + a) < 42).
- **Mechanism:** the locally built `strata-dsv41` (memory note) with its DSpark on and its acceptance stats, over the wiki256k text's tail; no engine code.
- **Expected gain if a = 0.6:** 55 / 1.6 = 34 ms per token (-19%) at decode256k; serve256k3 0 or negative (bytes). The findings' 41-46% break-even was derived at 1k where the fixed GPU share is smaller; at 256k the break-even is lower.
- **Cost:** 2-3 h, GPU busy for ~1 h; the build itself remains weeks (family 6), which is why the measurement must come first.
- **Row 9 after:** payoff column from "1.5-2x at 75%" to the measured curve.

Not ranked but noted: the exact IMMA dense tiles for the chunk's planar-Q8 segments (idea 11) are the exactness-preserving version of family 14's "lossy bf16 tile" and the only route to the dense passes' 42% of chunk GPU time; a multi-day kernel project that should wait for the author's current `mgemv_zt` thread to conclude. Family 8 (dense GEMV fusion at width 1, 2-4%) stays deferred.

## 3. Presupposition rows: what changes and how

| row | measurement that changes its status | payoff column becomes |
|---|---|---|
| 4 | already answered by `qx-poolbw` 1-core = 2.55 ms (16x ideal 0.16) and the DRAM floor 0.19 at 50 GB/s | "0.23 -> 0.19 at best: ~1.6 ms per 256k token (4%)", not 30% |
| 5 | E1's `pred_miss_hits/total` at 256k | r x (16 - 2) ms on decode256k; nothing on serve |
| 6 | E6's read-latency bench and `fetch` counter | -1.5 to -2.5 ms decode256k; 0 serve |
| 8 | E4's VRAM itemisation and the trace slope | "serve256k3 -20%, decode256k ~0" (the "-7 ms" does not survive the slope) |
| 9 | E8's acceptance curve | from the 256k break-even of ~31% |
| 10 | E3's chunk variant (GPU idle during 12.6 s of host waits per 82 s) | if idle: deterministic layer-ahead NVMe/stage prefetch for chunks, up to -15% prefill256k |
| 11 | E7's entropy | bytes fraction x serve256k3 |
| 12 | re-sweep after E2 | a few percent, as written |
| 14 | E3's bus-bubble fraction, then E5 | the bubble (<= 20% of serve256k3) |
| 15 | E2's `staged_cached` counter | "partly: frequency core, recency margin" |
| new | "the `gap` counter is GPU time" (E3) | up to the driver latency at layer 0 (idea 20) |
| new | "serve256k3 is DRAM-bound; bytes are the only lever" (E3's decode-only byte count) | decides E2/E4 vs E5 |

## 4. Backlog items to reject or re-rank, with the fact

- **Family 15 item 2** ("cut the per-record driver overhead to the 0.2 ms the bus takes", worth "~45 ms"): retire. The ready-words test showed the staged copies are not on the decode critical path, and in serve the copies run at bus rate (450-530 x 0.19 = 85-100 of 112 ms); there is no 0.5 ms per record to cut.
- **Family 15 item 9** (dataflow pool, "0.23 -> 0.17, 4-6 ms per token"): downgrade to <= 1.6 ms per token. The floor is DRAM (0.19 at 50 GB/s), and the in-engine gap to the bench is now 0.27 vs 0.23 after the placement fix.
- **Family 15 item 11** (per-layer DMA/pool makespan planner): reject for the scored cases. The share sweep is flat at 256k (3/8..8/8 within 2%) and both lanes draw on the same DRAM ceiling, so re-splitting cannot add bandwidth.
- **Family 15 item 10** (MWAITX): already closed by the 5.3 GHz clock measurement; remove.
- **Family 2's** "the next layer's routing is unknown, so no prefetch lever": not a fact, a presupposition (row 5); E1 is its test and colibri's 71.6% is a measured counter-example on a neighbouring model.
- **Rejected 2026-11-04** "GPU reads of RAM-resident experts in place at 7 GB/s": the bus fact no longer holds (`pcie_bench.cu`: a gather kernel lands records at 48 GB/s), but re-opening buys nothing for whole records since DMA staging already runs at bus rate; it matters only for E4's row gather, whose random 1 KB pattern is the open question.
- **Family 9(c)'s** "-7 ms of waits" for KV-to-host at decode256k: not supported by the trace slope (+400 records = -9% host-served = -1.5 ms); the lever is serve256k3 (+1100 records, -20%).
- **Family 6/13** (DSpark): re-rank the measurement up (E8: Strata's binary gives the acceptance for free); the build remains gated on it.

Files read: `/home/eric/git/QuixiInfer/perf/presuppositions.md`, `perf/scoreboard.md`, `perf/optimization_status.md` (tail), `perf/findings.md`, `perf/backlog.md`, `perf/results/2026-11-08/setd-flex-145532/logs/*` and `affin3-core15-162505/logs/*`, `crates/qx-engine/src/dsv41/driver.rs`, `crates/qx-engine/src/dsv41/mod.rs` (2380-2440), `crates/qx-cpu/src/pool.rs` (100-215), `crates/qx-engine/src/tiers.rs` (280-345), `tools/cache_sim3.py`, `~/git/colibri/docs/tuning.md`, `~/git/Strata/docs/DETAILS.md`.
