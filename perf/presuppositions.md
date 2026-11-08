# Presuppositions: the beliefs this engine's design rests on, and how each would be killed

Every row is a belief we act on. Each must be stated so that a measurement could prove it false, with the cheapest such
measurement named and the date it was last actually tested. A belief that has never been tested, or was tested before
the architecture changed, is a candidate for the next long shot, ranked by what we would gain if it turned out false.
The conjecturer's standing task is to pick from this table. Overturning a belief is a LANDED-class outcome: record it
in the notebook, update the row, and update `perf/findings.md`.

Status: **held** (tested, true as far as measured), **untested**, **overturned** (date), **partly** (true in one
regime only). Payoff: what changes if it is false.

| # | Belief | Would be killed by | Last tested | Status | Payoff if false |
|---|---|---|---|---|---|
| 1 | A 9.5 MB expert record cannot land in VRAM faster than ~0.7 ms. | A memcpy/gather bench of one record (`kernels/bench/pcie_bench.cu`). | 2026-11-08 | **overturned**: 0.19 ms, the bus ceiling | The DMA tier's cost model was 3.5x too pessimistic; the staging share and the planner change. |
| 2 | The CPU pool is memory-bandwidth-bound at ~0.37 ms per record. | One record on 1/4/8/16 cores (`qx-poolbw`); the pool in the engine with DMA off. | 2026-11-08 | **overturned**: 0.23 alone, 0.33 in the engine; 0.16 ideal; the 1.45x was thread placement | Placement fix -7%; the barrier structure is the next 30%. |
| 3 | Spinning threads lower the pool cores' clocks (power budget). | `/proc/cpuinfo` MHz sampled during generation vs the pool bench. | 2026-11-08 | **held** (5.3 GHz generating, 5.08 in the bench) | None; MWAIT work is closed here. |
| 4 | The pool must synchronize at stage barriers (gate/up, act, down). | A dataflow unit queue per record with per-job counters (Big-MoE-Small-Box `ft_n135.h`); or two records in flight overlapping phases. | never | **untested** | Up to 30% of the pool window (0.23 -> ~0.17 ms per record), ~4-6 ms per 256k token. |
| 5 | A token's experts are unknowable before its router runs, so nothing can be prefetched. | Offline: precision of layer l+1's router on layer l's residual, and of the previous token's set, from dumped traces. | 2026-11-07 (previous token: 13-23%) | **partly**: previous-token signal weak; layer-ahead never measured | A usable predictor hides the pool/DMA latency behind the previous layer: up to the whole 28 ms host wait. |
| 6 | An expert must be fetched and served as a whole record (gate, up, down together). | Serving gate/up as soon as their pieces land, down later (piece streaming into the pool). | never | **untested** | NVMe fetch waits (4.8 ms per 256k token) shrink; the pool starts earlier on NVMe records. |
| 7 | Exactness across GPU/CPU/chunk requires the dense GEMVs to stay in f32 (the "f32 path"). | An int8 or bf16 dense GEMV with a fixed reduction order that is bit-identical to a redefined CPU reference, under the ds-checks. | 2026-11-07 (int8 planar Q8 path exists for some segments; bf16 tmv differs by design) | **partly** | The dense passes are 42% of a chunk's GPU time; a tensor-core exact form is the Strata-class prefill gap. |
| 8 | The latent rows of the compressed KV store must live in VRAM. | 512 rows per layer per token gathered from pinned host memory at measured PCIe rate; decode256k with the freed ~8 GiB in the expert tier. | never (gather rate measured at 7 GB/s through the GEMV pattern, 2026-11-04) | **untested** in this form | ~8 GiB more experts in VRAM at 256k: fewer host-served records (-7 ms of waits predicted vs +3 of gathers). |
| 9 | Decode must produce one token per step. | DSpark-style drafting with a measured acceptance on our prompts (family 6). | 2026-11-06 (re-costed, not measured) | **untested** | At 75% acceptance (their number) 1.5-2x on decode; at <45% nothing. |
| 10 | The chunk path is launch-bound; fewer launches is the lever. | nsys of a chunk after the raw launcher: host enqueue share. | 2026-11-08 (raw launcher: 18.6 -> 15.7 s) | **partly**: still 8 s of 15.7 host-side per 4096 rows | Per-slab graphs or fused slab kernels: another 30-50% of the chunk. |
| 11 | The bus (PCIe) is the floor for host-served experts; compression in flight cannot help. | Records are 2-bit already; test: entropy-code the IQ2 blocks and decode on the GPU, measure bytes and time. | never | **untested** (likely true) | If records compress 20%, staged copies and NVMe reads shrink by that. |
| 12 | The best decode staging share is 3/8 regardless of context, placement or pool speed. | The share sweep after each change to the pool or the bus. | 2026-11-08 (flat 3/8..8/8 under the words path, before the placement fix) | **held**, re-test due after placement | A few percent of decode256k. |
| 13 | Results must not depend on scheduling (windows bit-identical to single steps). | This is the user's contract, not a belief; listed so nobody "tests" it away. | always | **contract** | n/a |
| 14 | The GPU is idle while it waits for the host (28 ms per 256k token) and nothing useful can fill that time. | Run the next layer's attention/dense work speculatively, or the next token's draft, during the wait. | never | **untested** | Up to the whole wait, if any work is independent of the pending experts. |
| 15 | The expert tier's ranking (golden profile) is the right residency policy at 256k. | Per-request hot sets; a ranking from the live run (`--profile-out`) vs the golden one (10.7% worse at 256k decode, 2026-11-07). | 2026-11-07 | **held** for now | Residency is the biggest lever on host-served records. |
| 16 | NVFP4/FP8/int8 "require hardware support" (the QuixiCore lesson): any format's math can be implemented exactly in software on any device. | Not a belief of this engine; recorded as the standing reminder that vendor claims are presuppositions. | n/a | **reminder** | n/a |

## How to use this file
- Before each ideation pass, read the table. Pick the untested or stale belief with the largest payoff and the cheapest
  killing test. Measurements that only confirm a belief are still outcomes: date the row.
- When an experiment's result contradicts a row, the row changes first, then the notebook and findings.
- Add a row whenever a design decision is justified by "everyone knows" or "the docs say"; those are the rows most
  likely to be wrong.
