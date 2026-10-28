# Why QuixiInfer was slower than Strata, and what was done about it (2026-10-26)

Same model (nvidia/Qwen3.8-Flash-Next-NVFP4), same class of machine. Strata's published figures for it are on an
RTX 5090 with a 9950X3D and 128 GB of DDR5-5600 (`../Strata/docs/NVFP4.md`): 114-116 tok/s on a 400-token answer
at 64k context, 3,900 tok/s on a 32k prompt. Strata could not be run here: its NVFP4 path needs its own converter
output, made from the safetensors that were deleted for disk space.

## How the time was measured

- Decode: `quixiinfer run` (a single request through the server's scheduler, speculative windows), profiled with
  Nsight Systems with per-node CUDA-graph tracing, broken down per verify window; the scheduler's per-200-window
  statistics (hit rate, misses, CPU pool time); `qx-batch-bench` for 1-16 sequences.
- RAM: `qx-membw`, the CPU pool's throughput on 6.6 GB of NVFP4 records with and without a concurrent copy stream.
- Dense matvec: `qx-kbench` on the engine's planar weights at 1-8 tokens.

## What a verify window cost (one 64k slot, ~3 tokens per window, 81 tok/s, before)

| part | ms per window |
|---|---|
| GPU waiting for the CPU's missed experts (`qx_moe_wait`) | 20-23 |
| dense matvecs (4.45 GiB of Q8_0 per window) | 10.8 |
| resident experts on the GPU (overlapped with the CPU) | 4.6 |
| everything else (routing, mixing, attention, GDN, sampling) | ~5 |
| **window** | **~41** |

## Root causes, in order of weight

1. **The CPU computes every missed expert, and the RAM is the limit.** The pool reads 2.64 MiB per missed
   expert at 53-63 GB/s, the RAM's bandwidth: 44-52 us per miss. A copy stream beside it does not add
   bandwidth, it splits it (pool 54 -> 29 GB/s, copies 28; 57 GB/s together), so Strata's PCIe share of the
   misses can be worth at most ~7% on this machine (Strata measured +4.5% for 0.25 against 0.10 on NVFP4). Time
   per window is set by how many experts miss, so by **how many experts fit in VRAM** and **how fast the cache
   adapts to the conversation**.
2. **VRAM spent on buffers that sit idle during decode.** The prompt buffers (2.1-2.7 GiB) were allocated beside
   the expert cache. Strata and Basalt lend them from the cache and give them back after the prompt. One 64k slot
   held 7,551 experts here against Strata's ~8,350.
3. **The production layout reserves 8 x 262k.** Per-slot state (the indexer's pooled keys are 402 MB per slot at
   262k and stay FP32, as Strata keeps them for selection exactness; KV rings; DeltaNet state) costs ~4 GiB of
   expert cache against one 64k session. This is a choice, not a defect: Strata's figures are for one session.
4. **The dense matvec did not scale with window width.** 1.6-2.5 TB/s at 1 token, 0.5-0.9 TB/s at 5-8 tokens: each
   token's input was read as f32 for every weight row.
5. **The cache adapted slowly.** A swapped-in expert waited out a 2-window quarantine and then had to be missed
   again; 8 swaps per token. Hit rate over a 2,000-token answer climbed 74% -> 87% -> 93% (Strata swaps up to 96
   experts every 4 rounds and admits on copy).

## What was integrated

| change | from | effect |
|---|---|---|
| Prompt buffers lent from the expert cache's tail (`prefill_lend`/`prefill_release`; the swapper evicts the tail's experts and copies them back after the prompt) | Strata, Basalt | one 64k slot 7,551 -> 8,360 experts; 8 x 262k 5,338 -> 6,371; bit-identical results |
| int8 activations for planar Q8_0 matvecs (`qx_mgemv_q8x` + dp4a) | llama.cpp / Strata decode precision | dense 5-token groups ~2x faster (e.g. 51.9 -> 25.4 us); engine vs reference KLD 6e-4 -> 1.1e-3 |
| 32 swaps per token (from 8) | Strata | single conversation +6% on 400-token answers; neutral under mixed load |
| Swaps admitted as their slot leaves quarantine (`QX_ADMIT=1`, off) | Strata (admit on copy) | **reverted to opt-in**: under mixed traffic the hit rate collapsed from ~70% to 13-20% and every concurrency level fell to ~33 tok/s; not understood yet |
| Resident NVFP4 experts grouped by slot (`qx_moe_groups`): each weight read once per window | Strata | bit-identical; +1% at 16 sequences (CPU-bound) |
| PCIe share of misses | Strata | not done: ~7% ceiling here (measured) |
| Route tail skip | Strata (off for NVFP4) | not done: changes the answers by more than the quantization |

## Results

Production API (8 x 262k slots, `loadgen.py --unique --warmup 64`, 3 rounds):

| clients | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| aggregate tok/s, before | 48 | 68 | 67 | 67 |
| aggregate tok/s, after | 50 | 80 | 81 | 78 |
| per request, after | 63 | 23 | 11 | 11 |

Prefill through the API, after: 25k-token prompt 1,070 tok/s, 100k 1,139 tok/s. Single requests:

| | before | after |
|---|---|---|
| 400-token answer, one 64k slot | 81-82 tok/s | 95 tok/s |
| 400-token answer, 8 x 262k slots | 67 tok/s | 75-76 tok/s |
| 2,000-token answer, one 64k slot | | 113 tok/s (cache warm by then: 93% hits) |
| `qx-batch-bench` 1 / 4 / 8 / 16 sequences | 53.5 / 98 / 108 / 117 | 64 / 167 / 174 / 210 |

Against Strata's 114-116 tok/s on a 400-token answer, one 64k slot is now at 95. The rest of the gap is the cold
cache at the start of a conversation: the same engine reaches Strata's figure after a few hundred tokens.
