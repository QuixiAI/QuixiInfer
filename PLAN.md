# QuixiInfer plan

Status: plan only, written 2026-10-24. No code yet.

## 1. Goal

Run Qwen3.8-Flash-Next `UD-IQ3_XXS` on this workstation as fast as possible, using DeepInfer's architecture and
algorithms, with these placements:

- model weights offloaded to CPU RAM, with a hot subset in VRAM;
- the KV cache in FP8 (E4M3), offloaded to CPU RAM, with a working set in VRAM;
- the PLE n-gram table ("engram", 26.8 GiB) offloaded to NVMe.

### Success criteria

All speed figures are measured on this machine, with the GPU power limit stated, against basalt running the same quant.

| Metric | Target |
|---|---|
| Quality, decode path, 20 wikitext-2 chunks vs Q8_0 | KLD mean at or below basalt's 0.244; same top-1 at or above 87% |
| Single-stream decode, no speculation | at or above basalt (about 184 tok/s at 400 W in basalt's docs) |
| Single-stream decode, with speculation | at or above basalt (about 320 tok/s at 400 W in basalt's docs) |
| Prefill, 16k-token prompt | at or above basalt; measured in phase 0 |
| Speculative output | identical to plain sampling for the same seed |
| Dependencies | builds from this repo alone; compiles its own kernels |

DeepInfer's own numbers are cluster throughput on 64 SM100 GPUs, so they are not a target here. Basalt is.

## 2. Machine

| Part | Detail |
|---|---|
| GPU | RTX 5090, 32 GiB, SM120, PCIe 5.0 x16, power limit 575 W (max 600 W) |
| CPU | Ryzen 9 9950X, 16 cores / 32 threads, 2 CCDs, full AVX-512 (VNNI, BF16, VBMI) |
| RAM | 120 GiB DDR5 |
| SSD | Samsung 9100 PRO 1 TB, PCIe 5.0; 174 GiB free |
| CUDA | Toolkits 13.1 and 13.3 in `/usr/local`, not on PATH; driver 595.99 supports CUDA 13.2 |
| Co-tenants | `win10` VM, 16 vCPUs and 16 GiB, always on; SlimServe disabled (it would need the GPU and 60 GiB) |

Use CUDA **13.1**. The driver tops out at 13.2, so NVRTC or PTX from 13.3 may not load.

## 3. Memory budget

Model facts are in [docs/model-facts.md](docs/model-facts.md).

### VRAM (32 GiB)

| Item | GiB |
|---|---|
| Desktop and driver reserve | 1.0 |
| Dense weights, shared experts, head, router | 4.2 |
| MTP drafter (Q8_0) | 3.9 |
| Activations, graphs, prefill staging | 2.0 |
| KV working set, indexer keys, DeltaNet states, one 32k session | 1.0 |
| **Routed-expert cache** | **about 20** |

About 20 GiB holds 44% of the 45.3 GiB of routed experts. The hit rate depends on routing skew and is measured in phase 4.

### RAM (120 GiB)

| Item | GiB |
|---|---|
| OS, desktop, page cache floor | 8 |
| `win10` VM | 16 |
| Pinned routed-expert master copy (all experts) | 45.3 |
| Pinned FP8 KV, 2 slots at 262k | 6 |
| PLE row cache | 2 to 8 |
| Prefix cache: KV pages and DeltaNet checkpoints | 8 |
| **Total** | **about 85 to 91** |

This fits with SlimServe off. If SlimServe is re-enabled at 60 GiB, QuixiInfer and SlimServe cannot both run.

### NVMe

| Item | GiB |
|---|---|
| Source GGUF shards during conversion | 76.3 |
| QuixiInfer pack, including the repacked PLE table | about 76 |
| MTP drafter | 3.9 |
| Prefix-cache spill | 10 to 20 |

The converter deletes each GGUF shard once it is packed, so peak use stays near 85 GiB instead of 156 GiB.

## 4. Architecture

### 4.1 Three tiers

| Tier | Holds | Moves |
|---|---|---|
| VRAM | Dense weights, hot experts, KV working set, DeltaNet states, indexer keys | Expert swaps in the background; KV pages on demand |
| Pinned RAM | Every routed expert, the full FP8 KV, prefix-cache checkpoints | PCIe 5 copy engine, H2D and D2H |
| NVMe | PLE table, prefix-cache spill | io_uring with O_DIRECT, 4 KiB reads |

### 4.2 Decode step, one layer

1. The GPU runs the mixer: either gated DeltaNet, or full attention with the sparse indexer.
2. The GPU runs the router and splits the 10 experts into VRAM hits and misses.
3. The GPU computes the hits with grouped quantized kernels. At the same time it copies the 2560-float activation to
   pinned memory, and the CPU pool computes the misses from RAM with AVX-512.
4. The GPU waits on a flag the CPU writes into mapped memory, adds the miss outputs, and applies the hyper-connection
   transition.

The CPU computes misses because, for one token, reading an expert from local RAM beats copying it over PCIe. A miss
costs about 1.9 MiB of RAM reads. The CPU pool spans both CCDs, because one CCD cannot saturate DDR5.

### 4.3 Scheduler (DeepInfer)

- A scheduler with a block manager, fixed state slots for the 36 DeltaNet layers, continuous batching across N slots
  and chunked prefill.
- **Zero-overhead scheduling.** While the GPU runs step N, the CPU schedules step N+1, launches it and post-processes
  step N-1. Each step shape replays as a CUDA graph.
- Prefill and decode run in one process on separate streams, with decode at higher stream priority.

### 4.4 Expert placement (DeepInfer EPLB applied to tiers)

- The starting placement comes from a routing profile, ranked by frequency. The profile is gathered by the engine
  itself on a calibration set.
- An adaptive tier counts routing in a sliding window and swaps experts in the background. Each swap is a 1.9 MiB
  H2D copy into a free slot, followed by a table update between steps.

### 4.5 FP8 KV with sparse-aware offload

- Twelve full-attention layers, 2 KV heads, head dim 256. K and V are stored in E4M3 with one scale per head and cell.
  K goes through a Walsh-Hadamard rotation before quantization. The quality gate decides whether rotation stays.
- The authoritative KV lives in pinned RAM, in paged layout. VRAM holds a page cache: the recent window plus pages the
  indexer selected recently.
- Each step uses the indexer to pick at most 2048 cells per layer. The engine prefetches the pages that changed since
  the previous step, predicted from the previous selection. The prefetch overlaps the three DeltaNet layers before
  each attention layer. Pages still missing are fetched after the exact selection, on the copy stream.
- The indexer's pooled keys stay in VRAM, about 0.4 GiB at 262k.
- The attention kernel is new. It uses SM120 native FP8 tensor cores (`mma` E4M3) with fp32 accumulation.

### 4.6 PLE table on NVMe

- The pack stores the table as 4 KiB pages of 45 rows (90 bytes each), so no row crosses a page and each row is one read.
- Each token needs 16 rows: one per hash head of the two n-gram orders.
- **Prefill.** All tokens are known up front, so chunk k+1's reads are issued while chunk k computes.
- **Decode.** Reads are issued for the drafter's proposed tokens and for the sampler's top few candidates before the
  token is final. Reads for candidates that lose are discarded. The fallback is to issue reads as soon as the token
  is sampled, and hide them behind the embedding and layer 0.
- A RAM row cache with set-associative eviction (basalt's design) catches repeated n-grams.
- io_uring with registered buffers and a fixed queue depth. A keep-alive read stops the SSD from dropping into a
  low-power state between requests.

### 4.7 Speculative decoding (DeepInfer DSpark)

- The drafter is the published MTP layer, `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`.
- The draft token's probability stands in for DSpark's confidence head.
- The verify width is chosen per step to maximize expected accepted tokens per unit time. Cost per width comes from a
  measured table that includes **the expert misses each extra token adds**, because on this machine the marginal cost of
  a verify token depends on cache hits.
- Verification is exact: rejection sampling against the target with pre-drawn uniforms, so output equals plain sampling.

### 4.8 Prefix cache (DeepInfer bounded replay)

- The engine saves KV pages and DeltaNet state checkpoints every 4096 tokens of a conversation, to RAM and then NVMe.
- A hit restores the nearest checkpoint at or before the shared prefix and replays at most 4096 tokens.

### 4.9 Prefill

- Prefill is compute-bound, so it touches every expert. Each chunk streams a layer's non-resident experts over PCIe
  into double-buffered staging, while the GPU computes the resident experts and the previous layer.
- Chunks are as large as staging allows, up to 16k tokens, because a pass over the streamed experts costs about the
  same at any chunk size.

### 4.10 Sampling, grammar and serving

- Fused GPU sampling: temperature, top-k, top-p, min-p and penalties. Draws are a pure function of (seed, position,
  token), so runs are reproducible.
- Structured output uses `llguidance`, a Rust grammar engine. Masks are computed on CPU threads while the forward runs.
- An OpenAI-compatible HTTP server on axum and tokio, bound to 127.0.0.1 unless an API key is set.
- The tokenizer is the Hugging Face `tokenizers` crate, loaded from the GGUF vocabulary.

## 5. Language and stack

Rust for the whole engine, with Burn's CUDA backend (CubeCL) as the GPU runtime.

| Layer | Choice |
|---|---|
| GPU runtime, tensors, kernel launch, CUDA graphs, stream priority | Burn with the CubeCL CUDA backend |
| Custom GPU kernels | QuixiInfer's own CUDA C++ in `kernels/`, run through Burn's `KernelSource` (NVRTC, `sm_120a`) |
| Simple elementwise and reduction ops | CubeCL kernels written in Rust |
| VMM arenas, pinned and mapped host memory, raw copies | `cudarc` driver API, where Burn has no wrapper |
| CPU expert pool | QuixiInfer's own Rust, `std::arch` AVX-512 intrinsics |
| NVMe | `io-uring` crate, O_DIRECT |
| Server | axum, tokio |
| Tokenizer, grammar | `tokenizers`, `llguidance` |

What the CubeCL CUDA source path does today, read from its code on 2026-10-23:

- It compiles CUDA C++ text with NVRTC for `sm_<arch>a`. The text names one entry point.
- It searches only the CUDA toolkit and CCCL include paths. It passes no `-std=c++20`.
- It captures and replays CUDA graphs, and creates streams with priorities.

### Self-contained kernels

QuixiInfer owns every kernel it runs. Nothing from basalt, llama.cpp, QuixiCore or DeepInfer is linked, vendored or
loaded at build or run time. Those projects are example code: we read them, learn the technique, and write our own.

- `kernels/` holds the CUDA C++ sources and a small header set of our own: FP8 and BF16 conversions, `mma` wrappers for
  SM120, the IQ3_S, IQ2_S and IQ4_NL decoders and their codebook tables, warp reductions.
- Headers are written to C++17, so NVRTC needs no extra flags.
- `build.rs` resolves each kernel's `#include "..."` lines inside `kernels/` and emits one self-contained source per
  kernel. That fits the one-entry-point and toolkit-only-include rules of Burn's source path. Template variants, such as
  each quant format or tile shape, become separate generated sources.
- `build.rs` also compiles every generated source with nvcc 13.1 for `sm_120a`. A kernel error fails `cargo build`
  instead of surfacing at first launch, and the cubins are ready if a kernel ever needs the `cudarc` fallback.
- At run time Burn compiles the sources with NVRTC and caches the results on disk.
- Every kernel has a plain Rust CPU reference in the same crate, and a parity test against it on random inputs and the
  model's real shapes.
- The IQ codebook tables are fixed by the GGUF format. If we transcribe them from llama.cpp, its MIT notice goes in
  `THIRD_PARTY_LICENSES`.

### Phase 1 risk gate

If Burn cannot do any of the following, the affected piece drops to `cudarc` with nvcc-built cubins. Rust stays either way.

- launch a multi-header QuixiInfer kernel after `build.rs` flattens it, compiled through NVRTC;
- share device pointers between Burn tensors and `cudarc`-allocated VMM or pinned memory;
- capture one full decode layer into a CUDA graph and replay it;
- keep per-step host overhead under 50 microseconds for a captured 48-layer step.

## 6. Kernel references

Details are in [docs/kernel-references.md](docs/kernel-references.md). In short:

Every kernel is written for QuixiInfer. The reference projects show techniques worth learning; none of their code is used.

- **Model numerics:** llama.cpp's `qwen4exp` model is the external oracle. It runs once, offline, to produce golden
  logits and per-layer activations. It is not a dependency.
- **Quantized expert kernels:** learn from basalt's fused int8 kernels and llama.cpp's MMQ and MMVQ.
- **CPU expert pool:** learn from llama.cpp's and basalt's AVX-512 dot products.
- **Indexer, DeltaNet, hyper-connections, sampling, speculative verify, paged KV:** learn from QuixiCore-CUDA's serving
  kernels and llama.cpp's CUDA ops.
- **FP8 attention:** no close example exists for SM120 native FP8, so this one is designed from scratch.

## 7. Phases

Each phase ends with a test that must pass before the next starts.

### Phase 0: environment and baselines

- Install rustup and stable Rust. Put CUDA 13.1 on PATH. Raise the memlock limit for pinned memory (it is 8 MiB now).
- Download `UD-IQ3_XXS` and the MTP drafter.
- Measure: PCIe H2D and D2H on pinned memory; DDR5 read bandwidth per CCD and for both; NVMe 4 KiB random-read
  latency and IOPS at queue depths 1 to 64.
- Run basalt on the same quant for decode, speculative decode and 16k prefill, at a fixed power limit. These are the
  numbers to beat.
- Dump llama.cpp logits and per-layer activations for a fixed prompt set as golden files. This is the only time
  llama.cpp runs; the files are stored with the tests.

**Exit:** a baseline table in `docs/baselines.md`, and golden files on disk.

### Phase 1: Rust skeleton and Burn prototype

- Cargo workspace: `qx-pack`, `qx-gpu`, `qx-cpu`, `qx-io`, `qx-engine`, `qx-server`.
- `kernels/` with the shared headers and `build.rs` that flattens, generates and nvcc-checks the kernels.
- Prototype the four risk-gate items from section 5, using one real kernel (the IQ4_NL expert GEMV) end to end.

**Exit:** the risk gate passes, or the fallback is chosen and written down.

### Phase 2: pack format and converter

- GGUF reader in Rust, including types IQ3_S, IQ2_S and IQ4_NL.
- A pack with 2 MiB-aligned sections: dense weights; routed experts contiguous per (layer, expert), with gate, up and
  down together; the PLE table as 4 KiB pages; tokenizer; routing profile.
- Tensors are copied byte for byte. Only the layout changes.
- The converter deletes each shard once it is packed and checks every section's SHA-256.

**Exit:** a round-trip test proves every tensor in the pack is byte-equal to the GGUF.

### Phase 3: correct but slow decode

- Dense layers on the GPU. All routed experts on the CPU. PLE rows read synchronously. FP16 KV in VRAM.
- No graphs, no batching, no speculation.

**Exit:** logits match llama.cpp's golden files within tolerance on the fixed prompts, and greedy output is identical.

### Phase 4: expert tiers

- The VRAM expert cache, the routing profile, GPU expert kernels and the hit/miss split with overlap.
- The CPU pool on both CCDs, with threads pinned away from the VM's vCPUs where possible.
- CUDA graphs for the decode step.

**Exit:** decode tok/s and expert hit rate recorded; parity still passes.

### Phase 5: FP8 KV and RAM offload

- The FP8 KV format, the new attention kernel, the pinned RAM KV, and page prefetch from the indexer's selection.

**Exit:** KLD within 0.01 of FP16 KV on the quality gate; no decode slowdown at 32k context vs VRAM-resident KV.

### Phase 6: PLE on NVMe

- The repacked table, io_uring reader, prefetch for prefill chunks, candidate prefetch in decode, row cache.

**Exit:** NVMe wait under 2% of decode time and under 5% of prefill time.

### Phase 7: prefill path

- Chunked prefill with experts streamed over PCIe, and prefill alongside decode at lower stream priority.

**Exit:** 16k-token prefill at or above basalt's rate.

### Phase 8: speculative decoding

- The MTP drafter, exact verification, and miss-aware variable verify width.

**Exit:** output identical to plain sampling for fixed seeds; decode with speculation at or above basalt.

### Phase 9: scheduler and serving

- Batch slots, zero-overhead scheduling, prefix cache with bounded replay, grammar, and the OpenAI-compatible server.

**Exit:** a soak test with 4 concurrent clients for 1 hour with no errors; a prefix-cache hit skips at least 90% of
prefill on a repeated conversation.

### Phase 10: tuning

- Profile with Nsight Systems, fix the largest gap, repeat. Compare against basalt at the same power limit.

**Exit:** every target in section 1 met, or the remaining gap explained by a measurement.

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Burn's NVRTC path rejects a kernel nvcc accepts | That kernel needs a different load path | Load the `build.rs` cubin through `cudarc` |
| Burn launch or graph overhead | Lost time on every step | Measured in phase 1; captured graphs; `cudarc` fallback |
| Expert hit rate below about 70% | CPU misses dominate decode | Larger cache by moving the drafter to RAM; better profiles; speculation |
| The VM competes for cores and memory bandwidth | Slower CPU pool | Pin the pool's threads; measure with the VM loaded and idle |
| NVMe latency spikes | Stalls at layer 1 | Candidate prefetch, row cache, keep-alive reads |
| FP8 KV hurts quality | Fails the quality gate | Hadamard rotation on K; per-head scales; K8 and V8 split |
| Disk space | Conversion fails | Delete shards as they are packed |

## 9. Open questions

- The project license. MIT matches basalt, llama.cpp and QuixiCore. DeepInfer is Apache-2.0, which is compatible.
- Whether to support a second GPU later as a peer expert tier, as basalt does.
- Whether image input (the model's vision tower) is in scope.
