# DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS in QuixiInfer

Revised 2026-11-03. The target is `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` **UD-IQ2_XXS** (280 GiB, 6 shards) on this
machine: RTX 5090 32 GB, Ryzen 9 9950X, 120 GiB of RAM (the `win10` VM is retired; its 16 GiB and 16 vCPUs are ours),
a 4 TB Crucial T705 PCIe 5.0 NVMe as the root drive (2.8 TB free; the drive reads at about 12 GB/s sequential and over
a million 4 KiB reads per second). The October version of this plan chose UD-IQ1_S because IQ2_XXS's experts did not
fit RAM; the NVMe makes IQ2_XXS the target, with a third expert tier on disk.

Sources: the official `inference/model.py`, `engram.py`, `kernel.py` (the arithmetic); llama.cpp PR ggml-org#30817
(`~/git/llama.cpp-dsv41`: `src/models/deepseek41.cpp`, built at `build/bin/llama-cli`; the oracle for golden logits,
KL 4.8e-3 against DeepSeek's own code; note its `.git` file points at a worktree parent that no longer exists, so the
source and binaries are intact but git commands fail there); the Unsloth fork PRs #285/#298 for the IQ1 quant types;
DeepInfer for the scheduler and the drafter; basalt for the expert tiers and the speculation policy
(`docs/reference-review-2026-11-03` notes in `progress.md`).

## What is in the file (from the shard headers)

| Part | Tensors | Bytes |
|---|---|---|
| Routed experts, 40 layers x 384 | gate/up: 15 layers IQ2_S, 15 IQ2_XXS, 10 IQ1_S; down: 15 IQ3_XXS, 15 IQ2_S, 6 IQ2_XS, 3 IQ2_XXS, 1 Q2_K | **146.1 GiB**; 7.3-11.5 MiB per expert record, mean 9.74 MiB |
| Engram table `per_layer_token_embd` | 768,022,850 rows x 256, Q5_K (176 B per row) = 48 tables of 16M rows (2 layers x 8 heads x n-grams 2..4) | **125.9 GiB** |
| Dense | attention Q8_0/Q6_K (5.3 GiB), shared experts (1.3), router F32 (0.3), `token_embd` Q8_0 (0.65), `output` Q6_K (0.5), `engram_kv` Q6_K (0.24), hc/indexer/compressor small | **~7.6 GiB** |

Per token: 6 experts x 40 layers = 240 expert records, **2.28 GiB** of expert weights (Qwen3.8-Flash-Next NVFP4 in
production moves 1.27 GiB per token). Per token the Engram reads 48 rows of 176 B (one per table), addressed by
hashes of the token ids alone.

Config (GGUF keys): 40 layers, hidden 5120, 4 hyper-connection streams (20480 flattened), vocab 129,280, 384 experts
with 6 used plus 1 shared, expert width 2304, gating `sqrt(softplus)` with a correction bias (a separate `_vl` bias for
image tokens), weights normalized and scaled 1.5; attention 64 heads x 512 (448 nope + 64 rope), one shared 512-wide
latent as K and V, q LoRA rank 1280, output LoRA rank 1024 in 8 groups, sliding window 128, YaRN 16x from 65,536 with
`compress_rope_freq_base` 160,000; KV source layers [2, 8, 14, 20], indexer source layers [2, 8, 14, 20, 24, 28, 32,
36] with 32 heads x 128, top-k 512, candidate blocks of 8 from layer 20 (top 2048); Engram at layers [1, 14], 4-gram,
8 heads per n-gram, compressed vocabulary 99,092, pad id 2; DSpark drafter: block size 5, noise token 128,799, target
layers [37, 38, 39], Markov rank 256, 128 experts with 3 used (separate GGUF). Context 1,048,576.

## Where everything lives

| Tier | Experts | Other |
|---|---|---|
| VRAM 32 GiB | **~20 GiB, ~2,100 records (14%)**, adaptive | dense 7 GiB (`token_embd` to host: 0.65 GiB saved); KV working set, indexer keys, Engram row ring, prefill staging and activations ~3-4 GiB, lent from the expert cache during a prompt |
| RAM 120 GiB | **~88 GiB pinned, ~9,250 records (60%)**, adaptive | Engram row cache 8 GiB (48M rows, 6% of the table); KV page pool 6 GiB; prefix arena 4 GiB; NVMe staging rings and mailboxes 4 GiB; OS and page-cache floor ~10 GiB |
| NVMe 2.8 TB free | **~38 GiB, ~4,000 records (26%)**, read in place from the pack | Engram table 126 GiB; KV page file for parked sequences and prefix entries (cap 200 GiB); the pack itself (280 GiB) beside the GGUF (280 GiB, deletable once the pack verifies) |

The Qwen service (`quixiinfer serve`, 8 slots, 262k) uses 31 GiB of VRAM and about 95 GiB of RAM, so it and the
DeepSeek engine cannot run at once; the Qwen service stays in production until the DeepSeek engine passes its checks.

## Decode cost model and what decides it

A token's cost is the misses, as for Qwen. Per expert record (mean 9.7 MiB): resident in VRAM about 7 us (1.5 TB/s,
inside the resident kernel); in RAM about 185 us (the CPU pool at 53-55 GB/s, the RAM's bandwidth; Qwen's NVFP4
records cost 54 us for 2.64 MiB, the same rate); on NVMe about 0.85 ms of drive time (12 GB/s) plus 0.1-0.2 ms of
latency, then either the CPU pool computes it from pinned staging or a DMA moves it to VRAM (0.2 ms at 50 GB/s).

| Routing shares VRAM / RAM / NVMe | RAM misses | NVMe misses | ms per token (single stream) | tok/s | at 8 sequences, aggregate |
|---|---|---|---|---|---|
| 60 / 37 / 3 | 89 x 0.185 = 16.5 | 7 x 0.85 = 6 (hidden by lookahead: 0) | 23-29 (+6 dense, attention, resident experts) | 35-44 | ~55-60 |
| 45 / 50 / 5 | 120 x 0.185 = 22 | 12 x 0.85 = 10 | 32-38 | 26-31 | ~40-45 |
| 30 / 60 / 10 | 144 x 0.185 = 27 | 24 x 0.85 = 20 | 40-53 | 19-25 | ~30 |

**Measured (D0b, 2026-11-03).** `tools/golden/route_trace` recorded the experts of 69,262 tokens (wikitext, this
repo's code, its docs; `golden/data/route-dsv41.trace`); `tools/cache_sim3.py` calibrates a static placement on the
first 20% and measures the adaptive three-tier policy (decayed counts, half-life 256 tokens, 4 promotions per
token per tier) on the rest:

| VRAM / RAM for experts | VRAM share | RAM share | NVMe share | RAM time | drive time | estimate |
|---|---|---|---|---|---|---|
| 20 GiB (2,113 records) / 88 GiB (9,235) | 74.4% | 25.0% (580 MiB per token) | 0.6% (14 MiB) | 11.7 ms | 2.3 ms (1.8 with lookahead) | **50 tok/s** |
| 20 / 80 | 74.4% | 24.6% | 1.0% | 11.7 ms | 3.6 ms | 47 |
| 16 / 88 | 69.4% | 29.9% | 0.8% | 13.8 ms | 2.9 ms | 44 |
| 20 / 88, half-life 1024 | 71.0% | 28.4% | 0.6% | 12.9 ms | 2.0 ms | 48 |

So the first row of the cost table above is the one to expect: the routing is as skewed as Qwen's, the NVMe tier is
rarely on the token's path, and the single-stream estimate is 45-50 tok/s (6 ms of dense, attention and resident
experts assumed; to be measured). **The adaptation is what gets there**: the same trace with the placement frozen
at the calibration ranking gives 52% / 40% / 8.5% (20 + 60 GiB), and the first dozen tokens of a text only
22% / 49% / 29%, which is what the first tiered run of the engine saw on a short prompt from a cold start. As for
Qwen, the policy (decayed counts, a few promotions per token into each tier, prompt routing seeding the counts)
is part of the design, not a tuning. Promotions run at 2 per token into VRAM and 1.2 into RAM (32 MiB per token of
copies, included in the times). At 8 sequences the pool's time grows with the sequence count (misses are per
token and rarely shared), so aggregate throughput levels off early, as it does for Qwen.

The old estimate for IQ2_XXS (3-6 tok/s) assumed the NVMe served every expert that RAM could not hold, with no
placement by frequency. With the cold quarter of the experts on disk and NVMe misses prefetched a layer ahead, the
drive is rarely on the token's critical path.

## Design

### Expert tiers: VRAM, RAM, NVMe

Reuse the Qwen engine's VRAM slot cache, swapper and pinned `HostExperts`; add the third tier and one ranking across
all three.

- **Pack.** Expert records `[gate | up | down]` 2 MiB-aligned as today, with per-layer record sizes (the types differ
  by layer; the planner and the pool already take a `gate_row_bytes` / `down_row_bytes` per layer). The NVMe tier
  reads records straight from the pack with O_DIRECT (`io_uring`, the PLE reader's machinery), so nothing is stored
  twice. Expert order inside the pack: by routing-profile rank within a layer, so the cold tail is contiguous and the
  warm set loads with few large reads.
- **One ranking, three tiers.** Decayed routing counts per (layer, expert) as now (basalt decays by 0.85 per round and
  ranks all layers together: candidates and victims from every layer in one sorted pass, swaps while candidate >
  victim + margin; measured 1.9% faster windows and 6-15% in a request's first 50). Resident set = VRAM; the next
  ~9,250 by rank = pinned RAM; the rest stays on disk. Promotion out of NVMe is a read into a spare pinned place
  (basalt keeps 4x the round's swap count of spare places); eviction from RAM is a free. Up to 96 swaps per round, 4x
  after a saturated round (basalt's "topic change": the previous round ranked 2x its cap). Spare swaps promote up to 16
  once-routed experts (basalt: -3%). A prompt seeds the counts from its routing histogram (old counts x 0.3 plus the
  prompt's scaled to 64 tokens) before decode starts. Placement persists in the routing profile, as now.
- **NVMe misses on the token path.** A miss not in RAM is read into a pinned staging ring (QD 4-8: colibri measured
  QD4 as the knee on a PCIe 4 drive, 6.3 GB/s with 5.6 MB reads; the T705 should take more) and computed by the CPU
  pool from the staging buffer with the same bit-exact dot products, then promoted if its count warrants. The pool's
  job list for a layer is built when the router runs; NVMe jobs are issued first and the pool starts on the RAM jobs
  while they land.
- **Lookahead for NVMe only.** Score layer l+1's router on layer l's FFN input with a 4-bit copy of the router
  (basalt `cpu_warm`: `amax/7` nibbles, scales per 32; colibri measured 71.6% recall of the true top experts on
  GLM-5.2 against 41% for "same as the previous token"), and issue NVMe reads for predicted experts that are neither
  in VRAM nor RAM. Prediction only prefetches; the real router decides, so exactness holds. With 1-10 NVMe misses per
  token this hides most of the drive latency; at 8 sequences the predicted set is the union over lanes.
- **Bit-exactness across tiers** as for Qwen: Q8_K / Q8_0 activations, `vpdpbusd` on the CPU and `dp4a` on the GPU
  with the same summation order per 256-chunk; decoders for IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and Q2_K on both
  sides (`qx-invariance` extended to the new types). The grids and sign tables come from `ggml-common.h` through
  `tools/gen_iq_tables.py`.

### Engram table on NVMe

The PLE path as it exists (O_DIRECT through `io_uring` on its own thread, a set-associative row cache in RAM, a
VRAM ring of rows ahead of the GPU) with these changes:

- **Pack section:** 4 KiB pages of 23 whole Q5_K rows (176 B; 1.2% padding), 48 tables of 16M rows each.
- **Addresses from token ids alone** (rolling XOR hash of `id * multiplier` modulo per-head primes over compressed
  ids, the metadata in the GGUF header). So for a prompt every row of a chunk is known before the chunk runs: issue
  them all when the chunk is scheduled (a 16k-token chunk is 786k rows, about 0.5-1 s of random reads, overlapped
  with the chunk's 4 s of weight movement). For decode the 48 rows of the next token are known the moment it is
  sampled; with the drafter they are known for every draft before verification, so prefetch the drafts' rows.
- **Row cache 8 GiB** (48M rows). Colibri notes the traffic is Zipfian; Qwen's PLE cache served 80% of rows from
  RAM. Measure the hit rate and resize.
- **GPU side:** `engram_kv` [6144 -> 25600] per layer gives 4 keys (one per hyper-connection stream) and one value;
  gate per stream = sigmoid(signed sqrt(dot(rms(R_c), rms(key_c)) x weight / sqrt(5120))); R_c += gate_c x value.
  Rows are dequantized in the gather kernel (Q5_K blocks of 256).

### KV cache: host pages as home, VRAM working set, NVMe for parked state

KV per sequence is small: a 128-cell window per layer (512 B rows), compressed latents at the 4 source layers
(positions / ratio, 512 B each), indexer keys at the 8 index source layers (128 per compressed position). At 262k in
FP8 that is about 400 MiB per sequence; DeepInfer stores the latent as FP4 e2m1 with an e4m3 scale per 16 (288 B),
the indexer key as MXFP4 (68 B) and the window as MXFP8 (528 B), which would halve it again.

- **Home in host pages** (`KvPages`, 256-token pages, mapped memory) with the latest window of cells in a VRAM ring
  and the indexer keys of the working set in VRAM, as the Qwen engine does. Eight 262k slots are 3.2 GiB of pages;
  the live set fits RAM easily.
- **NVMe page file** for pages of sequences that are parked (idle for 15 s of the engine's busy clock, basalt's
  rule) or whose prefix entry is evicted from RAM: pages written once at eviction with O_DIRECT, indexed by content
  hash, restored at drive speed (a 262k conversation in about 50 ms). The index persists across restarts.

### Automatic prefix caching with bounded replay

DeepInfer's deployment Q&A gives the rule that makes this cheap for this model: the sliding window and the
compressor state at any position depend only on the last `window + backtrack` tokens (128 + 6, keep 140), so a
prefix matched at any boundary needs **only its KV pages plus a replay of at most 140 tokens**, not a saved recurrent
state. The Engram history is 3 token ids.

- **Match per page, not per entry:** a hash chain over 256-token pages (hash of the page's tokens and the previous
  page's hash), so any sequence's pages are reusable by any later prompt with the same prefix, system prompts and
  multi-turn histories included. The existing `PrefixCache` entries become hash-chain heads with no arena state.
- **Restore:** load the matched pages (RAM, or NVMe page file), replay the last 140 tokens as one small prefill
  chunk to rebuild the window ring, the compressor ring state and the indexer's selection state, then continue.
  DeepInfer stores window state only at 128-token boundaries for the same reason; the replay replaces storage.
- **Eviction:** LRU over pages by last use, RAM first, then the page file by its cap. Pages shared by reference
  count as now.
- **Checks:** `qx-prefix-check` extended: a restored prefix gives the same logits as the uncut prompt, with the
  pages served from RAM and from the page file.

### Scheduler, speculation, structured outputs

Keep the Qwen engine's windows, lanes, asynchronous step queue and chunked prefill. Take from DeepInfer:

- **Multi-prompt prefill chunks** (`_schedule_prefill`): pack the waiting prompts FIFO into one prompt step up to a
  token budget and a prompt count, chunk only the last that does not fit, never mix prompt and decode tokens in a
  step, and ration prompt steps to every N decode steps so decode stalls only for them. This is the "multi-slot
  prefill chunks" item on the Qwen list (3x cheaper than lanes for 16 prompts).
- **Deferred release:** free a finished sequence's pages and slot at the next step's pre-sampler point, after the
  queued step is known not to touch them.
- **Not taken:** variable-length verify across sequences (DeepInfer measured -1.5% at 16 sequences per rank, gains
  only from ~48); the per-admission sync (DeepInfer pays one too).

Speculation with the **DSpark drafter** (separate GGUF; DeepInfer's `DFlash` is the same design): the main forward
writes the drafter's keys from the mean of the 4 streams at layers 37-39 into the drafter's own window ring, so the
drafter is always in sync and rejected drafts need no rollback (their ring slots are overwritten next step); one
non-causal pass over [bonus token, noise token x 4] drafts 5 tokens with a confidence head and a rank-256 Markov term.
Policy from basalt: speculate only when a sequence is alone (basalt caps a batch window at 8 rows in total, so it
does not speculate at 8 sequences either); draft length by cumulative acceptance against a threshold (0.35, adapted
per request from observed/expected acceptance) scaled 1.0-1.5x by the bytes the CPU and NVMe serve per expert, since
each extra row's misses are what a window costs here. Verification exact as now (`qx-spec-check`).

Structured outputs: the existing grammar engine, with the bitmask per draft position applied on the GPU and the
matcher rolled back after the verify, as DeepInfer does with XGrammar.

### Prefill

Chunks of 8k-16k tokens (activations borrowed from the expert cache: a 16k chunk's 4-stream residual is 1.3 GiB).
Per chunk every non-resident expert must move: 88 GiB from RAM over PCIe at ~45 GB/s (2 s) and 38 GiB from NVMe at
12 GB/s (3.2 s, into pinned staging then DMA), both streamed in (layer, expert) order through a VRAM ring by a
dedicated issuer thread while attention runs (basalt streams all non-resident experts from 6,144-token chunks on).
So the weight movement floors a chunk at 3.5-4 s: a 16k chunk at most ~4,000 tok/s, an 8k chunk ~2,000. With the
expert GEMMs (tokens sorted by expert, IQ decoders in the tile loader as `gemm_grouped` does for NVFP4) and the
dense bf16 GEMMs, expect **1,000-2,500 tok/s**; the Engram rows for the chunk are issued at schedule time. A 100k
prompt reads the NVMe tier about 7 times (265 GiB, read only). The CPU share of streamed experts (basalt: 0.6 of
those with 8 rows or fewer) helps only below 4k-token chunks, so it is for the short-prompt lane path.

## Plan

- **D0a. Download** (running): `tools/pardl.py`, 24 connections, resumable, to
  `~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/` (80 MB/s observed: about an hour).
- **D0b. Measure the routing skew.** Patch the oracle to dump the selected experts per (layer, token) on a 50k-token
  mixed corpus (chat, code, wikitext) and extend `tools/cache_sim.py` to three tiers with the sizes above and the
  adaptive policy. Output: the expected shares and tok/s row; the VRAM/RAM split (more Engram cache against more
  pinned experts) is decided here too.
- **D0c. Goldens.** `tools/golden/golden-dsv41` on the fixed prompts with the IQ2_XXS shards (mmap, 280 GiB on 120 GiB
  of RAM: fine for a few hundred tokens; the oracle's `LLAMA_DSV41_SPARSE` path on and off).
- **D1. Reference (`qx-ref`).** The forward pass below in f32 on the CPU from the GGUF: decoders for IQ1_S, IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, Q2_K, Q5_K, Q6_K; Engram hashing from the header metadata (multipliers, primes, offsets,
  token map). Exit: layer-0 activations match the oracle to float rounding, and over the golden positions the
  logits are as close to the oracle as the reference is to itself under a different activation rounding (this
  2-bit model amplifies Q8 rounding flips, so bit-level agreement past layer 0 is not attainable). **Done
  2026-11-03**, see `progress.md`.
- **D2. Pack.** Per-layer expert records, the Engram section in 4 KiB pages, dense tensors, SHA-256 per section;
  `qx-pack verify` round trip. **Done 2026-11-03** with the existing format (natural expert order; the routing
  profile and a rank order can be added later): `~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/*.qxp`.
- **D3. Engine.** Reuse: slot cache and swapper, pinned experts, CPU pool, PLE reader, `KvPages`, prefix cache,
  batching, server. New: the NVMe tier and three-tier ranking; the Engram reader's addressing and gather; the page
  hash chain, bounded replay and page file; kernels for `hc_mixes` with Sinkhorn, attention over [window | selected
  latents] with sinks and the un-RoPE, the compressor, the indexer with candidate blocks, the Engram gate, the
  sqrt-softplus router with bias, clamped SwiGLU, IQ decoders for GEMV and grouped GEMM, Q6_K/Q8_0 dense; CPU pool
  dots for the IQ types. Exit: the Qwen exactness checks on this model (`qx-invariance` across all three tiers,
  `qx-window-check`, `qx-batch-check`, `qx-prefill-check`, `qx-prefix-check` from RAM and from the page file) and
  parity with the reference. **Decode done 2026-11-03** (`crates/qx-engine/src/dsv41/`): the one-token path with
  all the kernels, the three tiers with the asynchronous driver (mailbox pool, stream-ordered promotions), the
  token captured as a CUDA graph, Engram rows gathered through io_uring; exact f32 parity with the reference and
  production parity within the rounding floor. **2026-11-04:** windows of up to 128 tokens (bit-identical to
  one-token steps on the f32 GEMV path), prefill through them, and `quixiinfer serve` / `run` on the DeepSeek
  engine (one request at a time, host sampling, the GGUF's chat template and tokenizer). Still to do under D3:
  Engram row cache, KV paging and the page file, bounded-replay prefix cache, multi-slot batching, structured
  outputs for this model.
- **D4. Speed.** The adaptive three-tier policy and its profile; NVMe lookahead; prefill chunk size against the
  stream; the DSpark drafter and the acceptance-threshold policy; `qx-ppl` and `loadgen.py` as for Qwen.
  **Measured so far** (p1 + 128 greedy tokens, 20 GiB VRAM / 80 GiB RAM, 32 promotions per token): the last 32
  tokens average 41.8 ms (24 tok/s); budget GPU 16.5 ms (dense GEMVs 6.5, attention, norms, 1,800 launches in a
  graph), pool compute about 14 (near RAM bandwidth: 0.22 ms per expert), NVMe wait about 5, Engram gather 4
  (being batched), gaps 2. Levers left: more VRAM/RAM hits (bigger tiers, better ranking), dense GEMV
  efficiency, kernel fusion of the small launches, DSpark.
- **D3b. Windows of T tokens** (the enabler for prefill chunks and DSpark verification; next). The decode path
  becomes `launch_window(T)`: every scratch buffer gets TMAX rows, the kernels that already index `blockIdx.y`
  (hc, rope, attention, indexer, combine, Engram gate) take grid.y = T, the per-position ones (compress, index-key
  commit, KV append) loop over the window's tokens in order inside one launch, and attention masks causally
  within the window (token t sees ring positions up to pos + t and the selected compressed rows committed before
  its group). The mailbox already posts T tokens' misses (`qx_moe_post` with info[13] = T) and the pool reads each
  expert once per window, so a window's CPU cost is near one token's. Engram rows for T tokens go in one io_uring
  batch. One graph per T. Exit: a window's logits equal the one-token path's bit for bit (`qx-window-check`), then
  prefill chunks of 32-64 forced tokens and the DSpark drafter on top.

Open:

- The second T705 (`nvme1n1`) is the Windows VM's disk and is **off limits**: everything here lives on the root
  drive (`nvme0n1`).
- RAM is now the hard limit on the RAM tier (88 GiB pinned), and every GiB given to the Engram row cache or the KV
  pool is 105 fewer pinned experts. D0b sizes this.

## The model (what the forward pass computes)

40 layers, hidden 5120, vocab 129,280, 384 routed experts of width 2304 with 6 active plus 1 shared, hyper-connections
with 4 copies (mHC with Sinkhorn), Engram at layers 1 and 14, sparse attention with a 128-token sliding window plus
compressed KV, DSpark drafter (3 layers, separate GGUF). Per layer, with the residual as 4 copies R[4][5120]:

1. **Engram (layers 1, 14):** 24 table rows per token (n-grams of 2..4 compressed token ids, 8 heads each; rolling
   XOR hash of `id * multiplier` modulo per-head primes, `token_map` compresses the vocabulary, pad id past the
   start). `engram_kv` [6144 -> 25600] gives 4 keys (one per copy) and one value; gate per copy =
   sigmoid(signed sqrt(dot(rms(R_c), rms(key_c)) * weight / sqrt(5120))); R_c += gate_c * value.
2. **hc_mixes:** rms over the flattened 20480-stream, one [24 x 20480] product, then pre (4: sigmoid*scale+base, +eps),
   post (4: 2*sigmoid), comb (4x4: softmax rows + eps, then 20 Sinkhorn row/column normalizations). The pre-mix a
   sublayer computes is used by the *next* sublayer (attention uses the previous FFN's).
3. **Attention:** x = rms(hc_pre(R, pre)); qr = rms(wq_a x) [1280]; q = wq_b qr -> 64 heads x 512 (448 nope + 64 rope,
   RoPE on the last 64 with theta 10000 for window-only layers, theta 160000 + YaRN(16, 65536) for compressed layers);
   kv = rms(wkv x) [512], one latent shared by all heads, K = V = kv (the window ring holds 128 of them).
   Compressed layers (ratios: 2 for layers 2-19, 1 for 20-39, 0 for 0-1): the 4 KV source layers (2, 8, 14, 20) pool
   `ratio` tokens with a learned softmax gate into one latent (rms, RoPE at group positions); consumers read the
   source's cache. Index source layers (2, 8, 14, 20, 24, 28, 32, 36) run the indexer: 32 heads x 128 queries from qr,
   keys from the latent (wk, rms, RoPE), score = sum_h relu(q_h.k) * w_h, top-512 visible compressed positions;
   layer 20 also picks 2048 candidate blocks of 8 that layers 24-36 restrict to. Each query attends over
   [window | selected compressed rows] with per-head sinks (added to the softmax denominator), then the output's
   RoPE is undone, wo_a per group of 8 heads [4096 -> 1024], wo_b [8192 -> 5120]. R = hc_post(out, R, post, comb).
4. **MoE:** scores = sqrt(softplus(router x)); selection by scores + bias (vision bias for image tokens), weights
   from the raw scores normalized and scaled by 1.5; experts SwiGLU with up clamped to [-10, 10] and gate to <= 10;
   shared expert the same; R = hc_post.
5. Head: hc_pre with the last FFN's pre-mix, rms, output matrix.
