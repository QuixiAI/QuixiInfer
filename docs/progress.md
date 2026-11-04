# Progress

What exists, what it measured, and what is next. Speeds are decode on this machine (RTX 5090 at its default
575 W limit), greedy, UD-IQ3_XXS, a 1024-token wikitext prompt then 256 generated tokens, adaptive expert cache.

## Done

- **Phase 0.** Toolchain (rustup, CUDA 13.1 NVRTC + 13.3 nvcc), model and drafter downloaded, golden files from
  llama.cpp (`tools/golden`), hardware probes (`qx-probe`): PCIe 50 GB/s H2D, RAM 54 GB/s (49 GB/s from one CCD),
  GPU DRAM read 1.58 TB/s.
- **Phase 1.** Burn/CubeCL gate passed (docs/phase1-gate.md). Self-contained kernel pipeline: `kernels/`,
  flattened and nvcc-checked by `crates/qx-gpu/build.rs`, NVRTC at run time.
- **Reference.** `qx-ref`: the whole forward pass on the CPU in f32 (docs/model-forward.md), checked against
  llama.cpp: layer-0 activations within 3e-7 when llama.cpp's activation rounding is emulated; trajectories
  diverge only through rounding flips, which llama.cpp's own 8-bit activations cause.
- **Phase 3.** GPU engine (`qx-engine`) matches the exact reference: logits within 4e-6, KLD 1e-11, every token.
- **Phase 4.**
  - GPU and CPU experts with identical integer arithmetic (Q8_K/Q8_0 activations, dp4a / AVX-512 VNNI, shared
    deterministic exp): logits are bitwise identical whatever share of experts the GPU holds (`qx-invariance`).
  - AVX-512 expert pool near DDR bandwidth (IQ2_S 49 GB/s, IQ4_NL 53 GB/s on 16 threads).
  - All 45 GiB of experts pinned in huge-page host memory; VRAM slot cache with adaptive residency (decayed
    routing frequency, swaps on a copy stream, per-layer slot snapshots, quarantined evictions).
    Held-out simulation (`tools/cache_sim.py`): 62% hits static vs 95%+ adaptive at 20 GiB.
  - Free-running decode: GPU <-> CPU handoffs through mapped-memory mailboxes; no host sync per token; the step is
    one captured CUDA graph.
  - Kernels: fused multi-matrix GEMV, flash-decoding attention, QSA block selection on the GPU, parallel GDN.

| Milestone | Decode tok/s |
|---|---|
| First free-running engine, static cache | 46 |
| Constant-memory codebooks moved to global | 51 |
| Fused GEMV groups | 58 |
| Adaptive cache | 61 (85-98% hits, text-dependent) |
| Flash-decoding attention | 91 |
| Kernel fixes, miss-free layers skip the CPU | 93-117 (text-dependent) |
| Fusions, CUDA graph | 105 |

- **Phase 5.** FP8 E4M3 KV cache, one scale per (cell, KV head): the whole cache in mapped host memory, the
  latest `QX_KV_WINDOW` cells (default 4096) also in a VRAM ring; attention reads older cells over PCIe.
  Mean KLD against the exact reference 1.65e-3 (f32 KV: 1e-11).
- **Phase 6.** PLE n-gram table read from NVMe: O_DIRECT through io_uring on its own thread, a set-associative
  row cache in RAM (about 80% of rows served from it), a ring of 4096 positions ahead of the GPU. No decode
  slowdown.
- **Phase 9.** Tokenizer (identical to llama.cpp on 297k wikitext tokens), sampling on the GPU (temperature,
  top-k, top-p, min-p, seed), OpenAI-compatible server (`quixiinfer serve`: chat and text completions, SSE,
  reasoning split, API key; refuses non-local hosts without a key) and `quixiinfer run`.
- **Phase 7.** Batched prefill (`crates/qx-engine/src/prefill.rs`, `kernels/prefill.cu`, `kernels/gemm.cu`):
  chunks of `QX_PREFILL_CHUNK` tokens (default 1024) through every layer at once.
  - bf16 tensor-core GEMMs for the dense weights (128-token tiles), a grouped expert GEMM over the chunk's
    tokens sorted by expert (64-row tiles), chunked GDN recurrence with the state in registers, split-free
    FP8 attention per (query, KV head), per-query QSA selection.
  - Experts the VRAM cache lacks are staged: a copy thread DMAs them into two VRAM halves (768 MiB) one group
    ahead of the MoE, which waits for them on the host. (A GPU spin-wait deadlocks: a first-launch module
    load holds a driver lock until the GPU idles, and the copy thread needs that lock.)
  - The prompt's routing feeds the adaptive cache: a burst of up to 3072 swaps over the first decode tokens.
  - Prompts of 96+ tokens use it (`QX_PREFILL_MIN`); the decode step for the prompt's last token follows.
  - Parity: against token-by-token decode, mean KLD 3e-4 at 300 tokens. On llama.cpp's logits at positions
    2560-2835 (sparse selection active), prefill+decode KLD 0.060 / top-1 247 of 276, decode-only 0.068 / 240,
    the exact CPU reference 0.069 / 243: prefill is as close as the other paths, which all drift apart at long
    context through routing and selection flips.

| Prompt processing | tok/s |
|---|---|
| Token by token (decode step per token) | 95 |
| Prefill, first version (experts read over PCIe in place) | 24-47 |
| Experts staged to VRAM by DMA | 1025 (4k prompt) |
| Typed decoders, 128-token dense tiles, 64-row expert tiles | 1212 |
| Split-free prefill attention | 1519 |
| Narrow f32 GEMM | 1547 (4k), 1487 (1k) |

Decode right after a 1k-token prefill: 95 tok/s (token-by-token prompt: 103), 90% of routings on the GPU.

- **Phase 8.** Exact speculative decoding with the MTP drafter (`crates/qx-engine/src/drafter.rs`).
  - Decode is a window of 1 to 8 tokens (`launch_window`); every kernel has a token dimension or an in-order
    token loop with the one-token arithmetic, so a window's logits equal one-token steps bit for bit
    (`qx-window-check`: 200 positions, widths 2-8 mixed, half the experts on the CPU, 0 rows differ).
  - A window verifies [last token, drafts]; the kept drafts are those the target's own per-position samples
    confirm, so output equals plain sampling for any seed (`qx-spec-check`: greedy and sampled identical).
    Rollback rebuilds conv and PLE histories from saved starts and replays the DeltaNet recurrence from its
    saved state; the PLE thread rewinds; the indexer ring has 16 slots and the FP8 ring a margin of 8 cells.
  - Drafter: the published MTP layer, experts requantized to IQ4_NL (1.42 GiB VRAM, cached in
    ~/.cache/quixiinfer), the trunk's embedding and head (same weights), a 32k-token draft head, catch-up in
    prefill chunks, one captured graph per catch-up width. First-draft acceptance on wikitext: 62-66% greedy.
  - Width per window: expected kept tokens per measured second (`pick_width`), from the draft probabilities and
    running window costs by width. `stream` (and the server) decode speculatively when the drafter loads;
    `QX_SPEC=0`, `QX_MTP=0`, `QX_SPEC_K` (drafts, default 4).
  - Speed: about 6% over plain decode on greedy wikitext so far. Windows are bound by CPU expert misses
    (RAM bandwidth, 38 us per miss): each extra window token brings its own misses. Width 2 costs 2 ms over
    width 1; widths 3+ cost much more.

- **Phase 2.** The pack (`crates/qx-pack/src/pack.rs`, `qx-pack pack|verify`): one GGUF-headered file with
  2 MiB-aligned sections: dense tensors, expert records [gate | up | down] in the engine's pinned-memory layout,
  the PLE table in 4 KiB pages of whole rows; routing profile and per-section SHA-256 in the header. Written in
  55 s with O_DIRECT; round trip checked byte for byte (every tensor, zero padding). The engine loads the experts
  with one O_DIRECT read (5.3 s for 45 GiB, 7 GB/s against the drive's 11; page faults on fresh memory) and reads
  PLE rows from the paged section. `quixiinfer` prefers the pack when it exists. GGUF and pack give identical tokens.
- **Routing profile.** The swapper saves its ranking to `~/.cache/quixiinfer/routing-*.txt` (every 4096 tokens and
  at exit after a long run); the next start fills the cache from it, else from a wikitext calibration ranking
  compiled in. Cold-start hit rate 74% -> 81%; after a reset the swapper's quarantine was timed by the GPU step count,
  which resets go back: it issued no swaps afterwards (51% hits). Now timed by its own tick count.
- **Phase 8 tuning.** Prefill staging 384 MiB (was 768, 2% of prefill speed for 384 MiB of experts). The cache
  budget is capped by the VRAM actually free less 1.25 GiB headroom (the server's estimate left 0.6 GiB and OOMed).
  Spec after a token-by-token prompt hung: the host rewrote the step words while the previous window's
  drafter catch-up still read them; a sync before each prompt window fixes it.
- **Continuous batching** (`crates/qx-serve/src/worker.rs`, `Engine::load_slots`, `begin_seq`, `step_segments`).
  - The engine holds `--slots` (default 16) sequences: per-slot GDN, conv, PLE history, indexer and FP8 KV state
    (strided buffers, a per-layer KV table of addresses), step words, context, sampling, PLE mailbox and output
    ring; the PLE thread serves every slot. VRAM: about 190 MiB per slot at 32k context.
  - A window is up to 16 tokens in lanes (`s.slot`): consecutive tokens of one slot are a segment run in order
    by the sequential kernels; `qx_win_begin/end`, sampling and `qx_moe_post` (one block per token) take a lane
    each. Speculative windows are one segment; batched decode one token per slot; prompt tokens 8 per step.
  - Exactness: with one product kernel for every width (`QX_TMV=0` or `1`) batched logits equal solo logits bit
    for bit (`qx-batch-check`: 16 slots, 0 rows differ; generation identical). By default windows of several
    sequences above 4 tokens use `qx_tmv` (below), so their logits differ from solo by bf16 rounding: KL 5e-2
    against solo, the same as either path's distance from the exact reference.
  - `qx_tmv` (kernels/gemm.cu): the fused matvec on bf16 tensor cores for up to 16 tokens, reading each weight
    once: 16-row tiles, warps split K (and blocks, with a deterministic in-order reduction), inputs prepared
    as bf16 by `qx_tmv_prep`. It reaches about 650-800 GB/s against `qx_mgemv`'s 1200-1600 for one token, so
    `qx_mgemv` keeps every window up to 4 tokens and speculative windows up to 8.
  - The scheduler: a request alone decodes speculatively; when another arrives it joins the batch
    (`spec_to_batch`). Prompts of 96+ tokens are prefilled on admission.
  - `tools/loadgen.py` (concurrent streamed requests, 256 tokens, 16 prompts) and `qx-batch-bench`.

| Decode (pure batched steps, 256-token prompts) | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| tok/s | 91 | 209 | 284 | 382 |
| ms per step | 11.0 | 19.1 | 28.2 | 41.9 |
| GPU hit rate | 91% | 96% | 97% | 97% |

| Server (`loadgen.py`, after 64 warm-up requests) | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| aggregate tok/s | 74 | 150 | 250 | 293 |
| per-request tok/s | 84 | 46 | 35 | 26 |
| TTFT median / p90 (s) | 0.39 / 0.41 | 0.35 / 0.74 | 0.71 / 1.43 | 2.4 / 4.4 |

  Before batching the server did 93 tok/s at every concurrency (one request at a time, TTFT 40 s at c=16).
  At c=16 the step is 41 ms of which 30 are the window: the rest is prefill of arriving prompts (blocking) and
  the per-step sync; the CPU expert pool is 8-12 ms per step at 97% hits.

- **Asynchronous scheduling.** The server queues step N+1 before reading step N's samples (from the mapped
  output rings the GPU publishes); the window's lanes come from a mapped ring the window's first kernel copies
  (`qx_lanes_take`), so a step needs no host sync or upload. Measured gain at c=16: about 5%.
- **Prompt scheduling.** Prompts under 256 tokens read through lanes with priority (up to 8 lanes first, decoders
  fill the rest in round robin, leftover lanes go back to prompts); longer ones take whole prefill chunks between
  steps. Windows widened to 32 tokens (TMAX 32, AMAX 320). Budgeted small prefill chunks were tried and dropped:
  at 44% residency a 64-token chunk touches about 80 cold experts per layer, so a prompt's expert bytes are the
  same whichever engine moves them; priority is what TTFT needed (c=16 p90 7 s -> 5 s).
- **Correction.** The 293 tok/s / "100% hits" at c=16 recorded above came from the broken split-K `qx_tmv`
  (zeroed products collapsed the routing). With correct kernels and 16 diverse chat prompts after warm-up:
  c=1 73 tok/s, c=4 136, c=8 135-157, c=16 164-176; TTFT median/p90 at c=16 1.5 / 5.1 s. The c=16 step is
  bound by RAM bandwidth on expert misses (84% hits, about 1000 misses per step, the CPU pool busy 45 of 46 ms).
  Half-life 1024/4096 changed nothing: 84-85% is the working set at 44% residency. 20% of lane-tokens at c=16 are
  prompt tokens (65-token prompts against 256 generated).
- **Paged host KV** (`crates/qx-engine/src/kvpages.rs`, `kernels/include/qx_kv.cuh`): the FP8 host copy is pages of
  256 cells holding every attention layer ([K | V | K scales | V scales] per layer, 3.05 MiB), in one pinned
  arena with reference counts; each slot has a page table in device memory written by queued `qx_set_u64`
  launches as its sequence grows (no syncs). The drafter has private pages. Parity unchanged (prefill-check
  16/17, windows and speculation exact).
- **Prefix cache** (`crates/qx-engine/src/prefix.rs`): an entry is a sequence's recurrent state after its first p
  tokens (DeltaNet S and conv states, PLE history, indexer ring and pooled keys: 113 MiB at 32k context, in a
  pinned arena, `QX_PREFIX_GIB`, default 8 for a server) plus references to its KV pages; a restore copies the
  state back, shares the pages (a partial last page is copied), refills the VRAM ring (`qx_kv_ring_fill`) and
  rebuilds the PLE thread's history from the forced tokens. Snapshots: after each prefill chunk, before a long
  prompt's last token (identical prompts hit), and at the end of every request (the next turn hits). LRU
  eviction; the longest cached prefix of a prompt is restored (`qx-prefix-check`: a restored prefix gives
  bit-identical logits and tokens). Speculative decoding starts from a restored prefix too.

| 3-turn conversations, 200 tokens per turn (`loadgen.py --turns 3`) | c=1 | c=4 | c=16 |
|---|---|---|---|
| aggregate tok/s, prefix cache on / off | 95 / 85 | 123 / 95 | 156 / 105 |
| TTFT p90 (s), on / off | 0.21 / 0.67 | 0.94 / 3.2 | 2.5 / 17.2 |

  41 hits for 42 follow-up turns. With unique single-turn prompts (`--unique`) the cache costs about 4%: each
  end-of-request snapshot syncs the pipeline for a step.

- **Structured outputs** (`crates/qx-serve/src/grammar.rs`): JSON schema (`response_format` json_schema /
  json_object, `guided_json`), regular expressions (`guided_regex`) and Lark grammars (`guided_grammar`) compiled
  by llguidance into a token matcher built from the tokenizer's token bytes; each step the matcher's allowed set
  goes to the sampler as a bit vector per lane (mapped memory, `qx_sample` sets the rest to -inf). A constrained
  request's next mask needs its last token, so the step after it waits (sync mode) while one runs; constrained
  requests never decode speculatively. Checked: schema-valid JSON, regex full match, json_object, alongside a plain
  request.
- `tools/loadgen.py`: `--unique` (a nonce per prompt: no prefix hits), `--turns N` (conversations with follow-ups,
  thinking off), prompt token counts in the table.

- **Production.** `quixiinfer.service` replaced SlimServe on port 8000 (same key file; `Conflicts=` keeps them apart);
  the win10 VM is stopped, autostart off, `libvirt-guests` ON_BOOT=ignore. Expert pool on the 16 physical cores
  (24 and 32 threads measured slower: idle-spinning workers starve the engine, PLE and CUDA threads).
- **262k context.** `--ctx 262144 --slots 8`: 16 slots did not fit (the KV page arena pins slots x 1024 pages of
  3.05 MiB: 52 GiB beside the 45 GiB of experts, the box thrashed in page reclaim); 8 slots pin 27 GiB, with
  2 GiB of PLE rows and 8 GiB of prefix-cache state (entries are 497 MiB at 262k because of the pooled indexer
  keys). The expert cache drops to 17.2 GiB (pooled keys take 384 MiB of VRAM per slot).

| Long prompts through the production API (wikitext, 64 generated) | 32k | 65k | 131k | 250k |
|---|---|---|---|---|
| prompt tokens | 28,558 | 55,764 | 110,640 | 208,210 |
| prefill tok/s | 438 | 634 | 577 | 515 |
| time to first token | 65 s | 88 s | 192 s | 404 s |
| decode tok/s after it | 67 | (stopped at once) | 74 | 62 |

  Prefill at long context runs at a third of its 4k-prompt speed: the per-query QSA selection and indexer
  scoring grow with the context (65k blocks at 262k). Decode holds 62-74 tok/s at 250k.
- **DeepSeek-V4.1-Flash-1017 (parked).** The user's intended model; sized and planned in
  `docs/deepseek41-plan.md` (IQ1_S fits this box with every expert resident; 25-35 tok/s expected), the llama.cpp
  oracle built (`~/llama.cpp-dsv41`), `tools/pardl.py` written for the 233 GB download. Parked until the machine
  has more storage; Qwen3.8-Flash-Next stays in production.
  **2026-11-03: unparked, target changed to UD-IQ2_XXS** (280 GiB: 146 GiB of experts, 126 GiB Engram table) on
  the new 4 TB NVMe with the win10 VM retired (all 120 GiB of RAM ours). Needs a third expert tier on NVMe plus
  NVMe for Engram rows, parked KV pages and prefix entries: `docs/deepseek41-plan.md` rewritten. Download in
  progress to `~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/`.

- **NVFP4 (nvidia/Qwen3.8-Flash-Next-NVFP4).** The production model moves from UD-IQ3_XXS (files deleted; the
  IQ paths stay supported) to NVIDIA's NVFP4 checkpoint: routed experts NVFP4 (ggml `block_nvfp4`: 64 values in
  36 bytes, four UE4M3 sub-block scales, doubled E2M1 codes) with ModelOpt's per-tensor `weight_scale_2`,
  attention/GDN/shared experts BF16, the PLE table FP8 with one scale. Conversion: llama.cpp PR #30817's
  converter (`~/llama.cpp-dsv41`, `--outtype q8_0 --no-mtp`) repacks the experts losslessly, writes
  `blk.N.ffn_*_exps.scale` (F32 x 512) beside them, Q8_0 for the dense weights and the PLE table (170-byte
  rows), and skips the FP8 block-scaled MTP head it cannot read (the drafter keeps
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`). Engine:
  - Expert records grow a 16-byte tail {gate, up, down, 0} of per-tensor scales (pack `tail_bytes`, header key
    `quixi.experts.tail_bytes`; `HostExperts`, slots and the prefill's pointers all see it). The scales apply
    where ggml applies them: gate and up outputs before the activation, the down output after it.
  - Kernels: `qx_moe_gateup_q8k_nvfp4` / `qx_moe_down_q80_nvfp4` (decode, integer dots on Q8_K / Q8_0
    activations), `qx_moe_gateup_nvfp4` / `qx_moe_down_nvfp4` (exact f32 mode), `gemm_grouped<QX_NVFP4>` with
    the scale in its epilogue (prefill), `decode8<QX_NVFP4>` for the dequant/GEMV paths.
  - CPU pool: `nvfp4_q8k` / `nvfp4_q80` (AVX-512 VNNI; magnitude LUT + sign mask, `vpdpbusd`), scalar
    references, `GateFmt::Nvfp4`. The tiers stay bit-equal: the 16 sub-block values of a 256-chunk are summed
    in one fixed pairwise order (`tree16`: a shuffle tree on the GPU, lane permutes on the CPU), then
    `acc += dq * chunk` in order. `qx-nvfp4-check` (random records, 12 selections x 2560 outputs): GPU and pool
    identical on every output, 8e-3 from an f64 reference (the activation quantization), the f32 path 2e-7.
  - `PleTable` dequantizes by tensor type (Q8_0 now, IQ4_NL before); `qx-ref` applies the scales;
    `default_model_path()` prefers the NVFP4 pack/GGUF (`QX_MODEL` overrides).
  - Found on the way: CubeCL attaches NVRTC compile failures to the buffers a launch binds, and QuixiInfer's
    launches bind none, so a kernel NVRTC rejects (it has no `<cstdint>`; nvcc at build time accepted `int8_t`)
    silently launched nothing. `Module::kernel` now compiles each module with NVRTC once per source hash
    (marker under `~/.cache/quixiinfer/nvrtc-ok`) and panics with the log (`examples/badkernel.rs` shows the
    silence).
  - Validation (GGUF, no pack yet; numbers after the PLE-scale fix below). llama.cpp goldens for the three
    short prompts (`tools/golden/golden-dsv41`, CPU, Q8_0 activations) predict their own next tokens at
    86% / 89% / 64% (the IQ3_XXS goldens: 86 / 87 / 57). The reference in compat mode agrees with them at KLD
    0.9e-3 / 1.5e-3 / 1.7e-3 over the first 8 positions, drifting at later positions as before (ref vs
    llama.cpp at 2560+ was 0.069 for IQ3_XXS: the draft PR's selection differs). The engine against the exact
    reference: with f32 KV and f32 experts KLD 1e-11 at every position (the port is exact); in production
    numerics (FP8 KV, Q8 expert activations) prompt 1 gives KLD 6.2e-4 over 64 positions (top-1 64/64) and
    prompt 0 5.5e-2 over 37 (36/37), where a few positions after position 8 flip expert routing on the
    FP8 KV's 3-5% attention error (one e4m3 scale per 256-dim head: dims far below the head's maximum land in
    subnormals). Per-32-dim KV scales would cut that at ~12% more KV bytes; not done.
  - The prefill path hung on the first NVFP4 prompt: `compute-sanitizer` found `qxp_route` writing past its
    register array. Its top-k loop indexes with the "no candidate" sentinel when every logit is NaN, and the
    logits were NaN because the chunk's embedding gather was hard-coded to Q6_K rows (the IQ3_XXS table's
    type) while the NVFP4 GGUF's table is Q8_0. The gather now goes through `ops::dequant_gather` by type, and
    both route kernels skip the sentinel (a poisoned context otherwise shows only as the staging copy thread
    waiting forever, since CubeCL reports faults on bound buffers only).
  - With that, every exactness check passes on the NVFP4 GGUF: prefill against decode (KL 4.3e-4, the bf16 GEMM),
    speculative (tokens identical), batched (4 slots x 80 positions in mixed windows: 0 rows differ), prefix
    restore (0 of 39 rows differ, tokens identical). The VRAM headroom the cache leaves now grows 0.4 GiB per
    slot (`QX_VRAM_HEADROOM_GIB` overrides): with the Q8_0 dense weights, 4 slots' graphs no longer fit in 1.25 GiB.
  - Speed (GGUF, 16 slots). The experts are 2.64 MiB each (4.5 bits per weight; IQ3_XXS: 1.65 MiB), 63.4 GiB
    pinned, 1.5 GB per token, and 32 GiB of VRAM holds 21% of them (13.6 GiB after the dense Q8_0 weights,
    drafter, KV rings, prefill buffers): hit rates 64% (c=1) to 74% (c=16) against 84-97% before.

| `qx-batch-bench` (256-token prompts, 64 steps) | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| tok/s | 53.5 | 98.3 | 107.9 | 116.5 |
| step ms | 18.7 | 40.7 | 74.2 | 137.4 |
| GPU hit rate | 64% | 73% | 74% | 74% |
| misses per step | 172 | 528 | 1010 | 1985 |
| CPU pool ms per step | 9.8 | 28.1 | 52.3 | 102.7 |

    The pool runs at the RAM's bandwidth (c=16: 1985 x 2.64 MiB in 103 ms = 53 GB/s against 54 measured), so
    throughput is set by the miss bytes; the GPU's part of a c=16 step is 35 ms. The first server sweep at 32k
    (`loadgen.py --unique`, 64 warm-ups) gave 36 / 65 / 73 / 73 tok/s at c=1/4/8/16 with the cache at 11.7 GiB:
    the per-slot headroom of that build forfeited 6 GiB at 16 slots (now 1.5 GiB + 0.15 per slot).
  - RAM at 262k. The service's first start with the NVFP4 experts was OOM-killed: 63.4 GiB of experts beside
    the 27 GiB KV page arena of 8 full contexts, 4 GiB of prefix state and the rest. The arena is now sized
    independently of slots x context (`QX_KV_GIB`, 16 GiB in the unit: about five 262k contexts), and the server
    reserves pages at admission: each running request's growth to prompt + max_tokens is reserved, a newcomer
    needs its whole span (counting what evicting the prefix cache would free) or waits in the queue; one that
    could never fit is refused. A restore no longer evicts the whole prefix cache to make room for a full
    context (it needs one page for the partial-page copy; growth takes pages as it goes).
  - **The GGUF was wrong, and every check agreed with it.** The server's text was random vocabulary. llama.cpp's
    own golden logits on the file had 0% next-token accuracy (the IQ3_XXS goldens: 86%), so the reference's
    KLD 7e-4 against llama.cpp and the engine's 1e-3 against the reference were three implementations agreeing
    on a broken model. Comparing tensors against the HF checkpoint: dense Q8_0 at 0.6% relative error, router
    exact, an NVFP4 expert at 2e-8 after its per-expert scale, norms with their +1, but the PLE table had an
    RMS of 37.5 where the checkpoint's is 0.0075: the converter's `_load_ple_shard` turns the FP8 shards into
    f32 without the table's per-tensor `weight_scale` (0.000199). Fixed in place with `qx-pack scale-q8`
    (every Q8_0 block scale of the 54 GB tensor times the factor, 40 s; rows then within 0.5% of the
    checkpoint) and in the converter (`conversion/qwen4exp.py` in `~/llama.cpp-dsv41`). Lesson kept: a golden
    file's next-token accuracy on its own text is part of the check now, not only parity between
    implementations.
  - Production (`--ctx 262144 --slots 8`, QX_KV_GIB=16, QX_PREFIX_GIB=4, QX_PLE_CACHE_ROWS=10000000): 15.3 GiB
    of expert cache (5861 experts, 24%), 113 GiB of RAM in use, the service enabled.

| Production API, unique prompts, 64 warm-ups (`loadgen.py`) | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| aggregate tok/s | 41 | 71 | 81 | 86 |
| per-request tok/s | 53 | 28 | 16 | 12 |
| TTFT median / p90 (s) | 1.1 / 1.1 | 0.8 / 2.4 | 1.1 / 4.4 | 15.6 / 21.8 |

| Long prompts (wikitext, `tools/longprompt.py`) | 25k | 100k | 189k |
|---|---|---|---|
| prefill tok/s | 299 | 239 | 228 |
| time to first token | 85 s | 417 s | 829 s |

    Against the IQ3_XXS service (67-74 / 119-136 / 119-157 / 146-176 tok/s; prefill 438-577 tok/s): decode is
    RAM-bandwidth-bound at a quarter of the experts resident, and prefill stages 1.6x the bytes per chunk over
    PCIe. A 25k prompt repeated right after the 189k one got no prefix hit: at 262k an entry's state is 497 MiB
    (the pooled indexer keys are sized by the context), so 4 GiB holds 8 entries, and a long prompt's chunk
    snapshots cycled them out. Chunk snapshots now happen only while the cache has a free entry (the prompt's
    end is always saved): the repeated 25k prompt then restores in 0.3 s instead of 86 s. Sizing entries by
    their position rather than the context would make long-prompt hits common (a 25k entry would be ~50 MiB).
  - A request without `max_tokens` used to get 4096: a thinking model spent all of it in `reasoning_content`
    on "implement A* in Rust" and returned no answer (`finish_reason: length`). The default is now what the
    prompt leaves of the context; `chat_template_kwargs.enable_thinking: false` turns thinking off per request.

- **Pack, finer KV scales, prefix entries by position (2026-10-25).**
  - The safetensors were deleted (user's call) and the pack written from the corrected GGUF with the saved
    routing profile: `Qwen3.8-Flash-Next-NVFP4.qxp`, 3 min 12 s, round trip byte-equal. Experts load in 7-12 s
    instead of 75 s. `qx-ref` and `qx-check` still need the GGUF (`QX_MODEL=...gguf`): the reference reads
    tensors as contiguous slices and a pack's expert records are padded.
  - FP8 KV: eight f32 scales per (cell, kv head), one per 32 of the 256 dims (`QX_KV_SG`, `SCALE_GROUPS`),
    in the host pages, the VRAM rings, both append kernels (a 16-lane shuffle max per group), the ring fill
    and the three attention consumers (the V scales stay in memory in `attn_fp8_full`: its three
    instantiations share one kernel's static shared memory and 8 KiB each passed the 99 KiB limit). A KV cell
    grows from 520 to 576 bytes per head (+11%). Against the exact reference on prompt 0 (37 positions): KLD
    mean 5.5e-2 -> 4.4e-2, the worst early position 0.42 -> 0.064, top-1 36/37 -> 37/37; a few positions
    still flip expert routing on E4M3's own 3-bit mantissa, so the remaining lever would be bf16 keys (twice
    the K bytes). A first version kept the V scales in memory to fit the 99 KiB static shared memory shared by
    the kernel's three instantiations, and prefill lost 25% to the per-element scale reads from host pages;
    each heads-per-kv-head count now has its own entry (`qxp_attn_fp8_g8/12/16`) with the scales in shared memory.
  - Prefix entries are sized by their position: the pooled indexer keys (1.5 KiB per 1k tokens across the 12
    attention layers) are saved for `pos / 4` rows only, on top of a fixed 113 MiB of recurrent state; the
    arena is a coalescing first-fit free list instead of fixed 497 MiB slots, evicting least recently used
    entries until a snapshot fits. At 262k, 4 GiB holds ~30 short-prompt entries instead of 8.
  - Checks on the pack: windows 0 rows differ, prefix restore 0 of 39 rows differ, batched 4 slots 0 rows
    differ, speculative tokens identical; with the correct PLE table the MTP drafter keeps 21 of 23 drafts
    (2.9-3.2 tokens per window; the broken table had it at 0).

- **Prefill: the KV mirror, deterministic selection, 2048-token chunks (2026-10-25/26).**
  - Profile of an 8k-token prefill (nsys): `qxp_attn_fp8` was 70% of GPU time (109 ms per layer-chunk), the
    grouped expert GEMMs 12%, the dense GEMMs 11%. Each of a chunk's 1024 queries read its 2048 selected cells
    from the host pages (only the latest 4096 cells live in the VRAM ring): ~2 GB per layer per chunk over
    PCIe, the same cells over and over. `qxp_kv_mirror` now copies the layer's cells into a VRAM mirror laid
    out as a ring larger than the context (0.3 GiB at 262k, one shared by all layers) before the chunk's
    attention, which then reads VRAM only. Bit-identical logits; 9k-token prompt 318 -> 715 tok/s.
  - While proving that, two identical prefill runs gave different logits past 2048 context cells: both QSA
    selection kernels (`qxp_qsa_select`, decode's `qx_qsa_select`) placed the selected blocks with an atomic
    counter, so the attention summed the same cells in a different order each run. Slots are now a prefix sum
    in block order; runs are bit-identical (checksummed).
  - Chunks of 4096 tokens deadlocked: the PLE ring holds 2048 positions and a chunk waits for all of its
    embeddings at once. Chunks are capped at `PLE_CAP`; the default is now 2048 (25k-token prompt: 696 tok/s
    at 1024, 1085 at 2048, engine level with a small expert cache; 206 before the mirror).

- **Production after all of the above (pack, 262k, 8 slots, 2048-token chunks).** Cache 13.9 GiB (5341 experts:
  the prefill buffers grew 1 GiB with the chunk and the mirror takes 0.3). Earlier NVFP4 throughput figures were
  measured on the broken-PLE file (its garbage text ended early and routed differently), so these are the first
  valid ones:

| Production API, unique prompts, 64 warm-ups | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| aggregate tok/s | 48 | 68 | 67 | 67 |
| per-request tok/s | 62 | 19 | 9.4 | 9.4 |
| TTFT median / p90 (s) | 1.2 / 1.3 | 1.3 / 2.6 | 1.9 / 6.3 | 32 / 36 (8 slots: half queue) |

| Long prompts | 25k | 100k | 189k |
|---|---|---|---|
| prefill tok/s | 1036 | 1109 | 1255 |
| time to first token | 24.5 s | 90 s | 151 s (was 829) |

  Startup from the pack: 91 s to listening (63 s pinning the experts in the service, 7-12 s in the check tools:
  not yet explained). The repeated 25k prompt after the 189k one missed the prefix cache (the long prompt's
  chunk snapshots filled the 4 GiB); decode at c>=4 is flat at 67 tok/s, bound by RAM bandwidth on misses.

- **Against Strata (2026-10-26):** root-cause analysis and the changes it led to in
  [docs/strata-rca.md](strata-rca.md): prompt buffers lent from the expert cache (+1,000 experts), int8
  activations for the Q8_0 matvecs, a larger swap budget, slot-grouped resident experts. Production API
  50 / 80 / 81 / 78 tok/s at c=1/4/8/16 (was 48 / 68 / 67 / 67); one 64k slot 95 tok/s on a 400-token answer
  (Strata publishes 114-116 for the same model). The CPU pool runs at the RAM's bandwidth (57 GB/s shared with any
  copy stream, `qx-membw`), so the expert hit rate decides decode speed.

- **Kernel perf pass: profile first (2026-10-26).** Method from QuixiCore-CUDA's `perf/perf.md` (profile, one
  factor at a time, keep only counter-backed wins).
  - 16k-token prefill (nsys, check tool): kernels busy 8.6 of 14 s; the copy engine moved 610 GB at 50 GB/s
    (the whole expert set once per 2048-token chunk). `PLE_CAP` (the PLE ring, which capped the chunk) is now 8192:
    1,160 / 1,713 / 2,145 tok/s at 2048 / 4096 / 8192-token chunks, logits bit-identical across chunk sizes.
  - 90k-token prefill: kernel-bound, chunk time growing with context. `qxp_qsa_select` was 39% of GPU time
    (20.5 s): the indexer scores (4 heads x 128 dims against one pooled key per 4 cells, ~12 TFLOP for the prompt)
    ran as one scalar thread per block with uncoalesced key reads, ~0.6 TFLOP/s. New `qxp_idx_score` computes them
    as a tiled product (64 x 64 tiles, 4 x 4 per thread, same fma chain per score) and the selection kernel skips
    its own scoring (`QX_IDX_TILED=0` restores it): 20.5 s -> 0.58 s, logits bit-identical, 90k prefill
    1,610 -> 2,471 tok/s (check tool, 8192-token chunks).
  - Remaining 90k prefill GPU time: dense GEMM 27%, expert GEMM 25% (both ~50 TFLOP/s on bf16 mma), GDN recurrence
    21% (token-serial), attention 9%.
  - CPU pool (`qx-rambw`, huge-page records): raw AVX-512 read 56 GB/s with 16 cores (50 from one CCD, 56 with SMT),
    the NVFP4 pool 54 GB/s (96%); SMT makes the pool slower (50.5). No headroom left in the pool kernel; only faster
    RAM (the DIMMs run JEDEC 5600 CL46) would move it.
  - Production layout (8 x 262k, loadgen --unique --warmup 64, 3 rounds; then a 90k prompt):

| chunk | c=1 | c=4 | c=8 | c=16 | 90k prompt prefill | decode after it |
|---|---|---|---|---|---|---|
| 2048 (before the tiled scores) | | | | | 903 tok/s | 115 |
| 2048 | 52.9 | 81.6 | 75.8 | 78.4 | 1,120 tok/s | 93 |
| 8192 | 45.8 | 79.3 | 75.3 | 75.0 | 1,691 tok/s | 74 |

    8192-token chunks lend 8.7 GiB of the expert cache to every prompt, short ones too, and decode pays for it,
    so the default stays 2048 (`QX_PREFILL_CHUNK=8192` for long-prompt work). Next: pick the chunk (and the lent
    bytes) per prompt.
  - Found, not caused by this: prefill against decode at 8k positions has KL 2.8e-2 (top-1 9/9), the same on the
    old path; the 4.3e-4 recorded earlier was at short context.

- **Prefill follow-ups (2026-10-26, continued).**
  - Two tiers of prompt buffers, both carved from the expert cache's tail: 2048-token chunks (2.67 GiB lent) for
    short prompts, 8192-token chunks (8.73 GiB, `QX_PREFILL_BIG`, 0 disables) when a prompt has at least that
    many tokens left. `prefill_tier` swaps them (giving back the other's lent tail; the staging counters carry
    over), `prefill_lend` lends the current tier's region. The worker no longer shrinks chunks to
    `--prefill-budget-ms` when no other request is decoding: every chunk streams the experts the cache lacks
    whatever its length, so the rate-sized chunks got smaller and slower in turn.
  - Found while measuring it: after a prompt the engine handed the swapper only the last call's routing counts,
    added to its scores at full weight. An 8192-token chunk's counts pulled the cache toward the prompt's
    experts (decode after a 90k prompt 72 tok/s against 118 with 2048-token chunks). Counts from a call longer
    than a small chunk are now scaled to a small chunk's worth: 127 tok/s after the same prompt.
  - `qx_gemm2` / `qx_gemm_grouped2` (`QX_GEMM2=0` for the old ones): two shared-memory stages with the next k
    step loaded while the current one's mmas run, ldmatrix fragments. Same bits (logits identical);
    `qx-kbench gemm`: +7-14% on wide shapes (qkv 10240x2560 at 8192 tokens 96 -> 110 TFLOP/s), +40-75% on the
    narrow long-K ones (hc down 320x10240); +3.3% on a 32k prefill.
  - GDN prefill in chunks of 32 tokens (the chunked gated delta rule, `QX_GDN`: 0 token by token, 1 one
    kernel, 2 two passes, the default): `qxp_gdn_prep` computes each chunk's (I + A)^-1 products
    (U, W, P, g) for all chunks and heads at once, `qxp_gdn_scan` walks the chunks with products against the
    state; run over 2048-token slices (113 MB of scratch). `qx-kbench gdn` (8192 tokens, 48 heads): token by
    token 16.9 ms, one kernel 10.6, two passes 9.1; outputs and state within 3e-7 of token by token. ncu: the
    scan is bound by shared-memory instructions and shuffles at 1 block per SM (0.56 waves); 16-column splits
    did not help. 90k prefill 2,548 -> 2,795 tok/s; logits KL against token by token 2e-6 / 1.5e-4.
  - Check tool, 90k tokens, 8192-token chunks: 1,610 (before the tiled scores) -> 2,795 tok/s.
  - Production layout (8 x 262k, loadgen --unique --warmup 64): c=1/4/8/16 51.7 / 81.8 / 74.5 / 74.6 against
    54.0 / 82.9 / 79.6 / 79.4 with the small tier only (short prompts do not reach the changed code; c=8/16
    vary ~5% run to run). A 90k-token prompt alone: TTFT 43.4 s (2,069 tok/s; 76.5 s at the start of the day),
    34.5 s with `QX_SPEC=0`: the MTP drafter catches up on the prompt in decode-sized windows (`draft_catch_up`,
    WMAX tokens per `draft_forward`), 9 s of the 43. Next: the drafter's layer through the batched prefill.
    23k-token prompt through the service: 1,495 tok/s (about 1,070 before).

- **Decode at c=4-16: the batched products (2026-10-26).** Per-level server stats (`QX_SCHED_STATS`) and an nsys
  profile of `qx-batch-bench` at B=8: an 86.7 ms step was 52.5 ms of the GPU waiting for the CPU pool and 33.3 ms
  of GPU work outside it, run while the pool idles; 15.7 ms of that was `qx_tmv` (the batched dense products, bf16
  tensor cores), at about 470 GB/s whatever the token count: 51 KB of static shared memory left one 8-warp block
  per SM (12.9% occupancy), and the inputs came through dependent generic loads before each mma.
  - Planar Q8_0 segments now take int8 inputs (`qx_mgemv_q8x`, as the one-token path already does; scales stored
    transposed for `qx_tmv`) and one int8 mma (m16n8k32) per Q8_0 block, its quants loaded straight into the
    fragments with the k order permuted alike on both sides (exact integer dots; 8-byte row loads), the next
    chunk's weights in flight during the current chunk's mmas, the chunk's inputs batched through the read-only
    path, the four scales of a row in one load. One entry per count of 8-token tiles (`qx_tmv_q8_1` .. `_4`, own
    registers), no bf16 staging tiles. The other types of a group (the router's f32 rows, f32 and bf16 segments)
    run in `qx_mgemv` beside it, so the router's logits are a token's solo bits (it went through bf16 before).
    `QX_TMV_INT8=0` restores the bf16 kernel.
  - `qx-tmv-check`: planar Q8_0 within 1.9e-7 of `qx_mgemv` (bf16: 7e-3), router 0. Kernel time at 8 tokens
    (nsys, `qx-kbench`): gdn in 90.7 -> 31.5 us, ssm_out-sized 37.9 -> 12.7, head (675 MB, from DRAM) 1,245 -> 617.
    Tried and dropped: 3-4 blocks per SM (spills), two chunks of weights ahead, inputs prefetched a chunk ahead.
  - Batched decode, B=8: products 15.85 -> 6.36 ms per step, GPU work outside the pool wait 33.6 -> 23.5 ms.
    `qx-batch-check` (8 slots): batched against solo KL mean 6.07e-2 -> 2.94e-2 (top-1 578 -> 593 of 640); with one
    kernel for every window (`QX_TMV=0` or `1`) still 0 rows differ. Against the CPU reference (`qx-check`, 48
    positions, every product through the int8 kernel): KL 5.21e-2 against 5.52e-2 for `qx_mgemv`, top-1 47/48 both.

| Production layout (8 x 262k, loadgen --unique, 3 rounds per level) | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| bf16 `qx_tmv` | 56.2 | 81.7 | 78.7 | 75.3 |
| int8 `qx_tmv_q8` | 53.5 | 82.7 | 86.7 | 85.7 |

  c=1 does not reach the changed code (single-sequence windows keep `qx_mgemv`); its spread run to run is about 5%.
  The c=8 step is still about 64 ms of CPU pool (expert misses at the RAM's bandwidth) and ~14 ms of GPU work in
  series with it. Next: overlap them (two half-batches, one's GPU layers while the other's misses run).

- **Why c=8 and c=16 barely beat c=4 (2026-10-26).** `qx-batch-bench` (256-token prompts, 96 steps, B in order
  1, 2, 4, 8, 16; cache 7706 experts): 64.5 / 101.9 / 135.4 / 119.1 / 122.4 tok/s; hit rate 75 / 82 / 82 / 76 / 74%;
  misses per token 119 / 87 / 87 / 115 / 127; CPU pool 6.5 / 9.7 / 18.9 / 54.8 / 107.4 ms of 15.5 / 19.6 / 29.5 /
  67.2 / 130.7 ms steps, at ~54 us per miss (the RAM's bandwidth). Misses are not shared between sequences (10 of
  512 per layer each), so the pool's cost is per token, and from 8 sequences on their working sets outgrow the
  cache: misses per token rise and cancel what batching saves on the GPU. A frozen cache (`QX_SWAPS=0`) falls to
  51-69% hits, half-lives 64 / 2048 change B=8 by under 2%. c=16 equals c=8 in the service because it has 8 slots:
  the c=16 requests queued a median 23 s. Levers: VRAM for the cache, faster RAM, overlap of the ~12 ms of GPU work.

- **Split windows: half the batch's GPU work under the other half's CPU experts (2026-10-27).** A window of several
  sequences (from `QX_SPLIT` tokens, default 4; 0 off) runs as two halves of whole segments, interleaved layer by
  layer on the one stream (`Engine::launch_split`): half A's front of layer l posts its misses, then half B's back of
  l - 1 and front of l run while the pool computes A's, then A's back of l. The second half has its own window
  buffers, product groups and MoE mailbox (`Shared::moe2`, its own job words); the driver answers the two
  mailboxes alternately. The slots' PLE history is shared (a view). One captured graph per width.
  - `qx-batch-check` (8 slots, `QX_TMV=0`): 0 rows differ, generation identical.
  - `qx-batch-bench` (B order 1, 2, 4, 8, 16, 96 steps), off -> on: 64.3 / 101.4 / 134.7 / 131.0 / 122.4 ->
    62.4 / 100.4 / 140.7 / 151.1 / 139.4 tok/s; B=8 step 61.1 -> 53.0 ms against 50.1 ms of pool time: decode now
    waits on the CPU almost only.

| Production layout (8 x 262k, loadgen --unique, 3 rounds per level) | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| int8 `qx_tmv_q8` | 53.5 | 82.7 | 86.7 | 85.7 |
| split windows | 54.2 | 90.0 | 95.2 | 90.1 |

  The win10 VM was running during c=1 and part of c=4 of the split run (and during every earlier run). c=16 still
  queues (median TTFT 22 s) with 8 slots. `--slots 16` was tried and dropped: every slot holds about 0.5 GiB of VRAM
  (the indexer's key pool for the full context, f32, plus graph headroom), so the cache fell 6,370 -> 4,226 experts
  and c=4 / c=8 to 62.6 / 62.8 tok/s.

- **Cache-aware routing, opt-in (2026-10-27).** `QX_ROUTE_CACHE=J,M` (`qx_moe_route`): each token keeps its top J
  experts and fills the other 10 - J picks from its top M, the ones the VRAM cache holds first (in rank order), then
  the best of the rest; weights renormalised over the picks. Off by default; the off path is the old kernel's bits.
  Prompted by colibri's `route_select`. Hit-rate context first: on a 4,096-token routing trace, a simulation of the
  swapper gives 86.4% at 6,370 experts against 86.3% for a per-layer LRU of the same size and 87.9% for a global LRU
  (colibri's ~90% reproduces), so the live 70-80% comes from short requests on fresh topics, not the policy.
  - Quality (`qx-ppl`, 4 texts x 512 tokens through the decode path, cache as in production, against exact routing):

| `QX_ROUTE_CACHE` | ppl | hit rate | KL from exact | same top-1 |
|---|---|---|---|---|
| off | 4.670 | 69.4% | | |
| 4,12 | 4.792 (+2.6%) | 77.3% | 5.3e-2 | 91.5% |
| 2,12 | 4.799 (+2.8%) | 77.7% | 6.0e-2 | 92.3% |
| 2,16 | 4.989 (+6.8%) | 86.1% | 1.25e-1 | 87.8% |

  - Throughput (`qx-batch-bench`, B 1 / 4 / 8 / 16, tok/s): off 59.8 / 136.9 / 149.6 / 140.2; 2,12 72.0 / 137.1 /
    186.2 / 208.6; 2,16 79.8 / 145.9 / 214.1 / 298.7. Misses per step at B=16: 2,055 -> 1,383 -> 743.
  - Not in production: it trades accuracy for speed, a decision for the operator.

- **Why cache-aware routing gained nothing at B=4 (2026-10-27).** It does cut the misses there (warm: 248 -> 191 per
  step, pool 13.5 -> 10.4 ms), but the split B=4 step is GPU-bound: 27 ms against 10-14 ms of pool time, because two
  halves of 2 tokens read every dense weight twice. `qx-batch-bench` 4,4,4,8 (third B=4 run, warm; win10 VM up):

| | split | not split |
|---|---|---|
| exact routing, B=4 | 148.2 tok/s (27.0 ms) | 162.6 (24.6 ms) |
| 2,12, B=4 | 149.3 (26.8 ms) | 190.5 (21.0 ms) |
| exact routing, B=8 | 163.1 (49.1 ms) | 150.5 (53.1 ms) |
| 2,12, B=8 | 196.6 (40.7 ms) | 202.4 (39.5 ms) |

  The split pays only while the pool's time exceeds the halves' GPU time; with a warm cache at B=4, or with
  cache-aware routing at B=8, it costs. The earlier B=4 gain (+4%) was with a colder cache. Next: choose split per
  width from measured step times instead of a fixed `QX_SPLIT`.

- **Split chosen per width by measurement (2026-10-28).** `QX_SPLIT` defaults to `auto` (`SplitAuto`): each width
  runs blocks of 8 windows in one mode, the first interval of a block dropped (the async pipeline still holds the
  other mode's window) and intervals over 2x the estimate (3x the other mode's, for a fresh one) dropped as other
  work; the faster mode runs and the other gets a fresh block after 4 blocks, the gap doubling to 32 while it keeps
  losing. `QX_SPLIT=N` forces the split from N tokens, 0 never; `QX_SPLIT_DEBUG` traces the choices.
  - `qx-batch-bench` with `QX_ROUTE_CACHE=2,12` (win10 VM up, runs noisy as hit rates vary): B=4 warm auto 172.6 /
    150.6 against always-split 148.5 / 148.4 and never 161.2 / 157.8; B=16 auto 216.6 / 227.0, split 240.8 / 228.8,
    never 186.0 / 192.4. `qx-batch-check` (8 slots, `QX_TMV=0`): 0 rows differ.
  - Server (production layout, exact routing, VM up): 53.0 / 92.2 / 95.1 / 93.6 tok/s at c=1/4/8/16 against
    54.2 / 90.0 / 95.2 / 90.1 with the fixed split. Deployed.

- **Cache-aware routing, more settings (2026-10-28).** `qx-ppl` as before (one run each, about +-1 point of hit
  rate run to run; win10 VM up):

| `QX_ROUTE_CACHE` | ppl | hit rate | KL from exact | same top-1 |
|---|---|---|---|---|
| off | 4.670 | 70.7% | | |
| 8,32 | 4.808 (+3.0%) | 78.7% | 4.8e-2 | 91.9% |
| **6,16** | 4.812 (+3.0%) | 82.6% | 6.9e-2 | 91.1% |
| 6,20 | 4.891 (+4.7%) | 84.2% | 7.5e-2 | 90.9% |
| 3,14 | 4.905 (+5.0%) | 83.7% | 8.9e-2 | 89.8% |
| **3,16** | 4.891 (+4.7%) | 86.6% | 1.13e-1 | 89.2% |
| 4,16 | 4.911 (+5.2%) | 85.6% | 9.7e-2 | 89.5% |
| 2,14 | 4.943 (+5.8%) | 84.1% | 1.04e-1 | 89.3% |

  With 2,12 (4.799, 77.7%) and 2,16 (4.989, 86.1%) from before, the frontier is 6,16 (2,12's cost, 5 points more
  hits) and 3,16 (2,16's hits at two thirds of its cost); keeping more true experts and widening M beats keeping 2.

  Throughput, `qx-batch-bench p100k.tokens 256 96 1,4,8,16` (auto split, production cache, win10 VM up, all five
  in one sitting), tok/s and hit rate at B = 1 / 4 / 8 / 16:

| `QX_ROUTE_CACHE` | tok/s | hits % |
|---|---|---|
| off | 64.6 / 132.3 / 160.4 / 134.0 | 76 / 80 / 78 / 73 |
| 2,12 | 73.1 / 145.5 / 189.0 / 204.4 | 85 / 85 / 84 / 83 |
| 6,16 | 77.6 / 160.0 / 204.9 / 241.4 | 88 / 88 / 86 / 86 |
| 3,16 | 81.3 / 165.6 / 237.9 / 296.0 | 92 / 89 / 91 / 91 |
| 2,16 | 80.0 / 173.6 / 242.3 / 300.5 | 93 / 92 / 91 / 91 |

  3,16 matches 2,16 within noise at B >= 8 (+48% / +121% over off) for less quality cost; 6,16 beats 2,12 at the
  same cost. The bench's hit rates run above `qx-ppl`'s (long decodes of one text per slot).

- **Pooled indexer keys in pages (2026-10-28).** The indexer's pooled block keys (f32, 1.5 MiB per 1k tokens over
  the 12 attention layers) were a flat buffer of the full context per slot: 384 MiB a slot at 262k, 3 GiB at 8
  slots, 6 GiB at 16. They now live in VRAM pages paired with the host KV pages (`KvPages::kp`, a second page
  table per slot, `kp_row` in kernels/include/qx_kv.cuh), so they cost what the KV arena holds: 4854 pages x
  384 KiB = 1.78 GiB with `QX_KV_GIB=16`, whatever the slot count. Prefix entries share them with the KV pages
  (a restored partial last page copies its kp page too) and no longer carry their own copy: 113 MiB an entry
  instead of up to 497. Exact: `qx-ppl` over 6144 tokens writes byte-identical log-probs to the previous build,
  `qx-prefill-check 4000` gives the previous build's numbers digit for digit, `qx-prefix-check 3000 300` and
  `qx-spec-check 2500 64` pass. (A 4 x 2400 `qx-batch-check` was cut short by host memory pressure.)

- **DeepSeek-V4.1-Flash UD-IQ2_XXS, D0-D1 groundwork (2026-11-03).** Target and machine changes in
  `docs/deepseek41-plan.md`. The Qwen service is stopped (user's instruction) so the GPU and RAM are free for this.
  - Downloaded (280 GiB, `tools/pardl.py`). `tools/gguf_layout.py` reads the shard headers alone: 146.1 GiB of
    routed experts in 7.3-11.5 MiB records (gate/up IQ2_S x15 layers, IQ2_XXS x15, IQ1_S x10; down IQ3_XXS x15,
    IQ2_S x15, IQ2_XS x6, IQ2_XXS x3, Q2_K x1), the 125.9 GiB Engram table (768M Q5_K rows of 176 B), 7.6 GiB
    dense.
  - Block decoders for IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, Q2_K and Q5_K (`qx-quant`, grids via `tools/gen_iq_tables.py`
    plus `KSIGNS`; GPU `decode8` twins in `qx_quant.cuh`): `tools/dequant_check.py` finds them identical to gguf-py
    on real tensors of the shards (max diff 0).
  - CPU pool: `GateFmt` became `ExpertFmt { gate, down, clamp }` with per-part `WType`s, the clamped SwiGLU
    (`silu(min(g, c)) * clamp(u, -c, c)`) and Q8_K hidden activations for 256-block down formats; AVX-512 VNNI dots
    for the five new types bit-equal to their scalar forms (IQ1_S's +-1/8 offset folded into the integer sum).
    GPU: `gateup_q8k_t<T, 20>` (5120-wide inputs) and `down_q8k_t<T>` (Q8_K h quantized in shared memory by the
    pool's rule). `qx-iq-check` (5120 x 2304, clamp 10, every gate/down pair the model uses): 0 of 61,440 outputs
    differ between GPU and pool per pair; 7e-8 to 1e-4 from an f64 reference on the rounded activations.
  - Reference: `crates/qx-ref/src/dsv41.rs`, the whole forward pass (Engram, single-pass mHC with Sinkhorn, MLA
    window attention with compressed latents, indexer with candidate blocks, YaRN RoPE in NORM mode, sqrt-softplus
    router, clamped SwiGLU), `qx-ref check` dispatching on `general.architecture`, activation names as the oracle's.
  - Oracle tooling: `tools/golden/route_trace.cpp` records the selected experts per (token, layer) through the
    eval callback (the top-k is a strided view); `tools/cache_sim3.py` simulates VRAM / RAM / NVMe tiers with the
    decode cost model; the offline tools now link with DT_RPATH so the surviving `~/git/llama.cpp-dsv41/build`
    resolves. Goldens must be recorded with `CUDA_VISIBLE_DEVICES=` (otherwise the oracle runs some ops on CUDA0).
  - Lesson: the oracle run, the trace and a Qwen pack load must not share the 120 GiB; together they OOMed the box.
  - **D1 result.** Against CPU-only goldens of the three short prompts (125 positions): layer 0's attention matches
    the oracle to 2e-7 once two ggml details are mirrored (the Q8_0 activation quantizer multiplies by `127 / amax`,
    not `1 / d`; RMS norm sums f32 squares in f64), and the FFN math replicated in Python on the oracle's own input
    matches to 1.4e-7. Everything after that is rounding flips: a 3e-5 input difference flips a dozen Q8 decisions
    and the 2-bit experts amplify them layer by layer, so positions 0-1 (BOS, first token) differ by KLD 1-2 even
    between two rounding modes of the reference itself. Excluding them: golden vs reference KLD mean 0.055, median
    0.004, top-1 93-95%; the reference against itself in two rounding modes 0.074 / 0.0034 / 95.8%. The reference
    is within the oracle's own noise floor; this model at UD-IQ2_XXS is far noisier than Qwen NVFP4 was (1e-3).
    `QX_REF_DUMP` / `QX_REF_DUMP_ACTS` write the reference's logits and activations for such comparisons.
  - **D0b result.** Routing trace of 69k mixed tokens; `tools/cache_sim3.py` with 20 GiB of VRAM and 88 GiB of RAM
    for experts: 74.4% of routings in VRAM, 25.0% in RAM (580 MiB per token for the pool), 0.6% on NVMe (14 MiB);
    11.7 ms of RAM time and 2.3 ms of drive time per token, so about 50 tok/s single-stream is the target. Robust
    to the budget split and the decay (44-50 across the variants); see the plan's table.
  - **D2 done.** `qx-pack pack` wrote the 280 GiB pack unchanged (per-layer record sizes and the 176-byte Q5_K
    Engram rows in 4 KiB pages were already supported) in about 2 minutes; `qx-pack verify` finds every tensor
    byte-equal. The verifier's expert comparison now reads 16 records at a time: a layer per rayon thread was
    118 GB of buffers on this model and OOMed the machine. New `tools/capped.sh` runs a command in a memory-capped
    systemd scope; every heavy job goes through it from now on.
  - **Tier store** (`crates/qx-engine/src/tiers.rs`, `qx-tier-check`): pinned RAM places per record-size class
    filled from the pack by rank (36 GiB in 3.1 s), misses read by io_uring O_DIRECT into staging places, promotion
    and eviction. On the pack: 17.7 GB/s for 48-record batches (577 us per 9.7 MiB record), 10.5 GB/s for 1-4
    record batches (2.5 ms per batch), every record byte-equal to the GGUF. The drive is faster than the plan's
    12 GB/s. The pack's header carried Qwen's calibration ranking (`qx-pack pack` defaulted to it for every model;
    now only for `qwen4exp`); readers ignore a ranking of the wrong size.
  - **D3 started: `crates/qx-engine/src/dsv41/` and `kernels/dsv41.cu`.** The one-token decode path with eager
    launches (hyper-connections with Sinkhorn, YaRN RoPE in NORM mode, the compressor and index keys, the indexer
    with candidate blocks and a radix top-k, attention over the window and the selected latents with sinks, the
    Engram gate, the sqrt-softplus router, the clamped shared expert), the routed experts on the CPU pool from
    the GGUF for now. `qx-ds-check` compares it with the reference op by op: in f32 mode (`QX_MGEMV_INT8=0`,
    `QX_EXACT_EXPERTS=1`, `QX_REF_COMPAT=0`) every activation of position 0 is within 1e-6 after two fixes (the
    fused GEMV silently returned zeros for the 24 x 20480 F32 mix matrix, now its own kernel; the GEMV dispatch
    lacked the new type ids, so Q5_K shared experts came out zero; the indexer's head weights lacked the
    1/sqrt(heads x dim) factor). Over the first 6 positions of prompt p0 (ratio-2 layers with up to 3 compressed
    rows, ratio-1 layers with up to 6, the candidate pin active): all 816-822 activations per position within
    1.3e-5, top-1 6/6, KLD 0.000000 against the reference.
  - **Tiered path** (`DsEngine::load_tiered`, `qx-ds-check` with `QX_DS_PACK`): dense weights from the pack, the
    hottest 20 GiB of expert records in VRAM by the trace's ranking (`golden/profile-dsv41.txt`; the top 2,113
    records cover 52% of routings, the top 11,348 cover 97.6%), 60 GiB pinned next, the rest on NVMe, filled in 7 s.
    Resident experts run the GPU integer kernels on Q8_K activations quantized on the device, the others go
    through the store to the pool: over 12 positions the output is **bit-identical** to the all-pool path, as the
    tiers are built to be. In production numerics (pool experts, int8 dense GEMV) the 12 positions give top-1
    11/12 and KLD 0.40 against the reference, 9/12 and 0.12 with the dense products in f32: both within the
    rounding-flip floor measured for the reference against the oracle. The reference's `experts` mode now rounds
    the down projection's input to Q8_K for 256-block formats (the DeepSeek pool does); a new `engine` mode mirrors
    the int8 GEMV's activation rounding too. The remaining 1e-3 differences are rounding flips: `QX_DS_TRACE`
    counts how many Q8_K values of each layer's expert input the two sides round differently; at position 0 the
    first three layers have 0 flips and match to 5e-7, layer 3 has 17 and its output is off by 5e-3, and from
    there the flips compound (81, 298, 531 at position 1). The experts' Q8_K quantization makes bit-level agreement
    past the first layers unreachable for two implementations that differ in f32 summation order; the f32 mode
    is where the kernels are checked.
  - **First end-to-end speed** (`QX_DS_BENCH`, host-driven path: a sync and a host round trip per layer, the
    experts dispatched by the host, promotions between tokens): prompt p1 (59 tokens) then 64 greedy tokens,
    20 GiB VRAM / 60 GiB RAM, 4 promotions per token into each tier: **108 ms per token, 9.2 tok/s**, 79 ms of it
    in the expert phase. The tiers adapt as the simulation said but slowly at 4 promotions per token: the first 8
    tokens route 21% / 41% / 38% to VRAM / RAM / NVMe, tokens 111-119 route 58% / 29% / 13%; an NVMe miss costs
    about 0.6 ms, so the early tokens are drive-bound. Promotions cost 0.7 ms each (5.8 ms per token at 8).
    With 16 promotions per token into each tier: 92 ms per token, 10.8 tok/s; by tokens 111-122 the routings go
    82% / 13% / 4% and the expert phase is 24-33 ms, but the promotions themselves cost 22 ms per token on the host
    (0.7 ms per 10 MiB copy) and the 40 per-layer host round trips the rest. Over all 59 positions of p1 with the
    tiers adapting, the reference agreement is top-1 57/59, KLD mean 0.14 (3.6 at position 0, mostly below 1e-2
    from position 8 on); the generated tokens are identical whatever the tiers hold.
    Next: the asynchronous pipeline (mailbox-driven pool on a driver thread, promotions on its copy stream, no
    per-layer sync) that the Qwen engine has; then the prompt's routing seeding the counts and NVMe lookahead.
  - **Asynchronous pipeline, and a race found by bisection.** The driver thread (`dsv41/driver.rs`) serves the
    mailbox (post, pool, done) and promotes between tokens on its own copy stream; the token runs with no host
    round trip. Its first benches were nondeterministic: two identical runs with promotions disabled diverged
    after 7 tokens, and both differed from the host-driven path from the first generated token, while a check
    with a sync after every op matched the reference exactly. A sync-point knob (`QX_DS_SYNC=post|wait|combine|
    layer|engram`) showed that *any* sync restores the host-driven sequence, even one only at the two Engram
    layers, so the fault was the host outrunning the device, not the mailbox protocol. Cause: the Engram rows
    gathered on the host went up with a synchronous `cuMemcpyHtoD`, which runs on the legacy stream and is not
    ordered after the kernels CubeCL has queued on its non-blocking stream; with the host far ahead, layer 14's
    rows landed before layer 1's Engram GEMV had read the buffer. Fix: the rows go into a pinned, device-mapped
    buffer with one region per Engram layer, and a copy kernel (`qx_ds_copy_f32`) moves them in stream order.
    After the fix two free-running asynchronous runs give the host-driven tokens exactly. Rule, now in
    `qx-gpu`'s `mem.rs` doc: a host copy into memory a queued kernel reads needs a sync first or a stream-ordered
    path; the Qwen engine's uploads sit at admission points that already sync.
  - **Where a steady-state token went** (positions 119-122 of p1 + 64, 16 promotions per token): 20 ms of GPU
    dense + attention + launches (`QX_DS_NOEXPERTS=1` measures the token without routed experts: 20 ms at
    position 7, 21 at 122, i.e. 50 tok/s is the ceiling before the experts), 30 ms of pool, 21 ms of promotions
    blocking the driver at the token boundary (a stream sync for the VRAM copies, then 16 synchronous NVMe reads
    into RAM); 64-68 ms in all. The promotions are now off the path: the VRAM copies and their table writes stay
    stream-ordered behind an event (the table entry lands after its record, so the next token reads it at once),
    the RAM places being copied are locked against eviction until the event completes, and the NVMe-to-RAM
    promotions trickle one per idle poll of the driver loop between jobs. Steady-state tokens: 52-57 ms
    (16 or 32 promotions per token), 75.9 ms averaged over the 64 generated tokens with the cold start (13.2
    tok/s); tokens unchanged. Left: the pool's 29 ms per token (about 60 expert evaluations: 48 from RAM, 11 from
    NVMe at 0.6 ms each; 16 threads on cores 0-15) and the GPU's launch overhead (about 1,800 launches per token
    at roughly 7 us each, the bulk of the 20 ms: the dense weights alone are 4-5 ms of bandwidth).
  - **Graph capture, profile, kernel rewrites, fetch overlap.** The token's launches are now captured once
    (`QX_GRAPH=0` disables; the first eligible token warms CubeCL's info cache, the second is recorded) and
    replayed: no gain, the GPU-only token stayed at 20 ms, so the cost is kernel time. An `nsys` profile of the
    GPU-only token (per-node graph tracing) gave 17.1 ms of kernels and 1,820 launches per token: `qx_mgemv` 6.5 ms
    (527 launches, about 1 TB/s effective), `qx_ds_hc_mixes` 4.7 ms (80 launches of 58 us: one block of 24 warps
    streaming 2 MB), `qx_ds_attn` 2.4 ms (59 us per layer at a 100-token context: a thread per key reading 2 KB
    rows), `qx_rms_groups` 1.2 ms (241 launches), the rest 2.3 ms. `hc_mixes` is now a block per output and
    `qx_ds_attn` scores with a warp per key and normalizes once: GPU-only token 16.5 ms (60 tok/s ceiling). The
    exact f32 check still agrees with the reference at the 1e-7 level; in production numerics p0's 12 positions
    give top-1 12/12, KLD mean 0.099 (the summation order changed, so the greedy text diverges from the earlier
    runs after 11 tokens, as any rounding change does). The fetch timer split the pool's time into NVMe waiting
    (13 ms per token) and compute (15 ms); the driver now computes the RAM-resident experts while the misses'
    reads are in flight (`ExpertStore::fetch_submit` / `fetch_wait`). The pool itself is near memory bandwidth:
    `qx-poolbw` measures 0.22 ms per 9.4 MiB expert at any batch size (44-54 GB/s on 16 cores), so a layer's 1-2
    CPU experts cost 0.3-0.5 ms and the only CPU-side lever left is fewer of them. Pinning the engine thread off
    the pool's cores (`QX_DS_CPUS`, `QX_DS_MAIN_CPU`) changed nothing.
  - **State at 128 generated tokens, RAM tier 80 GiB, 32 promotions per token:** the last 32 tokens average
    **41.8 ms, 23.9 tok/s** (58 ms over all 128 with the cold start); the Engram row gather on the host costs
    4.1 ms per token under NVMe contention (0.3 ms when the drive is idle), next to go through io_uring in one
    batch. Steady-state budget: GPU 16.5, pool compute about 14, NVMe wait about 5, Engram 4, gaps 2.
  - Engram rows now go through one io_uring batch per token (4.1 ms -> 0.1-0.3 ms averaged over the run; the
    steady-state token did not move, so the serial preads were mostly a cold-start cost). RAM tier 88 GiB (the
    plan's budget; 112 GiB cap): last 32 tokens **40.5 ms, 24.7 tok/s**, 53.8 ms over 128. At 20 GiB of VRAM
    experts the device has 2.5 GiB free after load (22 GiB does not fit: dense 7 GiB, embeddings 0.65, output
    0.5, scratch). Re-profiled GPU-only token: 13.8 ms of kernels in 15.7 ms: `qx_mgemv` 6.5 (7.0 GiB of dense
    weights per token, 1.15 TB/s), `hc_mixes` 2.35 (still 29 us a call: 24 blocks cannot pull 2 MB at bandwidth;
    next a split over more blocks with the partials summed in `hc_terms`), `attn` 1.35, `rms_groups` 1.2.
  - `hc_mixes` split 16 ways with the partials summed in `hc_terms` (exact check 6/6, KLD 0): GPU-only token
    **13.7 ms** (73 tok/s ceiling). The shared expert's SiLU and down GEMV now run between the mailbox post and
    the wait, overlapping the pool. **Teacher-forced benchmark** (`QX_DS_BENCH=0`, no generation, text-independent:
    the greedy text had turned repetitive after the kernel reorderings and inflated the hit rate): wiki1k's first
    256 tokens, 21 GiB VRAM / 88 GiB RAM, 32 promotions per token: last 32 tokens **40.8 ms, 24.5 tok/s**. Steady
    state there: GPU 13.7 + resident experts about 3, pool 20 (12 compute, 7.5 waiting on 8 NVMe misses per token
    at 0.9 ms each, partly overlapped), gaps 2; routings 82% VRAM, 15% RAM, 3.4% NVMe and the NVMe share not yet
    falling at 256 tokens (the tiers hold the trace's ranking, the forced text differs).
  - Driver fixes: victims' table zeros are issued before the copies (a zero queued behind 600 MiB of copies could
    land after the slot's quarantine had expired), no new copies while the previous tick's are in flight, RAM
    promotions as asynchronous io_uring reads (at most four in flight, a record counts as RAM-resident only once
    read; the driver never blocks between posts). 64 promotions per token still produces NaN logits from position
    205 of wiki1k, deterministically and with either tier's promotions disabled, so the cause is elsewhere.
    Isolated: `golden/data/wiki1k.tokens` was tokenized with another model's tokenizer (ids up to 227,135 against
    a 129,280 vocabulary; position 205 is id 210,905), so the embedding gather read past the table and the result
    depended on what followed it in VRAM, which is why 32 promotions happened to pass it. The engine now rejects
    out-of-range token ids; the forced benchmark uses the first 1,024 tokens of the DeepSeek routing trace
    (`golden/data/route1k.tokens`). Found on the way: `qx_ds_topk` stored its per-warp "equal" counts at offset 16
    of a 32-entry array with 32 warps (overlap and overflow once a store has more than 512 visible rows); fixed.
  - **Forced benchmark on valid tokens** (`golden/data/route1k.tokens`, the mixed corpus's first 1,024 tokens,
    21 GiB VRAM / 88 GiB RAM): **43.3 ms per token over the last 32 (23.1 tok/s)**, 44.3 over all 1,024, no NaN;
    64 promotions per token: 42.6 ms. The mixed text keeps the tiers churning (positions 767-1023: 76% VRAM, 22%
    RAM, 1.8% NVMe; pool 18.6 ms per token of which 3.4 waiting on NVMe). The GPU-only token grows with the
    context: 13.7 ms near position 100, 17 at 255, 21 at 511, 24 at 1023 (the indexer scores every visible row,
    the top-k is linear in them, attention saturates at window 128 + 512 selected rows); profiled next.
  - **The context growth was all attention**: `qx_ds_attn` averaged 224 us per layer over the 1k run (9 ms per
    token) against 34 us at a 100-token context; every other kernel was flat (`idx_score` 0.11 ms, `topk` 0.05).
    A block per head streamed all 640 key rows twice although MLA's 64 heads share one 512-wide latent as key and
    value. New `qx_ds_attn_part` (blocks over 8-head groups x 12-key chunks: each chunk loaded once into shared
    memory for its 8 heads, per-chunk max/sum/partial output) and `qx_ds_attn_merge` (the chunks and the sink,
    flash-decoding style). First version produced NaN: zero probabilities times uninitialized shared rows; the
    missing rows are now zeroed, and the shared key stride is 513 to keep consecutive rows on consecutive banks.
    Exact check 6/6 at KLD 0; production check p0 12/12, KLD 0.267. GPU-only token **13.1 ms at a 1k context**
    (was 24; 12-13 ms from position 255 on). Forced route1k 1024, 32 promotions: **31.9 ms per token over the
    last 32 (31.4 tok/s)**, 36.8 ms over all 1,024. Budget now: GPU 13 + resident experts 3, pool 20 (3.5 waiting
    on NVMe) partly overlapped, gaps about 2. The CPU side is the limit; the structural lever left is windows of
    several tokens (prefill chunks, DSpark verification), where the pool reads each expert once per window.
  - Process: chains started with `setsid nohup` died with two session restarts; they now run as transient
    systemd user services with the memory cap on the unit. Inside such a service cargo is not on PATH, which let
    two chains measure a stale binary before the build status was echoed.
  - **Windows of T tokens** (`DsEngine::forward_window`, `launch_window`; `QX_DS_WINDOW=T` in the forced bench,
    `QX_DS_DUMP_LOGITS` for comparisons): every per-token buffer has DS_TMAX = 32 rows, the kernels take grid T
    (they were written with positions arrays and strides), the compressor runs a window's tokens in order inside
    one block (a group's pending pairs accumulate across them), the latent ring is n_swa + DS_TMAX wide, the grouped
    output projection writes [group][token] and a `qx_ds_regroup` kernel lays the rows out per token for the second
    projection, the mailbox posts T tokens and the pool reads each expert once per window, one graph per width.
    Two bugs on the way: the per-layer top-k selection buffers were still one token wide (the top-k's "fewer rows
    than k" path wrote `sel[i] = i` for tokens 1..7 past them into neighbouring GEMV tables: compute-sanitizer
    showed the GEMV reading weights at 0x100000000, the u32 pair 0,1; found by printing the table words before
    each launch), and the sanitizer run itself was the way in. Result: **windows of 8 give logits bit-identical to
    one-token steps (128/128 rows)**. Windows of 32 differ everywhere: above 8 tokens the fused GEMV takes its
    tensor-core path with bf16 inputs (being confirmed with `QX_TMV=0`). First prefill rate, windows of 32: 34 ms
    per token at position 1023, pool-bound (promotions were per window, so the VRAM share fell to 53%; now scaled
    by the window's width). The chunk's distinct experts set the floor: about 100 per layer at 32 tokens, 0.9 GB per
    layer from RAM at the pool's 45 GB/s; wider chunks amortize further (the mailbox's 32-token limit is next).
  - Confirmed: windows of 32 with `QX_TMV=0` (the fused GEMV's f32 path in chunks of 8) are **bit-identical to
    one-token steps (128/128)**; the tensor-core path's bf16 inputs are the difference above 8 tokens. Prefill with
    promotions scaled by the window's width, route1k 1024, at the end of the run: windows of 32, f32 GEMV 24.0 ms
    per token (41.7 tok/s), tensor-core GEMV 22.6 (44.2); windows of 16, f32 GEMV 21.8-22.5 (44.5). Pool 24 ms per
    token in all three (the GPU is hidden): prefill is bound by the chunk's distinct experts streaming from RAM.
    Decode windows do not amortize the pool either (different tokens pick different experts), so speculation
    would not help the CPU-bound part. Next: the GPU reads a share of the RAM-resident experts in place over PCIe
    (the RAM tier is device-mapped pinned memory) in parallel with the pool, for decode and prefill alike.
  - **Tried and shelved: the GPU reading RAM-resident experts in place over PCIe.** A second device table holds
    each RAM record's mapped address; the post kernel hands a share (`QX_DS_GPU_SHARE`, eighths) of the RAM misses to
    the GPU's own expert kernels, the rest to the pool; RAM places are quarantined two ticks before reuse. The
    results stay bit-identical whichever side computes (128/128 rows, single steps and windows of 8), but the
    in-place reads run at about 7 GB/s: 1.3 ms per expert against the pool's 0.22, and decode went from 32 to
    135-175 ms per token at shares of 4-6 eighths (the per-layer round robin also sent nearly every RAM miss to
    the GPU, and mapped misses never reached the promotion candidates, so the VRAM tier stopped adapting). DMA
    copies into VRAM staging would reach 14-25 GB/s at best, still under the pool's 45; at most a 1.4x on the
    expert phase for a lot of machinery. The knob stays, default 0. Prefill's lever is wider windows: the distinct
    experts a window touches grow sublinearly with its width (about 100 per layer at 32 tokens, all 384 past
    about 500), so bytes per token fall; the mailbox's 32-token limit (QX_TMAX / QX_AMAX) is next.
  - Mailbox widened to 128 tokens (QX_TMAX 128, QX_AMAX 1280: MB_KIDX 5376, MB_ALL 10496, MB_X 16384; the pool's
    MAX_EXPERTS 1280; DS_TMAX 128 with the split attention launched in slices of 32 tokens so its partials stay at
    227 MB). The constants are shared with the Qwen engine, whose per-window scratch (`qkv`, `cv`, `beta`,
    `alpha`, the token-major buffers) grows with TMAX: check its VRAM budget and load before the Qwen service
    is re-enabled.
  - Windows of 128 are bit-identical to single steps on the f32 GEMV path (128/128); the tensor-core GEMV is
    capped at 32 tokens (`use_tmv` falls back above). A regression from the mapped path's deferred eviction
    (promotions stalled: 1 RAM promotion per run, decode at 40 ms) is fixed: places are quarantined only when the
    GPU reads RAM in place. **Prefill at the end of route1k 1024, 32 promotions per token: windows of 32 23.0 ms
    per token, 64 21.8, 128 20.8 (48 tok/s); decode 32.5 ms (30.8 tok/s).** Width stops paying because the pool
    turns compute-bound: a 128-token window needs 128 dot products per expert row, and 16 AVX-512 cores saturate
    around 4 TMAC/s (about 110 CPU experts per layer at 0.45 ms each for 128 tokens). The GPU could take those
    products only with the expert bytes moved over PCIe: DMA at 14-25 GB/s gives about the same 15-20 ms per
    token as the pool, both together perhaps 10 ms; the NVMe tier (58 GiB of experts at 10 GB/s) caps any
    whole-expert-set sweep near 20 ms per token. Prefill on this box lands at 50-100 tok/s, decode at 31; the
    engine's remaining value is in serving: KV paging, prefix cache, sampling, the API.
  - **Serving (first version).** `quixiinfer serve|run` dispatch on `general.architecture`: `deepseek41` loads the
    tiered engine on the pack (`--pack` or QX_DS_PACK; `--vram-gib`/`--ram-gib` or QX_DS_VRAM/QX_DS_RAM; the
    routing profile from `--profile`, QX_DS_PROFILE or `golden/profile-dsv41.txt`) and runs `worker_ds`: one
    request at a time, the prompt in windows of `--window` (64) tokens, then one token a step with host-side
    sampling (temperature, top-k, top-p, min-p, a splitmix draw keyed by seed and position), EOS and stop strings,
    the OpenAI routes and the chat template from the GGUF unchanged. `qx-tok` picks the pre-tokenizer from
    `tokenizer.ggml.pre`: `joyai-llm` (DeepSeek V4) is llama.cpp's chain of three splits. First runs: "What is the
    capital of France?" (thinking off, greedy) answers "The capital of France is Paris." and stops on EOS; a haiku
    request with thinking produces a coherent reasoning trace. Cold-start speeds in those short runs: prompt
    4.6-6.6 tok/s, decode 5.6-17.8 tok/s (the tiers warm over the first hundred tokens). Not yet: several
    requests at once (slots), the prefix cache, KV paging past the context, structured outputs for this model.
    The HTTP API answers chat (non-streaming and SSE with `reasoning_content` deltas) and completions with stop
    strings; the model id comes from the GGUF's `general.name`. Tokenization checked against llama.cpp's
    `llama-tokenize` on six texts (prose, code, CJK, whitespace runs, a chat prompt with control tokens, symbols
    and emoji): **identical ids on all, exact decode round trips.**
  - Two graph bugs found while adding a warm-up and a prefix rewind. (1) The first eligible window of a width
    called CubeCL's `graph_prepare` and ran eagerly, the second captured; with several widths in play a second
    `graph_prepare` before the first capture panics ("a graph capture is already prepared on this stream"). Now
    prepare, the executed eager run and the recording (not executed) happen in one call. (2) Three selection
    grids (`idx_score`, `block_max`, `cand_mask`) were sized from the visible rows at capture time, so a replayed
    graph never scored compressed rows beyond those of the capturing position: a silent long-context quality loss
    that the short reference checks could not see. They are sized from the maximum now (the kernels bound by
    position), as the Qwen engine does; re-checked against the reference over p1's 59 positions.
  - Prefix reuse for consecutive requests (`worker_ds`, `QX_DS_PREFIX=0` off): the engine keeps a host history of
    every position's latent per layer (copied out of the rings every 32 tokens), `rewind(p)` restores the ring
    cells of the attention window from it (the compressed stores and indexer keys are positional and the positions
    from p on are recomputed), and a new prompt sharing at least 64 tokens with the sequence in the engine
    continues from that prefix (rounded to a group boundary). A warm-up at load runs the window widths in use
    once (kernel compilation and graph captures off the first request).
    Two-turn chat (greedy): turn 2 reused 114 of its 134 prompt tokens (prompt 1.85 s against 3.88 s without),
    but its answer differed by a few words from the no-reuse run: the leftover 20-token window took the fused
    GEMV's tensor-core path (bf16 inputs, 9-32 tokens) while 64-token windows take the f32 path. DeepSeek windows
    now always use the f32 path (`QX_TMV=1` opts into the tensor cores), so single steps, windows and rewinds agree
    bit for bit; the reference check over p1's 59 positions stays at 57/59, KLD 0.129.
    Re-run: **turn 2's answer is identical with and without the reuse**; prompt 1.54 s (114 of 134 tokens reused)
    against 3.88 s. Cold start remains visible in these short chats (33% of the first request's routings on
    NVMe, decode 18 tok/s): the tiers start from the trace's ranking and adapt over the first few hundred tokens,
    so the next step is saving the live ranking as the profile to load from.
  - **Live routing profile** (`--profile-out FILE` / QX_DS_PROFILE_OUT, written after requests at most once a
    minute; `DsEngine::save_profile` asks the driver for its ranking: VRAM residents, then RAM, then NVMe, each by
    decayed score, every expert once). Two server rounds of the same three short chats: started from the trace
    profile the first request put 7,923 routings on NVMe; restarted from the saved ranking, 877 (VRAM hits 45k
    against 18k over the warm-up plus the request), prompt 13 against 8 tok/s. Short answers decode at 18-21
    tok/s: the first hundred tokens of a new text are the warm-up regime the forced bench showed before settling at
    31 tok/s. README has a DeepSeek section (pack, serve, profiles, the memory cap).
  - **Structured outputs** on the DeepSeek path: the server's llguidance matchers (built from the GGUF tokenizer,
    129,280 rows) give a token mask per step, applied to the logits before the host sampler (disallowed tokens to
    -inf), the sampled token fed back, the request ending when the grammar is complete. `response_format`
    (`json_schema`, `json_object`) and `guided_regex` / `guided_json` / `guided_grammar` all work: a schema
    request returned `{"name": "Tokyo", "population": 37114000, "country": "Japan"}`, a regex-guided completion
    `1969-07-20`. A start finding the `--profile-out` file loads the tiers from it;
    `tools/quixiinfer-deepseek.service.example` is a user unit with the memory cap.

## Not yet

- Admission without a GPU sync (`begin_seq`'s reset and uploads wait for the queued step: about 6% at c=16);
  snapshots without a sync (D2H copies ordered on the stream).
- Multi-slot prefill chunks (a burst of arriving prompts in one chunk: 3x cheaper than lanes for 16 prompts).
- Per-slot VRAM: the pooled keys in fp16 (0.9 GiB less); the 0.15 GiB per slot of runtime headroom.
- VRAM residency at c=16 (the throughput lever): token embedding to host memory (0.5 GiB), smaller KV ring, ...
- DeepSeek-V4.1-Flash-1017: the GGUF port on this box once storage allows (`docs/deepseek41-plan.md`); NVFP4 on
  a multi-GPU machine.
- Long-context prefill: the QSA selection and indexer scoring per query at 65k blocks (3x slower than at 4k).
- GPU-side accept to drop a sync per speculative window; pack-load page
  faults (prefault with MAP_POPULATE or huge pages).
