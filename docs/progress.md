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

## Not yet

- Admission without a GPU sync (`begin_seq`'s reset and uploads wait for the queued step: about 6% at c=16);
  snapshots without a sync (D2H copies ordered on the stream).
- Multi-slot prefill chunks (a burst of arriving prompts in one chunk: 3x cheaper than lanes for 16 prompts).
- Per-slot VRAM: the indexer key pool (f32, full context per slot) in fp16/fp8 or grown with the sequence, so 16
  slots fit without shrinking the cache.
- VRAM residency at c=16 (the throughput lever): token embedding to host memory (0.5 GiB), smaller KV ring, ...
- DeepSeek-V4.1-Flash-1017: the GGUF port on this box once storage allows (`docs/deepseek41-plan.md`); NVFP4 on
  a multi-GPU machine.
- Long-context prefill: the QSA selection and indexer scoring per query at 65k blocks (3x slower than at 4k).
- GPU-side accept to drop a sync per speculative window; pack-load page
  faults (prefault with MAP_POPULATE or huge pages).
