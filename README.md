# QuixiInfer

QuixiInfer is an inference server for **Qwen3.8-Flash-Next** on one workstation: one RTX 5090 (32 GB), a 16-core
CPU, 120 GiB of RAM and an NVMe SSD. It is written in Rust. Its CUDA kernels are its own, compiled at build time
and launched through Burn/CubeCL. It takes its architecture from [DeepInfer](../DeepInfer), DeepSeek's
cluster engine, and applies it to a single GPU with CPU expert offload: continuous batching over sequence slots,
asynchronous scheduling, paged KV, a prefix cache, exact speculative decoding and structured outputs.

It runs NVIDIA's [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) checkpoint,
which keeps the routed experts in 4-bit NVFP4 and the attention, GDN and shared experts in higher precision.
It also still runs Unsloth's IQ GGUFs (`UD-IQ3_XXS` was the first target).

## QuixiInfer, Strata and Basalt

All three run the same model on the same kind of machine. They share the basic design: the hottest experts sit
in a VRAM cache, every expert sits in pinned RAM, the CPU computes the experts the GPU lacks, and the PLE n-gram
table is read from the SSD. [Strata](../Strata) came first. [Basalt](../basalt) is a fork of Strata cut down to
Linux and Blackwell. QuixiInfer is a separate code base that only learns from them.

| | **Strata** | **Basalt** | **QuixiInfer** |
|---|---|---|---|
| Aim | Run the model on an ordinary gaming PC | Strata's engine, tuned for one Linux Blackwell workstation | A DeepInfer-style server: many concurrent requests on one GPU, every result checked |
| Hardware | NVIDIA RTX 20-50, AMD RDNA2-4, Apple Silicon, Intel Arc (community); 12 GB VRAM and up; Windows, Linux, macOS | NVIDIA sm_120 on Linux; optional second GPU as a peer expert tier | NVIDIA sm_120 on Linux, one GPU |
| Code | C++/CUDA/HIP/Metal, parts of ggml and llama.cpp | C++/CUDA, ggml parts extracted | Rust, its own CUDA C++ kernels via CubeCL, no ggml at runtime |
| Model files | Pre-built packs, many quants (Q2_0 to Q8_0), NVFP4 via a converter | `.basalt` packs: one file with weights, tokenizer, drafter, profile and vision | A `.qxp` pack: a GGUF-headered file in the engine's memory layout, written from a GGUF |
| Concurrency | One request by default; `parallel` slots on request | `--parallel 1..8`; the context is shared across slots unless `--slot-full-context` | Continuous batching over 8-16 slots, each with the full context (262k in production) |
| Speculation | MTP drafter, 1.6-1.8x | MTP drafter, self-distilled for its packs | MTP drafter, exact; single-stream only |
| KV cache | fp16 or int8 | fp16 or int8, elastic: borrowed from the expert cache | FP8 E4M3 with a scale per 32 dims, paged in pinned RAM, recent cells mirrored in VRAM |
| Prefix reuse | Message-boundary and prompt caches | Conversation checkpoints per slot | A shared prefix cache: KV pages by reference count, entries sized by position |
| Vision, tools, APIs | Images, OpenAI, Anthropic and Responses APIs, MCP, web app, installer | Images, OpenAI and Anthropic APIs, web UI | Text only; OpenAI chat and completions, JSON schema, regex and grammar outputs |
| Exactness | Speculative output equals plain sampling | Speculation exact unless relaxed drafting is on; greedy-identity check; PPL/KLD gate against Q8_0 | Bit-equal results across GPU and CPU expert placement, across batch composition, through prefix restore and speculation |

### Speed

These numbers come from different machines and settings. The NVFP4 rows are the closest comparison.

| | Single-stream decode | Concurrent decode | Prefill |
|---|---|---|---|
| Strata, NVFP4, RTX 5090 + 9950X3D, 128 GB RAM, one 64k session, int8 KV ([docs](../Strata/docs/NVFP4.md)) | 114-116 tok/s (400-token answer) | | 3,900 tok/s (32k prompt) |
| Basalt, IQ packs, 400 W (from Basalt's docs as recorded in [PLAN.md](PLAN.md); setup not restated there) | about 184 tok/s without speculation, 320 with | | |
| QuixiInfer, NVFP4, one 64k slot | 95 tok/s (400-token answer), 113 (2,000 tokens) | | |
| QuixiInfer, NVFP4, production: 8 slots x 262k, FP8 KV, through the API | 50 tok/s (63 per request) | 78-81 tok/s at 4-16 clients | 1,070-1,139 tok/s (25k-100k prompts) |

**QuixiInfer is still behind Strata for one conversation**, by about 20% on a 400-token answer, and 3-4x behind in
prefill. [docs/strata-rca.md](docs/strata-rca.md) has the root-cause analysis. In short: the CPU computes every
expert missing from VRAM at the RAM's bandwidth (57 GB/s here, shared with any PCIe copies), so speed follows the
expert hit rate. QuixiInfer now lends its prompt buffers from the expert cache as Strata does, which brings one 64k
slot to Strata's cache size (8,360 experts); the rest of the gap is a cold cache at the start of each
conversation. The production layout reserves eight 262k sessions, which costs about 2,000 cached experts.

What QuixiInfer does that the other two do not:

- **Many full-length conversations at once.** Eight slots each get the whole 262k context. New prompts join running
  sequences between decode steps. A 16 GiB pinned KV page pool is shared by all slots, and requests reserve their
  pages when they are admitted.
- **Results that do not depend on scheduling.** A token's logits are the same bits whether its experts ran on the
  GPU or the CPU, whether it was decoded alone or in a mixed batch, and whether its prefix came from the cache or was
  recomputed. Speculative tokens equal plain sampling. Prefill is deterministic from run to run. Each property has a
  check tool (below).
- **One Rust code base with its own kernels.** No ggml or llama.cpp code is linked. llama.cpp is used offline only:
  to convert the checkpoint and as a reference oracle.

## Status

- **Serving.** `quixiinfer.service` runs the NVFP4 pack on port 8000 with 8 slots at 262,144 tokens of context.
  The progress log with every measurement is [docs/progress.md](docs/progress.md).
- **Quality.** With f32 KV and f32 experts, the engine matches the f32 reference (`qx-ref`) at KLD 6e-12. With FP8 KV and Q8 expert
  activations, KLD to the reference is 6e-4 to 4e-2, depending on the prompt. The reference matches llama.cpp at
  about 1e-3. llama.cpp's golden logits predict the next wikitext token 86-89% of the time.
- **Known gaps.**
  - Single-stream speed (see above).
  - Startup takes about 90 s from the pack.
  - A very long prompt can push the shorter prompts out of the prefix cache.
  - No vision, no Anthropic API, no multi-GPU.

## Build and run

Requirements: Linux x86-64 with AVX-512 VNNI, an RTX 50-series GPU (sm_120), CUDA 13 (`nvcc` at
`/usr/local/cuda-13.3/bin`, or set `QX_NVCC`), and a recent stable Rust.

```sh
cargo build --release                       # target/release/quixiinfer and the check tools
```

The model: convert NVIDIA's checkpoint with the patched llama.cpp converter (it applies the FP8 PLE table's
per-tensor scale), then write the pack:

```sh
python convert_hf_to_gguf.py Qwen3.8-Flash-Next-NVFP4/ --outtype q8_0 --no-mtp --outfile Qwen3.8-Flash-Next-NVFP4.gguf
target/release/qx-pack pack Qwen3.8-Flash-Next-NVFP4.gguf Qwen3.8-Flash-Next-NVFP4.qxp --profile routing.txt
target/release/qx-pack verify Qwen3.8-Flash-Next-NVFP4.qxp Qwen3.8-Flash-Next-NVFP4.gguf
```

The drafter is the Q8_0 MTP GGUF (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`), which sits next to the model.

Serve:

```sh
QX_POOL_CPUS=16 QX_KV_GIB=16 QX_PREFIX_GIB=4 QX_PLE_CACHE_ROWS=10000000 \
  target/release/quixiinfer serve --host 127.0.0.1 --port 8000 --ctx 262144 --slots 8
```

The server refuses to listen beyond 127.0.0.1 without `--api-key`. Endpoints: `/v1/models`,
`/v1/chat/completions` and `/v1/completions`, streamed or not.

- **Completion limit.** A request without `max_tokens` may use the rest of the context. The model thinks before it
  answers, and that often takes thousands of tokens.
- **Thinking.** `chat_template_kwargs: {"enable_thinking": false}` turns it off.
- **Structured output.** `response_format` (`json_object`, `json_schema`) and the `guided_*` fields constrain the
  output.

| Setting | What it does |
|---|---|
| `--ctx N` | Tokens of context per slot. |
| `--slots N` | Concurrent sequences. |
| `QX_KV_GIB` | Size of the pinned KV page pool. |
| `QX_PREFIX_GIB` | Size of the prefix cache. |
| `QX_POOL_CPUS` | CPU expert pool threads (a count or a CPU list). Physical cores are faster than SMT siblings. |
| `QX_PREFILL_CHUNK` | Prompt tokens per prefill chunk. The default and the maximum is 2048. |
| `QX_MODEL` | Model path for the tools. A pack or a GGUF. |

## What is in the repository

| Crate | Contents |
|---|---|
| `qx-gpu` | CubeCL runtime glue, device memory, and the build script that flattens `kernels/*.cu` and checks them with nvcc. At run time each module is also checked once with NVRTC, so a kernel that fails to compile fails loudly. |
| `qx-engine` | The engine: the forward pass, the VRAM expert cache and its swapper, sequence slots, windows and batched decode, chunked prefill with staged experts, the paged FP8 KV, the prefix cache, speculation, and the PLE reader (io_uring, O_DIRECT). Also the check and bench binaries. |
| `qx-cpu` | The CPU expert pool: AVX-512 VNNI dot products for NVFP4, IQ2_S, IQ3_S and IQ4_NL, each bit-equal to the GPU kernels. |
| `qx-quant`, `qx-pack` | Dequantizers, the GGUF reader, and the pack writer and verifier. |
| `qx-ref` | The f32 CPU reference of the whole model, which the engine is checked against. |
| `qx-tok` | The tokenizer, which matches llama.cpp token for token. |
| `qx-serve` | The OpenAI-compatible server and its continuous-batching scheduler. |

`kernels/` holds the CUDA sources. `tools/` holds `loadgen.py` (concurrent load at c=1/4/8/16), `longprompt.py`,
`pardl.py` (a parallel resumable downloader) and `golden/` (llama.cpp golden logits).

## Checks

Each tool loads the model and checks one property:

| Tool | What it checks |
|---|---|
| `qx-check` | Engine logits against the f32 reference, per position and per activation. Needs a GGUF. |
| `qx-invariance` | Two expert-cache sizes (different experts on the GPU and the CPU) give bitwise identical logits. |
| `qx-window-check` | Multi-token windows equal one-token steps, bit for bit. |
| `qx-batch-check` | Slots decoded in mixed batches equal each slot decoded alone. |
| `qx-prefix-check` | A restored prefix continues exactly as an uninterrupted run. |
| `qx-spec-check` | Speculative decoding gives exactly the plain-decoding tokens, greedy and sampled. |
| `qx-prefill-check` | Batched prefill agrees with token-by-token decode. `QX_DUMP_LOGITS` compares two runs bit for bit. |
| `qx-gen-check` | The production generation paths produce the eager forward pass's tokens. "The capital of France is" must give " Paris". |
| `qx-nvfp4-check` | The GPU and CPU NVFP4 expert kernels agree bit for bit on random records. |

**Parity checks are not enough on their own.** The first NVFP4 GGUF passed all of them while producing nonsense: the
converter had dropped the PLE table's scale, so the reference, the engine and llama.cpp all agreed on a broken
file. Before benchmarking a new model file, also check:

- The golden logits' next-token accuracy on their own text.
- A few tensors of each kind against the original checkpoint.
- A question with a known answer, asked through the server.

## License

MIT. The model is under the [Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
QuixiInfer learns from DeepInfer, Strata, Basalt and llama.cpp (see [docs/kernel-references.md](docs/kernel-references.md)),
but it contains no code from them (the IQ lookup grids are ggml's data tables).
