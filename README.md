# QuixiInfer

QuixiInfer serves **Qwen3.8-Flash-Next** (125B MoE, 512 experts per layer, 262k context) to many users at once,
from a single workstation: one RTX 5090 (32 GB), a 16-core CPU, 128 GB of RAM and an NVMe SSD.

It is written in Rust, with its own CUDA kernels, compiled at build time and launched through CubeCL. No ggml or
llama.cpp code is linked. The server design comes from DeepInfer, DeepSeek's cluster engine, scaled down to one GPU
with CPU expert offload:

- continuous batching over sequence slots;
- asynchronous scheduling;
- paged FP8 KV;
- a shared prefix cache;
- exact speculative decoding;
- structured outputs.

It runs NVIDIA's [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) checkpoint, in
which the routed experts are 4-bit NVFP4 and attention, GDN and the shared experts are higher precision. It also
still runs Unsloth's IQ GGUFs (`UD-IQ3_XXS`).

## Where it fits

Several engines run this model, or this class of model, and each optimizes for something different:

| Engine | Built for | Hardware | Concurrency | What it optimizes |
|---|---|---|---|---|
| **QuixiInfer** | A shared server on one workstation | One NVIDIA sm_120 GPU, Linux | Continuous batching, 8 slots of 262k each | Aggregate throughput at 4-16 users, results that do not depend on scheduling |
| **Basalt** | One user on one Blackwell workstation | NVIDIA sm_120, Linux; an optional second GPU as an expert tier | `--parallel 1..8`, context shared across slots | Single-stream decode speed (tuned packs, self-distilled drafter) |
| **Strata** (Basalt's parent) | One user on a gaming PC | NVIDIA, AMD, Apple and Intel GPUs, 12 GB of VRAM and up; Windows, Linux, macOS | One request by default | Running on ordinary hardware; apps, APIs, installer |
| **[colibri](https://github.com/JustVugg/colibri)** | Models far bigger than RAM, on any machine | CPU first; CUDA, Vulkan and Metal tiers | One request | Reach: experts streamed from NVMe, so 284B-1T models still run |
| **SlimServe** | A multi-GPU server | 2-8 data-center or consumer GPUs (vLLM based) | vLLM batching | Throughput with tensor and expert parallelism |
| **DeepInfer** | DeepSeek's production cluster | SM100 GPUs, Ascend NPUs, 4-64 per deployment | Data and expert parallelism | DeepSeek-V4.1-Flash at data-center scale |
| **llama.cpp** | General local inference | Almost anything | `--parallel` | Breadth of models and hardware. QuixiInfer uses it offline only: to convert the checkpoint, and as a reference oracle |

How QuixiInfer differs from the engines that run this model on the same kind of machine (Strata, Basalt, colibri):

- **It is a server, not a chat app.** Eight concurrent conversations each get the whole 262k context. New prompts
  join running sequences between decode steps. All slots share one 16 GiB pool of pinned KV pages, and a request
  reserves its pages when it is admitted.
- **The GPU and CPU work concurrently, even under load.** Every expert missing from VRAM is computed on the CPU at
  the RAM's bandwidth. A batch of several sequences runs as two interleaved halves, so one half's GPU layers run
  while the CPU computes the other half's missing experts.
- **Results do not depend on scheduling.** A token's logits are the same bits whether its experts ran on the GPU or
  the CPU, whether its prefix came from the cache or was recomputed, and whether it was decoded alone or in a batch
  (for batch sizes of up to 4 tokens per half; wider batches use a tensor-core kernel that rounds differently).
  Speculative tokens equal plain sampling, and prefill is deterministic from run to run. Each of these properties
  has a check tool (see [Checks](#checks)).

What it does not do (yet): vision, the Anthropic API, multiple GPUs. Besides Qwen3.8-Flash-Next it serves
DeepSeek-V4.1-Flash at 256k context on the same machine, through a second engine in the same binary
([below](#deepseek-v41-flash)).

## Performance

The QuixiInfer figures below were measured on an RTX 5090, a Ryzen 9 9950X and 2 x 64 GB DDR5-5600 (JEDEC, about
55 GB/s measured), through the OpenAI API, in the production layout: 8 slots x 262,144 tokens, FP8 KV and NVFP4
experts. Every measurement is logged in [docs/progress.md](docs/progress.md).

**Concurrent decode** (`tools/loadgen.py --unique`, 256-token answers, 3 rounds per level, aggregate tok/s):

| Clients | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Aggregate tok/s | 54 | 90 | 95 | 90 |
| Per request | 69 | 26 | 14 | 13 |
| Median time to first token | 1.1 s | 1.1 s | 2.1 s | 22 s (8 slots: half the requests queue) |

**Prefill:** 1,070-1,139 tok/s on 25k-100k-token prompts.

**One conversation** (one 64k slot, so the expert cache is larger): 95 tok/s on a 400-token answer, 113 on a
2,000-token answer.

### Against the other engines

The figures below come from each project's own documentation, on different machines and settings, so treat them as
a rough guide. The Strata and Basalt rows are the closest to QuixiInfer's machine.

| Engine, setup | Single stream | Concurrent | Prefill |
|---|---|---|---|
| **QuixiInfer**, NVFP4, RTX 5090 + 9950X, 8 x 262k slots | 54 tok/s (API, 256-token answers); 95-113 with one 64k slot | **90-95 tok/s at 4-16 clients** | 1,070-1,139 tok/s |
| Strata, NVFP4, RTX 5090 + 9950X3D, 128 GB, one 64k session, int8 KV | 114-116 tok/s (400-token answer) | not published | 3,900 tok/s (32k prompt) |
| Basalt, IQ3_XXS pack, RTX 5090 at 400 W | 184 tok/s, 320 with speculation | not published | not published |
| colibri, Qwen3.8-Flash-Next, 8-core desktop, CPU only | 1.9-2.6 tok/s; 4.0 with int4 experts | single request | 512 tokens in 30-44 s |
| SlimServe, `qwen38fn-fp8-8`, 8 x RTX 3090, TP8 + EP | 140 tok/s | 591 at 8, 1,151 at 32 | not published |
| DeepInfer, DeepSeek-V4.1-Flash, 4-64 SM100 GPUs | different model and scale; see its README | | |

**For one conversation, QuixiInfer is slower than Strata and Basalt.** Strata is about 20% faster on a 400-token
answer and 3-4x faster on prefill; Basalt's tuned IQ3_XXS packs with speculation are about 3x faster. The reasons
are in [docs/strata-rca.md](docs/strata-rca.md):

- Decode speed follows the share of experts already in VRAM, and the production layout gives about 2,000
  experts of VRAM to eight full-length contexts.
- Each new conversation starts with a cold cache.
- Speculative decoding runs only when a single sequence is active.

QuixiInfer's place is serving several users at once: at 4-16 concurrent requests it keeps 90-95 tok/s across all
of them, on one GPU.

**DeepSeek-V4.1-Flash, measured on this machine, same model files and token ids** (2026-11-07, `perf/`): Strata's
`strata-dsv41` decodes at 27.5 tok/s against QuixiInfer's 30 at 1k context, so decode is even. Prefill is not:
Strata runs a 16k prompt at 935 tok/s (8192-row FP16 tensor-core chunks) against QuixiInfer's 190-250 (exact
int8 chunks). Quality, teacher-forced on the same ids against the f32 reference: QuixiInfer's production path is at
KL 0.01-0.02 and 96% top-1 (perplexity +0.1%); Strata's at KL 0.025-0.046 and 93-94% (perplexity within ±2%), its
FP16 prefill adding nothing measurable over its own decode path. Details in `perf/findings.md`.

**What limits it.** At 8 concurrent sequences a decode step is 53 ms, of which 50 ms is the CPU computing the
experts not in VRAM, at about 54 µs per expert. Misses are per token: each token picks 10 of 512 experts per layer,
so sequences rarely share them. Batching therefore saves GPU time but not CPU time, and throughput levels off from
about 4 users. The levers are:

- more VRAM for the expert cache;
- faster RAM (EXPO instead of JEDEC);
- cache-aware routing (`QX_ROUTE_CACHE`, experimental, see below).

## Quick start

Requirements:

- Linux x86-64 with AVX-512 VNNI;
- an RTX 50-series GPU (sm_120);
- CUDA 13 (`nvcc` at `/usr/local/cuda-13.3/bin`, or set `QX_NVCC`);
- a recent stable Rust;
- about 128 GB of RAM: 63 GiB of pinned experts, plus the KV pool and the PLE row cache;
- the pack on an NVMe drive.

**1. Build.**

```sh
cargo build --release      # target/release/quixiinfer, qx-pack and the check tools
```

**2. Make the model pack.** Download [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
(`tools/pardl.py` is a parallel, resumable downloader). Convert it with llama.cpp PR #30817's converter, which
applies the FP8 PLE table's per-tensor scale, then write and verify the pack:

```sh
python convert_hf_to_gguf.py Qwen3.8-Flash-Next-NVFP4/ --outtype q8_0 --no-mtp --outfile Qwen3.8-Flash-Next-NVFP4.gguf
target/release/qx-pack pack   Qwen3.8-Flash-Next-NVFP4.gguf Qwen3.8-Flash-Next-NVFP4.qxp --profile routing.txt
target/release/qx-pack verify Qwen3.8-Flash-Next-NVFP4.qxp  Qwen3.8-Flash-Next-NVFP4.gguf
```

Put the MTP drafter (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`) next to the model; it is found automatically.

**3. Serve.**

```sh
QX_POOL_CPUS=16 QX_KV_GIB=16 QX_PREFIX_GIB=4 QX_PLE_CACHE_ROWS=10000000 \
  target/release/quixiinfer serve --host 127.0.0.1 --port 8000 --ctx 262144 --slots 8
```

Startup takes about 90 s; the first start after a reboot takes longer while the experts are read into RAM. The
server refuses to listen beyond 127.0.0.1 without `--api-key KEY`.

**4. Ask it something.**

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false}}'
```

The endpoints are `/v1/models`, `/v1/chat/completions` and `/v1/completions`, streamed or not, and work with any
OpenAI client.

- **Completion limit.** Without `max_tokens`, a request may use the rest of its context. The model thinks before
  it answers, and thinking often takes thousands of tokens.
- **Thinking.** `chat_template_kwargs: {"enable_thinking": false}` turns it off.
- **Structured output.** `response_format` (`json_object`, `json_schema`) and the `guided_*` fields (regex,
  grammar) constrain the output.

**5. Run it as a service** (optional). In production QuixiInfer runs as a systemd unit; the parts that matter:

```ini
[Service]
EnvironmentFile=/etc/quixiinfer.env            # API_KEY=...
Environment=QX_POOL_CPUS=16 QX_KV_GIB=16 QX_PREFIX_GIB=4 QX_PLE_CACHE_ROWS=10000000
ExecStart=/usr/local/bin/quixiinfer serve --host 0.0.0.0 --port 8000 --api-key ${API_KEY} --ctx 262144 --slots 8
Restart=always
LimitMEMLOCK=infinity
```

Stop the service before benchmarking: two engines cannot both pin the experts.

### DeepSeek-V4.1-Flash

The same binary serves [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
at UD-IQ2_XXS (284 B parameters, 280 GiB of weights) on the same machine, at the model's full 262,144-token
context, with the experts in three tiers: what the VRAM has left after the context and the slots (15 GiB at one
262k slot, 12 at three, 21 at 32k), about 100 GiB pinned in RAM (the server is meant to run alone on the box;
about 13 GiB stay free for the system), the rest read from the pack on NVMe as routed. Several requests run at
once over sequence slots; a request sharing a prefix with an earlier one continues from it; parked sequences
resume bit-exactly; structured outputs go through the same grammar engine as the Qwen path.

**Measured** (RTX 5090, Ryzen 9 9950X, 2 x 64 GB DDR5-5600; the harness in `perf/`, every row in
`perf/scoreboard.md`):

| | ms per token | tok/s |
|---|--:|--:|
| Decode at 1k context (one stream) | 33 | 30 |
| Decode at a 256k position (one stream, resumed state) | 44-45 | 22-23 |
| Server at 262k context, 3 slots: one request | 43 | 23 |
| Server at 262k context, 3 slots: three concurrent requests | 108 per request | 9.2 each, 28 aggregate |
| Server at 8k context, 3 slots: three concurrent requests | 97 per request | 10.3 each, 31 aggregate |
| Prefill, the last 16k tokens of a 256k prompt (8192-row chunks) | 5.6 | 179 |
| Prefill, a 16k prompt from empty (4096-row chunks) | 4-5 | 190-250 |
| Time to first token, a 13.5k-token prompt on the 262k / 3-slot server | 44 s | 308 (prompt rate) |

Prompts run through a chunked, expert-major path: each chunk of rows goes layer by layer, and every expert the
chunk needs is streamed once per layer into exact int8 tensor-core kernels, so a prompt's cost no longer grows
with the number of distinct experts per window. Long prompts also skip the upper layers for positions outside the
tail they feed forward (`QX_DS_SKIP=0` turns that off). A 256k decode step is about 28 ms of waiting for
experts outside VRAM and 21 ms of kernels; the indexer and attention over the context are under 7 ms of it.

**Exactness.** The prompt path, windows of any width, parked and resumed sequences and prefix reuse all produce
the same bits as one-token decode (`tools/ds-checks.sh`, five checks, run after every change). The production
numerics (int8 activations for the dense projections and the expert kernels) sit at KL 0.01-0.02 from the f32
reference with 96% top-1 agreement and perplexity +0.1% over 1,023 teacher-forced positions
(`tools/logits_compare.py`); the engine's f32 mode matches the CPU reference exactly.

```sh
# the pack from the six GGUF shards (pass the first), once
target/release/qx-pack pack   DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp
target/release/qx-pack verify DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf

# serve at the full context: the engine is chosen by the GGUF's architecture; cap the memory, the tiers pin most of the RAM
systemd-run --user --scope -p MemoryMax=118G -p MemorySwapMax=0 -- \
  env CUDA_CACHE_MAXSIZE=4294967296 \
  target/release/quixiinfer serve --model DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --pack DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp --ctx 262144 --slots 3 --port 8000 \
  --ram-gib 95 --profile golden/profile-dsv41.txt --profile-out live-profile.txt
```

`--profile` ranks the experts the tiers are filled with at start (`golden/profile-dsv41.txt` comes from a routing
trace of a mixed corpus and is the default when present; it also places the decode tail of a resumed 256k
sequence 11% better than that sequence's own prompt ranking); `--profile-out` writes the live ranking after
requests, and a start finding that file loads from it instead. `--slots` (2) sets how many requests run at once;
each 262k slot takes about 1.5 GiB of VRAM from the expert tier. `--window` (128) is the prompt window below
32k of context. The chat template and tokenizer come from the GGUF (tokenization checked against llama.cpp). The
expert tier leaves `QX_DS_VRAM_RESERVE_MIB` (2560) of device memory for the kernel modules and the window graphs;
the warm-up line reports what is left, and a kernel that would need compiling after the warm-up aborts the server
(it means a module failed to load). `CUDA_CACHE_MAXSIZE` raises the driver's JIT cache from its 1 GiB default: the
engine's kernel modules evict each other from it, and a cold start then spends 10-20 minutes in ptxas.

| Setting (DeepSeek) | What it does |
|---|---|
| `QX_DS_CHUNK` | Rows per prompt chunk: 4096 by default at 32k of context and above, off below (prompts run in windows). 8192 takes 22% off the time to first token at 256k but costs three-stream decode 3.9% (the chunk's buffers come out of the expert tier). |
| `QX_DS_SKIP` | `0` runs every layer for every prompt position (the default skips the upper layers for positions outside the tail they feed forward; exact for the tokens that follow). |
| `QX_DS_STAGE_SHARE`, `QX_DS_STAGE_SHARE_DECODE` | Of 8: how many of the RAM-resident missing experts the GPU takes by DMA staging in windows (8) and in single decode steps (3, the measured optimum: DMA and the CPU pool cost the same per record). |
| `QX_DS_VRAM`, `QX_DS_RAM` | The tiers in GiB (`--vram-gib` 21, `--ram-gib` 100; the VRAM figure is a cap, the context and slots come first). |
| `QX_DS_PROMOS` | Expert promotions into VRAM per token (4). |
| `QX_DS_TOPK_KERNEL=1`, `QX_DS_IDX_KERNEL=1`, `QX_MGEMV_ZGROUPS=0` | The previous indexer top-k, scorer and dense-GEMV kernels, for A/B runs; every variant is bit-identical. |

### Settings

| Setting | What it does |
|---|---|
| `--ctx N` | Tokens of context per slot. |
| `--slots N` | Concurrent sequences. Each slot holds about 0.5 GiB of VRAM at 262k (taken from the expert cache), so more slots make every request slower; 8 is the measured best here. |
| `--api-key KEY` | Required to listen beyond 127.0.0.1. |
| `QX_KV_GIB` | Size of the pinned KV page pool. |
| `QX_PREFIX_GIB` | Size of the prefix cache. |
| `QX_POOL_CPUS` | CPU expert pool threads: a count or a CPU list. Physical cores are faster than SMT siblings. |
| `QX_PLE_CACHE_ROWS` | Rows of the PLE n-gram table cached in RAM. |
| `QX_PREFILL_CHUNK` | Prompt tokens per prefill chunk (default and maximum 2048). |
| `QX_SPLIT` | Batches of several sequences may run as two interleaved halves. `auto` (the default): from 4 tokens, split or not per batch width, whichever measures faster. `N`: always split from N tokens. `0`: never. |
| `QX_ROUTE_CACHE=J,M` | Experimental and lossy, off by default. Cache-aware routing: each token keeps its top J experts and fills the rest of its 10 from its top M, preferring experts already in VRAM. Measured at 8 / 16 users against off: `6,16` gives +28% / +80% for +3.0% perplexity; `3,16` gives +48% / +121% for +4.7% ([docs/progress.md](docs/progress.md)). |
| `QX_HALF_LIFE`, `QX_SWAPS` | The expert swapper's routing-score half-life (tokens) and swaps per token. |
| `QX_MODEL` | Model path for the tools: a pack or a GGUF. |

## How it works

- **Experts.** Every expert sits in pinned RAM (63 GiB for NVFP4). The hottest ones also sit in a VRAM cache, about
  6,400 of 24,576 in production. A swapper in the CPU driver thread ranks experts by decayed routing frequency and
  copies hot experts in over cold ones. A prompt's routing seeds it before decoding starts.
- **One decode step without the host.** The GPU posts each layer's missing experts to a mapped mailbox. The CPU
  pool (AVX-512 VNNI, bit-equal to the GPU kernels) computes them while the GPU computes the resident ones. Steps
  are CUDA graphs, and tokens are sampled on the GPU.
- **Batches.** One sequence's windows of up to 8 tokens, and batch halves of up to 4, use fused matrix-vector
  kernels with int8 activations. Wider batches use int8 tensor-core kernels (`qx_tmv_q8`). Batches of several
  sequences run split in two halves, so GPU and CPU work overlap.
- **Memory.** The KV cache is FP8 E4M3 with a scale per 32 dimensions. It is paged in pinned RAM, and the most
  recent cells are mirrored in VRAM. The prompt buffers are borrowed from the expert cache's tail during prefill.
  The PLE n-gram table is read from the SSD with io_uring and O_DIRECT, behind a RAM row cache.

## What is in the repository

| Crate | Contents |
|---|---|
| `qx-gpu` | CubeCL runtime glue and device memory. Its build script flattens `kernels/*.cu` and checks them with nvcc; at run time each module is also checked once with NVRTC, so a kernel that fails to compile fails loudly. |
| `qx-engine` | The engine: the forward pass, the expert cache and its swapper, sequence slots and batched decode, chunked prefill, paged FP8 KV, the prefix cache, speculation and the PLE reader. Also the check and bench binaries. |
| `qx-cpu` | The CPU expert pool: AVX-512 VNNI kernels for NVFP4, IQ2_S, IQ3_S and IQ4_NL, each bit-equal to its GPU kernel. |
| `qx-quant`, `qx-pack` | Dequantizers, the GGUF reader, and the pack writer and verifier. |
| `qx-ref` | The f32 CPU reference of the whole model, which the engine is checked against. |
| `qx-tok` | The tokenizer, which matches llama.cpp token for token. |
| `qx-serve` | The OpenAI-compatible server and its continuous-batching scheduler. |

`kernels/` holds the CUDA sources. `tools/` holds:

- `loadgen.py`: concurrent load at c=1/4/8/16;
- `longprompt.py`;
- `pardl.py`;
- `golden/`: llama.cpp golden logits;
- `ds-checks.sh`, `capped.sh`: the DeepSeek engine's regression checks, and a memory-capped scope to run heavy jobs in;
- `logits_compare.py`: teacher-forced top-1, KL and perplexity between runs (logits dumps or Strata score rows).

`perf/` is the measurement harness for the DeepSeek engine: `harness/bench.sh` (the standard cases, three repeats,
spreads, a correctness gate), `perf_diff.py` (gate against the promoted baseline, promote, score), the scoreboard,
the notebook of every experiment (`optimization_status.md`), `findings.md` (what is known and what was rejected)
and `backlog.md`.

## Checks

Each tool loads the model and checks one property:

| Tool | What it checks |
|---|---|
| `qx-check` | Engine logits against the f32 reference, per position and per activation. Needs a GGUF. |
| `qx-invariance` | Two expert-cache sizes (different experts on the GPU and the CPU) give bit-identical logits. |
| `qx-window-check` | Multi-token windows equal one-token steps, bit for bit. |
| `qx-batch-check` | Slots decoded in mixed and split batches equal each slot decoded alone. |
| `qx-prefix-check` | A restored prefix continues exactly as an uninterrupted run. |
| `qx-spec-check` | Speculative decoding gives exactly the plain-decoding tokens, greedy and sampled. |
| `qx-prefill-check` | Batched prefill agrees with token-by-token decode. |
| `qx-gen-check` | The production generation paths produce the eager forward pass's tokens; "The capital of France is" must give " Paris". |
| `qx-nvfp4-check` | The GPU and CPU NVFP4 expert kernels agree bit for bit on random records. |
| `qx-ppl` | Next-token NLL, expert hit rate, and KL and top-1 agreement against a saved run: the quality gate for lossy modes such as `QX_ROUTE_CACHE`. |
| `qx-batch-bench` | Decode throughput at several batch sizes, with the CPU pool's share of each step. |
| `qx-ds-check` (`tools/ds-checks.sh`) | The DeepSeek engine: the f32 mode against the CPU reference (exact), the production numerics against it (KL, top-1), windows of 8 and 64 against one-token steps (bit-identical), the tokenizer against llama.cpp, rewind and park/unpark against the straight pass (bit-identical). It also saves and resumes 256k states and dumps logits for `tools/logits_compare.py`. |

**Parity checks are not enough on their own.** The first NVFP4 GGUF passed every one of them while producing
nonsense: the converter had dropped the PLE table's scale, so the reference, the engine and llama.cpp all agreed
on a broken file. Before benchmarking a new model file, also check:

- the golden logits' next-token accuracy on their own text;
- a few tensors of each kind against the original checkpoint;
- a question with a known answer, asked through the server.

## License

MIT ([LICENSE](LICENSE)). The models keep their own licenses: Qwen3.8-Flash-Next under the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), DeepSeek-V4.1-Flash under
DeepSeek's model license. `third_party/cubecl-cuda` is a patched copy of the cubecl-cuda crate under its own license
(MIT/Apache-2.0). QuixiInfer learns from DeepInfer, Strata, Basalt, colibri, Big-MoE-Small-Box and llama.cpp (see
[docs/kernel-references.md](docs/kernel-references.md)), but it contains no code from them. The IQ lookup grids are
ggml's data tables.
