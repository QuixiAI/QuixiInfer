# QuixiInfer

QuixiInfer serves [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
at UD-IQ2_XXS (labelled 748B-A16B in the GGUF: 748 B parameters, 16 B active per token; six GGUF shards and a
282 GB pack; nearly half of the bytes are the Engram table at Q5_K, the routed experts are IQ1_S to IQ3_XXS by
layer) at the model's full 262,144-token context
from a single workstation: one RTX 5090 (32 GB), a 16-core CPU, 128 GB of RAM and an NVMe SSD. Its 15,360
routed-expert records (about 9.5 MB each) live in three tiers: what the VRAM
has left after the context and the slots (12.3 GiB, 1,275 records, at three 262k slots, plus 4.1 GiB lent from
the slots' unused tails), a pinned RAM tier (95 GiB in production: 9,590 records), and the rest read from the
pack on NVMe as routed. The tiers adapt to what requests route to. Several requests run at once over sequence
slots; a request sharing a prefix with an earlier one continues from it; parked sequences resume bit-exactly;
structured outputs are constrained by a grammar engine. Images are taken as the model was
trained to take them ([Image input](#image-input)).

It is written in Rust, with its own CUDA kernels, compiled at build time and launched through CubeCL. No ggml or
llama.cpp code is linked. The server design comes from DeepInfer, DeepSeek's cluster engine, scaled down to one GPU
with CPU expert offload and an NVMe tier:

- continuous batching over sequence slots;
- asynchronous scheduling;
- prefix reuse;
- exact speculative decoding;
- structured outputs;
- image input (the model's vision tower).

What it does not do (yet): fetching image URLs (images are sent inline), the Anthropic API, multiple GPUs.

## Where it fits

Several engines run this class of model, and each optimizes for something different:

| Engine | Built for | Hardware | Concurrency | What it optimizes |
|---|---|---|---|---|
| **QuixiInfer** | A shared server on one workstation | One NVIDIA sm_120 GPU, Linux | Continuous batching, 3 slots of 262k each | Several users on one box at full context; decode that does not depend on scheduling |
| **Basalt** | One user on one Blackwell workstation | NVIDIA sm_120, Linux; an optional second GPU as an expert tier | `--parallel 1..8`, context shared across slots | Single-stream decode speed (tuned packs, self-distilled drafter) |
| **Strata** (Basalt's parent) | One user on a gaming PC | NVIDIA, AMD, Apple and Intel GPUs, 12 GB of VRAM and up; Windows, Linux, macOS | One request by default | Running on ordinary hardware; apps, APIs, installer |
| **[colibri](https://github.com/JustVugg/colibri)** | Models far bigger than RAM, on any machine | CPU first; CUDA, Vulkan and Metal tiers | One request | Reach: experts streamed from NVMe, so models far larger than RAM still run |
| **DeepInfer** | DeepSeek's production cluster | SM100 GPUs, Ascend NPUs, 4-64 per deployment | Data and expert parallelism | DeepSeek-V4.1-Flash at data-center scale |
| **llama.cpp** | General local inference | Almost anything | `--parallel` | Breadth of models and hardware. QuixiInfer uses it offline only, as a reference oracle (tokenizer, golden logits) |

## Performance

**Measured** (RTX 5090, Ryzen 9 9950X, 2 x 64 GB DDR5-5600, win10 VM off; the harness in `perf/`, promoted rows
in `perf/scoreboard.md`; the figures are the gate runs of 2026-11-19 against the baseline of 2026-11-15, score
18.475):

| | ms per token | tok/s |
|---|--:|--:|
| Decode at 1k context (one stream) | 31 | 32 |
| Decode at a 256k position (one stream, resumed state) | 39 | 26 |
| Server at 262k context, 3 slots: one request (per decode step; the case sits at one of two levels) | 33-37 | 27-30 |
| Server at 262k context, 3 slots: three concurrent requests (per request) | 81 | 12.4 each, 37 aggregate |
| Prompt, the last 8k tokens of a 256k prompt (8192-row chunks) | 2.0 | 500 |
| Prompt in windows of 128 tokens, below 1k | 10.5 | 95 |
| A 16k prompt from empty (2026-11-15; kernels compiled) | 0.72 | 1,392 (1,132-1,158 as a fresh process, 2026-11-12) |
| Time to first token, a 12.5k-token prompt on the 262k / 3-slot server | 14.3-15.9 s | 790-880 (prompt rate) |

Long prompts run through a chunked, expert-major path (shorter ones in windows of up to 128 tokens): each chunk of up to 8,192 rows goes
layer by layer, and every expert the chunk needs is streamed once per layer into tensor-core kernels, so a
prompt's cost does not grow with the number of distinct experts per window. Long prompts also skip the upper
layers for positions outside the tail they feed forward (`QX_DS_SKIP=0` turns that off).

**Decode beyond one token a step.** A lone request drafts its next tokens from text it has already seen (its own
prompt and reply, and the finished replies of other requests) and verifies the draft in the same step; a
rejected draft is rewound. Replies are the same bytes with it on or off. Measured on one client: a file edit
27.3 -> 17.0 s, a quotation 12.8 -> 8.7 s, an essay asked a second time 10.8 -> 6.4 s; text with nothing to
copy is at parity, and two cases read 3-5% slower (a JSON list reusing another reply's values under other keys,
a sampled essay). `QX_DS_SPEC=0` turns it off, `QX_DS_SPEC_CORPUS=0` keeps drafts to the request's own text (on
a server shared between parties, reply speed can otherwise reveal that someone else produced the same text).

**Exactness.** What is bit-identical to one-token decode, checked by `tools/ds-checks.sh` (nine checks, run
before a change lands): windows of 3, 8 and 64 tokens, windows over several sequences, rewinds (by replay and
from checkpoints) and parked and resumed sequences; the engine's f32 mode matches the CPU reference exactly.
Prompt chunks are not bit-identical: since 2026-11-10 their dense products and attention run as f16 tensor-core
kernels, held to a bound against the exact path (mean KL at most 0.03, top-1 at least 95%; the pieces measured
KL 0.009-0.019 and top-1 95-96% when they were introduced). So the reply to a prompt prefilled afresh can differ from the reply when its prefix is reused;
`QX_DS_PREFILL_EXACT=1` restores exact chunks, which are slower (the scored 256k prompt case read 5.2-5.7 ms
per token before the f16 kernels and the changes that followed them, 2.0 now). The production decode numerics (int8
activations for the dense projections and the expert kernels) sit at KL 0.011-0.020 from the f32 reference with
96% top-1 agreement and perplexity +0.1% over 1,023 teacher-forced positions (`tools/logits_compare.py`,
2026-11-07).

**Against Strata, measured on this machine, same model files and token ids** (2026-11-15,
one run each, VM off; `perf/optimization_status.md` of that day): QuixiInfer is ahead on prompts and on decode.
A 16,384-token prompt: 11.7 s (1,392 tok/s) against Strata's 19.2 s (854); 2,048 tokens: 3.1 s against 10.4 s;
128 tokens: 1.1 s against 4.0 s. The 64 tokens after a prompt: 25-27 tok/s against 16.5-18.9. Steady decode at
1k context (2026-11-11): 32.4-32.7 tok/s against 27.2. Caveats: QuixiInfer's prompt time covers all but the last
64 tokens (those run as single steps), and Strata's process was cold while QuixiInfer's kernels were compiled;
as fresh processes on the 16k prompt it was 1,132-1,158 tok/s against 841-880. Strata has no server, so there is
no concurrent comparison. Quality, teacher-forced on the same ids against the f32 reference over 1k positions
(2026-11-07): QuixiInfer's decode path is at KL 0.011-0.020 and 96% top-1 (perplexity +0.1%); Strata's at KL
0.025-0.046 and 93-94% (perplexity within ±2%). Details in `perf/findings.md`.

## Quick start

Requirements:

- Linux x86-64 with AVX-512 VNNI;
- an NVIDIA GPU: developed and measured on an RTX 5090 (sm_120); the kernels also build for the RTX 40 series
  (sm_89) and the RTX 30 series (sm_86), see "Other GPUs" below;
- CUDA 13 (`nvcc` at `/usr/local/cuda-13.3/bin`, or set `QX_NVCC`);
- a recent stable Rust;
- about 128 GB of RAM (the process is about 107 GB with a 95 GiB RAM tier and the vision tower);
- the pack (282 GB) on an NVMe drive;
- for image input, `mmproj-F16.gguf` from the same model repository.

**1. Build.**

```sh
cargo build --release      # target/release/quixiinfer, qx-pack and the check tools
target/release/quixiinfer --help   # the commands and every flag; loads nothing
```

**Other GPUs (RTX 40 and 30 series).** The engine reads the card's compute capability at start and compiles its
kernels for it (the same target CubeCL uses: `sm_120a` on Blackwell, `sm_89` / `sm_86` on Ada / Ampere). The
expert tier is what is left of VRAM after the dense weights, the context slots and the staging area, so a 24 GB
card at 256k context holds few experts and serves most from RAM (set `--ctx` and `--slots` to what the card can
hold; `QX_DS_STAGE_CAP` shrinks the staging area); PCIe 4.0 lands a 9.5 MB expert record in 0.4 ms against 0.2 on
PCIe 5.0, which moves the best staging share (`QX_DS_STAGE_SHARE_DECODE`) toward the CPU pool.
`QX_NVCC_ARCHS=sm_86,sm_89,sm_120a cargo build` compiles every kernel for all three at build time (the default
check is `sm_120a` alone; the check takes about 20 minutes per target the first time). Only the 5090 has been
run here; the other two are build-checked.

**2. Make the model pack.** Download the six UD-IQ2_XXS shards (`tools/pardl.py` is a parallel, resumable
downloader), then write and verify the pack:

```sh
# the pack from the six GGUF shards (pass the first), once
target/release/qx-pack pack   DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp
target/release/qx-pack verify DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf
```

**3. Serve.**

```sh
# serve at the full context: the engine is chosen by the GGUF's architecture; cap the memory, the tiers pin most of the RAM
systemd-run --user --scope -p MemoryMax=112G -p MemorySwapMax=0 -- \
  env CUDA_CACHE_MAXSIZE=4294967296 QX_DS_VRAM=20 QX_DS_RAM=95 QX_DS_PROMOS=32 \
  target/release/quixiinfer serve --model DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --pack DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp --ctx 262144 --slots 3 --port 8000 \
  --profile golden/profile-dsv41.txt
```


That is the benchmarked and production configuration; the process is about 107 GB (105 without the vision tower) and is ready about 35 s
after start. `--profile` ranks the experts the tiers are filled with at start (`golden/profile-dsv41.txt` comes
from a routing trace of a mixed corpus and is the default when present); `--profile-out FILE` writes the live
ranking after requests, and a start finding that file loads from it instead. `--slots` (2) sets how many
requests run at once; each 262k slot takes VRAM from the expert tier. `--window` (128) is the prompt window for
prompts too short for a chunk. The chat template and tokenizer come from the GGUF (tokenization checked against
llama.cpp). The expert tier leaves `QX_DS_VRAM_RESERVE_MIB` (2560) of device memory for the kernel modules and
the window graphs, and a kernel that would need compiling after the warm-up aborts the server (it means a module
failed to load). `CUDA_CACHE_MAXSIZE` raises the driver's JIT cache from its 1 GiB default: the engine's kernel
modules evict each other from it, and a cold start then spends 10-20 minutes in ptxas.

The server refuses to listen beyond 127.0.0.1 without `--api-key KEY`.

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

**5. Run it as a service.** `tools/quixiinfer-deepseek.service.example` is the
production unit with the paths made generic. The service's memory cannot be swapped, so a second large tenant
(here a 16 GiB virtual machine started now and then) can end in an out-of-memory kill. With `QX_DS_RAM_YIELD=12`
the coldest 12 GiB of the RAM tier sit in arenas of their own; while `QX_DS_YIELD_PATH` names a live process
(the VM's pid file) or the host has under 4 GiB available, the service frees them and reads those records from
NVMe, and 20 s after the path is gone it pins them again and reads the records back. Replies are the same
bytes throughout; a benchmark run with the 12 GiB out scores 5.4% slower (decode at 256k 42.2 ms against 39.0),
and each hand-over pauses requests for 0.5-1.7 s. Measured with a stand-in for the VM (a pid file, and a 13 GiB
allocation), not yet with the VM itself; the journal's `ram tier:` lines record each hand-over.

Stop the service before benchmarking: two engines cannot both hold the GPU and pin the experts.

### Image input

DeepSeek-V4.1-Flash is an image-text model; its vision tower ships with the GGUFs as `mmproj-F16.gguf` (971 MB;
`--mmproj FILE`, or left beside or above the model's directory, where the server finds it). A chat message's
`image_url` parts (as `data:` URLs: PNG, JPEG, WebP, GIF, BMP) and Anthropic-style `image` parts are fitted,
encoded and placed in the prompt as the model's own code does it (`inference/image_processor.py`, `vision.py`):

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 picture.png)"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false}}'
```

- **What an image costs.** It is fitted to a multiple of 14 pixels (small images are enlarged to about 544 x 544,
  large ones reduced), and takes `rows x (columns + 1) + 2` positions of the context, a row and a column per 42
  pixels: about 200 positions for a 640 x 480 picture, at most 1,024 (reached from about 1.8 megapixels: 1280 x 960 takes 738, 1440 x 1080 takes 938). The vision tower runs on the CPU, one image at a time: 2-3 s for a small picture,
  19-24 s for a large photograph, and requests from all clients wait their turn; a decode running beside a large encode
  measured 22 against 28 tok/s meanwhile. An image sent again is not encoded again (the last images' blocks are
  kept, 335 MB at most), and a prompt that repeats an image and the text before it reuses that prefix like any
  other (a second turn about a 977-position image: 0.7 s).
- **In the language model** an image position takes its embedding from the vision block, the router's image
  bias (`exp_probs_b_vl`), no Engram contribution, and is dead to the Engram's n-grams, as in the reference.
  `qx-ds-img-check` verifies each of these numerically.
- **Limits.** 16 images and 20 MB per image in one request; `QX_DS_IMG_ROWS` (16,384) image positions held at
  once across the live contexts, 335 MB of pinned memory (the contexts of idle slots give theirs up first; a
  request that cannot fit is refused with a 400). Image URLs are not fetched: send the image inline. As in the
  model's own code, EXIF orientation is not applied (a phone's portrait photograph arrives sideways) and
  transparency is dropped. HEIC, AVIF and TIFF are not decoded; an animated GIF gives its first frame. Started
  with `--mmproj none`, or without the file, the server refuses requests with images.
- **Checked against** a NumPy + Pillow reference of the tower (`tools/vision_ref.py`): the fitted pixels are
  byte-identical to Pillow's for 8-bit PNG input, the block's values agree to 4e-5. A JPEG decodes up to 4
  levels apart from Pillow's decoder, and a 16-bit PNG is scaled where Pillow clips. The reference script is
  by the same hand as the tower: the independent evidence is that the model reads text, counts and describes
  photographs correctly through it.

### Settings

| Setting | What it does |
|---|---|
| `QX_DS_CHUNK` | Rows per prompt chunk: 8192 by default at 32k of context and above, off below (prompts run in windows); `0` turns chunks off. |
| `QX_DS_PREFILL_EXACT=1` | Exact kernels for prompt chunks (bit-identical to single steps), slower. |
| `QX_DS_SKIP` | `0` runs every layer for every prompt position (the default skips the upper layers for positions outside the tail they feed forward; exact for the tokens that follow). |
| `QX_DS_PREFIX=0` | No prefix reuse: every prompt is computed from its start. |
| `QX_DS_SPEC`, `QX_DS_SPEC_CORPUS` | Draft length cap for speculation (15; `0` off) and the tokens of other requests' finished replies kept to draft from (262144; `0` off). |
| `QX_DS_ADMIT_WINDOW` | Among this many waiting requests (4), the one sharing the longest prefix with a free slot is admitted first; the front request is passed over three times at most. |
| `QX_DS_ROUTE_RES=m`, `QX_DS_ROUTE_L0` | Lossy and off by default: the router's score of experts already in VRAM is multiplied by `m` from layer `L0` up. `1.3` from layer 20 measured 5-6% off decode at 256k for KL 0.010 and 96.4% top-1; replies then depend on what is resident. |
| `QX_DS_RAM_YIELD`, `QX_DS_YIELD_PATH`, `QX_DS_YIELD_LOW`, `QX_DS_YIELD_HIGH`, `QX_DS_YIELD_HOLD` | GiB of the RAM tier that can be handed back to the host (off when unset), the path whose live process asks for it, the available-memory mark below which it goes anyway (4 GiB), the room above its own size needed before it returns (8 GiB) and for how long (20 s). |
| `QX_DS_STAGE_SHARE`, `QX_DS_STAGE_SHARE_DECODE` | Of 8: how many of the RAM-resident missing experts the GPU takes by DMA staging in windows (8) and in single decode steps (3). |
| `QX_DS_VRAM`, `QX_DS_RAM` | The tiers in GiB (`--vram-gib` 21, `--ram-gib` 100; the VRAM figure is a cap, the context and slots come first). |
| `QX_DS_PROMOS` | Expert promotions into VRAM per token (4; the benchmarked configuration uses 32). |
| `QX_DS_TOPK_KERNEL=1`, `QX_DS_IDX_KERNEL=1`, `QX_MGEMV_ZGROUPS=0` | The previous indexer top-k, scorer and dense-GEMV kernels, for A/B runs. |

## How it works

- **Three expert tiers.** Each of the 15,360 routed-expert records is in VRAM, in pinned RAM or only on NVMe. A
  driver thread ranks experts by decayed routing frequency and promotes hot records over cold ones, NVMe to RAM
  and RAM to VRAM; `--profile` seeds the ranking at start.
- **One decode step.** The GPU posts each layer's experts that are not in VRAM to a mapped mailbox. A share of
  those in RAM is copied into a VRAM staging area by DMA; the CPU pool (AVX-512 VNNI) computes the rest from
  pinned memory; records on NVMe are read with io_uring and O_DIRECT into pinned landing places first. Steps and
  windows are CUDA graphs.
- **Prompts.** Long prompts run in chunks of up to 8,192 rows, layer by layer, each needed expert streamed once
  per layer into tensor-core kernels; the chunk's buffers borrow the tail of the VRAM expert arena for the
  prompt's duration.
- **Scheduling.** Requests occupy sequence slots; a new request takes the slot that shares the longest prefix
  with its prompt and continues from there. A lone decoder speculates (drafts verified in the same step).

## What is in the repository

| Crate | Contents |
|---|---|
| `qx-gpu` | CubeCL runtime glue and device memory. Its build script flattens `kernels/*.cu` and checks them with nvcc; at run time each module is also checked once with NVRTC, so a kernel that fails to compile fails loudly. |
| `qx-engine` | The engine (`dsv41/`): the forward pass, the expert tiers and their driver thread, sequence slots, windows and prompt chunks, rewind and park. Also the check and bench binaries. |
| `qx-cpu` | The CPU expert pool: AVX-512 VNNI kernels. |
| `qx-quant`, `qx-pack` | Dequantizers, the GGUF reader, and the pack writer and verifier. |
| `qx-ref` | The f32 CPU reference, which the engine is checked against. |
| `qx-tok` | The tokenizer, which matches llama.cpp token for token. |
| `qx-vision` | The vision tower on the CPU: image fitting (Pillow's bicubic, byte for byte), the ViT and the aligner, from the `mmproj` GGUF. |
| `qx-serve` | The OpenAI-compatible server and its continuous-batching scheduler. |

`kernels/` holds the CUDA sources. `tools/` holds:

- `loadgen.py`: concurrent load;
- `longprompt.py`: one long prompt through the API, with time to first token;
- `pardl.py`;
- `ds-checks.sh`, `capped.sh`: the engine's regression checks, and a memory-capped scope to run heavy jobs in;
- `quixiinfer-deepseek.service.example`: the production unit with generic paths;
- `logits_compare.py`: teacher-forced top-1, KL and perplexity between runs (logits dumps or Strata score rows).

`perf/` is the measurement harness: `harness/bench.sh` (the standard cases, a correctness gate), `perf_diff.py`
(gate against the promoted baseline, promote, score), the scoreboard, the notebook of every experiment
(`optimization_status.md`), `findings.md` (what is known and what was rejected), `presuppositions.md` (the
assumptions tested so far) and `backlog.md`.

## Checks

| Tool | What it checks |
|---|---|
| `qx-ds-check` (`tools/ds-checks.sh`) | Nine checks: the f32 mode against the CPU reference (exact), the production numerics against it (KL, top-1), windows of 3, 8 and 64 against one-token steps (bit-identical, also with the tiers adapting), the tokenizer against llama.cpp, rewind and park/unpark against the straight pass, windows over several sequences, rewinds by a replay chunk and from checkpoints (all bit-identical). It also saves and resumes 256k states and dumps logits for `tools/logits_compare.py`. |
| `qx-ds-img-check` | Image positions in the engine: the embedding is the vision block's row bit for bit, the Engram gates are zero, the router's picks are the top k under the image bias (and under the text bias elsewhere), and windows and an exact chunk over an image give the single steps' logits bit for bit. |
| `qx-vision-check`, `tools/vision_ref.py` | The vision tower on one image, and a NumPy + Pillow reference to compare it with. |
| `qx-tier-check` | The expert store: RAM places and NVMe reads against the source GGUF byte for byte; with `QX_TIER_ELASTIC=G`, every record through churn, release and restore of the elastic part. |
| `qx-ds-replay` | Replays a recorded scheduler trace as single steps and compares every row's hash (the exactness check for speculation and scheduling changes). |

## License

MIT ([LICENSE](LICENSE)). The model keeps its own license (MIT, per its model card). `third_party/cubecl-cuda` is
a patched copy of the cubecl-cuda crate under its own license (MIT/Apache-2.0). QuixiInfer learns from DeepInfer,
Strata, Basalt, colibri, Big-MoE-Small-Box and llama.cpp (see [docs/kernel-references.md](docs/kernel-references.md)),
but it contains no code from them. The IQ lookup grids are ggml's data tables.
