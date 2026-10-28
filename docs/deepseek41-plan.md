# DeepSeek-V4.1-Flash-1017 in QuixiInfer

The target the user wants: `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` UD-IQ1_S (233 GB) on this machine (RTX 5090 32 GB,
120 GiB RAM, one NVMe). Sources: the official `inference/model.py`, `engram.py`, `kernel.py` (the arithmetic), the
llama.cpp draft PR ggml-org#30817 (`~/llama.cpp-dsv41`, built with CUDA: the oracle for golden logits, KL 4.8e-3
against DeepSeek's own code), and the Unsloth fork PRs #285/#298 for the IQ1 quant types.

## Why IQ1_S fits this box without an NVMe expert tier

From the GGUF headers (shards' tensor indexes): routed experts 107.3 GiB (7.15 MiB per expert: 34 layers IQ1_S
gate/up, 6 layers IQ2_XXS; down 20 layers IQ2_XXS, 20 IQ1_S), the Engram table 103 GiB (IQ4_NL, 768,022,850 rows
x 256), dense 6.7 GiB (Q6_K/Q5_K/Q8_0/F32). Per token: 6 experts x 40 layers = 1.68 GiB of expert weights.

- VRAM: dense 6.7 GiB + KV + scratch, about 18-20 GiB of experts.
- RAM: the other ~88 GiB of experts pinned (as the Qwen engine does), the Engram row cache, mailboxes.
- NVMe: the Engram table only (24 row reads per Engram layer per token, 2 layers), the PLE path as it exists.
- KV per sequence is tiny: a 128-cell window per layer (512 B rows) plus compressed latents at the 4 source
  layers (positions / ratio x 512 B) and index keys (128 floats per compressed position at the 4 KV sources):
  about 400 MB at 262k.

Expected: decode 25-35 tok/s single-stream (CPU pool reading ~0.9 GB of misses per token from RAM), prefill
500-1500 tok/s (the non-resident 88 GiB stream over PCIe per chunk, so chunks of 2-4k tokens), 262k context cheap.
UD-IQ2_XXS (146 GiB of experts) would need an NVMe expert tier and run at 3-6 tok/s; NVFP4 is for the multi-GPU box.

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

## Plan

- **D0. Data and oracle.** Download IQ1_S (`tools/pardl.py`, 24 connections). llama.cpp PR build (done) produces
  golden logits on fixed prompts (slow: 233 GB mapped on 120 GiB RAM, fine for a few hundred tokens).
- **D1. Reference (qx-ref).** The forward pass above in f32 on the CPU from the GGUF: decoders for IQ1_S and IQ2_XXS
  (ggml's grids), Q5_K; Engram hashing from the GGUF metadata (multipliers, primes, offsets, token_map). Exit:
  logits match llama.cpp's within the usual tolerance on the fixed prompts.
- **D2. Engine.** Reuse: expert tiers (RAM pinned + VRAM cache + swapper), Engram on NVMe (the PLE reader with 24
  rows per layer), batching, prefix cache, pack format. New kernels: hc_mixes + Sinkhorn, sparse attention with
  512-wide shared latent and sinks, the compressor, the indexer with candidate blocks, Engram gate, sqrt-softplus
  router, clamped SwiGLU; dense GEMV/GEMM for Q5_K; CPU pool dot products for IQ1_S/IQ2_XXS (AVX-512). Exit: the
  decode and prefill exactness checks as for Qwen, and parity with the reference.
- **D3. Speed.** The adaptive cache profile for this model, prefill chunk size against PCIe staging, DSpark drafting.

Open: disk space (233 GB against 209 GB free: the Qwen GGUF shards (77 GB, the pack keeps the Qwen service) or
SlimServe (15 GB) plus `cargo clean` must go); the Qwen service stays in production until this works.
