# Model facts

Read from the GGUF headers of `unsloth/Qwen3.8-Flash-Next-GGUF`, directory `UD-IQ3_XXS`, on 2026-10-24.

## Geometry

| Key | Value |
|---|---|
| Architecture | `qwen4exp` ("A Preview of the Qwen4 Architecture") |
| Layers | 48 |
| Hidden size | 2560 |
| Layer pattern | 3 gated-DeltaNet (linear attention) layers, then 1 full-attention layer; 12 full-attention layers in all |
| Full attention | 24 query heads, 2 KV heads, head dim 256, partial RoPE on 64 dims (multi-section), base 1e7 |
| Sparse attention | Lightning indexer on every full-attention layer: 4 heads, key dim 128, top-k 2048, compress ratio 4 |
| Linear attention | Conv kernel 4, state size 128, 16 groups, 48 value heads (time-step rank), inner size 6144 |
| MoE | 512 routed experts, 10 active, expert FFN 640; one shared expert, FFN 640 |
| Hyper-connections | 4 streams, low rank 320 |
| PLE (n-gram embedding) | Layer 1 only; 3-gram; 16 hash heads of about 20M rows each; 160 dims per row |
| Context | 262,144 |
| Vocabulary | 248,320 tokens |

## Tensor sizes

| Group | Size | Formats |
|---|---|---|
| Whole file (3 shards) | 76.3 GiB | |
| PLE table `per_layer_token_embd` | 26.8 GiB | IQ4_NL, 90 bytes per row, 320,001,536 rows |
| Routed experts, down | 21.1 GiB | IQ4_NL |
| Routed experts, gate | 12.1 GiB | IQ3_S and IQ2_S by layer |
| Routed experts, up | 12.1 GiB | IQ3_S and IQ2_S by layer |
| Everything else (dense) | about 4.2 GiB | Q6_K, Q8_0, BF16, F32 |
| MTP drafter (separate file, Q8_0) | 3.9 GiB | Q8_0 |

## Derived figures

| Figure | Value |
|---|---|
| Routed expert size, per layer and expert | about 1.9 MiB |
| Routed expert bytes read per token | about 0.89 GiB |
| PLE reads per token | 16 rows, so 16 random 4 KiB pages |
| FP8 KV per token, 12 layers, K and V | about 12 KiB, so about 3 GiB at 262k context |
| Indexer pooled keys at 262k context, fp32 | about 0.4 GiB |
| KV cells read per attention layer per decode step | at most 2048, so about 2 MiB in FP8 |

## Reference quality

Basalt's measurement of the same quant, decode path, 20 wikitext-2 chunks against the Q8_0 reference:

| PPL | KLD mean | Same top-1 |
|---|---|---|
| 2.564 | 0.244 | 87.0% |
