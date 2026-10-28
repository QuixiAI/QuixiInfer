# Qwen3.8-Flash-Next forward pass

The computation QuixiInfer implements, read from llama.cpp's `src/models/qwen4exp.cpp` and the ggml ops it uses
(2026-10-24). Shapes are for one token. `rms(x, w)` is `x / sqrt(mean(x^2) + 1e-6) * w`; every norm weight in the
GGUF already has the `1 +` folded in. `E = 2560`, `HC = 4` streams, `HCD = HC*E = 10240`.

## Embedding and residual

- `e = token_embd[tok]` (Q6_K row, 2560).
- The residual is `R[HC][E]`, initialised to four copies of `e`.

## Layer `l` (48 layers; full attention when `(l+1) % 4 == 0`, else gated DeltaNet)

1. **PLE** (layer 1 only): see below. `R = ple(R)`.
2. **HC pre-mix** with `hc_attn_*`: `x, inject = mix(R)`.
3. **Mixer**: `y = attn(x)` or `y = gdn(x)`.
4. **HC combine**: `R[s] += y * w[s]`, `w = 2 * sigmoid(inject / HC)`.
5. **HC pre-mix** with `hc_ffn_*`: `x, inject = mix(R)`.
6. **MoE**: `y = moe(x)`.
7. **HC combine**.

## Head

`x = mix(R)` with `output_hc_norm/down/up` (no inject), `logits = output · x` (Q6_K, 248,320 rows). There is no separate
output norm.

## HC mix

- `xn[s] = rms(R[s], norm[s])` per stream (norm is `[E, HC]`), flattened to `HCD`.
- `lo = silu((down · xn) / HC)`, `down` is `HCD -> 320`.
- `gate = up · lo`, `up` is `320 -> HCD`.
- `x = mean_s(xn[s] * sigmoid(gate[s]))` (2560).
- `inject = inject_w · xn` (`HCD -> 4`, F32).

## Gated DeltaNet (36 layers)

Heads: 16 key heads and 48 value heads of dim 128. Value head `h` uses key head `h % 16`.

- `qkv = attn_qkv · x` (10240: q 2048 | k 2048 | v 6144), `z = attn_gate · x` (6144).
- `beta = sigmoid(ssm_beta · x)` (48), `g = softplus(ssm_alpha · x + ssm_dt.bias) * ssm_a` (48, negative).
- Causal conv, kernel 4, over the 10240 channels with the last 3 inputs as state:
  `c[ch] = silu(sum_k w[k][ch] * in[t-3+k][ch])`. The state keeps the last 3 raw `qkv` vectors.
- `q, k` per key head: `x / sqrt(sum(x^2) + 1e-6)`.
- Per value head, state `S[128 k][128 v]` (fp32):
  `S *= exp(g)`; `d_j = (v_j - sum_i S[i][j] k_i) * beta`; `S[i][j] += k_i d_j`; `o_j = sum_i S[i][j] q_i / sqrt(128)`.
- `o = rms(o_head, ssm_norm) * sigmoid(z_head)` per value head (`ssm_norm` is `[128]`).
- `y = ssm_out · o` (6144 -> 2560).

## Full attention with QSA sparse selection (12 layers)

- `qg = attn_q · x` (12288: per head 256 query then 256 gate, 24 heads). `q = rms(q_h, attn_q_norm)`.
- `k = rms(attn_k · x, attn_k_norm)`, `v = attn_v · x` (2 KV heads of 256). Query head `h` uses KV head `h / 12`.
- RoPE on the first 64 of 256 dims, NEOX pairing `(i, i+32)`, `theta_i = pos * 1e7^(-2i/64)`.
- **Indexer.** `kr = index_k_proj · x` (128) is cached raw per cell. Cells group into blocks of 4 in sequence order;
  a complete block's pooled key is `rope(rms(mean of its 4 kr, index_k_norm), pos of its first cell)`.
  `qi = rope(rms(index_q_proj · x per head, index_q_norm), pos)` (4 heads of 128).
  Block score: `sum_h relu(qi_h · K_b) / sqrt(128)`.
- **Selection** for the token at sequence rank `r`: visible blocks are `0 .. (r+1)/4 - 1`; keep the top
  `min(visible, 512)` by score, plus the tail: the `(r+1) % 4` most recent cells (the token itself included).
- Attention over the kept cells, scale `1/16`, softmax in fp32. `o_h = attn * sigmoid(gate_h)`.
- `y = attn_output · o` (6144 -> 2560).

## MoE

- `p = softmax(ffn_gate_inp · x)` over 512 experts (F32 router). Top 10 by `p`; weights `p_i / sum(p_top)`.
- Expert `e`: `down_e · (silu(gate_e · x) * (up_e · x))` (2560 -> 640 -> 2560).
- Shared expert: same form with `*_shexp`, scaled by `sigmoid(ffn_gate_inp_shexp · x)` (one scalar).
- `y = sum_i w_i expert_i + shared`.

## PLE (layer 1)

- **Hash.** `t0` is the token, `t1, t2` the previous two; a missing predecessor, or anything at or before an EOS
  (248044) in the window, reads as EOS (the token's own EOS does not cut). Image positions use token 248056.
  For `n = 2, 3`: `m = t0*M0 ^ ... ^ t(n-1)*M(n-1)` (u64 wrapping); for each of 8 heads `g`:
  `row = m % vocab[h] + offset[h]`, `h = (n-2)*8 + g`. 16 rows of 160 (IQ4_NL), concatenated head-major to 2560.
- `key = ple_key · emb` (2560 -> 10240), `value = ple_value · emb` (2560 -> 2560).
- `kn = rms(key[s], ple_norm_key[s])`, `qn = rms(R[s], ple_norm_query[s])` per stream.
- `s_s = (kn[s] · qn[s]) / sqrt(2560)`, `gate_s = sigmoid(sign(s_s) * sqrt(max(|s_s|, 1e-6)))`.
- `gated[s] = value * gate_s`; `cn[s] = rms(gated[s], ple_norm_conv[s])`.
- Depthwise causal conv over the 10240 channels of `cn`, kernel 4, dilation 3:
  `conv[c] = silu(sum_k w[k][c] * cn[t - (3-k)*3][c])`; the state keeps the last 9 `cn` vectors.
- `R[s] += gated[s] + conv[s]`.

## MTP drafter (layer 48, separate GGUF)

- `h` is the trunk residual `R` after the last layer, before the head mixer.
- `R' = eh_proj · concat(rms(e, enorm) repeated per stream, rms(h[s], hnorm[s]))` per stream (5120 -> 2560).
- One full-attention layer and one MoE layer as above, with the drafter's own weights and KV.
- Head: `mix` with `nextn.hc_head_*`, then the drafter's `output`.
