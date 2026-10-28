# Kernel references

QuixiInfer writes every kernel itself, under `kernels/` (GPU) and `crates/qx-cpu` (CPU). This page lists, for each
kernel, the example code worth reading first. Nothing listed here is linked, vendored or loaded by QuixiInfer.
Paths are relative to the home directory. Surveyed 2026-10-24.

## Per kernel

| QuixiInfer kernel | Read first | What to take from it |
|---|---|---|
| Model graph, op order | `llama.cpp/src/models/qwen4exp.cpp` | Exact op order and constants; it is also the offline golden oracle |
| IQ3_S, IQ2_S, IQ4_NL decoders and tables | `llama.cpp/ggml/src/ggml-common.h`, basalt `include/basalt/kernels/iq_kernels.hpp` | Block layouts and codebooks |
| Routed-expert GEMV, decode | basalt `native_moe.hpp`, `dec_moe.hpp`; llama.cpp `ggml-cuda/mmvq.cu` | int8 activations with dp4a or `mma`; grouping the 10 experts of a token |
| Routed-expert GEMM, prefill | basalt `src/prefill/moe_fused_iq.cu`, `moe_mmq.cu`; llama.cpp `ggml-cuda/mmq.cuh` | Tile shapes, expert-sorted token gather |
| Hit/miss expert split and combine | basalt `src/core/expert_cache.cpp`, `verify.cpp` | Residency table, mapped-memory handoff to the CPU |
| CPU expert dot products | llama.cpp `ggml-cpu/arch/x86/quants.c`; basalt Q8_K activation quantizer | AVX-512 unpacking of IQ codes, VNNI accumulation |
| Sparse lightning indexer | QuixiCore-CUDA `kernels/serving/indexer_logits_*.cuh`; llama.cpp `ggml-cuda/lightning-indexer.cu` | Score layout, ReLU-weighted head sum, block pooling |
| Top-k block selection | basalt `native_qsa_indexer.hpp` | Top-k over pooled block scores |
| Gated DeltaNet decode | QuixiCore-CUDA `kernels/serving/kda_decode_kernels.cuh`; llama.cpp `ggml-cuda/gated_delta_net.cu` | State layout in registers and shared memory; QuixiCore's is KDA, with a per-channel gate instead of this model's scalar gate |
| Gated DeltaNet prefill, chunked | llama.cpp `ggml-cuda/gated-delta-net-mma.cu`; QuixiCore-CUDA `kernels/lin_attn_tm/gdn_kernels.cuh` | Chunked recurrence on tensor cores |
| Short conv before DeltaNet | llama.cpp `ggml_ssm_conv` CUDA op | Rolling conv state |
| Hyper-connections | QuixiCore-CUDA `kernels/serving/mhc_ampere.cuh`; llama.cpp `ggml-cuda/dsv4-hc.cu` | Fused pre-mix and post-mix; warp split of partial sums |
| PLE n-gram block | llama.cpp `qwen4exp.cpp` PLE section; basalt `native_ple_postops.hpp` | Hashing, gather, conv and key/value projections |
| FP8 sparse attention | QuixiCore-CUDA `kernels/serving/attn_q_kernels.cuh` | In-loop dequant pattern only; the SM120 FP8 `mma` design is new |
| FP8 KV append and quantize | QuixiCore-CUDA `sparse_serving_kernels.cuh` (FP8 K quant and cache), basalt `kv_q8.hpp` | Per-head scale and page addressing |
| KV page gather and LSE merge | QuixiCore-CUDA `kv_cache_kernels.cuh`, `sparse_serving_kernels.cuh`; basalt `kv_stream.hpp` | Paged addressing, merging partial softmax |
| Dense GEMV and GEMM (Q6_K, Q8_0, BF16) | basalt `native_mmvq.hpp`, `dense_tc.hpp`; QuixiCore-CUDA `kernels/quant/qgemv.cu` | K-split for decode occupancy |
| RMSNorm, RoPE, elementwise | QuixiCore-CUDA `kernels/layernorm`, `kernels/rotary` | Simple ones can be CubeCL kernels instead |
| Sampling, penalties, grammar mask | QuixiCore-CUDA `kernels/serving/sampling_kernels.cuh`, `kernels/quant/lm_head_topkp.cu` | Seeded draws as a pure function; top-p by bisection without a sort |
| Speculative verify | QuixiCore-CUDA `kernels/serving/eagle_kernels.cuh` | Exact rejection with pre-drawn uniforms |
| PLE reader (CPU, io_uring) | basalt `src/ngram/ple_reader.cpp` | Page dedup, in-flight cap, row cache, keep-alive reads |
| Scheduler, async step loop | DeepInfer `csrc/engine/scheduler.hpp`, `deepinfer/engine/llm_engine.py` | Step-type vote, deferred release, overlap of schedule and run |
| Variable-length verify | DeepInfer `tilelang_ops/schedule_mtp_varlen.py`, `doc/inference_deployment_qa.md` | Picking the verify prefix from acceptance estimates and a throughput curve |

## Caveats about the examples

- QuixiCore-CUDA was measured almost entirely on an RTX 3090. Its timings say nothing about SM120.
- QuixiCore's FP8 paths decode E4M3 in software, because Ampere has no FP8 tensor cores. SM120 does, so our kernels
  use the hardware.
- QuixiCore has no IQ3_S or IQ2_S format, and its i-quant GEMV is lookup-bound. Basalt and llama.cpp show the faster
  approach for these formats.
- DeepInfer's kernels target SM100 and Ascend. Its scheduling and algorithms carry over; its kernels do not.
