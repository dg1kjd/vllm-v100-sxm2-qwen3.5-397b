# 1Cat-vLLM 1.2.1

1Cat-vLLM 1.2.1 is a V100/SM70 release built on upstream vLLM metadata
`0.21.1rc1.dev438+g4ff865c38.d20260603.cu128` with baseline commit
`4ff865c38`. The 1Cat release version is `1.2.1`.

This release keeps the 1.2.0 V100 migration and adds final patch hardening for
the public wheel.

## Highlights

1. Updated upstream vLLM base
   SM70/V100 support has been re-integrated into the newer backend,
   quantization, attention, CUDA graph, serving, OpenAI API, Qwen hybrid model,
   and scheduler paths instead of being carried as patches on the old 0.0.3
   tree.

2. Flash-V100 attention backend
   `FLASH_ATTN_V100` is the recommended V100 attention backend. It covers
   prefill, decode, paged KV, CUDA graph, and long-context serving on SM70.

3. V100 Marlin path
   0.0.3 Marlin required SM75+ and could not run on V100. The SM70 Marlin path
   is now enabled for V100 and can reuse the TurboMind dense/MoE fast paths
   where applicable.

4. TurboMind SM70 quantization paths
   AWQ, AWQ MoE, FP8, FP4, NVFP4, and MXFP4 routes are wired into the updated
   vLLM runtime. `VLLM_SM70_QUANT_BACKEND=marlin` and
   `VLLM_SM70_QUANT_BACKEND=turbomind` can be used to select routes explicitly.

5. Long-context decode and CUDA graph stability
   Flash-V100 uses graph-safe dynamic partition metadata and a conservative
   default partition policy. The public V100 default is no-MTP plus the
   Flash-V100 compile-graph fast path for long-context stability and decode
   throughput.

6. Long-context prefill optimization
   D=256 dense WMMA-QK is enabled by default for no-prefix dense prefill. The
   D=256 paged-prefix exact path uses a low-smem implementation plus page-id and
   page-offset caching to reduce chunked-prefill overhead.

7. Experimental BFLA sparse prefill
   `VLLM_FLASH_V100_BFLA_PREFILL=1` enables an experimental approximate sparse
   prefill route. It can provide large long-context prefill speedups, but it is
   not an exact attention path and remains default-off.

8. Qwen hybrid, GDN, prefix cache, and tool calling
   Qwen3.5/Qwen3.6 hybrid model support, GDN state handling, prefix cache,
   OpenAI-compatible serving, and Qwen tool parsing have been tightened for V100
   public serving.

9. V100 memory policy
   Public profiles target 256K context with more conservative
   `max_num_batched_tokens`, `max_num_seqs`, KV cache allocation, and CUDA graph
   capture settings to reduce 32 GB V100 OOM risk.

10. Multimodal serving defaults
    Image input is enabled by default on the V100 public path. Video remains
    disabled unless configured explicitly.

## 1.2.1 Patch Hardening

1. MTP is no longer enabled implicitly on SM70. Users can opt in with
   `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1` or an explicit speculative config.

2. NVFP4/MXFP4 default routing now uses the SM70 TurboMind path unless the user
   explicitly disables it or selects Marlin.

3. FlashInfer sampler import failures fall back to the native sampler by
   default, while `VLLM_USE_FLASHINFER_SAMPLER=1` still fails loudly for
   diagnostics.

4. V100 repetition, presence, and frequency penalty requests avoid unsupported
   custom CUDA kernels and use the torch fallback on SM70.

5. Prefix-cache release checks now require an actual cache-hit pass instead of
   accepting a skipped or missing probe.

6. The wheel build bundles the Flash-V100 extension modules together with the
   vLLM CUDA extensions.

## Build Target

The public 1.2.1 wheel is built for CUDA 12.8, Torch 2.10, and SM70/V100.
