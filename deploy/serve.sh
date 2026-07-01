#!/bin/bash
# Serve Qwen3.5-397B-A17B AWQ on 8x Tesla V100-SXM2-32GB (TP8) with the SM70 tuning.
#
# The GDN full-forward decode-corruption guard is auto-armed by this fork for the
# non-MTP profile; VLLM_SM70_QWEN_GDN_FULL_FORWARD=1 is also set below belt-and-suspenders.
#
# Edit the paths below (or pass them as env vars), then:  ./deploy/serve.sh [PORT]
set -u

# ---- edit these ----
MODEL="${MODEL:-/path/to/Qwen3.5-397B-A17B-AWQ}"    # local dir or HF repo id
PY="${PY:-python}"                                   # python from your venv (NOT the source tree)
WORKDIR="${WORKDIR:-/path/to/workdir}"               # any dir OUTSIDE this checkout
SERVED_NAME="${SERVED_NAME:-Qwen3.5-397B-A17B-AWQ}"
# --------------------

PORT="${1:-8000}"
LOG="${LOG:-/tmp/vllm_397b.log}"

# Refuse to stomp on GPUs that are already in use.
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -gt 0 ]; then
  echo "GPUs busy ($busy compute procs) - not starting:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  exit 1
fi

cd "$WORKDIR"   # never the source checkout (its vllm/ shadows the installed package)

launch() {
rm -f "$LOG"
nohup env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 \
  TORCH_BLAS_PREFER_CUBLASLT=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=2 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=1048576 \
  CUBLAS_WORKSPACE_CONFIG=:16:8 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_SM70_GDN_DELTA_H_BV=16 VLLM_SM70_GDN_DELTA_H_WARPS=4 VLLM_SM70_GDN_DELTA_H_STAGES=1 \
  VLLM_SM70_GDN_CHUNK_O_BK=64 VLLM_SM70_GDN_CHUNK_O_BV=64 VLLM_SM70_GDN_CHUNK_O_WARPS=8 VLLM_SM70_GDN_CHUNK_O_STAGES=2 \
  VLLM_SM70_FP16_GUARD=1 \
  VLLM_SM70_GDN_DECODE_FLASHQLA=0 \
  VLLM_SM70_QWEN_GDN_FULL_FORWARD=1 \
  "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED_NAME" \
  --attention-backend FLASH_ATTN_V100 --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8_e5m2 \
  --max-model-len 140000 --max-num-seqs 8 --max-num-batched-tokens 1059 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching \
  --no-async-scheduling \
  --mamba-ssm-cache-dtype float32 \
  --skip-mm-profiling \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --override-generation-config '{"thinking_token_budget": 4096}' \
  --trust-remote-code --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
PID=$!
}

# KV-fit retry: available KV varies run-to-run at the same util (worst-rank
# non-torch jitter ~0.41-0.50 GiB -> ceiling ~143k-158k). 140000 fits a good
# draw; a bad draw fails the startup KV-fit check loudly, so reroll up to 3x.
for attempt in 1 2 3; do
  launch
  echo "attempt $attempt: launched pid $PID, log $LOG"
  for i in $(seq 1 90); do
    sleep 10
    if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
      echo "READY after ~$((i*10))s; sending warmup"
      curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" >/dev/null
      echo "warm. endpoint http://127.0.0.1:$PORT/v1  (mml 140000, TP8, skip-mm)"
      exit 0
    fi
    if ! kill -0 $PID 2>/dev/null; then break; fi
  done
  if kill -0 $PID 2>/dev/null; then echo "TIMEOUT waiting for startup"; tail -30 "$LOG"; kill $PID 2>/dev/null; exit 1; fi
  if grep -q "estimated maximum model length" "$LOG" 2>/dev/null; then
    echo "attempt $attempt: KV-fit check failed (variance); retrying"; sleep 15; continue
  fi
  echo "server died (not a KV-fit failure), see $LOG"; tail -30 "$LOG"; exit 1
done
echo "giving up after 3 KV-fit failures; lower --max-model-len (log prints the ceiling)"; exit 1
