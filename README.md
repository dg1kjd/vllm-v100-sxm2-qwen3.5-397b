# 1Cat-vLLM

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判“过时”。
>
> 1Cat-vLLM 是面向 **SM70 / Tesla V100** 的 vLLM 工程分支。项目围绕
> AWQ、注意力后端、长上下文稳定性、MTP 投机解码、运行时默认值和部署
> 路径做了成体系的优化，让更多现代模型场景在 V100 上真正变得可用、
> 好用、能持续部署。
>
> 我们希望把一猫之下在 V100 上的工程经验、优化成果和验证过程贡献给
> 开源社区，也欢迎继续使用 V100 的个人开发者、工作室和团队一起反馈、
> 复现和改进。

1Cat-vLLM is a **Tesla V100 / SM70** focused vLLM fork for serving modern
Qwen-class AWQ and experimental FP8 models on Volta GPUs. It integrates
TurboMind-derived SM70 kernels, a V100 FlashAttention path, runtime defaults
for long-context serving, and OpenAI-compatible API fixes for common clients.

## Project Focus

- **V100 / SM70 first**: optimized for Tesla V100 rather than being a generic
  multi-hardware fork.
- **AWQ on Volta**: AWQ 4-bit inference paths for dense and MoE Qwen models on
  SM70.
- **V100 FlashAttention path**: `FLASH_ATTN_V100` decode and prefill backend
  for Volta GPUs.
- **Long-context serving**: public profiles default to 256K context where the
  model and memory budget allow it.
- **MTP serving**: Qwen3.6-class MTP speculative decoding with safer defaults
  for V100.
- **Tool calling and OpenAI API compatibility**: validated with OpenAI-style
  clients such as Cherry Studio, OpenClaw, and similar tools.
- **Experimental FP8 work**: FP8 model and KV-cache paths are included for
  validation, but they are not production defaults.
- **Experimental DFlash work**: included for continued research and validation.

## Recommended Model Providers

- `tclf90/Qwen3.6-27B-AWQ`
- `tclf90/Qwen3.6-35B-A3B-AWQ`
- `tclf90/Qwen3.5-122B-A10B-AWQ` for larger 4-GPU setups

The launch examples use local paths such as `/path/to/Qwen3.6-27B-AWQ`.
Replace them with your local model path or a Hugging Face repository id.

## Hardware Target

The public commands are written for V100 text-generation workloads.

| Host | Notes |
| --- | --- |
| 4 x Tesla V100 32 GB | Main public reference target |
| 2 x Tesla V100 32 GB | Supported for selected 27B MTP profiles with lower concurrency |

Typical model placement:

- `Qwen3.5-27B-AWQ`: TP1/TP2/TP4 supported; TP4 is the public reference.
- `Qwen3.6-27B-AWQ`: TP4 public MTP profile; TP2 memory-constrained profile.
- `Qwen3.6-35B-A3B-AWQ`: TP4 recommended.
- `Qwen3.5-122B-A10B-AWQ`: TP4 only in the public examples.

## Validated Stack

The public wheel path is validated on:

- OS: Ubuntu 24.04 LTS
- Python: 3.12
- CUDA toolkit: 12.8
- PyTorch: CUDA 12.8 runtime wheels
- GPU: Tesla V100 32 GB

## Runtime Notes

- The **first real request is not representative** of steady-state speed. On
  V100, the first request may spend 1 to 3 minutes compiling kernels, building
  graphs, and warming up execution paths.
- Public launch commands are text-generation profiles. Vision and multimodal
  workloads should be tuned separately.
- `FLASH_ATTN_V100` is the recommended attention backend for V100.
- Public serving examples default to 256K context with
  `--max-model-len 262144`.
- Keep `--max-num-seqs 1` for baseline serving until your workload has been
  profiled locally. The 27B MTP + prefix-cache profile intentionally uses
  `max_num_seqs=4` on TP4.
- `--gpu-memory-utilization` is an upper bound for model executor memory.
  1Cat-vLLM trims the final KV cache allocation for long-context single-user
  serving instead of always preallocating for many full-length requests.
- `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is experimental. Keep it disabled for
  public 35B/122B MoE serving unless you are benchmarking that path directly.
- Direct paged prefill can be forced with
  `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1`, but it is not the quality-safe
  public default.

## Quick Start

### 1. Install CUDA 12.8

Use the official NVIDIA repository on Ubuntu 24.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

If the machine also has another CUDA toolkit installed, force build-time and
runtime CUDA to 12.8:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
nvcc -V
```

### 2. Create the Python environment

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm-sm70 python=3.12
conda activate 1cat-vllm-sm70

python -m pip install --upgrade pip setuptools wheel
```

### 3. Install from Prebuilt Wheels

Prebuilt wheels are the recommended installation path for public users. Source
builds are intended for kernel development.

Download the latest wheel assets from:

```text
https://github.com/1CatAI/1Cat-vLLM/releases/latest
```

Install both wheels from the directory where you downloaded them:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./flash_attn_v100-*.whl \
  ./vllm-*.whl
```

Notes:

- Install `flash_attn_v100` and `vllm` together.
- Runtime installation from wheels does not require the bundled `lmdeploy`
  source tree.
- Use Python 3.12 and CUDA 12.8.
- If your shell has a broken local proxy configured, unset it before
  installing:
  `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy ...`.
- After installing from wheels, run `python -m vllm...` from a directory
  outside this source checkout, such as `cd ~` or `cd /tmp`. Running inside the
  cloned repository makes Python import the local source tree instead of the
  wheel-installed CUDA extensions.

### 4. Verify the Environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
import flash_attn_v100_cuda, paged_kv_utils
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", "ok")
PY
```

## Public Runtime Defaults

| Host | Model | TP | `max_model_len` | `max_num_seqs` | `max_num_batched_tokens` | Use case |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4-card 32 GB V100 | `Qwen3.5-27B-AWQ` | 4 | `262144` | `1` | `16384` | stable public default |
| 4-card 32 GB V100 | `Qwen3.6-27B-AWQ` + MTP | 4 | `262144` | `4` | `8192` | MTP + prefix-cache API serving |
| 2-card 32 GB V100 | `Qwen3.6-27B-AWQ` + MTP | 2 | `262144` | `1` | `8192` | memory-constrained MTP serving |
| 4-card 32 GB V100 | `Qwen3.6-35B-A3B-AWQ` | 4 | `262144` | `1` | `8192` | stable public default for MoE |
| 4-card 32 GB V100 | `Qwen3.5-122B-A10B-AWQ` | 4 | `262144` | `1` | `8096` | long-context large-model default |

Important defaults:

- Public baseline launch commands default to 256K context.
- Keep `max_num_seqs=1` for baseline public commands until your workload is
  profiled locally.
- On V100, Qwen3.6/Qwen3.5 AWQ checkpoints with bundled MTP layers can use the
  public MTP4 server profile unless the related flags are overridden.
- On 2 x 32 GB V100, keep the 27B MTP profile at `max_num_seqs=1`.
- Do not pass `--disable-custom-all-reduce` for the 27B TP4 decode baseline.
- `122B` uses a smaller prefill chunk budget to leave room for SM70 MoE
  temporary workspace during long-context serving.

## Launch Examples

All commands below are full runnable commands. When using prebuilt wheels, run
them outside the source checkout so Python loads the installed package and its
CUDA extensions.

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` only when you need to select a specific
four-card V100 set.

### Qwen3.5-27B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --served-model-name Qwen3.5-27B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-27B-AWQ, TP4, MTP + Prefix Cache

This profile keeps 256K context, enables tool calling, and uses the public MTP
serving defaults for the Qwen3.6 27B AWQ model family.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq-mtp \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

On V100, 1Cat-vLLM applies these defaults for this model family:

- MTP4 speculative decoding.
- 256K context from the model config.
- `max_num_seqs=4` and `max_num_batched_tokens=8192` for TP4.
- Prefix cache with `mamba_cache_mode=align`.
- Text-only multimodal defaults to avoid unnecessary vision cache/profiling.
- `gpu_memory_utilization=0.88`.
- MTP4 CUDA graph capture sizes used by the validated TP4/TP2 profiles.

If you need a non-default experiment, override the relevant flag explicitly.
For example:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":8}'
```

Do not set `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` for this public MTP profile.
That dense fast path is experimental and should be benchmarked separately.

If decode throughput is much lower than expected, check `/metrics` for the MTP
acceptance length first. Very low acceptance usually means the prompt is too
divergent for speculative decoding or a tuning flag was overridden.

### Qwen3.6-27B-AWQ, TP2, MTP + Prefix Cache

Use this profile for two 32 GB V100 cards. It keeps the 256K context limit, but
uses lower concurrency because the TP4 MTP setting does not fit on 64 GB.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq-mtp-tp2 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-35B-A3B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-35B-A3B-AWQ \
  --served-model-name Qwen3.6-35B-A3B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.5-122B-A10B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-122B-A10B-AWQ \
  --served-model-name Qwen3.5-122B-A10B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8096 \
  --host 0.0.0.0 \
  --port 8000
```

## OpenAI-Compatible Request Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "Qwen3.5-27B-AWQ",
    "messages": [{"role": "user", "content": "用一句话回答，2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

If the response is coherent and short, the API path is basically healthy.

## Experimental Features

### FP8

FP8 support is included for validation and research. It is not the stable
public default.

- FP8 model execution on V100 is experimental.
- `fp8_e5m2` KV cache can be used experimentally on V100.
- `fp8_e4m3` is not the recommended V100 option in the current path.
- Do not add `--calculate-kv-scales` unless you are specifically testing KV
  scale calculation behavior.

Example:

```bash
--kv-cache-dtype fp8_e5m2
```

### DFlash

DFlash is included as an experimental path for continued validation. Treat it
as a research feature until you have validated speed and output quality on your
own workload.

### Dense F16 Fast Path

`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is intended for targeted experiments.
Keep it disabled for public MoE serving profiles unless you are explicitly
benchmarking that path.

## Source Build

Source build is supported, but it is **not recommended** for normal runtime
deployment. Install the release wheels first unless you are changing CUDA,
C++, or Triton code.

This repository includes the validated `lmdeploy` source tree needed by the
SM70 AWQ build path.

```bash
cd /path/to/1Cat-vLLM/vllm
test -d lmdeploy
```

Install build dependencies:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-sm70

python -m pip install -r requirements/build.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

Build wheels:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir ../dist-cu128-sm70
popd

python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```

For editable development:

```bash
python -m pip install -e . --no-build-isolation
```

## Benchmarking Notes

- First-request warmup is slow on V100 and should not be included in
  steady-state throughput.
- Browser-side OpenAI streaming throughput includes request overhead and should
  not be compared directly with strict incremental decode TPS.
- Long-context throughput depends strongly on TP, `max_num_seqs`,
  `max_num_batched_tokens`, prompt shape, and attention backend.
- If you publish a baseline, include the full launch command, GPU model,
  driver, CUDA runtime, model checkpoint, sampling parameters, prompt length,
  and decode length.

## WeChat Community

**群聊：** 1Cat-vLLM 开源交流群

请使用微信扫描下方二维码加入群组：

![1Cat-vLLM 微信交流群二维码](docs/assets/wechat-group-qr.png)

> 提示：微信群二维码通常 7 天内有效。若扫描失败或提示过期，请重新打开本页查看最新图片，或关注仓库更新。

## Repository Notes

- Upstream project: [vLLM](https://github.com/vllm-project/vllm)
- This fork focuses on SM70 AWQ support, V100-oriented attention/runtime
  tuning, and experimental FP8/MTP/DFlash validation paths.
- Prebuilt wheels are the public installation path.
- Source builds are for development and kernel work.

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
