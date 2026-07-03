# vLLM on 8× Tesla V100-SXM2-32GB — Qwen3.5-397B-A17B (AWQ) for agentic coding & ops

A reproducible recipe for serving the **Qwen3.5-397B-A17B** mixture-of-experts model
(and derivatives such as [`prefeitura-rio/Rio-3.5-Open-397B`](https://huggingface.co/prefeitura-rio/Rio-3.5-Open-397B))
as a **4-bit AWQ** checkpoint across **eight Tesla V100-SXM2-32GB GPUs** (an original
NVIDIA DGX-1 / DGX-1-class board, hybrid cube-mesh NVLink, **no NVSwitch**), for use as
a **local, always-on backend for agentic coding and system-ops work**.

This is a downstream fork of **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)** (which
itself forks **[vLLM](https://github.com/vllm-project/vllm)**). 1Cat did the hard SM70/Volta
enablement — TurboMind-derived AWQ kernels, the `FLASH_ATTN_V100` attention path, and the
Gated-Delta-Net (GDN) support that a Qwen3.5 MoE needs. **This fork adds the integration and
tuning needed to push that stack to the top of its own hardware range: a full 8-GPU box and a
~400B-parameter model, served as a stable agentic baseload.** See
[How this differs from 1Cat-vLLM](#how-this-differs-from-1cat-vllm).

> The complete, unmodified upstream **1Cat-vLLM README** is reproduced verbatim at the
> [end of this document](#appendix--original-1cat-vllm-readme-verbatim).

---

## What this is, and who it's for

Ex-datacenter **DGX-1 blades** and loose **V100-SXM2** boards are now cheap on the
second-hand market. They are Volta (**SM70**): **fp16 only**, no bf16, no fp8 compute, and
AWQ normally wants sm_75+. 1Cat-vLLM solves the kernel side. What's been missing is a
worked, end-to-end recipe for the *largest* thing such a box can usefully run:

- **Hardware:** 8× Tesla V100-SXM2-32GB (256 GB aggregate VRAM), DGX-1-style hybrid
  cube-mesh NVLink, **no NVSwitch**.
- **Model:** Qwen3.5-397B-A17B — a ~403B-parameter MoE (17B active, 512 experts + shared
  expert, ~60 layers of which ~75% are GDN linear-attention, plus full-attention layers, a
  native vision tower, and an optional MTP head) — quantized to **4-bit AWQ** (~230 GB on
  disk, ~28 GiB per GPU under TP8).
- **Workload:** single- to few-stream **agentic coding and ops** — long context, tool
  calling, reasoning separation — not high-concurrency batch serving.

If that's your box and your goal, this repo is a clone-build-serve path that works.

## Results to expect (8× V100-SXM2-32GB, TP8, this config)

| Metric | Value |
|---|---|
| Decode, 1 stream | **~41 tok/s** (~90% of the fp16-golden practical ceiling on this HW) |
| Decode, aggregate (sum-tg) | **~41 / 60 / 96 / 131 tok/s** at 1 / 2 / 4 / 8 concurrent streams |
| Prefill (cold) | ~2,000 tok/s at 4k ctx → ~750 tok/s at 120k ctx (O(n²) attention) |
| Prefill (prefix-cache hit) | tens of thousands of tok/s |
| Context | **140,000 tokens** in production (empirical ceiling ~155–168k) |
| Startup to ready | ~5–6 min (weights ~3 min) |

Decode is **all-reduce-bound** on this PCIe/NVLink-hybrid topology; the custom 1-stage
all-reduce is already the best lever. Aggregate throughput scales cleanly to 8 streams.
**Do not run ≥16 concurrent streams** on this build — there is a known NCCL all-reduce
deadlock (see [Gotchas](#configuration-notes--gotchas)).

## How this differs from 1Cat-vLLM

1Cat-vLLM targets **2× and 4× V100** setups (they also make 2×/4× SXM2 baseboards and PCIe
adapters) and models up to ~122B. This fork is **complementary**, aimed one tier up:

| | 1Cat-vLLM public profiles | This recipe |
|---|---|---|
| GPUs | 2× / 4× V100 | **8× V100-SXM2 (DGX-1)** |
| Reference model | Qwen3.6-27B / 35B / 122B AWQ | **Qwen3.5-397B-A17B AWQ (~400B)** |
| Parallelism | TP1–TP4 | **TP8** |
| Use case | general V100 serving | **agentic coding/ops baseload** |

The substantive additions this fork carries on top of 1Cat v1.2.1 (all in git history):

- **Non-MTP GDN full-forward auto-arm** — 1Cat's v1.2.1 only auto-arms the FULL-cudagraph
  GDN decode-corruption guard when MTP speculative decoding is on. A non-MTP MoE (like this
  397B profile) otherwise gets silent decode corruption. This fork auto-arms it for non-MTP
  too. *(This is the one fix that helps any non-MTP SM70 MoE user, not just DGX-1 owners.)*
- **fp16 super-weight overflow saturation** and **shared-expert gate ordering** — SM70
  numerical-correctness fixes for the MoE path.
- **GDN autotune pin** for the Volta chunk-scaled-dot kernel (part of the memory fix).
- **All-reduce-algo env forwarded to workers**, **scheduler starvation warning**, an
  **Anthropic `/v1/messages` mid-conversation system-role** fix, and a **server-side
  `thinking_token_budget`** default that stops reasoning-mode from looping to empty output.
- A small **fleet regression harness** and an **MTP-branch AWQ quant** helper.

---

## Install

### 0. Prerequisites

- **8× Tesla V100-SXM2-32GB**, one host, NVLink up (`nvidia-smi topo -m`), persistence mode on.
- **Ubuntu 24.04 LTS**, **Python 3.12**, **CUDA toolkit 12.8**, a recent NVIDIA driver.
- **~250 GB free disk** for the AWQ checkpoint (and ~300 GB system RAM *only if you quantize
  it yourself*, see [step 3c](#3c-quantize-it-yourself)).
- Confirm the GPUs are free — this recipe uses all 8. Nothing else can share them while it runs.

### 1. Clone

```bash
git clone https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b.git
cd vllm-v100-sxm2-qwen3.5-397b
```

### 2. Build or install the engine

The SM70 patches in this fork are Python-only on top of 1Cat v1.2.1's compiled extensions,
so the fastest path is to **install 1Cat's prebuilt v1.2.1 wheel for its CUDA extensions,
then run this source tree** (which carries the patches):

```bash
# Python 3.12 env
python -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Get 1Cat's prebuilt SM70 wheel (CUDA 12.8 / torch 2.10) — bundles flash_attn_v100 + SM70 kernels
#   https://github.com/1CatAI/1Cat-vLLM/releases/latest   (asset: 1cat_vllm-1.2.1-cp312-cp312-linux_x86_64.whl)

# Install this fork editable, reusing that wheel's compiled .so (no nvcc needed):
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/1cat_vllm-1.2.1-cp312-cp312-linux_x86_64.whl \
  python -m pip install -e . --no-build-isolation --no-deps
# Copy the flash_attn_v100 .so out of the wheel into ./flash-attention-v100/ if the editable
# layout doesn't find it (the wheel places ext modules as siblings, not in-package).
```

Prefer a **full source build** (needed only if you change CUDA/C++/Triton code)? Follow the
[upstream Source Build section](#appendix--original-1cat-vllm-readme-verbatim) but set
`TORCH_CUDA_ARCH_LIST="7.0"` and `FLASH_ATTN_V100_CUDA_ARCH_LIST="7.0"`.

> **Always launch the server from *outside* this checkout** (e.g. `cd ~` or a work dir).
> Running inside the tree makes Python import the local `vllm/` package instead of the
> compiled extensions and fails with a `vllm._C` import error.

Verify:

```bash
cd /tmp && python - <<'PY'
import torch, triton, vllm, flash_attn_v100
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| fa_v100", flash_attn_v100.__version__)
PY
```

### 3. Get the weights

The model and its permissive derivative are public. Pick one of three:

#### 3a. A ready public AWQ (fastest, but validate load)

```
cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit      # Apache-2.0; keeps GDN/linear_attn in fp16 (correct)
```
Caveat: this checkpoint is **compressed-tensors "pack-quantized", symmetric, group_size 32** —
a *different* scheme than the `W4A16_ASYM g128` this stack is validated against. It may load
and serve, but **test coherence + a needle retrieval before trusting it**.

#### 3b. Not recommended: the official GPTQ-Int4

```
Qwen/Qwen3.5-397B-A17B-GPTQ-Int4         # Apache-2.0
```
This one sets `modules_to_not_convert: []` — it **quantizes the GDN/linear-attention path and
attention too**, which is exactly the corruption-prone region this stack keeps in fp16, and it
routes every GEMM through the slow Volta marlin backport. Fine for other engines; **not** this one.

#### 3c. Quantize it yourself (the validated path)

Reproduces the checkpoint this stack is built around. Base weights:
`Qwen/Qwen3.5-397B-A17B` (Apache-2.0) or `prefeitura-rio/Rio-3.5-Open-397B` (MIT).

- Tool: `llm-compressor`, `pipeline="sequential"`, `oneshot_device="cuda:0"` (single-GPU,
  the other 7 idle), `torch_dtype="float16"` (**never bf16**).
- Scheme: **`W4A16_ASYM`, `group_size=128`, AutoAWQ-compatible asymmetric INT4**.
- **Ignore list (keep in fp16):** `lm_head`, MoE router gates, the shared-expert gate, the
  **GDN / linear-attention projections**, and the **entire vision tower**. Copy the ignore
  list verbatim from your AWQ reference's `config.json` — do not invent regexes.
- Budget ~300 GB RAM, offload to CPU (**never spill to SSD**), ~3–6 h data-free. Checkpoint
  per layer so a crash resumes. Output `quantization_config` must diff-clean against the
  reference (format, bits, group_size, zero_point, ignore list).

The MTP-drafter branch (if you serve MTP) can be AWQ'd separately with
`tools/quantize_qwen3_5_mtp_awq.py`.

### 4. Serve

Two options in [`deploy/`](deploy/) — both default to **140k context, TP8, fp8_e5m2 KV,
the full SM70 tuning, and multi-stream cudagraph capture**. Edit the `/path/to/…` placeholders
and the model path first.

**a) One-shot script** (foreground, with GPU-busy guard, KV-fit retry, and a warmup ping):

```bash
MODEL=/path/to/Qwen3.5-397B-A17B-AWQ PY=/path/to/.venv/bin/python WORKDIR=/tmp \
  ./deploy/serve.sh 8000
```

**b) systemd service** (always-on, restart-on-crash, boot-enabled):

```bash
cp deploy/vllm-397b.env.example /path/to/vllm-397b.env    # edit MODEL + tunables
# edit deploy/vllm-397b.service: set User=, the venv python, WorkingDirectory, EnvironmentFile
sudo install -m 0644 deploy/vllm-397b.service /etc/systemd/system/vllm-397b.service
sudo systemctl daemon-reload && sudo systemctl enable --now vllm-397b
journalctl -u vllm-397b -f
```

> ⚠️ Both bind `0.0.0.0` with **no authentication**. On a shared or reachable network,
> firewall the port, bind `127.0.0.1` + SSH-tunnel, or add `--api-key`.

### 5. Smoke test

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-397B-A17B-AWQ","messages":[{"role":"user","content":"Capital of France, one word."}],"temperature":0,"max_tokens":16}'
```

A short coherent answer (e.g. `Barcelona`) means the GDN full-forward guard is doing its job and
the AWQ/kernel path is healthy. For agentic use, point any OpenAI-compatible coding agent at
`http://<host>:8000/v1`; tool calling (`qwen3_coder`) and reasoning separation (`qwen3`) are on.

---

## Configuration notes & gotchas

Hard-won settings baked into `deploy/` — change them only if you know why:

- **`--gpu-memory-utilization 0.95` is the wall.** Above ~0.95 you hit runtime CUDA OOM (per-rank
  memory jitters decide which); below it the requested context fails the KV-fit check. The KV
  pool fills whatever util allows.
- **`--kv-cache-dtype fp8_e5m2` is required**, not optional — the v1.2.1 base footprint OOMs
  KV init with fp16 KV at this context/util. Needle-validated well past 110k.
- **GDN decode-corruption guard** (`VLLM_SM70_QWEN_GDN_FULL_FORWARD`) — auto-armed by this fork
  for non-MTP; the env is set belt-and-suspenders. Without it, FULL-cudagraph decode is garbage.
- **`--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'`** — the SM70 default captures
  only `[1,2]`, so a batch of 4 falls to eager and *collapses* to ~5 tok/s/stream. This restores
  clean scaling to 8 streams. (Costs ~0.03–0.28 GiB of graph memory; re-check the 140k fit.)
- **`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`** — fixes an allocator
  fragmentation OOM in the SM70 MoE temp-buffer path (looked like an all-reduce hang). Do **not**
  use `expandable_segments` (breaks custom all-reduce graph capture on this topology).
- **`--skip-mm-profiling`** — the native vision tower's profiling pass inflates the activation
  estimate and costs ~90k tokens of context. Text-only serving is safe without it.
- **GDN kernel knobs** (`VLLM_SM70_GDN_*` in the env) — the Volta autotune that reclaims ~1.5
  GiB/rank; leave them as shipped.
- **Prefix caching** needs `--max-num-batched-tokens ≥` the mamba block size (1056 here); the
  script uses 1059 so one full prefill block fits alongside decodes.
- **`--override-generation-config '{"thinking_token_budget": 4096}'`** — reasoning mode can
  otherwise loop to an empty response on hard/unanswerable prompts. Clients override per request
  (`-1` = unlimited).
- **Do not run ≥16 concurrent streams.** There is an unresolved NCCL all-reduce deadlock at
  batch ≥16 on this build (two ranks diverge before the collective). Sweet spot: 4 streams for
  latency+throughput, 8 for max aggregate.

---

## Experimental features from this fork — vision fp32 & SwiReasoning

### Vision tower in fp32 (`VITFP32`)

The Qwen3.5 vision tower overflows fp16's precision on Volta: register-token
activations reach ~500 by block 9, and on text-dense images the worst merged-token
cosine vs an fp32 reference drops to 0.973 (visible as OCR/detail degradation).
This fork builds the tower in **fp32 on SM70** and casts back to the LLM dtype at
the boundary (after the deepstack concat).

- **Auto-gated:** fp32 only when SM70 *and* multimodal is enabled (any
  `--limit-mm-per-prompt` > 0; this base defaults `image=1`). Text-only configs
  keep the fp16 tower automatically. `VLLM_SM70_VIT_FP32=0` opts out explicitly.
- **Cost: ~1 GiB/rank** — the shipped 140k-context profile no longer passes the
  KV-fit check with it (ceiling drops to ~138k). `deploy/` therefore ships
  `VLLM_SM70_VIT_FP32=0` (vision still works, at upstream fp16 quality). To serve
  fp32 vision: drop that env line and set `--max-model-len` ≤ 138000.

### SwiReasoning (opt-in continuous-thought decoding)

An implementation of [SwiReasoning](https://arxiv.org/abs/2510.05069) — an
entropy-trend FSM that switches the model between latent and explicit reasoning
by injecting soft (probability-blended) input embeddings between decode steps.
Strictly **opt-in per request**; traffic without the key is byte-identical to
stock serving (fleet-regression verified).

```jsonc
// POST /v1/chat/completions — pass token ids from the model's tokenizer
"vllm_xargs": { "swireasoning": {
  "think_id": 248068, "end_think_id": 248069, "line_break_id": 1639,
  "eos_token_id": 248046,
  "convergence_ids": [248069],                         // encode("</think>")
  "termination_ids": [248069, 271, 760, 1534, 4087, 369],
  "alpha_0": 1.0, "beta_0": 0.7, "window_size": 512,
  "max_switch_count": null, "termination_max_tokens": 32,
  "max_new_tokens": 4096, "math_ids": null
} }
```

- **Server requirement:** `--no-async-scheduling` (already in `deploy/`). Runs on
  the normal graph-enabled server: on the multimodal input path the decode graph
  is embeds-entry, so injections ride graph replay natively; token-id-path
  configs fall back to per-step eager dispatch automatically.
  `VLLM_SWIR_TIER1=1` forces the per-step-eager mode; `VLLM_SWIR_DEBUG=1` logs
  the FSM per step.
- **Measured on this rig (Rio-397B AWQ, TP8) — read before using:** with the
  reference knobs above (`alpha_0=1.0`), SwiReasoning is **accuracy-neutral and
  token-neutral** under sampling (GSM8K n=200: 93.5% both arms, tokens +3%;
  AIME'24+'25 n=60×2: 0.82-0.83 vs 0.82 baseline). **Do not run `alpha_0 < 1.0`
  on this stack**: the earlier-reported token savings (−45% eager / −35% graphs)
  came from `alpha_0=0.5` and turned out to be substantially an artifact —
  stochastic premature closure of the thinking block, which reads as
  "compression" on easy problems and becomes wrong answers on hard ones
  (AIME high-difficulty subset: 11% vs 72% baseline on affected runs; full
  analysis in the fork's eval notes). The paper's efficiency claim did not
  reproduce at safe knobs on this fp16/AWQ/V100 stack; treat the feature as an
  experimental mechanism, not a proven efficiency win.
- Code: `vllm/v1/sample/swir_controller.py` (backend-agnostic FSM) +
  `vllm/v1/worker/swir_glue.py` (runner adapter, 5 one-line hooks).

---

## Contributing

**Issues and PRs are very welcome** — especially from other 8× V100-SXM2 / DGX-1 owners.
Good things to send back:

- Reproduction reports (your driver / CUDA / checkpoint / numbers).
- A root-cause or fix for the **≥16-stream NCCL deadlock**.
- Closing the last ~10% single-stream decode gap vs the fp16 golden (a version-level kernel
  rebuild).
- Prefill-side GDN retune on newer Triton.
- Other large MoE checkpoints validated on this box, or other 8-GPU Volta boards.

If a change is broadly useful (e.g. the non-MTP GDN guard), consider also raising it upstream
on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/issues).

## Disclaimer — no warranty

This software is provided **"AS IS", without warranty of any kind**, express or implied. See
the **Disclaimer of Warranty** and **Limitation of Liability** in [`LICENSE`](LICENSE)
(Apache-2.0, §7–8). The SM70 kernels, the AWQ-on-Volta path, and the tuning here are
experimental and hardware-specific; you run them at your own risk. The default launch binds
`0.0.0.0` with no auth — secure it before exposing it. Serving occupies all 8 GPUs.

## Credits & license

Licensed under **Apache-2.0** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). This is a fork
of **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)** → **[vLLM](https://github.com/vllm-project/vllm)**,
and builds on **[lmdeploy/TurboMind](https://github.com/InternLM/lmdeploy)**,
**[flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)**, and
**[marlin_v100](https://github.com/zhinianqin/marlin_v100)**. Model weights are © their
respective authors (**Qwen** / **prefeitura-rio**) under their own licenses.

*Recipe and SM70/V100 integration patches by **Jens David**, developed with assistance from
Claude (Opus 4.8 and Fable 5).*

---

## Appendix — original 1Cat-vLLM README (verbatim)

The following is the upstream `README.md` from **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)**
(tag v1.2.1), reproduced **verbatim and unmodified** for attribution and reference. Its
launch examples and hardware notes describe 1Cat's 2×/4× V100 profiles, not the 8-GPU 397B
profile documented above.

---

# 1Cat-vLLM

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判"过时"。
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
  for Volta GPUs, with SM70 compile-graph, guarded XQA decode, and D=256
  paged-prefix low-smem fast paths enabled by default.
- **Long-context serving**: public profiles default to 256K context where the
  model and memory budget allow it.
- **MTP serving**: Qwen3.6-class MTP speculative decoding remains available as
  an explicit opt-in path; long-context public profiles default to no MTP.
- **Image inputs by default**: SM70 `FLASH_ATTN_V100` profiles allow one image
  per prompt by default; video inputs remain opt-in.
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

The public commands are written for V100 Qwen serving workloads. Image inputs
are enabled by default on the SM70 `FLASH_ATTN_V100` path; video inputs are
disabled by default and should be enabled explicitly only after local memory
validation.

| Host | Notes |
| --- | --- |
| 4 x Tesla V100 32 GB | Main public reference target |
| 2 x Tesla V100 32 GB | Supported for selected 27B profiles with lower concurrency |

Typical model placement:

- `Qwen3.6-27B-AWQ`: TP1/TP2/TP4 supported; TP4 is the public reference.
- `Qwen3.6-27B-AWQ + MTP`: explicit opt-in profile for local validation, not
  the long-context public default.
- `Qwen3.6-35B-A3B-AWQ`: TP4 recommended.
- `Qwen3.5-122B-A10B-AWQ`: TP4 supported for larger deployments.

Multimodal defaults:

- Default SM70 `FLASH_ATTN_V100` serving allows `image=1`, `video=0` when
  `--limit-mm-per-prompt` is not set.
- For text-only serving, pass `--limit-mm-per-prompt '{"image":0,"video":0}'`
  or use `--language-model-only`.
- For video workloads, pass an explicit limit such as
  `--limit-mm-per-prompt '{"image":1,"video":1}'` and retune memory settings.

## Validated Stack

The public wheel path is validated on:

- OS: Ubuntu 24.04 LTS
- Python: 3.12
- CUDA toolkit: 12.8
- PyTorch: CUDA 12.8 runtime wheels
- GPU: Tesla V100 32 GB

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

Install the wheel from the directory where you downloaded it:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./1cat_vllm-*.whl
```

Notes:

- The `1cat_vllm` wheel already bundles the `flash_attn_v100` Python package
  and SM70 CUDA extensions.
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
import flash_attn_v100
from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", flash_attn_v100.__version__)
PY
```

## Recommended Launch Commands

These are the recommended public serving commands for the 27B AWQ and 35B AWQ
V100 profiles. When using prebuilt wheels, run them outside the source checkout
so Python loads the installed package and its CUDA extensions.

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` only when you need to select a specific
four-card V100 set.

### Qwen3.6-27B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-35B-A3B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-35B-A3B-AWQ \
  --served-model-name qwen3.6-35b-a3b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 \
  --port 8000
```

## OpenAI-Compatible Request Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.6-27b-awq",
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

### MTP

MTP is not enabled by default in the V100 public serving profile. Long-context
decode on V100 can slow down significantly when MTP is enabled, so keep the
default no-MTP path for 128K/256K style serving unless your own workload proves
otherwise.

To explicitly test the previous automatic SM70 MTP4 profile:

```bash
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
```

You can also pass an explicit `--speculative-config` when you want full control
over speculative decoding settings.

### Dense F16 Fast Path

`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is intended for targeted experiments.
Keep it disabled for public MoE serving profiles unless you are explicitly
benchmarking that path.

## Source Build

Source build is supported, but it is **not recommended** for normal runtime
deployment. Install the release wheels first unless you are changing CUDA,
C++, or Triton code.

This repository includes the validated `lmdeploy` source tree under
`csrc/sm70_turbomind/lmdeploy`, which is needed by the SM70 AWQ build path.

```bash
cd /path/to/1Cat-vLLM/vllm
test -d csrc/sm70_turbomind/lmdeploy
```

Install build dependencies:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-sm70

python -m pip install -r requirements/build/cuda.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

Build wheels:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
export FLASH_ATTN_V100_CUDA_ARCH_LIST="7.0"
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
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
