#!/usr/bin/env python3
"""Quantize the MTP branch of Qwen3.5-397B-A17B-AWQ to AWQ int4 g128.

Writes a new checkpoint dir (never modifies the source):
- mtp expert weights (mtp.layers.*.mlp.experts.*.{gate,up,down}_proj.weight):
  RTN int4 group-128 zero-point quantization into AWQ GEMM layout
  (qweight/qzeros/scales), matching the main model's expert tensors exactly.
- every other mtp.* tensor: cast BF16 -> FP16 (V100 has no bf16 compute).
- non-mtp tensors: byte-identical; shards with no mtp tensors are hardlinked
  when possible (NOTE: hardlinks share storage with the source — never edit
  those files in place), copied otherwise.
- config.json: modules_to_not_convert "mtp" -> "mtp.fc", so this fork's
  Qwen3_5MTP loader builds AWQ MTP experts (it checks for the exact string
  "mtp") while fc stays fp16; self_attn / shared_expert / mlp.gate are
  already covered by the existing substring patterns.

Usage: python tools/quantize_qwen3_5_mtp_awq.py /path/to/src /path/to/dst
Requires only torch, numpy, safetensors (no vllm import).

NOTE: this is RTN, not activation-aware AWQ search (no calibration data flows
through the MTP branch offline). Acceptance rate at runtime is the judge.

Expected per-rank weight cost at TP8: ~0.43 GiB (vs 1.54 GiB unquantized).
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("src", type=Path, help="source AWQ checkpoint dir")
_ap.add_argument("dst", type=Path, help="output dir (must not exist)")
_args = _ap.parse_args()
SRC_DIR: Path = _args.src
DST_DIR: Path = _args.dst
GROUP_SIZE = 128
BITS = 4

# A main-model expert tensor triplet used to self-test the packing code
# against ground truth before converting anything.
REF_QWEIGHT = "model.language_model.layers.5.mlp.experts.0.gate_proj.qweight"


def is_mtp_expert_weight(name: str) -> bool:
    return (
        name.startswith("mtp.")
        and ".mlp.experts." in name
        and name.endswith(".weight")
    )


# ---------------------------------------------------------------------------
# AWQ GEMM packing — vendored so this script does not import vllm
# (importing vllm next to a live server risks creating a CUDA context).
# Source: vllm/model_executor/layers/quantization/utils/quant_utils.py
# (pack_cols, awq_pack). Inverse order from moe_wna16.py:411.
# ---------------------------------------------------------------------------
AWQ_INTERLEAVE = np.array([0, 2, 4, 6, 1, 3, 5, 7])
AWQ_REVERSE = np.array([0, 4, 1, 5, 2, 6, 3, 7])


def pack_cols(q_w: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    assert q_w.shape == (size_k, size_n)
    assert size_n % 8 == 0
    q_w = q_w.astype(np.uint32)
    q_res = np.zeros((size_k, size_n // 8), dtype=np.uint32)
    for i in range(8):
        q_res |= q_w[:, i::8] << BITS * i
    return np.ascontiguousarray(q_res.astype(np.int32))


def awq_pack(q_w: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    assert q_w.shape == (size_k, size_n)
    q_w = q_w.reshape((-1, 8))[:, AWQ_INTERLEAVE].ravel()
    q_w = np.ascontiguousarray(q_w.reshape((-1, size_n)))
    return pack_cols(q_w, size_k, size_n)


def awq_unpack(packed: np.ndarray) -> np.ndarray:
    """[K, N/8] int32 -> [K, N] int32 in original column order."""
    p = packed.astype(np.uint32)
    K, Np = p.shape
    out = np.zeros((K, Np * 8), dtype=np.int32)
    for c in range(8):
        out[:, c::8] = ((p >> int(4 * AWQ_REVERSE[c])) & 0xF).astype(np.int32)
    return out


def self_test_packing() -> None:
    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, size=(256, 64), dtype=np.int32)
    assert (awq_unpack(awq_pack(q, 256, 64)) == q).all(), "synthetic roundtrip"

    # Ground truth: real main-model AWQ tensors must survive unpack->pack.
    idx = json.load(open(SRC_DIR / "model.safetensors.index.json"))
    shard = SRC_DIR / idx["weight_map"][REF_QWEIGHT]
    tensors = load_file(str(shard))
    for suffix in ("qweight", "qzeros"):
        ref = tensors[REF_QWEIGHT.replace("qweight", suffix)].numpy()
        rt = awq_pack(awq_unpack(ref), ref.shape[0], ref.shape[1] * 8)
        assert (rt == ref).all(), f"main-model {suffix} roundtrip failed"
    print("packing self-test vs main-model tensors: OK")


# ---------------------------------------------------------------------------
# RTN int4 zero-point quantization (AutoAWQ pseudo_quantize grid)
# ---------------------------------------------------------------------------
def quantize_rtn(w: torch.Tensor):
    """w: [out, in] -> (qweight [in, out/8] i32, qzeros [in/gs, out/8] i32,
    scales [in/gs, out] f16, max_abs_err, rel_fro_err)."""
    out_f, in_f = w.shape
    assert in_f % GROUP_SIZE == 0
    G = in_f // GROUP_SIZE
    wf = w.to(torch.float32).reshape(out_f, G, GROUP_SIZE)

    mx = wf.amax(dim=-1)
    mn = wf.amin(dim=-1)
    scales = (mx - mn).clamp(min=1e-5) / 15.0
    zeros = (-torch.round(mn / scales)).clamp(0, 15)
    q = (torch.round(wf / scales.unsqueeze(-1)) + zeros.unsqueeze(-1)).clamp(0, 15)

    deq = (q - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    err = (deq - wf).abs()
    max_abs_err = err.max().item()
    rel_fro = (err.square().sum().sqrt() / wf.square().sum().sqrt().clamp(min=1e-12)).item()

    q_in_out = q.reshape(out_f, in_f).T.contiguous().numpy().astype(np.int32)
    z_g_out = zeros.T.contiguous().numpy().astype(np.int32)
    qweight = torch.from_numpy(awq_pack(q_in_out, in_f, out_f))
    qzeros = torch.from_numpy(awq_pack(z_g_out, G, out_f))
    scales_t = scales.T.contiguous().to(torch.float16)
    return qweight, qzeros, scales_t, max_abs_err, rel_fro


def main() -> None:
    if DST_DIR.exists():
        raise FileExistsError(f"{DST_DIR} already exists; remove it first")
    if DST_DIR.resolve().is_relative_to(SRC_DIR.resolve()):
        raise ValueError("dst must not be inside the source checkpoint dir")
    self_test_packing()
    DST_DIR.mkdir(parents=True)

    idx = json.load(open(SRC_DIR / "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    old_total = idx["metadata"]["total_size"]

    shards = sorted({f for f in wmap.values()})
    mtp_shards = sorted({f for n, f in wmap.items() if n.startswith("mtp.")})
    print(f"shards: {len(shards)} total, {len(mtp_shards)} contain mtp tensors")

    # 1. hardlink untouched shards (copy if dst is on another filesystem)
    for fn in shards:
        if fn not in mtp_shards:
            try:
                os.link(SRC_DIR / fn, DST_DIR / fn)
            except OSError:
                shutil.copy2(SRC_DIR / fn, DST_DIR / fn)
    print(f"linked/copied {len(shards) - len(mtp_shards)} clean shards")

    # 2. convert mtp shards
    new_wmap = {n: f for n, f in wmap.items()}
    size_delta = 0
    n_quant = n_cast = n_kept = 0
    err_stats = {}  # proj type -> [max_abs, sum_rel, count]
    overflow = 0
    t0 = time.time()
    for i, fn in enumerate(mtp_shards, 1):
        tensors = load_file(str(SRC_DIR / fn))
        out = {}
        for name, t in tensors.items():
            if is_mtp_expert_weight(name):
                qweight, qzeros, scales, mx_err, rel = quantize_rtn(t)
                base = name[: -len(".weight")]
                out[base + ".qweight"] = qweight
                out[base + ".qzeros"] = qzeros
                out[base + ".scales"] = scales
                del new_wmap[name]
                for suf in (".qweight", ".qzeros", ".scales"):
                    new_wmap[base + suf] = fn
                size_delta += (
                    qweight.nbytes + qzeros.nbytes + scales.nbytes - t.nbytes
                )
                proj = base.rsplit(".", 1)[-1]
                s = err_stats.setdefault(proj, [0.0, 0.0, 0])
                s[0] = max(s[0], mx_err)
                s[1] += rel
                s[2] += 1
                n_quant += 1
            elif name.startswith("mtp.") and t.dtype == torch.bfloat16:
                overflow += int((t.abs() > 65504).sum())
                out[name] = t.to(torch.float16)
                n_cast += 1
            else:
                out[name] = t
                n_kept += 1
        save_file(out, str(DST_DIR / fn), metadata={"format": "pt"})
        mb = (DST_DIR / fn).stat().st_size / 2**20
        print(
            f"[{i}/{len(mtp_shards)}] {fn}: {len(out)} tensors, {mb:.0f} MB, "
            f"{time.time() - t0:.0f}s elapsed"
        )

    # 3. index
    idx_out = {
        "metadata": {"total_size": old_total + size_delta},
        "weight_map": new_wmap,
    }
    with open(DST_DIR / "model.safetensors.index.json", "w") as fh:
        json.dump(idx_out, fh, indent=2, sort_keys=True)

    # 4. config: enable MTP quantization for the loader, keep fc fp16
    cfg = json.load(open(SRC_DIR / "config.json"))
    mods = cfg["quantization_config"]["modules_to_not_convert"]
    assert "mtp" in mods, f"expected 'mtp' in modules_to_not_convert: {mods}"
    cfg["quantization_config"]["modules_to_not_convert"] = [
        "mtp.fc" if m == "mtp" else m for m in mods
    ]
    cfg["name_or_path"] = str(DST_DIR)
    with open(DST_DIR / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    # 5. remaining aux files
    for f in SRC_DIR.iterdir():
        if (
            f.name.startswith("model-")
            or f.name in ("model.safetensors.index.json", "config.json")
            or f.name.endswith(".bak")
        ):
            continue
        if f.is_dir():
            shutil.copytree(f, DST_DIR / f.name)
        else:
            shutil.copy2(f, DST_DIR / f.name)

    print(f"\nquantized {n_quant} expert tensors, cast {n_cast} bf16->fp16, "
          f"kept {n_kept} non-mtp tensors")
    print(f"bf16 values exceeding fp16 range: {overflow}")
    print(f"mtp size delta: {size_delta / 2**30:+.2f} GiB "
          f"(new total_size {(old_total + size_delta) / 2**30:.2f} GiB)")
    for proj, (mx, sum_rel, c) in sorted(err_stats.items()):
        print(f"  {proj}: max_abs_err {mx:.4f}, mean rel fro err "
              f"{sum_rel / c:.4f} over {c} tensors")
    print(f"done in {time.time() - t0:.0f}s -> {DST_DIR}")


if __name__ == "__main__":
    main()
