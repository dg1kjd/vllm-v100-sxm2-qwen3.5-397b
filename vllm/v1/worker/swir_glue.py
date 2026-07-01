# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiReasoning engine glue for the V1 GPU model runner (WORKORDER §5.P3).

Keeps the per-request SwiReasoning controller state and the two decode-loop
hooks OUT of the 5k-line ``gpu_model_runner.py`` so that file needs only a
handful of one-line call-sites.  All the control logic lives in the
backend-agnostic ``vllm.v1.sample.swir_controller`` (pinned token-exact to the
reference by the scratch oracle); this module is the vLLM-aware adapter.

Opt-in + OFF-by-default: a request is a SwiReasoning request iff its
``SamplingParams.extra_args["swireasoning"]`` is set (a dict of the model's
signal-token ids + the per-request knobs -- the client supplies them since it
holds the tokenizer).  When no request in a batch carries that key, every hook
is a cheap no-op, so the production text path is byte-for-byte unchanged
(workorder §5.P5 "text-only-when-off" gate).

INTERLOCKS (this first eager cut): SwiReasoning requires ``--enforce-eager``
(the per-decode-step ``inputs_embeds`` feedback breaks the token-id CUDA-graph
signature -- §4.D) and is built/validated on the no-MTP server with prefix
caching off (§4.E/§4.F).  Those are asserted / documented, not silently
worked around.
"""
from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.v1.sample.swir_controller import (
    CallableEmbedOps,
    SwirConfig,
    SwirController,
    SwirRequestState,
)

logger = init_logger(__name__)

SWIR_KEY = "swireasoning"
_DEBUG = os.environ.get("VLLM_SWIR_DEBUG", "0") == "1"


def get_swir_args(sampling_params) -> dict | None:
    """Return the swireasoning config dict from a request's SamplingParams, or
    None if the request is not a SwiReasoning request."""
    if sampling_params is None:
        return None
    extra = getattr(sampling_params, "extra_args", None)
    if not extra:
        return None
    return extra.get(SWIR_KEY)


def _build_cfg(d: dict) -> SwirConfig:
    return SwirConfig(
        think_id=int(d["think_id"]),
        end_think_id=int(d["end_think_id"]),
        line_break_id=int(d["line_break_id"]),
        eos_token_id=(None if d.get("eos_token_id") is None else int(d["eos_token_id"])),
        convergence_ids=[int(x) for x in d.get("convergence_ids", [])],
        termination_ids=[int(x) for x in d.get("termination_ids", [])],
        alpha_0=float(d.get("alpha_0", 1.0)),
        beta_0=float(d.get("beta_0", 0.7)),
        window_size=int(d.get("window_size", 512)),
        max_switch_count=(
            None if d.get("max_switch_count") is None else int(d["max_switch_count"])
        ),
        termination_max_tokens=int(d.get("termination_max_tokens", 32)),
        max_new_tokens=int(d.get("max_new_tokens", 32768)),
        math_ids=(None if d.get("math_ids") is None else [int(x) for x in d["math_ids"]]),
    )


def _find_embed_module(model):
    """Locate the input-embedding module (VocabParallelEmbedding) across model
    nestings.  We need the module (not just .weight) for its TP shard indices."""
    candidates = [
        lambda m: m.get_input_embeddings(),
        lambda m: m.model.embed_tokens,
        lambda m: m.model.model.embed_tokens,
        lambda m: m.language_model.model.embed_tokens,
        lambda m: m.language_model.embed_tokens,
    ]
    for fn in candidates:
        try:
            mod = fn(model)
        except AttributeError:
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            return mod
    raise RuntimeError("[swir] could not locate input-embedding module on model")


class SwirGlue:
    """Per-runner SwiReasoning state + decode-loop hooks."""

    def __init__(self, runner):
        self.runner = runner
        self.controller: SwirController | None = None
        self.states: dict[str, SwirRequestState] = {}
        # req_id -> [dim] embedding to feed as that request's NEXT input.
        self.pending_emb: dict[str, torch.Tensor] = {}
        self._emb_mod = None  # input-embedding module (for weight + TP shards)
        self._eager_checked = False

    # ---------------------------------------------------------------- lifecycle
    def on_finished(self, req_id: str) -> None:
        self.states.pop(req_id, None)
        self.pending_emb.pop(req_id, None)

    def _req_args(self, req_id: str) -> dict | None:
        rs = self.runner.requests.get(req_id)
        if rs is None:
            return None
        return get_swir_args(rs.sampling_params)

    def any_active(self) -> bool:
        """True if any request currently in the batch is a SwiReasoning request.
        Cheap: ``extra_args`` is None for all production requests."""
        if self.states:
            return True
        return any(
            self._req_args(r) is not None for r in self.runner.input_batch.req_ids
        )

    # ------------------------------------------------------------- embed ops
    def _embed_module(self):
        if self._emb_mod is None:
            self._emb_mod = _find_embed_module(self.runner.model)
        return self._emb_mod

    def _soft_emb(self, probs: torch.Tensor) -> torch.Tensor:
        """soft_emb = probs @ E with fp32 accumulation (FP16_GUARD lore, §4.B).

        ``probs`` is the FULL-vocab distribution (logits are all-gathered to the
        full vocab on every TP rank), so on TP>1 each rank contributes the
        partial matmul over the vocab slice it owns and we all-reduce(sum) to
        reconstruct ``probs @ E_full``.  fp32 accumulation throughout, cast back
        to the model dtype for the embedding feed."""
        mod = self._embed_module()
        E = mod.weight
        V = probs.shape[-1]
        tp = self.runner.vllm_config.parallel_config.tensor_parallel_size
        if tp == 1:
            Eh = E[:V] if E.shape[0] >= V else E
            return self._chunked_f32_matmul(probs, Eh, 0).to(E.dtype)

        # Vocab-parallel: matmul over this rank's org-vocab slice, then sum.
        si = mod.shard_indices
        if getattr(mod, "num_added_embeddings", 0):
            raise NotImplementedError(
                "[swir] soft_emb with added (LoRA) vocab + TP not supported"
            )
        s = si.org_vocab_start_index
        e = si.org_vocab_end_index
        ps = si.padded_org_vocab_start_index
        local_rows = E[(s - ps) : (e - ps)]  # [e-s, dim] this rank's real rows
        partial = self._chunked_f32_matmul(probs, local_rows, s)  # [B, dim]
        out = tensor_model_parallel_all_reduce(partial)
        return out.to(E.dtype)

    @staticmethod
    def _chunked_f32_matmul(
        probs: torch.Tensor, rows: torch.Tensor, col0: int, chunk: int = 8192
    ) -> torch.Tensor:
        """``probs[:, col0:col0+len(rows)] @ rows`` in fp32, chunked over vocab
        rows so the fp32 copy of the embedding shard is never materialized at
        once (a full ``rows.float()`` is ~0.5 GiB on the 397B and OOMs at
        gpu-memory-utilization 0.95 where runtime headroom is <0.4 GiB)."""
        out = torch.zeros(
            probs.shape[0], rows.shape[1], device=rows.device, dtype=torch.float32
        )
        n = rows.shape[0]
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            out += torch.matmul(
                probs[:, col0 + i : col0 + j].float(), rows[i:j].float()
            )
        return out

    def _ensure_controller(self) -> SwirController:
        if self.controller is None:
            ops = CallableEmbedOps(self.runner.model.embed_input_ids, self._soft_emb)
            self.controller = SwirController(ops)
        return self.controller

    def _check_interlocks(self) -> None:
        if self._eager_checked:
            return
        self._eager_checked = True
        if self.runner.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(
                "[swir] SwiReasoning requires eager execution "
                "(per-decode-step inputs_embeds feedback is incompatible with "
                "captured token-id CUDA graphs -- workorder §4.D). "
                "Launch with --enforce-eager."
            )
        if getattr(self.runner, "use_async_scheduling", False):
            raise RuntimeError(
                "[swir] SwiReasoning is incompatible with async scheduling "
                "(the controller overrides sampled tokens before bookkeeping). "
                "Launch with --no-async-scheduling."
            )

    # ----------------------------------------------------- hook A: inject embeds
    def maybe_inject(
        self, scheduler_output, num_input_tokens, input_ids, inputs_embeds
    ):
        """Called from ``_preprocess``: overwrite each SwiReasoning decode row's
        input embedding with the controller's stashed ``last_emb`` (the soft /
        signal-blended "continuous thought").  No-op unless a swir request has a
        pending embedding (i.e. it has produced at least one token)."""
        if not self.pending_emb:
            return input_ids, inputs_embeds
        req_ids = self.runner.input_batch.req_ids
        if not any(r in self.pending_emb for r in req_ids):
            return input_ids, inputs_embeds

        # Force the embeds path for the whole batch (non-swir rows are
        # numerically unchanged: E[token] == the token-id path).
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.runner.model.embed_input_ids(input_ids)
            input_ids = None

        offset = 0
        for r in req_ids:
            n = scheduler_output.num_scheduled_tokens[r]
            emb = self.pending_emb.get(r)
            if emb is not None:
                if n == 1:
                    inputs_embeds[offset] = emb.to(inputs_embeds.dtype)
                else:
                    # Should not happen in pure decode (spec-off, prompt already
                    # prefilled); skip rather than corrupt a chunked position.
                    logger.warning(
                        "[swir] req %s scheduled %d tokens at decode; "
                        "skipping embedding injection this step",
                        r,
                        n,
                    )
            offset += n
        return input_ids, inputs_embeds

    # ------------------------------------------------ hook B: controller step
    @torch.no_grad()
    def post_sample(self, scheduler_output, logits_orig, sampler_output) -> None:
        """Called from ``sample_tokens`` after ``_sample``: run the SwiReasoning
        FSM on the freshly-sampled swir rows, override the emitted token on
        forced injection, and stash each row's next input embedding.

        ``logits_orig`` must be a CLONE captured before the sampler mutated the
        logits in place (the sampler does float ops in place when logits are
        already fp32)."""
        runner = self.runner
        req_ids = runner.input_batch.req_ids
        rows: list[int] = []
        states: list[SwirRequestState] = []
        for i, r in enumerate(req_ids):
            d = self._req_args(r)
            if d is None:
                continue
            # Skip rows that did not sample a real output token (still in
            # chunked prefill -> their sampled token is discarded downstream).
            if bool(runner.discard_request_mask.np[i]):
                continue
            st = self.states.get(r)
            if st is None:
                st = SwirRequestState(cfg=_build_cfg(d))
                self.states[r] = st
            rows.append(i)
            states.append(st)
        if not rows:
            return

        self._check_interlocks()
        controller = self._ensure_controller()
        idx = torch.tensor(rows, device=logits_orig.device, dtype=torch.long)
        sub_logits = logits_orig.index_select(0, idx)  # [k, vocab]
        sampled = sampler_output.sampled_token_ids[:, 0].index_select(0, idx)  # [k]
        emitted, next_emb, finished = controller.step(sub_logits, sampled, states)

        for j, i in enumerate(rows):
            r = req_ids[i]
            tok = int(emitted[j].item())
            if bool(finished[j].item()):
                # Force-stop on answer-budget exhaustion by emitting EOS so the
                # engine's normal stop logic finishes the request.
                eos = states[j].cfg.eos_token_id
                if eos is not None and tok != eos:
                    tok = eos
            sampler_output.sampled_token_ids[i, 0] = tok
            self.pending_emb[r] = next_emb[j].detach()
            if _DEBUG:
                st = states[j]
                logger.info(
                    "[swir-dbg] req=%s step=%d mode=%d swc=%d inj=%s lock=%s "
                    "budget=%d emit=%d",
                    r,
                    st.step - 1,
                    st.mode,
                    st.switch_count,
                    st.injecting,
                    st.locked_normal,
                    st.answer_budget,
                    tok,
                )
