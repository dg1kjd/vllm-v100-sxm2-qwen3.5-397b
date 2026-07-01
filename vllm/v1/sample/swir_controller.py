# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiReasoning decode-time controller (WORKORDER_rio_swireasoning.md §5.P3).

Backend-agnostic transcription of the SwiReasoning per-step FSM from the
reference repo's ``generate_swir`` (github.com/sdc17/SwiReasoning), pinned
token-exact by ``wo_swir_oracle.py:my_generate_swir`` (HANDOFF_rio_swir_p2.md
§3.2).  This module is the SINGLE source of truth for the control logic:

  - the scratch oracle imports it by path to gate it against ``my_generate_swir``
    (and transitively the reference), and
  - the vLLM V1 ``GPUModelRunner`` imports it to drive the real eager decode loop.

IMPORTANT: keep this file free of any ``vllm`` import so the scratch
transformers env (no vllm) can ``sys.path``-import it standalone.  Pure
torch + dataclasses only.

Mechanics (per request, per decode step ``t`` -- mirrors the oracle op order
``wo_swir_oracle.py:132-218``):

  1.  ``probs_original = softmax(logits_original)`` over the FULL unfiltered vocab.
  2.  a token is sampled/emitted EVERY step by the host (vLLM's sampler); we take
      it as ``sampled_token`` (the latent-step "readout").  EOS / ``</think>``
      detection runs on it.
  3.  lock-to-explicit once ``</think>`` is sampled (``locked_normal``).
  4.  switch-count INJECTION may override the emitted token (force-feed a queued
      ``</think>`` / termination id).
  5.  entropy-trend FSM: ``cur_entropy = -sum p log p``; compare to a running
      ``ref_entropy`` (reset each switch) to flip mode soft<->normal, gated by a
      dwell ``window_size`` and the ``</think>`` lock.
  6.  the NEXT step's input embedding ``last_emb`` is either ``E[emitted]``
      (explicit) or ``probs_original @ E`` (latent "continuous thought"), with
      annealed signal-token blends on transitions.
  7.  switch-count control: after ``max_switch_count`` soft->normal completions,
      inject ``</think>``; past ``2*max_switch_count`` inject a termination
      string and arm an answer budget.

Divergence from the reference (correct generalization for vLLM): the reference
uses a single GLOBAL ``step`` shared by a batch that all starts together; here
each request carries its OWN step counter (its decode-step index).  For a batch
that starts together these coincide, so oracle parity is exact.

The soft-embedding op ``probs_original @ E`` is provided by an injected callable
(``soft_emb_fn``) so the TP8 vocab-parallel + all-reduce implementation (P4) can
be swapped in without touching the FSM; the default ``DenseEmbedOps`` does the
local matmul used on TP1 / the small-model oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Modes (match the reference: 0 == soft/latent, 1 == normal/explicit).
MODE_SOFT = 0
MODE_NORMAL = 1

_ENTROPY_EPS = 1e-12


@dataclass
class SwirConfig:
    """Per-request SwiReasoning knobs + the model-specific signal token ids.

    Defaults mirror the reference repo (HANDOFF §3.2 / workorder §3.7):
    alpha_0=1.0, beta_0=0.7, window_size=512, max_switch_count=2,
    termination_max_tokens=32.  ``max_new_tokens`` feeds the alpha/beta anneal
    denominator only (the host enforces the real ``max_tokens`` stop).
    """

    # Signal token ids (model/tokenizer specific -- caller supplies them).
    think_id: int
    end_think_id: int
    line_break_id: int
    eos_token_id: int | None
    # Forced-injection token id sequences.
    convergence_ids: list[int]
    termination_ids: list[int]
    # Anneal / dwell / switching.
    alpha_0: float = 1.0
    beta_0: float = 0.7
    window_size: int = 512
    max_switch_count: int | None = 2
    termination_max_tokens: int = 32
    max_new_tokens: int = 32768
    # Optional math-keep: force explicit mode when a math symbol is sampled.
    math_ids: list[int] | None = None


@dataclass
class SwirRequestState:
    """Mutable per-request controller state, threaded through the decode loop.

    Carries its OWN :class:`SwirConfig` so a single batched forward can mix
    requests with different knobs (window/alpha/switch-count) -- the controller
    reads ``st.cfg`` per row rather than a single global config.
    """

    cfg: "SwirConfig | None" = None
    mode: int = MODE_SOFT
    mode_stay_steps: int = 0
    cur_ref_entropy: float = 0.0
    locked_normal: bool = False
    switch_count: int = 0
    injecting: bool = False
    inject_queue: list[int] = field(default_factory=list)
    answer_budget: int = -1  # -1 == inactive; counts down to 0 == finish.
    step: int = 0  # this request's own decode-step index (0 == first sample).
    finished: bool = False


class DenseEmbedOps:
    """TP1 / oracle embedding ops: whole vocab embedding matrix on one device.

    ``E`` is [vocab, dim].  Both ops are exact local torch; the TP8 P4
    implementation replaces this with a vocab-parallel matmul + all-reduce and
    fp32-accumulated entropy, behind the same interface.
    """

    def __init__(self, embed_weight: torch.Tensor):
        self.E = embed_weight  # [vocab, dim]

    def embed_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [B] -> [B, dim]
        return self.E[token_ids]

    def soft_emb(self, probs_original: torch.Tensor) -> torch.Tensor:
        # [B, vocab] @ [vocab, dim] -> [B, dim]
        return torch.matmul(probs_original.to(self.E.dtype), self.E)


class CallableEmbedOps:
    """Adapter so the host (vLLM runner) injects its own TP-correct embedding
    ops without the controller importing anything host-specific:

      - ``embed_ids_fn(token_ids[B]) -> [B, dim]``  (e.g. ``model.embed_input_ids``,
        which is vocab-parallel-correct on TP>1).
      - ``soft_emb_fn(probs[B, vocab]) -> [B, dim]``  (``probs @ E``; on TP>1 the
        vocab-parallel partial matmul + all-reduce, with fp32 accumulation).
    """

    def __init__(self, embed_ids_fn, soft_emb_fn):
        self._embed_ids = embed_ids_fn
        self._soft_emb = soft_emb_fn

    def embed_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_ids(token_ids)

    def soft_emb(self, probs_original: torch.Tensor) -> torch.Tensor:
        return self._soft_emb(probs_original)


class SwirController:
    """Batched per-step SwiReasoning controller for a set of active rows.

    The host calls :meth:`step` once per forward, passing the rows (in any
    stable order) that have a sampled token this step together with their
    persistent :class:`SwirRequestState`.  Heavy ops (full-vocab softmax /
    entropy, ``probs @ E``) are batched; the scalar FSM / signal-token blends /
    injection bookkeeping run per row (agentic batches are tiny).

    :meth:`step` mutates each state in place and returns, per row:
      - ``emitted``  [B] int64  -- token to actually emit (post-injection).
      - ``next_emb`` [B, dim]   -- embedding to feed as that row's NEXT input.
      - ``finished`` [B] bool   -- request hit the answer-budget terminator.
    """

    def __init__(self, embed_ops):
        self.ops = embed_ops

    @torch.no_grad()
    def step(
        self,
        logits_original: torch.Tensor,  # [B, vocab] raw logits at sampled pos
        sampled_tokens: torch.Tensor,  # [B] int -- host sampler's readout token
        states: list[SwirRequestState],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = logits_original.device
        B = logits_original.shape[0]
        assert len(states) == B

        # --- batched heavy ops (oracle 133, 162, 186) ---------------------
        # Entropy / soft_emb use the ORIGINAL unfiltered logits (fp32 for the
        # softmax/entropy numerics -- FP16_GUARD lore, workorder §4.B).
        probs_original = torch.softmax(logits_original.float(), dim=-1)
        cur_entropy = -(
            probs_original * probs_original.clamp(min=_ENTROPY_EPS).log()
        ).sum(dim=-1)  # [B]
        soft_emb_all = self.ops.soft_emb(probs_original)  # [B, dim]

        emitted = sampled_tokens.clone().to(torch.int64)  # may be overridden
        dim = soft_emb_all.shape[-1]
        next_emb = torch.empty(
            (B, dim), dtype=soft_emb_all.dtype, device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for i, st in enumerate(states):
            cfg = st.cfg
            assert cfg is not None, "SwirRequestState.cfg must be set"
            use_switch = cfg.max_switch_count is not None
            math_ids = set(cfg.math_ids) if cfg.math_ids else None
            step = st.step
            ent = float(cur_entropy[i].item())

            # (2-3) lock to explicit once </think> is the SAMPLED token.
            if int(sampled_tokens[i].item()) == cfg.end_think_id:
                st.locked_normal = True

            # (4) consume a queued forced-injection token (overrides emit).
            if use_switch and st.injecting and st.inject_queue:
                emitted[i] = st.inject_queue.pop(0)
                if not st.inject_queue:
                    st.injecting = False

            # (5) entropy-trend FSM.
            to_normal = False
            to_soft = False
            if step == 0:
                st.cur_ref_entropy = ent
            else:
                st.mode_stay_steps += 1
                allow_switch = st.mode_stay_steps >= cfg.window_size
                if st.mode == MODE_SOFT and ent < st.cur_ref_entropy:
                    to_normal = True
                elif (
                    st.mode == MODE_NORMAL
                    and ent > st.cur_ref_entropy
                    and allow_switch
                    and not st.locked_normal
                ):
                    to_soft = True
                if to_normal:
                    st.mode = MODE_NORMAL
                if to_soft:
                    st.mode = MODE_SOFT
                if to_normal or to_soft:
                    st.mode_stay_steps = 0
                    st.cur_ref_entropy = ent
                if use_switch and to_normal:
                    st.switch_count += 1

            # (6) mode resolution (+ optional math-keep).
            is_normal = (st.mode == MODE_NORMAL) or st.locked_normal
            if math_ids is not None and int(emitted[i].item()) in math_ids:
                is_normal = True
            is_soft = not is_normal

            # (7) build this row's next-input embedding.
            soft_emb = soft_emb_all[i]
            normal_emb = self.ops.embed_ids(emitted[i : i + 1]).squeeze(0)

            alpha = cfg.alpha_0 + (1.0 - cfg.alpha_0) * float(step) / float(
                cfg.max_new_tokens
            )
            if step == 0:
                line_break_emb = self.ops.embed_ids(
                    torch.tensor([cfg.line_break_id], device=device)
                ).squeeze(0)
                soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
            elif to_soft:
                think_emb = self.ops.embed_ids(
                    torch.tensor([cfg.think_id], device=device)
                ).squeeze(0)
                soft_emb = alpha * soft_emb + (1.0 - alpha) * think_emb

            beta = cfg.beta_0 + (1.0 - cfg.beta_0) * float(step) / float(
                cfg.max_new_tokens
            )
            if step > 0 and to_normal:
                end_think_emb = self.ops.embed_ids(
                    torch.tensor([cfg.end_think_id], device=device)
                ).squeeze(0)
                normal_emb = beta * soft_emb + (1.0 - beta) * end_think_emb

            next_emb[i] = soft_emb if is_soft else normal_emb

            # (7b) switch-count control: queue forced injections + answer budget.
            if use_switch and step > 0:
                msc = cfg.max_switch_count
                if to_normal and msc <= st.switch_count <= 2 * msc:
                    st.inject_queue = list(cfg.convergence_ids)
                    st.injecting = True
                elif to_normal and st.switch_count > 2 * msc:
                    st.inject_queue = list(cfg.termination_ids)
                    st.injecting = True
                    st.answer_budget = cfg.termination_max_tokens
                if st.answer_budget >= 0:
                    st.answer_budget -= 1

            # (8) finish conditions: EOS readout or answer budget exhausted.
            tok = int(emitted[i].item())
            fin = (cfg.eos_token_id is not None and tok == cfg.eos_token_id)
            if use_switch and st.answer_budget == 0:
                fin = True
            st.finished = bool(st.finished or fin)
            finished[i] = st.finished

            st.step += 1

        return emitted, next_emb, finished
