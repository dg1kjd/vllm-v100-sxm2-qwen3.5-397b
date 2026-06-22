# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy DDTree sampler over compact target logits."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.ddtree_payload import (
    DDTreeDraftPayload,
    tree_from_payload,
)
from vllm.v1.spec_decode.ddtree_verify import (
    DDTreeVerificationResult,
    greedy_verify_payload_from_compact_logits,
)

PLACEHOLDER_TOKEN_ID = -1
logger = init_logger(__name__)


def _ddtree_debug_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_DEBUG", "0") == "1"


def _candidate_rank(
    payload: DDTreeDraftPayload,
    *,
    depth: int,
    token_id: int,
) -> int | None:
    if depth < 1 or depth > len(payload.topk_token_ids_by_depth):
        return None
    try:
        return payload.topk_token_ids_by_depth[depth - 1].index(token_id)
    except ValueError:
        return None


def _debug_log_verification(
    *,
    req_id: str,
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
    result: DDTreeVerificationResult,
) -> None:
    if not _ddtree_debug_enabled():
        return

    tree = tree_from_payload(payload)
    target_argmax = compact_logits.argmax(dim=-1).detach().cpu().tolist()
    cursor = 0
    rows: list[str] = []
    while True:
        if cursor >= len(target_argmax):
            rows.append(f"cursor={cursor} missing_compact_row")
            break
        next_token = int(target_argmax[cursor])
        children = tree.child_by_token(cursor)
        child_index = children.get(next_token)
        depth = tree.nodes[cursor].depth + 1
        rank = _candidate_rank(payload, depth=depth, token_id=next_token)
        rows.append(
            "row=%d parent=%d depth=%d target=%d child=%s topk_rank=%s "
            "child_tokens=%s"
            % (
                cursor,
                cursor,
                depth,
                next_token,
                child_index,
                rank,
                tuple(children.keys())[:8],
            )
        )
        if child_index is None:
            break
        cursor = child_index

    logger.info(
        "DFLASH_DDTREE_DEBUG verify req=%s accepted=%s output=%s "
        "flat=%s top1=%s tree_head=%s walk=%s",
        req_id,
        result.accepted_node_indices,
        result.output_token_ids,
        payload.flat_draft_token_ids,
        payload.top1_chain_token_ids,
        payload.tree_token_ids[: min(16, len(payload.tree_token_ids))],
        rows,
    )


@dataclass(frozen=True)
class DDTreeGreedySamplerResult:
    sampler_output: SamplerOutput
    verification_results: tuple[DDTreeVerificationResult, ...]


def _padded_int_tensor(
    rows: Sequence[tuple[int, ...]],
    *,
    device: torch.device,
) -> torch.Tensor:
    max_len = max((len(row) for row in rows), default=1)
    out = torch.full(
        (len(rows), max_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        out[row_idx, : len(row)] = torch.tensor(
            row,
            dtype=torch.int32,
            device=device,
        )
    return out


def greedy_sample_ddtree_payloads(
    *,
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload],
    compact_logits: torch.Tensor,
    num_draft_tokens: Sequence[int],
) -> DDTreeGreedySamplerResult | None:
    """Sample accepted DDTree paths from compact target logits.

    ``compact_logits`` is expected to use the same per-request row order as
    flat speculative decode today: root row followed by one row per scheduled
    draft/tree node. Until the verifier input path is tree-shaped, callers must
    only pass payloads whose tree node count equals the scheduled flat draft
    length.
    """

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must have shape [rows, vocab]")
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")

    rows_consumed = 0
    output_token_rows: list[tuple[int, ...]] = []
    accepted_node_rows: list[tuple[int, ...]] = []
    results: list[DDTreeVerificationResult] = []
    for req_id, draft_len in zip(req_ids, num_draft_tokens, strict=True):
        num_rows = int(draft_len) + 1
        payload = payload_by_req_id.get(req_id)
        if payload is None or payload.num_tree_nodes != int(draft_len):
            return None
        req_logits = compact_logits[rows_consumed : rows_consumed + num_rows]
        result = greedy_verify_payload_from_compact_logits(
            payload=payload,
            compact_logits=req_logits,
        )
        _debug_log_verification(
            req_id=req_id,
            payload=payload,
            compact_logits=req_logits,
            result=result,
        )
        if _ddtree_debug_enabled():
            logger.info(
                "DFLASH_DDTREE_DEBUG sampler req=%s draft_len=%d "
                "accepted_nodes=%s accepted_tokens=%s bonus=%s output=%s",
                req_id,
                int(draft_len),
                result.accepted_node_indices,
                result.accepted_token_ids,
                result.bonus_token_id,
                result.output_token_ids,
            )
        results.append(result)
        output_token_rows.append(result.output_token_ids)
        accepted_node_rows.append(result.accepted_node_indices)
        rows_consumed += num_rows

    if rows_consumed != compact_logits.shape[0]:
        raise ValueError(
            f"compact_logits row mismatch: consumed {rows_consumed}, "
            f"got {compact_logits.shape[0]}"
        )

    sampled_token_ids = _padded_int_tensor(
        output_token_rows,
        device=compact_logits.device,
    )
    accepted_node_indices = _padded_int_tensor(
        accepted_node_rows,
        device=compact_logits.device,
    )
    return DDTreeGreedySamplerResult(
        sampler_output=SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        ),
        verification_results=tuple(results),
    )
