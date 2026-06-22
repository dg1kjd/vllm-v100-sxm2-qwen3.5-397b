# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree draft payload construction.

This module converts a batched DFlash logits block into per-request DDTree
payloads. It is intentionally separate from the verifier hot path so the tree
builder can be tested and evolved without changing scheduler semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.ddtree_tree import DDTree, DDTreeNode, build_ddtree


@dataclass(frozen=True)
class DDTreeDraftPayload:
    """Flattened per-request DDTree payload produced by a draft model."""

    tree_token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    node_depths: tuple[int, ...]
    node_scores: tuple[float, ...]
    top1_chain_token_ids: tuple[int, ...]
    flat_draft_token_ids: tuple[int, ...]
    budget: int
    top_k: int
    chain_seed: bool
    topk_token_ids_by_depth: tuple[tuple[int, ...], ...] = ()
    topk_logprobs_by_depth: tuple[tuple[float, ...], ...] = ()

    @property
    def num_tree_nodes(self) -> int:
        return len(self.tree_token_ids)

    def flat_chain_matches_top1(self) -> bool:
        return self.flat_draft_token_ids == self.top1_chain_token_ids

    def is_flat_chain(self) -> bool:
        """Return whether verifier nodes are exactly the linear draft chain."""
        num_nodes = self.num_tree_nodes
        expected_parents = () if num_nodes == 0 else (-1,) + tuple(range(num_nodes - 1))
        expected_depths = tuple(range(1, num_nodes + 1))
        return (
            self.tree_token_ids == self.flat_draft_token_ids
            and self.parent_indices == expected_parents
            and self.node_depths == expected_depths
        )


def payload_from_tree(
    *,
    tree: DDTree,
    top1_chain_token_ids: tuple[int, ...],
    flat_draft_token_ids: tuple[int, ...],
    budget: int,
    top_k: int,
    chain_seed: bool,
    topk_token_ids_by_depth: tuple[tuple[int, ...], ...] = (),
    topk_logprobs_by_depth: tuple[tuple[float, ...], ...] = (),
) -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=tree.token_ids_for_verifier(),
        parent_indices=tree.parent_indices_for_verifier(),
        node_depths=tree.node_depths_for_verifier(),
        node_scores=tuple(node.score for node in tree.non_root_nodes),
        top1_chain_token_ids=top1_chain_token_ids,
        flat_draft_token_ids=flat_draft_token_ids,
        budget=budget,
        top_k=top_k,
        chain_seed=chain_seed,
        topk_token_ids_by_depth=topk_token_ids_by_depth,
        topk_logprobs_by_depth=topk_logprobs_by_depth,
    )


def tree_from_payload(
    payload: DDTreeDraftPayload,
    *,
    root_token_id: int = -1,
) -> DDTree:
    """Rebuild a DDTree from verifier-coordinate payload arrays."""

    num_nodes = len(payload.tree_token_ids)
    if len(payload.parent_indices) != num_nodes:
        raise ValueError("payload parent_indices length mismatch")
    if len(payload.node_depths) != num_nodes:
        raise ValueError("payload node_depths length mismatch")
    if len(payload.node_scores) != num_nodes:
        raise ValueError("payload node_scores length mismatch")

    nodes: list[DDTreeNode] = [
        DDTreeNode(
            index=0,
            parent_index=None,
            token_id=root_token_id,
            depth=0,
            score=0.0,
        )
    ]
    for verifier_index, (
        token_id,
        parent_index,
        depth,
        score,
    ) in enumerate(
        zip(
            payload.tree_token_ids,
            payload.parent_indices,
            payload.node_depths,
            payload.node_scores,
            strict=True,
        ),
        start=1,
    ):
        if parent_index < -1 or parent_index >= verifier_index - 1:
            raise ValueError(
                "payload parent_indices must reference an earlier verifier node"
            )
        tree_parent_index = 0 if parent_index == -1 else parent_index + 1
        expected_depth = nodes[tree_parent_index].depth + 1
        if depth != expected_depth:
            raise ValueError(
                f"payload depth mismatch for node {verifier_index}: "
                f"expected {expected_depth}, got {depth}"
            )
        nodes.append(
            DDTreeNode(
                index=verifier_index,
                parent_index=tree_parent_index,
                token_id=token_id,
                depth=depth,
                score=score,
            )
        )

    return DDTree(nodes=tuple(nodes))


def build_ddtree_payloads_from_logits(
    *,
    logits: torch.Tensor,
    batch_size: int,
    num_speculative_tokens: int,
    budget: int,
    top_k: int,
    chain_seed: bool,
    flat_draft_token_ids: torch.Tensor | None = None,
) -> tuple[DDTreeDraftPayload, ...]:
    """Build per-request DDTree payloads from DFlash first-pass logits.

    Args:
        logits: Tensor with shape ``[batch_size * num_speculative_tokens, vocab]``.
        batch_size: Number of active requests.
        num_speculative_tokens: DFlash proposal depth.
        budget: Maximum number of non-root DDTree verifier nodes.
        top_k: Candidate count per depth.
        chain_seed: Whether to seed the tree with the top-1 chain.
        flat_draft_token_ids: Existing flat draft output, shape
            ``[batch_size, num_speculative_tokens]``. This is carried for
            parity checks and does not change scheduler behavior.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch * k, vocab]")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    expected_rows = batch_size * num_speculative_tokens
    if logits.shape[0] != expected_rows:
        raise ValueError(
            f"logits row mismatch: expected {expected_rows}, got {logits.shape[0]}"
        )
    vocab_size = logits.shape[1]
    effective_top_k = min(top_k, vocab_size)

    float_logits = logits.float()
    topk_logits, topk_token_ids = torch.topk(
        float_logits,
        k=effective_top_k,
        dim=-1,
    )
    log_normalizer = torch.logsumexp(float_logits, dim=-1, keepdim=True)
    topk_logprobs = topk_logits - log_normalizer

    topk_token_ids_cpu = topk_token_ids.view(
        batch_size,
        num_speculative_tokens,
        effective_top_k,
    ).detach().cpu()
    topk_logprobs_cpu = topk_logprobs.view(
        batch_size,
        num_speculative_tokens,
        effective_top_k,
    ).detach().cpu()

    if flat_draft_token_ids is None:
        flat_draft_token_ids_cpu = topk_token_ids_cpu[:, :, 0]
    else:
        if flat_draft_token_ids.shape != (batch_size, num_speculative_tokens):
            raise ValueError(
                "flat_draft_token_ids must have shape "
                f"[{batch_size}, {num_speculative_tokens}]"
            )
        flat_draft_token_ids_cpu = flat_draft_token_ids.detach().cpu()

    payloads: list[DDTreeDraftPayload] = []
    for req_idx in range(batch_size):
        candidates_by_depth = [
            [
                (
                    int(topk_token_ids_cpu[req_idx, depth, candidate_idx].item()),
                    float(topk_logprobs_cpu[req_idx, depth, candidate_idx].item()),
                )
                for candidate_idx in range(effective_top_k)
            ]
            for depth in range(num_speculative_tokens)
        ]
        topk_token_ids_by_depth = tuple(
            tuple(token_id for token_id, _ in candidates)
            for candidates in candidates_by_depth
        )
        topk_logprobs_by_depth = tuple(
            tuple(logprob for _, logprob in candidates)
            for candidates in candidates_by_depth
        )
        tree = build_ddtree(
            candidates_by_depth,
            budget=budget,
            top_k=effective_top_k,
            chain_seed=chain_seed,
        )
        top1_chain = tuple(
            int(token_id.item()) for token_id in topk_token_ids_cpu[req_idx, :, 0]
        )
        flat_chain = tuple(
            int(token_id.item()) for token_id in flat_draft_token_ids_cpu[req_idx]
        )
        payloads.append(
            payload_from_tree(
                tree=tree,
                top1_chain_token_ids=top1_chain,
                flat_draft_token_ids=flat_chain,
                budget=budget,
                top_k=effective_top_k,
                chain_seed=chain_seed,
                topk_token_ids_by_depth=topk_token_ids_by_depth,
                topk_logprobs_by_depth=topk_logprobs_by_depth,
            )
        )

    return tuple(payloads)
