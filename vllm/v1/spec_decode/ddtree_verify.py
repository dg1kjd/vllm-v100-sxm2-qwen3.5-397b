# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree verifier helpers.

The helpers here consume the payload produced by DFlash and the compact logits
produced by a future tree verifier. They do not run the target model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.ddtree_metadata import (
    DDTreeVerifierMetadata,
    greedy_sample_from_compact_logits,
    make_prefill_tree_attention_mask,
)
from vllm.v1.spec_decode.ddtree_payload import (
    DDTreeDraftPayload,
    tree_from_payload,
)


@dataclass(frozen=True)
class DDTreeVerificationResult:
    accepted_node_indices: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int
    output_token_ids: tuple[int, ...]

    @property
    def num_accepted_tokens(self) -> int:
        return len(self.accepted_token_ids)


@dataclass(frozen=True)
class DDTreeAttentionVerifierInputs:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    compact_logits_indices: torch.Tensor
    metadata: DDTreeVerifierMetadata


def metadata_from_payload(
    *,
    prompt_len: int,
    payload: DDTreeDraftPayload,
) -> DDTreeVerifierMetadata:
    return DDTreeVerifierMetadata.from_tree(
        prompt_len=prompt_len,
        tree=tree_from_payload(payload),
    )


def make_attention_verifier_inputs(
    *,
    prompt_token_ids: torch.Tensor,
    payload: DDTreeDraftPayload,
    mask_dtype: torch.dtype,
) -> DDTreeAttentionVerifierInputs:
    """Build prompt-plus-tree tensors for an attention-only verifier oracle."""

    if prompt_token_ids.ndim != 1:
        raise ValueError("prompt_token_ids must be a 1D tensor")
    if prompt_token_ids.numel() < 1:
        raise ValueError("prompt_token_ids must contain at least one token")

    tree = tree_from_payload(payload)
    metadata = DDTreeVerifierMetadata.from_tree(
        prompt_len=int(prompt_token_ids.numel()),
        tree=tree,
    )
    tree_token_ids = torch.tensor(
        metadata.tree_token_ids,
        dtype=prompt_token_ids.dtype,
        device=prompt_token_ids.device,
    )
    input_ids = torch.cat((prompt_token_ids, tree_token_ids), dim=0)
    position_ids = torch.tensor(
        metadata.all_position_ids(),
        dtype=torch.long,
        device=prompt_token_ids.device,
    )
    attention_mask = make_prefill_tree_attention_mask(
        prompt_len=int(prompt_token_ids.numel()),
        tree=tree,
        device=prompt_token_ids.device,
        dtype=mask_dtype,
    )
    compact_logits_indices = torch.tensor(
        metadata.compact_logits_indices,
        dtype=torch.long,
        device=prompt_token_ids.device,
    )

    return DDTreeAttentionVerifierInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        compact_logits_indices=compact_logits_indices,
        metadata=metadata,
    )


def greedy_verify_payload_from_compact_logits(
    *,
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
) -> DDTreeVerificationResult:
    """Greedy-verify a DDTree payload from compact root-plus-node logits."""

    tree = tree_from_payload(payload)
    walk = greedy_sample_from_compact_logits(
        tree=tree,
        compact_logits=compact_logits,
    )
    return DDTreeVerificationResult(
        accepted_node_indices=walk.accepted_node_indices,
        accepted_token_ids=walk.accepted_token_ids,
        bonus_token_id=walk.bonus_token_id,
        output_token_ids=walk.output_token_ids,
    )
