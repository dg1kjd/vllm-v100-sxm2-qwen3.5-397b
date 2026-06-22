# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree_payload import build_ddtree_payloads_from_logits
from vllm.v1.spec_decode.dflash import DFlashProposer


class _GreedySamplingMetadata:
    all_greedy = True


def test_topk_one_payload_matches_flat_dflash_chain() -> None:
    batch_size = 2
    num_speculative_tokens = 4
    vocab_size = 128
    flat_draft_token_ids = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ],
        dtype=torch.int64,
    )
    logits = torch.full(
        (batch_size * num_speculative_tokens, vocab_size),
        -100.0,
        dtype=torch.float32,
    )
    for row, token_id in enumerate(flat_draft_token_ids.flatten()):
        logits[row, int(token_id.item())] = 10.0

    payloads = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        budget=num_speculative_tokens,
        top_k=1,
        chain_seed=True,
        flat_draft_token_ids=flat_draft_token_ids,
    )

    assert len(payloads) == batch_size
    for payload, flat_chain in zip(
        payloads,
        flat_draft_token_ids.tolist(),
        strict=True,
    ):
        assert payload.tree_token_ids == tuple(flat_chain)
        assert payload.parent_indices == (-1, 0, 1, 2)
        assert payload.node_depths == (1, 2, 3, 4)
        assert payload.top1_chain_token_ids == tuple(flat_chain)
        assert payload.flat_chain_matches_top1()
        assert payload.is_flat_chain()


def test_payload_builds_sibling_branch_from_topk_logits() -> None:
    vocab_size = 128
    logits = torch.full((3, vocab_size), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[0, 12] = 9.8
    logits[1, 21] = 10.0
    logits[1, 22] = 9.7
    logits[2, 31] = 10.0
    logits[2, 32] = 5.0

    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=5,
        top_k=2,
        chain_seed=True,
    )[0]

    assert payload.tree_token_ids[:3] == (11, 21, 31)
    assert payload.parent_indices[:3] == (-1, 0, 1)
    assert payload.num_tree_nodes == 5
    assert 12 in payload.tree_token_ids
    assert payload.top1_chain_token_ids == (11, 21, 31)
    assert not payload.is_flat_chain()


def test_dflash_ddtree_sampling_hook_builds_payload_from_logits() -> None:
    proposer = object.__new__(DFlashProposer)
    proposer.use_ddtree = True
    proposer.num_speculative_tokens = 3
    proposer.ddtree_budget = 3
    proposer.ddtree_top_k = 1
    proposer.ddtree_chain_seed = True
    proposer._enable_probabilistic_draft_probs = False
    proposer._last_ddtree_payloads = None

    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 31] = 10.0

    draft_token_ids, draft_probs = DFlashProposer._sample_draft_tokens(
        proposer,
        hidden_states=torch.empty((3, 4), dtype=torch.float32),
        sampling_metadata=_GreedySamplingMetadata(),  # type: ignore[arg-type]
        logits=logits,
        spec_step_idx=0,
    )

    assert draft_probs is None
    assert draft_token_ids.tolist() == [11, 21, 31]
    assert proposer._last_ddtree_payloads is not None
    payload = proposer._last_ddtree_payloads[0]
    assert payload.tree_token_ids == (11, 21, 31)
    assert payload.flat_chain_matches_top1()
