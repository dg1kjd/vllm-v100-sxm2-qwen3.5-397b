# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.ddtree_payload import DDTreeDraftPayload
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _branched_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 12),
        parent_indices=(-1, 0, -1),
        node_depths=(1, 2, 1),
        node_scores=(0.0, 0.0, -0.1),
        top1_chain_token_ids=(11, 21),
        flat_draft_token_ids=(11, 21),
        budget=3,
        top_k=2,
        chain_seed=True,
    )


def _flat_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 31),
        parent_indices=(-1, 0, 1),
        node_depths=(1, 2, 3),
        node_scores=(0.0, 0.0, 0.0),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=3,
        top_k=1,
        chain_seed=True,
    )


def _scheduler_output(payload: DDTreeDraftPayload) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r0": 5, "r1": 2},
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={"r0": list(payload.tree_token_ids)},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        scheduled_ddtree_payloads={"r0": payload},
    )


def test_apply_ddtree_position_overrides_uses_node_depths_on_spec_tail() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.input_batch = SimpleNamespace(req_ids=["r0", "r1"])
    runner.positions = torch.tensor(
        [100, 101, 102, 103, 104, 200, 201],
        dtype=torch.int64,
    )
    payload = _branched_payload()

    runner._apply_ddtree_position_overrides(
        _scheduler_output(payload),
        num_reqs=2,
        num_scheduled_tokens=np.array([5, 2], dtype=np.int32),
        cu_num_tokens=np.array([5, 7], dtype=np.int32),
    )

    assert runner.positions.tolist() == [100, 101, 102, 103, 102, 200, 201]


def test_apply_ddtree_position_overrides_requires_scheduled_tree_tokens() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.input_batch = SimpleNamespace(req_ids=["r0"])
    runner.positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    payload = _branched_payload()
    scheduler_output = _scheduler_output(payload)
    scheduler_output.scheduled_spec_decode_tokens = {"r0": [11, 21]}

    runner._apply_ddtree_position_overrides(
        scheduler_output,
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        cu_num_tokens=np.array([4], dtype=np.int32),
    )

    assert runner.positions.tolist() == [10, 11, 12, 13]


def test_ddtree_accepted_kv_local_copies_move_branch_to_prefix() -> None:
    copies = GPUModelRunner._ddtree_accepted_kv_local_copies(
        req_ids=["r0"],
        num_scheduled_tokens={"r0": 6},
        scheduled_spec_decode_tokens={"r0": [11, 21, 31, 12, 22]},
        accepted_node_indices=torch.tensor([[4, 5, -1]], dtype=torch.int32),
    )

    assert copies == [(4, 1), (5, 2)]


def test_ddtree_accepted_nodes_are_flat_prefix() -> None:
    assert GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[1, 2, 3], [-1, -1, -1]], dtype=torch.int32)
    )
    assert not GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[1, 3, -1]], dtype=torch.int32)
    )
    assert not GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[4, 5, -1]], dtype=torch.int32)
    )


def test_ddtree_state_slot_selectors_use_last_accepted_node() -> None:
    selectors = GPUModelRunner._ddtree_state_slot_selectors_from_accepted_nodes(
        torch.tensor(
            [
                [1, 2, -1],
                [-1, -1, -1],
                [4, 5, -1],
            ],
            dtype=torch.int32,
        )
    )

    assert selectors.tolist() == [3, 1, 6]


def test_update_states_after_model_execute_uses_ddtree_state_selector() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = object()
    runner.model_config = SimpleNamespace(is_hybrid=True)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    runner.num_accepted_tokens = SimpleNamespace(
        gpu=torch.ones(1, dtype=torch.int32)
    )
    runner.spec_state_slot_selectors = SimpleNamespace(
        gpu=torch.ones(1, dtype=torch.int32)
    )
    runner.input_batch = SimpleNamespace(
        num_accepted_tokens_cpu_tensor=torch.ones(1, dtype=torch.int32),
        spec_num_accepted_tokens_cpu_tensor=torch.ones(1, dtype=torch.int32),
    )
    runner.num_accepted_tokens_event = SimpleNamespace(record=lambda: None)

    runner._update_states_after_model_execute(
        torch.tensor([[12, 99]], dtype=torch.int32),
        _scheduler_output(_branched_payload()),
        torch.tensor([[3]], dtype=torch.int32),
    )

    assert runner.num_accepted_tokens.gpu.tolist() == [2]
    assert runner.spec_state_slot_selectors.gpu.tolist() == [4]
    assert runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [2]
    assert runner.input_batch.spec_num_accepted_tokens_cpu_tensor.tolist() == [4]


def test_compact_ddtree_drafter_context_moves_branch_to_prefix() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["r0"], num_reqs=1)
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor([10, 11, 21, 31, 12, 22], dtype=torch.int32)
    )
    hidden_states = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    aux_hidden = [torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)]
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[12, 22, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[4, 5]], dtype=torch.int32),
    )
    payload = DDTreeDraftPayload(
        tree_token_ids=(11, 21, 31, 12, 22),
        parent_indices=(-1, 0, 1, -1, 3),
        node_depths=(1, 2, 3, 1, 2),
        node_scores=(0.0, 0.0, 0.0, -0.1, -0.2),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=5,
        top_k=2,
        chain_seed=True,
    )
    scheduler_output = _scheduler_output(payload)
    scheduler_output.num_scheduled_tokens = {"r0": 6}
    scheduler_output.total_num_scheduled_tokens = 6
    scheduler_output.scheduled_spec_decode_tokens = {
        "r0": list(payload.tree_token_ids)
    }

    runner._compact_ddtree_drafter_context(
        hidden_states,
        aux_hidden,
        sampler_output,
        scheduler_output,
    )

    assert hidden_states[1].tolist() == [8.0, 9.0]
    assert hidden_states[2].tolist() == [10.0, 11.0]
    assert aux_hidden[0][1].tolist() == [12.0, 13.0, 14.0]
    assert aux_hidden[0][2].tolist() == [15.0, 16.0, 17.0]
    assert runner.input_ids.gpu.tolist() == [10, 12, 22, 31, 12, 22]


def test_validate_ddtree_hybrid_state_path_allows_flat_chain() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(is_hybrid=True)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11, 21, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[1, 2]], dtype=torch.int32),
    )

    runner._validate_ddtree_hybrid_state_path(
        sampler_output,
        _scheduler_output(_flat_payload()),
    )


def test_validate_ddtree_hybrid_state_path_rejects_branch_payload() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(is_hybrid=True)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[12, 21, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[4, 5]], dtype=torch.int32),
    )

    try:
        runner._validate_ddtree_hybrid_state_path(
            sampler_output,
            _scheduler_output(_branched_payload()),
        )
    except RuntimeError as exc:
        assert "tree-aware GDN/Mamba" in str(exc)
        assert "branched" in str(exc)
    else:
        raise AssertionError("expected hybrid DDTree branch guard to raise")


def test_copy_attention_kv_slot_copies_both_kv_planes() -> None:
    kv_cache = torch.arange(2 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(
        2,
        2,
        4,
        1,
        1,
    )
    src = kv_cache[1, :, 2].clone()

    GPUModelRunner._copy_attention_kv_slot(
        kv_cache,
        src_slot=6,
        dst_slot=1,
        block_size=4,
    )

    assert torch.equal(kv_cache[0, :, 1], src)
