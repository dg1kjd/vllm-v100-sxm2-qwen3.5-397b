# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.spec_decode.ddtree_tree import build_ddtree, greedy_tree_walk


def test_budget_one_matches_top1() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
        ],
        budget=1,
        top_k=2,
        root_token_id=7,
    )

    assert tree.token_ids_for_verifier() == (11,)
    assert tree.parent_indices_for_verifier() == (-1,)
    assert tree.node_depths_for_verifier() == (1,)


def test_chain_seed_preserves_top1_path() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=3,
        top_k=2,
    )

    assert tree.token_ids_for_verifier() == (11, 21, 31)
    assert tree.parent_indices_for_verifier() == (-1, 0, 1)
    assert tree.path_token_ids(3) == (11, 21, 31)


def test_best_first_adds_sibling_branch() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.15)],
            [(21, -0.1), (22, -0.12)],
            [(31, -0.1), (32, -2.0)],
        ],
        budget=5,
        top_k=2,
    )

    assert tree.token_ids_for_verifier()[:3] == (11, 21, 31)
    assert 12 in tree.child_by_token(0)
    assert 22 in tree.child_by_token(1)


def test_visibility_is_ancestor_only() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=5,
        top_k=2,
    )
    visibility = tree.visibility_mask()

    node_11 = tree.child_by_token(0)[11]
    node_12 = tree.child_by_token(0)[12]
    node_21 = tree.child_by_token(node_11)[21]

    assert visibility[node_21][0]
    assert visibility[node_21][node_11]
    assert visibility[node_21][node_21]
    assert not visibility[node_21][node_12]


def test_greedy_tree_walk_accepts_path_and_bonus() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=4,
        top_k=2,
    )

    next_tokens = {
        (): 11,
        (11,): 21,
        (11, 21): 99,
    }
    walk = greedy_tree_walk(tree, lambda path: next_tokens[path])

    assert walk.accepted_token_ids == (11, 21)
    assert walk.bonus_token_id == 99
    assert walk.output_token_ids == (11, 21, 99)


def test_topk_one_budget_sixteen_is_flat_dflash_chain() -> None:
    candidates = [[(1000 + depth, -0.1)] for depth in range(16)]

    tree = build_ddtree(candidates, budget=16, top_k=1, chain_seed=True)

    assert tree.token_ids_for_verifier() == tuple(range(1000, 1016))
    assert tree.parent_indices_for_verifier() == (-1,) + tuple(range(15))
    assert tree.node_depths_for_verifier() == tuple(range(1, 17))
