# SM70 DFlash DDTree Plan for Qwen3.6-27B-AWQ

This note records the DDTree reference checkout and the local integration plan
for accelerating Qwen3.6-27B-AWQ on V100/SM70. The goal is a vLLM-native
`dflash_ddtree` path that improves the DFlash16 verifier economics without
regressing exactness, long-context state, or the existing flat DFlash path.

## Reference Checkouts

The two public references were cloned under `third_party/ddtree_refs/`:

| Path | Upstream | Commit | License | Role |
|---|---|---|---|---|
| `third_party/ddtree_refs/official_ddtree` | `https://github.com/liranringel/ddtree` | `c96427a185677bf4133ed865dd1626a5041aef9b` | MIT | Algorithm reference for tree build, visibility mask, verifier walk, and cache compaction. |
| `third_party/ddtree_refs/aeon_qwen36_ddtree` | `https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree` | `d2610ae5e42ddcbccbf8cd800c67644d3c7843f6` | Apache-2.0 | vLLM integration map, prototypes, tests, and Qwen3.6 hybrid-state cautions. |

Treat both as references. Do not run the AEON overlay scripts against this tree:
they patch a different GB10/Blackwell branch and include research-only DDTree
paths that are explicitly not production safe.

## Local Baseline

Current 27B-AWQ evidence, all TP2 on GPU2/3 with `FLASH_ATTN_V100`:

- No-spec greedy baseline:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_nospec_i4096_o256_greedy_mnbt8192.json`
  measured `steady_decode_tps=55.86159315981125`.
- DFlash16 full-GDN greedy:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_dflash16_i4096_o256_greedy_mnbt8192.json`
  measured `steady_decode_tps=34.50339715311688`,
  `acceptance_length=4.642857142857142`, and
  `overall_acceptance_rate=0.22767857142857142`.
- DFlash16 official sampling:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_dflash16_i4096_o256_official_mnbt8192.json`
  measured `steady_decode_tps=27.732380043828957`,
  `acceptance_length=3.6956521739130435`, and
  `overall_acceptance_rate=0.16847826086956522`.
- A short DFlash profile showed the slowdown is not caused by repeatedly
  rebuilding the whole 4096-token context after the first round. The bottleneck
  is verifier/draft cost plus low accepted tokens per verifier step.

This makes DDTree the right algorithmic next step, but only if the tree verifier
and Qwen GDN state handling stay exact.

## What Transfers From The References

Useful pieces:

- Official `ddtree.py::build_ddtree_tree`: best-first heap expansion from
  per-position DFlash distributions.
- Official `compile_ddtree_tree`: root plus flattened nodes, position ids, and
  ancestor-only visibility.
- Official `follow_verified_tree`: target-logit walk through children until a
  recovery/bonus token is needed.
- AEON `prototypes/ddtree_tree.py`: cleaner vLLM-shaped tree dataclasses and
  chain-seeded best-first builder.
- AEON `prototypes/ddtree_vllm_metadata.py`: flattened tree token ids,
  parent ids, node depths, compact logit row conventions, and sibling-hiding
  attention mask tests.
- AEON `prototypes/ddtree_gdn_reference.py`: reference semantics for tree
  causal-conv and Gated DeltaNet state, where each node reads its parent state
  and writes its own scratch state.

Parts that should not be copied directly into the hot path:

- The official builder copies top-k logits/probs to CPU and builds the heap in
  Python. That is acceptable for correctness bring-up and instrumentation, but
  not a final V100 decode path.
- The official verifier uses Transformers `DynamicCache` plus a dense 4D mask.
  vLLM needs paged KV, scheduler integration, and Flash-V100-aware attention.
- AEON's patch scripts are overlay scripts for a different branch. The useful
  artifact is the design and prototype logic, not the mechanical patching.

## Local Patch Surface

The local vLLM path is still chain-shaped in the engine hot path:

- `vllm/config/speculative.py` accepts `method="dflash_ddtree"`, but the
  default `ddtree_disable_tree_verify=True` keeps it on the flat DFlash path.
- `vllm/v1/spec_decode/dflash.py::DFlashProposer` asserts
  `speculative_config.use_dflash()` and still produces one top-1 chain of
  length `num_speculative_tokens`.
- `vllm/v1/outputs.py::DraftTokenIds` carries optional per-request
  `DDTreeDraftPayload` objects aligned with `req_ids`.
- `vllm/v1/core/sched/output.py::SchedulerOutput` carries optional scheduled
  DDTree payloads after strict flat-token matching.
- `vllm/v1/spec_decode/metadata.py::SpecDecodeMetadata` describes
  linear draft tokens, target logit rows, and bonus logit rows.
- `vllm/v1/sample/rejection_sampler.py` and
  `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` both assume a
  contiguous chain. A greedy DDTree sampler now bypasses this only for a narrow
  all-greedy/no-logprobs/no-processor route.
- `vllm/v1/attention/backends/flex_attention.py` has an experimental logical
  DDTree ancestor mask. `flash_attn_v100.py` still has no native tree mask.
- Qwen GDN spec metadata now has a DDTree parent-id channel and state/cache
  slot sizing can follow `ddtree_budget`, but kernels still do not consume
  parent ids or commit selected tree-node state. The scheduler therefore
  refuses branched DDTree payloads for hybrid/Qwen3.6 models.

## Implementation Plan

### M0: Reference Capture

Status: complete. The reference repos above are cloned and this document records
their exact commits and licenses.

### M1: Local Tree Builder And Sampler Tests

Status: complete.

Add local, test-only modules under `vllm/v1/spec_decode/`:

- `ddtree_tree.py`: adapt the AEON dataclass builder, with attribution if code
  is copied.
- `ddtree_metadata.py`: compact tree metadata and CPU/torch helper functions.
- `tests/v1/spec_decode/test_ddtree_tree.py`: budget, chain seed, sibling
  branch, visibility, and greedy-walk tests.
- `tests/v1/spec_decode/test_ddtree_metadata.py`: parent offsets, compact logit
  row mapping, and sibling mask invariants.

Pass criteria:

- No engine behavior changes.
- Pure CPU tests pass in the releasecheck conda environment.
- `budget=16, top_k=1` is exactly equivalent to a DFlash16 flat chain.

### M2: Experimental Config Alias With Flat Fallback

Status: flat-fallback entry complete; no tree verifier is active yet.

Add `method="dflash_ddtree"` and explicit DDTree fields:

- `ddtree_budget`: default `None`; if unset, use
  `num_speculative_tokens`.
- `ddtree_top_k`: default `4` for early tests, then sweep.
- `ddtree_chain_seed`: default `True`.
- `ddtree_disable_tree_verify`: default `True` until M3/M4 pass.

Make `dflash_ddtree` instantiate the same DFlash proposer and return the same
flat `draft_token_ids` while carrying optional tree payload out-of-band. This is
a boot and regression milestone, not an acceleration milestone.

Current local state:

- `DFlashModelTypes` includes `dflash_ddtree`.
- `SpeculativeConfig.use_dflash()` covers both `dflash` and `dflash_ddtree`.
- `DFlashProposer` accepts both methods and records the DDTree knobs.
- Tree verifier remains disabled by default, so the runtime behavior is still
  the flat DFlash16 path.

Pass criteria:

- `method="dflash_ddtree"` with tree verify disabled has the same token hash and
  decode speed as `method="dflash"` within normal noise.
- Existing DFlash16 benchmark command still works unchanged.

### M3: DFlash Top-k Payload

Status: payload and scheduler bridge are present; no tree verifier is active
yet.

Extend the DFlash proposer to produce tree payloads from the same one-pass
DFlash block distribution:

- Force a logits/top-k path only for `dflash_ddtree`; current greedy DFlash can
  avoid full logits by using top-1, but DDTree needs top-k per depth.
- Keep returning the canonical top-1 chain as `draft_token_ids`.
- Attach per-request tree payloads:
  `tree_token_ids`, `parent_indices`, `node_depths`, `node_scores`,
  `top1_chain_token_ids`, `flat_draft_token_ids`, `budget`, `top_k`, and
  `chain_seed`.
- First local default for 27B-AWQ:
  `num_speculative_tokens=16`, `ddtree_budget=16`, `ddtree_top_k=1`,
  `ddtree_chain_seed=True`.
- First useful sweep after parity:
  `ddtree_budget in {16, 24, 32, 48}`, `ddtree_top_k in {2, 4}`.

Pass criteria:

- With `budget=16, top_k=1`, payload-derived flat chain equals current DFlash16
  draft ids.
- Tree payload creation overhead is measured separately from verifier time.

Current local state:

- `vllm/v1/spec_decode/ddtree_payload.py` converts
  `[batch * num_speculative_tokens, vocab]` DFlash logits into per-request
  `DDTreeDraftPayload` objects by reusing `build_ddtree()`.
- `DFlashProposer._sample_draft_tokens()` computes one logits block for
  `method="dflash_ddtree"`, reuses it for current flat sampling, and builds the
  DDTree payload from the same logits.
- `GPUModelRunner` caches the payload out-of-band via
  `take_dflash_ddtree_payloads()`. `DraftTokenIds` and scheduler output now
  carry the payload only when the flat scheduled draft token ids exactly match
  the payload's top-1 chain. Rejection sampling is still the flat-chain path.
- This path currently materializes top-k/logprob data to CPU for correctness
  bring-up. It is not the final performance implementation.

### M3.5: Scheduler And GDN Metadata Bridge

Status: complete as a data path; kernels still ignore DDTree parent metadata.

Current local state:

- `DraftTokenIds.ddtree_payloads` carries optional per-request payloads from the
  worker back to the scheduler.
- The scheduler caches payloads per request, clears them on prefill chunks,
  finish/free, preemption, structured-output trimming, and any token mismatch,
  and emits `SchedulerOutput.scheduled_ddtree_payloads` only for the matching
  scheduled speculative row.
- `ddtree_parent_metadata.py` converts payload parent indices into padded
  root-plus-tree parent-id rows aligned with the active batch.
- `SpeculativeConfig.num_speculative_state_tokens()` returns
  `max(num_speculative_tokens, ddtree_budget)` only when DDTree tree verify is
  explicitly enabled. Scheduler lookahead, Mamba speculative blocks,
  `GPUModelRunner.max_spec_state_slots`, and GDN metadata buffers use this
  state-token count.
- `GDNAttentionMetadata` carries `ddtree_parent_ids` and
  `ddtree_num_tree_tokens_cpu`, and the existing GDN diagnostic dumps include
  those tensors.

Remaining gap:

- Full-attention kernels still need ancestor-only tree visibility.
- GDN kernels still need to compute each node from its parent state and commit
  only the accepted path.

### M4: Attention-Only Verifier Correctness

Status: in progress. The vLLM runner now has the main attention-only pieces,
but the small-model forward oracle and Flash-V100 native tree mask are still
pending.

Before Qwen3.6, prove the tree verifier on a small full-attention model:

- Use a Qwen-family small dense target such as `Qwen/Qwen2.5-0.5B-Instruct`.
- Build a flattened tree, ancestor-only mask, compact logits rows, and greedy
  tree sampler.
- Compare one-pass tree verifier logits against per-path replay logits.

For V100, this can be a slower correctness path first. Do not call it a
performance route until Flash-V100 has a native ancestor-only tree verifier.

Pass criteria:

- `budget=1` equals ordinary one-token greedy verification.
- No sibling leakage in attention.
- One-pass tree logits match per-path replay top-1 for all verifier nodes.

Current local state:

- `tree_from_payload()` rebuilds a verifier-coordinate `DDTree` from
  `DDTreeDraftPayload`.
- `ddtree_verify.py` builds `DDTreeVerifierMetadata`, prompt-plus-tree
  attention verifier inputs, and greedy verification results from compact
  root-plus-node logits.
- Tests cover payload tree rebuild, compact-logit full accept, sibling-branch
  accept, and dense ancestor mask sibling hiding.
- `ddtree_sampler.py` greedily walks compact root-plus-node logits and returns
  `SamplerOutput.ddtree_accepted_node_indices` for accepted-path cache/state
  work.
- `FlexAttentionMetadata` can carry DDTree parent ids and wraps its logical
  causal mask with ancestor-only tree visibility.
- `GPUModelRunner` overrides DDTree tree-node position ids after slot mapping,
  so token storage and KV slot mapping stay linear while RoPE positions follow
  node depth.
- For non-hybrid attention-only models, `GPUModelRunner` compacts accepted
  DDTree KV rows from their tree slots back to the normal prefix slots before
  the next step can read cache.

### M5: Qwen3.6-27B-AWQ Safe Hybrid Bridge

Qwen3.6 has full-attention layers plus GDN/Mamba-style recurrent layers. A tree
attention mask alone is not enough:

- Each tree node must read GDN and conv state from its parent.
- Rejected branches must never mutate persistent state.
- Only the accepted path can be committed to normal vLLM KV and recurrent state.

The first bridge should be quality-first:

- Use tree verification for attention logits where supported.
- Commit only accepted path tokens.
- Replay the accepted path through the existing flat/full-GDN path to update
  recurrent state if tree-aware GDN state is not ready.

This may erase speedup, but it is the correctness bridge. If replay makes it
slower than flat DFlash16, keep it experimental and proceed to M6.

Pass criteria:

- Greedy output hash matches no-spec/flat-DFlash semantics for a fixed prompt.
- Long-context multi-turn smoke does not drift.
- Existing full-GDN guard remains the default unless a tree-aware state path is
  proven exact.

Current local state:

- Scheduler tree-row expansion is guarded by exact payload/scheduled-token
  matching and by token/max-len/long-prefill budget checks.
- `DDTreeDraftPayload.is_flat_chain()` now requires token ids, parent ids, and
  node depths to match a true linear chain; token equality alone is not enough.
- The scheduler rejects branched DDTree scheduling for hybrid models because
  accepted-path recurrent state compaction is not implemented yet.
- Flat-chain-equivalent DDTree payloads are still allowed on hybrid models, so
  `ddtree_budget=16, ddtree_top_k=1` remains a safe DFlash16 parity route.
- `GPUModelRunner` has a hybrid fail-fast guard before recurrent-state
  postprocess. If a branched DDTree payload reaches a hybrid model, the step
  raises instead of silently committing GDN/Mamba state from the wrong linear
  slot.
- `GPUModelRunner` now computes DDTree recurrent-state slot selectors from
  `SamplerOutput.ddtree_accepted_node_indices`. For a flat path this matches
  the old accepted-token count; for a branch it points to the last accepted
  compact tree node plus one. The guard still prevents using this on hybrid
  branched payloads until GDN/Mamba kernels consume parent state correctly.
- GDN metadata now has an internal `spec_state_slot_selectors` field. It
  defaults to the old `num_accepted_tokens` selector for flat MTP, but can
  represent the DDTree selected compact tree slot independently of generated
  token count. This is not yet part of the graph op signature.
- `GPUModelRunner` now keeps generated-token count and state-slot selector in
  separate GPU buffers before building GDN metadata. `num_accepted_tokens`
  remains the actual contiguous generated-token count, while
  `spec_state_slot_selectors` comes from
  `SamplerOutput.ddtree_accepted_node_indices` when DDTree metadata is present.
- Qwen GDN spec core now passes `spec_state_slot_selectors` to
  `causal_conv1d_update()` and `fused_recurrent_gated_delta_rule()` when the
  field is present. This selects the compact state slot independently from
  generated-token count, but it still does not compute each tree node from its
  DDTree parent.

### M6: SM70 Tree-Aware GDN And Flash-V100 Tree Verify

This is the real acceleration milestone for 27B-AWQ:

- Add tree parent ids to GDN attention metadata.
- Add scratch state buffers for per-node conv/GDN intermediate state.
- Add selected-path commit into persistent recurrent state.
- Add or adapt a Flash-V100 verifier path that supports ancestor-only tree
  visibility without falling back to a dense eager mask on every round.

Pass criteria:

- DFlash16 DDTree beats flat DFlash16 and no-spec on the same 27B-AWQ TP2
  benchmark, reporting pure decode separately from prefill.
- Token hash parity holds for greedy.
- Official sampling reports acceptance/tree-depth metrics and no sampler
  distribution shortcut is used.

Current local state, 2026-06-21:

- Scope is now narrowed to normal no-spec baseline versus DDTree tree verify.
  Flat DFlash-only speed tests are intentionally skipped because they do not
  answer whether DDTree acceleration is recovered.
- Added a fused DDTree GDN verifier path:
  `causal_conv1d_update_ddtree()` computes each verifier row from its selected
  parent/root conv state, and the fused sigmoid-gating delta-rule update now
  accepts `ddtree_parent_ids`. Qwen GDN pure-spec verification uses this path
  by default under `VLLM_DFLASH_DDTREE_FUSED_GDN=1`.
- Added `vllm/v1/attention/backends/ddtree_branch_triton.py`, a paged-KV
  Triton DDTree ancestor-mask correction kernel for Flash-V100 small-query
  verifier rows. It keeps root/prefix slot 0 visible for root children and
  works with graph capture when parent ids are already CUDA int32.
- `flash_attn_v100.py` now routes DDTree tree verifier rows through
  `prefill_ddtree_triton` before the old dense fallback. The dense fallback is
  still present only as an eager correctness bridge.
- `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE=1` is required for full hybrid
  tree-state runs. Without it, the scheduler rejects branched payloads and the
  run falls back to flat speculative rows; those results are invalid DDTree
  tree-verify evidence.
- Graph capture is now proven for the 27B-AWQ DDTree tree verifier by using
  `VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE=1 + ddtree_budget`.

Validated artifacts:

- Normal no-spec graph baseline:
  `bench_results/ddtree_smoke_20260621/27b_awq_nospec_flashv100_graph_i32_o16_w2r3.json`.
  It used Flash-V100 no-compile FULL_DECODE_ONLY graph, no speculative config,
  `input_len=32`, `output_len=16`, `warmup=2`, `repeat=3`, TP2, AWQ, and
  produced stable hash
  `d7e8a039cc9363c9d968bb2b1c0deea9cf6a53fecca39a7fdfdc853a869eaf7f`.
  Mean steady decode was `48.6173 tok/s`.
- Old full DDTree tree verify with dense attention fallback:
  `27b_awq_dflash_ddtree32_topk4_nochain_flashv100_i32_o16.json`, eager,
  `ddtree_budget=32`, `top_k=4`, `chain_seed=false`, spec tokens `16`.
  It produced the same hash, `draft_tokens_per_step=32`, `num_drafts=5`,
  `num_accepted_tokens=11`, acceptance length `3.2`, and mean steady decode
  `4.8949 tok/s`.
- Fused-GDN plus dense attention fallback:
  `27b_awq_dflash_ddtree32_topk4_nochain_fusedgdn_tree_eager_i32_o16.json`.
  Same hash and spec metrics, route summary included `prefill_ddtree_dense=160`,
  and mean steady decode improved to `8.8475 tok/s`.
- Fused-GDN plus Triton branch attention, eager:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_fusedgdn_tree_eager_i32_o16.json`.
  Same hash and spec metrics, route summary used `prefill_ddtree_triton=160`
  with no dense DDTree route, and mean steady decode improved to
  `11.4074 tok/s`.
- Fused-GDN plus Triton branch attention, no-compile FULL_DECODE_ONLY graph:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_fusedgdn_tree_graph_i32_o16_w2r3.json`.
  Capture size was `33`; graph capture completed; route summary included
  `prefill_capture_smallq=32` and `prefill_ddtree_triton=400`; all repeats
  matched the no-spec hash. Mean steady decode was `12.0542 tok/s`.
- Re-running after removing the Flash-V100 parent GPU-to-CPU branch probe did
  not show a stable speed win:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_nocpuprobe_fusedgdn_tree_graph_i32_o16_w2r3.json`
  measured `11.7603 tok/s`, with identical hash/spec metrics. Keep the code
  simplification, but do not count it as a speed improvement.
- AEON's recommended first shape adjusted to the user's spec-16 requirement
  (`ddtree_budget=22`, `top_k=8`, `chain_seed=true`, capture size `23`) was
  worse on this prompt:
  `27b_awq_dflash_ddtree22_topk8_chain_tritonattn_fusedgdn_tree_graph_i32_o16_w2r3.json`
  measured `9.5538 tok/s`, acceptance length `2.8333`, and `num_drafts=6`.
  It is not the current best local DDTree setting.
- `ddtree_budget=32`, `top_k=8`, `chain_seed=true` did not produce evidence:
  startup ended with engine initialization failure and process exit `139`
  before route/sampler metrics were available.

Interpretation:

- Kernel and graph blockers are no longer the main reason DDTree is slow on
  this short 27B-AWQ smoke. The best valid DDTree graph run is still only
  `12.0542 tok/s` versus the no-spec graph baseline at `48.6173 tok/s`.
- The immediate limiter is acceptance/proposal quality and per-step DDTree
  overhead: the current best tree accepts only `11` target tokens over `5`
  drafts for 16 output tokens, so each accepted token still pays a large
  drafter plus verifier cost.
- `build_ddtree_payloads_from_logits()` still synchronizes top-k logits and
  flat draft ids to CPU and builds the tree with Python heap logic. This should
  be profiled or moved to a graph-safe GPU/semigraph path before long-output
  speed claims.

## 2026-06-21 Acceptance Debug

- Added DDTree debug payload fields for per-depth DFlash top-k token ids and
  logprobs, and sampler logging that reports each greedy verifier step as
  `(compact row, parent node, depth, target argmax, tree child, top-k rank)`.
  This is gated by `VLLM_DFLASH_DDTREE_DEBUG=1`.
- Added `VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT=1` default-on. After a
  non-flat branch is accepted, the worker now copies accepted tree hidden-state
  rows, auxiliary hidden rows, and `input_ids` back into the committed flat
  spine before the next DFlash draft. This mirrors the accepted-attention-KV
  compaction and fixes a real DDTree/DFlash context mismatch.
- Direct tests cover the new context compaction behavior. `pytest` is not
  installed in `1cat-vllm-1.2.0-releasecheck`, so the test functions were run
  directly with `python -c`.
- Short debug run:
  `27b_awq_dflash_ddtree32_topk4_nochain_compactctx_debug_graph_i32_o16.json`.
  It still measured low acceptance length (`3.0`) and steady decode
  `11.2420 tok/s`, but now explains why: later verifier rejections were mostly
  because the target argmax was not in the current DFlash `top_k=4`, or was in
  the per-depth top-k but the budgeted tree did not allocate that parent/child.
  This points to candidate/tree coverage, not a compact-logit sampler bug.
- Reference comparison:
  official DDTree takes `topk = min(budget, vocab)`, while AEON's deployable
  Qwen3.6 profile uses `DDTREE_TOP_K=8`. The local default was raised from `4`
  to `8`; explicit benchmark configs still override it.
- Follow-up `top_k=8/32` validation was blocked by a separate TP4 OpenAI API
  server occupying all four V100s:
  `python -m vllm.entrypoints.openai.api_server ... --tensor-parallel-size 4`.
  Do not treat the failed `top_k=8/32` startup artifacts as DDTree evidence.

## Benchmark Gate

The first accepted 27B-AWQ DDTree benchmark must match the existing DFlash16
criterion:

```text
CUDA_VISIBLE_DEVICES=2,3
model=/home/ymzx/models/Qwen3.6-27B-AWQ
draft=/home/ymzx/models/Qwen3.6-27B-DFlash-FP16
TP=2
input_len=4096
output_len=256
max_model_len=8192
max_num_batched_tokens=8192
max_num_seqs=1
attention_backend=FLASH_ATTN_V100
mamba_cache_mode=align
num_speculative_tokens=16
```

Metrics to report:

- steady decode tok/s and output tok/s.
- prefill time and decode time.
- token hash for greedy.
- DDTree budget, verified nodes, accepted depth, sibling-branch hit count.
- tree build time, target verify time, GDN replay/commit time.
- flat DFlash16 and no-spec baselines from the same run group.

## Immediate Next Step

Recover DDTree speed by attacking acceptance and proposal overhead, not DFlash
flat benchmarking:

- Add DDTree timing instrumentation for tree payload build, drafter forward,
  target verifier forward, sampler, and accepted-state commit.
- Replace or bypass the Python/CPU tree builder in
  `build_ddtree_payloads_from_logits()` for the hot path.
- Sweep only DDTree tree-verify settings that are motivated by the reference
  plan (`budget=32/40/48`, `top_k=4/8`, chain seed on/off), using the existing
  no-spec graph baseline as the comparison.
- Keep hash parity and route summary (`prefill_ddtree_triton`, no
  `prefill_ddtree_dense`) as required acceptance checks.
