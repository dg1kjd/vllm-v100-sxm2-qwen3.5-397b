# Fleet regression runner (slim / systemd edition)

End-to-end "is this build still good?" check for the **qwen397b** service. This
is the slimmed descendant of the original multi-model fleet harness, retargeted
at the single **systemd** service (`vllm-397b.service` on
`http://127.0.0.1:11435`).

Key difference from the original: **it does not launch or tear down the engine.**
The systemd unit owns the lifecycle; the runner just addresses the already-running
endpoint. (Dropped for the same reason: the per-machine `~/launch_*.sh` contract,
the multi-model registry, and the `pi-coding-agent` dependency — `tool_roundtrip`
replaces `pi_toolcall`.)

## Usage

```bash
# Fast acceptance set (default)
python -m tests.fleet.run

# A subset
python -m tests.fleet.run --suites smoke,reasoning_split,flat_scan

# Everything, including the slow long-context / concurrency suites
python -m tests.fleet.run --suites all

# Point at a different host/port; write the summary to a file
python -m tests.fleet.run --base-url http://127.0.0.1:11435 --report /tmp/fleet.md
```

Exit code: `0` all selected suites passed, `1` any failed, `2` arg/preflight error.
Preflight hits `/v1/models` and checks `Qwen3.5-397b` is advertised.

## What gets checked

| Suite | What | Pass criteria | In default set |
|---|---|---|---|
| smoke | `/v1/completions`, 10 tok | ≥1 token | ✓ |
| perf_t1 | short prompt, 256-tok decode | record-only until baselined; then ≥ 0.85×baseline | ✓ |
| nul_scan | 4-turn polyfact decode, scan for `0x00` (fp16 AllReduce overflow class) | all turns return tokens, zero NUL bytes | ✓ |
| reasoning_split | 3 `chat_template_kwargs` variants — reasoning must land in `reasoning_content`, not leak into `content` | no-kwargs & explicit-true SEPARATED; explicit-false opts out cleanly | ✓ |
| flat_scan | greedy 6 prompts, flag chosen-token logprob < −9 (shared-expert sigmoid-gate / fp16-cast bug) | zero flat rows | ✓ |
| tool_roundtrip | OpenAI tool-call round-trip (`get_weather(Paris)`, feed result back) | one correct tool_call; final answer uses the result | ✓ |
| needle | needle-in-haystack, ~2k control + ~104k full ctx, 5 codes at depths 2/25/50/75/98% | every code appears in its answer | slow — opt-in |
| concurrency | 4-way + 25k-prefill anti-starvation + prefix-cache-under-load | all streams complete; prefill TTFT ≤ 120s; cache-hit TTFT ≤ 2s | slow — opt-in |

## Baselines

`perf_t1` is **record-only** (`baselines_tokps["perf_t1"] = 0.0` in `registry.py`)
until re-baselined on the current build — the old fleet number (50.05 tok/s,
2026-06-14) predates the v1.2.1 restack, the perf-to-10pct work, and this systemd
service, so it is not comparable. To enable the floor assertion: run `perf_t1`,
note the tok/s, set `baselines_tokps["perf_t1"]` to it.

## Notes for this build

- The server default `thinking_token_budget=4096` is active. The
  reasoning-heavy suites give the model room (they use `max_tokens` ≥ 512) and
  scan `reasoning_content` where relevant (`nul_scan`, `needle`).
- `reasoning_split` is the suite most likely to surface a live discrepancy —
  if `reasoning_content` comes back empty despite thinking being on, it will
  FAIL `no_kwargs_separated` / `explicit_true_separated`, which is the signal to
  investigate the qwen3 parser's reasoning surfacing.
- Preflight refuses nothing GPU-wise (systemd owns the process); if the endpoint
  is down, start it: `sudo systemctl start vllm-397b`.
