# SwiReasoning-ON miss analysis — AIME60, sampled ("fair") runs

Date: 2026-07-03. Status: miss-set + mechanism analysis COMPLETE from existing data; live trace capture BLOCKED in the analysis session (permission rules not honored for subagents — Appendix C). Basis: 1 complete ON sampled run vs 2 complete OFF sampled runs (fair3_off in-flight, 29/60 rows, folded in where available). Authored by the Fable analysis agent; transcribed to file by the main session (subagent file-writes were policy-blocked).

## 1. Executive summary

SwiReasoning-ON under sampling (temp 0.6 / top_p 0.95 / top_k 20, budget 30720, `max_switch_count: None`) loses ~5-7pp vs OFF (75.0% vs 80.0/83.3%) for exactly one reason: **stochastic premature closure of the thinking block on high-token-demand problems.**

- Low-demand (32 problems): ON 32/32, OFF 64/64 — no damage, full token savings.
- High-demand (greedy-OFF ≥6k tok), ON not collapsed (19): ON 63% vs OFF 55% — ON better; uniquely rescues idx 42, 57.
- High-demand, ON collapsed (<0.7× OFF mean) (9): **ON 11% vs OFF 72%** — the entire deficit (Fisher p≈0.014).
- OFF never finishes a high-demand problem under ~3.8k tokens; ON produced nine 1.4–2.4k finishes there. Bimodal, not a shifted distribution.
- Collapse is stochastic: greedy-ON solved 3 of the 5 pure regressions (idx 4, 27, 40) at full length; sampled-ON collapsed the same problems to 1.6–2.0k and missed.
- Forced convergence/termination structurally impossible: `max_switch_count: None` ⇒ `use_switch=False` (`swir_controller.py:201`). Verified.
- Garbling ruled out: all ON misses have `nul==0`; the only nonzero-nul rows (idx 16, 19 sampled; 4, 5 greedy) are correct.
- Wrong answers look like answers from an unfinished derivation (121 vs 110, 478 vs 699, 792 vs 588, 647 vs 259), occasionally derailed (−475, 12221 — non-AIME-range).

## 2. Data inventory

| file | arm | sampling | budget | acc | notes |
|---|---|---|---|---|---|
| results_aime_all_off_n60.jsonl | OFF | greedy | 8192 | 78.3% | ALL 13 misses length-capped |
| results_aime_all_on_n60.jsonl | ON | greedy | 8192 | 78.3% | 0 caps; all misses organic stop |
| results_aime_fair_off_n60.jsonl | OFF | t0.6 | 30720 | 80.0% | 12 misses |
| results_aime_fair2_off_n60.jsonl | OFF | t0.6 | 30720 | 83.3% | 10 misses |
| results_aime_fair3_off_n60.jsonl | OFF | t0.6 | 30720 | in-flight (29/60) | queue running (pgrep confirmed) |
| results_aime_fairon1_on_n60.jsonl | ON | t0.6 | 30720 | 75.0% | 15 misses; median 2.7k tok vs OFF ~4.6k |
| results_aime_fairon2_on_n60.jsonl | ON | t0.6 | 30720 | not yet created | queued |

Greedy@8192 was confounded: greedy-OFF lost all 13 misses to the cap, which ON's compression partly rescued — that masking produced the 78.3/78.3 tie; the uncapped fair runs expose the ON regression.

## 3. Miss-set analysis (sampled)

ON misses (15): idx 1, 2, 3, 4, 21, 27, 31, 38, 40, 43, 44, 49, 52, 58, 59.
- Shared hard tail (both OFF runs also miss): 2, 3, 21, 43, 44, 59 — not ON-specific. ON additionally solves hard-tail idx 42, 57.
- **Pure ON regressions** (both OFF solve): **1, 4, 27, 31, 40.**
- Mixed (OFF 1/2): 38, 49, 52, 58.

Pure regressions — 5/5 collapsed AND 5/5 high-demand (ratio = ON/mean(OFF1,OFF2); demand = greedy-OFF tokens, L = capped):

| idx | gold | ON tok (pred) | OFF1/OFF2 tok | ratio | demand | greedy-ON |
|---|---|---|---|---|---|---|
| 1 | 113 | 3412 (164) | 5131/5014 | 0.67 | 8192L miss | 4890 miss (12221) |
| 4 | 110 | 1598 (121) | 4658/4859 | 0.34 | 8192L hit | 5011 hit |
| 27 | 699 | 1760 (478) | 3926/5151 | 0.39 | 6014 hit | 4279 hit |
| 31 | 588 | 2401 (792) | 4769/4916 | 0.50 | 8192L miss | 1695 miss (690) |
| 40 | 259 | 2013 (647) | 5044/4788 | 0.41 | 8192L miss | 4961 hit |

Supporting cells: collapsed+high-demand extras idx 43 (1554, miss — same wrong answer 63 as OFF in ⅓ the tokens), 58 (2020 miss; greedy-ON 1622, pred −475), 59 (1416 miss), 56 (3314, ratio 0.68, the lone collapsed hit). Collapsed but low-demand = harmless and where the Pareto win lives: idx 41 (0.27, hit), 50 (0.29, hit), 47 (0.42, hit), 20 (0.47, hit), etc. Aggregate 2×3 table as in the summary. Eliminating the collapse cell recovers ~4.7 problems ⇒ ~83%, fully closing the gap. Budget checks: ON max 5179/30720 tokens (17%), zero caps, all `stop`.

## 4. Mechanism (code-verified; trace confirmation pending)

`/home/nvidia/work/1Cat-vLLM/vllm/v1/sample/swir_controller.py`:
1. **beta-blend** (`:267-274`): on every soft→normal transition, `next_input = beta*soft_emb + (1−beta)*E[</think>]`, beta = 0.7 + 0.3·step/30720 ≈ 0.70–0.72 ⇒ ~28–30% `</think>` embedding injected each switch. Only code path writing the "close thinking" direction into the input.
2. **Ungated to_normal** (`:224`): fires the moment entropy dips below the freshly-reset reference — no dwell (dwell gates only normal→soft, `:227-231`) ⇒ roughly one nudge per ~window(512) steps; collapsed lengths (1.4–2.4k) ≈ 3–4 window periods.
3. **Latch** (`:206-208`): once `</think>` is sampled, `locked_normal` is permanent — each nudge is a one-way ratchet under temp-0.6 noise (greedy shows rarer collapse: idx 49→970 tok, 58→1622, 31→1695).
4. **alpha-blend nonstandard** (`:261-265` + driver): driver sets `alpha_0=0.5` (reference default 1.0 = no blend) ⇒ every soft entry feeds 50% `E[<think>]` — an extra perturbation that may manufacture the entropy dips triggering the next beta-nudge.

Alternative hypothesis (latent thinking is legitimately denser; collapses are honest early convergence) is disfavored — greedy-ON solves the same problems at full length, and OFF never emits short finishes on them — but is cleanly separable via the beta_0=1.0 A/B.

## 5. Trace evidence — BLOCKED, staged

Capture plan (33 requests, concurrency 2 while queue runs): idx 1/4/27/31/40 → 3×ON+1×OFF; idx 43/58/59 → 2×ON; controls 41/50 (collapsed-correct) → 2×ON; 29/13 (full-length correct) → 1×ON. Body = driver-exact: temp 0.6/top_p 0.95/top_k 20/max_tokens 30720/thinking_token_budget −1, ON adds `vllm_xargs.swireasoning = build_swir_args(30720, 0.5, 512)`. Store full response JSON (reasoning+content+usage) as `trace_analysis/trace_idx{N}_{on|off}_rep{K}.json`. Read for: reasoning ending mid-derivation with abrupt final-answer jump (beta-closure signature); pred derived vs guessed; style-shift count (soft-mode readout spans); U+FFFD near transitions (expected none). For direct proof, add a debug knob in `vllm/v1/worker/swir_glue.py` logging (step, to_soft, to_normal, sampled==end_think_id) — hypothesis predicts `</think>` sampling clusters within tens of steps after to_normal events.

## 6. Ranked mitigations

1. **beta_0 0.7 → 1.0 A/B, then tune ~0.85–0.9** — 1-knob; disables the `</think>` blend; if collapse vanishes, mechanism proven. Cost: some easy-problem token savings.
2. **alpha_0 0.5 → 1.0** (reference default) — removes nonstandard `<think>` re-injection; may cut switch rate.
3. **Entropy-gate the beta-blend** (~5 lines in `SwirController.step`) — nudge only when entropy is genuinely low ("converged").
4. **Min-think floor** (glue): mask `end_think_id` logits while step < ~1500–3000; blunt but guaranteed; pairs with 3.
5. **window_size 512 → 1024/2048** — fewer nudges, less latent thinking.
6. **Operational retry**: ON completion <2500 tok ⇒ re-issue with OFF (≈15% retry rate; no code change).
7. Keep `max_switch_count: None` (don't add a second truncation mechanism).

Next experiments: (a) fairon @ beta_0=1.0; (b) fairon @ alpha_0=1.0; (c) let queued fairon2 finish to firm up n=1 ON stats.

## Appendix C — subagent permission failures (for the record)

All allowlisted Bash forms (venv python, curl to :11435, tail, ls) and Write into
/home/nvidia/models/swir_pareto/ were denied for the analysis subagent despite matching
`.claude/settings.local.json` allow rules created mid-session (only `pgrep` passed); report
file writes additionally blocked by harness policy (subagents return findings as text).
Likely cause: settings loaded at session start; `.claude/` did not exist then. Fix for
future agent runs: restart the Claude session so the rules load, then run the Appendix-B
capture plan.

---

## ADDENDUM 2026-07-03 — A/B results OVERTURN the mitigation ranking

One-knob isolation runs (n=60 AIME fair conditions each) vs OFF avg\@3 = 0.822:

| arm | acc | collapsed-HD cell | cell acc |
|---|---|---|---|
| ON default (fairon1/2) | 0.750 / 0.650 | 9 / 12 | 0.11 / 0.00 |
| **beta_0=1.0** (§6 rank 1) | 0.667 | **10 — still collapses** | 0.20 |
| **alpha_0=1.0** (§6 rank 2) | **0.817** | **0 — eliminated** | — |

**Verdict: the alpha-blend was the primary mechanism, not the beta-blend.** The driver's
nonstandard `alpha_0=0.5` (50% E[<think>] injected on every soft entry; reference default
1.0 = no blend) drove the stochastic premature closures. Reverting to 1.0 restores
baseline accuracy and empties the collapse cell; beta_0=1.0 does not help.

**Corollary:** alpha1's tokens ≈ OFF (3644 vs ~3700) — the AIME "token savings" vanished
with the collapse, i.e. the compression and the accuracy bug were largely the SAME
premature-closure mechanism (harmless-looking on low-demand problems, fatal on
high-demand ones). Whether legitimate savings survive correct knobs is being measured:
GSM8K n=200 alpha1-ON vs fair-OFF (queued as run_alpha1_confirm.sh) + AIME alpha1 rep2.
Driver default flipped to alpha0=1.0.
