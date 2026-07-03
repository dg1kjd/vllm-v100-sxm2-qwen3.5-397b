# SwiReasoning §5.P5 Pareto eval — results log

Driver: `wo_swir_pareto.py` (endpoint A/B; both arms temp=0, identical fixed
`max_tokens` budget, explicit `thinking_token_budget=-1`; same extractor both
arms so grading bias cancels). OFF = production graph server (:11435).
ON = eager swir server (`serve_rio_swir_eager_v121.sh`, :8000), swir cfg:
alpha_0=0.5, beta_0=0.7, window=512, math_ids from SwiReasoning repo.
Model: Rio-3.5-Open-397B-AWQ, TP8, 8x V100.

## GSM8K, n=100 (seed-0 sample of test), budget 4096 — 2026-07-02

| | OFF (plain CoT) | ON (SwiReasoning) |
|---|---|---|
| accuracy | 0.920 (8 miss) | **0.930 (7 miss)** |
| tokens mean/med | 350 / 259 | **192 / 178** (−45% / −31%) |
| budget-capped | 0 | 0 |
| nul/replacement | 0 | 0 |

Head-to-head: 6 missed by both; ON fixed {265, 1035}; ON broke {590}.
Accuracy delta = noise at n=100; token savings systematic.

**§5.P5 acceptance: PASS** — accuracy ≥ baseline + token-efficiency gain
(gate 1); ON flat-logits 0 (gate 2); memory flat 31858–32008 MiB GPU0 over
full ON run (gate 3); text-only-when-off fleet 8/8 (gate 4, 2026-07-01).

ON-arm wall ~28 s/problem (eager, sequential); OFF ~7 s (graphs, 4-way).
Raw: results_off_n100.jsonl / results_on_n100.jsonl, mem_on_arm.csv.

## AIME 2024+2025, n=60, budget 8192 — 2026-07-02

Both arms on the PRODUCTION graph server (:11435) — the ON arm ran under the
Tier-1 graph bypass (commit fb639bfd61) at concurrency 6, no server swap.

| | OFF (plain CoT) | ON (SwiReasoning) |
|---|---|---|
| accuracy | 0.783 (13 miss) | 0.783 (13 miss) |
| tokens mean/med | 4963 / 5197 | **3206 / 4236** (−35% / −18%) |
| budget-capped | **19** | **0** |
| nul/replacement | 0 | 2* |

Head-to-head: 9 missed by both; ON fixed 4, broke 4 (exact accuracy wash).
ON's compression **eliminated all 19 budget-cappings**; of those 19 capped
problems ON completed and solved 7 (4 of which OFF had missed).
*Both NUL-bearers are CORRECT solutions (idx 4/5, one replacement char each in
~5k-token outputs) — lone odd token at an injection splice, cosmetic; not
flat-logits corruption.

**§5.P5 acceptance on AIME: PASS** — accuracy tie at −35% tokens under a
binding budget (19/60 OFF runs hit the cap; the paper's fixed-budget regime).

## Tier-1 graph bypass validation (GSM8K ON n=20 vs eager reference)

Stable (20/20, 0 errors/NULs), accuracy identical (0.950 both), Pareto holds
vs OFF on the same problems (326 vs 434 tok). Known property: FSM trajectories
diverge from the eager server via kernel numerics — 7/20 problems >1.5×
eager-arm tokens, one 5.4× outlier (bound with max_switch_count if needed).
Swir requests still decode eager per-step; batching (concurrency 6) is the
practical throughput lever. Full-speed swir decode = Tier-2 (embeds-entry
graph capture), tracked as task #30.
Raw: results_aime_all_{off,on}_n60.jsonl, results_on_t1_n20.jsonl.

## Full GSM8K test set, n=1319, budget 4096 — 2026-07-03 (Tier-2, production graphs)

| | OFF (c4) | ON (Tier-2, c6) |
|---|---|---|
| accuracy | 0.9613 (51 miss) | **0.9636** (48 miss; 7 fixed / 4 broken) |
| tokens mean/med | 353 / 244 | 340 / 241 (**−3.5%**) |
| nul / errors | 0 / 0 | 0 / 0 |

**Key finding — token compression is numerics-fragile:** same problems, same knobs
(alpha0 0.5, window 512): eager server −45% tokens → Tier-1 (eager steps on graph
server) −25% → Tier-2 (full-graph replay) −3.5%. Accuracy is robust in every regime
(under graph-kernel numerics the FSM barely fires, so outputs converge to plain CoT).
Implication: for token-efficiency wins use the eager regime (or retune alpha/beta for
graph numerics — untested); for accuracy-neutral, no-swap, graph-speed serving of
swir requests, Tier-2 is safe. ON wall > OFF wall here (24.0 vs 15.8 s/prob): the
per-step controller + logits-clone tax is paid while the FSM earns no shorter output.
Raw: results_{off,on}_full.jsonl. Tier-2 commit 879ef52091 (per-step eager kept only
for token-id-path configs; VLLM_SWIR_TIER1=1 forces the Tier-1 behavior).

## FINAL VERDICT (2026-07-03, confirm runs) — the savings were the bug

alpha_0=1.0 confirm: AIME rep2 0.833 (rep1 0.817; OFF avg@3 0.822) — accuracy fully
restored, collapse cell empty both reps. GSM8K n=200 sampled: 93.5% BOTH arms
(12/13 misses shared), tokens **+3.2%** — NO savings at reference knobs.

Conclusion: the paper's token-efficiency claim does not reproduce at safe knobs on
this stack; the −45%/−35% compression was substantially the alpha_0=0.5 premature-
closure artifact (efficiency-looking on easy problems, wrong answers on hard ones).
SwiReasoning on this stack = accuracy-neutral, token-neutral, per-step overhead;
experimental mechanism only. Published: fork README fixed (example alpha_0 1.0 +
honest Measured section, commit d74db7f694). Full mechanism analysis:
REPORT_on_miss_analysis.md (+ addendum). Raw: results_aime_alpha1{,b}_on_n60.jsonl,
results_gsm_{fair_off,alpha1_on}_n200.jsonl.
