# SwiReasoning evaluation notes (8x V100, Rio/Qwen3.5-397B AWQ TP8)

Complete evaluation trail for this fork's SwiReasoning implementation, including the
negative result: **the paper's token-efficiency claim did not reproduce at safe knobs
on this stack** — the apparent savings traced to `alpha_0=0.5` causing stochastic
premature closure of the thinking block (accuracy-invisible on easy problems, fatal
on hard ones). See the main README's SwiReasoning section for usage.

- `RESULTS.md` — all benchmark runs in order (GSM8K n=100/n=1319, AIME'24+25 n=60,
  eager vs Tier-1 vs Tier-2 execution regimes, sampled "fair" reps, the alpha/beta
  one-knob A/B, final verdict).
- `REPORT_on_miss_analysis.md` — the miss-pattern analysis that localized the
  collapse cell (high-token-demand + shortened generation: 11% vs 72% baseline) and
  the subsequent A/B that overturned its own mechanism ranking (alpha, not beta).
- `wo_swir_pareto.py` — the endpoint A/B driver (self-contained; stdlib +
  transformers tokenizer). Datasets are NOT included: GSM8K test set is fetched as
  plain JSONL from the openai/grade-school-math repo; AIME 2024/2025 problem texts
  are MAA-copyrighted, fetch via the HF datasets-server first-rows API from
  HuggingFaceH4/aime_2024 and yentinglin/aime_2025 and normalize to
  {"question", "answer": "#### <int>"}.
- `results_*.jsonl` — raw per-problem outcomes (idx/correct/tokens/finish_reason/
  pred/gold only; no problem texts). File naming: `aime_fair*` = temp 0.6 budget
  30720; `aime_all*` = greedy budget 8192; `*_full` = GSM8K all 1319; `t1/t2` =
  Tier-1/Tier-2 execution-regime validation; `alpha1/beta1` = one-knob A/B.

Paths inside the documents refer to the original rig; treat them as provenance,
not instructions.
