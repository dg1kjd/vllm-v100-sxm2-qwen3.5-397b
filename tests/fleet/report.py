"""Markdown summary for a fleet run."""

from __future__ import annotations


def _check(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_model(model_name: str, kv_tokens: int | None, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"### {model_name}")
    if kv_tokens is not None:
        lines.append(f"- KV cache size: {kv_tokens:,} tokens")
    lines.append("")
    lines.append("| Suite | Result | Wall (s) | Details |")
    lines.append("|---|---|---|---|")
    for r in results:
        details = _format_details(r)
        lines.append(
            f"| {r['name']} | {_check(r['passed'])} | "
            f"{r['elapsed_s']:.1f} | {details} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_details(r: dict) -> str:
    if r["error"]:
        return f"ERROR: {r['error']}"
    d = r["details"]
    if r["name"] == "smoke":
        text = (d.get("text") or "").replace("|", "\\|").replace("\n", " ")[:50]
        return f"n_out={d.get('n_out')} text={text!r}"
    if r["name"].startswith("perf_"):
        tokps = d.get("tokps", 0)
        base = d.get("baseline_tokps", 0)
        if d.get("mode") == "record_only":
            return (
                f"tok/s={tokps:.2f} (record-only, no baseline) "
                f"n_out={d.get('n_out')} finish={d.get('finish_reason')}"
            )
        floor = d.get("floor_tokps", 0)
        delta_pct = (tokps - base) / base * 100 if base > 0 else 0
        sign = "+" if delta_pct >= 0 else ""
        return (
            f"tok/s={tokps:.2f} ({sign}{delta_pct:.1f}% vs baseline {base:.1f}, "
            f"floor {floor:.1f}) n_out={d.get('n_out')} finish={d.get('finish_reason')}"
        )
    if r["name"] == "nul_scan":
        return (
            f"turns={d.get('n_turns')} tokens={d.get('tokens')} "
            f"nul_bytes={d.get('nul_bytes')} other_ctrl={d.get('other_ctrl_bytes')}"
        )
    if r["name"] == "reasoning_split":
        c = d.get("checks", {})
        return (
            f"no_kwargs_sep={c.get('no_kwargs_separated')} "
            f"explicit_true_sep={c.get('explicit_true_separated')} "
            f"explicit_false_optout={c.get('explicit_false_optout')}"
        )
    if r["name"] == "flat_scan":
        return f"scanned={d.get('scanned_tokens')} flat_rows={d.get('flat_rows')}"
    if r["name"] == "tool_roundtrip":
        return (
            f"tool_called={d.get('tool_called')} args_ok={d.get('args_ok')} "
            f"city={d.get('city')!r} followup_uses_result={d.get('followup_uses_result')}"
        )
    if r["name"] == "needle":
        full_key = next((k for k in d if k.startswith("full_")), "full_misses")
        return (
            f"control_2k_misses={len(d.get('control_2k_misses', []))} "
            f"{full_key}={len(d.get(full_key, []))} of {d.get('n_needles')}"
        )
    if r["name"] == "concurrency":
        return (
            f"failures={len(d.get('failures', []))} "
            f"p2_prefill_ttft={d.get('phase2_longprefill_ttft_s')}s "
            f"p3_cache_ttft={d.get('phase3_repeat_ttft_s')}s"
        )
    return str(d)


def format_overall(per_model: list[tuple[str, list[dict]]]) -> str:
    total = sum(len(rs) for _, rs in per_model)
    passed = sum(1 for _, rs in per_model for r in rs if r["passed"])
    headline = (
        f"# Fleet regression summary\n\n"
        f"**{passed}/{total} suites passed**\n"
    )
    return headline
