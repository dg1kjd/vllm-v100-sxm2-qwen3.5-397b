"""Slim regression runner for the systemd vllm-397b service.

Unlike the original multi-model fleet runner, this does NOT launch or tear down
the engine — the systemd unit (``vllm-397b.service``) owns the lifecycle. It
preflights that the OpenAI endpoint is reachable and serving the expected model,
runs the selected suites against it, and prints (and optionally writes) a
markdown pass/fail summary.

Usage:
    python -m tests.fleet.run                              # default fast suites
    python -m tests.fleet.run --suites smoke,reasoning_split
    python -m tests.fleet.run --suites all                 # incl. slow needle/concurrency
    python -m tests.fleet.run --base-url http://127.0.0.1:11435
    python -m tests.fleet.run --report /tmp/fleet.md

Exit code: 0 if all selected suites passed, 1 if any failed, 2 for
argument/preflight errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import replace

from tests.fleet import report
from tests.fleet.measurements import SUITE_FUNCS
from tests.fleet.registry import DEFAULT_SUITES, MODEL, SLOW_SUITES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--suites", default="",
        help="Comma-separated suite names, or 'all' for default+slow "
             "(default: the fast acceptance set)",
    )
    p.add_argument(
        "--base-url", default="",
        help=f"OpenAI base URL (default: {MODEL.base_url})",
    )
    p.add_argument(
        "--report", default="",
        help="Write markdown summary to this file (also always printed to stdout)",
    )
    return p.parse_args()


def preflight(base_url: str, served_id: str) -> str | None:
    """Return an error string if the endpoint isn't reachable / serving the model."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return (f"endpoint {base_url} not reachable: {type(e).__name__}: {e}\n"
                f"Is the service up?  systemctl status vllm-397b")
    ids = [m.get("id") for m in data.get("data", [])]
    if served_id not in ids:
        return f"served_id {served_id!r} not advertised by {base_url}; got {ids}"
    return None


def resolve_suites(spec: str) -> list[str] | str:
    spec = spec.strip().lower()
    if spec == "all":
        names = list(DEFAULT_SUITES) + list(SLOW_SUITES)
    elif spec:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = list(DEFAULT_SUITES)
    unknown = [n for n in names if n not in SUITE_FUNCS]
    if unknown:
        return (f"unknown suite(s): {', '.join(unknown)}. "
                f"available: {', '.join(SUITE_FUNCS)}")
    return names


def main() -> int:
    args = parse_args()
    base_url = args.base_url.strip() or MODEL.base_url
    model = replace(MODEL, base_url=base_url)

    names = resolve_suites(args.suites)
    if isinstance(names, str):
        print(f"[fleet] {names}", file=sys.stderr)
        return 2

    pf = preflight(base_url, model.served_id)
    if pf is not None:
        print(f"[fleet] preflight failed: {pf}", file=sys.stderr)
        return 2

    print(f"[fleet] {model.served_id} @ {base_url} — suites: {', '.join(names)}",
          file=sys.stderr)
    results: list[dict] = []
    for name in names:
        print(f"[fleet] running {name} ...", file=sys.stderr, flush=True)
        r = SUITE_FUNCS[name](model, base_url)
        tag = "PASS" if r["passed"] else "FAIL"
        extra = f" error={r['error']}" if r["error"] else ""
        print(f"[fleet]   {name}: {tag} ({r['elapsed_s']:.1f}s){extra}",
              file=sys.stderr, flush=True)
        results.append(r)

    md = (report.format_overall([(model.name, results)]) + "\n"
          + report.format_model(model.name, None, results))
    print(md)
    if args.report.strip():
        with open(args.report.strip(), "w") as f:
            f.write(md + "\n")
        print(f"[fleet] wrote {args.report.strip()}", file=sys.stderr)

    n_fail = sum(1 for r in results if not r["passed"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
