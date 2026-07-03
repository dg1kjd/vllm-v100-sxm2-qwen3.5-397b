"""SwiReasoning §5.P5 Pareto eval driver — GSM8K OFF-vs-ON over the HTTP endpoint.

Self-contained (stdlib + transformers tokenizer + SwiReasoning.get_math_symbols_ids):
no datasets/math_verify deps, so the production venv stays untouched. Both arms use
temperature=0, an identical fixed token budget (--budget) and explicit
thinking_token_budget=-1 (production defaults it to 4096 server-side; the eager
server doesn't — pin it so the arms are comparable). Gold/pred extraction is the
same code for both arms, so extractor bias cancels in the A/B.

  OFF arm (production, :11435):  python wo_swir_pareto.py --mode off --n 100 --concurrency 4
  ON  arm (eager server, :8000): python wo_swir_pareto.py --mode on  --n 100 --url http://127.0.0.1:8000

Writes JSONL per problem (resume-safe: reruns skip already-answered idx) and a
summary line. Coherence gate: counts NUL/replacement chars per response.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

DATA = Path(__file__).parent / "gsm8k_test.jsonl"
PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
MODEL_DIR = "/home/nvidia/models/Rio-3.5-Open-397B-AWQ"
SERVED = "Rio-3.5-Open-397B-AWQ"

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(s: str):
    s = s.strip().strip("$").replace(",", "").rstrip(".")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def extract_gold(answer_field: str):
    m = answer_field.rsplit("####", 1)
    return _to_number(m[1]) if len(m) == 2 else None


def extract_pred(text: str):
    """Last \\boxed{...} if present, else the last number in the text."""
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    for cand in reversed(boxes):
        m = _NUM.search(cand)
        if m:
            n = _to_number(m.group(0))
            if n is not None:
                return n
    nums = _NUM.findall(text)
    for cand in reversed(nums):
        n = _to_number(cand)
        if n is not None:
            return n
    return None


def build_swir_args(budget: int, alpha_0: float, window: int, beta_0: float = 0.7) -> dict:
    from transformers import AutoTokenizer

    sys.path.insert(0, "/home/nvidia/work/SwiReasoning")
    from generation_utils import get_math_symbols_ids

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return {
        "think_id": tok.convert_tokens_to_ids("<think>"),
        "end_think_id": tok.convert_tokens_to_ids("</think>"),
        "line_break_id": tok.convert_tokens_to_ids("\\n"),
        "eos_token_id": tok.eos_token_id,
        "convergence_ids": tok.encode("</think>", add_special_tokens=False),
        "termination_ids": tok.encode(
            "</think>\n\nThe final answer is", add_special_tokens=False
        ),
        "alpha_0": alpha_0,
        "beta_0": beta_0,
        "window_size": window,
        "max_switch_count": None,
        "termination_max_tokens": 32,
        "max_new_tokens": budget,
        "math_ids": sorted(get_math_symbols_ids(tok)),
    }


def call(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "on"], required=True)
    ap.add_argument("--data", default=str(DATA), help="jsonl with question/answer('#### gold')")
    ap.add_argument("--tag", default="", help="output filename tag (defaults to data stem)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--beta0", type=float, default=0.7)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data_path = Path(args.data)
    tag = args.tag or (
        "" if data_path == DATA else data_path.stem.replace("_test", "") + "_"
    )
    out_path = Path(
        args.out or f"{Path(__file__).parent}/results_{tag}{args.mode}_n{args.n}.jsonl"
    )
    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                pass

    rows = [json.loads(l) for l in data_path.read_text().splitlines()]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    picked = order[: args.n]

    swir = build_swir_args(args.budget, args.alpha0, args.window, args.beta0) if args.mode == "on" else None
    lock = threading.Lock()
    todo = [i for i in picked if i not in done]
    print(f"[pareto] mode={args.mode} n={args.n} todo={len(todo)} out={out_path}")

    def work(idx: int) -> None:
        q = rows[idx]["question"]
        gold = extract_gold(rows[idx]["answer"])
        body = {
            "model": SERVED,
            "temperature": args.temp,
            "max_tokens": args.budget,
            "thinking_token_budget": -1,
            "messages": [{"role": "user", "content": q + PROMPT_SUFFIX}],
        }
        if args.top_p is not None:
            body["top_p"] = args.top_p
        if args.top_k is not None:
            body["top_k"] = args.top_k
        if swir is not None:
            body["vllm_xargs"] = {"swireasoning": swir}
        t0 = time.time()
        try:
            resp = call(args.url, body, args.timeout)
            msg = resp["choices"][0]["message"]
            text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
            rec = {
                "idx": idx,
                "correct": bool(
                    gold is not None
                    and extract_pred(text) is not None
                    and extract_pred(text) == gold
                ),
                "completion_tokens": resp["usage"]["completion_tokens"],
                "finish_reason": resp["choices"][0]["finish_reason"],
                "nul": text.count("\x00") + text.count("�"),
                "wall_s": round(time.time() - t0, 1),
                "pred": str(extract_pred(text)),
                "gold": str(gold),
            }
        except Exception as e:  # noqa: BLE001 — record and continue
            rec = {"idx": idx, "error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 1)}
        with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[pareto] idx={idx} {rec.get('correct', 'ERR')} "
                  f"tok={rec.get('completion_tokens', '-')} {rec.get('wall_s')}s", flush=True)

    threads: list[threading.Thread] = []
    for i in todo:
        while len([t for t in threads if t.is_alive()]) >= args.concurrency:
            time.sleep(0.5)
        t = threading.Thread(target=work, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # summary over the picked set
    recs = {}
    for line in out_path.read_text().splitlines():
        try:
            r = json.loads(line)
            recs[r["idx"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    got = [recs[i] for i in picked if i in recs and "correct" in recs[i]]
    errs = [recs[i] for i in picked if i in recs and "error" in recs[i]]
    if got:
        acc = sum(r["correct"] for r in got) / len(got)
        toks = sorted(r["completion_tokens"] for r in got)
        nuls = sum(r["nul"] for r in got)
        capped = sum(r["finish_reason"] == "length" for r in got)
        print(f"\n[pareto] ===== SUMMARY mode={args.mode} =====")
        print(f"  graded          : {len(got)}/{args.n} (errors: {len(errs)})")
        print(f"  accuracy        : {acc:.3f}")
        print(f"  tokens mean/med : {sum(toks)/len(toks):.0f} / {toks[len(toks)//2]}")
        print(f"  budget-capped   : {capped}")
        print(f"  nul/replacement : {nuls}")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
