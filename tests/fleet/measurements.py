"""Per-suite measurement functions (slim qwen397b / systemd :11435 edition).

Salvaged from the original multi-model fleet harness. Dropped: pi_toolcall
(needs pi-coding-agent), perf_t2/perf_t3 (long-context perf corpus). Kept the
V100-specific regression suites that still describe real failure modes.

Each suite returns:
    {"name": str, "passed": bool, "elapsed_s": float, "details": dict,
     "error": str | None}

Floor-style perf assertion: measured >= 0.85 * baseline (registry.PERF_FLOOR).
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request

from tests.fleet.registry import PERF_FLOOR, ModelConfig

BASE_URL = "http://127.0.0.1:11435"


def _post_json(url: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


# ---------- smoke ----------

def smoke(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Trivial 10-token completion — engine liveness."""
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        body = _post_json(
            f"{base_url}/v1/completions",
            {
                "model": model.served_id,
                "prompt": "Hello, world",
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout_s=60,
        )
        n_out = body["usage"]["completion_tokens"]
        text = body["choices"][0]["text"]
        details = {"n_out": n_out, "text": text[:80]}
        passed = n_out >= 1
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "smoke", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- perf ----------

def _decode_tokps(usage: dict, wall_s: float, prefill_subtract_s: float) -> float:
    n_out = usage.get("completion_tokens", 0)
    denom = max(wall_s - prefill_subtract_s, 1e-3)
    return n_out / denom


def _run_chat(base_url: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    return _post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout_s=600,
    )


def perf_t1(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Short prompt, 256-token decode. Record-only until baseline is set.

    NOTE: with the server-default thinking_token_budget, a portion of the
    decoded tokens may be reasoning. The tok/s number is still a consistent
    throughput probe; treat absolute value as build-specific.
    """
    baseline = model.baselines_tokps.get("perf_t1", 0.0)
    prefill_subtract_s = 0.5
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        body = _run_chat(
            base_url, model.served_id,
            "Explain photosynthesis in three short paragraphs.", 256,
        )
        wall = time.perf_counter() - t0
        usage = body.get("usage", {})
        finish = body["choices"][0].get("finish_reason")
        tokps = _decode_tokps(usage, wall, prefill_subtract_s)
        details = {
            "n_prompt": usage.get("prompt_tokens", 0),
            "n_out": usage.get("completion_tokens", 0),
            "wall_s": round(wall, 2),
            "tokps": round(tokps, 2),
            "baseline_tokps": baseline,
            "floor_tokps": round(PERF_FLOOR * baseline, 2) if baseline > 0 else 0.0,
            "finish_reason": finish,
        }
        produced_output = usage.get("completion_tokens", 0) > 0
        if baseline > 0:
            passed = produced_output and tokps >= PERF_FLOOR * baseline
        else:
            passed = produced_output
            details["mode"] = "record_only"
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "perf_t1", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- nul-byte scan ----------

# 4-turn polyfact-style decode (~4-6K tokens each) builds context depth to
# mirror the regime where the fp16 last-layer AllReduce overflow originally
# fired (NaN logits sampled to token id 0 -> 0x00 bytes via byte-fallback).
_NUL_SCAN_TURNS: tuple[str, ...] = (
    "Write a complete Python module implementing polynomial arithmetic over "
    "Z[x]: a Polynomial class with __init__, __add__, __sub__, __mul__, "
    "__divmod__, __mod__, __floordiv__, __eq__, __repr__, degree, "
    "leading_coeff, is_zero. 100+ lines, docstrings included, raw code -- no "
    "markdown wrapping, no bullet-point todo lists.",
    "Now write a complete gcd.py: Euclidean GCD over Z[x] using "
    "pseudo-remainder, extended Euclidean algorithm, content (GCD of "
    "coefficients), primitive_part. 80+ lines.",
    "Now write a complete finite_field.py: polynomial arithmetic mod p, "
    "multiplicative inverses mod p, polynomial GCD mod p, division mod p. "
    "100+ lines.",
    "Now write a complete factor.py: top-level factor() combining content "
    "extraction, square-free factorization, finite-field factorization, "
    "brute-force factor recombination. 100+ lines.",
)


def nul_scan(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Multi-turn decode + NUL-byte scan of all assistant output. Targets the
    fp16 last-layer AllReduce overflow class (NaN logits -> token id 0 -> 0x00).

    Pass = all turns return >=1 token AND zero 0x00 bytes across the transcript.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    transcript_bytes = b""
    n_turns = 0
    n_total_tokens = 0
    try:
        messages: list[dict] = []
        for prompt in _NUL_SCAN_TURNS:
            messages.append({"role": "user", "content": prompt})
            body = _post_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model.served_id,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    # Thinking off: bloats tokens without adding decode surface
                    # for this bug class.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout_s=600,
            )
            msg = body["choices"][0]["message"]
            parts: list[str] = []
            for k in ("content", "reasoning", "reasoning_content"):
                v = msg.get(k)
                if v:
                    parts.append(v)
            for tc in msg.get("tool_calls") or []:
                args = ((tc or {}).get("function") or {}).get("arguments")
                if args:
                    parts.append(args)
            assistant_text = "".join(parts)
            messages.append({"role": "assistant", "content": assistant_text})
            transcript_bytes += assistant_text.encode("utf-8", errors="surrogateescape")
            n_turns += 1
            n_total_tokens += body.get("usage", {}).get("completion_tokens", 0)

        nul_count = transcript_bytes.count(b"\x00")
        ctrl_count = sum(
            1 for b in transcript_bytes
            if b < 0x20 and b not in (0x09, 0x0A, 0x0D)
        )
        details = {
            "n_turns": n_turns,
            "bytes": len(transcript_bytes),
            "tokens": n_total_tokens,
            "nul_bytes": nul_count,
            "other_ctrl_bytes": ctrl_count,
        }
        passed = n_turns == len(_NUL_SCAN_TURNS) and nul_count == 0
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
        details = {
            "n_turns": n_turns,
            "bytes": len(transcript_bytes),
            "nul_bytes": transcript_bytes.count(b"\x00"),
        }
    return {
        "name": "nul_scan", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- reasoning / content separation ----------

# On a server started with --default-chat-template-kwargs '{"enable_thinking":
# true}', a no-kwargs request inherits that default, so the qwen3 parser sees
# the prompt's pre-opened <think> block and routes reasoning into
# reasoning_content instead of leaking it into content. The load-bearing case is
# `no_kwargs`: it proves the server default actually reaches the parser.
_REASONING_PROMPT = "What is 17 * 23? Reason step by step, then give the final number."
_REASONING_VARIANTS = (
    ("no_kwargs", None),
    ("explicit_true", {"enable_thinking": True}),
    ("explicit_false", {"enable_thinking": False}),
)


def _reasoning_call(base_url: str, model_id: str, ctk: dict | None) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": _REASONING_PROMPT}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    if ctk is not None:
        payload["chat_template_kwargs"] = ctk
    body = _post_json(f"{base_url}/v1/chat/completions", payload, timeout_s=300)
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    reasoning = reasoning or ""
    return {
        "reasoning_len": len(reasoning),
        "content_len": len(content),
        "leaked": ("<think>" in content) or ("</think>" in content),
        "finish": body["choices"][0].get("finish_reason"),
    }


def reasoning_split(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Reasoning text must land in reasoning_content, not leak into content.

    Pass criteria:
      * no_kwargs      -> SEPARATED (reasoning_content populated, no tags in
                          content). Proves the server enable_thinking default
                          reaches the parser.
      * explicit_true  -> SEPARATED.
      * explicit_false -> opt-out: content populated, no tags leaked.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        per = {
            name: _reasoning_call(base_url, model.served_id, ctk)
            for name, ctk in _REASONING_VARIANTS
        }

        def separated(v: dict) -> bool:
            return (not v["leaked"]) and v["reasoning_len"] > 0

        ef = per["explicit_false"]
        checks = {
            "no_kwargs_separated": separated(per["no_kwargs"]),
            "explicit_true_separated": separated(per["explicit_true"]),
            "explicit_false_optout": ef["content_len"] > 0 and not ef["leaked"],
        }
        details = {**per, "checks": checks}
        passed = all(checks.values())
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "reasoning_split", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- flat-logits scan ----------

# A chosen-token logprob below the threshold under temperature=0 means the model
# was near-indifferent at that step -- the signature of the shared-expert
# sigmoid-gate / fp16-cast ordering bug. Distinct from nul_scan (NaN->token-0
# byte corruption); this catches near-uniform logits.
_FLAT_SCAN_PROMPTS = (
    "def quicksort(arr):\n",
    "The history of the Roman Empire begins",
    "import numpy as np\n\ndef softmax(x):\n",
    "Once upon a time in a small village",
    "## Chapter 1: Introduction to Thermodynamics\n\n",
    "SELECT u.name, COUNT(o.id) FROM users u",
)
_FLAT_LOGPROB_THRESHOLD = -9.0


def flat_scan(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Greedy-decode several short prompts and flag near-uniform ("flat") logits.

    Pass = zero token logprobs below _FLAT_LOGPROB_THRESHOLD across all prompts.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    total = 0
    flats: list[dict] = []
    try:
        for p in _FLAT_SCAN_PROMPTS:
            body = _post_json(
                f"{base_url}/v1/completions",
                {
                    "model": model.served_id, "prompt": p,
                    "max_tokens": 400, "temperature": 0.0, "logprobs": 0,
                },
                timeout_s=600,
            )
            lp = body["choices"][0]["logprobs"]
            for i, (tok, l) in enumerate(zip(lp["tokens"], lp["token_logprobs"])):
                if l is not None and l < _FLAT_LOGPROB_THRESHOLD:
                    flats.append({"prompt": p[:25], "pos": i, "logprob": round(l, 2)})
            total += len(lp["tokens"])
        details = {
            "scanned_tokens": total, "flat_rows": len(flats), "samples": flats[:5],
        }
        passed = len(flats) == 0
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "flat_scan", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- tool-call round-trip ----------

_TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def tool_roundtrip(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Tool-call round-trip: model emits get_weather(city=Paris), we feed a tool
    result back, the model's final answer uses it (temp 7, light rain).

    Pass = exactly one correct tool_call AND the follow-up uses the result.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        msgs = [{"role": "user",
                 "content": "What is the weather in Paris right now? Use the tool."}]
        body = _post_json(
            f"{base_url}/v1/chat/completions",
            {"model": model.served_id, "messages": msgs, "tools": _TOOL_DEF,
             "chat_template_kwargs": {"enable_thinking": True},
             "temperature": 0.2, "max_tokens": 4096},
            timeout_s=600,
        )
        m = body["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        call_ok = len(tcs) == 1
        args_ok = False
        city = None
        if tcs:
            fn = tcs[0]["function"]
            try:
                args = json.loads(fn["arguments"])
                city = args.get("city")
                args_ok = fn["name"] == "get_weather" and "paris" in str(city or "").lower()
            except (json.JSONDecodeError, TypeError, KeyError):
                args_ok = False
        followup_ok = False
        if call_ok and args_ok:
            msgs2 = msgs + [
                {"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tcs[0]["id"],
                 "content": json.dumps(
                     {"city": "Paris", "temp_c": 7, "condition": "light rain"})},
            ]
            body2 = _post_json(
                f"{base_url}/v1/chat/completions",
                {"model": model.served_id, "messages": msgs2, "tools": _TOOL_DEF,
                 "chat_template_kwargs": {"enable_thinking": True},
                 "temperature": 0.2, "max_tokens": 4096},
                timeout_s=600,
            )
            content2 = body2["choices"][0]["message"].get("content") or ""
            followup_ok = ("7" in content2) and ("rain" in content2.lower())
            details["final_head"] = content2[:120].replace("\n", " ")
        details.update({
            "tool_called": call_ok, "args_ok": args_ok, "city": city,
            "followup_uses_result": followup_ok,
        })
        passed = call_ok and args_ok and followup_ok
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "tool_roundtrip", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- needle-in-a-haystack ----------

# 5 codes at depths 2/25/50/75/98%. Q1 pays the cold prefill; Q2-Q5 reuse the
# prefix cache, so this also gates cached long-range GDN-state quality. Slow
# (cold ~104k prefill) -> opt-in.
_NEEDLES = (
    ("NIGHTOWL", "7d3f-90b2"),
    ("CINNABAR", "k4q8-11zx"),
    ("MOONDIAL", "p9w2-55mh"),
    ("TIDEPOOL", "c6r1-83vd"),
    ("FOXGLOVE", "n2t7-46je"),
)
_NEEDLE_DEPTHS = (0.02, 0.25, 0.50, 0.75, 0.98)
_NEEDLE_SUBJECTS = (
    "the archive service", "node {n}", "the billing pipeline", "rack {n}",
    "the export daemon", "tenant {n}", "the audit log", "shard {n}",
    "the metrics relay", "queue {n}",
)
_NEEDLE_VERBS = (
    "rotates its credentials", "compacts its segments", "syncs snapshots",
    "rebalances partitions", "flushes its write-ahead log",
    "re-indexes documents", "prunes stale entries", "renews its lease",
)
_NEEDLE_TAILS = (
    "every {n} minutes", "at {n}:00 UTC", "after {n} requests",
    "when usage exceeds {n} percent", "on day {n} of each month",
)
# 104k stays inside the 140k max-model-len with room for the question + answer.
_NEEDLE_LONG_TOKENS = 104000


def _needle_filler(rng: random.Random) -> str:
    s = rng.choice(_NEEDLE_SUBJECTS).format(n=rng.randint(1, 97))
    v = rng.choice(_NEEDLE_VERBS)
    t = rng.choice(_NEEDLE_TAILS).format(n=rng.randint(2, 59))
    return f"{s.capitalize()} {v} {t}. "


def _count_tokens(base_url: str, model_id: str, text: str) -> int:
    body = _post_json(
        f"{base_url}/tokenize", {"model": model_id, "prompt": text}, timeout_s=120)
    return body["count"]


def _build_haystack(base_url: str, model_id: str, target_tokens: int,
                    rng: random.Random) -> str:
    probe = [_needle_filler(rng) for _ in range(200)]
    per_sent = _count_tokens(base_url, model_id, "".join(probe)) / 200
    n_sent = int(target_tokens / per_sent)
    sentences = [_needle_filler(rng) for _ in range(n_sent)]
    while _count_tokens(base_url, model_id, "".join(sentences)) > target_tokens:
        del sentences[len(sentences) // 2 :: 50]
    for (name, code), depth in zip(_NEEDLES, _NEEDLE_DEPTHS):
        pos = min(len(sentences) - 1, int(len(sentences) * depth))
        sentences.insert(pos, f"The vault access code for project {name} is {code}. ")
    return "".join(sentences)


def _needle_misses(base_url: str, model_id: str, target_tokens: int,
                   rng_seed: int = 110) -> list[dict]:
    rng = random.Random(rng_seed)
    haystack = _build_haystack(base_url, model_id, target_tokens, rng)
    misses: list[dict] = []
    for (name, code), depth in zip(_NEEDLES, _NEEDLE_DEPTHS):
        q = (f"{haystack}\n\nWhat is the vault access code for project {name}? "
             "Reply with the code only.")
        body = _post_json(
            f"{base_url}/v1/chat/completions",
            {"model": model_id, "messages": [{"role": "user", "content": q}],
             "max_tokens": 1024, "temperature": 0.0},
            timeout_s=900,
        )
        msg = body["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + " " + (msg.get("content") or "")
        if code not in text:
            misses.append({"needle": name, "depth": depth})
    return misses


def needle(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """Needle recall: ~2k control then ~104k full-context. Pass = every code
    string appears in its answer (reasoning text included)."""
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        control = _needle_misses(base_url, model.served_id, 2000)
        full = _needle_misses(base_url, model.served_id, _NEEDLE_LONG_TOKENS)
        details = {
            "n_needles": len(_NEEDLES),
            "control_2k_misses": control,
            f"full_{_NEEDLE_LONG_TOKENS // 1000}k_misses": full,
        }
        passed = not control and not full
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "needle", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- concurrency ----------

_CONC_PARA = (
    "Service {i} exposes endpoint /api/v{j} returning record {k}; its retry "
    "budget is {j} attempts with exponential backoff and a {k} ms deadline. "
)


def _conc_make_prompt(target_tokens: int, salt: str) -> str:
    n = max(1, target_tokens // 40)
    body = "".join(
        _CONC_PARA.format(i=i, j=i % 7 + 1, k=i * 13 % 503) for i in range(n))
    return f"[session {salt}] " + body + "\nDescribe the overall retry policy briefly."


def _conc_chat(base_url: str, model_id: str, prompt: str, max_tokens: int,
               out: dict, key: str) -> None:
    payload = {
        "model": model_id, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        # Scheduling/throughput probe, not a reasoning test: force thinking off
        # so the small token budgets aren't burned on a <think> block (ttft).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    ntok = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if (delta.get("content") or delta.get("reasoning")
                            or delta.get("reasoning_content")):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        ntok += 1
        total = time.monotonic() - t0
        n_out = usage["completion_tokens"] if usage else ntok
        out[key] = {
            "ttft": ttft, "total": total, "completion_tokens": n_out,
            "prompt_tokens": usage["prompt_tokens"] if usage else -1,
            "decode_tok_s": round(n_out / max(total - (ttft or 0), 1e-3), 1),
        }
    except Exception as e:  # noqa: BLE001 — per-stream failure, recorded not raised
        out[key] = {"error": repr(e)}


def concurrency(model: ModelConfig, base_url: str = BASE_URL) -> dict:
    """4-way concurrency + anti-starvation + prefix-cache-under-load.

    Phases (fail any -> suite fails):
      1. 4 concurrent ~3k-token sessions (150 tok out) all complete.
      2. a ~25k prefill submitted while 3 streams decode is admitted and
         finishes (TTFT <= 120s).
      3. a re-sent phase-1 prompt hits the prefix cache (TTFT <= 2s).
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    failures: list[str] = []
    mid = model.served_id
    try:
        res: dict = {}
        prompts = [_conc_make_prompt(3000, f"s{i}") for i in range(4)]
        threads = [
            threading.Thread(target=_conc_chat,
                             args=(base_url, mid, p, 150, res, f"s{i}"))
            for i, p in enumerate(prompts)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failures += [f"phase1 {k}: {res[k]['error']}"
                     for k in sorted(res) if "error" in res[k]]

        res2: dict = {}
        dthreads = [
            threading.Thread(target=_conc_chat,
                             args=(base_url, mid, _conc_make_prompt(800, f"d{i}"),
                                   300, res2, f"d{i}"))
            for i in range(3)
        ]
        for t in dthreads:
            t.start()
        time.sleep(1.0)
        _conc_chat(base_url, mid, _conc_make_prompt(25000, "long"), 32,
                   res2, "longprefill")
        for t in dthreads:
            t.join()
        failures += [f"phase2 {k}: {res2[k]['error']}"
                     for k in sorted(res2) if "error" in res2[k]]
        lp = res2.get("longprefill", {})
        lp_ttft = lp.get("ttft")
        if "error" not in lp:
            if lp_ttft is None:
                failures.append("phase2: long prefill produced no tokens")
            elif lp_ttft > 120:
                failures.append(f"phase2: long prefill TTFT {lp_ttft:.1f}s (starved?)")

        res3: dict = {}
        _conc_chat(base_url, mid, prompts[0], 32, res3, "repeat")
        rep = res3["repeat"]
        rep_ttft = rep.get("ttft")
        if "error" in rep:
            failures.append(f"phase3: {rep['error']}")
        elif rep_ttft is None:
            failures.append("phase3: repeat produced no tokens")
        elif rep_ttft > 2.0:
            failures.append(f"phase3: cache-hit TTFT {rep_ttft:.2f}s, expected < 2s")

        details = {
            "phase1_decode_tokps": {
                k: res[k].get("decode_tok_s") for k in sorted(res) if "error" not in res[k]
            },
            "phase2_longprefill_ttft_s": (round(lp["ttft"], 1)
                                          if isinstance(lp.get("ttft"), (int, float))
                                          else None),
            "phase3_repeat_ttft_s": (round(rep["ttft"], 2)
                                     if isinstance(rep.get("ttft"), (int, float))
                                     else None),
            "failures": failures,
        }
        passed = not failures
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "concurrency", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


SUITE_FUNCS = {
    "smoke": smoke,
    "perf_t1": perf_t1,
    "nul_scan": nul_scan,
    "reasoning_split": reasoning_split,
    "flat_scan": flat_scan,
    "tool_roundtrip": tool_roundtrip,
    "needle": needle,
    "concurrency": concurrency,
}
