"""Config for the slim qwen397b regression runner.

Unlike the original multi-model fleet runner, this targets the single
already-running systemd service (``vllm-397b.service`` on
``http://127.0.0.1:11435``). The systemd unit owns the engine lifecycle, so
there is no launch script / ready timeout here — the runner only addresses the
live endpoint over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PERF_FLOOR = 0.85  # measured tok/s must be >= floor * baseline to pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    served_id: str
    base_url: str = "http://127.0.0.1:11435"
    baselines_tokps: dict[str, float] = field(default_factory=dict)


# Served by the systemd unit:
#   --served-model-name Rio-3.5-Open-397B-AWQ Qwen3.5-397b  (we address the latter)
MODEL = ModelConfig(
    name="qwen397b",
    served_id="Qwen3.5-397b",
    base_url="http://127.0.0.1:11435",
    baselines_tokps={
        # Record-only (0.0 sentinel) until re-baselined on THIS build. The old
        # fleet baseline (perf_t1=50.05 tok/s, 2026-06-14) predates the v1.2.1
        # restack, the perf-to-10pct multi-stream work, and the :11435 systemd
        # service — it is not comparable. Run perf_t1, record the number, then
        # set it here to enable the 0.85x floor assertion.
        "perf_t1": 0.0,
    },
)

# Fast acceptance set (default). needle + concurrency are slow (104k prefill /
# multi-stream) -> opt in explicitly via `--suites`.
DEFAULT_SUITES = (
    "smoke",
    "perf_t1",
    "nul_scan",
    "reasoning_split",
    "flat_scan",
    "tool_roundtrip",
)
SLOW_SUITES = ("needle", "concurrency")
