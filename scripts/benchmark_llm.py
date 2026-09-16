#!/usr/bin/env python3
"""Measure the latency of the slow-loop endpoint from the machine that serves it.

The paper reports the cost of the slow loop as a distribution, never as a single
number, so this script produces the distribution. It sends the real telemetry
prompt that the client builds, records wall-clock latency per call, extracts the
token usage when the server reports it, and writes both a JSON report and a
markdown summary.

Usage
-----
    python scripts/benchmark_llm.py --base-url http://127.0.0.1:11434/v1 --model qwen3.5:4b --calls 10
    python scripts/benchmark_llm.py --base-url http://127.0.0.1:8080/v1 --model qwen3.5-4b-q4_k_m \
        --out results/llm_latency_llamacpp.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

REPRESENTATIVE_WINDOW: Dict[str, Any] = {
    "n_window": 10,
    "step": 40,
    "energy_series": [0.9193, 0.9275, 0.8955, 0.9089, 0.891, 0.8721, 0.8675, 0.8524, 0.8471, 0.8299],
    "energy": {"first": 0.9193, "last": 0.8299, "min": 0.8299, "mean": 0.8811, "std": 0.03091, "slope_per_step": -0.01046},
    "improvement": 0.08946,
    "grad_norm": {"last": 0.3821, "mean": 1.134, "max": 2.041},
    "eta": {"a_k_last": 0.1013, "c_k_last": 0.0871, "eta_scale": 1.0},
    "theta": {"var": 0.3142, "mean_abs": 0.5117, "max_abs": 1.284, "dim": 12},
}


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile without a SciPy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure(
    base_url: str,
    model: str,
    calls: int = 10,
    api_key: str = "",
    timeout: float = 120.0,
    session: Any = None,
    window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send ``calls`` decisions and return the latency distribution."""
    import requests

    from code.llm_client import LLMConfig, build_request_payload, parse_decision

    session = requests.Session() if session is None else session
    cfg = LLMConfig(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
    payload = build_request_payload(window or REPRESENTATIVE_WINDOW, cfg)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    latencies: List[float] = []
    completion_tokens: List[float] = []
    failures = 0
    for _ in range(int(calls)):
        started = time.perf_counter()
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            parse_decision(raw)
            latencies.append(elapsed)
            usage = data.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens.append(float(usage["completion_tokens"]))
        except Exception:  # noqa: BLE001 - failures are part of the reported cost
            failures += 1

    if not latencies:
        return {"model": model, "base_url": base_url, "calls": int(calls), "failures": failures, "ok": False}

    report: Dict[str, Any] = {
        "ok": True,
        "model": model,
        "base_url": base_url,
        "calls": int(calls),
        "failures": failures,
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
    }
    if completion_tokens:
        mean_tokens = statistics.fmean(completion_tokens)
        report["completion_tokens_mean"] = mean_tokens
        report["tokens_per_s_mean"] = mean_tokens / report["latency_s"]["mean"]
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    """Small markdown table for the paper notes."""
    if not report.get("ok"):
        return f"| {report['model']} | {report['calls']} | every call failed |\n"
    latency = report["latency_s"]
    tokens = report.get("tokens_per_s_mean")
    token_text = "n/a" if tokens is None else f"{tokens:.1f}"
    return (
        "| Metric | Value |\n|---|---|\n"
        f"| Model | {report['model']} |\n"
        f"| Endpoint | {report['base_url']} |\n"
        f"| Calls | {report['calls']} (failures: {report['failures']}) |\n"
        f"| Mean latency (s) | {latency['mean']:.3f} |\n"
        f"| Median latency (s) | {latency['median']:.3f} |\n"
        f"| p95 latency (s) | {latency['p95']:.3f} |\n"
        f"| Min / max (s) | {latency['min']:.3f} / {latency['max']:.3f} |\n"
        f"| Completion tokens per second | {token_text} |\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the latency of the slow-loop endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default="results/llm_latency.json")
    args = parser.parse_args(argv)

    report = measure(
        args.base_url,
        args.model,
        calls=args.calls,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(to_markdown(report), encoding="utf-8")

    print(to_markdown(report))
    print(f"[benchmark_llm] wrote {out_path} and {markdown_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
