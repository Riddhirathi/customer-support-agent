"""Eval harness: runs the real pipeline over eval/cases.jsonl and writes a
metrics summary plus failed-case detail to eval/results.md.

Run from the repo root as:
    python -m eval.run
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.agent import answer
from src.config import get_config
from src.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = REPO_ROOT / "eval" / "cases.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "results.md"
DELAY_BETWEEN_CALLS_SECONDS = 2.0  # be polite to the free-tier rate limit


def _load_cases() -> list[dict]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_last_trace_lines(config, count: int) -> list[dict]:
    """Read the last `count` lines from the trace log, in order. Assumes no
    other process wrote traces concurrently during this eval run."""
    if not config.logs_path.exists():
        return []
    lines = config.logs_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-count:]]


def _run_cases(config) -> tuple[list[dict], list[dict]]:
    """Run every case through the real agent pipeline. Returns
    (results, traces) aligned by index with the input cases."""
    cases = _load_cases()
    results = []
    for i, case in enumerate(cases):
        result = answer(case["question"])
        results.append(result.model_dump())
        print(f"  [{i + 1}/{len(cases)}] {case['question'][:60]!r} -> {result.action} ({result.confidence})")
        if i < len(cases) - 1:
            time.sleep(DELAY_BETWEEN_CALLS_SECONDS)
    traces = _read_last_trace_lines(config, len(cases))
    return results, traces


def _retrieval_hit_at_3(retriever: Retriever, cases: list[dict]) -> tuple[float, list[str]]:
    """Fraction of cases (with a non-null expected_section) whose expected
    section appears in the top-3 retrieved sections for that question."""
    scored_cases = [c for c in cases if c["expected_section"]]
    if not scored_cases:
        return 0.0, []
    misses = []
    hits = 0
    for case in scored_cases:
        top3 = [r["section_id"] for r in retriever.retrieve(case["question"], top_k=3)]
        if case["expected_section"] in top3:
            hits += 1
        else:
            misses.append(case["question"])
    return hits / len(scored_cases), misses


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), pct))


def compute_metrics(cases: list[dict], results: list[dict], traces: list[dict], retriever: Retriever) -> dict:
    """Compute every metric the spec asks for from the case/result/trace data."""
    n = len(cases)

    routing_correct = sum(1 for c, r in zip(cases, results) if r["action"] == c["expected_action"])
    routing_accuracy = routing_correct / n

    should_escalate = [(c, r) for c, r in zip(cases, results) if c["expected_action"] == "escalate"]
    false_responds = [c for c, r in should_escalate if r["action"] == "respond"]
    false_respond_rate = (len(false_responds) / len(should_escalate)) if should_escalate else 0.0

    hit_at_3, retrieval_misses = _retrieval_hit_at_3(retriever, cases)

    category_scored = [(c, r) for c, r in zip(cases, results) if c["expected_category"]]
    category_correct = sum(1 for c, r in category_scored if r["category"] == c["expected_category"])
    category_accuracy = (category_correct / len(category_scored)) if category_scored else 0.0

    responded = [(c, r, t) for c, r, t in zip(cases, results, traces) if r["action"] == "respond"]
    grounded = sum(1 for c, r, t in responded if t["checks"].get("citation_ok") and t["checks"].get("numbers_ok"))
    groundedness_rate = (grounded / len(responded)) if responded else 0.0

    retried = sum(1 for t in traces if t["checks"].get("llm_json_retried"))
    json_retry_rate = retried / n

    latencies = [t["total_latency_ms"] for t in traces]

    return {
        "n_cases": n,
        "routing_accuracy": routing_accuracy,
        "false_respond_rate": false_respond_rate,
        "false_respond_count": len(false_responds),
        "should_escalate_count": len(should_escalate),
        "retrieval_hit_at_3": hit_at_3,
        "retrieval_misses": retrieval_misses,
        "category_accuracy": category_accuracy,
        "groundedness_rate": groundedness_rate,
        "responded_count": len(responded),
        "json_retry_rate": json_retry_rate,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }


def _failed_cases(cases: list[dict], results: list[dict], traces: list[dict]) -> list[dict]:
    failed = []
    for case, result, trace in zip(cases, results, traces):
        if result["action"] != case["expected_action"]:
            failed.append(
                {
                    "question": case["question"],
                    "expected_action": case["expected_action"],
                    "actual_action": result["action"],
                    "reason": trace.get("reason"),
                }
            )
    return failed


def _write_results_md(metrics: dict, failed: list[dict]) -> None:
    lines = [
        "# Eval Results",
        "",
        f"Ran {metrics['n_cases']} cases from `eval/cases.jsonl` against the live pipeline.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Routing accuracy | {metrics['routing_accuracy']:.0%} |",
        f"| **False-respond rate** (answered when it should have escalated) | "
        f"{metrics['false_respond_rate']:.0%} ({metrics['false_respond_count']}/{metrics['should_escalate_count']}) |",
        f"| Retrieval hit@3 | {metrics['retrieval_hit_at_3']:.0%} |",
        f"| Category accuracy | {metrics['category_accuracy']:.0%} |",
        f"| Groundedness rate (of responded cases) | {metrics['groundedness_rate']:.0%} "
        f"({metrics['responded_count']} responded) |",
        f"| JSON retry rate | {metrics['json_retry_rate']:.0%} |",
        f"| Latency p50 | {metrics['latency_p50_ms']:.0f} ms |",
        f"| Latency p95 | {metrics['latency_p95_ms']:.0f} ms |",
        "",
    ]

    if metrics["retrieval_misses"]:
        lines.append("Retrieval hit@3 misses:")
        for q in metrics["retrieval_misses"]:
            lines.append(f"- {q}")
        lines.append("")

    lines.append("## Failed cases (routing mismatch)")
    lines.append("")
    if not failed:
        lines.append("None — every case routed to the expected action.")
    else:
        for f in failed:
            lines.append(
                f"- **{f['question']}** — expected `{f['expected_action']}`, got "
                f"`{f['actual_action']}` (reason: `{f['reason']}`)"
            )
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = get_config()
    cases = _load_cases()
    print(f"Running {len(cases)} eval cases against the live pipeline...")
    results, traces = _run_cases(config)

    retriever = Retriever(config.policies_dir, config.retrieval)
    metrics = compute_metrics(cases, results, traces, retriever)
    failed = _failed_cases(cases, results, traces)

    _write_results_md(metrics, failed)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Routing accuracy: {metrics['routing_accuracy']:.0%}")
    print(f"False-respond rate: {metrics['false_respond_rate']:.0%}")


if __name__ == "__main__":
    main()
