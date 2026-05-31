"""
src/evaluation_engine/pipeline.py
───────────────────────────────────
LangGraph evaluation pipeline — the 5-node DAG.

Graph topology:
    retrieve → generate → ground → [router] → judge → log
                                       ↓
                                  judge_skipped → log

The router after 'ground' is the key engineering decision:
  - If grounding PASSES: proceed to LLM judge (costs money, worth it)
  - If grounding FAILS: skip to judge_skipped (saves judge API cost,
    marks result as hallucinated deterministically)

This short-circuit is the OSFI E-23 alignment point: deterministic
validation gates probabilistic validation. Auditable, traceable, cheap.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import numpy as np
import structlog
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.models import (
    BootstrappedMetric,
    Domain,
    ModelConfig,
    RunSummary,
    TestCase,
)
from src.scorers.bootstrap import bootstrap_metric, is_regression

from .nodes import (
    clear_results,
    get_results,
    node_generate,
    node_ground,
    node_judge,
    node_judge_skipped,
    node_log,
    node_retrieve,
)
from .state import EvalState

logger = structlog.get_logger(__name__)


# ── Router ────────────────────────────────────────────────────────────────────


def route_after_ground(state: EvalState) -> str:
    """
    Conditional edge after the grounding node.

    Returns:
        "judge"         → grounding passed, proceed to LLM scoring
        "judge_skipped" → grounding failed, short-circuit to log
    """
    grounding = state.get("grounding_result")
    if grounding and grounding.deterministic_pass:
        return "judge"
    return "judge_skipped"


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_eval_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Called once at module load — the compiled graph is reused for every
    test case in a run (stateless between invocations).
    """
    graph = StateGraph(EvalState)

    # Register nodes
    graph.add_node("retrieve",      node_retrieve)
    graph.add_node("generate",      node_generate)
    graph.add_node("ground",        node_ground)
    graph.add_node("judge",         node_judge)
    graph.add_node("judge_skipped", node_judge_skipped)
    graph.add_node("log",           node_log)

    # Linear edges
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "ground")

    # Conditional edge: grounding gate
    graph.add_conditional_edges(
        "ground",
        route_after_ground,
        {
            "judge":         "judge",
            "judge_skipped": "judge_skipped",
        },
    )

    # Both judge paths converge at log
    graph.add_edge("judge",         "log")
    graph.add_edge("judge_skipped", "log")
    graph.add_edge("log",           END)

    return graph.compile()


# Compile once at import time
_eval_graph = build_eval_graph()


# ── Single test case runner ───────────────────────────────────────────────────


async def run_single(
    run_id: str,
    test_case: TestCase,
    model_cfg: ModelConfig,
    source_text: str,
) -> EvalState:
    """
    Run the full pipeline for a single test case.
    Returns the final EvalState after all nodes have executed.
    """
    initial_state: EvalState = {
        "run_id": run_id,
        "test_case": test_case,
        "model_cfg": model_cfg,
        "source_text": source_text,
        "retrieved_context": "",
        "model_output": "",
        "response_metrics": None,       # type: ignore[assignment]
        "grounding_result": None,       # type: ignore[assignment]
        "judge_result": None,           # type: ignore[assignment]
        "judge_skipped": False,
        "result_id": "",
        "error": None,
    }

    final_state: EvalState = await _eval_graph.ainvoke(initial_state)
    return final_state


# ── Full run orchestrator ─────────────────────────────────────────────────────


async def run_pipeline(
    model_cfg: ModelConfig,
    domain: Domain,
    run_id: str | None = None,
    n_cases: int | None = None,
    baseline_run_id: str | None = None,
    concurrency: int = 3,
) -> RunSummary:
    """
    Run the evaluation pipeline across a full dataset.

    Args:
        model_cfg:        The model to evaluate.
        domain:           Which dataset to use (medical / finance).
        run_id:           Optional — generated if not provided.
        n_cases:          Limit to first N cases (None = full dataset).
        baseline_run_id:  If provided, compare against this run for CI gate.
        concurrency:      Max parallel pipeline executions (default 3 —
                          keeps API rate limits happy).

    Returns:
        RunSummary with bootstrapped metrics and CI gate result.
    """
    run_id = run_id or str(uuid.uuid4())
    logger.info("pipeline_run_started", run_id=run_id, model=model_cfg.model_id, domain=domain.value)

    # ── Load test cases ──────────────────────────────────────────────
    test_cases = _load_test_cases(domain, n_cases)
    if not test_cases:
        raise ValueError(f"No test cases found for domain: {domain.value}")

    source_texts = _load_source_texts(domain)

    logger.info("pipeline_cases_loaded", n_cases=len(test_cases))

    # ── Run pipeline with bounded concurrency ────────────────────────
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(tc: TestCase) -> EvalState:
        async with semaphore:
            source = source_texts.get(tc.source_doc_id, "")
            return await run_single(run_id, tc, model_cfg, source)

    tasks = [run_with_semaphore(tc) for tc in test_cases]
    results_states = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Collect results ───────────────────────────────────────────────
    eval_results = get_results(run_id)

    if not eval_results:
        raise RuntimeError(f"Pipeline produced no results for run {run_id}")

    # ── Compute bootstrapped metrics ─────────────────────────────────
    accuracy_scores     = [r.judge.accuracy_score for r in eval_results]
    hallucination_flags = [float(r.judge.hallucination_detected) for r in eval_results]
    grounding_scores    = [r.grounding.grounding_score for r in eval_results]
    costs               = [r.metrics.cost_usd for r in eval_results]
    latencies           = [r.metrics.latency_ms for r in eval_results]

    n_iter = settings.bootstrap_iterations

    accuracy_metric     = bootstrap_metric(accuracy_scores,     n_iterations=n_iter)
    hallucination_metric= bootstrap_metric(hallucination_flags, n_iterations=n_iter)
    grounding_metric    = bootstrap_metric(grounding_scores,    n_iterations=n_iter)
    cost_metric         = bootstrap_metric(costs,               n_iterations=n_iter)

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))

    # ── CI gate ───────────────────────────────────────────────────────
    ci_gate_passed = True
    gate_failures: list[str] = []

    # Gate 1: hallucination rate must be below threshold
    if hallucination_metric.mean > settings.ci_hallucination_threshold:
        ci_gate_passed = False
        gate_failures.append(
            f"Hallucination rate {hallucination_metric.mean:.1%} exceeds "
            f"threshold {settings.ci_hallucination_threshold:.1%}"
        )

    # Gate 2: accuracy must not have regressed vs baseline
    # (only checked if a baseline run is provided)
    if baseline_run_id:
        baseline_summary = _load_baseline_summary(baseline_run_id)
        if baseline_summary:
            if is_regression(
                baseline_summary.accuracy,
                accuracy_metric,
                threshold=settings.ci_accuracy_drop_threshold,
            ):
                ci_gate_passed = False
                gate_failures.append(
                    f"Accuracy regressed from {baseline_summary.accuracy.mean:.1%} "
                    f"to {accuracy_metric.mean:.1%} (statistically significant)"
                )

    if gate_failures:
        logger.warning("ci_gate_failed", failures=gate_failures)
    else:
        logger.info("ci_gate_passed", run_id=run_id)

    # ── Build and return RunSummary ───────────────────────────────────
    summary = RunSummary(
        run_id=run_id,
        model_cfg=model_cfg,
        domain=domain,
        n_cases=len(eval_results),
        accuracy=accuracy_metric,
        hallucination_rate=hallucination_metric,
        grounding_score=grounding_metric,
        avg_cost_usd=cost_metric,
        p50_latency_ms=round(p50_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        ci_gate_passed=ci_gate_passed,
        baseline_run_id=baseline_run_id,
    )

    logger.info(
        "pipeline_run_complete",
        run_id=run_id,
        n_cases=summary.n_cases,
        accuracy=summary.accuracy.mean,
        hallucination_rate=summary.hallucination_rate.mean,
        ci_gate_passed=summary.ci_gate_passed,
        total_cost_usd=round(sum(costs), 4),
    )

    return summary


# ── Data loaders ──────────────────────────────────────────────────────────────


def _load_test_cases(domain: Domain, n: int | None = None) -> list[TestCase]:
    """Load test cases from data/{domain}/test_cases/*.json"""
    data_dir = Path(f"data/{domain.value}/test_cases")
    if not data_dir.exists():
        logger.error("test_cases_dir_not_found", path=str(data_dir))
        return []

    cases: list[TestCase] = []
    for json_file in sorted(data_dir.glob("*.json")):
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            try:
                cases.append(TestCase(**item))
            except Exception as exc:
                logger.warning("test_case_load_error", file=str(json_file), error=str(exc))

    if n is not None:
        cases = cases[:n]

    return cases


def _load_source_texts(domain: Domain) -> dict[str, str]:
    """
    Load source documents from data/{domain}/source_docs/*.txt
    Returns dict keyed by doc ID (filename without extension).
    """
    source_dir = Path(f"data/{domain.value}/source_docs")
    texts: dict[str, str] = {}

    if not source_dir.exists():
        return texts

    for txt_file in source_dir.glob("*.txt"):
        doc_id = txt_file.stem
        texts[doc_id] = txt_file.read_text(encoding="utf-8")

    return texts


# Baseline summaries in-memory store (Week 3: replace with Supabase query)
_baseline_store: dict[str, RunSummary] = {}


def save_baseline(summary: RunSummary) -> None:
    """Save a run summary as a baseline for future CI comparisons."""
    _baseline_store[summary.run_id] = summary


def _load_baseline_summary(run_id: str) -> RunSummary | None:
    return _baseline_store.get(run_id)
