"""
src/evaluation_engine/nodes.py
────────────────────────────────
The five evaluation pipeline nodes.

Each node:
  - Receives the full EvalState
  - Does one well-defined job
  - Returns a dict of ONLY the fields it changed

Node map:
  retrieve  → chunk source doc, return top-k context
  generate  → call model via ModelAdapter, log metrics
  ground    → deterministic entity extraction + verification
  judge     → LLM-as-judge scoring (skipped if ground failed)
  log       → persist EvalResult to Supabase / in-memory store
"""

from __future__ import annotations

import uuid

import structlog

from src.adapters.model_adapter import get_adapter
from src.models import (
    BootstrappedMetric,
    EvalResult,
    GroundingResult,
    HallucinationType,
    JudgeResult,
    ResponseMetrics,
)
from src.scorers.grounding import run_grounding_check
from src.scorers.judge import run_llm_judge

from .state import EvalState

logger = structlog.get_logger(__name__)


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────


async def node_retrieve(state: EvalState) -> dict:
    """
    Chunk the source document and return the most relevant context
    for this test case's prompt.

    Week 2 implementation: simple sliding-window chunking with keyword overlap.
    Week 3 upgrade path: swap for a proper vector store (pgvector on Supabase).
    """
    source_text = state.get("source_text", "")
    prompt = state["test_case"].prompt

    if not source_text:
        logger.warning("retrieve_empty_source", test_case_id=state["test_case"].id)
        return {"retrieved_context": "", "source_text": ""}

    # Simple retrieval: split into 200-word chunks, score by prompt keyword overlap
    chunks = _chunk_text(source_text, chunk_size=200, overlap=40)
    prompt_tokens = set(prompt.lower().split())

    scored = [
        (sum(1 for t in chunk.lower().split() if t in prompt_tokens), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top 2 chunks as context (stays within most LLM context windows)
    top_chunks = [chunk for _, chunk in scored[:2]]
    retrieved_context = "\n\n---\n\n".join(top_chunks)

    logger.debug(
        "retrieve_complete",
        test_case_id=state["test_case"].id,
        n_chunks=len(chunks),
        context_len=len(retrieved_context),
    )

    return {"retrieved_context": retrieved_context}


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


# ── Node 2: Generate ──────────────────────────────────────────────────────────


GENERATION_SYSTEM_PROMPT = """You are a knowledgeable assistant operating in a regulated
{domain} context. Answer the question accurately and concisely based on established
{domain} knowledge. If you are uncertain or the question is outside your scope,
say so explicitly rather than speculating."""


async def node_generate(state: EvalState) -> dict:
    """
    Call the model under evaluation via the unified ModelAdapter.
    Logs token usage, latency, and cost automatically.
    """
    test_case = state["test_case"]
    model_cfg = state["model_cfg"]
    context = state.get("retrieved_context", "")

    # Build prompt — inject retrieved context if available
    if context:
        prompt = (
            f"Use the following reference material to inform your answer:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {test_case.prompt}"
        )
    else:
        prompt = test_case.prompt

    system_prompt = GENERATION_SYSTEM_PROMPT.format(domain=test_case.domain.value)

    adapter = get_adapter(model_cfg)

    try:
        model_output, metrics = await adapter.generate_with_retry(
            prompt=prompt,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        logger.error(
            "generate_failed",
            test_case_id=test_case.id,
            model=model_cfg.model_id,
            error=str(exc),
        )
        # Return a failed output — pipeline continues and will flag as hallucination
        return {
            "model_output": f"[GENERATION ERROR: {exc}]",
            "response_metrics": ResponseMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                cost_usd=0.0,
            ),
            "error": str(exc),
        }

    logger.info(
        "generate_complete",
        test_case_id=test_case.id,
        model=model_cfg.model_id,
        tokens=metrics.total_tokens,
        cost_usd=metrics.cost_usd,
        latency_ms=metrics.latency_ms,
    )

    return {
        "model_output": model_output,
        "response_metrics": metrics,
    }


# ── Node 3: Ground ────────────────────────────────────────────────────────────


async def node_ground(state: EvalState) -> dict:
    """
    Deterministic mechanistic grounding check.

    Extracts entities from model output, verifies against source text.
    If grounding_score < threshold → sets deterministic_pass=False.
    The router after this node will skip the judge if it fails.

    This node never makes LLM calls — it's pure Python.
    """
    model_output = state.get("model_output", "")
    source_text = state.get("source_text", "")
    expected = state["test_case"].expected_answer

    # Short-circuit: if generation itself failed, mark as failed grounding
    if model_output.startswith("[GENERATION ERROR"):
        return {
            "grounding_result": GroundingResult(
                extracted_entities=[],
                verified_entities=[],
                grounding_score=0.0,
                hallucination_type=HallucinationType.FABRICATION,
                deterministic_pass=False,
            )
        }

    grounding_result = run_grounding_check(
        model_output=model_output,
        source_text=source_text,
        expected_answer=expected,
    )

    logger.info(
        "ground_complete",
        test_case_id=state["test_case"].id,
        grounding_score=grounding_result.grounding_score,
        passed=grounding_result.deterministic_pass,
        hallucination_type=grounding_result.hallucination_type.value,
    )

    return {"grounding_result": grounding_result}


# ── Node 4: Judge ─────────────────────────────────────────────────────────────


async def node_judge(state: EvalState) -> dict:
    """
    LLM-as-judge scoring.

    Only reached if grounding passed (enforced by the router).
    Uses a cross-family judge to avoid self-evaluation bias.
    Returns structured JSON scores: accuracy, hallucination, coherence.
    """
    test_case = state["test_case"]
    model_output = state.get("model_output", "")
    model_cfg = state["model_cfg"]

    try:
        judge_result = await run_llm_judge(
            question=test_case.prompt,
            expected=test_case.expected_answer,
            model_output=model_output,
            tested_provider=model_cfg.provider,
        )
    except Exception as exc:
        logger.error(
            "judge_failed",
            test_case_id=test_case.id,
            error=str(exc),
        )
        judge_result = JudgeResult(
            accuracy_score=0.0,
            hallucination_detected=True,
            hallucination_explanation=f"Judge call failed: {exc}",
            coherence_score=0.0,
            judge_confidence=0.0,
            reasoning="Judge node raised an exception.",
        )

    logger.info(
        "judge_complete",
        test_case_id=test_case.id,
        accuracy=judge_result.accuracy_score,
        hallucination=judge_result.hallucination_detected,
        confidence=judge_result.judge_confidence,
    )

    return {
        "judge_result": judge_result,
        "judge_skipped": False,
    }


async def node_judge_skipped(state: EvalState) -> dict:
    """
    Placeholder judge result when grounding gate fails.
    Saves judge API cost — no point scoring an output we know is ungrounded.
    """
    logger.info(
        "judge_skipped_grounding_failed",
        test_case_id=state["test_case"].id,
        grounding_score=state["grounding_result"].grounding_score,
    )
    return {
        "judge_result": JudgeResult(
            accuracy_score=0.0,
            hallucination_detected=True,
            hallucination_explanation="Skipped — failed deterministic grounding gate.",
            coherence_score=0.0,
            judge_confidence=1.0,  # We're certain it's wrong
            reasoning="Grounding check failed before judge was reached.",
        ),
        "judge_skipped": True,
    }


# ── Node 5: Log ───────────────────────────────────────────────────────────────


# In-memory store for Week 2 — replaced by Supabase in Week 3
_result_store: dict[str, EvalResult] = {}


async def node_log(state: EvalState) -> dict:
    """
    Persist the complete EvalResult.

    Week 2: stores in module-level dict (accessible via get_results()).
    Week 3: writes to Supabase eval_results table.
    """
    result_id = str(uuid.uuid4())

    # Build the complete result record
    result = EvalResult(
        id=result_id,
        run_id=state["run_id"],
        test_case_id=state["test_case"].id,
        model_cfg=state["model_cfg"],
        model_output=state.get("model_output", ""),
        grounding=state["grounding_result"],
        judge=state["judge_result"],
        metrics=state.get(
            "response_metrics",
            ResponseMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                cost_usd=0.0,
            ),
        ),
    )

    _result_store[result_id] = result

    logger.info(
        "log_complete",
        result_id=result_id,
        run_id=state["run_id"],
        test_case_id=state["test_case"].id,
        accuracy=result.judge.accuracy_score,
        hallucination=result.judge.hallucination_detected,
        cost_usd=result.metrics.cost_usd,
    )

    return {"result_id": result_id}


def get_results(run_id: str) -> list[EvalResult]:
    """Return all results for a given run_id."""
    return [r for r in _result_store.values() if r.run_id == run_id]


def clear_results() -> None:
    """Clear the in-memory store — used in tests."""
    _result_store.clear()
