"""
src/evaluation_engine/state.py
────────────────────────────────
LangGraph state schema for the evaluation pipeline.

Every node receives the full EvalState and returns a partial dict
of only the fields it modifies. LangGraph merges updates automatically.

Flow:
    retrieve → generate → ground → judge → log
    (with short-circuit: if ground fails, skip judge and go straight to log)
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from src.models import (
    GroundingResult,
    JudgeResult,
    ModelConfig,
    ResponseMetrics,
    TestCase,
)


class EvalState(TypedDict):
    """
    The single data envelope that flows through every pipeline node.
    All fields are optional at initialisation — nodes populate them progressively.
    """

    # ── Inputs (set before pipeline starts) ─────────────────────────
    run_id: str
    test_case: TestCase
    model_cfg: ModelConfig
    source_text: str                    # Contents of the source document

    # ── Node: retrieve ───────────────────────────────────────────────
    retrieved_context: str              # Top-k chunks from source doc

    # ── Node: generate ───────────────────────────────────────────────
    model_output: str                   # Raw text from model under test
    response_metrics: ResponseMetrics   # Tokens, latency, cost

    # ── Node: ground ─────────────────────────────────────────────────
    grounding_result: GroundingResult

    # ── Node: judge ──────────────────────────────────────────────────
    judge_result: JudgeResult
    judge_skipped: bool                 # True if grounding gate failed

    # ── Node: log ────────────────────────────────────────────────────
    result_id: str                      # UUID of saved EvalResult row
    error: str | None                   # Set if any node raises
