"""
src/api/main.py
────────────────
FastAPI application — the HTTP layer over the evaluation engine.

Endpoints:
  POST /evaluate          Run a full evaluation pipeline (async, returns run_id)
  GET  /runs/{run_id}     Fetch run summary
  GET  /leaderboard       Current leaderboard across all runs
  GET  /health            Liveness probe

All responses follow the same Pydantic schema used internally.
Async throughout — no blocking calls on the event loop.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.models import Domain, ModelConfig, ModelProvider, RunSummary

logger = structlog.get_logger(__name__)

# ── Lifespan: startup / shutdown ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "api_startup",
        environment=settings.environment,
        judge_model=settings.judge_model,
    )
    yield
    logger.info("api_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="GenAI Reliability Framework",
    description=(
        "Domain-grounded LLM evaluation harness for regulated industries. "
        "Medical and financial workflow validation aligned with OSFI E-23 "
        "model risk management guidelines."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────


class EvaluateRequest(BaseModel):
    model_config_: ModelConfig  # trailing underscore avoids Pydantic reserved name
    domain: Domain
    dataset_path: str | None = None  # relative to data/ — defaults to domain default
    n_cases: int | None = None       # None = run full dataset
    baseline_run_id: str | None = None  # compare against this run for CI gate


class EvaluateResponse(BaseModel):
    run_id: str
    status: str  # "queued" | "running" | "complete" | "failed"
    message: str


class LeaderboardEntry(BaseModel):
    run_id: str
    model_display_name: str
    model_id: str
    domain: str
    accuracy_mean: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    hallucination_rate_mean: float
    grounding_score_mean: float
    avg_cost_usd: float
    p95_latency_ms: float
    ci_gate_passed: bool


# ── In-memory run state (replace with Supabase in production) ─────────────────
# This simple dict is fine for Week 1 demo; the storage module will replace it.

_run_store: dict[str, RunSummary | str] = {}  # run_id → RunSummary or "running"


# ── Background evaluation task ────────────────────────────────────────────────


async def _run_evaluation(
    run_id: str,
    request: EvaluateRequest,
) -> None:
    """
    Kicked off as a FastAPI background task.
    Imports evaluation_engine here to keep startup time low.
    """
    try:
        _run_store[run_id] = "running"
        logger.info("eval_run_started", run_id=run_id, model=request.model_config_.model_id)

        from src.evaluation_engine.pipeline import run_pipeline
        summary = await run_pipeline(
            model_cfg=request.model_config_,
            domain=request.domain,
            run_id=run_id,
            n_cases=request.n_cases,
            baseline_run_id=request.baseline_run_id,
        )
        _run_store[run_id] = summary
        logger.info("eval_run_complete", run_id=run_id, ci_gate_passed=summary.ci_gate_passed)

    except Exception as exc:
        logger.error("eval_run_failed", run_id=run_id, error=str(exc))
        _run_store[run_id] = f"failed: {exc}"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/evaluate", response_model=EvaluateResponse, status_code=202)
async def start_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
) -> EvaluateResponse:
    """
    Queue an evaluation run. Returns immediately with a run_id.
    Poll GET /runs/{run_id} for results.
    """
    run_id = str(uuid.uuid4())
    background_tasks.add_task(_run_evaluation, run_id, request)
    logger.info("eval_queued", run_id=run_id, model=request.model_config_.model_id)
    return EvaluateResponse(
        run_id=run_id,
        status="queued",
        message=f"Evaluation queued. Poll GET /runs/{run_id} for results.",
    )


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    entry = _run_store.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if isinstance(entry, str):
        return {"run_id": run_id, "status": entry}
    return {"run_id": run_id, "status": "complete", "summary": entry.model_dump()}


@app.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    domain: Domain | None = Query(None, description="Filter by domain"),
) -> list[LeaderboardEntry]:
    """
    Returns one row per completed run, sorted by accuracy descending.
    Week 2: replace _run_store with Supabase query.
    """
    entries: list[LeaderboardEntry] = []
    for run_id, entry in _run_store.items():
        if not isinstance(entry, RunSummary):
            continue
        if domain and entry.domain != domain:
            continue
        entries.append(
            LeaderboardEntry(
                run_id=run_id,
                model_display_name=entry.model_config.display_name,
                model_id=entry.model_config.model_id,
                domain=entry.domain.value,
                accuracy_mean=entry.accuracy.mean,
                accuracy_ci_lower=entry.accuracy.ci_lower,
                accuracy_ci_upper=entry.accuracy.ci_upper,
                hallucination_rate_mean=entry.hallucination_rate.mean,
                grounding_score_mean=entry.grounding_score.mean,
                avg_cost_usd=entry.avg_cost_usd.mean,
                p95_latency_ms=entry.p95_latency_ms,
                ci_gate_passed=entry.ci_gate_passed,
            )
        )

    return sorted(entries, key=lambda e: e.accuracy_mean, reverse=True)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment, "version": "0.1.0"}
