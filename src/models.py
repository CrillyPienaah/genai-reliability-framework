"""
src/models.py
─────────────
Pydantic v2 schemas used throughout the framework.
These are the single source of truth for data shapes — API, DB, and
internal pipeline all speak the same types.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ─────────────────────────────────────────────────────────────


class Domain(str, Enum):
    MEDICAL = "medical"
    FINANCE = "finance"
    LEGAL = "legal"  # Future: Dockett integration


class Severity(str, Enum):
    """How bad is a failure in this test case?"""
    CRITICAL = "critical"    # Patient safety / regulatory breach
    HIGH = "high"            # Material factual error
    MEDIUM = "medium"        # Degraded quality
    LOW = "low"              # Style / minor omission


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class HallucinationType(str, Enum):
    FABRICATION = "fabrication"      # Entity/claim not in source at all
    CONTRADICTION = "contradiction"  # Claim inverts source fact
    OMISSION = "omission"            # Required fact absent from output
    NONE = "none"                    # No hallucination detected


# ── Test case schema ──────────────────────────────────────────────────────────


class TestCase(BaseModel):
    """A single evaluation item stored in data/{domain}/test_cases/*.json"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: Domain
    severity: Severity
    prompt: str = Field(..., description="The query sent to the model under test")
    expected_answer: str = Field(..., description="Gold-standard reference answer")
    source_doc_id: str = Field(
        ..., description="ID of the source document in data/{domain}/source_docs/"
    )
    tags: list[str] = Field(default_factory=list, description="e.g. drug-interaction, dosage")
    scenario_type: str = Field(
        ...,
        description="happy_path | adversarial | recoverable_error | out_of_scope",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CalibrationSample(BaseModel):
    """Human-labelled sample used to calibrate LLM judge bias."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    test_case_id: str
    model_output: str
    human_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    human_hallucination_label: bool
    human_grounding_score: float = Field(..., ge=0.0, le=1.0)
    annotator_id: str = Field(default="human_1")
    notes: str = ""


# ── Model adapter config ──────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Identifies a model to evaluate."""

    provider: ModelProvider
    model_id: str = Field(..., description="e.g. gpt-4o, claude-sonnet-4-6, gemini-1.5-pro")
    display_name: str
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    extra_params: dict[str, Any] = Field(default_factory=dict)


# ── Per-response result ───────────────────────────────────────────────────────


class GroundingResult(BaseModel):
    """Output from the mechanistic grounding node."""

    extracted_entities: list[str] = Field(
        default_factory=list,
        description="Entities/figures/dates pulled from model output",
    )
    verified_entities: list[str] = Field(
        default_factory=list,
        description="Subset of extracted_entities found in source doc",
    )
    grounding_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="% of extracted entities verified in source (deterministic)",
    )
    hallucination_type: HallucinationType = HallucinationType.NONE
    deterministic_pass: bool = Field(
        ..., description="True if grounding_score >= domain threshold"
    )


class JudgeResult(BaseModel):
    """Output from the LLM-as-judge node (GPT-4o structured output)."""

    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_detected: bool
    hallucination_explanation: str = ""
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    judge_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Judge's self-reported confidence"
    )
    reasoning: str = Field(..., description="Chain-of-thought from the judge")


class ResponseMetrics(BaseModel):
    """Token economics and latency for one model call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float = Field(..., description="Computed from provider pricing tables")


class EvalResult(BaseModel):
    """Complete result record for one (test_case × model) pair."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    test_case_id: str
    model_cfg: ModelConfig
    model_output: str
    grounding: GroundingResult
    judge: JudgeResult
    metrics: ResponseMetrics
    pipeline_version: str = Field("0.1.0")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("model_output")
    @classmethod
    def output_not_empty(cls, v: str) -> str:
        return v if v else "[EMPTY OUTPUT]"


# ── Aggregate run summary ─────────────────────────────────────────────────────


class BootstrappedMetric(BaseModel):
    """A scalar metric with a 95% CI from bootstrap resampling."""

    mean: float
    ci_lower: float
    ci_upper: float
    n_samples: int

    @property
    def ci_width(self) -> float:
        return round(self.ci_upper - self.ci_lower, 4)

    @property
    def is_significant_vs(self) -> bool:
        """Placeholder: call compare_metrics() in scorers/bootstrap.py instead."""
        return False


class RunSummary(BaseModel):
    """Aggregated results for one complete evaluation run."""

    run_id: str
    model_cfg: ModelConfig
    domain: Domain
    n_cases: int
    accuracy: BootstrappedMetric
    hallucination_rate: BootstrappedMetric
    grounding_score: BootstrappedMetric
    avg_cost_usd: BootstrappedMetric
    p50_latency_ms: float
    p95_latency_ms: float
    ci_gate_passed: bool
    baseline_run_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
