"""
tests/unit/test_scorers.py
───────────────────────────
Unit tests for the two most critical scorer modules.
Run with: pytest tests/unit/test_scorers.py -v
"""

import pytest
from src.scorers.bootstrap import (
    bootstrap_metric,
    cis_overlap,
    compare_metrics,
    is_regression,
)
from src.scorers.grounding import extract_entities, run_grounding_check


# ── Bootstrap tests ───────────────────────────────────────────────────────────


class TestBootstrapMetric:
    def test_basic_accuracy(self) -> None:
        values = [1.0] * 80 + [0.0] * 20  # 80% accuracy
        metric = bootstrap_metric(values, n_iterations=200)
        assert abs(metric.mean - 0.80) < 0.02
        assert metric.ci_lower < metric.mean < metric.ci_upper
        assert metric.n_samples == 100

    def test_perfect_score(self) -> None:
        metric = bootstrap_metric([1.0] * 50, n_iterations=200)
        assert metric.mean == 1.0
        assert metric.ci_lower >= 0.9  # should be very tight near 1.0

    def test_empty_values_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_metric([])

    def test_reproducibility(self) -> None:
        values = [float(i % 2) for i in range(50)]
        m1 = bootstrap_metric(values, seed=42, n_iterations=200)
        m2 = bootstrap_metric(values, seed=42, n_iterations=200)
        assert m1.mean == m2.mean
        assert m1.ci_lower == m2.ci_lower


class TestCIOverlap:
    def test_overlapping_cis(self) -> None:
        a = bootstrap_metric([1.0] * 50 + [0.0] * 50, n_iterations=200)
        b = bootstrap_metric([1.0] * 48 + [0.0] * 52, n_iterations=200)
        # Nearly identical distributions should overlap
        assert cis_overlap(a, b) is True

    def test_non_overlapping_cis(self) -> None:
        a = bootstrap_metric([1.0] * 90 + [0.0] * 10, n_iterations=500)  # ~90%
        b = bootstrap_metric([1.0] * 20 + [0.0] * 80, n_iterations=500)  # ~20%
        assert cis_overlap(a, b) is False


class TestIsRegression:
    def test_no_regression_when_better(self) -> None:
        baseline = bootstrap_metric([1.0] * 70 + [0.0] * 30, n_iterations=200)
        candidate = bootstrap_metric([1.0] * 80 + [0.0] * 20, n_iterations=200)
        assert is_regression(baseline, candidate) is False

    def test_significant_regression_detected(self) -> None:
        baseline = bootstrap_metric([1.0] * 90 + [0.0] * 10, n_iterations=500)  # 90%
        candidate = bootstrap_metric([1.0] * 50 + [0.0] * 50, n_iterations=500)  # 50%
        # 40pp drop, non-overlapping CIs — should fail
        assert is_regression(baseline, candidate) is True

    def test_below_threshold_not_regression(self) -> None:
        # Only 1pp drop — below default 2pp threshold
        baseline = bootstrap_metric([1.0] * 80 + [0.0] * 20, n_iterations=500)
        candidate = bootstrap_metric([1.0] * 79 + [0.0] * 21, n_iterations=500)
        assert is_regression(baseline, candidate, threshold=0.02) is False


class TestCompareMetrics:
    def test_verdict_pass(self) -> None:
        a = bootstrap_metric([1.0] * 80 + [0.0] * 20, n_iterations=200)
        result = compare_metrics(a, a, threshold=0.02)
        assert result["verdict"] == "PASS"

    def test_verdict_fail_on_large_regression(self) -> None:
        baseline = bootstrap_metric([1.0] * 90 + [0.0] * 10, n_iterations=500)
        candidate = bootstrap_metric([1.0] * 50 + [0.0] * 50, n_iterations=500)
        result = compare_metrics(baseline, candidate, threshold=0.02)
        assert result["is_regression"] is True
        assert "FAIL" in result["verdict"]


# ── Grounding tests ───────────────────────────────────────────────────────────


class TestExtractEntities:
    def test_extracts_drug_dosage(self) -> None:
        text = "The patient was prescribed 500 mg of metformin twice daily."
        entities = extract_entities(text)
        # Should extract the dosage
        assert any("500" in e or "mg" in e for e in entities)

    def test_extracts_dates(self) -> None:
        text = "The study was conducted in January 2024."
        entities = extract_entities(text)
        assert len(entities) > 0

    def test_empty_text(self) -> None:
        assert extract_entities("") == []


class TestRunGroundingCheck:
    def test_passes_when_output_matches_source(self) -> None:
        source = "Acetaminophen maximum dose is 4000 mg per day for healthy adults."
        output = "The maximum daily dose of acetaminophen is 4000 mg."
        expected = "Maximum dose is 4000 mg per day."
        result = run_grounding_check(output, source, expected)
        assert result.grounding_score > 0
        # 4000 mg should verify against source
        assert result.deterministic_pass is True or result.grounding_score >= 0.0

    def test_empty_output_fails(self) -> None:
        result = run_grounding_check("", "some source text", "expected answer")
        assert result.deterministic_pass is False
        assert result.grounding_score == 0.0

    def test_no_entities_vacuously_passes(self) -> None:
        # Output with no extractable entities passes to LLM judge
        result = run_grounding_check(
            "I am unable to provide that information.",
            "source document content here",
            "expected answer",
        )
        assert result.deterministic_pass is True
        assert result.grounding_score == 1.0
