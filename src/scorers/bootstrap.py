"""
src/scorers/bootstrap.py
─────────────────────────
Bootstrap resampling for all evaluation metrics.

WHY THIS EXISTS (interview talking point):
  A flat "84% accuracy" is meaningless without knowing the sampling distribution.
  If two models score 84% vs 86% but their 95% CIs overlap, the difference is
  statistical noise — deploying the "better" model provides zero real improvement.
  This module gates CI/CD on statistical significance, not raw numbers.

OSFI E-23 relevance:
  Model validation requires demonstrating that performance changes are
  material and not artefacts of test-set variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.models import BootstrappedMetric


def bootstrap_metric(
    values: list[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrappedMetric:
    """
    Compute mean + CI for a scalar metric via non-parametric bootstrap.

    Args:
        values:       Raw per-sample metric values (e.g. accuracy per test case).
        n_iterations: Number of bootstrap resamples. 1000 for production, 100 for dev.
        confidence:   CI level. 0.95 → 95% CI.
        seed:         Random seed for reproducibility.

    Returns:
        BootstrappedMetric with mean, ci_lower, ci_upper.

    Example:
        accuracy_scores = [1.0, 0.0, 1.0, 1.0, 0.0, ...]  # binary per case
        metric = bootstrap_metric(accuracy_scores)
        # → BootstrappedMetric(mean=0.72, ci_lower=0.61, ci_upper=0.83, n_samples=50)
    """
    if not values:
        raise ValueError("Cannot bootstrap an empty list of values")

    arr: NDArray[np.float64] = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Resample with replacement, compute mean each time
    boot_means: NDArray[np.float64] = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iterations)]
    )

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrappedMetric(
        mean=round(float(arr.mean()), 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        n_samples=len(values),
    )


def cis_overlap(a: BootstrappedMetric, b: BootstrappedMetric) -> bool:
    """
    Returns True if the two confidence intervals overlap —
    i.e. the difference between the two models is NOT statistically significant.

    Usage in CI gate:
        if cis_overlap(baseline.accuracy, candidate.accuracy):
            logger.info("No significant accuracy change — treating as equivalent")
    """
    return a.ci_lower <= b.ci_upper and b.ci_lower <= a.ci_upper


def is_regression(
    baseline: BootstrappedMetric,
    candidate: BootstrappedMetric,
    threshold: float = 0.02,
) -> bool:
    """
    Returns True if candidate has regressed vs baseline by more than `threshold`
    AND the difference is statistically significant (CIs do not overlap).

    This is the core CI gate predicate — used in .github/workflows/eval.yml.

    Args:
        baseline:   Metric from the last passing run (stored in Supabase).
        candidate:  Metric from the current PR run.
        threshold:  Minimum absolute drop to flag (default 2 percentage points).

    Returns:
        True  → fail the PR
        False → pass the PR
    """
    drop = baseline.mean - candidate.mean
    if drop <= 0:
        return False  # candidate is same or better
    if drop < threshold:
        return False  # drop is below noise floor
    # Only flag if the drop is statistically distinguishable
    return not cis_overlap(baseline, candidate)


def compare_metrics(
    baseline: BootstrappedMetric,
    candidate: BootstrappedMetric,
    threshold: float = 0.02,
) -> dict[str, object]:
    """
    Full comparison report — used in PR comment generation.
    """
    drop = baseline.mean - candidate.mean
    return {
        "baseline_mean": baseline.mean,
        "candidate_mean": candidate.mean,
        "absolute_change": round(candidate.mean - baseline.mean, 4),
        "direction": "improvement" if candidate.mean > baseline.mean else "regression",
        "cis_overlap": cis_overlap(baseline, candidate),
        "statistically_significant": not cis_overlap(baseline, candidate),
        "is_regression": is_regression(baseline, candidate, threshold),
        "verdict": (
            "FAIL — statistically significant regression exceeding threshold"
            if is_regression(baseline, candidate, threshold)
            else "PASS"
        ),
    }
