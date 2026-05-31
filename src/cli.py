"""
src/cli.py
───────────
CLI for running evaluations without starting the API server.

Usage:
    evaluate --model gpt-4o --domain medical --n 30
    evaluate --model claude-sonnet-4-6 --domain finance --baseline <run_id>
    evaluate calibrate --model gpt-4o
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.table import Table

from src.models import Domain, ModelConfig, ModelProvider

app = typer.Typer(
    name="evaluate",
    help="GenAI Reliability Framework — run evaluations from the command line.",
    add_completion=False,
)
console = Console()
logger = structlog.get_logger(__name__)

# ── Model shorthand map ────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
    ),
    "claude-sonnet-4-6": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
    ),
    "claude-haiku-4-5": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
    ),
    "gemini-1.5-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
    ),
}


# ── Commands ───────────────────────────────────────────────────────────────────


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model key (e.g. gpt-4o)"),
    domain: Domain = typer.Option(Domain.MEDICAL, "--domain", "-d"),
    n: int | None = typer.Option(None, "--n", help="Number of test cases (None = all)"),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Baseline run_id for CI comparison"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save JSON report to file"),
) -> None:
    """Run a full evaluation pipeline against a dataset."""

    model_config = MODEL_REGISTRY.get(model)
    if model_config is None:
        console.print(f"[red]Unknown model '{model}'. Available: {list(MODEL_REGISTRY.keys())}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]GenAI Reliability Framework[/bold]")
    console.print(f"  Model:   {model_config.display_name}")
    console.print(f"  Domain:  {domain.value}")
    console.print(f"  Cases:   {n or 'all'}")
    console.print(f"  Baseline: {baseline or 'none (first run)'}\n")

    from src.evaluation_engine.pipeline import run_pipeline, save_baseline

    console.print(f"\n[bold cyan]Starting evaluation run...[/bold cyan]\n")

    summary = asyncio.run(
        run_pipeline(
            model_cfg=model_config,
            domain=domain,
            n_cases=n,
            baseline_run_id=baseline,
        )
    )

    # Save as baseline for future CI comparisons
    save_baseline(summary)

    # ── Results table ────────────────────────────────────────────────
    table = Table(title=f"Run {summary.run_id[:8]}... · {model_config.display_name} · {domain.value}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("95% CI", justify="right", style="dim")

    table.add_row("Cases evaluated", str(summary.n_cases), "")
    table.add_row(
        "Accuracy",
        f"{summary.accuracy.mean:.1%}",
        f"[{summary.accuracy.ci_lower:.1%}, {summary.accuracy.ci_upper:.1%}]",
    )
    table.add_row(
        "Hallucination rate",
        f"{summary.hallucination_rate.mean:.1%}",
        f"[{summary.hallucination_rate.ci_lower:.1%}, {summary.hallucination_rate.ci_upper:.1%}]",
    )
    table.add_row(
        "Grounding score",
        f"{summary.grounding_score.mean:.1%}",
        f"[{summary.grounding_score.ci_lower:.1%}, {summary.grounding_score.ci_upper:.1%}]",
    )
    table.add_row("Avg cost / call", f"${summary.avg_cost_usd.mean:.4f}", "")
    table.add_row("p95 latency", f"{summary.p95_latency_ms:.0f}ms", "")

    ci_status = "[green]✓ PASSED[/green]" if summary.ci_gate_passed else "[red]✗ FAILED[/red]"
    table.add_row("CI gate", ci_status, "")

    console.print(table)
    console.print(f"\n[dim]Run ID: {summary.run_id}[/dim]")

    if output:
        import json as _json
        output.write_text(_json.dumps(summary.model_dump(), indent=2, default=str))
        console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def list_models() -> None:
    """List available model configurations."""
    table = Table(title="Available models")
    table.add_column("Key", style="cyan")
    table.add_column("Provider")
    table.add_column("Model ID")
    for key, config in MODEL_REGISTRY.items():
        table.add_row(key, config.provider.value, config.model_id)
    console.print(table)


@app.command()
def validate_data(
    domain: Domain = typer.Option(Domain.MEDICAL, "--domain", "-d"),
) -> None:
    """Validate all test cases in a domain against the TestCase schema."""
    from src.models import TestCase

    data_path = Path(f"data/{domain.value}/test_cases")
    if not data_path.exists():
        console.print(f"[red]No test cases found at {data_path}[/red]")
        raise typer.Exit(1)

    errors: list[str] = []
    total = 0

    for json_file in data_path.glob("*.json"):
        raw = json.loads(json_file.read_text())
        cases = raw if isinstance(raw, list) else [raw]
        for case_data in cases:
            total += 1
            try:
                TestCase(**case_data)
            except Exception as exc:
                errors.append(f"{json_file.name} [{case_data.get('id', '?')}]: {exc}")

    if errors:
        console.print(f"[red]✗ {len(errors)} validation errors across {total} cases:[/red]")
        for err in errors:
            console.print(f"  {err}")
        raise typer.Exit(1)

    console.print(f"[green]✓ All {total} test cases valid.[/green]")


if __name__ == "__main__":
    app()
