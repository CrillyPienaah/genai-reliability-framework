from __future__ import annotations
import json
import structlog
from sklearn.metrics import cohen_kappa_score
from src.adapters.model_adapter import get_adapter
from src.models import CalibrationSample, JudgeResult, ModelConfig, ModelProvider

logger = structlog.get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a rigorous AI evaluation expert. Return ONLY valid JSON with no markdown:
{"accuracy_score": <float 0.0-1.0>, "hallucination_detected": <bool>, "hallucination_explanation": "<string>", "coherence_score": <float 0.0-1.0>, "judge_confidence": <float 0.0-1.0>, "reasoning": "<1-3 sentences>"}
accuracy_score: 1.0=fully correct, 0.0=wrong. hallucination_detected: true only if output states facts not present in expected answer."""

JUDGE_USER_TEMPLATE = "QUESTION: {question}\n\nEXPECTED: {expected}\n\nOUTPUT: {output}"

def _select_judge_config(tested_provider: ModelProvider) -> ModelConfig:
    from src.config import settings
    judge_model = settings.judge_model
    provider_map = {
        "gpt-4o": ModelProvider.OPENAI,
        "gpt-4o-mini": ModelProvider.OPENAI,
        "claude-sonnet-4-6": ModelProvider.ANTHROPIC,
        "gemini-1.5-pro": ModelProvider.GOOGLE,
    }
    provider = provider_map.get(judge_model, ModelProvider.OPENAI)
    return ModelConfig(provider=provider, model_id=judge_model,
                       display_name=f"Judge ({judge_model})", temperature=0.0, max_tokens=512)

async def run_llm_judge(question: str, expected: str, model_output: str,
                        tested_provider: ModelProvider) -> JudgeResult:
    judge_config = _select_judge_config(tested_provider)
    adapter = get_adapter(judge_config)
    user_prompt = JUDGE_USER_TEMPLATE.format(question=question, expected=expected, output=model_output)
    raw_text, metrics = await adapter.generate_with_retry(prompt=user_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
    logger.debug("judge_call_complete", judge_model=judge_config.model_id, cost_usd=metrics.cost_usd)
    clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.error("judge_json_parse_failed", raw=raw_text[:200])
        return JudgeResult(accuracy_score=0.0, hallucination_detected=True,
                           hallucination_explanation="Parse failed", coherence_score=0.0,
                           judge_confidence=0.0, reasoning="JSON parse failure.")
    return JudgeResult(
        accuracy_score=float(data.get("accuracy_score", 0.0)),
        hallucination_detected=bool(data.get("hallucination_detected", True)),
        hallucination_explanation=str(data.get("hallucination_explanation", "")),
        coherence_score=float(data.get("coherence_score", 0.0)),
        judge_confidence=float(data.get("judge_confidence", 0.0)),
        reasoning=str(data.get("reasoning", "")),
    )

def compute_judge_calibration(samples: list[CalibrationSample],
                               judge_results: list[JudgeResult]) -> dict:
    human_labels = [int(s.human_hallucination_label) for s in samples]
    judge_labels = [int(r.hallucination_detected) for r in judge_results]
    kappa = float(cohen_kappa_score(human_labels, judge_labels))
    return {"cohen_kappa": round(kappa, 4), "n_samples": len(samples), "acceptable": kappa >= 0.60}