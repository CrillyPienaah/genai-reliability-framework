"""
src/adapters/model_adapter.py
──────────────────────────────
Unified interface for calling any LLM provider.
Swap models by changing ModelConfig — zero changes elsewhere in the pipeline.

Supported providers:
  - OpenAI   (GPT-4o, GPT-4o-mini, GPT-4-turbo)
  - Anthropic (Claude Sonnet 4.6, Claude Haiku 4.5)
  - Google    (Gemini 1.5 Pro via Vertex AI — uses your GCP credentials)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models import ModelConfig, ModelProvider, ResponseMetrics

logger = structlog.get_logger(__name__)

# ── Pricing table (USD per 1K tokens, as of June 2025) ───────────────────────
# Update these when providers change pricing.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":       {"input": 0.00015,"output": 0.0006},
    "gpt-4-turbo":       {"input": 0.01,   "output": 0.03},
    "claude-sonnet-4-6": {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":  {"input": 0.00025,"output": 0.00125},
    "gemini-1.5-pro":    {"input": 0.00125,"output": 0.005},
    "gemini-1.5-flash":  {"input": 0.000075,"output": 0.0003},
}


def compute_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model_id, {"input": 0.01, "output": 0.03})
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000.0


# ── Abstract base ─────────────────────────────────────────────────────────────


class ModelAdapter(ABC):
    """
    All adapters return (response_text, ResponseMetrics).
    The pipeline never touches provider SDKs directly.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.log = structlog.get_logger(
            __name__, provider=config.provider, model=config.model_id
        )

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> tuple[str, ResponseMetrics]:
        """
        Returns (response_text, ResponseMetrics).
        Implementations must measure latency and compute cost.
        """
        ...

    async def generate_with_retry(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> tuple[str, ResponseMetrics]:
        """Wraps generate() with exponential backoff — use this in the pipeline."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
            reraise=True,
        ):
            with attempt:
                return await self.generate(prompt, system_prompt, **kwargs)
        raise RuntimeError("generate_with_retry exhausted all attempts")


# ── OpenAI adapter ────────────────────────────────────────────────────────────


class OpenAIAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # Lazy import so other adapters don't require openai installed
        from openai import AsyncOpenAI
        from src.config import settings

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> tuple[str, ResponseMetrics]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        assert usage is not None  # always present for non-streaming calls

        metrics = ResponseMetrics(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(
                compute_cost(self.config.model_id, usage.prompt_tokens, usage.completion_tokens),
                6,
            ),
        )

        text = response.choices[0].message.content or ""
        self.log.debug("openai_call", latency_ms=metrics.latency_ms, cost_usd=metrics.cost_usd)
        return text, metrics


# ── Anthropic adapter ─────────────────────────────────────────────────────────


class AnthropicAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import anthropic
        from src.config import settings

        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> tuple[str, ResponseMetrics]:
        t0 = time.perf_counter()

        create_kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await self._client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        metrics = ResponseMetrics(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(
                compute_cost(
                    self.config.model_id, usage.input_tokens, usage.output_tokens
                ),
                6,
            ),
        )

        text = response.content[0].text if response.content else ""
        self.log.debug("anthropic_call", latency_ms=metrics.latency_ms, cost_usd=metrics.cost_usd)
        return text, metrics


# ── Google Vertex AI adapter ──────────────────────────────────────────────────


class GoogleVertexAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from google.cloud import aiplatform
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
        from src.config import settings

        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.google_cloud_region,
        )
        self._GenerativeModel = GenerativeModel
        self._GenerationConfig = GenerationConfig

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> tuple[str, ResponseMetrics]:
        import asyncio

        model = self._GenerativeModel(
            self.config.model_id,
            system_instruction=system_prompt,
        )
        gen_config = self._GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

        t0 = time.perf_counter()
        # Vertex SDK is sync — run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt, generation_config=gen_config)
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage_meta = response.usage_metadata
        prompt_tokens = usage_meta.prompt_token_count
        completion_tokens = usage_meta.candidates_token_count

        metrics = ResponseMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(
                compute_cost(self.config.model_id, prompt_tokens, completion_tokens), 6
            ),
        )

        text = response.text or ""
        self.log.debug("vertex_call", latency_ms=metrics.latency_ms, cost_usd=metrics.cost_usd)
        return text, metrics


# ── Factory ───────────────────────────────────────────────────────────────────


def get_adapter(config: ModelConfig) -> ModelAdapter:
    """
    Factory function — the only place in the codebase that knows about providers.

    Usage:
        adapter = get_adapter(ModelConfig(provider=ModelProvider.OPENAI, model_id="gpt-4o", ...))
        text, metrics = await adapter.generate_with_retry(prompt)
    """
    dispatch: dict[ModelProvider, type[ModelAdapter]] = {
        ModelProvider.OPENAI: OpenAIAdapter,
        ModelProvider.ANTHROPIC: AnthropicAdapter,
        ModelProvider.GOOGLE: GoogleVertexAdapter,
    }
    cls = dispatch.get(config.provider)
    if cls is None:
        raise ValueError(f"No adapter registered for provider: {config.provider}")
    return cls(config)
