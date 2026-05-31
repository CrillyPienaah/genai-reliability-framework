"""
src/config.py
─────────────
Central settings loaded from environment variables via pydantic-settings.
Import `settings` anywhere in the codebase — never read os.environ directly.
"""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM keys ────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), description="Anthropic API key")
    google_cloud_project: str = Field("", description="GCP project ID for Vertex AI")
    google_cloud_region: str = Field("us-central1")

    # ── Storage ──────────────────────────────────────────────────────
    supabase_url: str = Field("", description="Supabase project URL")
    supabase_anon_key: SecretStr = Field(SecretStr(""), description="Supabase anon key")
    supabase_service_role_key: SecretStr = Field(
        SecretStr(""), description="Supabase service role key (server-side only)"
    )
    database_url: str = Field(
        "postgresql+asyncpg://eval_user:eval_pass@localhost:5432/eval_db"
    )

    # ── App behaviour ────────────────────────────────────────────────
    environment: str = Field("development")
    log_level: str = Field("INFO")

    # ── Eval thresholds ──────────────────────────────────────────────
    bootstrap_iterations: int = Field(
        1000, description="Bootstrap resamples for CI calculation"
    )
    ci_hallucination_threshold: float = Field(
        0.15, description="Max allowed hallucination rate before CI gate fails"
    )
    ci_accuracy_drop_threshold: float = Field(
        0.02, description="Max allowed accuracy regression (absolute pp) before CI gate fails"
    )
    judge_model: str = Field(
        "gpt-4o",
        description="LLM used as evaluator judge — must differ from models under test",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience singleton
settings = get_settings()
