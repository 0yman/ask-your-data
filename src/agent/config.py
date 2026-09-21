"""Runtime configuration, loaded from environment or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

LLMBackend = Literal["gemini", "scripted"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", extra="ignore", protected_namespaces=()
    )

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    llm_backend: LLMBackend = "gemini"
    # Chosen by measurement, not by version number: on the free tier the
    # larger flash models are frequently 503 "high demand", while flash-lite
    # answered 3 probes in 4 at roughly a fifth of the latency. See the README.
    gemini_model: str = "gemini-3.1-flash-lite"

    temperature: float = 0.0
    max_output_tokens: int = 2048

    # Free-tier limits are per minute as well as per day, so a burst of agent
    # turns can hit 429 even well under the daily cap.
    max_retries: int = 5
    retry_base_delay: float = 2.0
    # Per-request HTTP timeout in milliseconds, passed to the SDK.
    request_timeout_ms: int = 60_000

    # --- agent loop ------------------------------------------------------
    # Put the whole schema in the system prompt, rather than making the agent
    # discover it with list_tables/describe_table. Costs tokens on every call;
    # `eval/run_eval.py --no-schema-prompt` measures what it buys.
    include_schema_in_prompt: bool = True
    # Each step is one model call. The cap is what stops a model that keeps
    # calling tools from running until the quota is gone.
    max_steps: int = 10
    # How many times a failing query may be handed back for correction before
    # the agent gives up on that approach.
    max_sql_retries: int = 3

    # --- warehouse -------------------------------------------------------
    db_path: Path = REPO_ROOT / "data" / "port.duckdb"
    max_rows: int = 1000
    query_timeout_seconds: float = 20.0
    # Rows pasted back into the model's context. Larger crowds out reasoning.
    max_rows_to_model: int = 30

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env."
            )
        return self.google_api_key


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
