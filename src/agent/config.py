"""Runtime configuration, loaded from environment or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"

LLMBackend = Literal["gemini", "openai", "scripted"]

# Public datasets offered next to the sample port warehouse. Each is one
# DuckDB file in `examples_dir`, named after its key.
EXAMPLE_DATASETS = ("retail", "co2")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute, so the app finds its .env whatever folder it is launched
        # from - a relative path silently loads nothing otherwise.
        env_file=ENV_FILE, env_prefix="AGENT_", extra="ignore", protected_namespaces=()
    )

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_backend: LLMBackend = "gemini"

    # Any OpenAI-format endpoint: leave blank for OpenAI itself, or point at
    # Groq, Together, OpenRouter, a local Ollama or vLLM. The wire format is
    # the same, so the adapter does not change.
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    # Tried in order when the main model is rate-limited or overloaded. On
    # Groq each model has its own quota, so this multiplies capacity.
    # From the environment as JSON: AGENT_OPENAI_FALLBACK_MODELS='["qwen/qwen3.8-27b"]'
    openai_fallback_models: list[str] = []
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
    # The sample dataset, generated from a fixed seed by build_warehouse.py.
    db_path: Path = DATA_DIR / "port.duckdb"
    # Tables built from the user's own CSV and Excel files. The agent only
    # ever opens this read-only; only the upload code writes to it.
    user_db_path: Path = DATA_DIR / "my_data.duckdb"
    uploads_dir: Path = DATA_DIR / "uploads"
    max_upload_mb: int = 50
    # Remembers which dataset the app was last showing.
    state_path: Path = DATA_DIR / "app_state.json"
    # Real public datasets, ready to ask about. Built by
    # scripts/build_examples.py; the app offers whichever of them exist.
    examples_dir: Path = DATA_DIR / "examples"
    max_rows: int = 1000
    query_timeout_seconds: float = 20.0
    # Rows pasted back into the model's context. Larger crowds out reasoning.
    max_rows_to_model: int = 30
    # DuckDB's own default is 80% of the machine's memory per database. Blank
    # keeps it; a public server sets a cap so one visitor's query cannot take
    # the memory every other visitor is using.
    duckdb_memory_limit: str = ""

    # --- public demo -----------------------------------------------------
    # Off, the app has one user: the person at this computer. On, every
    # visitor gets a private workspace, the server's key is rationed, and
    # nothing a visitor does is written to .env.
    public_mode: bool = False
    sessions_dir: Path = DATA_DIR / "sessions"
    session_idle_minutes: int = 60
    max_sessions: int = 300
    public_max_upload_mb: int = 10
    public_max_tables: int = 5
    # Questions answered on the server's key. Free-tier Gemini allows a fixed
    # number of model calls a day, and one question is 3-6 calls.
    public_questions_per_hour: int = 30
    public_questions_per_day: int = 250
    # Questions being worked on at once, across all visitors. The free tier
    # also limits calls per minute; queueing beats a wave of 429s.
    public_concurrent_questions: int = 3

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in .env, or switch back "
                "with AGENT_LLM_BACKEND=gemini."
            )
        return self.openai_api_key

    def has_model_key(self) -> bool:
        if self.llm_backend == "openai":
            return bool(self.openai_api_key)
        if self.llm_backend == "gemini":
            return bool(self.google_api_key)
        return True

    def dataset_path(self, dataset: str) -> Path:
        if dataset == "sample":
            return self.db_path
        if dataset == "mine":
            return self.user_db_path
        if dataset in EXAMPLE_DATASETS:
            return self.examples_dir / f"{dataset}.duckdb"
        raise ValueError(f"Unknown dataset {dataset!r}")

    def available_examples(self) -> list[str]:
        return [name for name in EXAMPLE_DATASETS if self.dataset_path(name).exists()]

    def upload_limit_mb(self) -> int:
        return self.public_max_upload_mb if self.public_mode else self.max_upload_mb

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env."
            )
        return self.google_api_key


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
