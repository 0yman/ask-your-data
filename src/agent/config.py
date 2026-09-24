"""Runtime configuration, loaded from environment or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"

LLMBackend = Literal["gemini", "openai", "scripted"]

# Public datasets offered next to the sample port warehouse. Each is one
# DuckDB file in `examples_dir`, named after its key.
EXAMPLE_DATASETS = ("retail", "co2")


class ModelOption(BaseModel):
    """A model the page can offer, on an OpenAI-format host.

    Offered when its key is set. Each free tier rations differently, so each
    carries its own daily cap and how many questions it can work on at once.
    """

    id: str
    name: str
    host: str
    base_url: str
    model: str
    key_field: str      # the Settings field holding its key
    note: str           # one line for the picker: what it is good at, what it costs you
    per_day: int        # questions a day on the server's key
    concurrent: int = 1

    @property
    def label(self) -> str:
        return f"{self.name} · {self.host}"


MODEL_CATALOG = (
    # Free plan: $10 of API credit a month. At $0.5/M input and $1.5/M output
    # a question costs about a third of a cent.
    ModelOption(
        id="mistral-large", name="Mistral Large", host="Mistral",
        base_url="https://api.mistral.ai/v1", model="mistral-large-latest",
        key_field="mistral_api_key",
        note="Mistral's largest model, on Mistral's free plan.",
        per_day=80, concurrent=2,
    ),
    # Free plan: 8K tokens a minute, 200K a day. Each step resends the
    # conversation, so a broad question waits on the per-minute limit.
    ModelOption(
        id="qwen-groq", name="Qwen 3.8 27B", host="Groq",
        base_url="https://api.groq.com/openai/v1", model="qwen/qwen3.8-27b",
        key_field="groq_api_key",
        note="Fast on focused questions. Broad ones wait on its per-minute limit.",
        per_day=35, concurrent=1,
    ),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute, so the app finds its .env whatever folder it is launched
        # from - a relative path silently loads nothing otherwise.
        env_file=ENV_FILE, env_prefix="AGENT_", extra="ignore", protected_namespaces=(),
        populate_by_name=True,  # tests may pass google_api_key=..., not only GOOGLE_API_KEY=...
    )

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_backend: LLMBackend = "gemini"

    # Keys for the models in MODEL_CATALOG. Each one set adds its models to
    # the page's model picker; with none set the app uses the single model
    # configured by llm_backend below, as before.
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    # Which catalog model a new visitor starts on ("" = the first available).
    default_model: str = ""

    # Any OpenAI-format endpoint: leave blank for OpenAI itself, or point at
    # Groq, Together, OpenRouter, a local Ollama or vLLM. The wire format is
    # the same, so the adapter does not change.
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    # Tried in order when the main model is rate-limited or overloaded. On
    # Groq each model has its own quota, so this multiplies capacity.
    # From the environment as JSON: AGENT_OPENAI_FALLBACK_MODELS='["qwen/qwen3.8-27b"]'
    openai_fallback_models: list[str] = []
    # A second OpenAI-format host, tried after every model on the first.
    # The live demo: Cerebras first, Groq behind it, the same Qwen on both.
    openai_backup_base_url: str = ""
    openai_backup_api_key: str | None = Field(default=None, alias="OPENAI_BACKUP_API_KEY")
    openai_backup_model: str = ""
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

    def available_models(self) -> list[ModelOption]:
        return [option for option in MODEL_CATALOG if getattr(self, option.key_field)]

    def model_option(self, model_id: str | None) -> ModelOption | None:
        return next((o for o in self.available_models() if o.id == model_id), None)

    def default_model_option(self) -> ModelOption | None:
        available = self.available_models()
        return self.model_option(self.default_model) or (available[0] if available else None)

    def with_model(self, option: ModelOption) -> Settings:
        """These settings, answering with `option`. Everything else - files,
        limits, the visitor's workspace - stays as it is."""
        return self.model_copy(update={
            "llm_backend": "openai",
            "openai_base_url": option.base_url,
            "openai_model": option.model,
            "openai_api_key": getattr(self, option.key_field),
            "openai_fallback_models": [],
            "openai_backup_api_key": None,
            "openai_backup_model": "",
        })

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
