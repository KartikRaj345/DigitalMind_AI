"""Application configuration.

Every setting is local-first by default. Anything that would cause data to leave
this machine (an LLM endpoint, a remote embedding API) is ``None`` unless the user
explicitly configures it, and the UI surfaces that state.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, overridable with ``DIGITALMIND_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DIGITALMIND_",
        env_file=".env",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")
    database_url: str | None = None

    # --- analysis ----------------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    risk_model_dir: Path = Field(default=REPO_ROOT / "artifacts" / "risk")

    # --- privacy -----------------------------------------------------------
    # Master switch for anything that could transmit record text off-device.
    allow_external_llm: bool = False
    llm_endpoint: str | None = None
    llm_model: str | None = None
    # When true, no record content is ever written to the log stream.
    redact_logs: bool = True

    # --- server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- analysis preferences ---------------------------------------------
    default_retrieval_mode: str = "hybrid"  # lexical | semantic | hybrid
    min_evidence_confidence: float = 0.5

    @property
    def db_path(self) -> Path:
        if self.database_url:
            return Path(self.database_url.replace("sqlite:///", ""))
        return self.data_dir / "digitalmind.db"

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url or f"sqlite:///{self.db_path}"

    @property
    def artifacts_dir(self) -> Path:
        return self.risk_model_dir

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Test helper: force settings to be re-read from the environment."""
    get_settings.cache_clear()
