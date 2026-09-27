"""Loads .env (secrets) and config.yaml (non-secret settings) into typed objects.

Kept deliberately separate from the DB/client modules so tests can construct
a Settings/CategoryConfig by hand without touching disk or the environment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "ccpf.db"
DEFAULT_RAW_CACHE_DIR = REPO_ROOT / "data" / "raw"


class Settings(BaseSettings):
    """Secrets and environment-derived settings. Source: .env, then real env vars."""

    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    indiankanoon_api_token: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    def require_indiankanoon_token(self) -> str:
        if not self.indiankanoon_api_token:
            raise RuntimeError(
                "INDIANKANOON_API_TOKEN is not set. Copy .env.example to .env "
                "and fill in your API key."
            )
        return self.indiankanoon_api_token

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and fill in your API key."
            )
        return self.anthropic_api_key


class QueryConfig(BaseModel):
    id: str
    text: str


class FilterConfig(BaseModel):
    ncdrc_source_pattern: str
    award_amount_patterns: list[str]
    award_verb_patterns: list[str]


class CategoryConfig(BaseModel):
    label: str
    budget_inr: Optional[float] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    queries: list[QueryConfig] = Field(default_factory=list)
    filters: Optional[FilterConfig] = None


class ApiConfig(BaseModel):
    base_url: str
    rate_limit_per_sec: float = 1.0
    max_retries: int = 5
    retry_min_wait_s: float = 2.0
    retry_max_wait_s: float = 30.0


class PricingConfig(BaseModel):
    search: int
    doc: int
    docfragment: int
    docmeta: int


class ExtractionConfig(BaseModel):
    budget_usd: float = 10.0


class NarrationConfig(BaseModel):
    # Haiku narration is tiny per query (~$0.003) — $5 covers roughly
    # 1500 chat queries before the app falls back to the raw display.
    budget_usd: float = 5.0


class AppConfig(BaseModel):
    api: ApiConfig
    pricing_paise: PricingConfig
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    narration: NarrationConfig = Field(default_factory=NarrationConfig)
    categories: dict[str, CategoryConfig]

    def get_category(self, name: str) -> CategoryConfig:
        try:
            cat = self.categories[name]
        except KeyError:
            raise KeyError(
                f"Unknown category {name!r}. Known: {sorted(self.categories)}"
            ) from None
        if not cat.queries:
            raise ValueError(
                f"Category {name!r} has no queries defined yet — not built out."
            )
        return cat


def load_app_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    # Loaded lazily (not at import time) so tests can monkeypatch os.environ
    # before the first call.
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings()
