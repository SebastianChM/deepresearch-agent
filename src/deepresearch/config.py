from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    openai_api_key: SecretStr
    tavily_api_key: SecretStr
    openai_model: str = "gpt-5.4-mini"
    max_iterations: Annotated[int, Field(ge=1, le=6)] = 2
    search_results_per_query: Annotated[int, Field(ge=1, le=10)] = 3
    fetch_timeout_seconds: Annotated[float, Field(gt=0.0, le=60.0)] = 10.0
    max_tokens_per_source: Annotated[int, Field(ge=500, le=16000)] = 2000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings fills fields from env at runtime
