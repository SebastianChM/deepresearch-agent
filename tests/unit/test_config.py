from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError
from pytest import MonkeyPatch

from deepresearch.config import Settings, get_settings


def test_settings_loads_from_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
    monkeypatch.setenv("MAX_ITERATIONS", "2")
    get_settings.cache_clear()
    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()
    assert settings.openai_api_key.get_secret_value() == "sk-from-env"
    assert settings.max_iterations == 2


def test_settings_rejects_max_iterations_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            openai_api_key=SecretStr("sk"),
            tavily_api_key=SecretStr("tv"),
            max_iterations=99,
        )


def test_settings_rejects_search_results_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            openai_api_key=SecretStr("sk"),
            tavily_api_key=SecretStr("tv"),
            search_results_per_query=42,
        )


def test_get_settings_returns_cached_instance(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cache-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-cache-test")
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
    finally:
        get_settings.cache_clear()
    assert first is second
