from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel, HttpUrl, SecretStr

from deepresearch.config import Settings
from deepresearch.domain.models import (
    Citation,
    Query,
    Report,
    ReportSection,
    Source,
    SubQuestion,
)
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import SearchHit


def _utc(year: int = 2026, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def make_source(url: str = "https://example.com/article", content_chars: int = 400) -> Source:
    return Source(
        url=HttpUrl(url),
        title="Example article",
        fetched_at=_utc(),
        content="x" * content_chars,
        token_count=max(1, content_chars // 4),
    )


def make_search_hit(url: str = "https://example.com") -> SearchHit:
    return SearchHit(url=HttpUrl(url), title="title", snippet="snippet", score=0.5)


class FakeLLMClient(OpenAIClient):
    def __init__(
        self,
        settings: Settings,
        *,
        structured: list[BaseModel] | None = None,
        text: list[str] | None = None,
    ) -> None:
        super().__init__(settings)
        self._structured_queue: list[BaseModel] = list(structured or [])
        self._text_queue: list[str] = list(text or [])

    async def complete_structured(
        self,
        messages: Any,
        response_model: Any,
    ) -> Any:
        if not self._structured_queue:
            raise RuntimeError("FakeLLMClient structured queue exhausted")
        return self._structured_queue.pop(0)

    async def complete_text(self, messages: Any) -> str:
        if not self._text_queue:
            raise RuntimeError("FakeLLMClient text queue exhausted")
        return self._text_queue.pop(0)


class FakeSearchProvider:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = list(hits or [])

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        return list(self._hits)


@pytest.fixture
def fake_settings() -> Settings:
    # pydantic-settings accepts `_env_file` at runtime to skip .env loading; mypy lacks the stub.
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        openai_api_key=SecretStr("sk-test"),
        tavily_api_key=SecretStr("tvly-test"),
    )


@pytest.fixture
def sample_query() -> Query:
    return Query(text="What are recent advances in retrieval-augmented generation?")


@pytest.fixture
def sample_sub_question(sample_query: Query) -> SubQuestion:
    return SubQuestion(text="What is RAG?", parent_query_id=sample_query.id, order=0)


@pytest.fixture
def sample_source() -> Source:
    return make_source()


@pytest.fixture
def sample_report(sample_query: Query, sample_source: Source) -> Report:
    citation = Citation(source_id=sample_source.id, snippet="RAG snippet", claim="claim")
    section = ReportSection(title="Intro", body="body [1]", citations=[citation])
    return Report(
        query=sample_query,
        sections=[section],
        iterations_used=1,
        sources_consulted=1,
    )


@pytest.fixture
def fake_search_provider() -> FakeSearchProvider:
    return FakeSearchProvider()


@pytest.fixture
def mock_http_factory() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]:
    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


@pytest.fixture
def fresh_uuid() -> Callable[[], Any]:
    return uuid4
