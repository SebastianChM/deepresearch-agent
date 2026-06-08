from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import httpx
import pytest
from pydantic import HttpUrl, SecretStr
from pytest import MonkeyPatch

from deepresearch.agent.graph import run_agent
from deepresearch.agent.nodes import (
    NodeDependencies,
    _CriticOutput,
    _PlannedSubQuestion,
    _PlannerOutput,
)
from deepresearch.config import Settings
from deepresearch.domain.exceptions import InvalidReportError
from deepresearch.domain.models import Source
from deepresearch.infrastructure.search import SearchProvider
from tests.conftest import FakeLLMClient, FakeSearchProvider, make_search_hit, make_source


def _settings(max_iterations: int = 2) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        openai_api_key=SecretStr("sk-test"),
        tavily_api_key=SecretStr("tvly-test"),
        max_iterations=max_iterations,
    )


def _patch_fetch_to_return(monkeypatch: MonkeyPatch, sources: Sequence[Source]) -> None:
    async def fake_fetch(
        urls: list[HttpUrl],
        client: httpx.AsyncClient,
        timeout_seconds: float,
        max_tokens: int,
    ) -> list[Source]:
        return list(sources)

    monkeypatch.setattr("deepresearch.agent.nodes.fetch_and_extract", fake_fetch)


def _single_sub_question_planner() -> _PlannerOutput:
    return _PlannerOutput(sub_questions=[_PlannedSubQuestion(text="What is RAG?", order=0)])


async def test_graph_completes_in_single_iteration_when_no_gaps(
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _settings()
    structured_queue = [
        _single_sub_question_planner(),
        _CriticOutput(has_gaps=False, missing_topics=[], reasoning="ok"),
    ]
    text_queue = ["This is the synthesized paragraph [1].", "# Final report\nBody [1]"]
    fake_llm = FakeLLMClient(settings, structured=structured_queue, text=text_queue)
    fake_search = FakeSearchProvider(hits=[make_search_hit("https://example.com")])
    _patch_fetch_to_return(monkeypatch, [make_source("https://example.com")])

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search),
            http_client=http_client,
            settings=settings,
        )
        report = await run_agent("What are recent advances in RAG?", deps)

    assert report.iterations_used == 1
    assert report.sources_consulted == 1
    assert "Final report" in report.sections[0].body


async def test_graph_loops_until_critic_clears(monkeypatch: MonkeyPatch) -> None:
    settings = _settings(max_iterations=3)
    structured_queue = [
        _single_sub_question_planner(),
        _CriticOutput(has_gaps=True, missing_topics=["X"], reasoning="missing X"),
        _single_sub_question_planner(),
        _CriticOutput(has_gaps=False, missing_topics=[], reasoning="ok"),
    ]
    text_queue = [
        "Iter 1 synth paragraph [1].",
        "Iter 2 synth paragraph [1].",
        "# Final report\nBody [1]",
    ]
    fake_llm = FakeLLMClient(settings, structured=structured_queue, text=text_queue)
    fake_search = FakeSearchProvider(hits=[make_search_hit("https://example.com")])
    _patch_fetch_to_return(monkeypatch, [make_source("https://example.com")])

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search),
            http_client=http_client,
            settings=settings,
        )
        report = await run_agent("Multi-iteration question for RAG?", deps)

    assert report.iterations_used == 2


async def test_graph_stops_when_max_iterations_reached(monkeypatch: MonkeyPatch) -> None:
    settings = _settings(max_iterations=2)
    structured_queue = [
        _single_sub_question_planner(),
        _CriticOutput(has_gaps=True, missing_topics=["X"], reasoning="still gaps"),
        _single_sub_question_planner(),
        _CriticOutput(has_gaps=True, missing_topics=["Y"], reasoning="still gaps"),
    ]
    text_queue = [
        "Iter 1 synth paragraph [1].",
        "Iter 2 synth paragraph [1].",
        "# Forced final report\nBody [1]",
    ]
    fake_llm = FakeLLMClient(settings, structured=structured_queue, text=text_queue)
    fake_search = FakeSearchProvider(hits=[make_search_hit("https://example.com")])
    _patch_fetch_to_return(monkeypatch, [make_source("https://example.com")])

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search),
            http_client=http_client,
            settings=settings,
        )
        report = await run_agent("Stop at cap question for RAG?", deps)

    assert report.iterations_used == 2
    assert "Forced final report" in report.sections[0].body


async def test_graph_raises_when_writer_emits_no_report(monkeypatch: MonkeyPatch) -> None:
    settings = _settings()
    fake_llm = FakeLLMClient(settings)

    async def fake_run(*_args: object, **_kwargs: object) -> dict[str, None]:
        return {"final_report": None}

    monkeypatch.setattr("deepresearch.agent.graph.build_graph", lambda deps: _StubGraph())

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, FakeSearchProvider()),
            http_client=http_client,
            settings=settings,
        )
        with pytest.raises(InvalidReportError):
            await run_agent("A valid sample question?", deps)


class _StubGraph:
    async def ainvoke(self, state: object) -> dict[str, None]:
        return {"final_report": None}
