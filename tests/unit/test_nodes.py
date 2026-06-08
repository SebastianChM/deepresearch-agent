from __future__ import annotations

from typing import cast
from uuid import uuid4

import httpx

from deepresearch.agent.nodes import (
    NodeDependencies,
    _build_source_payload,
    _build_summary_payload,
    _critic_node,
    _CriticOutput,
    _dedupe_urls,
    _PlannedSubQuestion,
    _planner_node,
    _PlannerOutput,
)
from deepresearch.agent.state import AgentState
from deepresearch.config import Settings
from deepresearch.domain.models import Query, SubQuestion
from deepresearch.infrastructure.search import SearchProvider
from tests.conftest import FakeLLMClient, FakeSearchProvider, make_search_hit, make_source


def test_build_source_payload_assigns_sequential_indices() -> None:
    sources = [make_source("https://a.com"), make_source("https://b.com")]
    payload = _build_source_payload(sources)
    assert payload[0]["index"] == 1
    assert payload[1]["index"] == 2
    assert payload[0]["url"] == "https://a.com/"


def test_build_summary_payload_skips_sub_questions_without_summary() -> None:
    parent = uuid4()
    sq_with = SubQuestion(text="answered", parent_query_id=parent, order=0)
    sq_without = SubQuestion(text="missing", parent_query_id=parent, order=1)
    payload = _build_summary_payload([sq_with, sq_without], {sq_with.id: "the answer"})
    assert len(payload) == 1
    assert payload[0]["sub_question"] == "answered"


def test_dedupe_urls_preserves_first_occurrence_order() -> None:
    hits_a = [make_search_hit("https://x.com"), make_search_hit("https://y.com")]
    hits_b = [make_search_hit("https://x.com"), make_search_hit("https://z.com")]
    unique = _dedupe_urls([hits_a, hits_b])
    assert [str(u) for u in unique] == [
        "https://x.com/",
        "https://y.com/",
        "https://z.com/",
    ]


def _empty_state(query: Query) -> AgentState:
    return {
        "query": query,
        "sub_questions": [],
        "sources": [],
        "partial_summaries": {},
        "iteration": 0,
        "critique": None,
        "missing_topics": [],
        "final_report": None,
    }


async def test_planner_node_builds_sub_questions_from_llm_output(
    sample_query: Query,
    fake_settings: Settings,
    fake_search_provider: FakeSearchProvider,
) -> None:
    planner_output = _PlannerOutput(
        sub_questions=[
            _PlannedSubQuestion(text="What is X?", order=0),
            _PlannedSubQuestion(text="What is Y?", order=1),
        ]
    )
    fake_llm = FakeLLMClient(fake_settings, structured=[planner_output])

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search_provider),
            http_client=http_client,
            settings=fake_settings,
        )
        result = await _planner_node(_empty_state(sample_query), deps)

    assert result["iteration"] == 1
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"][0].text == "What is X?"
    assert result["sub_questions"][0].parent_query_id == sample_query.id


async def test_critic_node_returns_none_critique_when_no_gaps(
    sample_query: Query,
    sample_sub_question: SubQuestion,
    fake_settings: Settings,
    fake_search_provider: FakeSearchProvider,
) -> None:
    critic_output = _CriticOutput(has_gaps=False, missing_topics=[], reasoning="all good")
    fake_llm = FakeLLMClient(fake_settings, structured=[critic_output])
    state = _empty_state(sample_query)
    state["sub_questions"] = [sample_sub_question]
    state["partial_summaries"] = {sample_sub_question.id: "an answer"}

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search_provider),
            http_client=http_client,
            settings=fake_settings,
        )
        result = await _critic_node(state, deps)

    assert result["critique"] is None
    assert result["missing_topics"] == []


async def test_critic_node_returns_critique_text_when_gaps_exist(
    sample_query: Query,
    sample_sub_question: SubQuestion,
    fake_settings: Settings,
    fake_search_provider: FakeSearchProvider,
) -> None:
    critic_output = _CriticOutput(
        has_gaps=True,
        missing_topics=["evaluation methods"],
        reasoning="missing benchmarks",
    )
    fake_llm = FakeLLMClient(fake_settings, structured=[critic_output])
    state = _empty_state(sample_query)
    state["sub_questions"] = [sample_sub_question]
    state["partial_summaries"] = {sample_sub_question.id: "an answer"}

    async with httpx.AsyncClient() as http_client:
        deps = NodeDependencies(
            llm=fake_llm,
            search=cast(SearchProvider, fake_search_provider),
            http_client=http_client,
            settings=fake_settings,
        )
        result = await _critic_node(state, deps)

    assert result["critique"] == "missing benchmarks"
    assert result["missing_topics"] == ["evaluation methods"]
