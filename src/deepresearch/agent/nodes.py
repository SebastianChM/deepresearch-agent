from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

import httpx
import structlog
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from deepresearch.agent.prompts import (
    CRITIC_PROMPT,
    PLANNER_PROMPT,
    SYNTHESIZER_PROMPT,
    WRITER_PROMPT,
)
from deepresearch.agent.state import AgentState
from deepresearch.config import Settings
from deepresearch.domain.models import Report, ReportSection, Source, SubQuestion
from deepresearch.infrastructure.fetcher import fetch_and_extract
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import SearchHit, SearchProvider

logger = structlog.get_logger(__name__)

_INSUFFICIENT_MARKER = "INSUFFICIENT_SOURCES"

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


class _PlannedSubQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    order: Annotated[int, Field(ge=0)]


class _PlannerOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    sub_questions: list[_PlannedSubQuestion]


class _CriticOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    has_gaps: bool
    missing_topics: list[str]
    reasoning: Annotated[str, Field(max_length=300)]


@dataclass(frozen=True)
class NodeDependencies:
    """Injectable runtime dependencies shared by every node in the agent graph."""

    llm: OpenAIClient
    search: SearchProvider
    http_client: httpx.AsyncClient
    settings: Settings


def _user_message(content: str) -> list[ChatCompletionMessageParam]:
    message: ChatCompletionUserMessageParam = {"role": "user", "content": content}
    return [message]


def _dedupe_urls(hits_per_query: list[list[SearchHit]]) -> list[HttpUrl]:
    seen: set[str] = set()
    unique: list[HttpUrl] = []
    for hits in hits_per_query:
        for hit in hits:
            key = str(hit.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(hit.url)
    return unique


def _build_source_payload(sources: list[Source]) -> list[dict[str, object]]:
    return [
        {
            "index": idx + 1,
            "url": str(src.url),
            "title": src.title,
            "content": src.content,
        }
        for idx, src in enumerate(sources)
    ]


def _build_summary_payload(
    sub_questions: list[SubQuestion],
    partial_summaries: dict[UUID, str],
) -> list[dict[str, str]]:
    return [
        {"sub_question": sq.text, "summary": partial_summaries[sq.id]}
        for sq in sub_questions
        if sq.id in partial_summaries
    ]


async def _synthesize_one(
    sub_question: SubQuestion,
    source_payload: list[dict[str, object]],
    llm: OpenAIClient,
) -> tuple[UUID, str]:
    prompt_text = SYNTHESIZER_PROMPT.render(
        sub_question=sub_question.text,
        sources=source_payload,
    )
    summary = (await llm.complete_text(_user_message(prompt_text))).strip()
    return sub_question.id, summary


async def _planner_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("node.planner.start")
    prompt_text = PLANNER_PROMPT.render(
        query=state["query"].text,
        previous_critique=state["critique"],
        missing_topics=state["missing_topics"],
    )
    output = await deps.llm.complete_structured(
        messages=_user_message(prompt_text),
        response_model=_PlannerOutput,
    )
    parent_id = state["query"].id
    sub_questions = [
        SubQuestion(text=item.text, parent_query_id=parent_id, order=item.order)
        for item in output.sub_questions
    ]
    next_iteration = state["iteration"] + 1
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "node.planner.end",
        sub_question_count=len(sub_questions),
        iteration=next_iteration,
        latency_ms=round(latency_ms, 2),
    )
    return {"sub_questions": sub_questions, "iteration": next_iteration}


async def _gather_sources_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("node.gather_sources.start")
    hits_per_query = await asyncio.gather(
        *(
            deps.search.search(sq.text, deps.settings.search_results_per_query)
            for sq in state["sub_questions"]
        )
    )
    unique_urls = _dedupe_urls(hits_per_query)
    sources = await fetch_and_extract(
        unique_urls,
        deps.http_client,
        deps.settings.fetch_timeout_seconds,
        deps.settings.max_tokens_per_source,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "node.gather_sources.end",
        unique_url_count=len(unique_urls),
        source_count=len(sources),
        latency_ms=round(latency_ms, 2),
    )
    return {"sources": sources}


async def _synthesizer_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("node.synthesizer.start")
    source_payload = _build_source_payload(state["sources"])
    results = await asyncio.gather(
        *(_synthesize_one(sq, source_payload, deps.llm) for sq in state["sub_questions"])
    )
    summaries: dict[UUID, str] = {
        sq_id: text for sq_id, text in results if text != _INSUFFICIENT_MARKER
    }
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "node.synthesizer.end",
        sub_question_count=len(state["sub_questions"]),
        partial_summary_count=len(summaries),
        latency_ms=round(latency_ms, 2),
    )
    return {"partial_summaries": summaries}


async def _critic_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("node.critic.start")
    payload = _build_summary_payload(state["sub_questions"], state["partial_summaries"])
    prompt_text = CRITIC_PROMPT.render(query=state["query"].text, partial_summaries=payload)
    output = await deps.llm.complete_structured(
        messages=_user_message(prompt_text),
        response_model=_CriticOutput,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "node.critic.end",
        has_gaps=output.has_gaps,
        missing_topic_count=len(output.missing_topics),
        latency_ms=round(latency_ms, 2),
    )
    return {
        "critique": output.reasoning if output.has_gaps else None,
        "missing_topics": output.missing_topics,
    }


async def _writer_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("node.writer.start")
    summary_payload = _build_summary_payload(state["sub_questions"], state["partial_summaries"])
    source_payload = _build_source_payload(state["sources"])
    prompt_text = WRITER_PROMPT.render(
        query=state["query"].text,
        partial_summaries=summary_payload,
        sources=source_payload,
    )
    markdown = await deps.llm.complete_text(_user_message(prompt_text))
    # Citations remain inline in the Markdown body; regex extraction of [N] markers is
    # fragile (matches inside code blocks and footnotes) and adds no value until the
    # evaluator scores per-claim attribution.
    report = Report(
        query=state["query"],
        sections=[ReportSection(title="Report", body=markdown, citations=[])],
        iterations_used=state["iteration"],
        sources_consulted=len(state["sources"]),
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "node.writer.end",
        markdown_chars=len(markdown),
        latency_ms=round(latency_ms, 2),
    )
    return {"final_report": report}


def _bind(
    deps: NodeDependencies,
    func: Callable[[AgentState, NodeDependencies], Awaitable[dict[str, Any]]],
) -> NodeFn:
    async def bound(state: AgentState) -> dict[str, Any]:
        return await func(state, deps)

    return bound


def build_nodes(deps: NodeDependencies) -> dict[str, NodeFn]:
    """Return the agent's nodes keyed by the names expected by the LangGraph builder."""
    return {
        "planner": _bind(deps, _planner_node),
        "gather_sources": _bind(deps, _gather_sources_node),
        "synthesizer": _bind(deps, _synthesizer_node),
        "critic": _bind(deps, _critic_node),
        "writer": _bind(deps, _writer_node),
    }
