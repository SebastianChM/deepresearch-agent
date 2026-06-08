from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from deepresearch.domain.models import Query, Report, Source, SubQuestion


class AgentState(TypedDict):
    query: Query
    sub_questions: list[SubQuestion]
    sources: list[Source]
    partial_summaries: dict[UUID, str]
    iteration: int
    critique: str | None
    missing_topics: list[str]
    final_report: Report | None
