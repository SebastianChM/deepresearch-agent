from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, computed_field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    text: Annotated[str, Field(min_length=5, max_length=500)]
    created_at: datetime = Field(default_factory=_utcnow)


class SubQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    text: Annotated[str, Field(min_length=1)]
    parent_query_id: UUID
    order: Annotated[int, Field(ge=0)]


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    title: str
    fetched_at: datetime
    content: str
    token_count: Annotated[int, Field(ge=0)]

    @computed_field  # type: ignore[prop-decorator]  # pydantic v2 expects bare @computed_field above @property
    @property
    def domain(self) -> str:
        host = self.url.host
        if host is None:
            raise ValueError("Source url has no host component")
        return host


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: UUID
    snippet: Annotated[str, Field(max_length=300)]
    claim: str


class ReportSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    body: str
    citations: list[Citation]


class Report(BaseModel):
    """Final research artifact assembled from per-sub-question summaries with their citations."""

    model_config = ConfigDict(frozen=True)

    query: Query
    sections: list[ReportSection]
    generated_at: datetime = Field(default_factory=_utcnow)
    iterations_used: Annotated[int, Field(ge=1)]
    sources_consulted: Annotated[int, Field(ge=0)]
