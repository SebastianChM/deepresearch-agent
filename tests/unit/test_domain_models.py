from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import HttpUrl, ValidationError

from deepresearch.domain.models import (
    Citation,
    Query,
    Report,
    ReportSection,
    Source,
    SubQuestion,
)


def test_query_accepts_text_within_length_bounds() -> None:
    query = Query(text="A valid research question")
    assert query.text.startswith("A valid")


def test_query_rejects_text_too_short() -> None:
    with pytest.raises(ValidationError):
        Query(text="hi")


def test_query_rejects_text_too_long() -> None:
    with pytest.raises(ValidationError):
        Query(text="x" * 501)


def test_sub_question_rejects_negative_order() -> None:
    with pytest.raises(ValidationError):
        SubQuestion(text="why?", parent_query_id=uuid4(), order=-1)


def test_sub_question_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        SubQuestion(text="", parent_query_id=uuid4(), order=0)


def test_source_domain_extracted_from_url() -> None:
    source = Source(
        url=HttpUrl("https://arxiv.org/abs/2005.11401"),
        title="paper",
        fetched_at=datetime.now(UTC),
        content="content",
        token_count=10,
    )
    assert source.domain == "arxiv.org"


def test_source_rejects_negative_token_count() -> None:
    with pytest.raises(ValidationError):
        Source(
            url=HttpUrl("https://x.com"),
            title="t",
            fetched_at=datetime.now(UTC),
            content="c",
            token_count=-1,
        )


def test_citation_rejects_snippet_longer_than_max() -> None:
    with pytest.raises(ValidationError):
        Citation(source_id=uuid4(), snippet="x" * 301, claim="claim")


def test_report_section_holds_citations() -> None:
    section = ReportSection(title="t", body="b", citations=[])
    assert section.citations == []


def test_report_requires_positive_iterations() -> None:
    with pytest.raises(ValidationError):
        Report(
            query=Query(text="A valid query"),
            sections=[],
            iterations_used=0,
            sources_consulted=0,
        )


def test_report_requires_non_negative_sources_consulted() -> None:
    with pytest.raises(ValidationError):
        Report(
            query=Query(text="A valid query"),
            sections=[],
            iterations_used=1,
            sources_consulted=-1,
        )


def test_query_is_immutable() -> None:
    query = Query(text="A valid query")
    with pytest.raises(ValidationError):
        query.text = "tampered"
