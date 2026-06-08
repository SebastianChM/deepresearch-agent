from __future__ import annotations

import httpx
from pydantic import HttpUrl

from deepresearch.infrastructure.fetcher import _truncate_by_tokens, fetch_and_extract

_PARAGRAPH = (
    "<p>Modern retrieval-augmented generation pipelines combine dense and "
    "sparse retrieval to balance precision and recall.</p>"
)
_RICH_HTML = "<html><body><article>" + (_PARAGRAPH * 6) + "</article></body></html>"


def test_truncate_by_tokens_passes_through_short_text() -> None:
    text, count = _truncate_by_tokens("hello world", max_tokens=10)
    assert text == "hello world"
    assert count >= 1


def test_truncate_by_tokens_truncates_when_over_limit() -> None:
    long_text = "word " * 200
    text, count = _truncate_by_tokens(long_text, max_tokens=20)
    assert count == 20
    assert len(text) < len(long_text)


async def test_fetch_and_extract_returns_source_for_valid_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_RICH_HTML)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sources = await fetch_and_extract(
            urls=[HttpUrl("https://example.com")],
            client=client,
            timeout_seconds=5.0,
            max_tokens=1000,
        )

    assert len(sources) == 1
    assert sources[0].token_count > 0


async def test_fetch_and_extract_drops_404_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sources = await fetch_and_extract(
            urls=[HttpUrl("https://example.com")],
            client=client,
            timeout_seconds=5.0,
            max_tokens=1000,
        )

    assert sources == []


async def test_fetch_and_extract_skips_short_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body><p>hi</p></body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sources = await fetch_and_extract(
            urls=[HttpUrl("https://example.com")],
            client=client,
            timeout_seconds=5.0,
            max_tokens=1000,
        )

    assert sources == []


async def test_fetch_and_extract_survives_partial_failures() -> None:
    call_counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, html=_RICH_HTML)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sources = await fetch_and_extract(
            urls=[HttpUrl("https://bad.com"), HttpUrl("https://good.com")],
            client=client,
            timeout_seconds=5.0,
            max_tokens=1000,
        )

    assert len(sources) == 1
