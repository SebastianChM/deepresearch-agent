from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from deepresearch.domain.exceptions import SearchProviderError
from deepresearch.infrastructure.search import (
    TavilySearchProvider,
    _parse_hits,
    _tavily_to_hit,
)


def test_tavily_to_hit_maps_content_to_snippet() -> None:
    raw = {
        "url": "https://example.com/article",
        "title": "Example",
        "content": "Example snippet text",
        "score": 0.9,
    }
    hit = _tavily_to_hit(raw)
    assert hit.snippet == "Example snippet text"
    assert hit.title == "Example"
    assert hit.score == 0.9


def test_parse_hits_drops_invalid_url_without_breaking_batch() -> None:
    results = [
        {"url": "not-a-valid-url", "title": "Bad", "content": "x"},
        {"url": "https://good.com", "title": "Good", "content": "y", "score": 0.5},
    ]
    hits = _parse_hits(results)
    assert len(hits) == 1
    assert hits[0].title == "Good"


async def test_tavily_provider_returns_hits_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a.com", "title": "A", "content": "snippet a", "score": 0.8},
                    {"url": "https://b.com", "title": "B", "content": "snippet b", "score": 0.7},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TavilySearchProvider(api_key=SecretStr("test"), client=client)
        hits = await provider.search("query", max_results=2)

    assert len(hits) == 2
    assert hits[0].title == "A"


async def test_tavily_provider_raises_on_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TavilySearchProvider(api_key=SecretStr("test"), client=client)
        with pytest.raises(SearchProviderError):
            await provider.search("query", max_results=2)


async def test_tavily_provider_raises_on_network_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TavilySearchProvider(api_key=SecretStr("test"), client=client)
        with pytest.raises(SearchProviderError):
            await provider.search("query", max_results=2)


async def test_tavily_provider_raises_when_results_field_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": "no results key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TavilySearchProvider(api_key=SecretStr("test"), client=client)
        with pytest.raises(SearchProviderError):
            await provider.search("query", max_results=2)
