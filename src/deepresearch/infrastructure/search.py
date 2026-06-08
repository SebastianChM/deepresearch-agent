from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, HttpUrl, SecretStr, ValidationError

from deepresearch.domain.exceptions import SearchProviderError

logger = structlog.get_logger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_REQUEST_TIMEOUT_SECONDS = 10.0
_TAVILY_SEARCH_DEPTH = "basic"
_ERROR_BODY_PREVIEW_CHARS = 200
_QUERY_LOG_TRUNCATE_CHARS = 80


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    title: str
    snippet: str
    score: float | None


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class TavilySearchProvider:
    def __init__(self, api_key: SecretStr, client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        payload = {
            "api_key": self._api_key.get_secret_value(),
            "query": query,
            "max_results": max_results,
            "search_depth": _TAVILY_SEARCH_DEPTH,
        }
        start = time.perf_counter()
        try:
            response = await self._client.post(
                _TAVILY_ENDPOINT,
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"Tavily request failed: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text[:_ERROR_BODY_PREVIEW_CHARS]
            raise SearchProviderError(
                f"Tavily returned {response.status_code}: {body_preview}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise SearchProviderError(f"Tavily returned non-JSON body: {exc}") from exc

        raw_results = body.get("results")
        if not isinstance(raw_results, list):
            raise SearchProviderError("Tavily response missing 'results' list")

        hits = _parse_hits(raw_results)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "search.completed",
            provider="tavily",
            query=query[:_QUERY_LOG_TRUNCATE_CHARS],
            result_count=len(hits),
            latency_ms=round(latency_ms, 2),
        )
        return hits


def _parse_hits(raw_results: list[Any]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            logger.warning("search.invalid_hit_dropped", provider="tavily", reason="not_a_dict")
            continue
        try:
            hits.append(_tavily_to_hit(raw))
        except ValidationError as exc:
            first = exc.errors()[0]
            logger.warning(
                "search.invalid_hit_dropped",
                provider="tavily",
                field=".".join(str(loc) for loc in first["loc"]),
                error_type=first["type"],
                message=first["msg"],
            )
    return hits


def _tavily_to_hit(raw: dict[str, Any]) -> SearchHit:
    return SearchHit(
        url=raw.get("url", ""),
        title=raw.get("title", ""),
        snippet=raw.get("content", ""),
        score=raw.get("score"),
    )
