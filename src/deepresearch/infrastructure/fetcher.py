from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from functools import lru_cache

import httpx
import structlog
import tiktoken
import trafilatura
from pydantic import HttpUrl

from deepresearch.domain.models import Source

logger = structlog.get_logger(__name__)

_USER_AGENT = "deepresearch-agent/0.1 (+https://github.com/sebastian/deepresearch-agent)"
_MIN_CONTENT_CHARS = 200
_TIKTOKEN_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_TIKTOKEN_ENCODING_NAME)


async def fetch_and_extract(
    urls: list[HttpUrl],
    client: httpx.AsyncClient,
    timeout_seconds: float,
    max_tokens: int,
) -> list[Source]:
    downloads = await asyncio.gather(
        *(_download(url, client, timeout_seconds) for url in urls),
        return_exceptions=True,
    )
    sources: list[Source] = []
    for url, outcome in zip(urls, downloads, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("fetch.unexpected_error", url=str(url), reason=str(outcome))
            continue
        if outcome is None:
            continue
        source = _build_source(url, outcome, max_tokens)
        if source is not None:
            sources.append(source)
    return sources


async def _download(
    url: HttpUrl,
    client: httpx.AsyncClient,
    timeout_seconds: float,
) -> str | None:
    start = time.perf_counter()
    try:
        response = await client.get(
            str(url),
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        logger.warning("fetch.download_failed", url=str(url), reason=str(exc))
        return None
    if response.status_code >= 400:
        logger.warning(
            "fetch.download_failed",
            url=str(url),
            reason=f"http_{response.status_code}",
        )
        return None
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "fetch.downloaded",
        url=str(url),
        bytes=len(response.content),
        latency_ms=round(latency_ms, 2),
    )
    return response.text


def _build_source(url: HttpUrl, html: str, max_tokens: int) -> Source | None:
    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if content is None or len(content) < _MIN_CONTENT_CHARS:
        logger.warning("fetch.skipped", url=str(url), reason="content_too_short")
        return None
    truncated, token_count = _truncate_by_tokens(content, max_tokens)
    return Source(
        url=url,
        title=_title_from(html, url),
        fetched_at=datetime.now(UTC),
        content=truncated,
        token_count=token_count,
    )


def _truncate_by_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    encoding = _encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens), len(truncated_tokens)


def _title_from(html: str, url: HttpUrl) -> str:
    metadata = trafilatura.extract_metadata(html)
    if metadata is not None and metadata.title:
        title: str = metadata.title
        return title
    host = url.host
    return host if host is not None else str(url)
