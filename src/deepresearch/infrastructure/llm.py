from __future__ import annotations

import time
from typing import TypeVar

import structlog
from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deepresearch.config import Settings
from deepresearch.domain.exceptions import LLMRefusalError

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_SDK_TIMEOUT_SECONDS = 60.0
_RETRY_ATTEMPTS = 3
_RETRY_MIN_WAIT_SECONDS = 2
_RETRY_MAX_WAIT_SECONDS = 10
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError)


_retry_transient_errors = retry(
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT_SECONDS, max=_RETRY_MAX_WAIT_SECONDS),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    reraise=True,
)


class OpenAIClient:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=_SDK_TIMEOUT_SECONDS,
        )
        self.total_tokens_used: int = 0

    async def complete_structured(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
    ) -> T:
        response, latency_ms = await self._parse_with_retry(messages, response_model)
        message = response.choices[0].message
        if message.refusal is not None:
            raise LLMRefusalError(reason=message.refusal)
        if message.parsed is None:
            raise LLMRefusalError(reason="Model returned no parsed content")
        self._record_usage(response.usage, latency_ms)
        return message.parsed

    async def complete_text(self, messages: list[ChatCompletionMessageParam]) -> str:
        response, latency_ms = await self._create_with_retry(messages)
        content = response.choices[0].message.content
        if content is None:
            raise LLMRefusalError(reason="Model returned empty content")
        self._record_usage(response.usage, latency_ms)
        return content

    @_retry_transient_errors
    async def _parse_with_retry(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
    ) -> tuple[ParsedChatCompletion[T], float]:
        start = time.perf_counter()
        response = await self._client.beta.chat.completions.parse(
            model=self._settings.openai_model,
            messages=messages,
            response_format=response_model,
        )
        return response, (time.perf_counter() - start) * 1000.0

    @_retry_transient_errors
    async def _create_with_retry(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> tuple[ChatCompletion, float]:
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=messages,
        )
        return response, (time.perf_counter() - start) * 1000.0

    def _record_usage(self, usage: CompletionUsage | None, latency_ms: float) -> None:
        if usage is None:
            return
        self.total_tokens_used += usage.total_tokens
        logger.info(
            "llm.completion",
            model=self._settings.openai_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
        )
