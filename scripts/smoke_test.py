from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import httpx
import structlog

from deepresearch.agent.graph import run_agent
from deepresearch.agent.nodes import NodeDependencies
from deepresearch.config import Settings, get_settings
from deepresearch.domain.exceptions import DeepResearchError
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import TavilySearchProvider

_DEFAULT_QUESTION = "What are the most recent advances in retrieval-augmented generation?"
_PLACEHOLDER_TOKEN = "replace-me"
_SEPARATOR = "=" * 80
_HTTP_CLIENT_TIMEOUT_SECONDS = 30.0
_EXIT_KEYS_NOT_CONFIGURED = 1
_EXIT_AGENT_FAILED = 2


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deepresearch agent end-to-end against a single question.",
    )
    parser.add_argument("--question", type=str, default=_DEFAULT_QUESTION)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_api_keys(settings: Settings) -> None:
    openai_key = settings.openai_api_key.get_secret_value()
    tavily_key = settings.tavily_api_key.get_secret_value()
    if _PLACEHOLDER_TOKEN in openai_key or _PLACEHOLDER_TOKEN in tavily_key:
        sys.stderr.write(
            "API keys are not configured. Edit .env and replace the placeholder values "
            "for OPENAI_API_KEY and TAVILY_API_KEY before running this script.\n"
        )
        sys.exit(_EXIT_KEYS_NOT_CONFIGURED)


def _print_report(
    body: str,
    iterations: int,
    source_count: int,
    tokens: int,
    latency_seconds: float,
) -> None:
    out = sys.stdout
    out.write(f"\n{_SEPARATOR}\n")
    out.write(body)
    if not body.endswith("\n"):
        out.write("\n")
    out.write(f"{_SEPARATOR}\n")
    out.write(
        f"iterations={iterations}  sources={source_count}  "
        f"tokens={tokens}  latency_s={latency_seconds:.2f}\n"
    )


async def main() -> int:
    """Drive the agent end-to-end against a single CLI-provided question and print the report."""
    args = _parse_args()
    _configure_logging(args.verbose)
    settings = get_settings()
    _validate_api_keys(settings)

    async with httpx.AsyncClient(timeout=_HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:
        llm = OpenAIClient(settings)
        search = TavilySearchProvider(api_key=settings.tavily_api_key, client=http_client)
        deps = NodeDependencies(
            llm=llm,
            search=search,
            http_client=http_client,
            settings=settings,
        )

        start = time.perf_counter()
        try:
            report = await run_agent(args.question, deps)
        except DeepResearchError as exc:
            sys.stderr.write(f"Agent failed: {exc}\n")
            return _EXIT_AGENT_FAILED
        latency_seconds = time.perf_counter() - start

    _print_report(
        body=report.sections[0].body,
        iterations=report.iterations_used,
        source_count=report.sources_consulted,
        tokens=llm.total_tokens_used,
        latency_seconds=latency_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
