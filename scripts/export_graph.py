from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import httpx
from pydantic import SecretStr

from deepresearch.agent.graph import AgentGraph, build_graph, export_graph_mermaid
from deepresearch.agent.nodes import NodeDependencies
from deepresearch.config import Settings
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import SearchHit

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
_MMD_PATH = _OUTPUT_DIR / "graph.mmd"
_PNG_PATH = _OUTPUT_DIR / "graph.png"


class _NullSearchProvider:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:  # noqa: ARG002
        return []  # parameter names kept to satisfy the SearchProvider Protocol contract


def _build_inert_deps(http_client: httpx.AsyncClient) -> NodeDependencies:
    # pydantic-settings accepts `_env_file` at runtime to skip .env loading; mypy lacks the stub.
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        openai_api_key=SecretStr("sk-dummy"),
        tavily_api_key=SecretStr("tvly-dummy"),
    )
    return NodeDependencies(
        llm=OpenAIClient(settings),
        search=_NullSearchProvider(),
        http_client=http_client,
        settings=settings,
    )


def _write_mermaid(graph: AgentGraph) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _MMD_PATH.write_text(export_graph_mermaid(graph), encoding="utf-8")


def _try_write_png(graph: AgentGraph) -> bool:
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
    except (httpx.HTTPError, ValueError, ImportError, RuntimeError) as exc:
        logger.warning("PNG generation skipped: %s", exc)
        return False
    _PNG_PATH.write_bytes(png_bytes)
    return True


async def _generate() -> int:
    async with httpx.AsyncClient() as http_client:
        deps = _build_inert_deps(http_client)
        graph = build_graph(deps)
        _write_mermaid(graph)
        logger.info("Wrote %s", _MMD_PATH)
        if _try_write_png(graph):
            logger.info("Wrote %s", _PNG_PATH)
    return 0


def main() -> int:
    """Export the agent's LangGraph topology to docs/images as .mmd and optional .png."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_generate())


if __name__ == "__main__":
    sys.exit(main())
