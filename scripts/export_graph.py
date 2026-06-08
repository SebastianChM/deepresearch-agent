from __future__ import annotations

import asyncio
import base64
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
_MMD_RAW_PATH = _OUTPUT_DIR / "graph_raw.mmd"
_MMD_PATH = _OUTPUT_DIR / "graph.mmd"
_PNG_PATH = _OUTPUT_DIR / "graph.png"
_MERMAID_INK_BASE = "https://mermaid.ink/img"

_STYLED_GRAPH = """graph TD;
    start([ start ]):::endpoint
    planner(Planner):::node
    gather(Search and Fetch):::node
    synth(Synthesize):::node
    critic(Critic):::loop
    writer(Writer):::node
    finish([ end ]):::endpoint
    start --> planner;
    planner --> gather;
    gather --> synth;
    synth --> critic;
    critic -.->|Gaps detected| planner;
    critic -.->|Approved| writer;
    writer --> finish;
    classDef node fill:#0e1117,stroke:#06b6d4,stroke-width:2px,color:#f1f5f9,rx:6,ry:6
    classDef loop fill:#1c1917,stroke:#f59e0b,stroke-width:2px,color:#fef3c7,rx:6,ry:6
    classDef endpoint fill:#1e293b,stroke:#64748b,color:#cbd5e1
    linkStyle default stroke:#475569,stroke-width:1.5px
"""


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


def _write_raw_topology(graph: AgentGraph) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _MMD_RAW_PATH.write_text(export_graph_mermaid(graph), encoding="utf-8")


def _write_styled_mermaid() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _MMD_PATH.write_text(_STYLED_GRAPH, encoding="utf-8")


async def _try_render_styled_png(client: httpx.AsyncClient) -> bool:
    encoded = base64.urlsafe_b64encode(_STYLED_GRAPH.encode("utf-8")).decode("ascii")
    url = f"{_MERMAID_INK_BASE}/{encoded}?type=png&bgColor=0a0e1a"
    try:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Styled PNG render failed: %s", exc)
        return False
    _PNG_PATH.write_bytes(response.content)
    return True


async def _generate() -> int:
    async with httpx.AsyncClient() as http_client:
        deps = _build_inert_deps(http_client)
        graph = build_graph(deps)
        _write_raw_topology(graph)
        logger.info("Wrote %s", _MMD_RAW_PATH)
        _write_styled_mermaid()
        logger.info("Wrote %s", _MMD_PATH)
        if await _try_render_styled_png(http_client):
            logger.info("Wrote %s", _PNG_PATH)
    return 0


def main() -> int:
    """Export the agent's LangGraph topology to docs/images as .mmd and optional .png."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_generate())


if __name__ == "__main__":
    sys.exit(main())
