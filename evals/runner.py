from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running this file directly (python evals/runner.py) by ensuring the
# project root is on sys.path so that `import evals.metrics` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import structlog
import yaml
from pydantic import BaseModel

from deepresearch.agent.graph import run_agent
from deepresearch.agent.nodes import NodeDependencies
from deepresearch.config import Settings, get_settings
from deepresearch.domain.exceptions import DeepResearchError
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import TavilySearchProvider
from evals.metrics import (
    citation_coverage,
    cost_usd,
    latency_percentiles,
    topic_coverage_judge,
    unique_citation_count,
)

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _PROJECT_ROOT / "evals" / "dataset.yaml"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "evals" / "results"
_HTTP_CLIENT_TIMEOUT_SECONDS = 30.0
_SMOKE_SAMPLE_SIZE = 3
_PLACEHOLDER_TOKEN = "replace-me"


class DatasetEntry(BaseModel):
    id: str
    question: str
    expected_topics: list[str]
    must_cite_min: int = 0


@dataclass(frozen=True)
class _RunArgs:
    mode: str
    with_judge: bool
    output_dir: Path
    dataset: Path


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def _parse_args() -> _RunArgs:
    parser = argparse.ArgumentParser(description="Run the deep research agent eval suite.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    namespace = parser.parse_args()
    return _RunArgs(
        mode=namespace.mode,
        with_judge=namespace.with_judge,
        output_dir=namespace.output_dir,
        dataset=namespace.dataset,
    )


def _load_dataset(path: Path) -> list[DatasetEntry]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return [DatasetEntry.model_validate(item) for item in raw]


def _ensure_keys(settings: Settings) -> None:
    openai_key = settings.openai_api_key.get_secret_value()
    tavily_key = settings.tavily_api_key.get_secret_value()
    if _PLACEHOLDER_TOKEN in openai_key or _PLACEHOLDER_TOKEN in tavily_key:
        sys.stderr.write("API keys not configured; edit .env before running evals.\n")
        sys.exit(1)


async def _evaluate_entry(
    entry: DatasetEntry,
    deps: NodeDependencies,
    with_judge: bool,
) -> dict[str, Any]:
    prompt_before = deps.llm.prompt_tokens_used
    completion_before = deps.llm.completion_tokens_used
    start = time.perf_counter()
    try:
        report = await run_agent(entry.question, deps)
    except DeepResearchError as exc:
        latency = time.perf_counter() - start
        return _failed_result(entry, str(exc), latency)
    latency = time.perf_counter() - start

    prompt_delta = deps.llm.prompt_tokens_used - prompt_before
    completion_delta = deps.llm.completion_tokens_used - completion_before
    body = "\n\n".join(section.body for section in report.sections)
    topic_score = (
        await topic_coverage_judge(body, entry.expected_topics, deps.llm)
        if with_judge
        else None
    )
    return {
        "id": entry.id,
        "question": entry.question,
        "citation_coverage": round(citation_coverage(report), 3),
        "unique_citations": unique_citation_count(report),
        "iterations_used": report.iterations_used,
        "sources_consulted": report.sources_consulted,
        "latency_seconds": round(latency, 2),
        "prompt_tokens": prompt_delta,
        "completion_tokens": completion_delta,
        "cost_usd": round(cost_usd(prompt_delta, completion_delta, deps.settings.openai_model), 4),
        "topic_coverage": round(topic_score, 3) if topic_score is not None else None,
        "error": None,
    }


def _failed_result(entry: DatasetEntry, error: str, latency: float) -> dict[str, Any]:
    return {
        "id": entry.id,
        "question": entry.question,
        "citation_coverage": 0.0,
        "unique_citations": 0,
        "iterations_used": 0,
        "sources_consulted": 0,
        "latency_seconds": round(latency, 2),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "topic_coverage": None,
        "error": error,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in results if r["error"] is None]
    latencies = [r["latency_seconds"] for r in successful]
    percentiles = latency_percentiles(latencies)
    mean_cit = (
        sum(r["citation_coverage"] for r in successful) / len(successful) if successful else 0.0
    )
    judge_scores = [r["topic_coverage"] for r in successful if r["topic_coverage"] is not None]
    mean_topic = sum(judge_scores) / len(judge_scores) if judge_scores else None
    return {
        "n_queries": len(results),
        "n_successful": len(successful),
        "mean_citation_coverage": round(mean_cit, 3),
        "mean_topic_coverage": round(mean_topic, 3) if mean_topic is not None else None,
        "latency_p50_seconds": round(percentiles["p50"], 2),
        "latency_p95_seconds": round(percentiles["p95"], 2),
        "latency_mean_seconds": round(percentiles["mean"], 2),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
    }


def _select_entries(entries: Sequence[DatasetEntry], mode: str) -> list[DatasetEntry]:
    if mode == "smoke":
        return list(entries[:_SMOKE_SAMPLE_SIZE])
    return list(entries)


def _write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"{stamp}.json"
    md_path = output_dir / f"{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path, md_path


_TABLE_HEADER = (
    "| ID | Citations | Unique | Iters | Sources | "
    "Tokens | Latency (s) | Cost ($) | Topic | Error |"
)
_TABLE_SEPARATOR = (
    "|----|-----------|--------|-------|---------|"
    "--------|-------------|----------|-------|-------|"
)


def _render_markdown(payload: dict[str, Any]) -> str:
    summary: dict[str, Any] = payload["summary"]
    results: list[dict[str, Any]] = payload["results"]
    judge_label = "enabled" if payload["with_judge"] else "disabled"
    topic_value = (
        summary["mean_topic_coverage"] if summary["mean_topic_coverage"] is not None else "n/a"
    )
    lines = [
        f"# Eval run - {payload['run_at']} - {payload['mode']}",
        "",
        f"- Mode: {payload['mode']} ({summary['n_queries']} queries)",
        f"- Model: {payload['model']}",
        f"- Judge: {judge_label}",
        f"- Successful: {summary['n_successful']} / {summary['n_queries']}",
        f"- Mean citation coverage: {summary['mean_citation_coverage']}",
        f"- Mean topic coverage: {topic_value}",
        f"- Latency p50 / p95 / mean (s): {summary['latency_p50_seconds']} / "
        f"{summary['latency_p95_seconds']} / {summary['latency_mean_seconds']}",
        f"- Total cost (USD): {summary['total_cost_usd']}",
        "",
        "## Per-query results",
        "",
        _TABLE_HEADER,
        _TABLE_SEPARATOR,
    ]
    for r in results:
        topic = r["topic_coverage"] if r["topic_coverage"] is not None else "-"
        error = r["error"] if r["error"] else ""
        total_tokens = r["prompt_tokens"] + r["completion_tokens"]
        lines.append(
            f"| {r['id']} | {r['citation_coverage']} | {r['unique_citations']} | "
            f"{r['iterations_used']} | {r['sources_consulted']} | {total_tokens} | "
            f"{r['latency_seconds']} | {r['cost_usd']} | {topic} | {error} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> int:
    """Drive the eval suite, write JSON and Markdown reports, return exit code."""
    args = _parse_args()
    _configure_logging()
    settings = get_settings()
    _ensure_keys(settings)
    entries = _select_entries(_load_dataset(args.dataset), args.mode)
    logger.info("eval.start", mode=args.mode, queries=len(entries), with_judge=args.with_judge)

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=_HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:
        llm = OpenAIClient(settings)
        search = TavilySearchProvider(api_key=settings.tavily_api_key, client=http_client)
        deps = NodeDependencies(llm=llm, search=search, http_client=http_client, settings=settings)
        for entry in entries:
            logger.info("eval.query.start", id=entry.id)
            result = await _evaluate_entry(entry, deps, args.with_judge)
            logger.info(
                "eval.query.end",
                id=entry.id,
                latency_s=result["latency_seconds"],
                cost_usd=result["cost_usd"],
                error=result["error"],
            )
            results.append(result)

    payload: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": settings.openai_model,
        "with_judge": args.with_judge,
        "results": results,
        "summary": _summarize(results),
    }
    json_path, md_path = _write_outputs(args.output_dir, payload)
    logger.info(
        "eval.done",
        json=str(json_path),
        markdown=str(md_path),
        total_cost_usd=payload["summary"]["total_cost_usd"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
