from __future__ import annotations

import re
import statistics

from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, ConfigDict

from deepresearch.domain.models import Report
from deepresearch.infrastructure.llm import OpenAIClient

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_NON_TRIVIAL_PARAGRAPH_MIN_CHARS = 50

# USD per 1k tokens (input, output). Update when provider pricing changes.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.00025, 0.002),
    "gpt-4o-mini": (0.00015, 0.0006),
}

_JUDGE_SYSTEM = (
    "You evaluate whether a research report substantively covers each topic from "
    "an expected list. A topic counts as covered only if the report has at least "
    "one paragraph that explains, defines, or analyzes that topic in non-trivial "
    "detail. Match topics by meaning, not exact wording."
)

_JUDGE_USER_TEMPLATE = """Expected topics:
{topics}

Report body:
{body}

Return JSON exactly: {{"covered_topics": [...]}}
Only include topics from the expected list, using the exact wording above.
"""


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    covered_topics: list[str]


def citation_coverage(report: Report) -> float:
    body = "\n\n".join(section.body for section in report.sections)
    paragraphs = [
        p for p in body.split("\n\n") if len(p.strip()) >= _NON_TRIVIAL_PARAGRAPH_MIN_CHARS
    ]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if _CITATION_PATTERN.search(p))
    return cited / len(paragraphs)


def unique_citation_count(report: Report) -> int:
    body = "\n\n".join(section.body for section in report.sections)
    return len({m.group(1) for m in _CITATION_PATTERN.finditer(body)})


def cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    if model not in _PRICING:
        return 0.0
    input_rate, output_rate = _PRICING[model]
    return (prompt_tokens / 1000.0) * input_rate + (completion_tokens / 1000.0) * output_rate


def latency_percentiles(latencies_seconds: list[float]) -> dict[str, float]:
    if not latencies_seconds:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(latencies_seconds)
    return {
        "p50": float(statistics.median(ordered)),
        "p95": _percentile(ordered, 0.95),
        "mean": float(statistics.fmean(ordered)),
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


async def topic_coverage_judge(
    report_body: str,
    expected_topics: list[str],
    llm: OpenAIClient,
) -> float:
    if not expected_topics:
        return 0.0
    topic_lines = "\n".join(f"- {t}" for t in expected_topics)
    user_content = _JUDGE_USER_TEMPLATE.format(topics=topic_lines, body=report_body)
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_content}
    system_message: ChatCompletionUserMessageParam = {"role": "user", "content": _JUDGE_SYSTEM}
    output = await llm.complete_structured(
        messages=[system_message, user_message],
        response_model=_JudgeOutput,
    )
    expected_set = {t.lower() for t in expected_topics}
    matched = sum(1 for t in output.covered_topics if t.lower() in expected_set)
    return matched / len(expected_topics)
