from __future__ import annotations

from dataclasses import dataclass

import jinja2

_env = jinja2.Environment(
    autoescape=False,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: jinja2.Template

    def render(self, **kwargs: object) -> str:
        return self.template.render(**kwargs)


_PLANNER_TEMPLATE = """\
You are a research planner. Break the user's question into 3 to 6 atomic, verifiable sub-questions that can each be answered by web sources.

Use the same language as the user's question (Spanish if the question is in Spanish, English otherwise).
{% if previous_critique and missing_topics %}

A previous decomposition was found insufficient. Critique:
{{ previous_critique }}

Topics that must be covered this iteration:
{% for topic in missing_topics %}- {{ topic }}
{% endfor %}
{% endif %}

User question:
{{ query }}

Output a single JSON object with this exact shape and nothing else:
{"sub_questions": [{"text": "...", "order": 0}, {"text": "...", "order": 1}]}

Do not invent sources or citations. Do not add explanatory text outside the JSON.
"""


_SYNTHESIZER_TEMPLATE = """\
You answer one focused sub-question using only the provided sources.

Sub-question:
{{ sub_question }}

Sources (cite each by its [index]):
{% for s in sources %}
[{{ s.index }}] {{ s.title }} ({{ s.url }})
{{ s.content }}

{% endfor %}

Write a single paragraph of 200 to 400 words that answers the sub-question. Every factual claim must end with an inline citation like [N] where N matches one of the source indices above. Use only information present in the sources; do not bring outside knowledge.

If the sources do not contain enough information to answer, output exactly:
INSUFFICIENT_SOURCES

Otherwise output the paragraph only. No headings, no bullet points, no preamble.
"""


_CRITIC_TEMPLATE = """\
You are a research critic. Decide whether the partial summaries together answer the original question well enough.

Original question:
{{ query }}

Partial summaries:
{% for item in partial_summaries %}
Sub-question: {{ item.sub_question }}
Summary: {{ item.summary }}

{% endfor %}

Evaluate for: contradictions between summaries, factual claims missing inline citations, aspects of the original question that are not yet covered.

Output a single JSON object with this exact shape and nothing else:
{"has_gaps": true, "missing_topics": ["topic 1", "topic 2"], "reasoning": "..."}

The "reasoning" field must be at most 300 characters. If has_gaps is false, missing_topics must be an empty list.
"""


_WRITER_TEMPLATE = """\
You assemble the final research report in Markdown.

Original question:
{{ query }}

Partial summaries (preserve their inline [index] citations exactly):
{% for item in partial_summaries %}
### {{ item.sub_question }}
{{ item.summary }}

{% endfor %}

Source registry (only cite sources actually referenced in the summaries above):
{% for s in sources %}- [{{ s.index }}] {{ s.title }} - {{ s.url }}
{% endfor %}

Produce a Markdown report with:
1. A short introduction relating the summaries to the original question.
2. Logical sections grouping related summaries (rename and merge headings where it improves flow).
3. A final "## Sources" section listing only the sources whose [index] appears in the body, formatted as: `[N] Title - URL`.

Output Markdown only. No frontmatter, no commentary, no closing remarks.
"""


PLANNER_PROMPT = Prompt(
    name="planner",
    version="1.0",
    template=_env.from_string(_PLANNER_TEMPLATE),
)

SYNTHESIZER_PROMPT = Prompt(
    name="synthesizer",
    version="1.0",
    template=_env.from_string(_SYNTHESIZER_TEMPLATE),
)

CRITIC_PROMPT = Prompt(
    name="critic",
    version="1.0",
    template=_env.from_string(_CRITIC_TEMPLATE),
)

WRITER_PROMPT = Prompt(
    name="writer",
    version="1.0",
    template=_env.from_string(_WRITER_TEMPLATE),
)
