---
title: Deep Research Agent
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LLM research agent that produces cited Markdown reports
colorFrom: indigo
colorTo: blue
---

# Deep Research Agent

> An LLM agent that plans, searches the web, reads sources, and synthesizes a Markdown research report with verifiable inline citations.

[![Open in HF Spaces](https://img.shields.io/badge/HuggingFace-Open%20in%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/SebastianChM/deepresearch-agent)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-46AAA8?style=for-the-badge)](https://docs.astral.sh/ruff/)
[![mypy strict](https://img.shields.io/badge/types-mypy%20strict-2A6DB4?style=for-the-badge)](https://mypy.readthedocs.io/)
[![tests](https://img.shields.io/badge/tests-42%20passing-22C55E?style=for-the-badge)](#testing)
[![coverage](https://img.shields.io/badge/coverage-90%25-22C55E?style=for-the-badge)](#testing)
[![evals](https://img.shields.io/badge/evals-20%20queries%20%C2%B7%200.93%20topic-22C55E?style=for-the-badge)](#evaluation-results)
[![License MIT](https://img.shields.io/badge/license-MIT-A8C95A?style=for-the-badge)](LICENSE)

**Live demo:** [huggingface.co/spaces/SebastianChM/deepresearch-agent](https://huggingface.co/spaces/SebastianChM/deepresearch-agent)

---

## What it does

Deep Research Agent turns a single research question into a cited Markdown report. Given a question, it decomposes it into 3–6 atomic sub-questions, runs parallel web searches via Tavily, downloads and extracts the most relevant sources with `trafilatura`, and synthesizes a grounded paragraph per sub-question with inline citations to the real URLs that backed each claim.

A critic node inspects the partial summaries between iterations. If it detects gaps, missing topics, or claims without citations, the planner runs again with the critique as context — the agent keeps refining until the report is comprehensive or the iteration budget is exhausted.

The output is downloadable as Markdown for power users and as a print-ready PDF for everyone else. A Streamlit UI streams each node's progress in real time, so the agent's reasoning is visible while it runs rather than hidden behind a spinner.

The whole pipeline is orchestrated with [LangGraph](https://langchain-ai.github.io/langgraph/), uses OpenAI's structured outputs for typed responses, and is built with `mypy --strict` from day one.

## Architecture

```mermaid
graph TD;
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
```

- **Planner** — Decomposes the user question into 3–6 atomic, verifiable sub-questions using OpenAI structured outputs.
- **Search and Fetch** — Parallel Tavily searches, deduplicates URLs, concurrent HTTP fetch, content extraction with trafilatura, token-budget truncation.
- **Synthesize** — One LLM call per sub-question, paragraph output grounded only in the fetched sources with inline `[N]` citations.
- **Critic** — Reviews the partial summaries; returns `has_gaps: bool` plus missing topics if any.
- **Writer** — Assembles the final Markdown report when the critic clears or the iteration budget is hit.

## Architecture decisions

- **LangGraph over a custom loop** — Streaming, conditional edges, and checkpointing for free. Re-implementing the state machine would have been two weeks of glue without portfolio upside.
- **Tavily over DuckDuckGo or HTML scraping** — Tavily's API is designed for LLM agents (relevance-scored JSON, free tier). Scraping search engines is brittle and rate-limited.
- **OpenAI structured outputs over JSON-mode** — `beta.chat.completions.parse` validates each response against a Pydantic model at the API boundary. Invalid shapes fail fast with `LLMRefusalError` instead of producing silent parse errors downstream.
- **Tenacity retries only on transient errors** — `RateLimitError` and `APITimeoutError` get exponential backoff. Every other exception propagates immediately; bugs deserve visibility.
- **Onion architecture (`domain` / `infrastructure` / `agent`)** — `Query`, `Source`, `Report` know nothing about HTTP or LangGraph. The `SearchProvider` Protocol means swapping Tavily for another backend is a single class.
- **Pure-Python PDF export** — `markdown` + `xhtml2pdf`, no WeasyPrint or wkhtmltopdf binaries. Deploys on any OS including HuggingFace Spaces and Streamlit Cloud with a single `pip install`.
- **`astream(stream_mode="updates")`** — The UI receives only the diff each node emits, not the full state. Lower latency in the websocket roundtrip and clearer rendering logic.
- **Strict tooling from day one** — `mypy --strict`, ruff with `N`, `B`, `C90`, `SIM`, `RUF`, `UP`, `ARG`, `PTH`. Pydantic models with `frozen=True`. Upfront cost; no surprise behavior at runtime.

## Quickstart

```bash
git clone https://github.com/<your-user>/deepresearch-agent.git
cd deepresearch-agent
cp .env.example .env   # then edit .env with your OPENAI_API_KEY and TAVILY_API_KEY
uv sync
uv run streamlit run app/streamlit_app.py
```

To run the agent end-to-end from the CLI without the UI:

```bash
uv run python scripts/smoke_test.py --question "Your research question here"
```

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| LLM | OpenAI (default `gpt-5.4-mini`) |
| Search | Tavily |
| Models / validation | Pydantic v2 |
| HTTP | httpx (async) |
| Content extraction | trafilatura |
| Tokenization | tiktoken |
| UI | Streamlit with custom CSS |
| PDF export | markdown + xhtml2pdf |
| Logging | structlog |
| Retries | tenacity |
| Package manager | uv |
| Lint / types | ruff + mypy strict |

## Project structure

```
deepresearch-agent/
├── src/deepresearch/
│   ├── config.py                  # pydantic-settings, reads .env
│   ├── domain/
│   │   ├── models.py              # Query, SubQuestion, Source, Citation, Report
│   │   └── exceptions.py          # Domain-specific error hierarchy
│   ├── infrastructure/
│   │   ├── llm.py                 # OpenAI client with retries, structured outputs
│   │   ├── search.py              # SearchProvider Protocol + Tavily impl
│   │   └── fetcher.py             # Concurrent fetch + trafilatura extraction
│   └── agent/
│       ├── state.py               # AgentState TypedDict
│       ├── prompts.py             # Versioned Jinja2 prompt templates
│       ├── nodes.py               # planner, gather_sources, synthesizer, critic, writer
│       └── graph.py               # LangGraph wiring + run_agent entrypoint
├── app/
│   └── streamlit_app.py           # Web UI with streaming progress
├── scripts/
│   ├── smoke_test.py              # End-to-end CLI run
│   └── export_graph.py            # Renders the graph to docs/images
├── tests/
│   ├── unit/                      # Domain, config, search, fetcher, nodes
│   └── integration/               # End-to-end graph with mocked deps
├── evals/
│   ├── dataset.yaml               # 20 research questions with expected topics
│   ├── metrics.py                 # citation_coverage, cost_usd, judge, p50/p95
│   └── runner.py                  # CLI runner with smoke and full modes
├── docs/images/                   # Architecture diagram (.mmd and .png)
├── .streamlit/config.toml         # Forces dark theme for the app
├── .env.example
├── pyproject.toml                 # uv, ruff (strict), mypy (strict)
└── README.md
```

## Testing

The repository ships with a 42-test suite that runs offline (`httpx.MockTransport` for the search and fetch paths, a `FakeLLMClient` for the agent flow). `pytest` finishes in under 4 seconds and the project gates on `coverage >= 80%`.

```bash
uv run pytest -v
uv run pytest --cov=src/deepresearch
```

Current coverage breakdown:

| Module | Coverage |
|---|---|
| `agent/graph.py` | 98% |
| `domain/models.py` | 96% |
| `infrastructure/search.py` | 93% |
| `infrastructure/fetcher.py` | 89% |
| `infrastructure/llm.py` | 51% (retry branches exercised in integration only) |
| **Total** | **90%** |

## Evaluation results

The agent is evaluated against a YAML dataset of 20 research questions across science, history, technology, health, and the arts. Each entry declares the topics a faithful report must cover. The runner executes the agent end-to-end against the real OpenAI and Tavily APIs and writes a timestamped JSON plus a Markdown table per run.

```bash
uv run python evals/runner.py --mode full --with-judge
```

The metrics computed per query:

- **citation_coverage** — fraction of non-trivial paragraphs that contain at least one `[N]` inline citation.
- **unique_citations** — distinct source indices referenced in the body.
- **topic_coverage** — LLM-as-judge score (`covered_topics / expected_topics`) using `gpt-5.4-mini` as the judge.
- **latency_seconds**, **prompt_tokens**, **completion_tokens**, **cost_usd** — captured from the OpenAI client's per-request usage.

### Latest run

`gpt-5.4-mini`, 20 queries, LLM judge enabled, 2026-06-08.

| Metric | Value |
|---|---|
| Successful queries | 20 / 20 |
| Mean citation coverage | **0.787** |
| Mean topic coverage (judge) | **0.925** |
| Latency p50 / p95 / mean | 36.9s / 100.0s / 46.0s |
| Total cost | $0.81 USD |

<details>
<summary>Per-query breakdown</summary>

| ID | Citation cov. | Topic cov. | Iterations | Sources | Latency (s) | Cost ($) |
|---|---|---|---|---|---|---|
| rag-advances | 0.824 | 1.00 | 3 | 8 | 42.4 | 0.064 |
| crispr-basics | 0.850 | 0.75 | 2 | 6 | 29.6 | 0.032 |
| apollo-11 | 0.889 | 1.00 | 2 | 6 | 26.7 | 0.024 |
| quantum-entanglement | 0.786 | 1.00 | 1 | 5 | 19.9 | 0.022 |
| climate-impacts | 0.800 | 1.00 | 3 | 12 | 190.3 | 0.065 |
| bitcoin-protocol | 0.885 | 0.75 | 3 | 9 | 43.5 | 0.059 |
| mediterranean-diet | 0.750 | 0.75 | 2 | 7 | 37.1 | 0.042 |
| photosynthesis | 0.818 | 1.00 | 1 | 7 | 26.0 | 0.015 |
| jwst-discoveries | 0.737 | 1.00 | 2 | 9 | 39.6 | 0.044 |
| ww2-causes | 0.692 | 1.00 | 2 | 8 | 34.0 | 0.047 |
| ml-bias | 0.882 | 1.00 | 1 | 5 | 23.8 | 0.024 |
| french-revolution | 0.867 | 1.00 | 2 | 7 | 36.6 | 0.040 |
| cancer-immunotherapy | 0.846 | 1.00 | 2 | 7 | 33.0 | 0.043 |
| internet-origins | 0.778 | 1.00 | 2 | 9 | 47.4 | 0.041 |
| mars-exploration | 0.625 | 1.00 | 2 | 8 | 38.6 | 0.046 |
| renewable-energy | 0.562 | 0.50 | 1 | 6 | 32.8 | 0.024 |
| antibiotic-resistance | 0.750 | 1.00 | 2 | 9 | 48.1 | 0.049 |
| renaissance | 0.818 | 1.00 | 1 | 4 | 30.7 | 0.021 |
| ev-adoption | 0.762 | 0.75 | 3 | 11 | 95.3 | 0.049 |
| sleep-science | 0.826 | 1.00 | 2 | 11 | 45.7 | 0.064 |

</details>

The `renewable-energy` query is the clearest outlier: the planner produced sub-questions whose web results were thin on the specific topics the judge expected, dragging both metrics down. This is the kind of failure the eval suite is meant to surface for follow-up.

## Roadmap

- Per-sub-question source filtering (BM25 or embedding similarity) to cut synthesizer token cost on long contexts
- LangSmith tracing for hosted observability
- Multi-provider LLM abstraction (Anthropic, OpenRouter, local Ollama)
- GitHub Actions workflow to run the smoke eval on every PR

## License

MIT. See [LICENSE](LICENSE).
