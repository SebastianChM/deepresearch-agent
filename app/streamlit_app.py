from __future__ import annotations

import asyncio
import hmac
import logging
import time
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from typing import Any

import httpx
import markdown as md_lib
import streamlit as st
import structlog
from xhtml2pdf import pisa

from deepresearch.agent.graph import build_graph
from deepresearch.agent.nodes import NodeDependencies
from deepresearch.agent.state import AgentState
from deepresearch.config import Settings, get_settings
from deepresearch.domain.exceptions import DeepResearchError, InvalidReportError
from deepresearch.domain.models import Query, Report
from deepresearch.infrastructure.llm import OpenAIClient
from deepresearch.infrastructure.search import TavilySearchProvider

_PLACEHOLDER_TOKEN = "replace-me"
_HTTP_CLIENT_TIMEOUT_SECONDS = 30.0
_SS_REPORT = "research_report"
_SS_LATENCY = "research_latency"
_SS_TOKENS = "research_tokens"
_SS_QUESTION = "research_question"
_PDF_TITLE = "Deep Research Report"
_PDF_CSS = """
@page { size: A4; margin: 2cm 2cm 2.5cm 2cm; }
body { font-family: Helvetica, Arial, sans-serif; color: #1f2937; line-height: 1.55; font-size: 10.5pt; }
h1 { color: #0f172a; font-size: 20pt; margin: 0 0 6pt 0; letter-spacing: -0.01em; }
h2 { color: #0f172a; font-size: 13pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 3pt; }
h3 { color: #1f2937; font-size: 11pt; margin: 12pt 0 4pt 0; }
a { color: #0891b2; text-decoration: none; }
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 8pt 18pt; }
li { margin-bottom: 3pt; }
.report-meta { color: #6b7280; font-size: 8.5pt; margin: 0 0 18pt 0; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 16pt 0; }
"""
_PIPELINE_NODES = ("planner", "gather_sources", "synthesizer", "critic", "writer")
_NODE_LABELS = {
    "planner": "Planner",
    "gather_sources": "Search & fetch",
    "synthesizer": "Synthesize",
    "critic": "Critic",
    "writer": "Writer",
}
_STATE_TEXT = {
    "pending": "pending",
    "running": "running",
    "done": "done",
    "skipped": "skipped",
}
_LINEAR_NEXT_NODE = {
    "planner": "gather_sources",
    "gather_sources": "synthesizer",
    "synthesizer": "critic",
}
_EXAMPLE_QUESTION = "What are the most recent advances in retrieval-augmented generation?"

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stTextArea, .stButton, .stMetric {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 12% -10%, rgba(6, 182, 212, 0.10) 0%, transparent 45%),
        radial-gradient(circle at 88% 110%, rgba(129, 140, 248, 0.08) 0%, transparent 45%),
        linear-gradient(180deg, #0a0e1a 0%, #0b1020 100%) !important;
}

.block-container {
    padding-top: 2.4rem !important;
    padding-bottom: 4rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1600px !important;
}

#MainMenu, footer, .stDeployButton, header[data-testid="stHeader"] { visibility: hidden; }

.hero { margin: 0 0 28px 0; }
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    color: #06b6d4;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.02;
    letter-spacing: -0.035em;
    color: #f8fafc;
    margin: 0 0 14px 0;
}
.hero-accent {
    background: linear-gradient(135deg, #06b6d4 0%, #818cf8 60%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-tagline {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.6;
    max-width: 620px;
    margin: 0;
}

.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin: 28px 0 12px 0;
}
.section-label.tight { margin-top: 8px; }

.stTextArea textarea {
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.96rem !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: rgba(6, 182, 212, 0.55) !important;
    box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.10) !important;
}
.stTextArea label { color: #94a3b8 !important; font-size: 0.78rem !important; }

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    border-radius: 10px !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
}
.stButton button[kind="primary"]:hover:not(:disabled) {
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 28px rgba(6, 182, 212, 0.28) !important;
}
.stButton button[kind="primary"]:disabled {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #475569 !important;
    cursor: not-allowed !important;
}

.iter-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    background: rgba(6, 182, 212, 0.10);
    border: 1px solid rgba(6, 182, 212, 0.28);
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #67e8f9;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 16px;
}
.iter-badge.dim {
    background: rgba(100, 116, 139, 0.08);
    border-color: rgba(100, 116, 139, 0.25);
    color: #94a3b8;
}
.iter-badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #06b6d4;
}
.iter-badge.dim .iter-badge-dot { background: #64748b; }

.node-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    transition: background 0.25s ease, border-color 0.25s ease;
}
.node-card.running {
    background: rgba(245, 158, 11, 0.08);
    border-color: rgba(245, 158, 11, 0.32);
}
.node-card.done {
    background: rgba(16, 185, 129, 0.05);
    border-color: rgba(16, 185, 129, 0.18);
}

.node-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.node-dot.pending { background: #475569; }
.node-dot.running {
    background: #f59e0b;
    box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.45);
    animation: pulse-amber 1.4s infinite;
}
.node-dot.done { background: #10b981; }
.node-dot.skipped { background: #475569; opacity: 0.45; }

@keyframes pulse-amber {
    0%   { box-shadow: 0 0 0 0   rgba(245, 158, 11, 0.45); }
    70%  { box-shadow: 0 0 0 9px rgba(245, 158, 11, 0); }
    100% { box-shadow: 0 0 0 0   rgba(245, 158, 11, 0); }
}

.node-name { flex: 1; font-weight: 500; color: #e2e8f0; font-size: 0.95rem; }
.node-status {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.node-card.running .node-status { color: #fbbf24; }
.node-card.done .node-status { color: #34d399; }

.metric-card {
    padding: 18px 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(6, 182, 212, 0.32);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
    letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
}
.metric-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 10px;
    font-weight: 600;
}

.config-list { display: flex; flex-direction: column; gap: 0; }
.config-entry {
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.config-entry:first-child { padding-top: 0; }
.config-entry:last-child { border-bottom: none; padding-bottom: 0; }
.config-entry-label {
    font-size: 0.62rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 4px;
}
.config-entry-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: #e2e8f0;
    font-weight: 500;
}

.stack-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 22px;
}
.stack-badge {
    padding: 5px 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}

.flow {
    display: flex;
    gap: 6px;
    align-items: stretch;
    margin-top: 18px;
}
.flow-stage {
    flex: 1;
    padding: 18px 16px;
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.flow-stage:hover {
    border-color: rgba(6, 182, 212, 0.30);
    transform: translateY(-2px);
}
.flow-stage-num {
    font-size: 0.62rem;
    font-weight: 700;
    color: #06b6d4;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 8px;
}
.flow-stage-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 8px;
    letter-spacing: -0.005em;
}
.flow-stage-desc {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.5;
}
.flow-arrow {
    display: flex;
    align-items: center;
    color: #334155;
    font-size: 1.6rem;
    font-weight: 300;
    padding: 0 2px;
    flex-shrink: 0;
}
.flow-stage.loop {
    border-color: rgba(245, 158, 11, 0.18);
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.04) 0%, rgba(255,255,255,0.01) 100%);
}
.flow-stage.loop .flow-stage-num { color: #f59e0b; }

.how-section { margin-top: 36px; }

.gate-card {
    max-width: 520px;
    margin: 64px auto 8px auto;
    padding: 32px 36px;
    background: linear-gradient(140deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
}
.gate-eyebrow {
    font-size: 0.66rem;
    font-weight: 700;
    color: #f59e0b;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 14px;
}
.gate-title {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
    color: #f8fafc;
    margin: 0 0 10px 0;
    letter-spacing: -0.025em;
}
.gate-subtitle {
    font-size: 0.94rem;
    color: #94a3b8;
    line-height: 1.6;
    margin: 0;
}

[data-testid="stCodeBlock"] {
    background: rgba(0, 0, 0, 0.28) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
}
[data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #cbd5e1 !important;
}

[data-testid="stMarkdownContainer"] h2 {
    color: #f1f5f9;
    font-weight: 700;
    letter-spacing: -0.015em;
    margin-top: 1.8em;
}
[data-testid="stMarkdownContainer"] h3 {
    color: #cbd5e1;
    font-weight: 600;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    color: #cbd5e1;
    line-height: 1.72;
}
[data-testid="stMarkdownContainer"] a { color: #67e8f9; text-decoration: none; border-bottom: 1px solid rgba(103, 232, 249, 0.3); }
[data-testid="stMarkdownContainer"] a:hover { border-bottom-color: #67e8f9; }
[data-testid="stMarkdownContainer"] hr { border-color: rgba(255, 255, 255, 0.08); }
</style>
"""

_HERO_HTML = """
<div class="hero">
    <div class="hero-eyebrow">LLM agent · LangGraph · OpenAI</div>
    <h1 class="hero-title">Deep <span class="hero-accent">Research</span> Agent</h1>
    <p class="hero-tagline">
        Planifica sub-preguntas, busca fuentes en la web, sintetiza con citas verificables
        y produce un reporte en Markdown listo para publicar.
    </p>
    <div class="stack-badges">
        <span class="stack-badge">Python 3.11+</span>
        <span class="stack-badge">LangGraph</span>
        <span class="stack-badge">OpenAI</span>
        <span class="stack-badge">Pydantic v2</span>
        <span class="stack-badge">Tavily</span>
        <span class="stack-badge">Streamlit</span>
    </div>
</div>
"""

_GATE_HTML = """
<div class="gate-card">
    <div class="gate-eyebrow">Restricted demo</div>
    <h1 class="gate-title">Access key required</h1>
    <p class="gate-subtitle">
        This live demo runs on a personal OpenAI budget. Enter the access key
        shared in your invitation to continue.
    </p>
</div>
"""

_HOW_IT_WORKS_HTML = """
<div class="how-section">
    <div class="section-label">Cómo funciona</div>
    <div class="flow">
        <div class="flow-stage">
            <div class="flow-stage-num">01 · Plan</div>
            <div class="flow-stage-name">Planner</div>
            <div class="flow-stage-desc">Descompone la pregunta en 3–6 sub-preguntas atómicas y verificables.</div>
        </div>
        <div class="flow-arrow">›</div>
        <div class="flow-stage">
            <div class="flow-stage-num">02 · Discover</div>
            <div class="flow-stage-name">Search &amp; Fetch</div>
            <div class="flow-stage-desc">Búsqueda paralela vía Tavily, descarga y extracción con trafilatura.</div>
        </div>
        <div class="flow-arrow">›</div>
        <div class="flow-stage">
            <div class="flow-stage-num">03 · Reason</div>
            <div class="flow-stage-name">Synthesize</div>
            <div class="flow-stage-desc">Un párrafo por sub-pregunta con citas inline a las fuentes reales.</div>
        </div>
        <div class="flow-arrow">›</div>
        <div class="flow-stage loop">
            <div class="flow-stage-num">04 · Verify ↻</div>
            <div class="flow-stage-name">Critic</div>
            <div class="flow-stage-desc">Detecta huecos y replanifica hasta cubrirlos o agotar iteraciones.</div>
        </div>
        <div class="flow-arrow">›</div>
        <div class="flow-stage">
            <div class="flow-stage-num">05 · Compose</div>
            <div class="flow-stage-name">Writer</div>
            <div class="flow-stage-desc">Ensambla el reporte en Markdown con secciones lógicas y bibliografía.</div>
        </div>
    </div>
</div>
"""


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def _enforce_access_gate(settings: Settings) -> None:
    if settings.app_access_password is None:
        return
    if st.session_state.get("_gate_passed"):
        return
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(_GATE_HTML, unsafe_allow_html=True)
        candidate = st.text_input(
            "Access key",
            type="password",
            label_visibility="collapsed",
            placeholder="Paste your access key",
            key="_gate_input",
        )
        if st.button("Unlock", type="primary", use_container_width=True):
            expected = settings.app_access_password.get_secret_value()
            if hmac.compare_digest(candidate, expected):
                st.session_state["_gate_passed"] = True
                st.rerun()
            else:
                st.error("That access key is not recognized.")
    st.stop()


def _ensure_keys_configured(settings: Settings) -> None:
    openai_key = settings.openai_api_key.get_secret_value()
    tavily_key = settings.tavily_api_key.get_secret_value()
    if _PLACEHOLDER_TOKEN in openai_key or _PLACEHOLDER_TOKEN in tavily_key:
        st.error(
            "Las API keys no están configuradas. Edita `.env` y reemplaza "
            "los valores placeholder de `OPENAI_API_KEY` y `TAVILY_API_KEY`."
        )
        st.stop()


def _initial_node_states() -> dict[str, str]:
    return {name: "pending" for name in _PIPELINE_NODES}


def _pipeline_html(node_states: dict[str, str], iteration: int) -> str:
    badge_class = "" if iteration > 0 else "dim"
    badge_text = f"Iteration {iteration}" if iteration > 0 else "Pipeline idle"
    badge = (
        f'<div class="iter-badge {badge_class}">'
        f'<span class="iter-badge-dot"></span>{badge_text}'
        "</div>"
    )
    cards = "".join(_node_card_html(name, node_states[name]) for name in _PIPELINE_NODES)
    return badge + cards


def _node_card_html(name: str, state: str) -> str:
    return (
        f'<div class="node-card {state}">'
        f'<span class="node-dot {state}"></span>'
        f'<span class="node-name">{_NODE_LABELS[name]}</span>'
        f'<span class="node-status">{_STATE_TEXT[state]}</span>'
        "</div>"
    )


def _render_pipeline(placeholder: Any, node_states: dict[str, str], iteration: int) -> None:
    placeholder.markdown(_pipeline_html(node_states, iteration), unsafe_allow_html=True)


def _render_progress(placeholder: Any, lines: list[str]) -> None:
    if lines:
        placeholder.code("\n".join(lines), language=None)


def _advance_after_event(node_states: dict[str, str], finished_node: str) -> None:
    node_states[finished_node] = "done"
    next_node = _LINEAR_NEXT_NODE.get(finished_node)
    if next_node is not None:
        node_states[next_node] = "running"


def _reset_loop_iteration(node_states: dict[str, str]) -> None:
    for name in ("gather_sources", "synthesizer", "critic"):
        node_states[name] = "pending"
    node_states["planner"] = "running"


async def _run_research(
    question: str,
    pipeline_placeholder: Any,
    progress_placeholder: Any,
) -> tuple[Report, float, int]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=_HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:
        llm = OpenAIClient(settings)
        search = TavilySearchProvider(api_key=settings.tavily_api_key, client=http_client)
        deps = NodeDependencies(llm=llm, search=search, http_client=http_client, settings=settings)
        report, latency = await _stream_graph(
            deps, question, pipeline_placeholder, progress_placeholder
        )
        return report, latency, llm.total_tokens_used


async def _stream_graph(
    deps: NodeDependencies,
    question: str,
    pipeline_placeholder: Any,
    progress_placeholder: Any,
) -> tuple[Report, float]:
    graph = build_graph(deps)
    initial_state: AgentState = {
        "query": Query(text=question),
        "sub_questions": [],
        "sources": [],
        "partial_summaries": {},
        "iteration": 0,
        "critique": None,
        "missing_topics": [],
        "final_report": None,
    }
    node_states = _initial_node_states()
    node_states["planner"] = "running"
    iteration = 1
    _render_pipeline(pipeline_placeholder, node_states, iteration)

    progress_lines: list[str] = []
    final_report: Report | None = None
    start = time.perf_counter()
    last_node_start = start

    async for event in graph.astream(initial_state, stream_mode="updates"):
        for finished_node, update in event.items():
            now = time.perf_counter()
            elapsed = now - last_node_start
            last_node_start = now

            if finished_node == "planner" and node_states["planner"] == "done":
                iteration += 1
                _reset_loop_iteration(node_states)

            _advance_after_event(node_states, finished_node)
            progress_lines.append(f"{_NODE_LABELS[finished_node]}: completed in {elapsed:.1f}s")
            _render_pipeline(pipeline_placeholder, node_states, iteration)
            _render_progress(progress_placeholder, progress_lines)

            if finished_node == "writer":
                final_report = update.get("final_report")

    latency_seconds = time.perf_counter() - start
    if not isinstance(final_report, Report):
        raise InvalidReportError("Graph completed without producing a final report")
    return final_report, latency_seconds


def _has_persisted_report() -> bool:
    return _SS_REPORT in st.session_state


def _persist_report(report: Report, latency: float, tokens: int, question: str) -> None:
    st.session_state[_SS_REPORT] = report
    st.session_state[_SS_LATENCY] = latency
    st.session_state[_SS_TOKENS] = tokens
    st.session_state[_SS_QUESTION] = question


def _clear_persisted_report() -> None:
    for key in (_SS_REPORT, _SS_LATENCY, _SS_TOKENS, _SS_QUESTION):
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _markdown_to_pdf_bytes(body_md: str, question: str) -> bytes:
    html_body = md_lib.markdown(body_md, extensions=["extra", "sane_lists"])
    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_PDF_CSS}</style></head><body>"
        f"<h1>{escape(_PDF_TITLE)}</h1>"
        f"<div class='report-meta'>Pregunta: {escape(question)}</div>"
        f"{html_body}"
        "</body></html>"
    )
    buffer = BytesIO()
    result = pisa.CreatePDF(src=full_html, dest=buffer, encoding="utf-8")
    if result.err:
        raise InvalidReportError(f"PDF generation produced {result.err} error(s)")
    return buffer.getvalue()


def _metric_card_html(label: str, value: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        "</div>"
    )


def _render_metrics(report: Report, latency_seconds: float, total_tokens: int) -> None:
    cards = [
        ("Iteraciones", str(report.iterations_used)),
        ("Fuentes", str(report.sources_consulted)),
        ("Tokens", f"{total_tokens:,}"),
        ("Latencia", f"{latency_seconds:.1f}s"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value) in zip(cols, cards, strict=True):
        col.markdown(_metric_card_html(label, value), unsafe_allow_html=True)


def _render_results(report: Report, latency_seconds: float, total_tokens: int, question: str) -> None:
    st.markdown('<div class="section-label">Resultados</div>', unsafe_allow_html=True)
    _render_metrics(report, latency_seconds, total_tokens)
    st.markdown('<div class="section-label">Reporte</div>', unsafe_allow_html=True)
    body = report.sections[0].body
    with st.container(border=True):
        st.markdown(body, unsafe_allow_html=False)
    _render_download_actions(body, question)


def _render_download_actions(body: str, question: str) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    pdf_bytes = _markdown_to_pdf_bytes(body, question)
    col_pdf, col_md, col_new = st.columns([1, 1, 1], gap="small")
    with col_pdf:
        st.download_button(
            label="Descargar PDF",
            data=pdf_bytes,
            file_name=f"research_{timestamp}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with col_md:
        st.download_button(
            label="Descargar Markdown",
            data=body,
            file_name=f"research_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_new:
        if st.button("Nueva investigación", use_container_width=True):
            _clear_persisted_report()
            st.rerun()


def _config_card_html(settings: Settings) -> str:
    rows = [
        ("Model", settings.openai_model),
        ("Max iterations", str(settings.max_iterations)),
        ("Results / query", str(settings.search_results_per_query)),
        ("Tokens / source", f"{settings.max_tokens_per_source:,}"),
        ("Fetch timeout", f"{settings.fetch_timeout_seconds:.0f}s"),
    ]
    entries = "".join(
        f'<div class="config-entry">'
        f'<div class="config-entry-label">{label}</div>'
        f'<div class="config-entry-value">{value}</div>'
        "</div>"
        for label, value in rows
    )
    return f'<div class="config-list">{entries}</div>'


def _render_sidebar(settings: Settings) -> None:
    st.markdown('<div class="section-label tight">Pipeline</div>', unsafe_allow_html=True)
    st.session_state.setdefault("_pipeline_placeholder", st.empty())
    _render_pipeline(st.session_state["_pipeline_placeholder"], _initial_node_states(), 0)
    st.markdown('<div class="section-label">Configuración</div>', unsafe_allow_html=True)
    st.markdown(_config_card_html(settings), unsafe_allow_html=True)


def main() -> None:
    _configure_logging()
    st.set_page_config(
        page_title="Deep Research Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    settings = get_settings()
    _ensure_keys_configured(settings)
    _enforce_access_gate(settings)

    left, right = st.columns([7, 3], gap="large")
    with right:
        _render_sidebar(settings)

    with left:
        st.markdown(_HERO_HTML, unsafe_allow_html=True)
        question = st.text_area(
            "Pregunta de investigación",
            height=110,
            placeholder=_EXAMPLE_QUESTION,
            label_visibility="visible",
        )
        disabled = not question.strip() or _has_persisted_report()
        run_clicked = st.button("Investigar", type="primary", disabled=disabled)
        progress_placeholder = st.empty()

        if run_clicked:
            pipeline_placeholder = st.session_state["_pipeline_placeholder"]
            try:
                report, latency, tokens = asyncio.run(
                    _run_research(question.strip(), pipeline_placeholder, progress_placeholder)
                )
            except DeepResearchError as exc:
                st.error(f"El agente falló: {exc}")
                return
            _persist_report(report, latency, tokens, question.strip())
            _render_results(report, latency, tokens, question.strip())
            return

        if _has_persisted_report():
            _render_results(
                st.session_state[_SS_REPORT],
                st.session_state[_SS_LATENCY],
                st.session_state[_SS_TOKENS],
                st.session_state[_SS_QUESTION],
            )
            return

        st.markdown(_HOW_IT_WORKS_HTML, unsafe_allow_html=True)


main()
