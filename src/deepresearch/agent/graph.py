from __future__ import annotations

import time
from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deepresearch.agent.nodes import NodeDependencies, build_nodes
from deepresearch.agent.state import AgentState
from deepresearch.domain.exceptions import InvalidReportError
from deepresearch.domain.models import Query, Report

logger = structlog.get_logger(__name__)

AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def _should_continue(state: AgentState, max_iterations: int) -> Literal["planner", "writer"]:
    if state["critique"] is not None and state["iteration"] < max_iterations:
        return "planner"
    return "writer"


def build_graph(deps: NodeDependencies) -> AgentGraph:
    """Wire the research agent's nodes into a LangGraph state machine."""
    nodes = build_nodes(deps)
    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        # langgraph stubs reject async nodes returning partial dict updates; runtime accepts them.
        graph.add_node(name, fn)  # type: ignore[call-overload]
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "gather_sources")
    graph.add_edge("gather_sources", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    max_iterations = deps.settings.max_iterations

    def route(state: AgentState) -> Literal["planner", "writer"]:
        return _should_continue(state, max_iterations)

    graph.add_conditional_edges("critic", route, {"planner": "planner", "writer": "writer"})
    graph.add_edge("writer", END)
    return graph.compile()


async def run_agent(question: str, deps: NodeDependencies) -> Report:
    """Execute the agent end-to-end over one question and return the final report."""
    start = time.perf_counter()
    logger.info("agent.run.start", query=question)
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
    result: dict[str, Any] = await graph.ainvoke(initial_state)
    report = result.get("final_report")
    if not isinstance(report, Report):
        raise InvalidReportError("Graph completed without producing a final report")
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "agent.run.end",
        iterations=result.get("iteration", 0),
        source_count=len(result.get("sources", [])),
        total_tokens=deps.llm.total_tokens_used,
        latency_ms=round(latency_ms, 2),
    )
    return report


def export_graph_mermaid(graph: AgentGraph) -> str:
    return graph.get_graph().draw_mermaid()
