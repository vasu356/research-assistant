"""
graph/state.py
--------------
Shared state schema for the multi-agent research assistant workflow.

All agents read from and write to this state. LangGraph manages
propagation of state updates between nodes automatically.

Typical state progression through the workflow:
    query → search_results → summary → fact_check → final_answer
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """
    Central state object passed between all agents in the graph.

    Attributes:
        query: Original user research query.
        search_results: Raw text results from the web search agent.
        summary: Structured summary produced by the summarizer agent.
        fact_check: Fact-check report produced by the fact-checker agent.
        reflection_notes: Self-reflection notes from the supervisor's review pass.
        final_answer: The polished final answer ready to return to the user.
        next_action: Routing signal set by the supervisor
            ("search", "summarize", "fact_check", "reflect", "finish").
        iteration_count: Number of supervisor decision loops completed so far.
        messages: Accumulated log of agent messages (append-only across nodes).
        error: Optional error message if something goes wrong.
    """

    query: str
    search_results: str | None
    summary: str | None
    fact_check: str | None
    reflection_notes: str | None
    final_answer: str | None
    next_action: str | None
    iteration_count: int
    messages: Annotated[list[str], operator.add]
    error: str | None


def initial_state(query: str) -> ResearchState:
    """
    Build a clean initial state from a user query.

    Args:
        query: The research question posed by the user.

    Returns:
        A fully-initialised ResearchState with sensible defaults.
    """
    return ResearchState(
        query=query,
        search_results=None,
        summary=None,
        fact_check=None,
        reflection_notes=None,
        final_answer=None,
        next_action=None,
        iteration_count=0,
        messages=[f"[System] Research query received: {query}"],
        error=None,
    )
