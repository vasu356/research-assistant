"""
graph/workflow.py
-----------------
LangGraph workflow definition for the multi-agent research assistant.

Defines all graph nodes, conditional edges, and the compiled StateGraph.
The graph implements a supervisor-driven hierarchical delegation pattern:

                    [START]
                       │
                       ▼
              [supervisor_decision]  ◄──────────────────────────┐
              ┌─────────┼──────────┐                            │
              │         │          │                            │
              ▼         ▼          ▼                            │
       [search]  [summarize]  [fact_check]                      │
              │         │          │                            │
              └─────────┼──────────┘                            │
                       │                                        │
                       ▼ "reflect"                              │
                 [self-reflection] ── "NEEDS_IMPROVEMENT" ──────┘
                       │
                       ▼ "COMPLETE"
                     [END]
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from agents.fact_checker import fact_checker_agent_node
from agents.search_agent import search_agent_node
from agents.summarizer import summarizer_agent_node
from agents.supervisor import (
    route_after_decision,
    route_after_reflection,
    supervisor_decision_node,
    supervisor_reflect_node,
)
from config import settings
from graph.state import ResearchState

logger = logging.getLogger(__name__)


def build_workflow(llm: ChatGroq) -> Any:
    """
    Construct and compile the LangGraph StateGraph for the research assistant.

    Uses functools.partial to inject the shared LLM instance into each node
    function (LangGraph nodes must be single-argument callables accepting state).

    Args:
        llm: A configured ChatGroq instance shared across all agents.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    logger.info("[Workflow] Building research assistant graph...")

    graph = StateGraph(ResearchState)

    # Wrap each async node with the shared LLM via partial application
    supervisor_decide = functools.partial(supervisor_decision_node, llm=llm)
    supervisor_reflect = functools.partial(supervisor_reflect_node, llm=llm)
    search_node = functools.partial(search_agent_node, llm=llm)
    summarizer_node = functools.partial(summarizer_agent_node, llm=llm)
    fact_checker_node = functools.partial(fact_checker_agent_node, llm=llm)

    # Register nodes
    graph.add_node("supervisor_decision", supervisor_decide)
    graph.add_node("search_agent", search_node)
    graph.add_node("summarizer_agent", summarizer_node)
    graph.add_node("fact_checker_agent", fact_checker_node)
    graph.add_node("reflect", supervisor_reflect)

    # Entry point
    graph.set_entry_point("supervisor_decision")

    # Conditional edges from supervisor_decision
    graph.add_conditional_edges(
        "supervisor_decision",
        route_after_decision,
        {
            "search_agent": "search_agent",
            "summarizer_agent": "summarizer_agent",
            "fact_checker_agent": "fact_checker_agent",
            "reflect": "reflect",
            "__end__": END,
        },
    )

    # After each worker, return to supervisor for re-evaluation
    graph.add_edge("search_agent", "supervisor_decision")
    graph.add_edge("summarizer_agent", "supervisor_decision")
    graph.add_edge("fact_checker_agent", "supervisor_decision")

    # Conditional edge from reflect: COMPLETE → END, NEEDS_IMPROVEMENT → loop
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "search_agent": "search_agent",
            "__end__": END,
        },
    )

    compiled = graph.compile()
    logger.info("[Workflow] Graph compiled successfully.")
    return compiled


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
) -> ChatGroq:
    """
    Create a configured ChatGroq LLM instance from centralised settings.

    All parameters fall back to values from ``config.settings``, which in
    turn read from environment variables.  Pass explicit arguments only
    when you need to override for testing.

    Args:
        model:       Groq model name (default: settings.GROQ_MODEL).
        temperature: Sampling temperature (default: settings.GROQ_TEMPERATURE).
        max_retries: Retry count for failed requests (default: settings.GROQ_MAX_RETRIES).
        timeout:     Request timeout in seconds (default: settings.GROQ_TIMEOUT).

    Returns:
        A configured ChatGroq instance.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not configured.
    """
    settings.validate()

    resolved_model = model or settings.GROQ_MODEL
    resolved_temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
    resolved_retries = max_retries if max_retries is not None else settings.GROQ_MAX_RETRIES
    resolved_timeout = timeout if timeout is not None else settings.GROQ_TIMEOUT

    logger.info(
        "[Workflow] Initialising LLM: model=%s, temperature=%.2f, max_tokens=%d",
        resolved_model,
        resolved_temp,
        settings.GROQ_MAX_TOKENS,
    )

    return ChatGroq(
        model=resolved_model,
        temperature=resolved_temp,
        max_tokens=settings.GROQ_MAX_TOKENS,
        max_retries=resolved_retries,
        timeout=resolved_timeout,
    )
