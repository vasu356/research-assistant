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
from graph.state import ResearchState

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0


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

    # ------------------------------------------------------------------
    # 1. Initialise the StateGraph with our shared state schema
    # ------------------------------------------------------------------
    graph = StateGraph(ResearchState)

    # ------------------------------------------------------------------
    # 2. Wrap each async node with the shared LLM via partial application
    # ------------------------------------------------------------------
    supervisor_decide = functools.partial(supervisor_decision_node, llm=llm)
    supervisor_reflect = functools.partial(supervisor_reflect_node, llm=llm)
    search_node = functools.partial(search_agent_node, llm=llm)
    summarizer_node = functools.partial(summarizer_agent_node, llm=llm)
    fact_checker_node = functools.partial(fact_checker_agent_node, llm=llm)

    # ------------------------------------------------------------------
    # 3. Register nodes
    # ------------------------------------------------------------------
    graph.add_node("supervisor_decision", supervisor_decide)
    graph.add_node("search_agent", search_node)
    graph.add_node("summarizer_agent", summarizer_node)
    graph.add_node("fact_checker_agent", fact_checker_node)
    graph.add_node("reflect", supervisor_reflect)

    # ------------------------------------------------------------------
    # 4. Set entry point
    # ------------------------------------------------------------------
    graph.set_entry_point("supervisor_decision")

    # ------------------------------------------------------------------
    # 5. Conditional edges from supervisor_decision
    #    route_after_decision maps next_action → node name
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. After each worker, always return to supervisor for re-evaluation
    # ------------------------------------------------------------------
    graph.add_edge("search_agent", "supervisor_decision")
    graph.add_edge("summarizer_agent", "supervisor_decision")
    graph.add_edge("fact_checker_agent", "supervisor_decision")

    # ------------------------------------------------------------------
    # 7. Conditional edge from reflect:
    #    COMPLETE → END, NEEDS_IMPROVEMENT → search_agent (loop)
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "search_agent": "search_agent",
            "__end__": END,
        },
    )

    # ------------------------------------------------------------------
    # 8. Compile and return
    # ------------------------------------------------------------------
    compiled = graph.compile()
    logger.info("[Workflow] Graph compiled successfully.")
    return compiled


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> ChatGroq:
    """
    Create a configured ChatGroq LLM instance.

    Reads ``GROQ_API_KEY`` from environment (loaded via python-dotenv in
    :func:`main.main`).  Additional configuration can be supplied via
    environment variables:

    * ``GROQ_MODEL`` — override the default model name.
    * ``GROQ_TEMPERATURE`` — override the default temperature.

    Args:
        model: Groq model name (default: llama-3.3-70b-versatile).
        temperature: Sampling temperature (low = more deterministic).
        max_retries: Number of retries for failed requests (default: 3).
        timeout: Request timeout in seconds (default: 60.0).

    Returns:
        A configured ChatGroq instance.
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
        max_retries=max_retries,
        timeout=timeout,
    )