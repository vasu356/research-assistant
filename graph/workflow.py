"""
graph/workflow.py
-----------------
LangGraph workflow definition for the multi-agent research assistant.
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
from config import settings
from graph.state import ResearchState

logger = logging.getLogger(__name__)


def build_workflow(llm: ChatGroq) -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Args:
        llm: A configured ChatGroq instance shared across all agents.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    logger.info("[Workflow] Building research assistant graph...")

    graph = StateGraph(ResearchState)

    graph.add_node("supervisor_decision", functools.partial(supervisor_decision_node, llm=llm))
    graph.add_node("reflect", functools.partial(supervisor_reflect_node, llm=llm))
    graph.add_node("search_agent", functools.partial(search_agent_node, llm=llm))
    graph.add_node("summarizer_agent", functools.partial(summarizer_agent_node, llm=llm))
    graph.add_node("fact_checker_agent", functools.partial(fact_checker_agent_node, llm=llm))

    graph.set_entry_point("supervisor_decision")

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

    graph.add_edge("search_agent", "supervisor_decision")
    graph.add_edge("summarizer_agent", "supervisor_decision")
    graph.add_edge("fact_checker_agent", "supervisor_decision")

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

    Args:
        model:       Groq model name (default: settings.GROQ_MODEL).
        temperature: Sampling temperature (default: settings.GROQ_TEMPERATURE).
        max_retries: Retry count (default: settings.GROQ_MAX_RETRIES).
        timeout:     Request timeout in seconds (default: settings.GROQ_TIMEOUT).

    Returns:
        A configured ChatGroq instance.

    Raises:
        OSError: If GROQ_API_KEY is not configured.
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
