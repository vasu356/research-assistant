"""
agents/search_agent.py
-----------------------
Web Search Agent — retrieves relevant, high-quality information from the web.

Implements the ReAct (Reason + Act) pattern:
  1. REASON  — The LLM analyses the query and plans targeted sub-searches.
  2. ACT     — It calls DuckDuckGo tools with focused queries.
  3. OBSERVE — It reviews tool results and decides if more searches are needed.
  4. DONE    — When satisfied, it stops the loop and returns consolidated results.

The agent autonomously decides how many searches to run (up to MAX_TOOL_ITERATIONS)
without any hard-coded query logic — all search strategy emerges from the LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from config import settings
from graph.state import ResearchState
from tools.search_tools import TOOLS_BY_NAME, duckduckgo_news_search, duckduckgo_search

logger = logging.getLogger(__name__)

SEARCH_AGENT_SYSTEM = """\
You are a specialized Web Search Agent in a multi-agent research system.

Your ONLY job is to search the web and retrieve relevant, high-quality information.

## Your Process (ReAct Pattern)
1. REASON  — Analyse the research query; identify 2–3 specific sub-questions.
2. ACT     — Call search tools with targeted queries for each sub-question.
3. OBSERVE — Review results; run additional searches if gaps remain.
4. SYNTHESIZE — Return ALL raw information as a single labelled block.

## Search Guidelines
- Break complex queries into focused sub-searches (e.g. "LLM reasoning 2025" + "chain-of-thought vs reasoning models benchmarks")
- Prefer recent sources — append the current year to time-sensitive queries
- Use both web search and news search for comprehensive coverage
- Return ALL raw information without heavy filtering; the Summarizer will structure it

## Output Format
Begin with "=== WEB SEARCH RESULTS ===" then list every retrieved item with source labels.
"""


async def search_agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
    """
    Web Search Agent node for LangGraph.

    Receives the research query from state, executes the ReAct tool-calling
    loop to gather information, and returns raw consolidated search results.

    The agent independently determines:
    - How many searches to run (up to ``settings.MAX_TOOL_ITERATIONS``)
    - What queries to formulate
    - When sufficient information has been gathered

    Args:
        state: Current workflow state (must contain ``query``).
        llm:   Shared ChatGroq instance injected by the workflow.

    Returns:
        Partial state dict with ``search_results`` (str) and appended ``messages``.
        On failure, also sets ``error``.
    """
    logger.info("[SearchAgent] Starting web search for query: %r", state["query"])

    query = state["query"]
    tools = [duckduckgo_search, duckduckgo_news_search]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SEARCH_AGENT_SYSTEM),
        HumanMessage(
            content=(
                f"Research Query: {query}\n\n"
                "Search for comprehensive information on this topic using multiple "
                "targeted sub-queries. Retrieve both general web results and recent news."
            )
        ),
    ]

    collected_results: list[str] = []

    try:
        for iteration in range(1, settings.MAX_TOOL_ITERATIONS + 1):
            logger.debug("[SearchAgent] Tool iteration %d/%d", iteration, settings.MAX_TOOL_ITERATIONS)

            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # LLM finished reasoning — capture any final synthesis text
                if response.content:
                    collected_results.append(str(response.content))
                break

            for tool_call in response.tool_calls:
                tool_name: str = tool_call.get("name", "unknown")
                tool_args: Dict[str, Any] = tool_call.get("args", {})
                tool_id: str = tool_call.get("id", "")

                logger.info("[SearchAgent] Tool call: %s(%s)", tool_name, tool_args)

                if tool_name in TOOLS_BY_NAME:
                    result = TOOLS_BY_NAME[tool_name].invoke(tool_args)
                else:
                    result = f"Unknown tool requested: {tool_name!r}"
                    logger.warning("[SearchAgent] %s", result)

                collected_results.append(f"[{tool_name}({tool_args})]:\n{result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        raw_results = "\n\n".join(collected_results) if collected_results else "No results retrieved."
        final_output = f"=== WEB SEARCH RESULTS ===\nQuery: {query}\n\n{raw_results}"

        logger.info(
            "[SearchAgent] Complete — %d result blocks, %d chars",
            len(collected_results),
            len(final_output),
        )

        return {
            "search_results": final_output,
            "messages": [
                f"[SearchAgent] Retrieved {len(collected_results)} result block(s) "
                f"across {iteration} iteration(s)."
            ],
        }

    except Exception as exc:
        error_msg = f"Search agent error: {exc}"
        logger.error("[SearchAgent] %s", error_msg, exc_info=True)
        return {
            "search_results": f"Search failed: {error_msg}",
            "messages": [f"[SearchAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }
