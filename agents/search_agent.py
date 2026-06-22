"""
agents/search_agent.py
-----------------------
Web Search Agent — searches the web for relevant information.

Uses DuckDuckGo (no API key required) via LangChain tool-calling.
Implements the ReAct pattern: the LLM reasons about what to search for,
then calls the search tool, then reasons about whether results are sufficient.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState
from tools.search_tools import TOOLS_BY_NAME, duckduckgo_search, duckduckgo_news_search

logger = logging.getLogger(__name__)

# System prompt for the Search Agent
SEARCH_AGENT_SYSTEM = """You are a specialized Web Search Agent in a multi-agent research system.

Your ONLY job is to search the web and retrieve relevant, high-quality information.

## Your Process (ReAct Pattern):
1. REASON: Analyse the research query and identify 2-3 specific sub-questions to answer.
2. ACT: Call the search tools with targeted queries for each sub-question.
3. OBSERVE: Review the results and decide if you need additional searches.
4. SYNTHESIZE: Combine all raw results into a single coherent information block.

## Guidelines:
- Break complex queries into focused sub-searches (e.g., "LLM reasoning 2025" + "chain-of-thought vs reasoning models")
- Prefer recent sources — include the year in queries when looking for current info
- Include both general web results and news results for comprehensive coverage
- Return ALL raw information — do NOT summarise or filter heavily; the Summarizer will do that
- Clearly label each result source

## Output Format:
Start with "=== WEB SEARCH RESULTS ===" then list all retrieved information with source labels.
"""

MAX_TOOL_ITERATIONS = 4


async def search_agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
    """
    Web Search Agent node function for LangGraph.

    Receives the research query from state, uses tool-calling to search
    the web, and returns raw search results.

    The agent follows the ReAct pattern, autonomously deciding:
    * How many searches to run
    * What queries to formulate
    * When sufficient information has been gathered

    Args:
        state: Current workflow state containing the query.
        llm: The Groq LLM instance (passed in from workflow).

    Returns:
        Partial state update with 'search_results' and appended messages.
    """
    logger.info("[SearchAgent] Starting web search...")

    query = state["query"]

    # Bind both search tools to the LLM
    tools = [duckduckgo_search, duckduckgo_news_search]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SEARCH_AGENT_SYSTEM),
        HumanMessage(
            content=(
                f"Research Query: {query}\n\n"
                "Please search for comprehensive information on this topic. "
                "Use multiple targeted searches to cover different aspects. "
                "Retrieve both general web results and recent news."
            )
        ),
    ]

    collected_results: list[str] = []
    iteration = 0

    try:
        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            logger.debug("[SearchAgent] Tool iteration %d/%d", iteration, MAX_TOOL_ITERATIONS)

            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # Check if the LLM wants to call tools
            if not response.tool_calls:
                # No more tool calls — LLM is done reasoning
                if response.content:
                    collected_results.append(str(response.content))
                break

            # Execute each requested tool call
            for tool_call in response.tool_calls:
                tool_name: str = tool_call.get("name", "unknown")
                tool_args: Dict[str, Any] = tool_call.get("args", {})
                tool_id: str = tool_call.get("id", "")

                logger.info(
                    "[SearchAgent] Calling tool '%s' with args: %s",
                    tool_name,
                    tool_args,
                )

                if tool_name in TOOLS_BY_NAME:
                    result = TOOLS_BY_NAME[tool_name].invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                collected_results.append(f"[{tool_name}({tool_args})]:\n{result}")

                # Feed tool result back into message history
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

        # Assemble final raw results blob
        raw_results = "\n\n".join(collected_results) if collected_results else "No results retrieved."
        final_output = f"=== WEB SEARCH RESULTS ===\nQuery: {query}\n\n{raw_results}"

        logger.info(
            "[SearchAgent] Search complete. Result length: %d chars, iterations: %d",
            len(final_output),
            iteration,
        )

        return {
            "search_results": final_output,
            "messages": [
                f"[SearchAgent] Completed {iteration} tool iteration(s). "
                f"Retrieved {len(collected_results)} result block(s)."
            ],
        }

    except Exception as e:
        error_msg = f"Search agent error: {e}"
        logger.error("[SearchAgent] %s", error_msg)
        return {
            "search_results": f"Search failed: {error_msg}",
            "messages": [f"[SearchAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }