"""
agents/summarizer.py
---------------------
Summarizer Agent — converts raw search results into a structured markdown summary.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM = """\
You are a specialized Research Summarizer Agent in a multi-agent system.

Your job is to transform raw, noisy web search results into a clean, structured, insightful summary.

## Task
1. READ the raw search results carefully.
2. IDENTIFY key themes, facts, data points, and perspectives.
3. ORGANISE them into a coherent structure.
4. PRESERVE important details: numbers, dates, names, and source attributions.
5. HIGHLIGHT the most important insights.

## Output Format

### 📋 Summary: [Topic]

**Key Findings:**
- [3–5 bullet points of the most important findings]

**Detailed Analysis:**

#### [Section 1 Title]
[2–3 paragraphs on this aspect]

#### [Section 2 Title]
[2–3 paragraphs on this aspect]

**Notable Data Points & Statistics:**
- [List specific numbers, dates, statistics from sources]

**Source References:**
- [URL or source name] — [what it contributed]

## Rules
- Be comprehensive but concise
- Do NOT invent information absent from the search results
- Always attribute specific claims to their sources
"""


async def summarizer_agent_node(state: ResearchState, llm: ChatGroq) -> dict[str, Any]:
    """
    Summarizer Agent node for LangGraph.

    Args:
        state: Current workflow state (must contain ``search_results``).
        llm:   Shared ChatGroq instance.

    Returns:
        Partial state dict with ``summary`` and an appended log message.
        On failure, also sets ``error``.
    """
    logger.info("[SummarizerAgent] Starting summarization...")

    query = state["query"]
    search_results = state.get("search_results", "")

    if not search_results:
        logger.warning("[SummarizerAgent] No search results available — skipping.")
        return {
            "summary": "No search results were available to summarize.",
            "messages": ["[SummarizerAgent] Skipped — no search results available."],
        }

    try:
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM),
            HumanMessage(
                content=(
                    f"Original Research Query: {query}\n\n"
                    f"Raw Search Results to Summarize:\n\n{search_results}\n\n"
                    "Produce a comprehensive, well-structured summary following the required format."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        summary = str(response.content).strip()

        logger.info("[SummarizerAgent] Summary generated — %d chars.", len(summary))

        return {
            "summary": summary,
            "messages": [f"[SummarizerAgent] Summary complete ({len(summary)} chars)."],
        }

    except Exception as exc:
        error_msg = f"Summarizer agent error: {exc}"
        logger.error("[SummarizerAgent] %s", error_msg, exc_info=True)
        return {
            "summary": f"Summarization failed: {error_msg}",
            "messages": [f"[SummarizerAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }
