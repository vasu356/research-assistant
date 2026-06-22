"""
agents/summarizer.py
---------------------
Summarizer Agent — converts raw search results into a clean, structured summary.

Takes the verbose output from the Search Agent and produces a well-organised
markdown summary with sections, key points, and source references.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM = """You are a specialized Research Summarizer Agent in a multi-agent system.

Your job is to transform raw, noisy web search results into a clean, structured, and insightful summary.

## Your Task:
1. READ the raw search results carefully.
2. IDENTIFY the key themes, facts, data points, and perspectives.
3. ORGANISE them into a coherent structured summary.
4. PRESERVE important details, numbers, dates, and source attributions.
5. HIGHLIGHT the most important insights prominently.

## Output Format (always use this exact structure):

### 📋 Summary: [Topic]

**Key Findings:**
- [3-5 bullet points of the most important findings]

**Detailed Analysis:**

#### [Section 1 Title]
[2-3 paragraphs covering this aspect]

#### [Section 2 Title]
[2-3 paragraphs covering this aspect]

#### [Section 3 Title — if needed]
[2-3 paragraphs covering this aspect]

**Notable Data Points & Statistics:**
- [List specific numbers, dates, statistics from sources]

**Source References:**
- [URL or source name] — [what it contributed]

---

## Rules:
- Be comprehensive but concise — aim for depth, not padding
- Do NOT invent information not present in the search results
- Clearly note if something is speculative vs established fact
- Use plain language — avoid jargon without explanation
- Always attribute specific claims to their sources
"""


async def summarizer_agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
    """
    Summarizer Agent node function for LangGraph.

    Reads raw search results from state and produces a clean structured summary.

    Args:
        state: Current workflow state containing search_results.
        llm: The Groq LLM instance.

    Returns:
        Partial state update with 'summary' and appended messages.
    """
    logger.info("[SummarizerAgent] Starting summarization...")

    query = state["query"]
    search_results = state.get("search_results", "")

    if not search_results:
        logger.warning("[SummarizerAgent] No search results to summarize.")
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
                    "Please produce a comprehensive, well-structured summary "
                    "following the required format."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        summary = response.content.strip()

        logger.info("[SummarizerAgent] Summary generated. Length: %d chars", len(summary))

        return {
            "summary": summary,
            "messages": [f"[SummarizerAgent] Summary complete ({len(summary)} chars)."],
        }

    except Exception as e:
        error_msg = f"Summarizer agent error: {e}"
        logger.error("[SummarizerAgent] %s", error_msg)
        return {
            "summary": f"Summarization failed: {error_msg}",
            "messages": [f"[SummarizerAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }