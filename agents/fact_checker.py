"""
agents/fact_checker.py
-----------------------
Fact Checker Agent — validates claims and flags uncertain information.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState

logger = logging.getLogger(__name__)

FACT_CHECKER_SYSTEM = """\
You are a specialized Fact Checker Agent in a multi-agent research system.

Your job is to critically evaluate the summary and flag issues.

## Evaluation Criteria
For each major claim, assess:
1. VERIFIABILITY — Can this be independently verified from the sources cited?
2. SPECIFICITY   — Is it a specific, falsifiable statement or a vague generalisation?
3. RECENCY       — Is the information current and relevant?
4. CONSISTENCY   — Do the claims contradict each other or well-known facts?
5. SOURCE QUALITY — Is the claim backed by credible, identifiable sources?

## Confidence Levels
- ✅ HIGH CONFIDENCE:  Specific, cited, verifiable, from credible sources
- ⚠️  MEDIUM CONFIDENCE: Plausible but vague, weakly sourced, or slightly dated
- ❌ LOW CONFIDENCE / FLAG: Unverified, potentially incorrect, or speculative

## Output Format

### 🔍 Fact-Check Report

**Overall Assessment:** [1–2 sentence overall verdict]

**Claim-by-Claim Analysis:**

| Claim | Confidence | Notes |
|-------|-----------|-------|
| [Claim 1] | ✅ HIGH | [Reason] |
| [Claim 2] | ⚠️ MEDIUM | [Reason] |
| [Claim 3] | ❌ FLAG | [Reason] |

**Flagged Issues:**
- [Issue 1]: [Explanation and recommendation]

**Well-Supported Claims:**
- [Claims that are solid and should be highlighted]

**Recommendations for Final Answer:**
- [Specific suggestions to improve accuracy]
- [Caveats that should be added]
"""


async def fact_checker_agent_node(state: ResearchState, llm: ChatGroq) -> dict[str, Any]:
    """
    Fact Checker Agent node for LangGraph.

    Args:
        state: Current workflow state (must contain ``summary``).
        llm:   Shared ChatGroq instance.

    Returns:
        Partial state dict with ``fact_check`` and an appended log message.
        On failure, also sets ``error``.
    """
    logger.info("[FactCheckerAgent] Starting fact-check...")

    query = state["query"]
    summary = state.get("summary", "")
    search_results = state.get("search_results", "")

    if not summary:
        logger.warning("[FactCheckerAgent] No summary available — skipping.")
        return {
            "fact_check": "No summary was available to fact-check.",
            "messages": ["[FactCheckerAgent] Skipped — no summary available."],
        }

    try:
        messages = [
            SystemMessage(content=FACT_CHECKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Original Research Query: {query}\n\n"
                    f"Summary to Fact-Check:\n\n{summary}\n\n"
                    f"Original Source Material (for reference):\n\n"
                    f"{search_results[:3000]}...\n\n"
                    "Perform a thorough fact-check using the required format."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        fact_check = str(response.content).strip()

        logger.info("[FactCheckerAgent] Fact-check complete — %d chars.", len(fact_check))

        return {
            "fact_check": fact_check,
            "messages": [f"[FactCheckerAgent] Fact-check complete ({len(fact_check)} chars)."],
        }

    except Exception as exc:
        error_msg = f"Fact checker agent error: {exc}"
        logger.error("[FactCheckerAgent] %s", error_msg, exc_info=True)
        return {
            "fact_check": f"Fact-check failed: {error_msg}",
            "messages": [f"[FactCheckerAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }
