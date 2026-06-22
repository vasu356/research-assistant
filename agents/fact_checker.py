"""
agents/fact_checker.py
-----------------------
Fact Checker Agent — validates claims and flags uncertain information.

Analyses the summary produced by the Summarizer Agent, classifies each
major claim by confidence level, and produces a structured fact-check report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState

logger = logging.getLogger(__name__)

FACT_CHECKER_SYSTEM = """You are a specialized Fact Checker Agent in a multi-agent research system.

Your job is to critically evaluate the summary produced by the Summarizer Agent and flag any issues.

## Your Evaluation Criteria:

**For each major claim or assertion, assess:**
1. VERIFIABILITY — Can this be independently verified from the sources cited?
2. SPECIFICITY — Is it a specific, falsifiable statement or vague generalisation?
3. RECENCY — Is the information current and relevant to the query timeframe?
4. CONSISTENCY — Do the claims contradict each other or known facts?
5. SOURCE QUALITY — Is the claim backed by credible sources?

## Confidence Levels:
- ✅ HIGH CONFIDENCE: Specific, cited, verifiable, from credible sources
- ⚠️  MEDIUM CONFIDENCE: Plausible but vague, weakly sourced, or slightly dated
- ❌ LOW CONFIDENCE / FLAG: Unverified, potentially incorrect, contradictory, or speculative

## Output Format:

### 🔍 Fact-Check Report

**Overall Assessment:** [1-2 sentence overall verdict]

**Claim-by-Claim Analysis:**

| Claim | Confidence | Notes |
|-------|-----------|-------|
| [Claim 1] | ✅ HIGH | [Reason] |
| [Claim 2] | ⚠️ MEDIUM | [Reason] |
| [Claim 3] | ❌ FLAG | [Reason] |

**Flagged Issues:**
- [Issue 1]: [Explanation and recommendation]
- [Issue 2]: [Explanation and recommendation]

**Well-Supported Claims:**
- [List claims that are solid and should be highlighted]

**Recommendations for Final Answer:**
- [Specific suggestions to improve accuracy and reliability]
- [Any caveats that should be added to the final answer]

---

## Important Rules:
- Be rigorous but fair — do NOT over-flag reasonable claims
- Distinguish between "unverified by these sources" vs "likely false"
- Focus on factual claims, not writing quality
- Note if key aspects of the query were NOT addressed by the summary
"""


async def fact_checker_agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
    """
    Fact Checker Agent node function for LangGraph.

    Reviews the summary for accuracy, flags uncertain claims, and produces
    a structured fact-check report.

    Args:
        state: Current workflow state containing query, search_results, and summary.
        llm: The Groq LLM instance.

    Returns:
        Partial state update with 'fact_check' and appended messages.
    """
    logger.info("[FactCheckerAgent] Starting fact-check...")

    query = state["query"]
    summary = state.get("summary", "")
    search_results = state.get("search_results", "")

    if not summary:
        logger.warning("[FactCheckerAgent] No summary to fact-check.")
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
                    "Please perform a thorough fact-check of the summary "
                    "using the required format."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        fact_check = response.content.strip()

        logger.info(
            "[FactCheckerAgent] Fact-check complete. Length: %d chars",
            len(fact_check),
        )

        return {
            "fact_check": fact_check,
            "messages": [
                f"[FactCheckerAgent] Fact-check complete ({len(fact_check)} chars)."
            ],
        }

    except Exception as e:
        error_msg = f"Fact checker agent error: {e}"
        logger.error("[FactCheckerAgent] %s", error_msg)
        return {
            "fact_check": f"Fact-check failed: {error_msg}",
            "messages": [f"[FactCheckerAgent] ERROR: {error_msg}"],
            "error": error_msg,
        }