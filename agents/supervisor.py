"""
agents/supervisor.py
---------------------
Supervisor Agent — orchestrates the multi-agent research workflow.

Implements two core agentic patterns:
  1. **ReAct (Reason + Act)**: The supervisor reasons about the current
     state and decides which worker agent to dispatch next.
  2. **Self-Reflection**: After producing an initial final answer, the
     supervisor reviews it and decides whether another pass is needed.

The supervisor uses hierarchical delegation — it never does research itself
but coordinates the specialised worker agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from graph.state import ResearchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing action type
# ---------------------------------------------------------------------------

Action = Literal["search", "summarize", "fact_check", "reflect", "finish"]

MAX_ITERATIONS = 6

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUPERVISOR_DECISION_SYSTEM = """You are the Supervisor Agent in a multi-agent research system.

You orchestrate specialised worker agents to answer research queries thoroughly and accurately.

## Your Worker Agents:
- **search**: Web Search Agent — retrieves fresh information from the internet
- **summarize**: Summarizer Agent — structures raw search results into a clear summary
- **fact_check**: Fact Checker Agent — validates claims and flags uncertain information
- **reflect**: Self-Reflection step — YOU review the draft answer and improve it
- **finish**: Return the final polished answer to the user

## Your Decision Process (ReAct Pattern):
1. REASON: Analyse what has been done so far and what is still needed.
2. ACT: Choose the next agent to call OR decide to finish.

## Routing Rules:
- Always start with "search" if search_results is missing.
- Move to "summarize" once you have good search results but no summary.
- Move to "fact_check" once you have a summary but no fact-check.
- Move to "reflect" once you have all three components — write the first draft answer.
- After reflection, decide: "finish" if quality is acceptable, or "search" again if major gaps remain.
- ALWAYS "finish" if iteration_count >= 6 to prevent infinite loops.

## Response Format:
You MUST respond with ONLY valid JSON (no markdown, no explanation):
{
  "reasoning": "Your step-by-step analysis of the current state and what is needed",
  "next_action": "search|summarize|fact_check|reflect|finish",
  "rationale": "One sentence explaining why you chose this action"
}
"""

SUPERVISOR_REFLECT_SYSTEM = """You are the Supervisor Agent performing a SELF-REFLECTION review.

You have all research components assembled. Your job is to:
1. Write a comprehensive, accurate final answer to the original query.
2. Critically evaluate the draft answer you just wrote.
3. Decide if it meets the bar or needs improvement.

## Components Available to You:
- Original query
- Web search results
- Structured summary
- Fact-check report with flagged issues

## Your Output Format:

### 🎯 Final Research Answer

**Executive Summary:**
[2-3 sentence overview of the key answer]

**Detailed Findings:**
[Comprehensive answer addressing all aspects of the query, organised by theme]

**Key Comparisons / Contrasts** (if applicable):
[Side-by-side comparison of concepts if the query asks for comparison]

**Confidence Assessment:**
[Based on the fact-check, what is the overall reliability of this answer?]

**Limitations & Caveats:**
[What the research could not fully address, or areas of uncertainty]

---

After writing the answer, add a REFLECTION block:

### 🔄 Self-Reflection

**Quality Assessment:**
- Coverage: [Did I address all parts of the query?]
- Accuracy: [Are all claims well-supported?]
- Depth: [Is the analysis sufficiently deep?]

**Verdict:** [COMPLETE — answer is ready | NEEDS_IMPROVEMENT — specify what is missing]
"""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def supervisor_decision_node(
    state: ResearchState, llm: ChatGroq
) -> Dict[str, Any]:
    """
    Supervisor decision node — implements the ReAct reasoning loop.

    Evaluates the current state and decides which worker to dispatch next,
    or whether the workflow is complete.

    Args:
        state: Current workflow state.
        llm: The Groq LLM instance.

    Returns:
        Partial state update with 'next_action', incremented 'iteration_count',
        and appended messages.
    """
    logger.info(
        "[Supervisor] Decision node — iteration %d", state["iteration_count"]
    )

    # Hard stop to prevent infinite loops
    if state["iteration_count"] >= MAX_ITERATIONS:
        logger.warning("[Supervisor] Max iterations reached — forcing finish.")
        return {
            "next_action": "finish",
            "iteration_count": state["iteration_count"] + 1,
            "messages": ["[Supervisor] Max iterations reached. Forcing finish."],
        }

    # Build a state summary for the LLM to reason about
    state_summary = _build_state_summary(state)

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_DECISION_SYSTEM),
            HumanMessage(
                content=(
                    f"Research Query: {state['query']}\n\n"
                    f"Current State:\n{state_summary}\n\n"
                    "What is the next action?"
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        decision = json.loads(raw)
        next_action: Action = decision.get("next_action", "finish")
        rationale: str = decision.get("rationale", "")

        logger.info("[Supervisor] Decision: %s | %s", next_action, rationale)

        return {
            "next_action": next_action,
            "iteration_count": state["iteration_count"] + 1,
            "messages": [
                f"[Supervisor] Iteration {state['iteration_count'] + 1}: "
                f"Action={next_action} | {rationale}"
            ],
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error("[Supervisor] Decision failed: %s", e)
        # Fallback: determine action from state heuristically
        next_action = _heuristic_next_action(state)
        return {
            "next_action": next_action,
            "iteration_count": state["iteration_count"] + 1,
            "messages": [
                f"[Supervisor] Fallback decision: {next_action} (error: {e})"
            ],
        }


async def supervisor_reflect_node(
    state: ResearchState, llm: ChatGroq
) -> Dict[str, Any]:
    """
    Supervisor self-reflection node — writes the final answer and reviews it.

    Synthesises all research components into a polished final answer and
    performs a quality self-assessment to decide if the work is complete.

    Args:
        state: Current workflow state with all research components populated.
        llm: The Groq LLM instance.

    Returns:
        Partial state update with 'final_answer', 'reflection_notes',
        updated 'next_action', and appended messages.
    """
    logger.info("[Supervisor] Reflection node — synthesising final answer...")

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_REFLECT_SYSTEM),
            HumanMessage(
                content=(
                    f"Original Query: {state['query']}\n\n"
                    f"=== SEARCH RESULTS ===\n"
                    f"{(state.get('search_results') or '')[:2000]}\n\n"
                    f"=== SUMMARY ===\n{state.get('summary', 'None')}\n\n"
                    f"=== FACT-CHECK REPORT ===\n{state.get('fact_check', 'None')}\n\n"
                    "Please write the comprehensive final answer "
                    "and perform self-reflection."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        full_response = response.content.strip()

        # Split answer from reflection block
        if "### 🔄 Self-Reflection" in full_response:
            parts = full_response.split("### 🔄 Self-Reflection", 1)
            final_answer = parts[0].strip()
            reflection_notes = "### 🔄 Self-Reflection\n" + parts[1].strip()
        else:
            final_answer = full_response
            reflection_notes = "Self-reflection block not found in response."

        # Parse verdict from reflection to decide next action
        needs_improvement = (
            "NEEDS_IMPROVEMENT" in reflection_notes.upper()
            and state["iteration_count"] < MAX_ITERATIONS
        )
        next_action: Action = "search" if needs_improvement else "finish"

        logger.info("[Supervisor] Reflection complete. Verdict → %s", next_action)

        return {
            "final_answer": final_answer,
            "reflection_notes": reflection_notes,
            "next_action": next_action,
            "messages": [
                f"[Supervisor] Reflection complete. "
                f"Verdict: {'NEEDS_IMPROVEMENT — re-searching' if needs_improvement else 'COMPLETE'}"
            ],
        }

    except Exception as e:
        error_msg = f"Supervisor reflection error: {e}"
        logger.error("[Supervisor] %s", error_msg)
        return {
            "final_answer": state.get("summary", "Unable to generate final answer."),
            "reflection_notes": f"Reflection failed: {error_msg}",
            "next_action": "finish",
            "messages": [f"[Supervisor] Reflection ERROR: {error_msg}"],
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_state_summary(state: ResearchState) -> str:
    """Produce a compact text summary of what is present in the state."""
    lines = [
        "- search_results : "
        + (
            f"✅ present ({len(state['search_results'])} chars)"
            if state.get("search_results")
            else "❌ missing"
        ),
        "- summary        : "
        + (
            f"✅ present ({len(state['summary'])} chars)"
            if state.get("summary")
            else "❌ missing"
        ),
        "- fact_check     : "
        + (
            f"✅ present ({len(state['fact_check'])} chars)"
            if state.get("fact_check")
            else "❌ missing"
        ),
        "- final_answer   : "
        + ("✅ present" if state.get("final_answer") else "❌ missing"),
        "- reflection     : "
        + ("✅ present" if state.get("reflection_notes") else "❌ missing"),
        f"- iteration_count: {state['iteration_count']}",
    ]
    return "\n".join(lines)


def _heuristic_next_action(state: ResearchState) -> Action:
    """Fallback routing when the LLM decision cannot be parsed."""
    if not state.get("search_results"):
        return "search"
    if not state.get("summary"):
        return "summarize"
    if not state.get("fact_check"):
        return "fact_check"
    if not state.get("final_answer"):
        return "reflect"
    return "finish"


def route_after_decision(state: ResearchState) -> str:
    """
    Conditional edge function called by LangGraph after the supervisor decision node.

    Maps the 'next_action' value in state to the appropriate graph node name.

    Args:
        state: Current workflow state.

    Returns:
        Name of the next node to execute.
    """
    action = state.get("next_action", "finish")
    mapping: Dict[str, str] = {
        "search": "search_agent",
        "summarize": "summarizer_agent",
        "fact_check": "fact_checker_agent",
        "reflect": "reflect",
        "finish": "__end__",
    }
    destination = mapping.get(action, "__end__")
    logger.info("[Router] next_action='%s' → node='%s'", action, destination)
    return destination


def route_after_reflection(state: ResearchState) -> str:
    """
    Conditional edge function called by LangGraph after the reflection node.

    If the reflection verdict is NEEDS_IMPROVEMENT, routes back to search.
    Otherwise routes to the end.

    Args:
        state: Current workflow state.

    Returns:
        Name of the next node to execute.
    """
    action = state.get("next_action", "finish")
    if action == "search" and state["iteration_count"] < MAX_ITERATIONS:
        logger.info(
            "[Router] Reflection says NEEDS_IMPROVEMENT → re-routing to search_agent"
        )
        return "search_agent"
    logger.info("[Router] Reflection complete → __end__")
    return "__end__"