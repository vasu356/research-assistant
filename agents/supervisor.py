"""
agents/supervisor.py
---------------------
Supervisor Agent — orchestrates the multi-agent research workflow.

Implements two core agentic patterns:
  1. ReAct (Reason + Act): The supervisor reasons about the current
     state and decides which worker agent to dispatch next.
  2. Self-Reflection: After synthesising a final answer, the supervisor
     reviews its own output and decides whether another research pass is
     needed.

The supervisor uses hierarchical delegation — it never performs research
itself but coordinates the specialised worker agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import settings
from graph.state import ResearchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Action = Literal["search", "summarize", "fact_check", "reflect", "finish"]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SUPERVISOR_DECISION_SYSTEM = """\
You are the Supervisor Agent in a multi-agent research system.

You orchestrate specialised worker agents to answer research queries thoroughly and accurately.

## Worker Agents
- **search**     — Web Search Agent: retrieves fresh information from the internet
- **summarize**  — Summarizer Agent: structures raw search results into a clear summary
- **fact_check** — Fact Checker Agent: validates claims and flags uncertain information
- **reflect**    — Self-Reflection step: YOU write the draft answer and review it
- **finish**     — Return the final polished answer to the user

## Decision Process (ReAct Pattern)
1. REASON — Analyse what has been done so far and what is still needed.
2. ACT    — Choose the next agent OR decide to finish.

## Routing Rules
- Always start with "search" when search_results is missing.
- Move to "summarize" when search_results is present but summary is missing.
- Move to "fact_check" when summary is present but fact_check is missing.
- Move to "reflect" when all three components are present — write the first draft.
- After reflection: "finish" if quality is acceptable; "search" again if major gaps remain.
- ALWAYS "finish" when iteration_count >= {max_iterations} to prevent infinite loops.

## Response Format
Respond ONLY with valid JSON — no markdown fences, no preamble:
{{
  "reasoning": "Step-by-step analysis of the current state and what is needed",
  "next_action": "search|summarize|fact_check|reflect|finish",
  "rationale": "One sentence explaining this choice"
}}
""".format(max_iterations=settings.MAX_ITERATIONS)

SUPERVISOR_REFLECT_SYSTEM = """\
You are the Supervisor Agent performing a SELF-REFLECTION review.

All research components are assembled. Your job:
1. Write a comprehensive, accurate final answer to the original query.
2. Critically evaluate your draft.
3. Decide if it meets the bar or needs improvement.

## Available Components
- Original research query
- Web search results (raw)
- Structured summary
- Fact-check report with confidence scoring

## Output Format

### 🎯 Final Research Answer

**Executive Summary:**
[2–3 sentence overview of the key answer]

**Detailed Findings:**
[Comprehensive answer organised by theme, addressing all aspects of the query]

**Key Comparisons / Contrasts** (if applicable):
[Side-by-side comparison if the query asks for one]

**Confidence Assessment:**
[Overall reliability based on the fact-check results]

**Limitations & Caveats:**
[What the research could not fully address, or areas of uncertainty]

---

After the answer, add a REFLECTION block:

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

    Evaluates the current workflow state and decides which worker to dispatch
    next, or whether the workflow is complete.  Includes a hard iteration cap
    to prevent infinite loops, and a heuristic fallback for unparseable LLM
    responses.

    Args:
        state: Current workflow state.
        llm:   Shared ChatGroq instance.

    Returns:
        Partial state dict with ``next_action``, incremented ``iteration_count``,
        and an appended log message.
    """
    iteration = state["iteration_count"]
    logger.info("[Supervisor] Decision node — iteration %d", iteration)

    # Hard stop to prevent infinite loops
    if iteration >= settings.MAX_ITERATIONS:
        logger.warning("[Supervisor] Max iterations (%d) reached — forcing finish.", settings.MAX_ITERATIONS)
        return {
            "next_action": "finish",
            "iteration_count": iteration + 1,
            "messages": [f"[Supervisor] Max iterations ({settings.MAX_ITERATIONS}) reached. Forcing finish."],
        }

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

        # Strip optional markdown code fences
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
            "iteration_count": iteration + 1,
            "messages": [
                f"[Supervisor] Iteration {iteration + 1}: action={next_action} | {rationale}"
            ],
        }

    except (json.JSONDecodeError, Exception) as exc:
        logger.error("[Supervisor] Decision failed (%s) — using heuristic fallback.", exc)
        next_action = _heuristic_next_action(state)
        return {
            "next_action": next_action,
            "iteration_count": iteration + 1,
            "messages": [f"[Supervisor] Fallback decision: {next_action} (parse error: {exc})"],
        }


async def supervisor_reflect_node(
    state: ResearchState, llm: ChatGroq
) -> Dict[str, Any]:
    """
    Supervisor self-reflection node — synthesises all research into a final answer.

    Writes a polished final answer from all assembled research components, then
    performs a self-assessment to decide whether the answer is ready or whether
    another research pass is warranted.

    Args:
        state: Current workflow state (all research components should be populated).
        llm:   Shared ChatGroq instance.

    Returns:
        Partial state dict with ``final_answer``, ``reflection_notes``,
        updated ``next_action``, and an appended log message.
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
                    f"=== SUMMARY ===\n{state.get('summary') or 'None'}\n\n"
                    f"=== FACT-CHECK REPORT ===\n{state.get('fact_check') or 'None'}\n\n"
                    "Write the comprehensive final answer and perform self-reflection."
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        full_response = response.content.strip()

        # Split final answer from reflection block
        if "### 🔄 Self-Reflection" in full_response:
            parts = full_response.split("### 🔄 Self-Reflection", 1)
            final_answer = parts[0].strip()
            reflection_notes = "### 🔄 Self-Reflection\n" + parts[1].strip()
        else:
            final_answer = full_response
            reflection_notes = "Self-reflection block not found in response."

        # Parse verdict: loop back only if improvement is needed AND budget remains
        needs_improvement = (
            "NEEDS_IMPROVEMENT" in reflection_notes.upper()
            and state["iteration_count"] < settings.MAX_ITERATIONS
        )
        next_action: Action = "search" if needs_improvement else "finish"

        logger.info("[Supervisor] Reflection verdict → %s", next_action)

        return {
            "final_answer": final_answer,
            "reflection_notes": reflection_notes,
            "next_action": next_action,
            "messages": [
                f"[Supervisor] Reflection complete. "
                f"Verdict: {'NEEDS_IMPROVEMENT — re-searching' if needs_improvement else 'COMPLETE'}"
            ],
        }

    except Exception as exc:
        error_msg = f"Supervisor reflection error: {exc}"
        logger.error("[Supervisor] %s", error_msg, exc_info=True)
        return {
            "final_answer": state.get("summary", "Unable to generate final answer."),
            "reflection_notes": f"Reflection failed: {error_msg}",
            "next_action": "finish",
            "messages": [f"[Supervisor] Reflection ERROR: {error_msg}"],
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Routing functions (conditional edge callbacks for LangGraph)
# ---------------------------------------------------------------------------

def route_after_decision(state: ResearchState) -> str:
    """
    Conditional edge function invoked by LangGraph after the supervisor decision node.

    Maps the ``next_action`` state value to the corresponding graph node name.

    Args:
        state: Current workflow state.

    Returns:
        Node name string recognised by LangGraph.
    """
    action = state.get("next_action", "finish")
    mapping: Dict[str, str] = {
        "search":     "search_agent",
        "summarize":  "summarizer_agent",
        "fact_check": "fact_checker_agent",
        "reflect":    "reflect",
        "finish":     "__end__",
    }
    destination = mapping.get(action, "__end__")
    logger.info("[Router] next_action=%r → node=%r", action, destination)
    return destination


def route_after_reflection(state: ResearchState) -> str:
    """
    Conditional edge function invoked by LangGraph after the reflection node.

    Routes back to search if the self-reflection verdict is NEEDS_IMPROVEMENT
    and the iteration budget has not been exhausted.

    Args:
        state: Current workflow state.

    Returns:
        Node name string recognised by LangGraph.
    """
    action = state.get("next_action", "finish")
    if action == "search" and state["iteration_count"] < settings.MAX_ITERATIONS:
        logger.info("[Router] Reflection NEEDS_IMPROVEMENT → search_agent")
        return "search_agent"
    logger.info("[Router] Reflection COMPLETE → __end__")
    return "__end__"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_state_summary(state: ResearchState) -> str:
    """Produce a compact human-readable summary of which state fields are populated."""

    def _status(key: str) -> str:
        value = state.get(key)
        if not value:
            return "❌ missing"
        return f"✅ present ({len(value)} chars)"

    lines = [
        f"- search_results : {_status('search_results')}",
        f"- summary        : {_status('summary')}",
        f"- fact_check     : {_status('fact_check')}",
        f"- final_answer   : {'✅ present' if state.get('final_answer') else '❌ missing'}",
        f"- reflection     : {'✅ present' if state.get('reflection_notes') else '❌ missing'}",
        f"- iteration_count: {state['iteration_count']}",
    ]
    return "\n".join(lines)


def _heuristic_next_action(state: ResearchState) -> Action:
    """Deterministic fallback routing when the LLM response cannot be parsed."""
    if not state.get("search_results"):
        return "search"
    if not state.get("summary"):
        return "summarize"
    if not state.get("fact_check"):
        return "fact_check"
    if not state.get("final_answer"):
        return "reflect"
    return "finish"
