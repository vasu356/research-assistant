"""
tests/test_end_to_end.py
------------------------
Comprehensive end-to-end test suite for the Multi-Agent Research Assistant.

Tests every component without requiring network access or a real API key.
All LLM calls are intercepted by a mock that returns canned responses in
the exact sequence the workflow expects.

Run:
    python tests/test_end_to_end.py
"""

import sys
import asyncio
import json
import unittest
from unittest.mock import MagicMock

# Fix Unicode encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Make sure the project root is on the path
sys.path.insert(0, ".")

from langchain_core.messages import AIMessage

# ── Canned LLM responses ─────────────────────────────────────────────────────

CANNED_RESPONSES = [
    # 1. Supervisor: no results yet → search
    AIMessage(content=json.dumps({
        "reasoning": "search_results is missing; must gather information first.",
        "next_action": "search",
        "rationale": "Need to retrieve information before anything else."
    })),
    # 2. Search agent iteration 1: tool call (web search)
    AIMessage(
        content="",
        tool_calls=[{
            "name": "duckduckgo_search",
            "args": {"query": "LLM reasoning models 2025", "max_results": 4},
            "id": "call_001",
            "type": "tool_call",
        }]
    ),
    # 3. Search agent iteration 2: tool call (news search)
    AIMessage(
        content="",
        tool_calls=[{
            "name": "duckduckgo_news_search",
            "args": {"query": "chain-of-thought vs o3 DeepSeek-R1", "max_results": 4},
            "id": "call_002",
            "type": "tool_call",
        }]
    ),
    # 4. Search agent: done reasoning
    AIMessage(content=(
        "=== WEB SEARCH RESULTS ===\n"
        "In 2025, reasoning models such as OpenAI o3 and DeepSeek-R1 surpassed "
        "traditional chain-of-thought prompting by internalising multi-step reasoning."
    )),
    # 5. Supervisor: have search → summarize
    AIMessage(content=json.dumps({
        "reasoning": "search_results is present; summary is missing.",
        "next_action": "summarize",
        "rationale": "Raw results need to be structured into a clear summary."
    })),
    # 6. Summarizer
    AIMessage(content=(
        "### 📋 Summary: LLM Reasoning Models 2025\n\n"
        "**Key Findings:**\n"
        "- Reasoning models (o3, DeepSeek-R1) outperform standard CoT prompting\n"
        "- These models internalise deliberation rather than exposing it\n"
        "- Performance gains are largest on complex maths and coding tasks\n\n"
        "**Detailed Analysis:**\n\n"
        "#### Rise of Native Reasoning\n"
        "Unlike chain-of-thought prompting, 2025 reasoning models embed extended "
        "deliberation directly into the model architecture.\n\n"
        "#### Comparison with CoT\n"
        "Traditional CoT requires explicit step-by-step prompts; reasoning models "
        "achieve this automatically with higher reliability.\n\n"
        "**Source References:**\n"
        "- duckduckgo_search — background context\n"
        "- duckduckgo_news_search — 2025 updates"
    )),
    # 7. Supervisor: have summary → fact_check
    AIMessage(content=json.dumps({
        "reasoning": "summary present; fact_check missing.",
        "next_action": "fact_check",
        "rationale": "Validate claims before composing the final answer."
    })),
    # 8. Fact checker
    AIMessage(content=(
        "### 🔍 Fact-Check Report\n\n"
        "**Overall Assessment:** Claims are well-supported by the search results.\n\n"
        "**Claim-by-Claim Analysis:**\n\n"
        "| Claim | Confidence | Notes |\n"
        "|-------|-----------|-------|\n"
        "| Reasoning models outperform CoT | ✅ HIGH | Multiple sources confirm |\n"
        "| o3 and DeepSeek-R1 lead in 2025 | ✅ HIGH | Corroborated by news search |\n"
        "| Gains largest on maths/coding | ⚠️ MEDIUM | Plausible, weaker sourcing |\n\n"
        "**Recommendations:** Qualify the maths/coding claim with 'reportedly'."
    )),
    # 9. Supervisor: all three components → reflect
    AIMessage(content=json.dumps({
        "reasoning": "search_results, summary, and fact_check all present.",
        "next_action": "reflect",
        "rationale": "All components assembled; time to synthesise the final answer."
    })),
    # 10. Reflect node
    AIMessage(content=(
        "### 🎯 Final Research Answer\n\n"
        "**Executive Summary:**\n"
        "In 2025, LLM reasoning models such as OpenAI o3, DeepSeek-R1, and "
        "Google Gemini 2.0 Thinking represent a fundamental shift from "
        "prompting-based chain-of-thought techniques. These models embed "
        "deliberation internally rather than exposing it in the output stream.\n\n"
        "**Detailed Findings:**\n"
        "Traditional chain-of-thought prompting asks the model to 'think step by "
        "step', surfacing intermediate reasoning as text. Reasoning models instead "
        "run an internal 'thinking' process invisible to the user, producing more "
        "reliable answers, especially on complex problems.\n\n"
        "**Key Comparisons:**\n"
        "CoT prompting is explicit and tunable; reasoning models are automatic but "
        "less transparent. Both benefit from longer compute budgets.\n\n"
        "**Confidence Assessment:** HIGH — findings corroborated by multiple sources.\n\n"
        "**Limitations:** Rapidly evolving field; some benchmarks are contested.\n\n"
        "---\n\n"
        "### 🔄 Self-Reflection\n\n"
        "**Quality Assessment:**\n"
        "- Coverage: The answer addresses both 'latest developments' and the "
        "comparison with chain-of-thought.\n"
        "- Accuracy: All high-confidence claims retained; medium-confidence claim "
        "appropriately qualified.\n"
        "- Depth: Sufficient for a research summary.\n\n"
        "**Verdict:** COMPLETE — answer is ready"
    )),
]


def make_mock_llm() -> MagicMock:
    """
    Build a mock ChatGroq that replays CANNED_RESPONSES in order.
    Supports .ainvoke() and .bind_tools() (returns self).
    After the canned list is exhausted, returns a safe 'finish' response.
    """
    call_idx = [0]
    fallback = AIMessage(content=json.dumps({
        "reasoning": "All done.",
        "next_action": "finish",
        "rationale": "Workflow complete.",
    }))

    async def fake_ainvoke(messages, *args, **kwargs):
        i = call_idx[0]
        call_idx[0] += 1
        return CANNED_RESPONSES[i] if i < len(CANNED_RESPONSES) else fallback

    mock = MagicMock()
    mock.ainvoke = fake_ainvoke
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


def make_mock_tools() -> dict:
    """
    Build mock tool callables that return fixed strings without network access.
    """
    def web_result(args):
        return f"[MOCK WEB] Results for: {args.get('query', 'unknown')}"

    def news_result(args):
        return f"[MOCK NEWS] Articles for: {args.get('query', 'unknown')}"

    return {
        "duckduckgo_search": web_result,
        "duckduckgo_news_search": news_result,
    }


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestStateSchema(unittest.TestCase):
    """Tests for graph/state.py"""

    def test_initial_state_fields(self):
        from graph.state import initial_state
        state = initial_state("test query")
        self.assertEqual(state["query"], "test query")
        self.assertIsNone(state["search_results"])
        self.assertIsNone(state["summary"])
        self.assertIsNone(state["fact_check"])
        self.assertIsNone(state["reflection_notes"])
        self.assertIsNone(state["final_answer"])
        self.assertIsNone(state["next_action"])
        self.assertEqual(state["iteration_count"], 0)
        self.assertIsInstance(state["messages"], list)
        self.assertEqual(len(state["messages"]), 1)
        self.assertIn("test query", state["messages"][0])
        self.assertIsNone(state["error"])

    def test_initial_state_message_contains_query(self):
        from graph.state import initial_state
        state = initial_state("hello world")
        self.assertTrue(any("hello world" in m for m in state["messages"]))


class TestSearchTools(unittest.TestCase):
    """Tests for tools/search_tools.py"""

    def test_tools_registry_populated(self):
        from tools.search_tools import AVAILABLE_TOOLS, TOOLS_BY_NAME
        self.assertIn("duckduckgo_search", TOOLS_BY_NAME)
        self.assertIn("duckduckgo_news_search", TOOLS_BY_NAME)
        self.assertEqual(len(AVAILABLE_TOOLS), 2)

    def test_search_tool_handles_error_gracefully(self):
        """Tool must return a string even when network fails."""
        from tools.search_tools import duckduckgo_search
        result = duckduckgo_search.invoke({"query": "completely_unreachable_xyz_abc_123"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_news_tool_handles_error_gracefully(self):
        from tools.search_tools import duckduckgo_news_search
        result = duckduckgo_news_search.invoke({"query": "completely_unreachable_xyz_abc_123"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestSupervisorRouting(unittest.TestCase):
    """Tests for routing functions in agents/supervisor.py"""

    def test_route_after_decision_all_actions(self):
        from agents.supervisor import route_after_decision
        from graph.state import initial_state

        cases = {
            "search":      "search_agent",
            "summarize":   "summarizer_agent",
            "fact_check":  "fact_checker_agent",
            "reflect":     "reflect",
            "finish":      "__end__",
            "unknown_val": "__end__",
        }
        for action, expected in cases.items():
            state = initial_state("q")
            state["next_action"] = action
            result = route_after_decision(state)
            self.assertEqual(result, expected, f"action={action}")

    def test_route_after_reflection_complete(self):
        from agents.supervisor import route_after_reflection
        from graph.state import initial_state
        state = initial_state("q")
        state["next_action"] = "finish"
        state["iteration_count"] = 2
        self.assertEqual(route_after_reflection(state), "__end__")

    def test_route_after_reflection_needs_improvement_under_limit(self):
        from agents.supervisor import route_after_reflection
        from graph.state import initial_state
        state = initial_state("q")
        state["next_action"] = "search"
        state["iteration_count"] = 2   # below 6
        self.assertEqual(route_after_reflection(state), "search_agent")

    def test_route_after_reflection_maxed_out(self):
        from agents.supervisor import route_after_reflection
        from graph.state import initial_state
        state = initial_state("q")
        state["next_action"] = "search"
        state["iteration_count"] = 7   # above 6 — must stop
        self.assertEqual(route_after_reflection(state), "__end__")

    def test_heuristic_next_action_progression(self):
        from agents.supervisor import _heuristic_next_action
        from graph.state import initial_state

        s = initial_state("q")
        self.assertEqual(_heuristic_next_action(s), "search")

        s["search_results"] = "some results"
        self.assertEqual(_heuristic_next_action(s), "summarize")

        s["summary"] = "some summary"
        self.assertEqual(_heuristic_next_action(s), "fact_check")

        s["fact_check"] = "some fact check"
        self.assertEqual(_heuristic_next_action(s), "reflect")

        s["final_answer"] = "some answer"
        self.assertEqual(_heuristic_next_action(s), "finish")

    def test_build_state_summary_shows_missing(self):
        from agents.supervisor import _build_state_summary
        from graph.state import initial_state
        s = initial_state("q")
        summary = _build_state_summary(s)
        self.assertIn("missing", summary)

    def test_build_state_summary_shows_present(self):
        from agents.supervisor import _build_state_summary
        from graph.state import initial_state
        s = initial_state("q")
        s["search_results"] = "x" * 100
        summary = _build_state_summary(s)
        self.assertIn("present", summary)


class TestWorkflowGraph(unittest.TestCase):
    """Tests for graph/workflow.py — graph structure (no LLM calls)."""

    def setUp(self):
        import os
        os.environ["GROQ_API_KEY"] = "test_key_for_graph_build_only"
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=4096,
            groq_api_key="test_key_for_graph_build_only",
        )

    def test_graph_compiles(self):
        from graph.workflow import build_workflow
        wf = build_workflow(self.llm)
        self.assertIsNotNone(wf)

    def test_graph_has_all_nodes(self):
        from graph.workflow import build_workflow
        wf = build_workflow(self.llm)
        nodes = list(wf.get_graph().nodes.keys())
        for expected in [
            "__start__", "supervisor_decision", "search_agent",
            "summarizer_agent", "fact_checker_agent", "reflect", "__end__"
        ]:
            self.assertIn(expected, nodes, f"Missing node: {expected}")

    def test_graph_has_correct_edge_count(self):
        from graph.workflow import build_workflow
        wf = build_workflow(self.llm)
        edges = list(wf.get_graph().edges)
        self.assertGreaterEqual(len(edges), 8)


# ── Integration Tests (async) ─────────────────────────────────────────────────

class TestEndToEndWorkflow(unittest.IsolatedAsyncioTestCase):
    """
    Full end-to-end tests using mock LLM and mock tools.
    Validates the complete workflow from initial state to final answer.
    """

    async def asyncSetUp(self):
        """Set up mock tools before each test."""
        from tools import search_tools as st
        self._original_tools = {k: v for k, v in st.TOOLS_BY_NAME.items()}
        mock_fns = make_mock_tools()
        for name, fn in mock_fns.items():
            mock = MagicMock()
            mock.invoke = fn
            st.TOOLS_BY_NAME[name] = mock

    async def asyncTearDown(self):
        """Restore real tools after each test."""
        from tools import search_tools as st
        st.TOOLS_BY_NAME.update(self._original_tools)

    async def _run_workflow(self) -> dict:
        from graph.state import initial_state
        from graph.workflow import build_workflow
        state = initial_state(
            "What are the latest developments in LLM reasoning models in 2025 "
            "and how do they compare to traditional chain-of-thought approaches?"
        )
        wf = build_workflow(make_mock_llm())
        return await wf.ainvoke(state)

    async def test_all_state_fields_populated(self):
        """Every output field must be non-None after a full run."""
        final = await self._run_workflow()
        self.assertIsNotNone(final.get("search_results"),   "search_results")
        self.assertIsNotNone(final.get("summary"),          "summary")
        self.assertIsNotNone(final.get("fact_check"),       "fact_check")
        self.assertIsNotNone(final.get("final_answer"),     "final_answer")
        self.assertIsNotNone(final.get("reflection_notes"), "reflection_notes")

    async def test_no_errors(self):
        """The workflow must complete without errors."""
        final = await self._run_workflow()
        self.assertIsNone(final.get("error"), f"Unexpected error: {final.get('error')}")

    async def test_messages_accumulated(self):
        """All agent log messages must be present in the final state."""
        final = await self._run_workflow()
        msgs = final.get("messages", [])
        self.assertGreaterEqual(len(msgs), 6, f"Got only {len(msgs)} messages: {msgs}")
        combined = "\n".join(msgs)
        self.assertIn("SearchAgent",       combined, "Search agent message missing")
        self.assertIn("SummarizerAgent",   combined, "Summarizer message missing")
        self.assertIn("FactCheckerAgent",  combined, "Fact checker message missing")
        self.assertIn("Supervisor",        combined, "Supervisor message missing")

    async def test_iteration_count_incremented(self):
        """iteration_count must increase from 0."""
        final = await self._run_workflow()
        self.assertGreater(final.get("iteration_count", 0), 0)

    async def test_search_results_format(self):
        """search_results must start with the expected header."""
        final = await self._run_workflow()
        self.assertIn("WEB SEARCH RESULTS", final.get("search_results", ""))

    async def test_summary_has_key_sections(self):
        """Summary must contain Key Findings and Detailed Analysis sections."""
        final = await self._run_workflow()
        summary = final.get("summary", "")
        self.assertIn("Key Findings",     summary)
        self.assertIn("Detailed Analysis", summary)

    async def test_fact_check_has_table(self):
        """Fact-check report must contain a markdown table."""
        final = await self._run_workflow()
        fact_check = final.get("fact_check", "")
        self.assertIn("Confidence", fact_check)
        self.assertIn("|", fact_check)

    async def test_final_answer_has_executive_summary(self):
        """Final answer must contain the Executive Summary section."""
        final = await self._run_workflow()
        self.assertIn("Executive Summary", final.get("final_answer", ""))

    async def test_reflection_verdict_complete(self):
        """Reflection notes must contain the COMPLETE verdict."""
        final = await self._run_workflow()
        self.assertIn("COMPLETE", final.get("reflection_notes", ""))

    async def test_node_visit_order(self):
        """
        Nodes must be visited in the correct causal order:
        supervisor → search → supervisor → summarize → supervisor
        → fact_check → supervisor → reflect
        """
        from graph.state import initial_state
        from graph.workflow import build_workflow

        state      = initial_state("test ordering query")
        wf         = build_workflow(make_mock_llm())
        node_order = []

        async for chunk in wf.astream(state, stream_mode="updates"):
            node_order.extend(chunk.keys())

        def first(name):
            return next(i for i, n in enumerate(node_order) if n == name)

        self.assertIn("supervisor_decision", node_order)
        self.assertIn("search_agent",        node_order)
        self.assertIn("summarizer_agent",    node_order)
        self.assertIn("fact_checker_agent",  node_order)
        self.assertIn("reflect",             node_order)

        self.assertLess(first("supervisor_decision"), first("search_agent"))
        self.assertLess(first("search_agent"),        first("summarizer_agent"))
        self.assertLess(first("summarizer_agent"),    first("fact_checker_agent"))
        self.assertLess(first("fact_checker_agent"),  first("reflect"))

    async def test_max_iteration_guard(self):
        """
        If we send a mock LLM that always returns 'search',
        the supervisor must hard-stop at max iterations and not loop forever.
        """
        from graph.state import initial_state
        from graph.workflow import build_workflow

        # LLM that always says "search" — should be stopped by the guard
        infinite_mock = MagicMock()
        infinite_mock.bind_tools = MagicMock(return_value=infinite_mock)
        call_count = [0]

        async def always_search(msgs, *a, **kw):
            call_count[0] += 1
            # Return different things so search agent can complete
            # Alternate: supervisor decisions and search agent completions
            if call_count[0] % 2 == 1:
                # Supervisor call → always search
                return AIMessage(content=json.dumps({
                    "reasoning": "always search",
                    "next_action": "search",
                    "rationale": "keep searching"
                }))
            else:
                # Search agent call → no tool calls, return text
                return AIMessage(content="Mock search results for infinite loop test.")

        infinite_mock.ainvoke = always_search

        state = initial_state("infinite loop test")
        wf = build_workflow(infinite_mock)

        # This must terminate — if the guard is broken, it would run forever
        final = await asyncio.wait_for(wf.ainvoke(state), timeout=30.0)
        self.assertIsNotNone(final)
        # Must have been stopped by the guard
        self.assertGreaterEqual(final.get("iteration_count", 0), 6)


# ── Individual Agent Unit Tests ───────────────────────────────────────────────

class TestSearchAgentNode(unittest.IsolatedAsyncioTestCase):
    """Tests for agents/search_agent.py"""

    async def asyncSetUp(self):
        from tools import search_tools as st
        self._orig = dict(st.TOOLS_BY_NAME)
        for name in st.TOOLS_BY_NAME:
            mk = MagicMock()
            mk.invoke = lambda args, n=name: f"Mock result for {n}"
            st.TOOLS_BY_NAME[name] = mk

    async def asyncTearDown(self):
        from tools import search_tools as st
        st.TOOLS_BY_NAME.update(self._orig)

    async def test_search_agent_returns_results(self):
        from agents.search_agent import search_agent_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Here are the search results I found.")
        )

        state  = initial_state("test query for search")
        result = await search_agent_node(state, llm)

        self.assertIn("search_results", result)
        self.assertIsInstance(result["search_results"], str)
        self.assertIn("messages", result)

    async def test_search_agent_handles_llm_error(self):
        """Even when the LLM raises, the node must return a dict (not raise)."""
        from agents.search_agent import search_agent_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)

        async def raise_error(msgs, *a, **kw):
            raise RuntimeError("Simulated LLM failure")

        llm.ainvoke = raise_error
        state  = initial_state("error test")
        result = await search_agent_node(state, llm)

        self.assertIn("search_results", result)
        self.assertIn("error", result)
        self.assertIsNotNone(result["error"])


class TestSummarizerAgentNode(unittest.IsolatedAsyncioTestCase):
    """Tests for agents/summarizer.py"""

    async def test_summarizer_produces_summary(self):
        from agents.summarizer import summarizer_agent_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="### 📋 Summary\n**Key Findings:**\n- Finding 1\n**Detailed Analysis:**\nDetail here.")
        )

        state = initial_state("test")
        state["search_results"] = "Some raw search results about LLMs."
        result = await summarizer_agent_node(state, llm)

        self.assertIn("summary", result)
        self.assertIn("Key Findings", result["summary"])

    async def test_summarizer_skips_when_no_results(self):
        from agents.summarizer import summarizer_agent_node
        from graph.state import initial_state

        llm    = MagicMock()
        state  = initial_state("test")
        # search_results is None — summarizer should skip gracefully
        result = await summarizer_agent_node(state, llm)

        self.assertIn("summary", result)
        self.assertFalse(llm.ainvoke.called, "LLM should not be called with no input")

    async def test_summarizer_handles_llm_error(self):
        from agents.summarizer import summarizer_agent_node
        from graph.state import initial_state

        llm = MagicMock()
        async def raise_error(msgs, *a, **kw):
            raise RuntimeError("LLM down")
        llm.ainvoke = raise_error

        state = initial_state("test")
        state["search_results"] = "some results"
        result = await summarizer_agent_node(state, llm)

        self.assertIn("error", result)
        self.assertIsNotNone(result["error"])


class TestFactCheckerAgentNode(unittest.IsolatedAsyncioTestCase):
    """Tests for agents/fact_checker.py"""

    async def test_fact_checker_produces_report(self):
        from agents.fact_checker import fact_checker_agent_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="### 🔍 Fact-Check Report\n| Claim | ✅ HIGH | Supported |")
        )

        state = initial_state("test")
        state["summary"]        = "Some summary content."
        state["search_results"] = "Some raw results."
        result = await fact_checker_agent_node(state, llm)

        self.assertIn("fact_check", result)
        self.assertIn("Fact-Check", result["fact_check"])

    async def test_fact_checker_skips_when_no_summary(self):
        from agents.fact_checker import fact_checker_agent_node
        from graph.state import initial_state

        llm   = MagicMock()
        state = initial_state("test")
        result = await fact_checker_agent_node(state, llm)

        self.assertIn("fact_check", result)
        self.assertFalse(llm.ainvoke.called)


class TestSupervisorNodes(unittest.IsolatedAsyncioTestCase):
    """Tests for agents/supervisor.py decision and reflect nodes."""

    async def test_decision_node_parses_json_correctly(self):
        from agents.supervisor import supervisor_decision_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content=json.dumps({
                "reasoning": "search missing",
                "next_action": "search",
                "rationale": "need to search"
            }))
        )

        state  = initial_state("test")
        result = await supervisor_decision_node(state, llm)

        self.assertEqual(result["next_action"], "search")
        self.assertEqual(result["iteration_count"], 1)

    async def test_decision_node_hard_stop_at_max_iterations(self):
        from agents.supervisor import supervisor_decision_node
        from graph.state import initial_state

        llm   = MagicMock()  # should not be called
        state = initial_state("test")
        state["iteration_count"] = 6   # at max

        result = await supervisor_decision_node(state, llm)
        self.assertEqual(result["next_action"], "finish")
        self.assertFalse(llm.ainvoke.called, "LLM must not be called at max iterations")

    async def test_decision_node_falls_back_on_bad_json(self):
        from agents.supervisor import supervisor_decision_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="this is not json at all {{{}}")
        )

        state  = initial_state("test")
        result = await supervisor_decision_node(state, llm)

        # Must still return a valid next_action via heuristic
        self.assertIn(result["next_action"], ["search", "summarize", "fact_check", "reflect", "finish"])

    async def test_reflect_node_splits_answer_from_reflection(self):
        from agents.supervisor import supervisor_reflect_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content=(
                "### 🎯 Final Research Answer\n\nThe answer is X.\n\n---\n\n"
                "### 🔄 Self-Reflection\n\n**Verdict:** COMPLETE — answer is ready"
            ))
        )

        state                  = initial_state("test")
        state["search_results"] = "results"
        state["summary"]        = "summary"
        state["fact_check"]     = "fact check"

        result = await supervisor_reflect_node(state, llm)

        self.assertIn("final_answer",     result)
        self.assertIn("reflection_notes", result)
        self.assertIn("Final Research Answer", result["final_answer"])
        self.assertIn("Self-Reflection",       result["reflection_notes"])
        self.assertEqual(result["next_action"], "finish")

    async def test_reflect_node_handles_missing_reflection_block(self):
        from agents.supervisor import supervisor_reflect_node
        from graph.state import initial_state

        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Just the answer, no reflection block.")
        )

        state = initial_state("test")
        state["search_results"] = "r"
        state["summary"]        = "s"
        state["fact_check"]     = "f"

        result = await supervisor_reflect_node(state, llm)
        self.assertIn("final_answer",     result)
        self.assertIn("reflection_notes", result)
        self.assertEqual(result["next_action"], "finish")


# ── Helper: AsyncMock for Python 3.7 compatibility ───────────────────────────

class AsyncMock(MagicMock):
    """Simple AsyncMock for environments without unittest.mock.AsyncMock."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🧪  MULTI-AGENT RESEARCH ASSISTANT — FULL TEST SUITE")
    print("=" * 70 + "\n")

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestStateSchema,
        TestSearchTools,
        TestSupervisorRouting,
        TestWorkflowGraph,
        TestSearchAgentNode,
        TestSummarizerAgentNode,
        TestFactCheckerAgentNode,
        TestSupervisorNodes,
        TestEndToEndWorkflow,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"  🎉  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  ❌  {len(result.failures)} failure(s), {len(result.errors)} error(s) "
              f"out of {result.testsRun} tests")
    print("=" * 70 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
