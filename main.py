"""
main.py
-------
Entry point for the Multi-Agent Research Assistant.

Usage:
    python main.py
    python main.py "Your custom research query here"
    LOG_LEVEL=DEBUG python main.py "Deep dive query"
"""

from __future__ import annotations

# NOTE: load_dotenv() must run before any config imports so env vars are
# available when settings.py is first evaluated. The E402 noqa comments
# silence ruff's "module-level import not at top" warning which is
# unavoidable here by design.
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402
from graph.state import ResearchState, initial_state  # noqa: E402
from graph.workflow import build_workflow, get_llm  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUERY = (
    "What are the latest developments in LLM reasoning models in 2025 "
    "and how do they compare to traditional chain-of-thought approaches?"
)
SEPARATOR = "=" * 80

# ---------------------------------------------------------------------------
# CLI progress indicator
# ---------------------------------------------------------------------------


class Spinner:
    """Async braille spinner shown during long-running pipeline steps."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Working") -> None:
        self.message = message
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _spin(self) -> None:
        i = 0
        while self._running:
            print(f"\r  {self.FRAMES[i]} {self.message}...", end="", flush=True)
            i = (i + 1) % len(self.FRAMES)
            await asyncio.sleep(0.1)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._spin())

    async def stop(self, final_message: str = "") -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None
        msg = final_message or "Done"
        print(f"\r  \u2713 {msg}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_section(title: str, content: str) -> None:
    """Pretty-print a labelled section to stdout."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(content)


def merge_state_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial LangGraph node update into the accumulated full state."""
    merged = dict(state)
    for key, value in update.items():
        if key == "messages" and value:
            merged.setdefault("messages", [])
            if isinstance(value, list):
                merged["messages"].extend(value)
            else:
                merged["messages"].append(value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def _sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert arbitrary text to a safe filename fragment."""
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in text)
    return safe.strip().replace(" ", "_")[:max_length]


def save_results_to_markdown(state: dict[str, Any], query: str) -> Path:
    """
    Persist all research outputs to a timestamped markdown file.

    Args:
        state: Final workflow state dict.
        query: Original research query (used in filename).

    Returns:
        Path to the written file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = _sanitize_filename(query)
    output_path = settings.OUTPUT_DIR / f"research_{timestamp}_{safe_query}.md"
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# Research Results\n",
        f"**Query:** {query}\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Model:** {settings.GROQ_MODEL}\n",
        "---\n",
        "## Search Results\n",
        state.get("search_results") or "N/A",
        "\n---\n",
        "## Structured Summary\n",
        state.get("summary") or "N/A",
        "\n---\n",
        "## Fact-Check Report\n",
        state.get("fact_check") or "N/A",
        "\n---\n",
        "## Self-Reflection Notes\n",
        state.get("reflection_notes") or "N/A",
        "\n---\n",
        "## Final Answer\n",
        state.get("final_answer") or "N/A",
        "\n---\n",
        "## Workflow Metadata\n",
        f"- **Iterations:** {state.get('iteration_count', 0)}",
        f"- **Agent messages:** {len(state.get('messages', []))}",
        f"- **Error:** {state.get('error') or 'None'}\n",
    ]

    output_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Results saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def run_research(query: str) -> dict[str, Any]:
    """
    Execute the full multi-agent research pipeline for a given query.

    Args:
        query: The research question to investigate.

    Returns:
        Final, fully-populated workflow state dict.
    """
    print(f"\n{'MULTI-AGENT RESEARCH ASSISTANT':^80}")
    print(SEPARATOR)
    print(f"Query : {query}")
    print(f"Model : {settings.GROQ_MODEL}")
    print(SEPARATOR)

    llm = get_llm()
    workflow = build_workflow(llm)
    state: ResearchState = initial_state(query)
    final_state: dict[str, Any] = dict(state)

    start_time = time.perf_counter()
    logger.info("Workflow execution started.")

    spinner = Spinner("Running pipeline")
    await spinner.start()

    async for chunk in workflow.astream(state, stream_mode="updates"):
        for _node_name, node_output in chunk.items():
            elapsed = time.perf_counter() - start_time
            for msg in node_output.get("messages", []):
                print(f"\r  [{elapsed:5.1f}s] {msg}")
            final_state = merge_state_update(final_state, node_output)

    elapsed_total = time.perf_counter() - start_time
    await spinner.stop(f"Pipeline complete in {elapsed_total:.1f}s")
    logger.info("Workflow complete in %.1fs", elapsed_total)

    return final_state


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def display_results(state: dict[str, Any], query: str) -> None:
    """
    Print all research outputs to stdout and save a markdown export.

    Args:
        state: Final workflow state dict.
        query: Original research query.
    """
    print(f"\n\n{'RESEARCH COMPLETE':^80}")

    search = state.get("search_results") or "N/A"
    preview = search[:500] + "\n... [truncated]" if len(search) > 500 else search
    print_section("SEARCH RESULTS (preview)", preview)
    print_section("STRUCTURED SUMMARY", state.get("summary") or "N/A")
    print_section("FACT-CHECK REPORT", state.get("fact_check") or "N/A")
    print_section("SELF-REFLECTION NOTES", state.get("reflection_notes") or "N/A")
    print_section("FINAL ANSWER", state.get("final_answer") or "N/A")
    print_section(
        "WORKFLOW METADATA",
        f"  Iterations    : {state.get('iteration_count', 0)}\n"
        f"  Agent messages: {len(state.get('messages', []))}\n"
        f"  Error         : {state.get('error') or 'None'}",
    )
    print_section("AGENT MESSAGE LOG", "\n".join(state.get("messages", [])))

    try:
        output_path = save_results_to_markdown(state, query)
        print(f"\n  \U0001f4c1 Results saved → {output_path}")
    except Exception as exc:
        logger.error("Failed to save markdown export: %s", exc)
        print(f"\n  \u26a0\ufe0f  Could not save results: {exc}")

    print(f"\n{SEPARATOR}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Async CLI entry point."""
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    try:
        final_state = await run_research(query)
        display_results(final_state, query)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except OSError as exc:
        print(f"\n\u274c CONFIG ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        print(f"\n\u274c ERROR: {exc}")
        print("Ensure GROQ_API_KEY is set in your .env file.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
