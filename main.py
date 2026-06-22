"""
main.py
-------
Entry point for the Multi-Agent Research Assistant.

Orchestrates a supervisor-driven multi-agent workflow that:
1. Searches the web for relevant information (Search Agent)
2. Structures raw results into a coherent summary (Summarizer Agent)
3. Validates claims and flags uncertainties (Fact Checker Agent)
4. Synthesises a final answer with self-reflection (Supervisor Agent)

Usage:
    python main.py
    python main.py "Your custom research query here"
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load .env before any LangChain/Groq imports that read env vars
load_dotenv()

from graph.state import ResearchState, initial_state
from graph.workflow import build_workflow, get_llm

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
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
OUTPUT_DIR = Path("research_outputs")

# ---------------------------------------------------------------------------
# Spinner / Progress Indicator
# ---------------------------------------------------------------------------


class Spinner:
    """Simple async spinner for progress indication during long operations."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Working") -> None:
        self.message = message
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _spin(self) -> None:
        """Animate spinner frames in-place."""
        i = 0
        while self._running:
            print(f"\r  {self.FRAMES[i]} {self.message}...", end="", flush=True)
            i = (i + 1) % len(self.FRAMES)
            await asyncio.sleep(0.1)

    async def start(self) -> None:
        """Start the spinner."""
        self._running = True
        self._task = asyncio.create_task(self._spin())

    async def stop(self, final_message: str = "") -> None:
        """Stop the spinner and print final message."""
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None
        if final_message:
            print(f"\r  \u2713 {final_message}")
        else:
            print("\r  \u2713 Done")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_section(title: str, content: str) -> None:
    """Pretty-print a labelled section to stdout."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(content)


def merge_state_update(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a partial LangGraph update into the accumulated state."""
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


def save_results_to_markdown(state: Dict[str, Any], query: str) -> Path:
    """
    Save research results to a timestamped markdown file.

    Args:
        state: Final workflow state.
        query: Original research query.

    Returns:
        Path to the saved markdown file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = _sanitize_filename(query)
    filename = f"research_{timestamp}_{safe_query}.md"
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    sections: List[str] = [
        "# Research Results\n",
        f"**Query:** {query}\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
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
    logger.info("Results saved to: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


async def run_research(query: str) -> Dict[str, Any]:
    """
    Execute the full multi-agent research pipeline for a given query.

    Streams live progress messages from each agent as they run, then
    performs a final ainvoke to obtain the fully-merged state.

    Args:
        query: The research question to investigate.

    Returns:
        The final, fully-populated workflow state dict.
    """
    print(f"\n{'MULTI-AGENT RESEARCH ASSISTANT':^80}")
    print(SEPARATOR)
    print(f"Query: {query}")
    print(SEPARATOR)

    llm = get_llm()
    workflow = build_workflow(llm)
    state: ResearchState = initial_state(query)
    final_state: Dict[str, Any] = dict(state)

    start_time = time.time()
    logger.info("Starting workflow execution...")

    spinner = Spinner("Initializing workflow")
    await spinner.start()

    # Stream once for live progress and accumulate the final merged state.
    async for chunk in workflow.astream(state, stream_mode="updates"):
        for _node_name, node_output in chunk.items():
            elapsed = time.time() - start_time
            for msg in node_output.get("messages", []):
                print(f"  [{elapsed:5.1f}s] {msg}")
            final_state = merge_state_update(final_state, node_output)

    await spinner.stop("Workflow complete")

    elapsed_total = time.time() - start_time
    logger.info("Workflow complete in %.1fs", elapsed_total)

    return final_state


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def display_results(state: Dict[str, Any], query: str) -> None:
    """
    Display all research outputs from the final state in a readable format
    and save to markdown file.

    Args:
        state: Final workflow state.
        query: Original research query (for filename).
    """
    print(f"\n\n{'RESEARCH COMPLETE':^80}")

    # Search Results (truncated preview)
    search = state.get("search_results") or "N/A"
    search_preview = (
        search[:500] + "\n... [truncated]" if len(search) > 500 else search
    )
    print_section("SEARCH RESULTS (preview)", search_preview)

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

    # Save to markdown file
    try:
        output_path = save_results_to_markdown(state, query)
        print(f"\n  \U0001f4c1 Results saved to: {output_path}")
    except Exception as e:
        logger.error("Failed to save results to file: %s", e)
        print(f"\n  \u26a0\ufe0f  Could not save results to file: {e}")

    print(f"\n{SEPARATOR}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Main async entry point."""
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    try:
        final_state = await run_research(query)
        display_results(final_state, query)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Workflow failed with unhandled error: %s", e)
        print(f"\n\u274c ERROR: {e}")
        print("Make sure GROQ_API_KEY is set in your .env file.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())