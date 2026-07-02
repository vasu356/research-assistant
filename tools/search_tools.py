"""
tools/search_tools.py
----------------------
DuckDuckGo-based web search tool definitions for the research assistant.

Provides LangChain-compatible tools that search the web for real-time
information without requiring an API key.  Both tools implement graceful
degradation: network failures or rate-limits return a descriptive error
string rather than raising, keeping the agent loop alive.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDGS import — support both old (duckduckgo-search) and new (ddgs) packages.
# We import into a plain Any-typed name to keep mypy happy regardless of
# which package is installed; both expose an identical context-manager API.
# ---------------------------------------------------------------------------
try:
    from ddgs import DDGS as _DDGSImpl
except ImportError:
    from duckduckgo_search import DDGS as _DDGSImpl  # type: ignore[no-redef]

_DDGS: Any = _DDGSImpl


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_result(prefix: str, index: int, result: dict[str, Any]) -> str:
    """Format a single web search result into a labelled text block."""
    title = result.get("title", "N/A")
    url = result.get("href", result.get("url", "N/A"))
    snippet = result.get("body", result.get("snippet", result.get("excerpt", "N/A")))
    return f"[{prefix} {index}]\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"


def _format_news_result(prefix: str, index: int, result: dict[str, Any]) -> str:
    """Format a single news result into a labelled text block."""
    title = result.get("title", "N/A")
    url = result.get("url", result.get("href", "N/A"))
    date = result.get("date", "N/A")
    source = result.get("source", "N/A")
    snippet = result.get("body", result.get("excerpt", "N/A"))
    return (
        f"[{prefix} {index}]\n"
        f"Title: {title}\nURL: {url}\nDate: {date}\nSource: {source}\nSnippet: {snippet}\n"
    )


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


@tool
def duckduckgo_search(query: str, max_results: int = settings.SEARCH_MAX_RESULTS) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted string containing titles, URLs, and snippets.
        Returns a descriptive error string on failure — never raises.
    """
    try:
        logger.info("[SearchTool] Querying web: %r (max=%d)", query, max_results)
        with _DDGS() as ddgs:
            results: list[dict[str, Any]] = list(ddgs.text(query, max_results=max_results) or [])

        if not results:
            return f"No results found for query: {query!r}"

        blocks = [_format_result("Result", i, r) for i, r in enumerate(results, 1)]
        logger.info("[SearchTool] Retrieved %d web results.", len(results))
        return "\n".join(blocks)

    except Exception as exc:
        logger.error("[SearchTool] Web search failed: %s", exc)
        return f"Web search error: {exc}. Ensure outbound internet access is available."


@tool
def duckduckgo_news_search(query: str, max_results: int = settings.NEWS_MAX_RESULTS) -> str:
    """
    Search recent news using DuckDuckGo News and return formatted results.

    Args:
        query: The news search query string.
        max_results: Maximum number of news articles to return.

    Returns:
        Formatted string containing titles, URLs, dates, sources, and snippets.
        Returns a descriptive error string on failure — never raises.
    """
    try:
        logger.info("[NewsTool] Querying news: %r (max=%d)", query, max_results)
        with _DDGS() as ddgs:
            results: list[dict[str, Any]] = list(ddgs.news(query, max_results=max_results) or [])

        if not results:
            return f"No news results found for query: {query!r}"

        blocks = [_format_news_result("News", i, r) for i, r in enumerate(results, 1)]
        logger.info("[NewsTool] Retrieved %d news articles.", len(results))
        return "\n".join(blocks)

    except Exception as exc:
        logger.error("[NewsTool] News search failed: %s", exc)
        return f"News search error: {exc}. Ensure outbound internet access is available."


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [duckduckgo_search, duckduckgo_news_search]
TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in AVAILABLE_TOOLS}
