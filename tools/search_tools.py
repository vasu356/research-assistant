"""
tools/search_tools.py
----------------------
DuckDuckGo-based web search tool definitions for the research assistant.

Provides LangChain-compatible tools that search the web for real-time
information without requiring an API key.

Network note: These tools require outbound internet access. They will fail
gracefully if the network is unavailable or rate-limited.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Import DDGS — support both old and new package names
try:
    from ddgs import DDGS  # new package name (pip install ddgs)
except ImportError:
    from duckduckgo_search import DDGS  # legacy name (pip install duckduckgo-search)


def _format_result(prefix: str, index: int, result: Dict[str, Any]) -> str:
    """Format a single search result into a consistent string block."""
    title = result.get("title", "N/A")
    url = result.get("href", result.get("url", "N/A"))
    snippet = result.get("body", result.get("snippet", result.get("excerpt", "N/A")))
    return (
        f"[{prefix} {index}]\n"
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Snippet: {snippet}\n"
    )


def _format_news_result(prefix: str, index: int, result: Dict[str, Any]) -> str:
    """Format a single news result into a consistent string block."""
    title = result.get("title", "N/A")
    url = result.get("url", result.get("href", "N/A"))
    date = result.get("date", "N/A")
    source = result.get("source", "N/A")
    snippet = result.get("body", result.get("excerpt", "N/A"))
    return (
        f"[{prefix} {index}]\n"
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Date: {date}\n"
        f"Source: {source}\n"
        f"Snippet: {snippet}\n"
    )


@tool
def duckduckgo_search(query: str, max_results: int = 6) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 6).

    Returns:
        A formatted string containing search results with titles, URLs, and snippets.
    """
    try:
        logger.info("[SearchTool] Searching for: %r", query)

        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            results: List[Dict[str, Any]] = list(raw) if raw else []

        if not results:
            return f"No results found for query: {query}"

        formatted = [
            _format_result("Result", i, r) for i, r in enumerate(results, 1)
        ]

        return "\n".join(formatted)

    except Exception as e:
        logger.error("[SearchTool] Search failed: %s", e)
        return (
            f"Search encountered an error: {e}. "
            "Please ensure network access is available."
        )


@tool
def duckduckgo_news_search(query: str, max_results: int = 5) -> str:
    """
    Search recent news using DuckDuckGo News and return formatted results.

    Args:
        query: The news search query string.
        max_results: Maximum number of news articles to return (default 5).

    Returns:
        A formatted string containing news results with titles, URLs, dates, and snippets.
    """
    try:
        logger.info("[NewsTool] Searching news for: %r", query)

        with DDGS() as ddgs:
            raw = ddgs.news(query, max_results=max_results)
            results: List[Dict[str, Any]] = list(raw) if raw else []

        if not results:
            return f"No news results found for query: {query}"

        formatted = [
            _format_news_result("News", i, r) for i, r in enumerate(results, 1)
        ]

        return "\n".join(formatted)

    except Exception as e:
        logger.error("[NewsTool] News search failed: %s", e)
        return (
            f"News search encountered an error: {e}. "
            "Please ensure network access is available."
        )


# Registry of all available tools for easy import by other modules
AVAILABLE_TOOLS = [duckduckgo_search, duckduckgo_news_search]
TOOLS_BY_NAME: Dict[str, tool] = {t.name: t for t in AVAILABLE_TOOLS}