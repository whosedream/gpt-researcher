"""GPT Researcher MCP Server.

Exposes three tools via the Model Context Protocol:
  - deep_search: full research + report generation
  - web_search:  fast search across configured retrievers
  - browse_page: extract text content from a URL

Features:
  - Token budget (8k max per response)
  - LRU response cache (1000 entries, 1 h TTL)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover
    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
            self.name = name
        def tool(self):
            def decorator(func):
                return func
            return decorator
        def run(self) -> None:
            raise RuntimeError("mcp SDK is not installed")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_NAME = "GPTResearcherMCP"
MAX_TOKEN_BUDGET = 8000  # ~8k tokens ≈ ~32k chars (conservative 4 chars/token)
CHAR_BUDGET = MAX_TOKEN_BUDGET * 4
CACHE_MAX_ENTRIES = 1000
CACHE_TTL_SECONDS = 3600  # 1 hour

# ---------------------------------------------------------------------------
# LRU Cache with TTL
# ---------------------------------------------------------------------------


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiration."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl: float = CACHE_TTL_SECONDS):
        self._max = max_entries
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_key(tool_name: str, kwargs: dict) -> str:
        raw = f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def get(self, tool_name: str, kwargs: dict) -> Any | None:
        key = self._make_key(tool_name, kwargs)
        async with self._lock:
            if key in self._store:
                ts, value = self._store[key]
                if time.monotonic() - ts < self._ttl:
                    self._store.move_to_end(key)
                    return value
                del self._store[key]
        return None

    async def set(self, tool_name: str, kwargs: dict, value: Any) -> None:
        key = self._make_key(tool_name, kwargs)
        async with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[key] = (time.monotonic(), value)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


_cache = TTLCache()

# ---------------------------------------------------------------------------
# Token budget helper
# ---------------------------------------------------------------------------


def _truncate_to_budget(text: str, budget: int = CHAR_BUDGET) -> str:
    """Truncate *text* so it stays within the character budget."""
    if len(text) <= budget:
        return text
    return text[: budget - 50] + "\n\n... [truncated to fit token budget]"

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(SERVER_NAME)


@mcp.tool()
async def deep_search(question: str, report_type: str = "research_report") -> str:
    """Run a full deep research: search, scrape, and generate a report.

    Args:
        question: The research question (supports Chinese and English).
        report_type: Report type (default: research_report). Options:
            research_report, detailed_report, resource_report, outline_report.
    """
    cache_key_kwargs = {"question": question, "report_type": report_type}
    cached = await _cache.get("deep_search", cache_key_kwargs)
    if cached is not None:
        return cached

    try:
        from gpt_researcher.agent import GPTResearcher

        researcher = GPTResearcher(query=question, report_type=report_type)
        await researcher.conduct_research()
        report = await researcher.write_report()
        result = _truncate_to_budget(report)
        await _cache.set("deep_search", cache_key_kwargs, result)
        return result
    except Exception as e:
        logger.error("deep_search failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using configured retrievers. Returns top results.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5).
    """
    cache_key_kwargs = {"query": query, "max_results": max_results}
    cached = await _cache.get("web_search", cache_key_kwargs)
    if cached is not None:
        return cached

    try:
        from gpt_researcher.retrievers.multi_source.multi_source_search import MultiSourceSearch
        from gpt_researcher.config.config import Config

        cfg = Config()
        sources = getattr(cfg, "search_sources", ["tavily"])
        rate_limit = getattr(cfg, "search_rate_limit_per_source", 10.0)

        searcher = MultiSourceSearch(
            query=query,
            sources=sources,
            max_results_per_source=max_results,
            rate_limit_per_source=rate_limit,
        )
        results = await searcher.search()
        trimmed = results[:max_results]

        output = json.dumps(
            [{"title": r.get("body", "")[:80], "url": r["href"], "snippet": r["body"]}
             for r in trimmed],
            ensure_ascii=False,
        )
        result = _truncate_to_budget(output)
        await _cache.set("web_search", cache_key_kwargs, result)
        return result
    except Exception as e:
        logger.error("web_search failed: %s", e)
        # Fallback: try single retriever
        try:
            from gpt_researcher.actions.retriever import get_retriever
            from gpt_researcher.config.config import Config

            cfg = Config()
            retriever_name = getattr(cfg, "retriever", "tavily")
            retriever_cls = get_retriever(retriever_name)
            if retriever_cls is None:
                return json.dumps({"error": f"No retriever found: {retriever_name}"}, ensure_ascii=False)

            search_retriever = retriever_cls(query)
            raw_results = search_retriever.search(max_results=max_results)
            output = json.dumps(raw_results, ensure_ascii=False)
            result = _truncate_to_budget(output)
            await _cache.set("web_search", cache_key_kwargs, result)
            return result
        except Exception as e2:
            logger.error("web_search fallback also failed: %s", e2)
            return json.dumps({"error": str(e2)}, ensure_ascii=False)


@mcp.tool()
async def browse_page(url: str) -> str:
    """Extract text content from a web page (max 5000 chars).

    Args:
        url: The URL to fetch and extract text from.
    """
    cache_key_kwargs = {"url": url}
    cached = await _cache.get("browse_page", cache_key_kwargs)
    if cached is not None:
        return cached

    try:
        from gpt_researcher.actions.web_scraping import scrape_urls
        from gpt_researcher.utils.workers import WorkerPool
        from gpt_researcher.config.config import Config

        cfg = Config()
        worker_pool = WorkerPool(max_workers=cfg.max_scraper_workers)
        scraped_data, _ = await scrape_urls([url], cfg, worker_pool)

        if scraped_data and len(scraped_data) > 0:
            raw = scraped_data[0].get("raw_content", "") or scraped_data[0].get("content", "")
        else:
            raw = ""

        text = raw[:5000] if raw else ""
        result = json.dumps({"url": url, "content": text, "length": len(text)}, ensure_ascii=False)
        await _cache.set("browse_page", cache_key_kwargs, result)
        return result
    except Exception as e:
        logger.error("browse_page failed for %s: %s", url, e)
        return json.dumps({"url": url, "content": "", "error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def clear_cache() -> str:
    """Clear the response cache."""
    await _cache.clear()
    return json.dumps({"status": "cache cleared"}, ensure_ascii=False)


def main() -> None:
    """Entry point for running the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
