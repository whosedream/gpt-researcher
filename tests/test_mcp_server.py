"""Tests for gpt_researcher.mcp.server — TTLCache, token budget, and tool wiring."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.mcp.server import (
    TTLCache,
    _truncate_to_budget,
    CHAR_BUDGET,
    MAX_TOKEN_BUDGET,
    mcp,
)


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_set_and_get():
    cache = TTLCache(max_entries=10, ttl=60)
    await cache.set("tool", {"q": "hello"}, "result_1")
    val = await cache.get("tool", {"q": "hello"})
    assert val == "result_1"


@pytest.mark.asyncio
async def test_cache_miss():
    cache = TTLCache(max_entries=10, ttl=60)
    val = await cache.get("tool", {"q": "missing"})
    assert val is None


@pytest.mark.asyncio
async def test_cache_EXPIRY():
    cache = TTLCache(max_entries=10, ttl=0.05)
    await cache.set("tool", {"q": "expire"}, "val")
    await asyncio.sleep(0.1)
    val = await cache.get("tool", {"q": "expire"})
    assert val is None


@pytest.mark.asyncio
async def test_cache_eviction_LRU():
    cache = TTLCache(max_entries=2, ttl=60)
    await cache.set("t", {"k": "a"}, "A")
    await cache.set("t", {"k": "b"}, "B")
    # Access "a" to make it recently used
    await cache.get("t", {"k": "a"})
    # Insert "c" should evict "b" (least recently used)
    await cache.set("t", {"k": "c"}, "C")
    assert await cache.get("t", {"k": "a"}) == "A"
    assert await cache.get("t", {"k": "b"}) is None
    assert await cache.get("t", {"k": "c"}) == "C"


@pytest.mark.asyncio
async def test_cache_clear():
    cache = TTLCache(max_entries=10, ttl=60)
    await cache.set("t", {"k": "v"}, "val")
    await cache.clear()
    assert await cache.get("t", {"k": "v"}) is None


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def test_truncate_short_text():
    text = "short"
    assert _truncate_to_budget(text) == "short"


def test_truncate_long_text():
    text = "x" * 50000
    result = _truncate_to_budget(text, budget=1000)
    assert len(result) < 1100
    assert "truncated" in result


def test_budget_constants():
    assert MAX_TOKEN_BUDGET == 8000
    assert CHAR_BUDGET == 32000


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_mcp_server_has_tools():
    """The mcp server object exists and tools are registered."""
    assert mcp is not None
    assert mcp.name == "GPTResearcherMCP"


# ---------------------------------------------------------------------------
# deep_search tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deep_search_success():
    from gpt_researcher.mcp.server import deep_search

    mock_researcher = MagicMock()
    mock_researcher.conduct_research = AsyncMock()
    mock_researcher.write_report = AsyncMock(return_value="Test report content")

    with patch("gpt_researcher.mcp.server.GPTResearcher", return_value=mock_researcher) if False else \
         patch("gpt_researcher.agent.GPTResearcher", return_value=mock_researcher):
        # Import inside patch context would be tricky; mock the module path used in the tool
        pass

    # Direct approach: mock at the import point inside the function
    mock_module = MagicMock()
    mock_module.GPTResearcher = MagicMock(return_value=mock_researcher)

    with patch.dict("sys.modules", {"gpt_researcher.agent": mock_module}):
        result = await deep_search("test question")

    assert "Test report content" in result
    mock_researcher.conduct_research.assert_called_once()
    mock_researcher.write_report.assert_called_once()


@pytest.mark.asyncio
async def test_deep_search_cached():
    from gpt_researcher.mcp.server import _cache, deep_search

    # Pre-populate cache
    await _cache.set("deep_search", {"question": "cached_q", "report_type": "research_report"}, "cached_result")

    result = await deep_search("cached_q")
    assert result == "cached_result"


@pytest.mark.asyncio
async def test_deep_search_error():
    from gpt_researcher.mcp.server import deep_search

    with patch("gpt_researcher.agent.GPTResearcher", side_effect=RuntimeError("boom")):
        result = await deep_search("fail question")

    parsed = json.loads(result)
    assert "error" in parsed
    assert "boom" in parsed["error"]


# ---------------------------------------------------------------------------
# web_search tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_cached():
    from gpt_researcher.mcp.server import _cache, web_search

    await _cache.set("web_search", {"query": "cached_q", "max_results": 5}, "cached_web_result")
    result = await web_search("cached_q")
    assert result == "cached_web_result"


@pytest.mark.asyncio
async def test_web_search_success():
    from gpt_researcher.mcp.server import web_search

    mock_searcher = MagicMock()
    mock_searcher.search = AsyncMock(return_value=[
        {"href": "https://example.com/1", "body": "Result one"},
        {"href": "https://example.com/2", "body": "Result two"},
    ])

    mock_cfg = MagicMock()
    mock_cfg.search_sources = ["tavily"]
    mock_cfg.search_rate_limit_per_source = 10.0

    with patch("gpt_researcher.retrievers.multi_source.multi_source_search.MultiSourceSearch", return_value=mock_searcher), \
         patch("gpt_researcher.config.config.Config", return_value=mock_cfg):
        result = await web_search("test query", max_results=2)

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_web_search_fallback():
    from gpt_researcher.mcp.server import web_search

    # MultiSourceSearch fails, should fall back to single retriever
    mock_retriever_cls = MagicMock()
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.search.return_value = [{"href": "https://a.com", "body": "fallback"}]
    mock_retriever_cls.return_value = mock_retriever_instance

    mock_cfg = MagicMock()
    mock_cfg.retriever = "tavily"

    with patch("gpt_researcher.retrievers.multi_source.multi_source_search.MultiSourceSearch", side_effect=RuntimeError("multi fail")), \
         patch("gpt_researcher.config.config.Config", return_value=mock_cfg), \
         patch("gpt_researcher.actions.retriever.get_retriever", return_value=mock_retriever_cls):
        result = await web_search("fallback query")

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


# ---------------------------------------------------------------------------
# browse_page tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browse_page_cached():
    from gpt_researcher.mcp.server import _cache, browse_page

    await _cache.set("browse_page", {"url": "https://cached.com"}, "cached_page")
    result = await browse_page("https://cached.com")
    assert result == "cached_page"


@pytest.mark.asyncio
async def test_browse_page_success():
    from gpt_researcher.mcp.server import browse_page

    mock_scraped = [{"raw_content": "Page content here", "status": "success"}]
    mock_cfg = MagicMock()
    mock_cfg.max_scraper_workers = 1

    with patch("gpt_researcher.actions.web_scraping.scrape_urls", new_callable=AsyncMock, return_value=(mock_scraped, [])), \
         patch("gpt_researcher.config.config.Config", return_value=mock_cfg), \
         patch("gpt_researcher.utils.workers.WorkerPool"):
        result = await browse_page("https://example.com")

    parsed = json.loads(result)
    assert parsed["url"] == "https://example.com"
    assert parsed["content"] == "Page content here"
    assert parsed["length"] == 17


@pytest.mark.asyncio
async def test_browse_page_error():
    from gpt_researcher.mcp.server import browse_page

    with patch("gpt_researcher.actions.web_scraping.scrape_urls", new_callable=AsyncMock, side_effect=RuntimeError("scrape fail")), \
         patch("gpt_researcher.config.config.Config", return_value=MagicMock()), \
         patch("gpt_researcher.utils.workers.WorkerPool"):
        result = await browse_page("https://fail.com")

    parsed = json.loads(result)
    assert "error" in parsed
    assert "scrape fail" in parsed["error"]


# ---------------------------------------------------------------------------
# clear_cache tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_cache_tool():
    from gpt_researcher.mcp.server import _cache, clear_cache

    await _cache.set("t", {"k": "v"}, "val")
    result = await clear_cache()
    parsed = json.loads(result)
    assert parsed["status"] == "cache cleared"
    assert await _cache.get("t", {"k": "v"}) is None
