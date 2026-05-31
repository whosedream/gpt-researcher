"""E2E integration tests for the 4 transplanted Agent-Swarm components.

Tests all components together in realistic scenarios:
1. Multi-Source Search Orchestrator
2. URL Bloom Filter
3. Token-Aware Context Compression
4. MCP Tool Server

These are unit-level integration tests (mocked external APIs) so they
run without network access or API keys.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.benchmark.dataset import BENCHMARK_QUESTIONS


# ---------------------------------------------------------------------------
# Component 1: Multi-Source Search
# ---------------------------------------------------------------------------

class TestMultiSourceSearchIntegration:
    """Integration tests for the multi-source search orchestrator."""

    def _make_retriever(self, results):
        class MockRetriever:
            def __init__(self, query, query_domains=None, **kw):
                self.query = query
            def search(self, max_results=10):
                return results
        return MockRetriever

    @pytest.mark.asyncio
    async def test_search_all_benchmark_questions(self):
        """Each benchmark question can be searched without errors."""
        from gpt_researcher.retrievers.multi_source.multi_source_search import MultiSourceSearch

        mock_results = [
            {"href": "https://example.com/result1", "body": "Test result body"},
            {"href": "https://example.com/result2", "body": "Another result"},
        ]
        mock_cls = self._make_retriever(mock_results)

        for q in BENCHMARK_QUESTIONS[:5]:  # Test first 5 for speed
            searcher = MultiSourceSearch(
                query=q["question"],
                sources=["mock_source"],
            )
            with patch.object(searcher, "_get_retriever_class", return_value=mock_cls):
                results = await searcher.search()
            assert isinstance(results, list)
            assert len(results) > 0
            assert all("href" in r and "body" in r for r in results)

    @pytest.mark.asyncio
    async def test_deduplication_across_sources(self):
        """Same URL from multiple sources appears only once."""
        from gpt_researcher.retrievers.multi_source.multi_source_search import MultiSourceSearch

        shared = [{"href": "https://shared.com/article", "body": "Shared content"}]
        source_a = shared + [{"href": "https://a.com/unique", "body": "A only"}]
        source_b = shared + [{"href": "https://b.com/unique", "body": "B only"}]

        mock_classes = {
            "src_a": self._make_retriever(source_a),
            "src_b": self._make_retriever(source_b),
        }
        searcher = MultiSourceSearch(query="test", sources=["src_a", "src_b"])
        with patch.object(searcher, "_get_retriever_class", side_effect=lambda n: mock_classes.get(n)):
            results = await searcher.search()

        urls = [r["href"] for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    @pytest.mark.asyncio
    async def test_quality_sorting(self):
        """Official domains rank higher than generic."""
        from gpt_researcher.retrievers.multi_source.multi_source_search import MultiSourceSearch

        low = [{"href": "https://random-blog.com/post", "body": "blog post content"}]
        high = [{"href": "https://www.nih.gov/study", "body": "research findings methodology"}]

        mock_classes = {
            "low": self._make_retriever(low),
            "high": self._make_retriever(high),
        }
        searcher = MultiSourceSearch(query="research", sources=["low", "high"])
        with patch.object(searcher, "_get_retriever_class", side_effect=lambda n: mock_classes.get(n)):
            results = await searcher.search()

        # .gov should rank higher
        assert results[0]["href"] == "https://www.nih.gov/study"


# ---------------------------------------------------------------------------
# Component 2: URL Bloom Filter
# ---------------------------------------------------------------------------

class TestBloomFilterIntegration:
    """Integration tests for the URL Bloom filter."""

    def _load_bloom(self):
        import importlib.util, os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "gpt_researcher", "utils", "bloom_filter.py")
        spec = importlib.util.spec_from_file_location("bf", os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.URLBloomFilter

    def test_url_dedup_in_research_scenario(self):
        """Simulate research session: add URLs, check duplicates."""
        BF = self._load_bloom()
        bf = BF(expected_items=10_000, fp_rate=0.001)

        visited = []
        urls_to_visit = [f"https://example.com/page/{i}" for i in range(100)]

        for url in urls_to_visit:
            if url not in bf:
                bf.add(url)
                visited.append(url)

        assert len(visited) == 100

        # Re-visiting should be detected
        dup_count = sum(1 for u in urls_to_visit if u in bf)
        assert dup_count == 100

    def test_memory_efficiency(self):
        """100k URLs use < 200KB memory."""
        BF = self._load_bloom()
        bf = BF(expected_items=100_000, fp_rate=0.001)
        for i in range(100_000):
            bf.add(f"https://site{i}.com/path")
        assert bf.size_bytes < 200 * 1024

    @pytest.mark.asyncio
    async def test_concurrent_url_adds(self):
        """Concurrent adds don't corrupt state."""
        BF = self._load_bloom()
        bf = BF(expected_items=50_000, fp_rate=0.001)

        async def add_batch(start):
            for i in range(start, start + 1000):
                await bf.async_add(f"https://concurrent.com/{i}")

        await asyncio.gather(*[add_batch(i * 1000) for i in range(10)])
        for i in range(10_000):
            assert f"https://concurrent.com/{i}" in bf


# ---------------------------------------------------------------------------
# Component 3: Token-Aware Context Compression
# ---------------------------------------------------------------------------

class TestTokenBudgetIntegration:
    """Integration tests for token budget and context compression."""

    def _load_token_budget(self):
        import importlib.util, os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "gpt_researcher", "context", "token_budget.py")
        spec = importlib.util.spec_from_file_location("tb", os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_budget_enforcement_across_documents(self):
        """TokenBudget enforces limit across multiple document insertions."""
        tb_mod = self._load_token_budget()
        budget = tb_mod.TokenBudget(max_tokens=500)

        docs = [f"Document {i}: {'word ' * 50}" for i in range(20)]
        for doc in docs:
            if budget.can_fit(doc):
                budget.consume(doc)

        assert budget.used <= 500
        assert budget.is_exhausted or budget.remaining >= 0

    def test_truncation_preserves_beginning(self):
        """Truncated text keeps the beginning (most relevant)."""
        tb_mod = self._load_token_budget()
        text = "IMPORTANT START. " + "filler " * 1000
        result = tb_mod.truncate_to_tokens(text, 20)
        assert result.startswith("IMPORTANT START")

    def test_compression_ratio_8k_to_3k(self):
        """8k tokens compresses to <= 3.2k tokens."""
        tb_mod = self._load_token_budget()
        sentence = "The quick brown fox jumps over the lazy dog and runs through the forest. "
        text = sentence * 800
        tokens_in = tb_mod.count_tokens(text)
        assert tokens_in > 7000, f"Need >7k tokens, got {tokens_in}"

        result = tb_mod.truncate_to_tokens(text, 3200)
        tokens_out = tb_mod.count_tokens(result)
        assert tokens_out <= 3200
        reduction = 1 - tokens_out / tokens_in
        assert reduction > 0.5

    def test_context_compressor_with_budget(self):
        """ContextCompressor integrates with token_budget parameter."""
        from gpt_researcher.context.compression import ContextCompressor

        compressor = ContextCompressor(
            documents=[],
            embeddings=None,
            token_budget=3200,
        )
        assert compressor.token_budget == 3200


# ---------------------------------------------------------------------------
# Component 4: MCP Tool Server
# ---------------------------------------------------------------------------

class TestMCPServerIntegration:
    """Integration tests for the MCP tool server."""

    @pytest.mark.asyncio
    async def test_deep_search_tool_wiring(self):
        """deep_search tool calls GPTResearcher correctly."""
        from gpt_researcher.mcp.server import deep_search

        mock_researcher = MagicMock()
        mock_researcher.conduct_research = AsyncMock()
        mock_researcher.write_report = AsyncMock(return_value="Report content")

        with patch("gpt_researcher.agent.GPTResearcher", return_value=mock_researcher):
            result = await deep_search("test question")

        assert "Report content" in result
        mock_researcher.conduct_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_tool_wiring(self):
        """web_search tool calls MultiSourceSearch correctly."""
        from gpt_researcher.mcp.server import web_search

        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value=[
            {"href": "https://example.com/1", "body": "Result"},
        ])
        mock_cfg = MagicMock()
        mock_cfg.search_sources = ["tavily"]
        mock_cfg.search_rate_limit_per_source = 10.0

        with patch("gpt_researcher.retrievers.multi_source.multi_source_search.MultiSourceSearch",
                    return_value=mock_searcher), \
             patch("gpt_researcher.config.config.Config", return_value=mock_cfg):
            result = await web_search("test query")

        parsed = json.loads(result)
        assert len(parsed) >= 1

    @pytest.mark.asyncio
    async def test_browse_page_tool_wiring(self):
        """browse_page tool calls scrape_urls correctly."""
        from gpt_researcher.mcp.server import browse_page

        mock_scraped = [{"raw_content": "Page text here", "status": "success"}]
        mock_cfg = MagicMock()
        mock_cfg.max_scraper_workers = 1

        with patch("gpt_researcher.actions.web_scraping.scrape_urls",
                    new_callable=AsyncMock, return_value=(mock_scraped, [])), \
             patch("gpt_researcher.config.config.Config", return_value=mock_cfg), \
             patch("gpt_researcher.utils.workers.WorkerPool"):
            result = await browse_page("https://example.com")

        parsed = json.loads(result)
        assert parsed["content"] == "Page text here"

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Cached responses return in < 500ms."""
        from gpt_researcher.mcp.server import _cache, web_search

        await _cache.set("web_search", {"query": "cached", "max_results": 5}, "cached_result")

        t0 = time.monotonic()
        result = await web_search("cached")
        elapsed = (time.monotonic() - t0) * 1000

        assert result == "cached_result"
        assert elapsed < 500, f"Cache hit took {elapsed:.0f}ms (> 500ms)"

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self):
        """Responses exceeding 8k tokens are truncated."""
        from gpt_researcher.mcp.server import _truncate_to_budget, CHAR_BUDGET

        large_text = "x" * 50000
        result = _truncate_to_budget(large_text)
        assert len(result) <= CHAR_BUDGET + 100  # some slack for the marker


# ---------------------------------------------------------------------------
# Cross-component integration
# ---------------------------------------------------------------------------

class TestCrossComponentIntegration:
    """Tests that verify components work together."""

    @pytest.mark.asyncio
    async def test_search_then_compress_flow(self):
        """Search results feed into token budget for context assembly."""
        from gpt_researcher.retrievers.multi_source.multi_source_search import (
            MultiSourceSearch, quality_score,
        )
        tb_mod = self._load_token_budget()

        # Simulate: search returns results, we score and budget them
        mock_results = [
            {"href": "https://nih.gov/study", "body": "research findings " * 50},
            {"href": "https://example.com/blog", "body": "blog post " * 50},
        ]

        scored = sorted(
            [(quality_score("research", r), r) for r in mock_results],
            key=lambda x: x[0], reverse=True,
        )

        budget = tb_mod.TokenBudget(max_tokens=200)
        for score, result in scored:
            text = result["body"]
            if budget.can_fit(text):
                budget.consume(text)

        assert budget.used <= 200

    def _load_token_budget(self):
        import importlib.util, os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "gpt_researcher", "context", "token_budget.py")
        spec = importlib.util.spec_from_file_location("tb", os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_bloom_filter_with_search_dedup(self):
        """Bloom filter prevents re-scraping already visited URLs."""
        import importlib.util, os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "gpt_researcher", "utils", "bloom_filter.py")
        spec = importlib.util.spec_from_file_location("bf", os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BF = mod.URLBloomFilter

        bf = BF(expected_items=10_000, fp_rate=0.001)

        # Simulate two searches returning overlapping URLs
        search_a = ["https://a.com/1", "https://shared.com/page"]
        search_b = ["https://b.com/1", "https://shared.com/page"]

        new_urls = []
        for url in search_a + search_b:
            if url not in bf:
                bf.add(url)
                new_urls.append(url)

        # shared.com/page should only be scraped once
        assert new_urls.count("https://shared.com/page") == 1
        assert len(new_urls) == 3
