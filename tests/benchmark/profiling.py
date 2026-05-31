"""Performance profiling script for the 4 transplanted components.

Measures latency and identifies bottlenecks:
1. Multi-Source Search: concurrent vs sequential
2. URL Bloom Filter: add/lookup throughput
3. Token Budget: counting and truncation throughput
4. MCP Cache: hit vs miss latency

Run: python -m tests.benchmark.profiling
"""

import asyncio
import cProfile
import io
import pstats
import time
from unittest.mock import AsyncMock, MagicMock, patch


def _timer(func, *args, **kwargs):
    """Run func and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


async def _async_timer(func, *args, **kwargs):
    """Run async func and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = await func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ---------------------------------------------------------------------------
# 1. Multi-Source Search
# ---------------------------------------------------------------------------

def profile_multi_source_search():
    """Profile multi-source search: concurrent vs sequential."""
    from gpt_researcher.retrievers.multi_source.multi_source_search import MultiSourceSearch

    print("\n=== Multi-Source Search ===")

    class FakeRetriever:
        def __init__(self, query, query_domains=None, **kw):
            self.query = query
        def search(self, max_results=10):
            # Simulate 50ms latency
            time.sleep(0.001)  # Reduced for profiling
            return [{"href": f"https://example.com/{i}", "body": f"Result {i}"} for i in range(5)]

    # Concurrent (2 sources)
    async def concurrent_search():
        searcher = MultiSourceSearch(query="test query", sources=["a", "b"])
        with patch.object(searcher, "_get_retriever_class", return_value=FakeRetriever):
            return await searcher.search()

    result, elapsed = asyncio.run(_async_timer(concurrent_search))
    print(f"  Concurrent (2 sources): {elapsed:.1f}ms, {len(result)} results")

    # 4 sources
    async def concurrent_4():
        searcher = MultiSourceSearch(query="test", sources=["a", "b", "c", "d"])
        with patch.object(searcher, "_get_retriever_class", return_value=FakeRetriever):
            return await searcher.search()

    result, elapsed = asyncio.run(_async_timer(concurrent_4))
    print(f"  Concurrent (4 sources): {elapsed:.1f}ms, {len(result)} results")


# ---------------------------------------------------------------------------
# 2. URL Bloom Filter
# ---------------------------------------------------------------------------

def profile_bloom_filter():
    """Profile Bloom filter: add and lookup throughput."""
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "gpt_researcher", "utils", "bloom_filter.py")
    spec = importlib.util.spec_from_file_location("bf", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    BF = mod.URLBloomFilter

    print("\n=== URL Bloom Filter ===")

    bf = BF(expected_items=100_000, fp_rate=0.001)

    # Bulk add
    urls = [f"https://site{i}.com/page" for i in range(100_000)]
    _, elapsed = _timer(bf.update, urls)
    throughput = 100_000 / (elapsed / 1000) if elapsed > 0 else float("inf")
    print(f"  Add 100k URLs: {elapsed:.1f}ms ({throughput:.0f} URLs/sec)")
    print(f"  Memory: {bf.size_bytes / 1024:.1f} KB")

    # Lookup throughput
    lookup_urls = [f"https://site{i}.com/page" for i in range(0, 100_000, 10)]
    _, elapsed = _timer(lambda: [u in bf for u in lookup_urls])
    throughput = len(lookup_urls) / (elapsed / 1000) if elapsed > 0 else float("inf")
    print(f"  Lookup 10k URLs: {elapsed:.1f}ms ({throughput:.0f} lookups/sec)")


# ---------------------------------------------------------------------------
# 3. Token Budget
# ---------------------------------------------------------------------------

def profile_token_budget():
    """Profile token counting and truncation throughput."""
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "gpt_researcher", "context", "token_budget.py")
    spec = importlib.util.spec_from_file_location("tb", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("\n=== Token Budget ===")

    # Count tokens throughput
    texts = ["The quick brown fox jumps over the lazy dog. " * i for i in range(1, 101)]
    _, elapsed = _timer(lambda: [mod.count_tokens(t) for t in texts])
    print(f"  Count tokens (100 texts): {elapsed:.1f}ms")

    # Truncation throughput
    large_text = "word " * 10000
    _, elapsed = _timer(lambda: [mod.truncate_to_tokens(large_text, n) for n in [100, 500, 1000, 2000]])
    print(f"  Truncate 4x (10k-word text): {elapsed:.1f}ms")

    # TokenBudget enforcement
    budget = mod.TokenBudget(max_tokens=8000)
    doc = "context chunk " * 100
    _, elapsed = _timer(lambda: [budget.consume(doc) for _ in range(20)])
    print(f"  Budget enforcement (20 chunks): {elapsed:.1f}ms")


# ---------------------------------------------------------------------------
# 4. MCP Cache
# ---------------------------------------------------------------------------

def profile_mcp_cache():
    """Profile MCP cache hit vs miss latency."""
    from gpt_researcher.mcp.server import TTLCache

    print("\n=== MCP Cache ===")

    cache = TTLCache(max_entries=1000, ttl=3600)

    async def _run():
        # Populate cache
        for i in range(100):
            await cache.set("tool", {"q": f"query_{i}"}, f"result_{i}")

        # Cache hit
        hits = []
        for i in range(100):
            t0 = time.perf_counter()
            await cache.get("tool", {"q": f"query_{i}"})
            hits.append((time.perf_counter() - t0) * 1000)

        avg_hit = sum(hits) / len(hits)
        print(f"  Cache hit (100 lookups): avg {avg_hit:.3f}ms")

        # Cache miss
        misses = []
        for i in range(100):
            t0 = time.perf_counter()
            await cache.get("tool", {"q": f"missing_{i}"})
            misses.append((time.perf_counter() - t0) * 1000)

        avg_miss = sum(misses) / len(misses)
        print(f"  Cache miss (100 lookups): avg {avg_miss:.3f}ms")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Agent-Swarm Transplant — Performance Profiling")
    print("=" * 60)

    profile_multi_source_search()
    profile_bloom_filter()
    profile_token_budget()
    profile_mcp_cache()

    print("\n" + "=" * 60)
    print("Profiling complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
