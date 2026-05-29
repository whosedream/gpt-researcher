"""Tests for multi_source_search.py — rate limiter, backoff, concurrent search, result merging."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from gpt_researcher.retrievers.multi_source.multi_source_search import (
    MultiSourceSearch,
    TokenBucketRateLimiter,
    _backoff_sleep,
    quality_score,
    _source_tier,
    _query_tokens,
)


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Bucket allows burst up to capacity, then throttles."""
    rl = TokenBucketRateLimiter(rate=5.0, capacity=5)
    t0 = time.monotonic()
    for _ in range(5):
        await rl.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"Burst should be instant, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rate_limiter_throttles():
    """After burst, subsequent acquires wait for refill."""
    rl = TokenBucketRateLimiter(rate=10.0, capacity=1)
    await rl.acquire()  # consume the single token
    t0 = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.08, f"Should wait ~100ms, took {elapsed:.3f}s"
    assert elapsed < 0.3, f"Should not wait too long, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rate_limiter_concurrent():
    """Multiple concurrent acquires are serialized correctly."""
    rl = TokenBucketRateLimiter(rate=20.0, capacity=2)
    results = []

    async def worker(i):
        await rl.acquire()
        results.append(i)

    t0 = time.monotonic()
    await asyncio.gather(*[worker(i) for i in range(4)])
    elapsed = time.monotonic() - t0
    assert len(results) == 4
    assert elapsed >= 0.05, f"4 acquires at 20/s should take ~150ms, took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Exponential Backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backoff_sleep_returns_duration():
    """backoff_sleep returns the actual sleep duration."""
    t0 = time.monotonic()
    duration = await _backoff_sleep(0, base=0.01, cap=0.1)
    elapsed = time.monotonic() - t0
    assert duration >= 0
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_backoff_sleep_increases():
    """Later attempts sleep longer."""
    durations = []
    for attempt in range(4):
        d = await _backoff_sleep(attempt, base=0.01, cap=0.1)
        durations.append(d)
    # Each duration should be >= previous (on average, with jitter)
    assert durations[-1] >= durations[0]


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------

def test_source_tier_gov():
    assert _source_tier("https://www.nih.gov/study") == "official"


def test_source_tier_wikipedia():
    assert _source_tier("https://en.wikipedia.org/wiki/Quantum") == "encyclopedia"


def test_source_tier_news():
    assert _source_tier("https://www.reuters.com/article/123") == "news"


def test_source_tier_generic():
    assert _source_tier("https://example.com/blog") == "generic"


def test_quality_score_high_for_official():
    result = {"href": "https://www.nih.gov/study", "body": "Research findings show results"}
    score = quality_score("quantum computing research", result)
    assert score > 0.6


def test_quality_score_low_for_noise():
    result = {"href": "https://wiki.com/random_article", "body": "random article contents"}
    score = quality_score("quantum computing", result)
    assert score < 0.3


def test_quality_score_query_overlap_bonus():
    r1 = {"href": "https://example.com", "body": "quantum computing algorithms"}
    r2 = {"href": "https://example.com", "body": "unrelated topic"}
    s1 = quality_score("quantum computing", r1)
    s2 = quality_score("quantum computing", r2)
    assert s1 > s2


def test_query_tokens():
    tokens = _query_tokens("quantum computing 2025")
    assert "quantum" in tokens
    assert "computing" in tokens
    assert "2025" in tokens


# ---------------------------------------------------------------------------
# MultiSourceSearch — concurrent search + deduplication
# ---------------------------------------------------------------------------

def _make_retriever_class(return_value):
    """Create a mock retriever class matching gpt-researcher convention."""
    class MockRetriever:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query
            self.query_domains = query_domains

        def search(self, max_results=10):
            return return_value
    return MockRetriever


@pytest.mark.asyncio
async def test_search_concurrent_and_dedup():
    """Multiple sources return overlapping URLs; results are deduplicated."""
    source_a = [
        {"href": "https://example.com/a", "body": "Result A"},
        {"href": "https://example.com/b", "body": "Result B"},
    ]
    source_b = [
        {"href": "https://example.com/b", "body": "Result B duplicate"},
        {"href": "https://example.com/c", "body": "Result C"},
    ]

    mock_classes = {
        "source_a": _make_retriever_class(source_a),
        "source_b": _make_retriever_class(source_b),
    }

    searcher = MultiSourceSearch(
        query="test query",
        sources=["source_a", "source_b"],
    )

    with patch.object(searcher, "_get_retriever_class", side_effect=lambda n: mock_classes.get(n)):
        results = await searcher.search()

    urls = [r["href"] for r in results]
    assert len(urls) == 3
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
    assert "https://example.com/c" in urls


@pytest.mark.asyncio
async def test_search_single_source_failure_continues():
    """If one source fails, others still return results."""
    good_source = [{"href": "https://good.com/1", "body": "Good result"}]

    mock_classes = {
        "good": _make_retriever_class(good_source),
        "bad": None,  # will cause _get_retriever_class to return None
    }

    searcher = MultiSourceSearch(
        query="test",
        sources=["bad", "good"],
    )

    with patch.object(searcher, "_get_retriever_class", side_effect=lambda n: mock_classes.get(n)):
        results = await searcher.search()

    assert len(results) == 1
    assert results[0]["href"] == "https://good.com/1"


@pytest.mark.asyncio
async def test_search_empty_sources():
    """Empty source list returns empty results."""
    searcher = MultiSourceSearch(query="test", sources=[])
    results = await searcher.search()
    assert results == []


@pytest.mark.asyncio
async def test_search_sorted_by_quality():
    """Results are sorted by quality score (highest first)."""
    low_quality = [{"href": "https://example.com/low", "body": "random article stuff"}]
    high_quality = [{"href": "https://www.nih.gov/high", "body": "research findings methodology"}]

    mock_classes = {
        "low_src": _make_retriever_class(low_quality),
        "high_src": _make_retriever_class(high_quality),
    }

    searcher = MultiSourceSearch(
        query="research methodology",
        sources=["low_src", "high_src"],
    )

    with patch.object(searcher, "_get_retriever_class", side_effect=lambda n: mock_classes.get(n)):
        results = await searcher.search()

    assert len(results) == 2
    # High quality (.gov + research signals) should come first
    assert results[0]["href"] == "https://www.nih.gov/high"


# ---------------------------------------------------------------------------
# Registration checks
# ---------------------------------------------------------------------------

def test_multi_source_in_valid_retrievers():
    """multi_source is registered in VALID_RETRIEVERS."""
    from gpt_researcher.retrievers.utils import VALID_RETRIEVERS
    assert "multi_source" in VALID_RETRIEVERS


def test_get_retriever_returns_multi_source():
    """get_retriever('multi_source') returns MultiSourceSearch."""
    from gpt_researcher.actions.retriever import get_retriever
    from gpt_researcher.retrievers import MultiSourceSearch
    cls = get_retriever("multi_source")
    assert cls is MultiSourceSearch
