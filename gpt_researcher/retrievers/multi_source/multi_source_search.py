"""Multi-source async search orchestrator for GPT Researcher.

Merges results from multiple gpt-researcher retrievers concurrently,
with token bucket rate limiting, exponential backoff, result deduplication,
and quality-based ranking adapted from Agent-Swarm's SearchAdapter.
"""

import asyncio
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Async token bucket rate limiter (per-source).

    Tokens refill at a steady rate up to the bucket capacity. Each request
    consumes one token; if the bucket is empty the caller waits until a
    token becomes available.
    """

    def __init__(self, rate: float, capacity: Optional[int] = None):
        """
        Args:
            rate: Tokens added per second (e.g. 10.0 for 10 req/s).
            capacity: Maximum tokens in the bucket. Defaults to *rate*
                      (burst of 1 second).
        """
        self._rate = max(rate, 0.01)
        self._capacity = capacity if capacity is not None else int(self._rate)
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Time until the next token arrives.
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Exponential Backoff Helper
# ---------------------------------------------------------------------------

async def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Sleep with exponential backoff + jitter. Returns the sleep duration."""
    delay = min(base * (2 ** attempt), cap)
    jitter = delay * 0.25 * (2 * (time.monotonic() % 1) - 1)  # +/-25%
    sleep_for = max(0, delay + jitter)
    await asyncio.sleep(sleep_for)
    return sleep_for


# ---------------------------------------------------------------------------
# Quality Scoring (generalized from Agent-Swarm SearchAdapter)
# ---------------------------------------------------------------------------

# Research-oriented domain tiers — generalized from sports-specific lists.
_OFFICIAL_DOMAINS = frozenset([
    ".gov", ".edu", ".mil",
])
_AUTHORITY_DOMAINS = frozenset([
    "who.int", "un.org", "europa.eu", "nih.gov", "cdc.gov",
    "nasa.gov", "ieee.org", "acm.org", "nature.com", "science.org",
    "springer.com", "wiley.com", "elsevier.com", "oxford.ac.uk",
    "cambridge.org", "pubmed.ncbi.nlm.nih.gov",
])
_ENCYCLOPEDIA_DOMAINS = frozenset([
    "wikipedia.org", "britannica.com", "baike.baidu.com",
    "encyclopedia.com", "scholarpedia.org",
])
_NEWS_DOMAINS = frozenset([
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "theguardian.com", "wsj.com", "ft.com", "economist.com",
    "technologyreview.com", "sciencedaily.com",
])
_STATISTICS_DOMAINS = frozenset([
    "statista.com", "data.gov", "worldbank.org", "un.org/statistics",
    "census.gov", "oecd.org",
])

# Generic research signal words (replaces sports-specific infobox/stats words).
_POSITIVE_SIGNALS = frozenset([
    "abstract", "findings", "results", "conclusion", "methodology",
    "experiment", "analysis", "study", "research", "review",
    "paper", "journal", "conference", "proceedings", "doi",
    "pdf", "supplementary", "appendix", "references",
])

# Hard noise page kinds.
_NOISE_SIGNALS = frozenset([
    "random article", "main page", "special:", "portal:",
    "on this day", "calendar", "birthdays", "deaths",
    "current events", "contents", "disambiguation",
])


def _source_tier(url: str) -> str:
    """Classify a URL into a trust tier."""
    domain = urlparse(url).netloc.lower()
    if any(domain.endswith(s) for s in _OFFICIAL_DOMAINS):
        return "official"
    if any(auth in domain for auth in _AUTHORITY_DOMAINS):
        return "authority"
    if any(enc in domain for enc in _ENCYCLOPEDIA_DOMAINS):
        return "encyclopedia"
    if any(news in domain for news in _NEWS_DOMAINS):
        return "news"
    if any(stat in domain for stat in _STATISTICS_DOMAINS):
        return "statistics"
    return "generic"


_TIER_WEIGHTS = {
    "official": 1.0,
    "authority": 0.95,
    "statistics": 0.90,
    "encyclopedia": 0.82,
    "news": 0.72,
    "generic": 0.55,
}

_DOMAIN_CREDIBILITY: Dict[str, float] = {
    ".gov": 0.95, ".edu": 0.95, ".mil": 0.95,
}


def _domain_credibility(url: str) -> float:
    domain = urlparse(url).netloc.lower()
    for suffix, score in _DOMAIN_CREDIBILITY.items():
        if domain.endswith(suffix):
            return score
    if any(auth in domain for auth in _AUTHORITY_DOMAINS):
        return 0.90
    if any(enc in domain for enc in _ENCYCLOPEDIA_DOMAINS):
        return 0.85
    return 0.70


def _query_tokens(query: str) -> List[str]:
    """Extract meaningful tokens from a query."""
    import re
    lowered = query.lower()
    latin = re.findall(r"[a-z0-9]{2,}", lowered)
    cjk = re.findall(r"[一-鿿]{2,6}", query)
    return list(dict.fromkeys(latin + cjk))


def quality_score(query: str, result: Dict[str, str]) -> float:
    """Compute a quality score for a search result (0-1 scale).

    Adapted from Agent-Swarm's SearchAdapter._quality_score, generalized
    from sports-specific to broad research domain.
    """
    url = result.get("href", "")
    body = result.get("body", "")
    tier = _source_tier(url)
    tier_weight = _TIER_WEIGHTS.get(tier, 0.55)

    tokens = _query_tokens(query)
    haystack_body = f" {body.lower()} "
    overlap_body = sum(1 for t in tokens if t in haystack_body)

    score = tier_weight
    score += 0.12 * min(overlap_body, 5)  # query-token overlap bonus
    score += 0.25 * _domain_credibility(url)

    # Positive research signals in body.
    lowered_combo = f"{body} {url}".lower()
    if any(sig in lowered_combo for sig in _POSITIVE_SIGNALS):
        score += 0.10

    # Noise penalty.
    if any(sig in lowered_combo for sig in _NOISE_SIGNALS):
        score -= 0.60

    return round(max(min(score, 1.0), 0.0), 4)


# ---------------------------------------------------------------------------
# Multi-Source Search Orchestrator
# ---------------------------------------------------------------------------

class MultiSourceSearch:
    """Async concurrent search across multiple gpt-researcher retrievers.

    Usage::

        searcher = MultiSourceSearch(
            query="quantum computing 2025",
            sources=["tavily", "serper", "duckduckgo"],
        )
        results = await searcher.search()

    Returns a list of ``{"href": url, "body": snippet}`` dicts sorted by
    quality score, with duplicates removed by URL.
    """

    def __init__(
        self,
        query: str,
        headers: Optional[Dict[str, str]] = None,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 10,
        query_domains: Optional[List[str]] = None,
        rate_limit_per_source: float = 10.0,
        max_retries: int = 3,
        topic: str = "general",
    ):
        """
        Args:
            query: The search query.
            headers: Optional headers dict (passed to retrievers).
            sources: List of retriever names to query. Defaults to
                     ``["tavily"]``.
            max_results_per_source: Max results to request per source.
            query_domains: Optional domain filter for searches.
            rate_limit_per_source: Max requests/second per source.
            max_retries: Max retries per source on failure.
            topic: Search topic (default "general").
        """
        self.query = query
        self.headers = headers or {}
        self.sources = sources or ["tavily"]
        self.max_results_per_source = max_results_per_source
        self.query_domains = query_domains
        self.topic = topic
        self.max_retries = max_retries

        # Per-source rate limiters (lazy-init in _get_limiter).
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._rate_limit_per_source = rate_limit_per_source

    # -- Retriever instantiation (lazy) ---------------------------------

    def _get_retriever_class(self, name: str) -> Optional[Type]:
        """Resolve a retriever name to its class via the factory."""
        from gpt_researcher.actions.retriever import get_retriever
        return get_retriever(name)

    def _get_limiter(self, source: str) -> TokenBucketRateLimiter:
        if source not in self._limiters:
            self._limiters[source] = TokenBucketRateLimiter(
                rate=self._rate_limit_per_source
            )
        return self._limiters[source]

    # -- Single-source search with retry + backoff ----------------------

    async def _search_single_source(
        self, source: str
    ) -> List[Dict[str, str]]:
        """Search one source with rate limiting, retry, and backoff."""
        retriever_cls = self._get_retriever_class(source)
        if retriever_cls is None:
            logger.warning("Unknown retriever '%s', skipping.", source)
            return []

        limiter = self._get_limiter(source)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            await limiter.acquire()
            try:
                # Existing retrievers are sync; run in a thread to avoid
                # blocking the event loop.
                results = await asyncio.to_thread(
                    retriever_cls,
                    self.query,
                    query_domains=self.query_domains,
                )
                # Some retrievers accept `topic`; try to pass it for those
                # that support it (tavily, etc.). If the constructor
                # rejects it we just re-init without it.
                try:
                    search_results = await asyncio.to_thread(
                        results.search,
                        max_results=self.max_results_per_source,
                    )
                except TypeError:
                    search_results = await asyncio.to_thread(
                        results.search
                    )
                return self._normalize_results(search_results)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Source '%s' attempt %d/%d failed: %s",
                    source, attempt + 1, self.max_retries + 1, exc,
                )
                if attempt < self.max_retries:
                    await _backoff_sleep(attempt)

        logger.error(
            "Source '%s' exhausted all retries. Last error: %s",
            source, last_error,
        )
        return []

    @staticmethod
    def _normalize_results(
        raw: Any,
    ) -> List[Dict[str, str]]:
        """Ensure every result has ``href`` and ``body`` keys."""
        out: List[Dict[str, str]] = []
        if not raw:
            return out
        for item in raw:
            if isinstance(item, dict):
                href = item.get("href") or item.get("url", "")
                body = item.get("body") or item.get("snippet", "")
                if href:
                    out.append({"href": href, "body": body})
        return out

    # -- Concurrent orchestration ----------------------------------------

    async def search(self) -> List[Dict[str, str]]:
        """Search all sources concurrently and return merged, deduplicated
        results sorted by quality score.
        """
        if not self.sources:
            return []

        tasks = [
            self._search_single_source(src) for src in self.sources
        ]
        per_source_results = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        # Merge & deduplicate by normalised URL.
        seen_urls: set = set()
        merged: List[Dict[str, str]] = []
        for source_results in per_source_results:
            for result in source_results:
                normalised = self._normalise_url(result["href"])
                if normalised not in seen_urls:
                    seen_urls.add(normalised)
                    merged.append(result)

        # Score and sort.
        scored = [
            (quality_score(self.query, r), r) for r in merged
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    @staticmethod
    def _normalise_url(url: str) -> str:
        """Strip trailing slashes, fragments for deduplication."""
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), p.params, "", ""))
