"""URL Bloom Filter for memory-efficient URL deduplication.

This module provides a session-scoped Bloom filter that replaces the
visited_urls set for large-scale URL deduplication with significantly
lower memory usage (~120KB for 100k URLs vs ~20MB for a Python set).
"""

import asyncio
import math
import mmh3
from bitarray import bitarray


class URLBloomFilter:
    """A thread-safe Bloom filter for URL deduplication.

    Provides a set-like interface (``in``, ``add``, ``update``, ``clear``,
    ``__len__``) while using far less memory than a Python ``set``.

    Parameters match the typical GPT-Researcher session scale:
        - Expected items: 100 000
        - False-positive rate: 0.1 %
        - Hash functions: 7
        - Memory: ~120 KB
    """

    def __init__(
        self,
        expected_items: int = 100_000,
        fp_rate: float = 0.001,
    ):
        """Initialise the Bloom filter.

        Args:
            expected_items: Expected number of unique URLs.
            fp_rate: Desired false-positive probability (default 0.1%).
        """
        self._expected_items = expected_items
        self._fp_rate = fp_rate
        self._count = 0

        # Optimal bit-array size: m = -n * ln(p) / (ln2)^2
        self._size = self._optimal_size(expected_items, fp_rate)
        # Optimal number of hash functions: k = (m/n) * ln2
        self._num_hashes = self._optimal_hash_count(self._size, expected_items)

        self._bits = bitarray(self._size)
        self._bits.setall(0)

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Set-like interface
    # ------------------------------------------------------------------

    def __contains__(self, url: str) -> bool:
        """Check membership (may return false positives at the configured rate)."""
        for i in self._get_indices(url):
            if not self._bits[i]:
                return False
        return True

    def add(self, url: str) -> None:
        """Add a URL to the filter."""
        for i in self._get_indices(url):
            self._bits[i] = 1
        self._count += 1

    def update(self, urls) -> None:
        """Add multiple URLs (iterable of strings)."""
        for url in urls:
            self.add(url)

    def clear(self) -> None:
        """Reset the filter to empty state."""
        self._bits.setall(0)
        self._count = 0

    def __len__(self) -> int:
        """Return the number of items added (not the number of unique URLs)."""
        return self._count

    def copy(self) -> "URLBloomFilter":
        """Create a shallow copy of this Bloom filter.

        Returns:
            A new URLBloomFilter with the same bit-array state.
        """
        new_filter = URLBloomFilter(
            expected_items=self._expected_items,
            fp_rate=self._fp_rate,
        )
        new_filter._bits = self._bits.copy()
        new_filter._count = self._count
        return new_filter

    def __iter__(self):
        """Iteration is not supported for Bloom filters.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Bloom filters do not support enumeration. "
            "Use a regular set if you need to iterate URLs."
        )

    # ------------------------------------------------------------------
    # Session isolation
    # ------------------------------------------------------------------

    def reconstruct(self) -> "URLBloomFilter":
        """Create a fresh, empty Bloom filter with the same parameters.

        Useful for session isolation in multi-turn research workflows.
        """
        return URLBloomFilter(
            expected_items=self._expected_items,
            fp_rate=self._fp_rate,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit-array size."""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return int(math.ceil(k))

    def _get_indices(self, url: str):
        """Generate ``_num_hashes`` bit indices for *url* using double hashing."""
        h1 = mmh3.hash(url, 0, signed=False)
        h2 = mmh3.hash(url, h1, signed=False)
        for i in range(self._num_hashes):
            idx = (h1 + i * h2) % self._size
            yield idx

    # ------------------------------------------------------------------
    # Async helpers (thread-safe wrappers)
    # ------------------------------------------------------------------

    async def async_add(self, url: str) -> None:
        """Thread-safe ``add``."""
        async with self._lock:
            self.add(url)

    async def async_contains(self, url: str) -> bool:
        """Thread-safe membership check."""
        # Reads are safe without the lock on CPython (GIL), but we
        # acquire for consistency with writers.
        async with self._lock:
            return url in self

    async def async_update(self, urls) -> None:
        """Thread-safe ``update``."""
        async with self._lock:
            self.update(urls)

    async def async_clear(self) -> None:
        """Thread-safe ``clear``."""
        async with self._lock:
            self.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size_bits(self) -> int:
        """Total number of bits in the filter."""
        return self._size

    @property
    def size_bytes(self) -> int:
        """Memory usage in bytes (bit array only)."""
        return self._size // 8

    @property
    def num_hashes(self) -> int:
        """Number of hash functions used."""
        return self._num_hashes

    @property
    def false_positive_rate(self) -> float:
        """Configured false-positive rate."""
        return self._fp_rate
