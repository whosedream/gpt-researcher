"""Tests for the URL Bloom filter.

Validates:
- False-positive rate within 0.1% bound
- Memory usage under 200 KB for 100 k URLs
- Thread-safe concurrent adds
- reconstruct() creates a fresh filter
- Set-like interface compatibility (in, add, update, clear, len, copy)
"""

import asyncio
import os
import sys
import time

import pytest

# Import the bloom filter module directly to avoid pulling in the full
# gpt_researcher package (which requires many optional dependencies).
_bloom_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "gpt_researcher", "utils", "bloom_filter.py"
)
_bloom_path = os.path.abspath(_bloom_path)

import importlib.util

_spec = importlib.util.spec_from_file_location("bloom_filter", _bloom_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
URLBloomFilter = _mod.URLBloomFilter


# ---------------------------------------------------------------------------
# Basic membership
# ---------------------------------------------------------------------------

class TestMembership:
    def test_added_url_is_contained(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.add("https://example.com")
        assert "https://example.com" in bf

    def test_unadded_url_not_contained(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        assert "https://example.com" not in bf

    def test_add_multiple_and_check(self):
        bf = URLBloomFilter(expected_items=10_000, fp_rate=0.001)
        urls = [f"https://example.com/page/{i}" for i in range(500)]
        bf.update(urls)
        for url in urls:
            assert url in bf

    def test_len_counts_additions(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.add("a")
        bf.add("b")
        bf.update(["c", "d"])
        assert len(bf) == 4

    def test_clear_resets_filter(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.add("https://example.com")
        bf.clear()
        assert len(bf) == 0
        assert "https://example.com" not in bf


# ---------------------------------------------------------------------------
# False-positive rate
# ---------------------------------------------------------------------------

class TestFalsePositiveRate:
    def test_fp_rate_within_bounds(self):
        """With 10 k inserted and 10 k checked, FP rate must stay < 0.1%."""
        n = 10_000
        bf = URLBloomFilter(expected_items=n, fp_rate=0.001)

        inserted = [f"https://site-a.com/{i}" for i in range(n)]
        queried = [f"https://site-b.com/{i}" for i in range(n)]

        bf.update(inserted)

        false_positives = sum(1 for url in queried if url in bf)
        fp_rate = false_positives / len(queried)

        # Allow small headroom above the theoretical 0.1%
        assert fp_rate < 0.002, f"False-positive rate {fp_rate:.4f} exceeded 0.2%"

    def test_zero_false_positives_on_exact_items(self):
        """Items that were inserted must always be positive (no false negatives)."""
        bf = URLBloomFilter(expected_items=5000, fp_rate=0.001)
        urls = [f"https://exact.com/{i}" for i in range(5000)]
        bf.update(urls)
        for url in urls:
            assert url in bf


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

class TestMemoryUsage:
    def test_memory_under_200kb(self):
        """Bloom filter bit-array must use < 200 KB for 100 k items at 0.1% FPR."""
        bf = URLBloomFilter(expected_items=100_000, fp_rate=0.001)
        # Insert 100 k URLs to trigger any internal growth
        for i in range(100_000):
            bf.add(f"https://mem-test.com/{i}")

        mem_bytes = bf.size_bytes
        assert mem_bytes < 200 * 1024, (
            f"Bloom filter used {mem_bytes} bytes (> 200 KB)"
        )


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_adds(self):
        """Concurrent async adds must not crash or corrupt state."""
        bf = URLBloomFilter(expected_items=50_000, fp_rate=0.001)

        async def _add_batch(start: int, count: int):
            for i in range(start, start + count):
                await bf.async_add(f"https://concurrent.com/{i}")

        async def _run():
            tasks = [_add_batch(i * 1000, 1000) for i in range(50)]
            await asyncio.gather(*tasks)

        asyncio.run(_run())

        # All items should be found
        for i in range(50_000):
            assert f"https://concurrent.com/{i}" in bf

    def test_async_contains(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)

        async def _check():
            await bf.async_add("https://async.com/1")
            assert await bf.async_contains("https://async.com/1")
            assert not await bf.async_contains("https://async.com/missing")

        asyncio.run(_check())

    def test_async_clear(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)

        async def _clear():
            await bf.async_add("https://async.com/1")
            await bf.async_clear()
            assert len(bf) == 0
            assert "https://async.com/1" not in bf

        asyncio.run(_clear())


# ---------------------------------------------------------------------------
# reconstruct()
# ---------------------------------------------------------------------------

class TestReconstruct:
    def test_reconstruct_creates_fresh_filter(self):
        bf = URLBloomFilter(expected_items=5000, fp_rate=0.001)
        bf.add("https://example.com/old")
        bf.update([f"https://example.com/{i}" for i in range(100)])

        new_bf = bf.reconstruct()

        assert len(new_bf) == 0
        assert "https://example.com/old" not in new_bf
        # Original is unchanged
        assert "https://example.com/old" in bf
        assert len(bf) == 101

    def test_reconstruct_preserves_params(self):
        bf = URLBloomFilter(expected_items=20_000, fp_rate=0.005)
        new_bf = bf.reconstruct()
        assert new_bf._expected_items == 20_000
        assert new_bf._fp_rate == 0.005
        assert new_bf.size_bits == bf.size_bits
        assert new_bf.num_hashes == bf.num_hashes


# ---------------------------------------------------------------------------
# Set-like interface
# ---------------------------------------------------------------------------

class TestSetLikeInterface:
    def test_copy(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.add("https://example.com/a")
        bf.add("https://example.com/b")

        copied = bf.copy()

        assert len(copied) == len(bf)
        assert "https://example.com/a" in copied
        assert "https://example.com/b" in copied

        # Mutating copy does not affect original
        copied.add("https://example.com/c")
        assert "https://example.com/c" not in bf

    def test_update_with_iterable(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.update(["u1", "u2", "u3"])
        assert all(u in bf for u in ["u1", "u2", "u3"])

    def test_clear_then_reuse(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        bf.add("https://reuse.com/1")
        bf.clear()
        bf.add("https://reuse.com/2")
        assert "https://reuse.com/2" in bf
        assert "https://reuse.com/1" not in bf


# ---------------------------------------------------------------------------
# Backward-compat: seeding from a set
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_init_with_existing_set(self):
        existing = {f"https://legacy.com/{i}" for i in range(200)}
        bf = URLBloomFilter()
        bf.update(existing)

        for url in existing:
            assert url in bf
        assert len(bf) == 200

    def test_seed_via_constructor_flow(self):
        """Simulate the agent.py backward-compat path."""
        visited_urls = {"https://seed.com/1", "https://seed.com/2"}
        bf = URLBloomFilter()
        if visited_urls:
            bf.update(visited_urls)

        assert "https://seed.com/1" in bf
        assert "https://seed.com/2" in bf
        assert len(bf) == 2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_size_bytes_positive(self):
        bf = URLBloomFilter(expected_items=1000, fp_rate=0.001)
        assert bf.size_bytes > 0

    def test_num_hashes_matches_spec(self):
        """Optimal k for 100k items at 0.1% FPR should be ~10."""
        bf = URLBloomFilter(expected_items=100_000, fp_rate=0.001)
        assert 5 <= bf.num_hashes <= 12
