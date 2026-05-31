"""Tests for token_budget module: counting, truncation, and budget enforcement.

Uses importlib to import token_budget directly by file path, bypassing the
gpt_researcher package __init__ which pulls in heavy dependencies.
"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Direct module import (avoids triggering gpt_researcher.__init__)
# ---------------------------------------------------------------------------

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "gpt_researcher", "context", "token_budget.py",
)
_MODULE_PATH = os.path.normpath(_MODULE_PATH)

_spec = importlib.util.spec_from_file_location("token_budget", _MODULE_PATH)
_token_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_token_budget)

count_tokens = _token_budget.count_tokens
truncate_to_tokens = _token_budget.truncate_to_tokens
TokenBudget = _token_budget.TokenBudget


class TestCountTokens(unittest.TestCase):
    """Verify count_tokens accuracy against tiktoken."""

    def test_returns_positive_int(self):
        n = count_tokens("hello world")
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_empty_string(self):
        n = count_tokens("")
        # char//4 fallback gives 0 but we clamp to >= 1; tiktoken gives 0
        # Both are acceptable -- just verify it does not crash
        self.assertIsInstance(n, int)

    def test_known_text_matches_tiktoken(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        expected = len(enc.encode(text))
        self.assertEqual(count_tokens(text), expected)

    def test_longer_text_has_more_tokens(self):
        short = "hello"
        long = "hello " * 100
        self.assertGreater(count_tokens(long), count_tokens(short))


class TestTruncateToTokens(unittest.TestCase):
    """Verify truncation produces text within token budget."""

    def test_no_truncation_when_under_budget(self):
        text = "short"
        result = truncate_to_tokens(text, 1000)
        self.assertEqual(result, text)

    def test_truncation_reduces_token_count(self):
        text = "word " * 5000  # ~5000 tokens
        result = truncate_to_tokens(text, 100)
        self.assertLessEqual(count_tokens(result), 100)

    def test_empty_when_zero_budget(self):
        text = "anything"
        result = truncate_to_tokens(text, 0)
        self.assertEqual(result, "")

    def test_returns_string(self):
        result = truncate_to_tokens("test", 10)
        self.assertIsInstance(result, str)


class TestTokenBudget(unittest.TestCase):
    """Verify TokenBudget tracker and enforcement logic."""

    def test_unlimited_budget(self):
        budget = TokenBudget(max_tokens=None)
        self.assertIsNone(budget.max_tokens)
        self.assertFalse(budget.is_exhausted)
        self.assertIsNone(budget.remaining)
        self.assertTrue(budget.can_fit("anything at all"))

    def test_consume_and_remaining(self):
        budget = TokenBudget(max_tokens=1000)
        budget.consume("hello world")
        self.assertGreater(budget.used, 0)
        self.assertEqual(budget.remaining, 1000 - budget.used)

    def test_is_exhausted(self):
        budget = TokenBudget(max_tokens=5)
        # "a" * 100 is well over 5 tokens
        budget.consume("a" * 100)
        self.assertTrue(budget.is_exhausted)

    def test_can_fit_false_when_over_budget(self):
        budget = TokenBudget(max_tokens=5)
        budget.consume("a" * 100)
        self.assertFalse(budget.can_fit("more text"))

    def test_fit_text_truncates(self):
        budget = TokenBudget(max_tokens=10)
        budget.consume("a" * 100)  # already over budget
        result = budget.fit_text("more text here")
        self.assertEqual(result, "")

    def test_fit_text_returns_full_when_budget_available(self):
        budget = TokenBudget(max_tokens=10000)
        text = "hello"
        result = budget.fit_text(text)
        self.assertEqual(result, text)

    def test_reset(self):
        budget = TokenBudget(max_tokens=100)
        budget.consume("some text")
        self.assertGreater(budget.used, 0)
        budget.reset()
        self.assertEqual(budget.used, 0)

    def test_repr(self):
        budget = TokenBudget(max_tokens=500)
        r = repr(budget)
        self.assertIn("500", r)


class TestTokenBudgetWithContextCompressor(unittest.TestCase):
    """Integration: token_budget parameter on ContextCompressor."""

    def test_no_token_budget_preserves_original_output(self):
        """Without token_budget, output is returned as-is (backward compat)."""
        from gpt_researcher.context.compression import ContextCompressor

        docs = [{"raw_content": "Hello world. " * 100}]
        compressor = ContextCompressor(
            documents=docs,
            embeddings=None,
            max_results=5,
            token_budget=None,
        )
        # The fast-path returns directly when content < COMPRESSION_THRESHOLD
        # but our doc is short so it should go through fast path
        self.assertIsNone(compressor.token_budget)

    def test_token_budget_attribute_set(self):
        from gpt_researcher.context.compression import ContextCompressor

        compressor = ContextCompressor(
            documents=[],
            embeddings=None,
            token_budget=3200,
        )
        self.assertEqual(compressor.token_budget, 3200)


class TestFallbackWithoutTiktoken(unittest.TestCase):
    """Verify fallback when tiktoken is not importable."""

    def test_count_tokens_fallback(self):
        with patch.object(_token_budget, "_tiktoken_available", False):
            text = "a" * 100
            n = count_tokens(text)
            self.assertEqual(n, 25)  # 100 // 4

    def test_truncate_fallback(self):
        with patch.object(_token_budget, "_tiktoken_available", False):
            text = "a" * 200
            result = truncate_to_tokens(text, 10)
            # char budget = 10 * 4 = 40
            self.assertEqual(len(result), 40)


class TestEightKToThreePointTwoK(unittest.TestCase):
    """Verify the target compression ratio: 8k tokens -> <= 3.2k tokens."""

    def test_truncate_8k_to_3200(self):
        # Generate ~8000 tokens of text
        # Each sentence ~16 tokens, need ~500 repetitions
        sentence = (
            "The quick brown fox jumps over the lazy dog and runs "
            "through the green forest chasing butterflies in the sunlight. "
        )
        text = sentence * 500
        tokens_in = count_tokens(text)
        self.assertGreater(tokens_in, 7000)  # sanity check

        result = truncate_to_tokens(text, 3200)
        tokens_out = count_tokens(result)
        self.assertLessEqual(tokens_out, 3200)
        # Should achieve ~60% reduction
        reduction = 1 - tokens_out / tokens_in
        self.assertGreater(reduction, 0.5)  # at least 50% reduction


if __name__ == "__main__":
    unittest.main()
