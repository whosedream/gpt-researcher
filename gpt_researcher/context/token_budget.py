"""Token counting and budget enforcement utilities.

Provides accurate token counting via tiktoken with a graceful fallback
to a char/4 heuristic when tiktoken is unavailable.

Classes:
    TokenBudget: Tracks and enforces a token budget across context assembly.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tiktoken lazy import with fallback
# ---------------------------------------------------------------------------

_tiktoken_available = False
try:
    import tiktoken as _tiktoken

    _tiktoken_available = True
except ImportError:
    _tiktoken = None  # type: ignore[assignment]

DEFAULT_ENCODING = "cl100k_base"


def count_tokens(text: str, model: str = DEFAULT_ENCODING) -> int:
    """Return the token count for *text*.

    Uses tiktoken when available; falls back to ``len(text) // 4``.
    """
    if _tiktoken_available:
        try:
            enc = _tiktoken.get_encoding(model)
            return len(enc.encode(text))
        except Exception:
            logger.debug("tiktoken encoding failed, falling back to char//4")
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int, model: str = DEFAULT_ENCODING) -> str:
    """Truncate *text* to at most *max_tokens* tokens.

    Decodes back to a string so the result is always valid text.
    """
    if max_tokens <= 0:
        return ""
    if _tiktoken_available:
        try:
            enc = _tiktoken.get_encoding(model)
            token_ids = enc.encode(text)
            if len(token_ids) <= max_tokens:
                return text
            return enc.decode(token_ids[:max_tokens])
        except Exception:
            logger.debug("tiktoken truncation failed, falling back to char-based")
    # Fallback: estimate char budget from token budget
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget]


class TokenBudget:
    """Mutable token budget tracker.

    Parameters
    ----------
    max_tokens : int | None
        Hard token limit.  ``None`` means unlimited (original behaviour).
    """

    def __init__(self, max_tokens: Optional[int] = None) -> None:
        self.max_tokens = max_tokens
        self._used: int = 0

    # -- properties ----------------------------------------------------------

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> Optional[int]:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self._used)

    @property
    def is_exhausted(self) -> bool:
        if self.max_tokens is None:
            return False
        return self._used >= self.max_tokens

    # -- mutation ------------------------------------------------------------

    def consume(self, text: str, model: str = DEFAULT_ENCODING) -> int:
        """Count tokens in *text*, add to running total, return tokens added."""
        n = count_tokens(text, model=model)
        self._used += n
        return n

    def can_fit(self, text: str, model: str = DEFAULT_ENCODING) -> bool:
        """Return True if *text* fits within the remaining budget."""
        if self.max_tokens is None:
            return True
        n = count_tokens(text, model=model)
        return (self._used + n) <= self.max_tokens

    def fit_text(self, text: str, model: str = DEFAULT_ENCODING) -> str:
        """Return *text* truncated to fit the remaining budget.

        If no budget is set the full text is returned unchanged.
        """
        if self.max_tokens is None:
            return text
        budget_left = max(0, self.max_tokens - self._used)
        return truncate_to_tokens(text, budget_left, model=model)

    def reset(self) -> None:
        self._used = 0

    def __repr__(self) -> str:
        return (
            f"TokenBudget(max={self.max_tokens}, "
            f"used={self._used}, remaining={self.remaining})"
        )
