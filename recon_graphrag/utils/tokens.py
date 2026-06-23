"""Token counting and packing utilities.

These primitives are intentionally dependency-light. The default
``ApproximateTokenCounter`` uses a configurable character-to-token ratio so that
budgeting works without installing a tokenizer. An optional ``tiktoken`` adapter
is available when the library is installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting and truncation."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return the longest prefix of ``text`` that fits in ``max_tokens``."""
        ...


class ApproximateTokenCounter:
    """Lightweight token estimator.

    Uses ``ceil(len(text) / ratio)`` as the token count. This is a coarse
    estimate suitable for budgeting when no exact tokenizer is available.
    Callers requiring provider-level accuracy should pass an exact counter.
    """

    DEFAULT_RATIO = 4.0

    def __init__(self, ratio: float | None = None):
        ratio = ratio if ratio is not None else self.DEFAULT_RATIO
        if ratio <= 0:
            raise ValueError("ratio must be > 0")
        self._ratio = ratio

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._ratio)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if max_tokens == 0:
            return ""
        if not text:
            return ""
        max_chars = int(max_tokens * self._ratio)
        return text[:max_chars]


class TiktokenTokenCounter:
    """Exact token counter backed by ``tiktoken``, when available.

    Falls back to a clear import-time error if ``tiktoken`` is not installed.
    """

    def __init__(self, model: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TiktokenTokenCounter requires the 'tiktoken' package. "
                "Install it or use ApproximateTokenCounter."
            ) from exc
        self._encoding = tiktoken.get_encoding(model)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if max_tokens == 0:
            return ""
        if not text:
            return ""
        encoded = self._encoding.encode(text)
        return self._encoding.decode(encoded[:max_tokens])


def create_token_counter(name: str = "approximate", **kwargs) -> TokenCounter:
    """Factory for token counters.

    Supported names:
    - ``"approximate"``: ``ApproximateTokenCounter`` (always available).
      Accepts ``ratio``.
    - ``"tiktoken"``: ``TiktokenTokenCounter`` (requires ``tiktoken``).
      Accepts ``model`` (defaults to ``"cl100k_base"``).
    """
    if name == "approximate":
        return ApproximateTokenCounter(**kwargs)
    if name == "tiktoken":
        return TiktokenTokenCounter(**kwargs)
    raise ValueError(f"Unknown token counter: {name!r}")


def count_tokens(text: str, counter: TokenCounter | None = None) -> int:
    """Convenience helper: count tokens in ``text``."""
    return (counter or ApproximateTokenCounter()).count(text)


def truncate_text(text: str, max_tokens: int, counter: TokenCounter | None = None) -> str:
    """Convenience helper: truncate ``text`` to ``max_tokens``."""
    return (counter or ApproximateTokenCounter()).truncate(text, max_tokens)


__all__ = [
    "TokenCounter",
    "ApproximateTokenCounter",
    "TiktokenTokenCounter",
    "create_token_counter",
    "count_tokens",
    "truncate_text",
]
