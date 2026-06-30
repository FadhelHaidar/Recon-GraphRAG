"""Retriever protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from recon_graphrag.models.types import SearchResult


@runtime_checkable
class BaseRetriever(Protocol):
    """Protocol for retriever implementations.

    All search modes (local, global, drift) implement this protocol.
    """

    async def search(self, query: str, **kwargs) -> SearchResult:
        """Run a search and return structured results."""
        ...
