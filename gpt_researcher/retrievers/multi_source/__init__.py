"""Multi-source search retriever for GPT Researcher.

This module provides the MultiSourceSearch class that concurrently queries
multiple retriever backends and merges/deduplicates results.
"""

from .multi_source_search import MultiSourceSearch

__all__ = ["MultiSourceSearch"]
