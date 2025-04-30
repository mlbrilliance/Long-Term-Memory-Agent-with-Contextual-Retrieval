"""
Caching utilities for memory operations.

This module provides caching mechanisms to improve performance for
frequently accessed knowledge units and reduce database load.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from ltm_agent.core.models import KnowledgeUnit

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
V = TypeVar("V")


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.

    This class provides a size-limited cache that evicts the least recently used
    items when the cache reaches its capacity.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to store in the cache
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, float] = {}
        self.expiry_times: dict[str, float] = {}
        logger.info(f"Initialized LRU cache with max_size={max_size}, ttl={ttl_seconds}s")

    def get(self, key: str) -> Any | None:
        """
        Get an item from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        # Check if key exists and hasn't expired
        current_time = time.time()
        if key in self.cache and key in self.expiry_times and current_time < self.expiry_times[key]:
            # Update access time
            self.access_times[key] = current_time
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]

        # Remove if expired
        if (
            key in self.cache
            and key in self.expiry_times
            and current_time >= self.expiry_times[key]
        ):
            self._remove(key)
            logger.debug(f"Removed expired cache entry for key: {key}")

        logger.debug(f"Cache miss for key: {key}")
        return None

    def put(self, key: str, value: Any) -> None:
        """
        Put an item in the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        current_time = time.time()

        # If the cache is full, remove the least recently used item
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        # Add or update the entry
        self.cache[key] = value
        self.access_times[key] = current_time
        self.expiry_times[key] = current_time + self.ttl_seconds
        logger.debug(f"Added/updated cache entry for key: {key}")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: The cache key to invalidate

        Returns:
            True if the key was found and invalidated, False otherwise
        """
        if key in self.cache:
            self._remove(key)
            logger.debug(f"Invalidated cache entry for key: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        self.access_times.clear()
        self.expiry_times.clear()
        logger.info("Cleared all cache entries")

    def _remove(self, key: str) -> None:
        """
        Remove an item from the cache.

        Args:
            key: The cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.expiry_times:
            del self.expiry_times[key]

    def _evict_lru(self) -> None:
        """Evict the least recently used item from the cache."""
        if not self.access_times:
            return

        # Find the least recently accessed key
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove(lru_key)
        logger.debug(f"Evicted LRU cache entry with key: {lru_key}")


class KnowledgeCache:
    """
    Specialized cache for knowledge units with additional query capabilities.

    This class extends the basic LRU cache with functionality specific to
    caching and retrieving knowledge units.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 600):
        """
        Initialize the knowledge cache.

        Args:
            max_size: Maximum number of knowledge units to cache
            ttl_seconds: Time-to-live for cached knowledge units in seconds
        """
        self.id_cache = LRUCache(max_size, ttl_seconds)
        self.query_cache = LRUCache(
            max_size // 10, ttl_seconds // 2
        )  # Smaller, shorter-lived query cache
        logger.info(f"Initialized knowledge cache with max_size={max_size}, ttl={ttl_seconds}s")

    def get_by_id(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get a knowledge unit by its unique ID.

        Args:
            unique_id: The unique ID of the knowledge unit

        Returns:
            The cached knowledge unit or None if not found
        """
        cached = self.id_cache.get(unique_id)
        return cached

    def put(self, knowledge_unit: KnowledgeUnit) -> None:
        """
        Cache a knowledge unit.

        Args:
            knowledge_unit: The knowledge unit to cache
        """
        self.id_cache.put(knowledge_unit.unique_id, knowledge_unit)

    def invalidate(self, unique_id: str) -> bool:
        """
        Invalidate a cached knowledge unit.

        Args:
            unique_id: The unique ID of the knowledge unit to invalidate

        Returns:
            True if the knowledge unit was found and invalidated, False otherwise
        """
        # Invalidate query cache when a unit is invalidated
        # since query results might include this unit
        self.query_cache.clear()

        return self.id_cache.invalidate(unique_id)

    def get_query_results(self, query_key: str) -> list[tuple[KnowledgeUnit, float]] | None:
        """
        Get cached query results.

        Args:
            query_key: The cache key for the query

        Returns:
            Cached query results or None if not found
        """
        return self.query_cache.get(query_key)

    def cache_query_results(
        self, query_key: str, results: list[tuple[KnowledgeUnit, float]]
    ) -> None:
        """
        Cache query results.

        Args:
            query_key: The cache key for the query
            results: The query results to cache
        """
        self.query_cache.put(query_key, results)

        # Also cache individual knowledge units
        for ku, _ in results:
            self.put(ku)

    def clear(self) -> None:
        """Clear all cached knowledge units and query results."""
        self.id_cache.clear()
        self.query_cache.clear()
        logger.info("Cleared knowledge cache")


def cache_key_for_query(
    query: str, limit: int, filter_criteria: dict[str, Any] | None = None
) -> str:
    """
    Generate a cache key for a query.

    Args:
        query: The search query
        limit: The result limit
        filter_criteria: Optional filter criteria

    Returns:
        A cache key string
    """
    filter_str = ""
    if filter_criteria:
        # Sort keys for consistent key generation
        sorted_items = sorted(filter_criteria.items())
        filter_str = ",".join(f"{k}:{v}" for k, v in sorted_items)

    return f"q:{query}|l:{limit}|f:{filter_str}"


def with_cache(
    cache: KnowledgeCache,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for caching async function results.

    Args:
        cache: The knowledge cache to use

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Extract ID for get operations
            if func.__name__ == "get" and len(args) > 1:
                unique_id = args[1]  # args[0] is self
                cached = cache.get_by_id(unique_id)
                if cached is not None:
                    return cached

            # Handle search operations
            elif func.__name__ in ("search", "similarity_search") and len(args) > 1:
                query = args[1]  # args[0] is self
                limit = kwargs.get("limit", 10)
                filter_criteria = kwargs.get("filter_criteria")

                query_key = cache_key_for_query(str(query), limit, filter_criteria)

                cached_results = cache.get_query_results(query_key)
                if cached_results is not None:
                    return cached_results

            # Call the original function
            result = await func(*args, **kwargs)

            # Cache the result
            if func.__name__ == "get" and result is not None:
                cache.put(result)
            elif func.__name__ in ("add", "update") and result is not None:
                # For add, result is the ID; for update, we need the KnowledgeUnit
                if func.__name__ == "add" and len(args) > 1:
                    knowledge_unit = args[1]  # args[0] is self
                    cache.put(knowledge_unit)
                # For update, invalidate the cache
                elif func.__name__ == "update" and len(args) > 1:
                    knowledge_unit = args[1]  # args[0] is self
                    cache.invalidate(knowledge_unit.unique_id)
                    cache.put(knowledge_unit)
            elif func.__name__ == "delete" and result and len(args) > 1:
                unique_id = args[1]  # args[0] is self
                cache.invalidate(unique_id)
            elif func.__name__ in ("search", "similarity_search"):
                query = args[1]  # args[0] is self
                limit = kwargs.get("limit", 10)
                filter_criteria = kwargs.get("filter_criteria")

                query_key = cache_key_for_query(str(query), limit, filter_criteria)

                cache.cache_query_results(query_key, result)

            return result

        return wrapper

    return decorator
