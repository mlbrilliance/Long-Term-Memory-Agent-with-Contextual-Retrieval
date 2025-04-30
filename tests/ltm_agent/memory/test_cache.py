"""
Tests for the memory caching system.

This module tests the caching functionality to ensure efficient
retrieval operations.
"""

import sys
import time
from pathlib import Path
from typing import Any

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.cache import KnowledgeCache, LRUCache, cache_key_for_query, with_cache
from ltm_agent.utils.test_utils import create_test_knowledge_unit


class TestLRUCache:
    """Test suite for the LRU cache."""

    def test_basic_operations(self):
        """Test basic get and put operations."""
        cache = LRUCache(max_size=3, ttl_seconds=10)

        # Put items in the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Get items from the cache
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") is None  # Non-existent key

    def test_cache_eviction(self):
        """Test that the cache evicts the least recently used item when full."""
        cache = LRUCache(max_size=2, ttl_seconds=10)

        # Fill the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it more recently used
        assert cache.get("key1") == "value1"

        # Add a new item, should evict key2
        cache.put("key3", "value3")

        # Check that key2 was evicted
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_invalidate(self):
        """Test that items can be invalidated manually."""
        cache = LRUCache(max_size=3, ttl_seconds=10)

        # Put items in the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Invalidate one item
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        # Try to invalidate a non-existent key
        assert cache.invalidate("key3") is False

    def test_ttl_expiration(self):
        """Test that items expire after their TTL."""
        cache = LRUCache(max_size=3, ttl_seconds=1)  # 1 second TTL

        # Put an item in the cache
        cache.put("key1", "value1")

        # Access it immediately
        assert cache.get("key1") == "value1"

        # Wait for it to expire
        time.sleep(1.1)

        # Should be gone now
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing the entire cache."""
        cache = LRUCache(max_size=3, ttl_seconds=10)

        # Put items in the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Clear the cache
        cache.clear()

        # All items should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestKnowledgeCache:
    """Test suite for the knowledge cache."""

    def test_get_put_by_id(self):
        """Test getting and putting knowledge units by ID."""
        cache = KnowledgeCache(max_size=10, ttl_seconds=10)

        # Create a test knowledge unit
        ku = create_test_knowledge_unit(content="Test caching", source="action")

        # Put it in the cache
        cache.put(ku)

        # Get it back
        retrieved_ku = cache.get_by_id(ku.unique_id)

        # Check it's the same
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == ku.unique_id
        assert retrieved_ku.original_chunk == ku.original_chunk

    def test_invalidate_knowledge(self):
        """Test invalidating cached knowledge units."""
        cache = KnowledgeCache(max_size=10, ttl_seconds=10)

        # Create test knowledge units
        ku1 = create_test_knowledge_unit(content="Test unit 1", source="action")
        ku2 = create_test_knowledge_unit(content="Test unit 2", source="corpus")

        # Put them in the cache
        cache.put(ku1)
        cache.put(ku2)

        # Invalidate one
        assert cache.invalidate(ku1.unique_id) is True

        # Check it's gone
        assert cache.get_by_id(ku1.unique_id) is None
        assert cache.get_by_id(ku2.unique_id) is not None

    def test_query_caching(self):
        """Test caching and retrieving query results."""
        cache = KnowledgeCache(max_size=10, ttl_seconds=10)

        # Create test knowledge units
        ku1 = create_test_knowledge_unit(content="Test query 1", source="action")
        ku2 = create_test_knowledge_unit(content="Test query 2", source="corpus")

        # Create query results
        results = [(ku1, 0.9), (ku2, 0.8)]

        # Generate query key
        query_key = cache_key_for_query("test query", 10, {"source": "action"})

        # Cache the results
        cache.cache_query_results(query_key, results)

        # Retrieve the results
        cached_results = cache.get_query_results(query_key)

        # Check they match
        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0][0].unique_id == ku1.unique_id
        assert cached_results[1][0].unique_id == ku2.unique_id
        assert cached_results[0][1] == 0.9
        assert cached_results[1][1] == 0.8

    def test_query_key_generation(self):
        """Test that query key generation is consistent."""
        # Simple query
        key1 = cache_key_for_query("test", 10)
        assert key1 == "q:test|l:10|f:"

        # Query with filter
        key2 = cache_key_for_query("test", 10, {"source": "action"})
        assert key2 == "q:test|l:10|f:source:action"

        # Query with multiple filters
        key3 = cache_key_for_query("test", 10, {"source": "action", "tag": "important"})
        # Order should be consistent
        assert key3 == "q:test|l:10|f:source:action,tag:important"

    def test_clear_knowledge_cache(self):
        """Test clearing the entire knowledge cache."""
        cache = KnowledgeCache(max_size=10, ttl_seconds=10)

        # Create a test knowledge unit
        ku = create_test_knowledge_unit(content="Test clear cache", source="action")

        # Put it in the cache
        cache.put(ku)

        # Create query results
        results = [(ku, 0.9)]
        query_key = cache_key_for_query("test clear", 10)
        cache.cache_query_results(query_key, results)

        # Clear the cache
        cache.clear()

        # Both ID and query caches should be empty
        assert cache.get_by_id(ku.unique_id) is None
        assert cache.get_query_results(query_key) is None


class MockMemoryStore:
    """Mock memory store for testing the cache decorator."""

    def __init__(self):
        self.items = {}
        self.get_count = 0
        self.add_count = 0
        self.search_count = 0

    async def get(self, unique_id: str) -> KnowledgeUnit | None:
        """Mock get operation."""
        self.get_count += 1
        return self.items.get(unique_id)

    async def add(self, knowledge_unit: KnowledgeUnit) -> str:
        """Mock add operation."""
        self.add_count += 1
        self.items[knowledge_unit.unique_id] = knowledge_unit
        return knowledge_unit.unique_id

    async def search(
        self, query: str, limit: int = 10, filter_criteria: dict[str, Any] | None = None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Mock search operation."""
        self.search_count += 1
        # Simple mock that returns all items
        return [(ku, 0.9) for ku in self.items.values()][:limit]


class TestCacheDecorator:
    """Test suite for the cache decorator."""

    @pytest.mark.asyncio
    async def test_with_cache_decorator(self):
        """Test that the with_cache decorator properly caches results."""
        # Create a mock store and a cache
        store = MockMemoryStore()
        cache = KnowledgeCache(max_size=10, ttl_seconds=10)

        # Apply the decorator to methods
        store.get = with_cache(cache)(store.get)
        store.add = with_cache(cache)(store.add)
        store.search = with_cache(cache)(store.search)

        # Create a test knowledge unit
        ku = create_test_knowledge_unit(content="Test decorator", source="action")

        # Add it
        await store.add(ku)
        assert store.add_count == 1

        # Get it (should hit the store)
        result1 = await store.get(ku.unique_id)
        assert result1 is not None
        assert result1.unique_id == ku.unique_id
        assert store.get_count == 1

        # Get it again (should be cached)
        result2 = await store.get(ku.unique_id)
        assert result2 is not None
        assert result2.unique_id == ku.unique_id
        # Counter shouldn't increase because it's cached
        assert store.get_count == 1

        # Search (should hit the store)
        search_results1 = await store.search("test", 10)
        assert len(search_results1) == 1
        assert store.search_count == 1

        # Search again with same parameters (should be cached)
        search_results2 = await store.search("test", 10)
        assert len(search_results2) == 1
        # Counter shouldn't increase because it's cached
        assert store.search_count == 1
