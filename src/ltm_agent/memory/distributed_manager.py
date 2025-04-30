"""
Distributed memory manager with caching and sharding capabilities.

This module provides a distributed approach to memory management,
enabling horizontal scaling, fault tolerance, and improved performance.
"""

import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from typing import Any

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)


class LRUCache:
    """
    LRU (Least Recently Used) Cache implementation.

    This cache automatically evicts least recently used items when it reaches capacity.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to store
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """
        Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None

            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key: str, value: Any) -> None:
        """
        Put an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)

            self.cache[key] = value

            # Evict least recently used item if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def remove(self, key: str) -> bool:
        """
        Remove an item from the cache.

        Args:
            key: Cache key

        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()

    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)


class ShardedVectorStore(VectorStore):
    """
    Sharded vector store that distributes data across multiple stores.

    This class enables horizontal scaling by partitioning data across
    multiple vector store instances, while presenting a unified interface.
    """

    def __init__(
        self, stores: list[VectorStore], strategy: str = "consistent_hash", replicas: int = 1
    ):
        """
        Initialize the sharded vector store.

        Args:
            stores: List of vector store instances to use as shards
            strategy: Sharding strategy ('consistent_hash', 'mod_hash', or 'random')
            replicas: Number of replicas for each knowledge unit
        """
        if not stores:
            raise ValueError("At least one store is required")

        self.stores = stores
        self.strategy = strategy
        self.replicas = min(replicas, len(stores))

        # Virtual nodes for consistent hashing
        if strategy == "consistent_hash":
            self.virtual_nodes = 100
            self.ring = self._build_hash_ring()

        # Cache for routing info
        self.routing_cache = {}

    def _build_hash_ring(self) -> dict[int, int]:
        """
        Build the consistent hash ring.

        Returns:
            Dictionary mapping hash positions to store indices
        """
        ring = {}

        for i, _ in enumerate(self.stores):
            for j in range(self.virtual_nodes):
                key = f"{i}:{j}"
                hash_key = self._hash_key(key)
                ring[hash_key] = i

        return ring

    def _hash_key(self, key: str) -> int:
        """
        Hash a key to an integer.

        Args:
            key: String to hash

        Returns:
            Integer hash value
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_store_indices(self, key: str) -> list[int]:
        """
        Get store indices for a key based on sharding strategy.

        Args:
            key: Key to determine sharding

        Returns:
            List of store indices
        """
        # Check cache first
        if key in self.routing_cache:
            return self.routing_cache[key]

        indices = []

        if self.strategy == "consistent_hash":
            # Find closest points on the hash ring
            key_hash = self._hash_key(key)
            sorted_keys = sorted(self.ring.keys())

            # Find the first point on the ring
            for ring_key in sorted_keys:
                if ring_key >= key_hash:
                    indices.append(self.ring[ring_key])
                    break

            # If we didn't find a point, use the first one (wrap around)
            if not indices:
                indices.append(self.ring[sorted_keys[0]])

            # Add additional replicas if needed, avoiding duplicates
            remaining = set(range(len(self.stores))) - set(indices)
            indices.extend(list(remaining)[: self.replicas - 1])

        elif self.strategy == "mod_hash":
            # Simple modulo-based hashing
            primary = self._hash_key(key) % len(self.stores)
            indices.append(primary)

            # Add replicas
            for i in range(1, self.replicas):
                replica = (primary + i) % len(self.stores)
                indices.append(replica)

        else:  # random
            # Random allocation
            primary = random.randint(0, len(self.stores) - 1)
            indices.append(primary)

            # Add replicas
            remaining = set(range(len(self.stores))) - {primary}
            indices.extend(random.sample(remaining, min(self.replicas - 1, len(remaining))))

        # Update cache
        self.routing_cache[key] = indices
        return indices

    def add(self, unit: KnowledgeUnit) -> None:
        """
        Add a knowledge unit to the sharded store.

        Args:
            unit: Knowledge unit to add
        """
        # Get store indices for this unit
        indices = self._get_store_indices(unit.unique_id)

        # Add to all replicas
        for idx in indices:
            self.stores[idx].add(unit)

    def update(self, unit: KnowledgeUnit) -> None:
        """
        Update a knowledge unit in the sharded store.

        Args:
            unit: Knowledge unit to update
        """
        # Get store indices for this unit
        indices = self._get_store_indices(unit.unique_id)

        # Update all replicas
        for idx in indices:
            self.stores[idx].update(unit)

    def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the sharded store.

        Args:
            unique_id: ID of the knowledge unit to delete

        Returns:
            True if deleted from at least one store
        """
        # Get store indices for this unit
        indices = self._get_store_indices(unique_id)

        # Delete from all replicas
        success = False
        for idx in indices:
            if self.stores[idx].delete(unique_id):
                success = True

        # Clear routing cache
        if unique_id in self.routing_cache:
            del self.routing_cache[unique_id]

        return success

    def get(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get a knowledge unit by ID from the sharded store.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        # Get store indices for this unit
        indices = self._get_store_indices(unique_id)

        # Try to get from any replica
        for idx in indices:
            unit = self.stores[idx].get(unique_id)
            if unit:
                return unit

        return None

    def search(
        self, embedding: list[float], limit: int = 5, threshold: float = 0.0
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for similar knowledge units across all shards.

        Args:
            embedding: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (knowledge_unit, similarity_score) tuples
        """
        # Search across all stores
        all_results = []

        for store in self.stores:
            results = store.search(embedding, limit=limit, threshold=threshold)
            all_results.extend(results)

        # Remove duplicates (keep highest score)
        unique_results = {}
        for unit, score in all_results:
            if unit.unique_id not in unique_results or score > unique_results[unit.unique_id][1]:
                unique_results[unit.unique_id] = (unit, score)

        # Sort by score and limit
        results = list(unique_results.values())
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def list_all(self) -> list[KnowledgeUnit]:
        """
        List all knowledge units across all stores.

        Returns:
            List of all knowledge units
        """
        # Get all units from all stores
        all_units = []
        unique_ids = set()

        for store in self.stores:
            units = store.list_all()
            for unit in units:
                if unit.unique_id not in unique_ids:
                    all_units.append(unit)
                    unique_ids.add(unit.unique_id)

        return all_units

    def clear(self) -> None:
        """Clear all knowledge units from all stores."""
        for store in self.stores:
            store.clear()

        self.routing_cache.clear()

    def count(self) -> int:
        """
        Count the total number of unique knowledge units.

        Returns:
            Number of unique knowledge units
        """
        # This is expensive as we need to get all units and deduplicate
        # In a production system, we would maintain a separate counter
        unique_ids = set()

        for store in self.stores:
            units = store.list_all()
            for unit in units:
                unique_ids.add(unit.unique_id)

        return len(unique_ids)


class DistributedMemoryManager(EnhancedMemoryManager):
    """
    Memory manager with distributed caching and sharding capabilities.

    This class extends the EnhancedMemoryManager with:
    1. Distributed caching for frequently accessed knowledge
    2. Sharding for horizontal scaling
    3. Replication for fault tolerance
    4. Automatic retry mechanisms
    """

    def __init__(
        self,
        memory_stores: list[VectorStore],
        cache_size: int = 10000,
        sharding_strategy: str = "consistent_hash",
        replicas: int = 2,
        max_retries: int = 3,
        contextualizer=None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the distributed memory manager.

        Args:
            memory_stores: List of vector stores to use as shards
            cache_size: Size of the in-memory LRU cache
            sharding_strategy: Strategy for distributing data ('consistent_hash', 'mod_hash', or 'random')
            replicas: Number of replicas for each knowledge unit
            max_retries: Maximum number of retry attempts for operations
            contextualizer: Contextualizer for processing knowledge
            config: Additional configuration parameters
        """
        # Create sharded store
        sharded_store = ShardedVectorStore(
            stores=memory_stores, strategy=sharding_strategy, replicas=replicas
        )

        # Initialize parent class
        super().__init__(memory_store=sharded_store, contextualizer=contextualizer, config=config)

        # Create LRU cache
        self.cache = LRUCache(max_size=cache_size)
        self.max_retries = max_retries

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_reset_time = None
        self.failure_count = 0
        self.failure_threshold = 5
        self.circuit_timeout = 60  # seconds

    def _with_retry(self, operation, *args, **kwargs):
        """
        Execute an operation with automatic retry.

        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retries fail
        """
        # Check circuit breaker
        if self.circuit_open:
            current_time = time.time()

            if self.circuit_reset_time and current_time >= self.circuit_reset_time:
                # Reset circuit breaker
                logger.info("Resetting circuit breaker")
                self.circuit_open = False
                self.failure_count = 0
                self.circuit_reset_time = None
            else:
                # Circuit is open, fast fail
                raise RuntimeError("Circuit breaker is open, operation rejected")

        # Execute with retry
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)

                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = 0

                return result

            except Exception as e:
                last_exception = e
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")

                # Increment failure count
                self.failure_count += 1

                # Check if circuit breaker should open
                if self.failure_count >= self.failure_threshold:
                    logger.warning("Circuit breaker opened due to repeated failures")
                    self.circuit_open = True
                    self.circuit_reset_time = time.time() + self.circuit_timeout
                    raise RuntimeError(f"Circuit breaker opened: {e}") from e

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    backoff = (2**attempt) * 0.1
                    time.sleep(backoff)

        # All retries failed
        if last_exception:
            raise last_exception

        return None

    def add_knowledge(
        self, content: str, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> KnowledgeUnit:
        """
        Add new knowledge to memory with retry and caching.

        Args:
            content: Text content to add
            source: Optional source of the knowledge
            metadata: Optional metadata about the knowledge

        Returns:
            Added knowledge unit
        """

        # Define operation
        def _add():
            unit = super().add_knowledge(content, source, metadata)
            # Add to cache
            self.cache.put(unit.unique_id, unit)
            return unit

        # Execute with retry
        return self._with_retry(_add)

    def update_knowledge(
        self, unique_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> KnowledgeUnit | None:
        """
        Update existing knowledge with retry and cache update.

        Args:
            unique_id: ID of knowledge unit to update
            content: New content for the knowledge unit
            metadata: Optional updated metadata

        Returns:
            Updated knowledge unit or None if not found
        """

        # Define operation
        def _update():
            unit = super().update_knowledge(unique_id, content, metadata)

            # Update cache if successful
            if unit:
                self.cache.put(unit.unique_id, unit)

            # Remove from cache if not found or update failed
            else:
                self.cache.remove(unique_id)

            return unit

        # Execute with retry
        return self._with_retry(_update)

    def delete_knowledge(self, unique_id: str) -> bool:
        """
        Delete knowledge with retry and cache update.

        Args:
            unique_id: ID of knowledge unit to delete

        Returns:
            True if deleted, False if not found
        """

        # Define operation
        def _delete():
            result = super().delete_knowledge(unique_id)

            # Remove from cache
            self.cache.remove(unique_id)

            return result

        # Execute with retry
        return self._with_retry(_delete)

    def get_knowledge(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get knowledge unit by ID with caching.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        # Check cache first
        cached = self.cache.get(unique_id)
        if cached:
            return cached

        # Define operation
        def _get():
            unit = super().get_knowledge(unique_id)

            # Add to cache if found
            if unit:
                self.cache.put(unit.unique_id, unit)

            return unit

        # Execute with retry
        return self._with_retry(_get)

    def retrieve_relevant(
        self, query: str, limit: int = 5, threshold: float = 0.0
    ) -> list[KnowledgeUnit]:
        """
        Retrieve knowledge relevant to the query with caching.

        Args:
            query: Query string to find relevant knowledge
            limit: Maximum number of units to return
            threshold: Minimum relevance score (0.0-1.0)

        Returns:
            List of relevant knowledge units
        """

        # Can't easily cache this operation as it depends on the query
        # Just use retry
        def _retrieve():
            return super().retrieve_relevant(query, limit, threshold)

        return self._with_retry(_retrieve)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.cache.max_size,
            "utilization": len(self.cache) / self.cache.max_size if self.cache.max_size > 0 else 0,
        }

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """
        Get status of the circuit breaker.

        Returns:
            Dictionary with circuit breaker status
        """
        reset_in = None
        if self.circuit_reset_time:
            reset_in = max(0, self.circuit_reset_time - time.time())

        return {
            "open": self.circuit_open,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "reset_in_seconds": reset_in,
        }
