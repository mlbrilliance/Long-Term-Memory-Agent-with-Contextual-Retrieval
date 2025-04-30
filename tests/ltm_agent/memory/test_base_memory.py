"""
Tests for base memory store implementations.

This module contains tests for the base memory store classes, ensuring they
properly implement the memory interfaces and provide the expected functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base import BaseMemoryStore, BaseVectorStore
from ltm_agent.memory.interfaces import MemoryStore, VectorStore


# Create a minimal concrete implementation of BaseMemoryStore for testing
class TestMemoryStore(BaseMemoryStore):
    """Concrete implementation of BaseMemoryStore for testing."""

    async def initialize(self) -> None:
        """Initialize the memory store."""
        self.initialized = True

    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """Implementation for adding a knowledge unit."""
        self.storage[knowledge_unit.unique_id] = knowledge_unit
        return knowledge_unit.unique_id

    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """Implementation for retrieving a knowledge unit."""
        return self.storage.get(unique_id)

    async def _delete_implementation(self, unique_id: str) -> bool:
        """Implementation for deleting a knowledge unit."""
        if unique_id in self.storage:
            del self.storage[unique_id]
            return True
        return False

    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Implementation for updating a knowledge unit."""
        if knowledge_unit.unique_id in self.storage:
            self.storage[knowledge_unit.unique_id] = knowledge_unit
            return True
        return False

    async def _search_implementation(
        self, query: str, limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Implementation for searching knowledge units."""
        results = []
        for ku in self.storage.values():
            if query.lower() in ku.original_chunk.lower():
                # Apply filter criteria if provided
                if filter_criteria:
                    matches = True
                    for key, value in filter_criteria.items():
                        if hasattr(ku, key) and getattr(ku, key) != value:
                            matches = False
                            break
                    if not matches:
                        continue

                results.append((ku, 0.9))  # Mock relevance score

        return results[:limit]

    async def _list_implementation(
        self,
        limit: int,
        offset: int,
        filter_criteria: dict[str, Any] | None,
        sort_by: str | None,
        sort_order: str,
    ) -> list[KnowledgeUnit]:
        """Implementation for listing knowledge units."""
        results = list(self.storage.values())

        # Apply filter criteria if provided
        if filter_criteria:
            filtered_results = []
            for ku in results:
                matches = True
                for key, value in filter_criteria.items():
                    if hasattr(ku, key) and getattr(ku, key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append(ku)
            results = filtered_results

        # Apply sorting if requested
        if sort_by and hasattr(results[0], sort_by) if results else False:
            reverse = sort_order.lower() == "desc"
            results.sort(key=lambda ku: getattr(ku, sort_by), reverse=reverse)

        # Apply pagination
        return results[offset : offset + limit]

    async def _count_implementation(self, filter_criteria: dict[str, Any] | None) -> int:
        """Implementation for counting knowledge units."""
        if not filter_criteria:
            return len(self.storage)

        # Count with filtering
        count = 0
        for ku in self.storage.values():
            matches = True
            for key, value in filter_criteria.items():
                if hasattr(ku, key) and getattr(ku, key) != value:
                    matches = False
                    break
            if matches:
                count += 1

        return count


# Create a minimal concrete implementation of BaseVectorStore for testing
class TestVectorStore(BaseVectorStore):
    """Concrete implementation of BaseVectorStore for testing."""

    async def initialize(self) -> None:
        """Initialize the vector store."""
        self.initialized = True

    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """Implementation for adding a knowledge unit."""
        self.storage[knowledge_unit.unique_id] = knowledge_unit
        return knowledge_unit.unique_id

    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """Implementation for retrieving a knowledge unit."""
        return self.storage.get(unique_id)

    async def _delete_implementation(self, unique_id: str) -> bool:
        """Implementation for deleting a knowledge unit."""
        if unique_id in self.storage:
            del self.storage[unique_id]
            return True
        return False

    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Implementation for updating a knowledge unit."""
        if knowledge_unit.unique_id in self.storage:
            self.storage[knowledge_unit.unique_id] = knowledge_unit
            return True
        return False

    async def _search_implementation(
        self, query: str, limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Implementation for searching knowledge units."""
        # Simple text-based search using the same logic as TestMemoryStore
        results = []
        for ku in self.storage.values():
            if query.lower() in ku.original_chunk.lower():
                # Apply filter criteria if provided
                if filter_criteria:
                    matches = True
                    for key, value in filter_criteria.items():
                        if hasattr(ku, key) and getattr(ku, key) != value:
                            matches = False
                            break
                    if not matches:
                        continue

                results.append((ku, 0.9))  # Mock relevance score

        return results[:limit]

    async def _list_implementation(
        self,
        limit: int,
        offset: int,
        filter_criteria: dict[str, Any] | None,
        sort_by: str | None,
        sort_order: str,
    ) -> list[KnowledgeUnit]:
        """Implementation for listing knowledge units."""
        # Same implementation as TestMemoryStore
        results = list(self.storage.values())

        # Apply filter criteria if provided
        if filter_criteria:
            filtered_results = []
            for ku in results:
                matches = True
                for key, value in filter_criteria.items():
                    if hasattr(ku, key) and getattr(ku, key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append(ku)
            results = filtered_results

        # Apply sorting if requested
        if sort_by and hasattr(results[0], sort_by) if results else False:
            reverse = sort_order.lower() == "desc"
            results.sort(key=lambda ku: getattr(ku, sort_by), reverse=reverse)

        # Apply pagination
        return results[offset : offset + limit]

    async def _count_implementation(self, filter_criteria: dict[str, Any] | None) -> int:
        """Implementation for counting knowledge units."""
        # Same implementation as TestMemoryStore
        if not filter_criteria:
            return len(self.storage)

        # Count with filtering
        count = 0
        for ku in self.storage.values():
            matches = True
            for key, value in filter_criteria.items():
                if hasattr(ku, key) and getattr(ku, key) != value:
                    matches = False
                    break
            if matches:
                count += 1

        return count

    async def _similarity_search_implementation(
        self, embedding: list[float], limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Implementation for similarity search."""
        # Mock implementation that returns items with embeddings
        results = []
        for ku in self.storage.values():
            if ku.embedding_vector is not None:
                # Apply filter criteria if provided
                if filter_criteria:
                    matches = True
                    for key, value in filter_criteria.items():
                        if hasattr(ku, key) and getattr(ku, key) != value:
                            matches = False
                            break
                    if not matches:
                        continue

                # Calculate a mock similarity score (in a real implementation this would use vector similarity)
                # Here we just return a fixed score of 0.8
                results.append((ku, 0.8))

        return results[:limit]

    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """Implementation for generating embeddings."""
        # Mock implementation that returns a fixed-size embedding
        # In a real implementation, this would use a model to generate the embedding
        return [0.1] * 10  # 10-dimensional mock embedding


class TestBaseMemoryStore:
    """Test suite for BaseMemoryStore."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides a test memory store."""
        return TestMemoryStore({})  # Initialize with empty dictionary as storage

    @pytest.fixture
    def sample_ku(self):
        """Fixture that provides a sample knowledge unit."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk for the base memory store",
            knowledge_source="action",
        )

    @pytest.mark.asyncio
    async def test_initialization(self, memory_store):
        """Test that the memory store initializes correctly."""
        assert memory_store.initialized is False
        await memory_store.initialize()
        assert memory_store.initialized is True

    @pytest.mark.asyncio
    async def test_automatic_initialization(self, memory_store, sample_ku):
        """Test that methods automatically initialize the store if needed."""
        assert memory_store.initialized is False
        await memory_store.add(sample_ku)
        assert memory_store.initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get(self, memory_store, sample_ku):
        """Test adding and retrieving a knowledge unit."""
        await memory_store.initialize()

        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Verify the add_implementation was called with the correct arguments
        assert unique_id == sample_ku.unique_id

        # Retrieve the knowledge unit
        retrieved_ku = await memory_store.get(unique_id)

        # Verify it's the same knowledge unit
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == sample_ku.unique_id
        assert retrieved_ku.original_chunk == sample_ku.original_chunk

    @pytest.mark.asyncio
    async def test_update(self, memory_store, sample_ku):
        """Test updating a knowledge unit."""
        await memory_store.initialize()

        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Update the knowledge unit
        updated_ku = KnowledgeUnit(
            unique_id=unique_id, original_chunk="Updated chunk", knowledge_source="feedback"
        )
        result = await memory_store.update(updated_ku)

        # Verify the update succeeded
        assert result is True

        # Retrieve the updated knowledge unit
        retrieved_ku = await memory_store.get(unique_id)

        # Verify it has the updated content
        assert retrieved_ku is not None
        assert retrieved_ku.original_chunk == "Updated chunk"
        assert retrieved_ku.knowledge_source == "feedback"

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_ku):
        """Test deleting a knowledge unit."""
        await memory_store.initialize()

        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Verify it exists
        assert await memory_store.get(unique_id) is not None

        # Delete it
        result = await memory_store.delete(unique_id)

        # Verify the delete succeeded
        assert result is True

        # Verify it no longer exists
        assert await memory_store.get(unique_id) is None


class TestBaseVectorStore:
    """Test suite for BaseVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Fixture that provides a test vector store."""
        return TestVectorStore({})  # Initialize with empty dictionary as storage

    @pytest.fixture
    def sample_ku_with_embedding(self):
        """Fixture that provides a sample knowledge unit with an embedding."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk with an embedding",
            knowledge_source="action",
            embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

    @pytest.mark.asyncio
    async def test_generate_embedding(self, vector_store):
        """Test generating an embedding."""
        await vector_store.initialize()

        # Generate an embedding
        embedding = await vector_store.generate_embedding("Test text")

        # Verify the embedding has the expected format
        assert isinstance(embedding, list)
        assert len(embedding) == 10  # Our mock implementation returns a 10-dimensional embedding
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store, sample_ku_with_embedding):
        """Test similarity search."""
        await vector_store.initialize()

        # Add a knowledge unit with an embedding
        await vector_store.add(sample_ku_with_embedding)

        # Perform a similarity search
        query_embedding = [0.2, 0.3, 0.4, 0.5, 0.6]
        results = await vector_store.similarity_search(query_embedding)

        # Verify the search returned the expected results
        assert len(results) == 1
        assert results[0][0].unique_id == sample_ku_with_embedding.unique_id
        assert isinstance(results[0][1], float)  # Similarity score should be a float

    @pytest.mark.asyncio
    async def test_inheritance(self, vector_store):
        """Test that BaseVectorStore properly inherits from BaseMemoryStore."""
        # Verify it's an instance of both classes
        assert isinstance(vector_store, BaseMemoryStore)
        assert isinstance(vector_store, BaseVectorStore)

        # Verify it implements both interfaces
        assert isinstance(vector_store, MemoryStore)
        assert isinstance(vector_store, VectorStore)
