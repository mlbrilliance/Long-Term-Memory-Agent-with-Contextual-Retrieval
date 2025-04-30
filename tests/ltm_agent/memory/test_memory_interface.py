"""
Tests for memory interfaces.

This module contains tests for the memory interface contracts, ensuring that
any implementation adhering to these interfaces will function correctly with
the rest of the system.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.interfaces import MemoryStore


# Define a mock implementation of MemoryStore for testing
class MockMemoryStore(MemoryStore):
    """A mock implementation of MemoryStore for testing the interface."""

    def __init__(self):
        self.store = {}  # Simple in-memory dictionary storage

    async def add(self, knowledge_unit: KnowledgeUnit) -> str:
        self.store[knowledge_unit.unique_id] = knowledge_unit
        return knowledge_unit.unique_id

    async def get(self, unique_id: str) -> KnowledgeUnit | None:
        return self.store.get(unique_id)

    async def delete(self, unique_id: str) -> bool:
        if unique_id in self.store:
            del self.store[unique_id]
            return True
        return False

    async def update(self, knowledge_unit: KnowledgeUnit) -> bool:
        if knowledge_unit.unique_id in self.store:
            self.store[knowledge_unit.unique_id] = knowledge_unit
            return True
        return False

    async def search(
        self, query: str, limit: int = 10, filter_criteria: dict[str, Any] | None = None
    ) -> list[tuple[KnowledgeUnit, float]]:
        # Simple mock implementation that just returns knowledge units containing the query string
        results = []
        for ku in self.store.values():
            if query.lower() in ku.original_chunk.lower():
                results.append((ku, 0.9))  # Mock relevance score

        # Apply filter criteria if provided
        if filter_criteria:
            filtered_results = []
            for ku, score in results:
                matches = True
                for key, value in filter_criteria.items():
                    if hasattr(ku, key) and getattr(ku, key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append((ku, score))
            results = filtered_results

        return results[:limit]

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_criteria: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> list[KnowledgeUnit]:
        # Extract all values
        results = list(self.store.values())

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

    async def count(self, filter_criteria: dict[str, Any] | None = None) -> int:
        if not filter_criteria:
            return len(self.store)

        # Count with filtering
        count = 0
        for ku in self.store.values():
            matches = True
            for key, value in filter_criteria.items():
                if hasattr(ku, key) and getattr(ku, key) != value:
                    matches = False
                    break
            if matches:
                count += 1

        return count


class TestMemoryStore:
    """Test suite for the MemoryStore interface."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides a mock memory store for testing."""
        return MockMemoryStore()

    @pytest.fixture
    def sample_ku(self):
        """Fixture that provides a sample knowledge unit for testing."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk for memory storage", knowledge_source="action"
        )

    @pytest.mark.asyncio
    async def test_add_and_get(self, memory_store, sample_ku):
        """Test adding a knowledge unit and retrieving it by ID."""
        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Retrieve it
        retrieved_ku = await memory_store.get(unique_id)

        # Verify it's the same knowledge unit
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == sample_ku.unique_id
        assert retrieved_ku.original_chunk == sample_ku.original_chunk
        assert retrieved_ku.knowledge_source == sample_ku.knowledge_source

    @pytest.mark.asyncio
    async def test_update(self, memory_store, sample_ku):
        """Test updating a knowledge unit."""
        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Modify and update
        updated_ku = KnowledgeUnit(
            unique_id=unique_id, original_chunk="Updated chunk content", knowledge_source="feedback"
        )
        success = await memory_store.update(updated_ku)

        # Verify update was successful
        assert success is True

        # Retrieve and verify changes
        retrieved_ku = await memory_store.get(unique_id)
        assert retrieved_ku is not None
        assert retrieved_ku.original_chunk == "Updated chunk content"
        assert retrieved_ku.knowledge_source == "feedback"

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_ku):
        """Test deleting a knowledge unit."""
        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Delete it
        success = await memory_store.delete(unique_id)

        # Verify deletion was successful
        assert success is True

        # Verify it's no longer retrievable
        retrieved_ku = await memory_store.get(unique_id)
        assert retrieved_ku is None

    @pytest.mark.asyncio
    async def test_search(self, memory_store):
        """Test searching for knowledge units."""
        # Add multiple knowledge units
        ku1 = KnowledgeUnit(
            original_chunk="Python is a programming language", knowledge_source="corpus"
        )
        ku2 = KnowledgeUnit(
            original_chunk="LangChain is a framework for LLM applications",
            knowledge_source="corpus",
        )
        ku3 = KnowledgeUnit(
            original_chunk="Memory systems help agents retain information",
            knowledge_source="action",
        )

        await memory_store.add(ku1)
        await memory_store.add(ku2)
        await memory_store.add(ku3)

        # Search for "language"
        results = await memory_store.search("language")
        assert len(results) == 1
        assert results[0][0].original_chunk == "Python is a programming language"

        # Search for "memory" with a source filter
        results = await memory_store.search(
            "memory", filter_criteria={"knowledge_source": "action"}
        )
        assert len(results) == 1
        assert results[0][0].original_chunk == "Memory systems help agents retain information"

        # Search with no matches
        results = await memory_store.search("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_list_and_count(self, memory_store):
        """Test listing and counting knowledge units."""
        # Add multiple knowledge units
        for i in range(5):
            ku = KnowledgeUnit(
                original_chunk=f"Test chunk {i}",
                knowledge_source="corpus" if i % 2 == 0 else "action",
            )
            await memory_store.add(ku)

        # Count all
        count = await memory_store.count()
        assert count == 5

        # Count with filter
        count = await memory_store.count(filter_criteria={"knowledge_source": "corpus"})
        assert count == 3  # 0, 2, 4 are corpus

        # List all
        results = await memory_store.list()
        assert len(results) == 5

        # List with pagination
        results = await memory_store.list(limit=2, offset=1)
        assert len(results) == 2

        # List with filter
        results = await memory_store.list(filter_criteria={"knowledge_source": "action"})
        assert len(results) == 2  # 1, 3 are action
        assert all(ku.knowledge_source == "action" for ku in results)


# Note: VectorStore tests would be implemented similarly, but would require a mock
# implementation that supports vector operations, which would be more complex.
# Those tests would be added in a separate test file.
