"""
Tests for the memory manager.

This module contains tests for the MemoryManager class, which provides a
high-level interface for interacting with the agent's memory system.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


class TestMemoryManager:
    """Test suite for the MemoryManager class."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides a mock memory store for testing."""
        store = InMemoryVectorStore(embedding_dim=10)
        return store

    @pytest.fixture
    async def initialized_memory_manager(self, memory_store):
        """Fixture that provides an initialized memory manager for testing."""
        manager = MemoryManager(memory_store=memory_store)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_initialization(self, memory_store):
        """Test that the memory manager initializes correctly."""
        # Create a memory manager
        manager = MemoryManager(memory_store=memory_store)
        assert manager.initialized is False

        # Initialize it
        await manager.initialize()
        assert manager.initialized is True
        assert memory_store.initialized is True

    @pytest.mark.asyncio
    async def test_add_knowledge(self, initialized_memory_manager):
        """Test adding knowledge to memory."""
        # Add a knowledge unit
        unique_id = await initialized_memory_manager.add_knowledge(
            content="Test content", source="action", context="Test context"
        )

        # Verify it was added
        assert unique_id is not None

        # Retrieve it and verify the contents
        knowledge_unit = await initialized_memory_manager.retrieve_knowledge(unique_id)
        assert knowledge_unit is not None
        assert knowledge_unit.original_chunk == "Test content"
        assert knowledge_unit.knowledge_source == "action"
        assert knowledge_unit.contextual_text == "Test context"

    @pytest.mark.asyncio
    async def test_update_knowledge(self, initialized_memory_manager):
        """Test updating knowledge in memory."""
        # Add a knowledge unit
        unique_id = await initialized_memory_manager.add_knowledge(
            content="Original content", source="action"
        )

        # Update it
        result = await initialized_memory_manager.update_knowledge(
            unique_id=unique_id, content="Updated content", source="feedback"
        )

        # Verify the update succeeded
        assert result is True

        # Retrieve the updated knowledge unit
        knowledge_unit = await initialized_memory_manager.retrieve_knowledge(unique_id)
        assert knowledge_unit is not None
        assert knowledge_unit.original_chunk == "Updated content"
        assert knowledge_unit.knowledge_source == "feedback"

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, initialized_memory_manager):
        """Test deleting knowledge from memory."""
        # Add a knowledge unit
        unique_id = await initialized_memory_manager.add_knowledge(
            content="Test content", source="action"
        )

        # Verify it exists
        assert await initialized_memory_manager.retrieve_knowledge(unique_id) is not None

        # Delete it
        result = await initialized_memory_manager.delete_knowledge(unique_id)

        # Verify the delete succeeded
        assert result is True

        # Verify it no longer exists
        assert await initialized_memory_manager.retrieve_knowledge(unique_id) is None

    @pytest.mark.asyncio
    async def test_search_knowledge(self, initialized_memory_manager):
        """Test searching for knowledge."""
        # Add multiple knowledge units
        await initialized_memory_manager.add_knowledge(
            content="Python is a programming language", source="corpus"
        )
        await initialized_memory_manager.add_knowledge(
            content="LangChain is a framework for LLM applications", source="corpus"
        )
        await initialized_memory_manager.add_knowledge(
            content="Memory systems help agents retain information", source="action"
        )

        # Search for "programming"
        results = await initialized_memory_manager.search_knowledge("programming")

        # Verify we got results
        assert len(results) > 0
        assert any("Python" in ku.original_chunk for ku, _ in results)

        # Search with a source filter
        results = await initialized_memory_manager.search_knowledge(
            "memory", source_filter="action"
        )

        # Verify we got filtered results
        assert len(results) > 0
        assert all(ku.knowledge_source == "action" for ku, _ in results)

    @pytest.mark.asyncio
    async def test_get_related_knowledge(self, initialized_memory_manager):
        """Test finding related knowledge."""
        # Add multiple knowledge units
        await initialized_memory_manager.add_knowledge(
            content="Python is used for data science and machine learning", source="corpus"
        )
        await initialized_memory_manager.add_knowledge(
            content="JavaScript is used for web development", source="corpus"
        )

        # Find knowledge related to machine learning
        results = await initialized_memory_manager.get_related_knowledge(
            "How to use Python for machine learning"
        )

        # Verify we got related results
        assert len(results) > 0

        # The first result should be most related to machine learning
        most_related, score = results[0]
        assert "Python" in most_related.original_chunk
        assert "machine learning" in most_related.original_chunk
        assert 0 <= score <= 1  # Score should be between 0 and 1

    @pytest.mark.asyncio
    async def test_list_knowledge(self, initialized_memory_manager):
        """Test listing knowledge units."""
        # Add multiple knowledge units with different sources
        await initialized_memory_manager.add_knowledge(content="Knowledge unit 1", source="corpus")
        await initialized_memory_manager.add_knowledge(content="Knowledge unit 2", source="action")
        await initialized_memory_manager.add_knowledge(
            content="Knowledge unit 3", source="feedback"
        )

        # List all knowledge units
        results = await initialized_memory_manager.list_knowledge()

        # Verify we got all the knowledge units
        assert len(results) == 3

        # List with a source filter
        results = await initialized_memory_manager.list_knowledge(source_filter="action")

        # Verify we got filtered results
        assert len(results) == 1
        assert results[0].knowledge_source == "action"

        # List with pagination
        results = await initialized_memory_manager.list_knowledge(limit=1, offset=1)

        # Verify pagination works
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_count_knowledge(self, initialized_memory_manager):
        """Test counting knowledge units."""
        # Add multiple knowledge units with different sources
        await initialized_memory_manager.add_knowledge(content="Knowledge unit 1", source="corpus")
        await initialized_memory_manager.add_knowledge(content="Knowledge unit 2", source="action")
        await initialized_memory_manager.add_knowledge(
            content="Knowledge unit 3", source="feedback"
        )

        # Count all knowledge units
        count = await initialized_memory_manager.count_knowledge()
        assert count == 3

        # Count with a source filter
        count = await initialized_memory_manager.count_knowledge(source_filter="corpus")
        assert count == 1
