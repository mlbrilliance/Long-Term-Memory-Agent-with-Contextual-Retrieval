"""
Tests for the Memory Manager's Process & Update Mechanism (Phase 7).

This module contains tests for the MemoryManager's process_potential_knowledge method
and related update mechanisms implemented in Phase 7.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.memory.contextualizer import Contextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


class MockContextualizer(Contextualizer):
    """Mock Contextualizer for testing."""

    async def generate_context(
        self, content: str, existing_context: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Mock implementation that enhances context by adding a prefix.

        Args:
            content: The main content of the knowledge unit
            existing_context: Optional existing context
            metadata: Optional existing metadata

        Returns:
            Enhanced context and metadata
        """
        enhanced_context = (
            f"Enhanced: {existing_context}" if existing_context else "Enhanced context"
        )
        enhanced_metadata = metadata.copy() if metadata else {}
        enhanced_metadata["enhanced"] = True

        return {"context": enhanced_context, "metadata": enhanced_metadata}


class TestMemoryManagerPhase7:
    """Test suite for the Memory Manager's Process & Update Mechanism (Phase 7)."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides a memory store for testing."""
        store = InMemoryVectorStore(embedding_dim=10)
        return store

    @pytest.fixture
    def contextualizer(self):
        """Fixture that provides a mock contextualizer for testing."""
        return MockContextualizer()

    @pytest.fixture
    async def initialized_memory_manager(self, memory_store, contextualizer):
        """Fixture that provides an initialized memory manager for testing."""
        manager = MemoryManager(memory_store=memory_store, contextualizer=contextualizer)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_process_potential_knowledge_new(self, initialized_memory_manager):
        """Test processing new knowledge when no similar content exists."""
        # Process new knowledge
        unique_id = await initialized_memory_manager.process_potential_knowledge(
            content="Python is a versatile programming language",
            source="corpus",
            context="Programming languages",
        )

        # Verify it was added
        knowledge_unit = await initialized_memory_manager.retrieve_knowledge(unique_id)
        assert knowledge_unit is not None
        assert knowledge_unit.original_chunk == "Python is a versatile programming language"
        assert knowledge_unit.contextual_text == "Enhanced: Programming languages"
        assert knowledge_unit.metadata["enhanced"] is True

        # Count should be 1
        count = await initialized_memory_manager.count_knowledge()
        assert count == 1

    @pytest.mark.asyncio
    async def test_process_potential_knowledge_update(self, initialized_memory_manager):
        """Test processing knowledge that's similar to existing content."""
        # Add initial knowledge unit
        original_id = await initialized_memory_manager.add_knowledge(
            content="JavaScript is used for web development",
            source="corpus",
            context="Web technologies",
        )

        # Mock get_related_knowledge to return the existing unit
        original_unit = await initialized_memory_manager.retrieve_knowledge(original_id)

        with patch.object(
            initialized_memory_manager, "_find_similar_knowledge", new_callable=AsyncMock
        ) as mock_find:
            # Set up the mock to return our existing unit
            mock_find.return_value = original_unit

            # Process similar knowledge
            updated_id = await initialized_memory_manager.process_potential_knowledge(
                content="JavaScript is widely used for frontend web development",
                source="corpus",
                context="Frontend technologies",
            )

            # Verify the IDs match (indicating update rather than new creation)
            assert updated_id == original_id

            # Verify content was updated
            updated_unit = await initialized_memory_manager.retrieve_knowledge(updated_id)
            assert (
                updated_unit.original_chunk
                == "JavaScript is widely used for frontend web development"
            )
            assert updated_unit.contextual_text == "Enhanced: Frontend technologies"
            assert updated_unit.metadata["enhanced"] is True

            # Verify mock was called
            mock_find.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_potential_knowledge_force_new(self, initialized_memory_manager):
        """Test forcing new knowledge creation even when similar content exists."""
        # Add initial knowledge unit
        original_id = await initialized_memory_manager.add_knowledge(
            content="Python is used for data science", source="corpus"
        )

        # Process similar knowledge but force new unit
        new_id = await initialized_memory_manager.process_potential_knowledge(
            content="Python is excellent for data science tasks",
            source="corpus",
            update_if_similar=False,  # Force creating a new unit
        )

        # Verify new unit was created (IDs are different)
        assert new_id != original_id

        # Verify both units exist
        original_unit = await initialized_memory_manager.retrieve_knowledge(original_id)
        new_unit = await initialized_memory_manager.retrieve_knowledge(new_id)

        assert original_unit is not None
        assert new_unit is not None
        assert original_unit.original_chunk == "Python is used for data science"
        assert new_unit.original_chunk == "Python is excellent for data science tasks"

        # Count should be 2
        count = await initialized_memory_manager.count_knowledge()
        assert count == 2

    @pytest.mark.asyncio
    async def test_contextualizer_integration(self, initialized_memory_manager):
        """Test that the contextualizer properly enhances knowledge."""
        # Process knowledge with contextualizer
        unique_id = await initialized_memory_manager.process_potential_knowledge(
            content="Natural Language Processing helps computers understand human language",
            source="corpus",
            context="AI technologies",
            metadata={"category": "AI", "complexity": "high"},
        )

        # Verify contextualizer enhanced the knowledge
        knowledge_unit = await initialized_memory_manager.retrieve_knowledge(unique_id)
        assert knowledge_unit is not None
        assert knowledge_unit.contextual_text == "Enhanced: AI technologies"
        assert knowledge_unit.metadata["category"] == "AI"
        assert knowledge_unit.metadata["complexity"] == "high"
        assert knowledge_unit.metadata["enhanced"] is True
