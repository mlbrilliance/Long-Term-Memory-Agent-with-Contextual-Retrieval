"""
Tests for the Add vs. Update Logic in the memory manager.

This module contains tests specifically for the "Add vs. Update Logic" feature,
which decides whether to add a new knowledge unit or update an existing one
based on content similarity.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


class TestAddUpdateLogic:
    """Test suite for the Add vs. Update Logic feature."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides a memory store for testing."""
        store = InMemoryVectorStore(embedding_dim=10)
        return store

    @pytest.fixture
    async def initialized_memory_manager(self, memory_store):
        """Fixture that provides an initialized memory manager for testing."""
        manager = MemoryManager(memory_store=memory_store)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_add_new_knowledge(self, initialized_memory_manager):
        """Test adding new knowledge when no similar content exists."""
        # Add a knowledge unit
        unique_id = await initialized_memory_manager.add_knowledge(
            content="Python is a programming language",
            source="corpus",
            context="Programming languages",
        )

        # Verify it was added
        knowledge_unit = await initialized_memory_manager.retrieve_knowledge(unique_id)
        assert knowledge_unit is not None
        assert knowledge_unit.original_chunk == "Python is a programming language"
        assert knowledge_unit.contextual_text == "Programming languages"

        # Count the total number of knowledge units
        count = await initialized_memory_manager.count_knowledge()
        assert count == 1

    @pytest.mark.asyncio
    async def test_update_similar_knowledge(self, initialized_memory_manager):
        """Test that similar content updates existing knowledge rather than creating duplicates."""
        # Mock get_related_knowledge to return a similar knowledge unit
        original_knowledge = KnowledgeUnit(
            unique_id="test-id-1",
            original_chunk="Python is a popular programming language used for data science",
            contextual_text="Programming languages",
            knowledge_source="corpus",
        )

        # Patch get_related_knowledge to return our mock data
        with patch.object(
            initialized_memory_manager, "get_related_knowledge", new_callable=AsyncMock
        ) as mock_get_related:
            # Set up the mock to return our test knowledge unit with a high similarity score
            mock_get_related.return_value = [(original_knowledge, 0.95)]

            # Add similar content that should update rather than create a new unit
            updated_id = await initialized_memory_manager.add_knowledge(
                content="Python is a widely used programming language for data science and AI",
                source="corpus",
                context="Programming languages and AI",
            )

            # Verify the IDs match (indicating update rather than new creation)
            assert updated_id == "test-id-1"

            # Verify update_knowledge was called
            assert mock_get_related.called

    @pytest.mark.asyncio
    async def test_force_add_similar_knowledge(self, initialized_memory_manager):
        """Test that update_if_similar=False forces adding a new unit even if similar content exists."""
        # Add initial knowledge unit
        original_id = await initialized_memory_manager.add_knowledge(
            content="JavaScript is used for web development", source="corpus"
        )

        # Mock get_related_knowledge to return the existing knowledge unit
        original_knowledge = await initialized_memory_manager.retrieve_knowledge(original_id)

        with patch.object(
            initialized_memory_manager, "get_related_knowledge", new_callable=AsyncMock
        ) as mock_get_related:
            # Set up the mock to return our existing knowledge unit with a high similarity score
            mock_get_related.return_value = [(original_knowledge, 0.9)]

            # Add similar content but force it to be a new unit
            new_id = await initialized_memory_manager.add_knowledge(
                content="JavaScript is a language commonly used for web development",
                source="corpus",
                update_if_similar=False,  # Force creating a new unit
            )

            # Verify the IDs are different (indicating a new unit was created)
            assert new_id != original_id

            # Verify get_related_knowledge was not called (since update_if_similar=False)
            assert not mock_get_related.called

    @pytest.mark.asyncio
    async def test_similarity_threshold(self, initialized_memory_manager):
        """Test that the similarity threshold controls when to update vs. add."""
        # Add initial knowledge unit
        original_id = await initialized_memory_manager.add_knowledge(
            content="Machine learning is a subset of artificial intelligence", source="corpus"
        )

        # Get the original knowledge unit
        original_knowledge = await initialized_memory_manager.retrieve_knowledge(original_id)

        # Test with high threshold - should create new unit
        with patch.object(
            initialized_memory_manager, "get_related_knowledge", new_callable=AsyncMock
        ) as mock_get_related:
            # Return the existing knowledge with a similarity score below the high threshold
            mock_get_related.return_value = [(original_knowledge, 0.8)]

            # Add somewhat similar content with high threshold (should create new unit)
            new_id_high_threshold = await initialized_memory_manager.add_knowledge(
                content="Deep learning is an advanced form of machine learning",
                source="corpus",
                similarity_threshold=0.95,  # Very high threshold
            )

            # Verify new unit was created (IDs are different)
            assert new_id_high_threshold != original_id

        # Test with low threshold - should update existing unit
        with patch.object(
            initialized_memory_manager, "get_related_knowledge", new_callable=AsyncMock
        ) as mock_get_related:
            # Return the existing knowledge with a similarity score above the low threshold
            mock_get_related.return_value = [(original_knowledge, 0.6)]

            # Add similar content with low threshold (should update existing unit)
            updated_id = await initialized_memory_manager.add_knowledge(
                content="Machine learning is a field of AI that focuses on algorithms",
                source="corpus",
                similarity_threshold=0.5,  # Low threshold
            )

            # Verify original unit was updated
            assert updated_id == original_id

    @pytest.mark.asyncio
    async def test_metadata_merging(self, initialized_memory_manager):
        """Test that metadata is merged properly when updating."""
        # Add initial knowledge unit with metadata
        original_id = await initialized_memory_manager.add_knowledge(
            content="Natural Language Processing (NLP) helps computers understand human language",
            source="corpus",
            metadata={"category": "AI", "importance": "high"},
        )

        # Get the original knowledge unit for mocking
        original_knowledge = await initialized_memory_manager.retrieve_knowledge(original_id)

        # Setup update test with mocked similar knowledge
        with patch.object(
            initialized_memory_manager, "get_related_knowledge", new_callable=AsyncMock
        ) as mock_get_related:
            # Return the existing knowledge with a high similarity score
            mock_get_related.return_value = [(original_knowledge, 0.9)]

            # Mock update_knowledge to capture the metadata
            with patch.object(
                initialized_memory_manager, "update_knowledge", new_callable=AsyncMock
            ) as mock_update:
                mock_update.return_value = True

                # Update with new metadata
                updated_id = await initialized_memory_manager.add_knowledge(
                    content="NLP is a field that helps computers process and understand human language",
                    source="corpus",
                    metadata={"applications": ["chatbots", "translation"], "complexity": "medium"},
                )

                # Verify the IDs match
                assert updated_id == original_id

                # Verify update_knowledge was called with merged metadata
                mock_update.assert_called_once()
                call_args = mock_update.call_args[1]
                assert "metadata" in call_args
                assert call_args["metadata"].get("category") == "AI"
                assert call_args["metadata"].get("importance") == "high"
                assert call_args["metadata"].get("applications") == ["chatbots", "translation"]
                assert call_args["metadata"].get("complexity") == "medium"
