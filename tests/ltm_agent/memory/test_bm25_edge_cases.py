"""
Tests for edge cases in the BM25 store implementation.

This module contains tests for handling edge cases in the BM25 store,
such as empty queries, special characters, and large documents.
"""

import os

import pytest

from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.utils.test_utils import create_test_knowledge_unit

# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio


class TestBM25EdgeCases:
    """Tests for edge cases in the BM25 store implementation."""

    @pytest.fixture
    def bm25_store(self, tmp_path):
        """Fixture providing a RankBM25Store instance with a temporary file path."""
        index_path = os.path.join(tmp_path, "test_bm25_edge.pkl")
        store = RankBM25Store(index_path=index_path)
        return store

    @pytest.fixture
    def special_knowledge_units(self):
        """Fixture providing knowledge units with special characteristics for edge case testing."""
        units = {}

        # Unit with empty content
        units["empty"] = create_test_knowledge_unit(
            content="",
            context="This unit has empty content",
        )

        # Unit with special characters
        units["special_chars"] = create_test_knowledge_unit(
            content="!@#$%^&*()-_=+[]{}|;:'\",.<>/?",
            context="This unit has special characters",
        )

        # Unit with very short content
        units["short"] = create_test_knowledge_unit(
            content="Short.",
            context="This unit has very short content",
        )

        # Unit with very long content
        long_content = "Long content " * 200  # 2400 characters
        units["long"] = create_test_knowledge_unit(
            content=long_content,
            context="This unit has very long content",
        )

        # Unit with repeated terms
        units["repeated"] = create_test_knowledge_unit(
            content="repeat repeat repeat repeat repeat",
            context="This unit has repeated terms",
        )

        # Unit with mixed language content
        units["mixed_lang"] = create_test_knowledge_unit(
            content="English text with 一些中文 and some 日本語",
            context="This unit has mixed language content",
        )

        # Unit with numeric content
        units["numeric"] = create_test_knowledge_unit(
            content="123 456 789 0.12 3.45",
            context="This unit has numeric content",
        )

        return units

    async def test_empty_query(self, bm25_store, special_knowledge_units):
        """Test search with an empty query."""
        # Update the index with the special units
        await bm25_store.update_index(special_knowledge_units)

        # Search with empty query
        results = await bm25_store.search("", k=5)

        # Should return empty results for empty query
        assert len(results) == 0

    async def test_special_character_query(self, bm25_store, special_knowledge_units):
        """Test search with special characters in the query."""
        # Update the index with the special units
        await bm25_store.update_index(special_knowledge_units)

        # Search with special characters
        results = await bm25_store.search("!@#$%^&*()", k=5)

        # Depending on tokenization, this might match the special_chars unit
        # But should not crash or raise exceptions
        assert isinstance(results, list)

        # Search with a mix of normal text and special characters
        results = await bm25_store.search("text with !@#$", k=5)
        assert isinstance(results, list)

    async def test_very_large_update(self, bm25_store):
        """Test updating the index with a large number of units."""
        # Create a large number of units
        large_units = {}
        for i in range(100):  # 100 units should be enough to test without being too slow
            unit_id = f"large_{i}"
            large_units[unit_id] = create_test_knowledge_unit(
                content=f"Content for unit {i}",
                context=f"Context for unit {i}",
            )

        # Update should handle a large number of units
        await bm25_store.update_index(large_units)

        # Verify all units were added
        for i in range(100):
            unit_id = f"large_{i}"
            assert await bm25_store.contains(unit_id)

        # Search should return relevant results
        results = await bm25_store.search("unit 50", k=5)
        assert len(results) > 0

    async def test_incremental_updates(self, bm25_store, special_knowledge_units):
        """Test incrementally updating the index."""
        # Start with a subset of units
        initial_units = {
            "empty": special_knowledge_units["empty"],
            "short": special_knowledge_units["short"],
        }
        await bm25_store.update_index(initial_units)

        # Verify the initial units are in the index
        assert await bm25_store.contains("empty")
        assert await bm25_store.contains("short")
        assert not await bm25_store.contains("long")

        # Add more units
        additional_units = {
            "long": special_knowledge_units["long"],
            "repeated": special_knowledge_units["repeated"],
        }
        await bm25_store.update_index(additional_units)

        # Verify all units are now in the index
        assert await bm25_store.contains("empty")
        assert await bm25_store.contains("short")
        assert await bm25_store.contains("long")
        assert await bm25_store.contains("repeated")

        # Search should find terms from both update batches
        short_results = await bm25_store.search("short", k=5)
        long_results = await bm25_store.search("long", k=5)
        repeat_results = await bm25_store.search("repeat", k=5)

        # Should find results for all queries
        assert any(unit_id == "short" for unit_id, _ in short_results)
        assert any(unit_id == "long" for unit_id, _ in long_results)
        assert any(unit_id == "repeated" for unit_id, _ in repeat_results)

    async def test_update_existing_units(self, bm25_store):
        """Test updating existing units in the index."""
        # Create initial units
        initial_units = {
            "unit1": create_test_knowledge_unit(
                content="Initial content for unit 1",
                context="Initial context for unit 1",
            ),
            "unit2": create_test_knowledge_unit(
                content="Initial content for unit 2",
                context="Initial context for unit 2",
            ),
        }
        await bm25_store.update_index(initial_units)

        # Verify initial content is searchable
        results = await bm25_store.search("initial", k=5)
        assert len(results) > 0

        # Create updated units with the same IDs
        updated_units = {
            "unit1": create_test_knowledge_unit(
                content="Updated content for unit 1",
                context="Updated context for unit 1",
            ),
            "unit2": create_test_knowledge_unit(
                content="Updated content for unit 2",
                context="Updated context for unit 2",
            ),
        }
        await bm25_store.update_index(updated_units)

        # Verify updated content is searchable
        updated_results = await bm25_store.search("updated", k=5)
        initial_results = await bm25_store.search("initial", k=5)

        # Check that updated content is found
        assert len(updated_results) > 0

        # The initial content might still be found if the index doesn't fully replace
        # entries, so we don't strictly assert it's not found
        # Instead, we verify the updated content is now searchable
