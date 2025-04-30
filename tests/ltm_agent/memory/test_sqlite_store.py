"""
Tests for the SQLite-based persistent vector store.

This module tests the SQLiteVectorStore implementation to ensure
proper storage and retrieval of knowledge units from a SQLite database.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.memory.sqlite_store import SQLiteVectorStore
from ltm_agent.utils.test_utils import (
    assert_knowledge_unit_equal,
    create_sample_knowledge_units,
    create_test_knowledge_unit,
)


class TestSQLiteVectorStore:
    """Test suite for the SQLiteVectorStore."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary file path for database testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        yield db_path

        # Clean up the file after the test
        try:
            os.unlink(db_path)
        except Exception as e:
            print(f"Failed to clean up temporary database file: {str(e)}")

    @pytest.fixture
    async def sqlite_store(self, temp_db_path):
        """Fixture providing an initialized SQLiteVectorStore."""
        store = SQLiteVectorStore(database_path=temp_db_path, embedding_dim=128)
        await store.initialize()
        yield store

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db_path):
        """Test store initialization creates the database and tables."""
        store = SQLiteVectorStore(database_path=temp_db_path)
        await store.initialize()

        # Check that the database file exists
        assert os.path.exists(temp_db_path)
        assert os.path.getsize(temp_db_path) > 0

    @pytest.mark.asyncio
    async def test_add_and_get(self, sqlite_store):
        """Test adding and retrieving a knowledge unit."""
        # Create a test knowledge unit
        ku = create_test_knowledge_unit(
            content="SQLite test content",
            source="action",
            context="SQLite test context",
            metadata={"test_id": 1},
            tags=["sqlite", "test"],
        )

        # Add it to the store
        unique_id = await sqlite_store.add(ku)
        assert unique_id == ku.unique_id

        # Retrieve it
        retrieved_ku = await sqlite_store.get(unique_id)

        # Verify it matches the original
        assert retrieved_ku is not None
        await assert_knowledge_unit_equal(retrieved_ku, ku)

    @pytest.mark.asyncio
    async def test_embedding_generation(self, sqlite_store):
        """Test that embeddings are generated when adding knowledge units."""
        # Create a test knowledge unit without embedding
        ku = create_test_knowledge_unit(content="SQLite embedding test", source="corpus")
        assert ku.embedding_vector is None

        # Add it to the store
        unique_id = await sqlite_store.add(ku)

        # Retrieve it
        retrieved_ku = await sqlite_store.get(unique_id)

        # Verify embedding was generated
        assert retrieved_ku is not None
        assert retrieved_ku.embedding_vector is not None
        assert len(retrieved_ku.embedding_vector) == sqlite_store.embedding_dim

    @pytest.mark.asyncio
    async def test_update(self, sqlite_store):
        """Test updating a knowledge unit."""
        # Create and add a test knowledge unit
        ku = create_test_knowledge_unit(content="Original content", source="action")
        unique_id = await sqlite_store.add(ku)

        # Update the knowledge unit
        ku.original_chunk = "Updated content"
        ku.contextual_text = "Updated context"
        ku.knowledge_source = "feedback"
        ku.metadata = {"updated": True}
        ku.tags = ["updated", "test"]

        # Submit the update
        result = await sqlite_store.update(ku)
        assert result is True

        # Retrieve the updated unit
        updated_ku = await sqlite_store.get(unique_id)

        # Verify updates were applied
        assert updated_ku is not None
        assert updated_ku.original_chunk == "Updated content"
        assert updated_ku.contextual_text == "Updated context"
        assert updated_ku.knowledge_source == "feedback"
        assert updated_ku.metadata == {"updated": True}
        assert sorted(updated_ku.tags) == ["test", "updated"]

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store):
        """Test deleting a knowledge unit."""
        # Create and add a test knowledge unit
        ku = create_test_knowledge_unit(content="Content to delete", source="corpus")
        unique_id = await sqlite_store.add(ku)

        # Verify it exists
        assert await sqlite_store.get(unique_id) is not None

        # Delete it
        result = await sqlite_store.delete(unique_id)
        assert result is True

        # Verify it's gone
        assert await sqlite_store.get(unique_id) is None

    @pytest.mark.asyncio
    async def test_batch_operations(self, sqlite_store):
        """Test adding and retrieving multiple knowledge units."""
        # Create sample knowledge units
        units = create_sample_knowledge_units(5)

        # Add them to the store
        for ku in units:
            await sqlite_store.add(ku)

        # Retrieve each one and verify
        for original_ku in units:
            retrieved_ku = await sqlite_store.get(original_ku.unique_id)
            assert retrieved_ku is not None
            await assert_knowledge_unit_equal(retrieved_ku, original_ku)

    @pytest.mark.asyncio
    async def test_text_search(self, sqlite_store):
        """Test text-based search functionality."""
        # Add sample knowledge units with specific content
        ku1 = create_test_knowledge_unit(content="Python programming language", source="corpus")
        ku2 = create_test_knowledge_unit(content="JavaScript for web development", source="corpus")
        ku3 = create_test_knowledge_unit(content="Python database libraries", source="action")

        await sqlite_store.add(ku1)
        await sqlite_store.add(ku2)
        await sqlite_store.add(ku3)

        # Search for Python
        results = await sqlite_store.search("Python")

        # Should find both Python-related units
        assert len(results) == 2
        result_ids = [ku.unique_id for ku, _ in results]
        assert ku1.unique_id in result_ids
        assert ku3.unique_id in result_ids
        assert ku2.unique_id not in result_ids

        # All results should have a score
        for _, score in results:
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_filtered_search(self, sqlite_store):
        """Test search with filters."""
        # Add sample knowledge units with different sources
        ku1 = create_test_knowledge_unit(content="Python for data science", source="corpus")
        ku2 = create_test_knowledge_unit(content="Python for web development", source="action")

        await sqlite_store.add(ku1)
        await sqlite_store.add(ku2)

        # Search with source filter
        results = await sqlite_store.search(
            "Python", filter_criteria={"knowledge_source": "action"}
        )

        # Should only find the action source unit
        assert len(results) == 1
        assert results[0][0].unique_id == ku2.unique_id

    @pytest.mark.asyncio
    async def test_similarity_search(self, sqlite_store):
        """Test embedding-based similarity search."""
        # Add sample knowledge units
        ku1 = create_test_knowledge_unit(content="Machine learning concepts", source="corpus")
        ku2 = create_test_knowledge_unit(content="Deep learning frameworks", source="corpus")
        ku3 = create_test_knowledge_unit(content="Web development basics", source="action")

        await sqlite_store.add(ku1)
        await sqlite_store.add(ku2)
        await sqlite_store.add(ku3)

        # Generate a query embedding
        query_embedding = await sqlite_store.generate_embedding("AI and machine learning")

        # Perform similarity search
        results = await sqlite_store.similarity_search(query_embedding)

        # Should find all units but in order of relevance
        assert len(results) == 3
        # First results should be more relevant to machine learning
        assert "learning" in results[0][0].original_chunk.lower()

        # All results should have a similarity score
        for _, score in results:
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_list_and_count(self, sqlite_store):
        """Test listing and counting knowledge units."""
        # Add sample knowledge units
        units = create_sample_knowledge_units(5)
        for ku in units:
            await sqlite_store.add(ku)

        # List all units
        all_units = await sqlite_store.list()
        assert len(all_units) == 5

        # List with limit
        limited_units = await sqlite_store.list(limit=2)
        assert len(limited_units) == 2

        # List with filter
        action_units = await sqlite_store.list(filter_criteria={"knowledge_source": "action"})
        for ku in action_units:
            assert ku.knowledge_source == "action"

        # Count all units
        count = await sqlite_store.count()
        assert count == 5

        # Count with filter
        action_count = await sqlite_store.count(filter_criteria={"knowledge_source": "action"})
        assert action_count == len(action_units)

    @pytest.mark.asyncio
    async def test_tag_filtering(self, sqlite_store):
        """Test filtering by tags."""
        # Add units with different tags
        ku1 = create_test_knowledge_unit(
            content="Content with tag1", source="corpus", tags=["tag1", "common"]
        )
        ku2 = create_test_knowledge_unit(
            content="Content with tag2", source="action", tags=["tag2", "common"]
        )

        await sqlite_store.add(ku1)
        await sqlite_store.add(ku2)

        # Filter by a specific tag
        results = await sqlite_store.list(filter_criteria={"tags": ["tag1"]})
        assert len(results) == 1
        assert results[0].unique_id == ku1.unique_id

        # Filter by common tag
        results = await sqlite_store.list(filter_criteria={"tags": ["common"]})
        assert len(results) == 2
