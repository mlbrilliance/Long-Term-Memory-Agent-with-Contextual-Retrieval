"""
Tests for the ChromaDB vector store implementation.

Note: On Windows, teardown errors (PermissionError) may occur when deleting the temp directory due to SQLite file locks held by ChromaDB. This is a known issue with ChromaDB/SQLite on Windows and does NOT affect the correctness of the tests or the vector store implementation. See: https://github.com/chroma-core/chroma/issues/620

This module contains tests for the ChromaDB vector store adapter, ensuring
it properly implements the vector store interface and provides the expected
functionality for storing and retrieving embeddings.
"""

import gc
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    """Test suite for the ChromaDB vector store implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Fixture that provides a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Additional cleanup: force GC to help release file locks
        gc.collect()
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Fixture that provides a mock configuration for testing."""
        config = Settings(
            anthropic_api_key="test_key",
            perplexity_api_key="test_key",
            vector_db_path=temp_dir,
            embedding_model_name="all-MiniLM-L6-v2",  # Small model for faster tests
        )
        return config

    @pytest.fixture
    def vector_store(self, mock_config):
        """Fixture that provides a ChromaDB vector store for testing."""
        store = ChromaVectorStore(config=mock_config)
        yield store
        # Cleanup: ensure ChromaDB client is closed and GC runs before temp_dir removal
        store.client = None
        del store
        gc.collect()

    @pytest.fixture
    def sample_ku(self):
        """Fixture that provides a sample knowledge unit for testing."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk for the ChromaDB vector store",
            knowledge_source="action",
        )

    @pytest.mark.asyncio
    async def test_initialization(self, vector_store):
        """Test that the vector store initializes correctly."""
        assert vector_store.initialized is False
        await vector_store.initialize()
        assert vector_store.initialized is True
        assert vector_store.collection is not None

    @pytest.mark.asyncio
    async def test_generate_embedding(self, vector_store):
        """Test generating embeddings."""
        await vector_store.initialize()

        # Generate embedding for a test text
        embedding = await vector_store.generate_embedding("This is a test text")

        # Verify the embedding format
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_add_and_get(self, vector_store, sample_ku):
        """Test adding and retrieving a knowledge unit."""
        await vector_store.initialize()

        # Add the knowledge unit
        unique_id = await vector_store.add(sample_ku)

        # Verify an embedding was generated
        assert sample_ku.embedding_vector is not None

        # Retrieve the knowledge unit
        retrieved_ku = await vector_store.get(unique_id)

        # Verify it matches the original
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == sample_ku.unique_id
        assert retrieved_ku.original_chunk == sample_ku.original_chunk
        assert retrieved_ku.knowledge_source == sample_ku.knowledge_source
        assert len(retrieved_ku.embedding_vector) == len(sample_ku.embedding_vector)

    @pytest.mark.asyncio
    async def test_update(self, vector_store, sample_ku):
        """Test updating a knowledge unit."""
        await vector_store.initialize()

        # Add the knowledge unit
        unique_id = await vector_store.add(sample_ku)

        # Create an updated version
        updated_ku = KnowledgeUnit(
            unique_id=unique_id, original_chunk="Updated test chunk", knowledge_source="feedback"
        )

        # Update it
        result = await vector_store.update(updated_ku)

        # Verify the update succeeded
        assert result is True

        # Retrieve the updated knowledge unit
        retrieved_ku = await vector_store.get(unique_id)

        # Verify it has the updated content
        assert retrieved_ku is not None
        assert retrieved_ku.original_chunk == "Updated test chunk"
        assert retrieved_ku.knowledge_source == "feedback"

    @pytest.mark.asyncio
    async def test_delete(self, vector_store, sample_ku):
        """Test deleting a knowledge unit."""
        await vector_store.initialize()

        # Add the knowledge unit
        unique_id = await vector_store.add(sample_ku)

        # Verify it exists
        assert await vector_store.get(unique_id) is not None

        # Delete it
        result = await vector_store.delete(unique_id)

        # Verify the delete succeeded
        assert result is True

        # Verify it no longer exists
        assert await vector_store.get(unique_id) is None

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store):
        """Test similarity search functionality."""
        await vector_store.initialize()

        # Add multiple knowledge units
        ku1 = KnowledgeUnit(
            original_chunk="Python is a programming language used for AI development",
            knowledge_source="corpus",
        )
        ku2 = KnowledgeUnit(
            original_chunk="LangChain is a framework for building LLM applications",
            knowledge_source="corpus",
        )
        ku3 = KnowledgeUnit(
            original_chunk="Vector databases store and search embeddings efficiently",
            knowledge_source="action",
        )

        await vector_store.add(ku1)
        await vector_store.add(ku2)
        await vector_store.add(ku3)

        # Generate a query embedding
        query_embedding = await vector_store.generate_embedding(
            "How are programming languages used in AI?"
        )

        # Search based on this embedding
        results = await vector_store.similarity_search(query_embedding, limit=2)

        # Verify we got results
        assert len(results) > 0

        # The first result should be ku1 since it's most related to programming languages and AI
        # Note: This is probabilistic, so we're not asserting specific order, just checking format
        assert len(results) <= 2  # Limit should be respected
        assert all(isinstance(ku, KnowledgeUnit) for ku, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        assert all(0 <= score <= 1 for _, score in results)

    @pytest.mark.asyncio
    async def test_filter_criteria(self, vector_store):
        """Test search with filter criteria."""
        await vector_store.initialize()

        # Add multiple knowledge units with different sources
        ku1 = KnowledgeUnit(
            original_chunk="Knowledge from corpus source", knowledge_source="corpus"
        )
        ku2 = KnowledgeUnit(
            original_chunk="Knowledge from action source", knowledge_source="action"
        )
        ku3 = KnowledgeUnit(
            original_chunk="Knowledge from feedback source", knowledge_source="feedback"
        )

        await vector_store.add(ku1)
        await vector_store.add(ku2)
        await vector_store.add(ku3)

        # Search with source filter
        results = await vector_store.search(
            "Knowledge", filter_criteria={"knowledge_source": "action"}
        )

        # Verify we only got the action-sourced knowledge unit
        assert len(results) == 1
        assert results[0][0].knowledge_source == "action"

    @pytest.mark.asyncio
    async def test_list_and_count(self, vector_store):
        """Test listing and counting knowledge units."""
        await vector_store.initialize()

        # Add multiple knowledge units
        for i in range(5):
            ku = KnowledgeUnit(
                original_chunk=f"Test chunk {i}",
                knowledge_source="corpus" if i % 2 == 0 else "action",
            )
            await vector_store.add(ku)

        # Test count
        count = await vector_store.count()
        assert count == 5

        # Test count with filter
        count = await vector_store.count(filter_criteria={"knowledge_source": "corpus"})
        assert count == 3  # 0, 2, 4 are corpus

        # Test list with pagination
        results = await vector_store.list(limit=2, offset=1)
        assert len(results) == 2

        # Test list with filter
        results = await vector_store.list(filter_criteria={"knowledge_source": "action"})
        assert len(results) == 2  # 1, 3 are action
        assert all(ku.knowledge_source == "action" for ku in results)
