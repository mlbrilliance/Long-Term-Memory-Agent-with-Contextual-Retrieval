"""
Tests for the in-memory store implementations.

This module contains tests for the in-memory store implementations of the
memory interfaces, ensuring they function correctly for development and testing.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.in_memory_store import InMemoryStore, InMemoryVectorStore


class TestInMemoryStore:
    """Test suite for the InMemoryStore."""

    @pytest.fixture
    def memory_store(self):
        """Fixture that provides an in-memory store for testing."""
        return InMemoryStore()

    @pytest.fixture
    def sample_ku(self):
        """Fixture that provides a sample knowledge unit for testing."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk for the in-memory store", knowledge_source="action"
        )

    @pytest.mark.asyncio
    async def test_initialization(self, memory_store):
        """Test that the store initializes correctly."""
        assert memory_store.initialized is False
        await memory_store.initialize()
        assert memory_store.initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get(self, memory_store, sample_ku):
        """Test adding and retrieving a knowledge unit."""
        await memory_store.initialize()

        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Retrieve it
        retrieved_ku = await memory_store.get(unique_id)

        # Verify it's the same
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == sample_ku.unique_id
        assert retrieved_ku.original_chunk == sample_ku.original_chunk
        assert retrieved_ku.knowledge_source == sample_ku.knowledge_source

    @pytest.mark.asyncio
    async def test_update(self, memory_store, sample_ku):
        """Test updating a knowledge unit."""
        await memory_store.initialize()

        # Add the knowledge unit
        unique_id = await memory_store.add(sample_ku)

        # Update it
        updated_ku = KnowledgeUnit(
            unique_id=unique_id, original_chunk="Updated chunk", knowledge_source="feedback"
        )
        result = await memory_store.update(updated_ku)

        # Verify the update succeeded
        assert result is True

        # Retrieve and verify the update
        retrieved_ku = await memory_store.get(unique_id)
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

    @pytest.mark.asyncio
    async def test_search(self, memory_store):
        """Test searching for knowledge units."""
        await memory_store.initialize()

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
        assert len(results) > 0
        assert any(ku.original_chunk == "Python is a programming language" for ku, _ in results)

        # Search with filter
        results = await memory_store.search(
            "memory", filter_criteria={"knowledge_source": "action"}
        )
        assert len(results) > 0
        assert all(ku.knowledge_source == "action" for ku, _ in results)

    @pytest.mark.asyncio
    async def test_list_and_count(self, memory_store):
        """Test listing and counting knowledge units."""
        await memory_store.initialize()

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


class TestInMemoryVectorStore:
    """Test suite for the InMemoryVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Fixture that provides an in-memory vector store for testing."""
        return InMemoryVectorStore(embedding_dim=10)

    @pytest.fixture
    def sample_ku(self):
        """Fixture that provides a sample knowledge unit for testing."""
        return KnowledgeUnit(
            original_chunk="This is a test chunk for the in-memory vector store",
            knowledge_source="action",
        )

    @pytest.mark.asyncio
    async def test_initialization(self, vector_store):
        """Test that the store initializes correctly."""
        assert vector_store.initialized is False
        await vector_store.initialize()
        assert vector_store.initialized is True
        assert vector_store.embedding_dim == 10

    @pytest.mark.asyncio
    async def test_generate_embedding(self, vector_store):
        """Test generating embeddings."""
        await vector_store.initialize()

        # Generate embedding
        text = "This is a test text"
        embedding = await vector_store.generate_embedding(text)

        # Verify the embedding format
        assert isinstance(embedding, list)
        assert len(embedding) == vector_store.embedding_dim
        assert all(isinstance(x, float) for x in embedding)

        # Generate another embedding for the same text
        embedding2 = await vector_store.generate_embedding(text)

        # Verify deterministic behavior for the same text
        assert embedding == embedding2

    @pytest.mark.asyncio
    async def test_add_with_embedding_generation(self, vector_store, sample_ku):
        """Test adding a knowledge unit with automatic embedding generation."""
        await vector_store.initialize()

        # Add the knowledge unit without an embedding
        assert sample_ku.embedding_vector is None
        unique_id = await vector_store.add(sample_ku)

        # Retrieve it
        retrieved_ku = await vector_store.get(unique_id)

        # Verify an embedding was generated
        assert retrieved_ku is not None
        assert retrieved_ku.embedding_vector is not None
        assert len(retrieved_ku.embedding_vector) == vector_store.embedding_dim

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store):
        """Test similarity search functionality."""
        await vector_store.initialize()

        # Add multiple knowledge units
        ku1 = KnowledgeUnit(
            original_chunk="Python is a programming language used for AI", knowledge_source="corpus"
        )
        ku2 = KnowledgeUnit(
            original_chunk="LangChain is a framework for LLM applications",
            knowledge_source="corpus",
        )
        ku3 = KnowledgeUnit(
            original_chunk="Vector databases store embeddings efficiently",
            knowledge_source="action",
        )

        await vector_store.add(ku1)
        await vector_store.add(ku2)
        await vector_store.add(ku3)

        # Generate a query embedding
        query_embedding = await vector_store.generate_embedding("Python programming for AI")

        # Search based on this embedding
        results = await vector_store.similarity_search(query_embedding)

        # Verify we got results in the correct format
        assert len(results) > 0
        assert all(isinstance(ku, KnowledgeUnit) for ku, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        assert all(0 <= score <= 1 for _, score in results)

    @pytest.mark.asyncio
    async def test_search_with_text_query(self, vector_store):
        """Test searching with a text query."""
        await vector_store.initialize()

        # Add multiple knowledge units
        ku1 = KnowledgeUnit(
            original_chunk="Python is a programming language used for AI", knowledge_source="corpus"
        )
        ku2 = KnowledgeUnit(
            original_chunk="LangChain is a framework for LLM applications",
            knowledge_source="corpus",
        )

        await vector_store.add(ku1)
        await vector_store.add(ku2)

        # Search with a text query
        results = await vector_store.search("Python AI")

        # Verify we got results
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_filter_with_similarity_search(self, vector_store):
        """Test filtering with similarity search."""
        await vector_store.initialize()

        # Add knowledge units with different sources
        ku1 = KnowledgeUnit(original_chunk="Python for data science", knowledge_source="corpus")
        ku2 = KnowledgeUnit(original_chunk="Python for web development", knowledge_source="action")

        await vector_store.add(ku1)
        await vector_store.add(ku2)

        # Generate a query embedding
        query_embedding = await vector_store.generate_embedding("Python programming")

        # Search with a source filter
        results = await vector_store.similarity_search(
            query_embedding, filter_criteria={"knowledge_source": "action"}
        )

        # Verify we only got action-sourced results
        assert len(results) > 0
        assert all(ku.knowledge_source == "action" for ku, _ in results)
