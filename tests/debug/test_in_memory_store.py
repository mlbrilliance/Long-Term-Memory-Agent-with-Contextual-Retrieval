"""
Custom test script for testing the in-memory store implementations.

This script directly tests the in-memory store functionality without using pytest.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("in_memory_store_tests")

# Add the project root and src directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

logger.info(f"Python version: {sys.version}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Working directory: {os.getcwd()}")


def log_separator(title: str = None):
    """Print a separator line with optional title."""
    if title:
        logger.info(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        logger.info(f"\n{'=' * 50}")


async def test_in_memory_store():
    """Test the InMemoryStore implementation."""
    try:
        from ltm_agent.core.models import KnowledgeUnit
        from ltm_agent.memory.in_memory_store import InMemoryStore

        log_separator("TESTING IN-MEMORY STORE")

        # Initialize the store
        logger.info("Creating InMemoryStore instance...")
        store = InMemoryStore()

        # Test initialization
        logger.info("Testing initialization...")
        assert store.initialized is False, "Store should not be initialized initially"

        await store.initialize()
        assert store.initialized is True, "Store should be initialized after initialize() call"
        logger.info("PASSED: Initialization test passed")

        # Create a test knowledge unit
        test_ku = KnowledgeUnit(
            original_chunk="This is a test knowledge unit for the in-memory store",
            knowledge_source="action",
        )

        # Test adding a knowledge unit
        logger.info("\nTesting add operation...")
        unique_id = await store.add(test_ku)
        assert unique_id is not None, "add() should return a unique ID"
        assert unique_id == test_ku.unique_id, "Returned ID should match the knowledge unit's ID"
        logger.info(f"PASSED: Add test passed, created knowledge unit with ID: {unique_id}")

        # Test retrieving a knowledge unit
        logger.info("\nTesting get operation...")
        retrieved_ku = await store.get(unique_id)
        assert retrieved_ku is not None, "get() should return the knowledge unit"
        assert retrieved_ku.unique_id == unique_id, "Retrieved unit should have the correct ID"
        assert (
            retrieved_ku.original_chunk == test_ku.original_chunk
        ), "Retrieved content should match"
        logger.info(f"PASSED: Get test passed, retrieved: {retrieved_ku.original_chunk}")

        # Test updating a knowledge unit
        logger.info("\nTesting update operation...")
        updated_ku = KnowledgeUnit(
            unique_id=unique_id,
            original_chunk="This is an updated knowledge unit",
            knowledge_source="feedback",
        )
        result = await store.update(updated_ku)
        assert result is True, "update() should return True on success"

        # Verify the update
        retrieved_ku = await store.get(unique_id)
        assert (
            retrieved_ku.original_chunk == "This is an updated knowledge unit"
        ), "Content should be updated"
        assert retrieved_ku.knowledge_source == "feedback", "Source should be updated"
        logger.info(f"PASSED: Update test passed, updated content: {retrieved_ku.original_chunk}")

        # Add more knowledge units for searching and filtering
        logger.info("\nAdding additional knowledge units for search tests...")
        ku1 = KnowledgeUnit(
            original_chunk="Python is a versatile programming language", knowledge_source="corpus"
        )
        ku2 = KnowledgeUnit(
            original_chunk="LangChain is a framework for LLM applications",
            knowledge_source="corpus",
        )
        ku3 = KnowledgeUnit(
            original_chunk="Memory management is crucial for agents", knowledge_source="action"
        )

        await store.add(ku1)
        await store.add(ku2)
        await store.add(ku3)
        logger.info("Added 3 additional knowledge units")

        # Test searching
        logger.info("\nTesting search operation...")
        search_results = await store.search("Python programming")
        assert len(search_results) > 0, "Search should return matching results"
        top_result, score = search_results[0]
        assert "Python" in top_result.original_chunk, "Top result should be relevant to query"
        logger.info(
            f"PASSED: Search test passed, top result: {top_result.original_chunk} (score: {score:.4f})"
        )

        # Test filtering
        logger.info("\nTesting search with filtering...")
        filtered_results = await store.search(
            "memory", filter_criteria={"knowledge_source": "action"}
        )
        assert len(filtered_results) > 0, "Filtered search should return results"
        assert all(
            ku.knowledge_source == "action" for ku, _ in filtered_results
        ), "All results should match filter"
        logger.info(
            f"PASSED: Filter test passed, got {len(filtered_results)} result(s) matching filter"
        )

        # Test listing
        logger.info("\nTesting list operation...")
        all_units = await store.list()
        assert len(all_units) == 4, "List should return all knowledge units"

        # Test with pagination
        paginated_units = await store.list(limit=2, offset=1)
        assert len(paginated_units) == 2, "Paginated list should return requested number of units"
        logger.info(f"PASSED: List test passed, found {len(all_units)} total units")

        # Test counting
        logger.info("\nTesting count operation...")
        total_count = await store.count()
        assert total_count == 4, "Count should return the total number of units"

        corpus_count = await store.count(filter_criteria={"knowledge_source": "corpus"})
        assert corpus_count == 2, "Filtered count should return matching count"
        logger.info(f"PASSED: Count test passed, total: {total_count}, corpus: {corpus_count}")

        # Test deletion
        logger.info("\nTesting delete operation...")
        result = await store.delete(unique_id)
        assert result is True, "delete() should return True on success"

        # Verify deletion
        retrieved_ku = await store.get(unique_id)
        assert retrieved_ku is None, "Deleted unit should no longer be retrievable"

        # Check count after deletion
        new_total = await store.count()
        assert new_total == total_count - 1, "Count should decrease after deletion"
        logger.info(f"PASSED: Delete test passed, count before: {total_count}, after: {new_total}")

        logger.info("\nAll InMemoryStore tests passed successfully!")
        return True

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


async def test_in_memory_vector_store():
    """Test the InMemoryVectorStore implementation."""
    try:
        from ltm_agent.core.models import KnowledgeUnit
        from ltm_agent.memory.in_memory_store import InMemoryVectorStore

        log_separator("TESTING IN-MEMORY VECTOR STORE")

        # Initialize the vector store
        logger.info("Creating InMemoryVectorStore instance...")
        vector_store = InMemoryVectorStore(embedding_dim=10)

        # Test initialization
        logger.info("Testing initialization...")
        assert vector_store.initialized is False, "Store should not be initialized initially"

        await vector_store.initialize()
        assert (
            vector_store.initialized is True
        ), "Store should be initialized after initialize() call"
        assert vector_store.embedding_dim == 10, "Embedding dimension should be set correctly"
        logger.info("PASSED: Initialization test passed")

        # Test embedding generation
        logger.info("\nTesting embedding generation...")
        text = "This is a test sentence for embedding generation"
        embedding = await vector_store.generate_embedding(text)

        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == 10, "Embedding should have the specified dimension"
        assert all(isinstance(x, float) for x in embedding), "Embedding values should be floats"

        # Test deterministic behavior
        embedding2 = await vector_store.generate_embedding(text)
        assert embedding == embedding2, "Embeddings for the same text should be identical"
        logger.info(f"PASSED: Embedding generation test passed, dimension: {len(embedding)}")

        # Create a test knowledge unit
        test_ku = KnowledgeUnit(
            original_chunk="This is a test knowledge unit for vector similarity",
            knowledge_source="action",
        )

        # Test adding a knowledge unit with automatic embedding
        logger.info("\nTesting add with automatic embedding...")
        assert test_ku.embedding_vector is None, "Initial embedding should be None"

        unique_id = await vector_store.add(test_ku)
        retrieved_ku = await vector_store.get(unique_id)

        assert retrieved_ku.embedding_vector is not None, "Embedding should be generated on add"
        assert len(retrieved_ku.embedding_vector) == 10, "Embedding should have correct dimension"
        logger.info("PASSED: Automatic embedding test passed")

        # Add more knowledge units for similarity search
        logger.info("\nAdding knowledge units for similarity search...")
        ku1 = KnowledgeUnit(
            original_chunk="Python is great for machine learning applications",
            knowledge_source="corpus",
        )
        ku2 = KnowledgeUnit(
            original_chunk="JavaScript is commonly used for web development",
            knowledge_source="corpus",
        )
        ku3 = KnowledgeUnit(
            original_chunk="Vector databases store embeddings for similarity search",
            knowledge_source="action",
        )

        await vector_store.add(ku1)
        await vector_store.add(ku2)
        await vector_store.add(ku3)
        logger.info("Added 3 additional knowledge units with embeddings")

        # Test similarity search with embedding
        logger.info("\nTesting similarity search...")
        query_embedding = await vector_store.generate_embedding("machine learning with Python")

        sim_results = await vector_store.similarity_search(query_embedding)
        assert len(sim_results) > 0, "Similarity search should return results"

        top_result, score = sim_results[0]
        assert "Python" in top_result.original_chunk, "Top result should be relevant to query"
        assert 0 <= score <= 1, "Similarity score should be between 0 and 1"
        logger.info(
            f"PASSED: Similarity search test passed, top result: {top_result.original_chunk} (score: {score:.4f})"
        )

        # Test text-based search
        logger.info("\nTesting text search...")
        text_results = await vector_store.search("vector database")
        assert len(text_results) > 0, "Text search should return results"

        top_text_result, text_score = text_results[0]
        assert "Vector" in top_text_result.original_chunk, "Top result should be relevant to query"
        logger.info(
            f"PASSED: Text search test passed, top result: {top_text_result.original_chunk}"
        )

        # Test filtering with similarity search
        logger.info("\nTesting filtered similarity search...")
        filtered_results = await vector_store.similarity_search(
            query_embedding, filter_criteria={"knowledge_source": "corpus"}
        )

        assert len(filtered_results) > 0, "Filtered search should return results"
        assert all(
            ku.knowledge_source == "corpus" for ku, _ in filtered_results
        ), "All results should match filter"
        logger.info("PASSED: Filtered similarity search test passed")

        logger.info("\nAll InMemoryVectorStore tests passed successfully!")
        return True

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


async def run_tests():
    """Run all tests."""
    log_separator("STARTING IN-MEMORY STORE TESTS")

    # Run InMemoryStore tests
    in_memory_result = await test_in_memory_store()

    # Run InMemoryVectorStore tests
    vector_store_result = await test_in_memory_vector_store()

    # Print overall results
    log_separator("TEST RESULTS")
    logger.info(f"InMemoryStore Tests: {'PASSED' if in_memory_result else 'FAILED'}")
    logger.info(f"InMemoryVectorStore Tests: {'PASSED' if vector_store_result else 'FAILED'}")

    return in_memory_result and vector_store_result


if __name__ == "__main__":
    # Run the tests using asyncio
    asyncio.run(run_tests())
