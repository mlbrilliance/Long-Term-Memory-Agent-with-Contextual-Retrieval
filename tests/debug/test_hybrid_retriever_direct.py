"""
Direct test of the HybridRetriever implementation.

This script tests the core functionality of the HybridRetriever
without relying on pytest, providing clear error output.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
src_path = os.path.join(project_root, "src")
sys.path.insert(0, str(src_path))

from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.interfaces import VectorStore
from ltm_agent.utils.test_utils import create_test_knowledge_unit


def create_mock_vector_store():
    """Create a mock vector store for testing."""
    mock = MagicMock(spec=VectorStore)

    # Set up mock methods
    mock.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock.similarity_search = AsyncMock()
    mock.get = AsyncMock()

    return mock


def create_mock_bm25_store():
    """Create a mock BM25 store for testing."""
    mock = MagicMock(spec=BaseBM25Store)

    # Set up mock methods
    mock.search = AsyncMock()

    return mock


def create_sample_units():
    """Create sample knowledge units for testing."""
    units = {}

    # Create units with different content for testing
    units["id1"] = create_test_knowledge_unit(
        content="Python is a high-level programming language",
        context="Python is known for its readability and versatility",
    )
    units["id2"] = create_test_knowledge_unit(
        content="Machine learning algorithms learn from data",
        context="ML systems improve through experience with data",
    )
    units["id3"] = create_test_knowledge_unit(
        content="Neural networks are inspired by the human brain",
        context="Deep learning models use multiple layers of neurons",
    )

    # Store unique_ids for reference
    unit_ids = {
        "id1": units["id1"].unique_id,
        "id2": units["id2"].unique_id,
        "id3": units["id3"].unique_id,
    }

    print("Created test units with the following IDs:")
    for key, uid in unit_ids.items():
        print(f"  - {key}: {uid}")

    return units, unit_ids


async def test_hybrid_retriever():
    """Run tests for the HybridRetriever."""
    try:
        print("=== Testing HybridRetriever ===")

        print("\n--- Test 1: Initialize with valid weights ---")
        vector_store = create_mock_vector_store()
        bm25_store = create_mock_bm25_store()
        retriever = LangchainHybridRetriever(vector_store, bm25_store, 0.5, 0.5)
        print(
            f"✓ Initialized with weights: vector={retriever.vector_weight}, bm25={retriever.bm25_weight}"
        )

        print("\n--- Test 2: Set weights ---")
        await retriever.set_weights(0.7, 0.3)
        print(f"✓ Updated weights: vector={retriever.vector_weight}, bm25={retriever.bm25_weight}")

        print("\n--- Test 3: Test vector-only search ---")
        units, unit_ids = create_sample_units()
        unit_list = list(units.values())

        # Set up vector search mock
        vector_results = [(unit_list[0], 0.9), (unit_list[1], 0.7), (unit_list[2], 0.5)]
        vector_store.similarity_search.return_value = vector_results

        # Set vector weight to 1.0, BM25 weight to 0.0
        await retriever.set_weights(1.0, 0.0)

        # Perform search
        results = await retriever.search("python", limit=2)

        # Verify results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert results[0][0] == unit_list[0], "First result should be unit_list[0]"
        assert results[1][0] == unit_list[1], "Second result should be unit_list[1]"

        print(f"✓ Vector-only search returned {len(results)} results")
        print(f"  - First result: {results[0][0].unique_id} with score {results[0][1]:.4f}")
        print(f"  - Second result: {results[1][0].unique_id} with score {results[1][1]:.4f}")

        print("\n--- Test 4: Test BM25-only search ---")
        # Create a dict mapping unit_id to unit for easier lookups
        unit_map = {unit.unique_id: unit for unit in unit_list}

        # Set up BM25 search mock with actual unique_ids
        bm25_id_results = [(unit_list[0].unique_id, 0.8), (unit_list[1].unique_id, 0.6)]
        bm25_store.search.return_value = bm25_id_results

        # Set up vector store get method to return units by ID
        async def mock_get(unit_id):
            unit = unit_map.get(unit_id)
            if not unit:
                print(f"Warning: No unit found for ID {unit_id}")
            return unit

        vector_store.get.side_effect = mock_get

        # Set vector weight to 0.0, BM25 weight to 1.0
        await retriever.set_weights(0.0, 1.0)

        # Reset mock call counts
        vector_store.similarity_search.reset_mock()
        vector_store.generate_embedding.reset_mock()
        bm25_store.search.reset_mock()

        # Perform search
        print("Executing BM25-only search...")
        results = await retriever.search("python", limit=2)

        print(f"Raw results from search: {results}")

        # Verify results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Extract IDs for easier comparison
        result_ids = [result[0].unique_id for result in results]
        expected_ids = [unit_list[0].unique_id, unit_list[1].unique_id]

        print(f"Result IDs: {result_ids}")
        print(f"Expected IDs: {expected_ids}")

        assert (
            result_ids[0] == expected_ids[0]
        ), f"First result ID should be {expected_ids[0]}, got {result_ids[0]}"
        assert (
            result_ids[1] == expected_ids[1]
        ), f"Second result ID should be {expected_ids[1]}, got {result_ids[1]}"

        # Verify correct methods were called
        assert (
            not vector_store.similarity_search.called
        ), "Vector similarity_search should not be called"
        assert (
            not vector_store.generate_embedding.called
        ), "Vector generate_embedding should not be called"
        assert bm25_store.search.called, "BM25 search should be called"
        assert (
            vector_store.get.call_count == 2
        ), f"Vector get should be called twice, got {vector_store.get.call_count} calls"

        print(f"✓ BM25-only search returned {len(results)} results")
        print(f"  - First result: {results[0][0].unique_id} with score {results[0][1]:.4f}")
        print(f"  - Second result: {results[1][0].unique_id} with score {results[1][1]:.4f}")

        print("\n--- Test 5: Test hybrid search ---")
        # Set up vector search mock
        vector_results = [(unit_list[0], 0.9), (unit_list[2], 0.7)]
        vector_store.similarity_search.return_value = vector_results

        # Set up BM25 search mock
        bm25_id_results = [(unit_list[0].unique_id, 0.8), (unit_list[1].unique_id, 0.9)]
        bm25_store.search.return_value = bm25_id_results

        # Set hybrid weights
        await retriever.set_weights(0.4, 0.6)

        # Reset mock call counts
        vector_store.similarity_search.reset_mock()
        vector_store.generate_embedding.reset_mock()
        bm25_store.search.reset_mock()
        vector_store.get.reset_mock()

        # Perform search
        print("Executing hybrid search...")
        results = await retriever.search("python", limit=3)

        # Verify results
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Extract IDs for easier comparison
        result_ids = [result[0].unique_id for result in results]

        print(f"Hybrid search result IDs: {result_ids}")
        print(f"Hybrid search scores: {[score for _, score in results]}")

        # Verify correct methods were called
        assert vector_store.similarity_search.called, "Vector similarity_search should be called"
        assert vector_store.generate_embedding.called, "Vector generate_embedding should be called"
        assert bm25_store.search.called, "BM25 search should be called"
        assert (
            vector_store.get.call_count == 2
        ), f"Vector get should be called twice, got {vector_store.get.call_count} calls"

        print(f"✓ Hybrid search returned {len(results)} results")
        for i, (unit, score) in enumerate(results):
            print(f"  - Result {i + 1}: {unit.unique_id} with score {score:.4f}")

        print("\n=== All tests passed successfully! ===")
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_hybrid_retriever())
    sys.exit(0 if success else 1)
