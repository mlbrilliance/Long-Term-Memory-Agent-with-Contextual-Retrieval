"""
Simple test for the hybrid retriever with mock data.

This script tests the HybridRetriever using mock vector and BM25 stores,
eliminating dependencies on actual database implementations.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.interfaces import VectorStore


def create_mock_knowledge_unit(id_num, content="Test content", context="Test context"):
    """Create a simple knowledge unit for testing."""
    return KnowledgeUnit(
        unique_id=f"test-{id_num}",
        original_chunk=content,
        contextual_text=context,
        knowledge_source="corpus",
        metadata={"source": "test", "created_at": "2023-01-01"},
    )


async def main():
    """Run a simple test for the hybrid retriever with mock data."""
    print("=== Simple Hybrid Retriever Test ===\n")

    # Create mock units
    print("1. Creating test knowledge units...")
    units = {
        f"test-{i}": create_mock_knowledge_unit(i, f"Content for unit {i}", f"Context for unit {i}")
        for i in range(1, 6)
    }
    for unit_id, unit in units.items():
        print(f"  - Created unit {unit_id}: {unit.contextual_text}")

    # Create mock vector store
    print("\n2. Setting up mock vector store...")
    mock_vector_store = MagicMock(spec=VectorStore)

    # Setup search results for the vector store
    vector_results = [(units["test-1"], 0.9), (units["test-3"], 0.7), (units["test-5"], 0.5)]
    mock_vector_store.similarity_search = AsyncMock(return_value=vector_results)
    mock_vector_store.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    # Fix the mock get method to properly return knowledge units
    async def mock_get(unit_id):
        return units.get(unit_id)

    mock_vector_store.get = AsyncMock(side_effect=mock_get)

    # Create mock BM25 store
    print("3. Setting up mock BM25 store...")
    mock_bm25_store = MagicMock(spec=BaseBM25Store)

    # Setup search results for the BM25 store
    bm25_results = [("test-2", 0.85), ("test-4", 0.75), ("test-1", 0.6)]
    mock_bm25_store.search = AsyncMock(return_value=bm25_results)

    # Create hybrid retriever
    print("4. Creating hybrid retriever...")
    hybrid_retriever = LangchainHybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        vector_weight=0.6,
        bm25_weight=0.4,
        min_score_threshold=0.0,
    )

    # Test hybrid search
    print("\n5. Testing hybrid search...")
    query = "test query"

    # Display expected vector results
    print(f"\nExpected vector search results for query: '{query}'")
    for i, (unit, score) in enumerate(vector_results):
        print(f"  {i + 1}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")

    # Display expected BM25 results
    print(f"\nExpected BM25 search results for query: '{query}'")
    for i, (unit_id, score) in enumerate(bm25_results):
        unit = units.get(unit_id)
        if unit:
            print(f"  {i + 1}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")

    # Perform hybrid search
    print(f"\nActual hybrid search results for query: '{query}'")
    try:
        hybrid_results = await hybrid_retriever.search(query, limit=5)

        for i, (unit, score) in enumerate(hybrid_results):
            print(f"  {i + 1}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")

        print("\nSearch successful!")
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        import traceback

        traceback.print_exc()

    # Test with different weights
    print("\n6. Testing with different weights (BM25 dominant)...")
    await hybrid_retriever.set_weights(vector_weight=0.3, bm25_weight=0.7)
    print("New weights - Vector: 0.3, BM25: 0.7")

    try:
        hybrid_results = await hybrid_retriever.search(query, limit=5)

        print("\nHybrid search results with BM25 dominant:")
        for i, (unit, score) in enumerate(hybrid_results):
            print(f"  {i + 1}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")

        print("\nSearch successful!")
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Test completed ===")


if __name__ == "__main__":
    asyncio.run(main())
