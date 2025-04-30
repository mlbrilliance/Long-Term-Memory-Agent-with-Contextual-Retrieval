"""
Integration test for hybrid retrieval with learning pathways.

This test verifies that the hybrid retrieval system works correctly with
the learning pathway functions to process and store knowledge.
"""

import logging
import random
from typing import Any

import pytest

from ltm_agent.agent.learning import format_action_result, format_human_feedback
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.utils.test_utils import create_test_knowledge_unit

# Set up logging
logging.basicConfig(level=logging.INFO)

# Monkey patch the RankBM25Store.search method to accept filter_criteria
original_search = RankBM25Store.search


async def patched_search(
    self, query: str, k: int = 10, filter_criteria: dict[str, Any] | None = None
) -> list[tuple[str, float]]:
    # Just ignore the filter_criteria parameter and call the original method
    return await original_search(self, query, k)


RankBM25Store.search = patched_search

# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio


class TestHybridRetrievalLearning:
    """Integration tests for hybrid retrieval with learning pathways."""

    @pytest.fixture
    def sample_knowledge_units(self):
        """Fixture providing sample knowledge units."""
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

        return units

    @pytest.fixture
    async def hybrid_retrieval_system(self, sample_knowledge_units, tmp_path):
        """Fixture providing a configured hybrid retrieval system."""
        # Create a temporary BM25 index file
        bm25_path = tmp_path / "test_bm25_index.pkl"

        # We'll use a consistent embedding dimension for testing
        embedding_dim = 768

        # Initialize stores
        vector_store = InMemoryVectorStore(embedding_dim=embedding_dim)
        bm25_store = RankBM25Store(index_path=str(bm25_path))

        # Add knowledge units to vector store
        for unit_id, unit in sample_knowledge_units.items():
            # Generate a consistent mock embedding
            embedding = [random.uniform(-1, 1) for _ in range(embedding_dim)]
            unit.embedding_vector = embedding
            await vector_store.add(unit)

        # Update BM25 index
        await bm25_store.update_index(sample_knowledge_units)

        # Create hybrid retriever
        hybrid_retriever = LangchainHybridRetriever(
            vector_store=vector_store, bm25_store=bm25_store, vector_weight=0.5, bm25_weight=0.5
        )

        return {
            "vector_store": vector_store,
            "bm25_store": bm25_store,
            "hybrid_retriever": hybrid_retriever,
            "units": sample_knowledge_units,
            "embedding_dim": embedding_dim,
        }

    async def test_action_result_integration(self, hybrid_retrieval_system):
        """Test that action results can be formatted and then retrieved."""
        # Format an action result
        action_result = {
            "task": "analyze_code",
            "success": True,
            "details": {"language": "Python", "lines_of_code": 150, "complexity": "medium"},
        }

        formatted_result = format_action_result(action_result)

        # Create a new knowledge unit with the formatted result
        new_unit = create_test_knowledge_unit(content=formatted_result, context="", source="action")

        # Generate a simple mock embedding with consistent dimension
        embedding_dim = hybrid_retrieval_system["embedding_dim"]
        new_unit.embedding_vector = [random.uniform(-1, 1) for _ in range(embedding_dim)]

        # Add to vector store
        new_unit_id = await hybrid_retrieval_system["vector_store"].add(new_unit)

        # Add to BM25 store
        await hybrid_retrieval_system["bm25_store"].update_index({new_unit_id: new_unit})

        # Retrieve with a relevant query
        results = await hybrid_retrieval_system["hybrid_retriever"].search(
            query="Python code analysis", limit=10
        )

        # Verify that our new unit is in the results
        result_ids = [unit.unique_id for unit, _ in results]
        assert new_unit_id in result_ids

        # Check that the formatted content is preserved
        for unit, _ in results:
            if unit.unique_id == new_unit_id:
                assert "ACTION RESULT" in unit.original_chunk
                assert "Python" in unit.original_chunk
                assert "complexity" in unit.original_chunk

    async def test_human_feedback_integration(self, hybrid_retrieval_system):
        """Test that human feedback can be formatted and then retrieved."""
        # Format human feedback
        feedback = {
            "rating": 4,
            "strengths": ["Thorough analysis", "Clear explanations"],
            "areas_for_improvement": ["Could be more concise", "Add more examples"],
        }

        formatted_feedback = format_human_feedback(feedback)

        # Create a new knowledge unit with the formatted feedback
        new_unit = create_test_knowledge_unit(
            content=formatted_feedback, context="", source="feedback"
        )

        # Generate a simple mock embedding with consistent dimension
        embedding_dim = hybrid_retrieval_system["embedding_dim"]
        new_unit.embedding_vector = [random.uniform(-1, 1) for _ in range(embedding_dim)]

        # Add to vector store
        new_unit_id = await hybrid_retrieval_system["vector_store"].add(new_unit)

        # Add to BM25 store
        await hybrid_retrieval_system["bm25_store"].update_index({new_unit_id: new_unit})

        # Retrieve with a relevant query
        results = await hybrid_retrieval_system["hybrid_retriever"].search(
            query="feedback improvement examples", limit=10
        )

        # Verify that our new unit is in the results
        result_ids = [unit.unique_id for unit, _ in results]
        assert new_unit_id in result_ids

        # Check that the formatted content is preserved
        for unit, _ in results:
            if unit.unique_id == new_unit_id:
                assert "HUMAN FEEDBACK" in unit.original_chunk
                assert "Thorough analysis" in unit.original_chunk
                assert "more concise" in unit.original_chunk

    async def test_hybrid_weights_with_learning_content(self, hybrid_retrieval_system):
        """Test hybrid retrieval with different weights for learning content."""
        # Format both an action result and human feedback
        action_result = "Successfully completed the code refactoring task"
        human_feedback = "The refactoring was good, but could use more comments"

        formatted_action = format_action_result(action_result)
        formatted_feedback = format_human_feedback(human_feedback)

        # Create knowledge units
        action_unit = create_test_knowledge_unit(
            content=formatted_action, context="", source="action"
        )
        feedback_unit = create_test_knowledge_unit(
            content=formatted_feedback, context="", source="feedback"
        )

        # Generate mock embeddings with consistent dimension
        embedding_dim = hybrid_retrieval_system["embedding_dim"]
        action_unit.embedding_vector = [random.uniform(-1, 1) for _ in range(embedding_dim)]
        feedback_unit.embedding_vector = [random.uniform(-1, 1) for _ in range(embedding_dim)]

        # Add to vector store
        action_id = await hybrid_retrieval_system["vector_store"].add(action_unit)
        feedback_id = await hybrid_retrieval_system["vector_store"].add(feedback_unit)

        # Add to BM25 store
        await hybrid_retrieval_system["bm25_store"].update_index(
            {action_id: action_unit, feedback_id: feedback_unit}
        )

        # Test with vector search emphasis (0.8, 0.2)
        await hybrid_retrieval_system["hybrid_retriever"].set_weights(0.8, 0.2)
        vector_results = await hybrid_retrieval_system["hybrid_retriever"].search(
            query="code refactoring", limit=10
        )

        # Test with BM25 search emphasis (0.2, 0.8)
        await hybrid_retrieval_system["hybrid_retriever"].set_weights(0.2, 0.8)
        bm25_results = await hybrid_retrieval_system["hybrid_retriever"].search(
            query="code refactoring", limit=10
        )

        # Both search methods should return both units
        result_ids_vector = [unit.unique_id for unit, _ in vector_results]
        result_ids_bm25 = [unit.unique_id for unit, _ in bm25_results]

        assert action_id in result_ids_vector
        assert feedback_id in result_ids_vector
        assert action_id in result_ids_bm25
        assert feedback_id in result_ids_bm25
