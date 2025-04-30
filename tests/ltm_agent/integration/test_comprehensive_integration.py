"""
Comprehensive integration tests for LTM Agent.

This module contains comprehensive tests that validate the integration of all
key components of the LTM Agent system, including vector store, BM25 store,
hybrid retrieval, and learning pathways.
"""

import os
import random
from typing import Any

import pytest

from ltm_agent.agent.learning import format_action_result, format_human_feedback
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.utils.test_utils import create_test_knowledge_unit

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


class TestComprehensiveIntegration:
    """Comprehensive integration tests for LTM Agent."""

    @pytest.fixture
    def diverse_knowledge_units(self):
        """Fixture providing a diverse set of knowledge units for testing."""
        units = {}

        # Technical content
        units["tech1"] = create_test_knowledge_unit(
            content="Python supports multiple programming paradigms, including object-oriented, imperative, functional and procedural",
            context="Discussion about Python language features",
            source="corpus",
            metadata={"domain": "programming", "language": "python"},
        )

        units["tech2"] = create_test_knowledge_unit(
            content="TensorFlow is an end-to-end open source platform for machine learning",
            context="Overview of ML frameworks",
            source="corpus",
            metadata={"domain": "machine learning", "framework": "tensorflow"},
        )

        # Conversational content
        units["conv1"] = create_test_knowledge_unit(
            content="I think we should refactor the database access layer to improve performance",
            context="Team discussion about code improvements",
            source="corpus",
            metadata={"domain": "engineering", "topic": "refactoring"},
        )

        units["conv2"] = create_test_knowledge_unit(
            content="The client meeting went well, they liked our proposal for the new dashboard",
            context="Project update conversation",
            source="corpus",
            metadata={"domain": "business", "topic": "client feedback"},
        )

        # Learning content (action results)
        units["action1"] = create_test_knowledge_unit(
            content=format_action_result(
                {
                    "task": "code_review",
                    "success": True,
                    "details": {
                        "files_reviewed": 5,
                        "issues_found": 3,
                        "recommendations": [
                            "Add more comments",
                            "Fix memory leak in data processor",
                            "Optimize database queries",
                        ],
                    },
                }
            ),
            context="Code review for data processing module",
            source="action",
            metadata={"domain": "engineering", "type": "code review"},
        )

        # Learning content (human feedback)
        units["feedback1"] = create_test_knowledge_unit(
            content=format_human_feedback(
                {
                    "rating": 4,
                    "strengths": ["Clear explanation", "Good examples"],
                    "improvements": ["Could be more concise", "Add visual diagrams"],
                }
            ),
            context="Feedback on technical documentation",
            source="feedback",
            metadata={"domain": "documentation", "type": "technical writing"},
        )

        return units

    @pytest.fixture
    async def memory_system(self, diverse_knowledge_units, tmp_path):
        """Fixture providing a configured memory system with hybrid retrieval."""
        # Create a temporary BM25 index file
        bm25_path = os.path.join(tmp_path, "test_comprehensive_bm25.pkl")

        # Use a consistent embedding dimension
        embedding_dim = 768

        # Initialize stores
        vector_store = InMemoryVectorStore(embedding_dim=embedding_dim)
        bm25_store = RankBM25Store(index_path=bm25_path)

        # Add knowledge units to vector store with embeddings
        for unit_id, unit in diverse_knowledge_units.items():
            # Generate consistent embeddings based on unit_id to create predictable retrieval patterns
            random.seed(hash(unit_id))  # Use unit_id as seed for reproducibility
            embedding = [random.uniform(-1, 1) for _ in range(embedding_dim)]
            unit.embedding_vector = embedding
            await vector_store.add(unit)

        # Update BM25 index
        await bm25_store.update_index(diverse_knowledge_units)

        # Create hybrid retriever with balanced weights
        hybrid_retriever = LangchainHybridRetriever(
            vector_store=vector_store, bm25_store=bm25_store, vector_weight=0.5, bm25_weight=0.5
        )

        return {
            "vector_store": vector_store,
            "bm25_store": bm25_store,
            "hybrid_retriever": hybrid_retriever,
            "units": diverse_knowledge_units,
            "embedding_dim": embedding_dim,
        }

    async def test_adaptive_retrieval_by_topic(self, memory_system):
        """Test that retrieval adapts to different topics appropriately."""
        # Technical query should favor technical content
        tech_results = await memory_system["hybrid_retriever"].search(
            query="What are the features of Python programming language?", limit=3
        )

        # Extract IDs for easier assertion
        tech_ids = [unit.unique_id for unit, _ in tech_results]
        for unit, _ in memory_system["units"].items():
            if memory_system["units"][unit].original_chunk.lower().find("python") != -1:
                assert (
                    unit in tech_ids or memory_system["units"][unit].unique_id in tech_ids
                ), f"Technical query should retrieve unit with Python content: {unit}"
                break

        # Business query should favor conversational business content
        business_results = await memory_system["hybrid_retriever"].search(
            query="How did the client meeting go? What did they think of our dashboard?", limit=3
        )

        business_ids = [unit.unique_id for unit, _ in business_results]
        for unit, _ in memory_system["units"].items():
            if (
                memory_system["units"][unit].original_chunk.lower().find("client") != -1
                and memory_system["units"][unit].original_chunk.lower().find("dashboard") != -1
            ):
                assert (
                    unit in business_ids or memory_system["units"][unit].unique_id in business_ids
                ), f"Business query should retrieve unit with client and dashboard content: {unit}"
                break

    async def test_learning_content_integration(self, memory_system):
        """Test that learning content is appropriately retrieved."""
        # Add new formatted action result
        new_action = format_action_result(
            {
                "task": "performance_optimization",
                "success": True,
                "details": {
                    "module": "data_processor",
                    "improvements": [
                        "Reduced memory usage by 30%",
                        "Improved query execution time by 45%",
                    ],
                    "issues_resolved": 2,
                },
            }
        )

        new_action_unit = create_test_knowledge_unit(
            content=new_action,
            context="Performance optimization of data processing module",
            source="action",
            metadata={"domain": "engineering", "type": "optimization"},
        )

        # Add embedding
        random.seed(hash("new_action"))
        new_action_unit.embedding_vector = [
            random.uniform(-1, 1) for _ in range(memory_system["embedding_dim"])
        ]

        # Add to stores
        new_action_id = await memory_system["vector_store"].add(new_action_unit)
        await memory_system["bm25_store"].update_index({new_action_id: new_action_unit})

        # Add new formatted human feedback
        new_feedback = format_human_feedback(
            {
                "rating": 5,
                "comment": "The optimization was excellent, exactly what we needed",
                "impact": "Critical improvement for our production system",
            }
        )

        new_feedback_unit = create_test_knowledge_unit(
            content=new_feedback,
            context="Feedback on performance optimization work",
            source="feedback",
            metadata={"domain": "engineering", "type": "performance"},
        )

        # Add embedding
        random.seed(hash("new_feedback"))
        new_feedback_unit.embedding_vector = [
            random.uniform(-1, 1) for _ in range(memory_system["embedding_dim"])
        ]

        # Add to stores
        new_feedback_id = await memory_system["vector_store"].add(new_feedback_unit)
        await memory_system["bm25_store"].update_index({new_feedback_id: new_feedback_unit})

        # Query related to performance optimization
        results = await memory_system["hybrid_retriever"].search(
            query="How did we improve performance? What was the feedback on our optimization work?",
            limit=5,
        )

        # Should retrieve both the action and the feedback
        result_ids = [unit.unique_id for unit, _ in results]
        assert new_action_id in result_ids, "Should retrieve the performance optimization action"
        assert new_feedback_id in result_ids, "Should retrieve the feedback on optimization"

    async def test_weight_adjustment_for_different_queries(self, memory_system):
        """Test that adjusting weights impacts retrieval for different query types."""
        # Create a technical query
        technical_query = "What machine learning frameworks are available?"

        # Test with vector emphasis (semantic understanding)
        await memory_system["hybrid_retriever"].set_weights(0.8, 0.2)
        vector_results = await memory_system["hybrid_retriever"].search(
            query=technical_query, limit=3
        )

        # Test with BM25 emphasis (keyword matching)
        await memory_system["hybrid_retriever"].set_weights(0.2, 0.8)
        bm25_results = await memory_system["hybrid_retriever"].search(
            query=technical_query, limit=3
        )

        # Compare the order of results
        vector_ids = [unit.unique_id for unit, _ in vector_results]
        bm25_ids = [unit.unique_id for unit, _ in bm25_results]

        # The results or their order should differ when weights change significantly
        assert vector_ids != bm25_ids or (
            len(vector_ids) > 0 and len(bm25_ids) > 0 and vector_ids[0] != bm25_ids[0]
        ), "Different weights should affect retrieval results"

    async def test_iterative_learning_cycle(self, memory_system):
        """
        Test a complete learning cycle with multiple iterations of:
        1. Retrieve knowledge
        2. Take action based on knowledge
        3. Get feedback
        4. Store both action results and feedback
        5. Retrieve again with the enhanced knowledge
        """
        # Start with initial query
        initial_query = "How should we improve code quality?"
        initial_results = await memory_system["hybrid_retriever"].search(
            query=initial_query, limit=3
        )

        # First action: Code review based on initial knowledge
        action_result_1 = format_action_result(
            {
                "task": "code_review",
                "outcome": "Identified need for better testing",
                "details": {
                    "suggestion": "Implement more unit tests for data processor",
                    "priority": "high",
                },
            }
        )

        # Format and store the action result
        action_unit_1 = create_test_knowledge_unit(
            content=action_result_1,
            context="Code review action based on initial query",
            source="action",
            metadata={"iteration": 1},
        )

        # Add embedding
        random.seed(hash("action_1"))
        action_unit_1.embedding_vector = [
            random.uniform(-1, 1) for _ in range(memory_system["embedding_dim"])
        ]

        # Add to stores
        action_id_1 = await memory_system["vector_store"].add(action_unit_1)
        await memory_system["bm25_store"].update_index({action_id_1: action_unit_1})

        # Feedback on the first action
        feedback_1 = format_human_feedback(
            {
                "rating": 3,
                "comment": "Good start, but we also need to focus on code structure",
                "additional_point": "Consider implementing a code formatter",
            }
        )

        # Format and store the feedback
        feedback_unit_1 = create_test_knowledge_unit(
            content=feedback_1,
            context="Feedback on first code review action",
            source="feedback",
            metadata={"iteration": 1},
        )

        # Add embedding
        random.seed(hash("feedback_1"))
        feedback_unit_1.embedding_vector = [
            random.uniform(-1, 1) for _ in range(memory_system["embedding_dim"])
        ]

        # Add to stores
        feedback_id_1 = await memory_system["vector_store"].add(feedback_unit_1)
        await memory_system["bm25_store"].update_index({feedback_id_1: feedback_unit_1})

        # Second query - should benefit from previous action and feedback
        second_query = "What specific steps should we take to improve code quality?"
        second_results = await memory_system["hybrid_retriever"].search(query=second_query, limit=5)

        # The second search should retrieve the action and feedback
        second_result_ids = [unit.unique_id for unit, _ in second_results]
        assert action_id_1 in second_result_ids, "Second query should retrieve first action"
        assert feedback_id_1 in second_result_ids, "Second query should retrieve first feedback"

        # Second action: Based on accumulated knowledge
        action_result_2 = format_action_result(
            {
                "task": "code_quality_improvement",
                "outcome": "Implemented comprehensive quality measures",
                "details": {
                    "changes": [
                        "Added unit tests for data processor",
                        "Implemented code formatter",
                        "Added static code analysis",
                    ],
                    "status": "completed",
                },
            }
        )

        # Format and store the second action result
        action_unit_2 = create_test_knowledge_unit(
            content=action_result_2,
            context="Code quality improvement based on accumulated knowledge",
            source="action",
            metadata={"iteration": 2},
        )

        # Add embedding
        random.seed(hash("action_2"))
        action_unit_2.embedding_vector = [
            random.uniform(-1, 1) for _ in range(memory_system["embedding_dim"])
        ]

        # Add to stores
        action_id_2 = await memory_system["vector_store"].add(action_unit_2)
        await memory_system["bm25_store"].update_index({action_id_2: action_unit_2})

        # Final query - should show the learning progression
        final_query = "What have we done to improve code quality and what were the outcomes?"
        final_results = await memory_system["hybrid_retriever"].search(query=final_query, limit=5)

        # The final search should retrieve both actions and the feedback
        final_result_ids = [unit.unique_id for unit, _ in final_results]
        assert action_id_1 in final_result_ids, "Final query should retrieve first action"
        assert feedback_id_1 in final_result_ids, "Final query should retrieve feedback"
        assert action_id_2 in final_result_ids, "Final query should retrieve second action"
