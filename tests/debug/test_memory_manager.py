"""
Custom test script for testing the memory manager implementation.

This script directly tests the memory manager functionality without using pytest.
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
logger = logging.getLogger("memory_manager_tests")

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


async def test_memory_manager():
    """Test the MemoryManager implementation."""
    try:
        from ltm_agent.memory.in_memory_store import InMemoryVectorStore
        from ltm_agent.memory.manager import MemoryManager

        log_separator("TESTING MEMORY MANAGER")

        # Initialize the memory store and manager
        logger.info("Creating InMemoryVectorStore for testing...")
        vector_store = InMemoryVectorStore(embedding_dim=10)

        logger.info("Creating MemoryManager instance...")
        manager = MemoryManager(memory_store=vector_store)

        # Test initialization
        logger.info("Testing initialization...")
        assert manager.initialized is False, "Manager should not be initialized initially"

        await manager.initialize()
        assert manager.initialized is True, "Manager should be initialized after initialize() call"
        assert vector_store.initialized is True, "Underlying store should also be initialized"
        logger.info("PASSED: Initialization test passed")

        # Test adding knowledge
        logger.info("\nTesting add_knowledge operation...")
        knowledge_id = await manager.add_knowledge(
            content="Python is a versatile programming language",
            source="corpus",
            context="From programming languages discussion",
        )

        assert knowledge_id is not None, "add_knowledge() should return a unique ID"
        logger.info(f"PASSED: add_knowledge test passed, created knowledge with ID: {knowledge_id}")

        # Test retrieving knowledge
        logger.info("\nTesting retrieve_knowledge operation...")
        retrieved_ku = await manager.retrieve_knowledge(knowledge_id)

        assert retrieved_ku is not None, "retrieve_knowledge() should return the knowledge unit"
        assert retrieved_ku.unique_id == knowledge_id, "Retrieved unit should have the correct ID"
        assert (
            retrieved_ku.original_chunk == "Python is a versatile programming language"
        ), "Content should match"
        assert (
            retrieved_ku.contextual_text == "From programming languages discussion"
        ), "Context should match"
        assert retrieved_ku.knowledge_source == "corpus", "Source should match"
        logger.info("PASSED: retrieve_knowledge test passed")

        # Test updating knowledge
        logger.info("\nTesting update_knowledge operation...")
        update_result = await manager.update_knowledge(
            unique_id=knowledge_id,
            content="Python is a versatile programming language for AI and data science",
            context="Updated context information",
            source="feedback",
        )

        assert update_result is True, "update_knowledge() should return True on success"

        # Verify the update
        updated_ku = await manager.retrieve_knowledge(knowledge_id)
        assert "AI and data science" in updated_ku.original_chunk, "Content should be updated"
        assert (
            updated_ku.contextual_text == "Updated context information"
        ), "Context should be updated"
        assert updated_ku.knowledge_source == "feedback", "Source should be updated"
        logger.info("PASSED: update_knowledge test passed")

        # Add more knowledge units for search and retrieval tests
        logger.info("\nAdding additional knowledge units for search tests...")
        await manager.add_knowledge(
            content="LangChain is a framework for building LLM applications", source="corpus"
        )
        await manager.add_knowledge(
            content="Vector databases are efficient for similarity search operations",
            source="action",
        )
        await manager.add_knowledge(
            content="Memory management is crucial for long-term agent capabilities", source="action"
        )
        logger.info("Added 3 additional knowledge entries")

        # Test searching knowledge
        logger.info("\nTesting search_knowledge operation...")
        search_results = await manager.search_knowledge("Python programming")

        assert len(search_results) > 0, "search_knowledge should return matching results"
        top_result, score = search_results[0]
        assert "Python" in top_result.original_chunk, "Top result should be relevant to query"
        logger.info(
            f"PASSED: search_knowledge test passed, found {len(search_results)} relevant results"
        )

        # Test filtered search
        logger.info("\nTesting search with source filtering...")
        filtered_results = await manager.search_knowledge(query="memory", source_filter="action")

        assert len(filtered_results) > 0, "Filtered search should return results"
        assert all(
            ku.knowledge_source == "action" for ku, _ in filtered_results
        ), "All results should match filter"
        logger.info("PASSED: Filtered search test passed")

        # Test related knowledge retrieval
        logger.info("\nTesting get_related_knowledge operation...")
        related_results = await manager.get_related_knowledge(
            content="How does memory work in AI systems?"
        )

        assert len(related_results) > 0, "get_related_knowledge should return relevant results"
        top_related, rel_score = related_results[0]
        assert "memory" in top_related.original_chunk.lower(), "Top result should be about memory"
        logger.info("PASSED: get_related_knowledge test passed")

        # Test knowledge listing
        logger.info("\nTesting list_knowledge operation...")
        all_knowledge = await manager.list_knowledge()

        assert len(all_knowledge) == 4, "list_knowledge should return all knowledge units"

        # Test with source filter
        action_knowledge = await manager.list_knowledge(source_filter="action")
        assert len(action_knowledge) == 2, "Filtered list should return 2 action units"

        # Test with pagination
        page_knowledge = await manager.list_knowledge(limit=2, offset=1)
        assert len(page_knowledge) == 2, "Paginated list should return requested number of units"
        logger.info(
            f"PASSED: list_knowledge test passed, total: {len(all_knowledge)}, action: {len(action_knowledge)}"
        )

        # Test counting knowledge
        logger.info("\nTesting count_knowledge operation...")
        total_count = await manager.count_knowledge()
        assert total_count == 4, "count_knowledge should return the total number of units"

        action_count = await manager.count_knowledge(source_filter="action")
        assert action_count == 2, "Filtered count should return matching count"
        logger.info(
            f"PASSED: count_knowledge test passed, total: {total_count}, action: {action_count}"
        )

        # Test deletion
        logger.info("\nTesting delete_knowledge operation...")
        delete_result = await manager.delete_knowledge(knowledge_id)

        assert delete_result is True, "delete_knowledge() should return True on success"

        # Verify deletion
        deleted_ku = await manager.retrieve_knowledge(knowledge_id)
        assert deleted_ku is None, "Deleted unit should no longer be retrievable"

        # Check count after deletion
        new_count = await manager.count_knowledge()
        assert new_count == total_count - 1, "Count should decrease after deletion"
        logger.info(
            f"PASSED: delete_knowledge test passed, count before: {total_count}, after: {new_count}"
        )

        logger.info("\nAll MemoryManager tests passed successfully!")
        return True

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


async def run_tests():
    """Run all tests."""
    log_separator("STARTING MEMORY MANAGER TESTS")

    # Run MemoryManager tests
    result = await test_memory_manager()

    # Print overall results
    log_separator("TEST RESULTS")
    logger.info(f"MemoryManager Tests: {'PASSED' if result else 'FAILED'}")

    return result


if __name__ == "__main__":
    # Run the tests using asyncio
    asyncio.run(run_tests())
