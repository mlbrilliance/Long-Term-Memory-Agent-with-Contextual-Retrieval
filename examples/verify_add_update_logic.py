"""
Verify the Add vs. Update Logic implementation with a simple integration test.

This script demonstrates how the "Add vs. Update Logic" feature works by:
1. Adding a knowledge unit
2. Adding similar content to test the update logic
3. Adding dissimilar content to test the add logic
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run a simple integration test of the Add vs. Update Logic."""
    # Create and initialize a memory manager
    memory_store = InMemoryVectorStore(embedding_dim=128)
    memory_manager = MemoryManager(memory_store=memory_store)
    await memory_manager.initialize()

    logger.info("Memory manager initialized")

    # 1. Add initial knowledge unit
    logger.info("Adding initial knowledge unit about Python...")
    python_id = await memory_manager.add_knowledge(
        content="Python is a high-level programming language known for its readability and versatility.",
        source="corpus",
        context="Programming languages overview",
    )
    logger.info(f"Added knowledge unit with ID: {python_id}")

    # 2. Add a completely different knowledge unit
    logger.info("Adding different knowledge unit about JavaScript...")
    js_id = await memory_manager.add_knowledge(
        content="JavaScript is primarily used for web development and runs in browsers.",
        source="corpus",
        context="Web development languages",
    )
    logger.info(f"Added knowledge unit with ID: {js_id}")

    # Count knowledge units - should be 2
    count = await memory_manager.count_knowledge()
    logger.info(f"Knowledge unit count: {count} (expected: 2)")

    # 3. Add similar content to the Python knowledge unit
    logger.info("Adding similar content about Python (should update existing unit)...")
    updated_python_id = await memory_manager.add_knowledge(
        content="Python is a widely used high-level programming language valued for its readability and versatility in data science.",
        source="corpus",
        context="Programming languages and data science",
        similarity_threshold=0.7,  # Reasonable threshold for similar content
    )
    logger.info(f"Result ID: {updated_python_id}")

    # Check if update happened (IDs should match)
    is_update = updated_python_id == python_id
    logger.info(f"Was this an update? {is_update} (expected: True)")

    # Retrieve the updated knowledge unit
    updated_ku = await memory_manager.retrieve_knowledge(updated_python_id)
    logger.info(f"Updated content: {updated_ku.original_chunk}")
    logger.info(f"Updated context: {updated_ku.contextual_text}")

    # Count knowledge units - should still be 2
    count = await memory_manager.count_knowledge()
    logger.info(f"Knowledge unit count: {count} (expected: 2)")

    # 4. Add similar content but force it to be a new unit
    logger.info("Adding similar content about Python but forcing it to be new...")
    forced_new_id = await memory_manager.add_knowledge(
        content="Python is an interpreted language used in web development, data analysis, and AI.",
        source="corpus",
        context="Python applications",
        update_if_similar=False,  # Force add as new
    )
    logger.info(f"Result ID: {forced_new_id}")

    # Verify it's a new unit
    is_new = forced_new_id != python_id and forced_new_id != updated_python_id
    logger.info(f"Is this a new unit? {is_new} (expected: True)")

    # Count knowledge units - should now be 3
    count = await memory_manager.count_knowledge()
    logger.info(f"Knowledge unit count: {count} (expected: 3)")

    # 5. Add content with high similarity threshold
    logger.info("Adding somewhat related content with high similarity threshold...")
    high_threshold_id = await memory_manager.add_knowledge(
        content="Python libraries like NumPy and Pandas are essential for data analysis.",
        source="corpus",
        context="Python libraries",
        similarity_threshold=0.95,  # Very high threshold
    )
    logger.info(f"Result ID: {high_threshold_id}")

    # Verify it's a new unit (due to high threshold)
    is_new = high_threshold_id != python_id and high_threshold_id != forced_new_id
    logger.info(f"Is this a new unit? {is_new} (expected: True)")

    # Count knowledge units - should now be 4
    count = await memory_manager.count_knowledge()
    logger.info(f"Knowledge unit count: {count} (expected: 4)")

    logger.info("Integration test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
