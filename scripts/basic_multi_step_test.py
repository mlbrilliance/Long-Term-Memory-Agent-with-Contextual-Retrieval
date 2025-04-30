"""
Basic test for multi-step learning in LongTermMemoryAgent.
This minimalist script tests knowledge building across interactions.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_basic_multi_step_learning():
    """Test basic knowledge building over multiple steps."""
    print("\n===== Basic Multi-Step Learning Test =====\n")

    # Create memory components
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(memory_store=vector_store, contextualizer=SimpleContextualizer())

    # Initialize memory manager
    await memory_manager.initialize()
    print("Memory manager initialized")

    # Step 1: Add initial knowledge about Python
    print("\nStep 1: Adding initial knowledge about Python")
    python_id = await memory_manager.add_knowledge(
        content="Python is a high-level programming language known for its readability and versatility.",
        source="corpus",
        metadata={"topic": "programming", "subject": "Python"},
    )
    print(f"Added knowledge unit: {python_id}")

    # Step 2: Add related knowledge about Python libraries
    print("\nStep 2: Adding knowledge about Python libraries")
    libraries_id = await memory_manager.add_knowledge(
        content="Python has many libraries including NumPy for numerical computing and Pandas for data analysis.",
        source="corpus",
        metadata={
            "topic": "programming",
            "subject": "Python libraries",
            "related_to": python_id,  # Reference to first knowledge unit
        },
    )
    print(f"Added knowledge unit: {libraries_id}")

    # Step 3: Add knowledge that connects Python to a new domain (ML)
    print("\nStep 3: Adding knowledge connecting Python to machine learning")
    ml_id = await memory_manager.add_knowledge(
        content="Python is widely used in machine learning due to libraries like TensorFlow and PyTorch.",
        source="corpus",
        metadata={
            "topic": "machine learning",
            "subject": "Python in ML",
            "related_to": [python_id, libraries_id],  # References to previous knowledge
        },
    )
    print(f"Added knowledge unit: {ml_id}")

    # Step 4: Retrieve knowledge with a query that should connect all domains
    print("\nStep 4: Testing knowledge retrieval across domains")
    query = "How is Python used in machine learning and what libraries are important?"

    # Retrieve related knowledge
    related_units = await memory_manager.get_related_knowledge(query, limit=5)
    print(f"Query: '{query}'")
    print(f"Found {len(related_units)} related knowledge units:")

    for i, (unit, score) in enumerate(related_units):
        print(f"{i + 1}. {unit.original_chunk} (relevance: {score:.2f})")

        # Show metadata to verify connections
        if unit.metadata and "related_to" in unit.metadata:
            print(f"   Connected to: {unit.metadata['related_to']}")

    # Step 5: Verify that we can retrieve all connected knowledge
    print("\nStep 5: Verifying connected knowledge retrieval")

    # Get the ML knowledge unit
    ml_unit = await memory_manager.retrieve_knowledge(ml_id)

    # Check if it has connections
    if ml_unit.metadata and "related_to" in ml_unit.metadata:
        related_ids = ml_unit.metadata["related_to"]
        print(f"ML knowledge unit has {len(related_ids)} connections")

        # Retrieve each connected unit
        for related_id in related_ids:
            related_unit = await memory_manager.retrieve_knowledge(related_id)
            print(f"Connected to: {related_unit.original_chunk}")

    print("\n===== Test Complete =====")


if __name__ == "__main__":
    asyncio.run(test_basic_multi_step_learning())
