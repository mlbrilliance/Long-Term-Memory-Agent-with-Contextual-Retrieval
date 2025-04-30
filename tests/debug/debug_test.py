"""Debug script for running tests and troubleshooting failures."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import the test classes
from tests.ltm_agent.memory.test_in_memory_store import TestInMemoryStore


async def run_test():
    """Run a single test to check for issues."""
    # Create test instance
    test_instance = TestInMemoryStore()

    # Setup
    memory_store = test_instance.memory_store()
    sample_ku = test_instance.sample_ku()

    # Run the test
    print("Testing initialization...")
    assert memory_store.initialized is False
    await memory_store.initialize()
    assert memory_store.initialized is True
    print("Initialization test passed!")

    print("Testing add and get...")
    await memory_store.initialize()
    unique_id = await memory_store.add(sample_ku)
    retrieved_ku = await memory_store.get(unique_id)
    assert retrieved_ku is not None
    assert retrieved_ku.unique_id == sample_ku.unique_id
    print("Add and get test passed!")

    print("All tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
