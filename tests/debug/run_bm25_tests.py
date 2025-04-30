"""
Direct test runner for BM25Store tests.

This script runs tests for the BM25Store implementation directly,
providing detailed output about any failures.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = os.path.join(project_root, "src")
sys.path.insert(0, str(src_path))

try:
    from ltm_agent.core.models import KnowledgeUnit
    from ltm_agent.memory.bm25_store import RankBM25Store
    from ltm_agent.utils.test_utils import create_test_knowledge_unit

    print("Successfully imported required modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def create_test_units():
    """Create test knowledge units."""
    units = {}
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


async def test_bm25_store_basic_functionality():
    """Test basic functionality of the BM25Store."""
    try:
        print("\n--- TEST 1: Basic Initialization ---")
        store = RankBM25Store(index_path="test_runner.pkl")
        print("✓ Store initialized successfully")

        print("\n--- TEST 2: Update Index ---")
        units = create_test_units()
        await store.update_index(units)
        print("✓ Index updated successfully")
        print(f"  - Tokenized corpus size: {len(store.tokenized_corpus)}")
        print(f"  - ID map entries: {len(store.id_map)}")

        print("\n--- TEST 3: Contains ---")
        contains_id1 = await store.contains("id1")
        contains_nonexistent = await store.contains("nonexistent")
        print(f"✓ Contains 'id1': {contains_id1}")
        print(f"✓ Contains 'nonexistent': {contains_nonexistent}")

        print("\n--- TEST 4: Search ---")
        results = await store.search("python programming", k=2)
        print(f"✓ Search results: {results}")

        print("\n--- TEST 5: Remove ---")
        await store.remove(["id1"])
        contains_after = await store.contains("id1")
        print(f"✓ Contains 'id1' after removal: {contains_after}")

        print("\n--- TEST 6: Save Index ---")
        test_file = "test_runner_save.pkl"
        await store.save_index(test_file)
        print(f"✓ Index saved to {test_file}")

        print("\n--- TEST 7: Load Index ---")
        new_store = RankBM25Store(index_path=test_file)
        await new_store.load_index()
        contains_id2 = await new_store.contains("id2")
        print(f"✓ Contains 'id2' after loading: {contains_id2}")

        print("\nAll tests passed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_bm25_store_basic_functionality())
    sys.exit(0 if success else 1)
