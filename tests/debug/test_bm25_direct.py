"""
Direct test script for RankBM25Store.

This script tests the RankBM25Store implementation directly
without using pytest to help diagnose issues.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.utils.test_utils import create_test_knowledge_unit


async def test_bm25_store():
    """Test the RankBM25Store implementation directly."""
    try:
        print("[INFO] Creating BM25Store...")
        store = RankBM25Store(index_path="test_bm25_index.pkl")

        print("[INFO] Creating test knowledge units...")
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

        # Print debug info about the created units
        for unit_id, unit in units.items():
            print(f"[DEBUG] Unit {unit_id}:")
            print(f"  - unique_id: {unit.unique_id}")
            print(f"  - original_chunk: {unit.original_chunk}")
            print(f"  - contextual_text: {unit.contextual_text}")
            print(f"  - knowledge_source: {unit.knowledge_source}")

        print("[INFO] Updating BM25 index...")
        await store.update_index(units)

        print("[INFO] Testing search functionality...")
        results = await store.search("python programming", k=2)

        print(f"[INFO] Search results: {results}")

        print("[INFO] Testing contains functionality...")
        for unit_id in ["id1", "id2", "id3", "non_existent_id"]:
            contains = await store.contains(unit_id)
            print(f"[INFO] Contains '{unit_id}': {contains}")

        print("[INFO] Testing remove functionality...")
        await store.remove(["id1"])

        contains = await store.contains("id1")
        print(f"[INFO] After remove, contains 'id1': {contains}")

        print("[INFO] Saving BM25 index...")
        await store.save_index()

        print("[INFO] Loading BM25 index...")
        new_store = RankBM25Store(index_path="test_bm25_index.pkl")
        await new_store.load_index()

        print("[INFO] Verifying loaded index...")
        for unit_id in ["id1", "id2", "id3"]:
            contains = await new_store.contains(unit_id)
            print(f"[INFO] After load, contains '{unit_id}': {contains}")

        print("[SUCCESS] BM25Store tests completed.")
    except Exception as e:
        print(f"[ERROR] Test failed with error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_bm25_store())
