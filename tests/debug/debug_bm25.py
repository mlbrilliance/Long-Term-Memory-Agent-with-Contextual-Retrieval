"""
Debug script for BM25Store.

This standalone script tests each aspect of the BM25Store implementation individually
and outputs detailed error information to help diagnose issues.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
import logging

from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.utils.test_utils import create_test_knowledge_unit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("BM25Debug")


async def debug_bm25_store():
    """Debug the RankBM25Store implementation step by step."""
    try:
        print("\n=== STEP 1: Initialize BM25Store ===")
        store = RankBM25Store(index_path="test_bm25_index.pkl")
        print("✓ Initialization successful")

        print("\n=== STEP 2: Create test knowledge units ===")
        units = {}
        units["id1"] = create_test_knowledge_unit(
            content="Python is a high-level programming language",
            context="Python is known for its readability and versatility",
        )
        units["id2"] = create_test_knowledge_unit(
            content="Machine learning algorithms learn from data",
            context="ML systems improve through experience with data",
        )
        print(f"✓ Created {len(units)} test knowledge units")

        print("\n=== STEP 3: Update index ===")
        try:
            await store.update_index(units)
            print("✓ Index update successful")
            print(f"  - BM25 instance: {store.bm25 is not None}")
            print(f"  - Tokenized corpus length: {len(store.tokenized_corpus)}")
            print(f"  - ID map entries: {len(store.id_map)}")
        except Exception as e:
            print(f"✗ Index update failed: {e}")
            traceback.print_exc()
            return

        print("\n=== STEP 4: Test contains ===")
        try:
            contains_id1 = await store.contains("id1")
            print(f"✓ Contains 'id1': {contains_id1}")
        except Exception as e:
            print(f"✗ Contains check failed: {e}")
            traceback.print_exc()
            return

        print("\n=== STEP 5: Test search ===")
        try:
            results = await store.search("python programming", k=2)
            print(f"✓ Search successful, returned {len(results)} results")
            for i, (unit_id, score) in enumerate(results):
                print(f"  - Result {i + 1}: ID={unit_id}, Score={score}")
        except Exception as e:
            print(f"✗ Search failed: {e}")
            traceback.print_exc()
            return

        print("\n=== STEP 6: Test remove ===")
        try:
            await store.remove(["id1"])
            contains_after = await store.contains("id1")
            print(f"✓ Remove successful, contains 'id1' after removal: {contains_after}")
        except Exception as e:
            print(f"✗ Remove failed: {e}")
            traceback.print_exc()
            return

        print("\n=== STEP 7: Test save index ===")
        try:
            await store.save_index("test_debug_bm25.pkl")
            print("✓ Save successful to test_debug_bm25.pkl")
        except Exception as e:
            print(f"✗ Save failed: {e}")
            traceback.print_exc()
            return

        print("\n=== STEP 8: Test load index ===")
        try:
            new_store = RankBM25Store(index_path="test_debug_bm25.pkl")
            await new_store.load_index()
            contains_id2 = await new_store.contains("id2")
            print(f"✓ Load successful, contains 'id2': {contains_id2}")
        except Exception as e:
            print(f"✗ Load failed: {e}")
            traceback.print_exc()
            return

        print("\n=== All tests passed successfully ===")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async debug function
    asyncio.run(debug_bm25_store())
