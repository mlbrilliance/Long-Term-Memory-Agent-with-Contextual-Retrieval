"""
Simplified test for BM25Store.

This script tests just the core BM25Store functionality without dependencies.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
src_path = os.path.join(project_root, "src")
sys.path.insert(0, str(src_path))


# Create a simplified mock version of KnowledgeUnit
@dataclass
class MockKnowledgeUnit:
    unique_id: str
    original_chunk: str
    contextual_text: str

    # We'll only implement the properties needed by BM25Store


# Import the BM25Store (this should work since we fixed the path)
try:
    from ltm_agent.memory.bm25_store import RankBM25Store

    print("Successfully imported RankBM25Store")
except ImportError as e:
    print(f"Error importing RankBM25Store: {e}")
    sys.exit(1)


async def run_test():
    """Run a minimal test of the BM25Store."""
    try:
        print("\nCreating BM25Store...")
        store = RankBM25Store(index_path="simplified_test.pkl")
        print("✓ Created BM25Store successfully")

        print("\nCreating mock knowledge units...")
        units = {}
        units["id1"] = MockKnowledgeUnit(
            unique_id="id1",
            original_chunk="Python is a programming language",
            contextual_text="Python is known for its readability",
        )
        units["id2"] = MockKnowledgeUnit(
            unique_id="id2",
            original_chunk="Machine learning is fascinating",
            contextual_text="ML algorithms learn from data",
        )
        print(f"✓ Created {len(units)} mock knowledge units")

        # Patch the _combine_texts method if needed
        original_combine_texts = store._combine_texts

        def patched_combine_texts(unit):
            return f"{unit.contextual_text} {unit.original_chunk}".strip()

        store._combine_texts = patched_combine_texts

        print("\nUpdating index...")
        await store.update_index(units)
        print("✓ Updated index successfully")
        print(f"  - Tokenized corpus size: {len(store.tokenized_corpus)}")
        print(f"  - ID map entries: {len(store.id_map)}")

        print("\nTesting search...")
        results = await store.search("python", k=1)
        print(f"✓ Search results: {results}")

        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())
