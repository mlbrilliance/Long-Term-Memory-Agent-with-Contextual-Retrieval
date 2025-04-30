"""
Minimal test for BM25Store.

This script tests just the core BM25Store functionality to diagnose issues.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
src_path = os.path.join(project_root, "src")
sys.path.insert(0, str(src_path))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.bm25_store import RankBM25Store


# Test data
def create_simple_knowledge_unit(id_num, text):
    """Create a simple knowledge unit for testing."""
    return KnowledgeUnit(
        unique_id=f"id{id_num}",
        original_chunk=text,
        contextual_text="",
        embedding_vector=[0.1] * 10,  # Simple mock embedding
        knowledge_source="test",
        timestamp=None,
        metadata={},
        tags=[],
    )


async def run_test():
    """Run a minimal test of the BM25Store."""
    print("Creating BM25Store...")
    store = RankBM25Store(index_path="minimal_test.pkl")

    print("Creating test units...")
    units = {}
    units["id1"] = create_simple_knowledge_unit(1, "Python is a programming language")
    units["id2"] = create_simple_knowledge_unit(2, "Machine learning is fascinating")

    print("Updating index...")
    await store.update_index(units)

    print("Testing search...")
    results = await store.search("python", k=1)
    print(f"Search results: {results}")

    print("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_test())
