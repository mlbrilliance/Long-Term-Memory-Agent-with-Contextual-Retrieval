"""
Example demonstrating hybrid retrieval using vector and BM25 stores.

This script showcases how to use the hybrid retriever to combine
results from both vector similarity and BM25 lexical search for
improved retrieval performance.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.sqlite_store import SQLiteVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Utility function to create a knowledge unit directly
def create_knowledge_unit(content: str, context: str, unique_id: str) -> KnowledgeUnit:
    """Create a knowledge unit without relying on test utilities."""
    return KnowledgeUnit(
        unique_id=unique_id,
        original_chunk=content,
        contextual_text=context,
        knowledge_source="corpus",  # Required field: must be "corpus", "action", or "feedback"
        metadata={"source": "example", "created_at": "2023-01-01"},
    )


async def main():
    """Run the hybrid retrieval example."""
    print("=== Hybrid Retrieval Example ===\n")

    # Create temporary file paths
    database_path = "example_vector_store.db"
    bm25_path = "example_bm25_index.pkl"

    # Clean up old files if they exist
    for path in [database_path, bm25_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed existing file: {path}")

    # Initialize vector store with direct configuration
    print("\n1. Initializing vector store...")
    vector_store = SQLiteVectorStore(
        database_path=database_path,  # Use database_path instead of db_path
        embedding_dim=384,  # Dimension for the embeddings
        create_tables=True,  # Create tables if they don't exist
    )

    # Initialize BM25 store
    print("\n2. Initializing BM25 store...")
    bm25_store = RankBM25Store(index_path=bm25_path)

    # Create hybrid retriever
    print("\n3. Creating hybrid retriever...")
    hybrid_retriever = LangchainHybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        vector_weight=0.6,  # Slightly favor vector search
        bm25_weight=0.4,
        min_score_threshold=0.1,
    )

    # Create sample documents
    print("\n4. Creating sample documents...")
    documents = [
        (
            "Python programming",
            "Python is a high-level programming language known for its readability and versatility.",
        ),
        (
            "Machine learning",
            "Machine learning algorithms learn patterns from data to make predictions.",
        ),
        (
            "Neural networks",
            "Neural networks are inspired by the human brain and consist of interconnected nodes.",
        ),
        (
            "Deep learning",
            "Deep learning uses multi-layered neural networks to learn hierarchical representations.",
        ),
        (
            "Natural language processing",
            "NLP focuses on the interaction between computers and human language.",
        ),
        ("Python libraries", "Popular Python libraries include NumPy, Pandas, and TensorFlow."),
        (
            "Data science workflow",
            "A typical data science workflow includes data collection, cleaning, analysis, and visualization.",
        ),
        (
            "Feature engineering",
            "Feature engineering involves creating new features from existing data to improve model performance.",
        ),
    ]

    # Add documents to both stores
    for i, (title, content) in enumerate(documents):
        # Create knowledge unit
        unit = create_knowledge_unit(content=content, context=title, unique_id=f"doc{i + 1}")

        # Add to vector store
        await vector_store.add(unit)
        print(f"  - Added document {i + 1}: {title}")

    # Update BM25 index with all units
    units_dict = {}
    async for unit in vector_store.list():
        units_dict[unit.unique_id] = unit

    await bm25_store.update_index(units_dict)
    print(f"  - Updated BM25 index with {len(units_dict)} documents")

    # Test queries
    print("\n5. Testing retrieval with different methods...")
    queries = [
        "Python programming language",
        "neural networks and deep learning",
        "data analysis libraries",
    ]

    for query in queries:
        print(f"\n--- Query: '{query}' ---")

        # Vector search
        print("Vector search results:")
        try:
            embedding = await vector_store.generate_embedding(query)
            vector_results = await vector_store.similarity_search(embedding, limit=3)

            for i, (unit, score) in enumerate(vector_results):
                print(
                    f"  {i + 1}. [{score:.4f}] {unit.contextual_text}: {unit.original_chunk[:50]}..."
                )
        except Exception as e:
            print(f"  Error in vector search: {e}")

        # BM25 search
        print("\nBM25 search results:")
        try:
            bm25_id_results = await bm25_store.search(query, k=3)
            bm25_results = []
            for unit_id, score in bm25_id_results:
                unit = await vector_store.get(unit_id)
                if unit:
                    bm25_results.append((unit, score))

            for i, (unit, score) in enumerate(bm25_results):
                print(
                    f"  {i + 1}. [{score:.4f}] {unit.contextual_text}: {unit.original_chunk[:50]}..."
                )
        except Exception as e:
            print(f"  Error in BM25 search: {e}")

        # Hybrid search
        print("\nHybrid search results:")
        try:
            hybrid_results = await hybrid_retriever.search(query, limit=3)

            for i, (unit, score) in enumerate(hybrid_results):
                print(
                    f"  {i + 1}. [{score:.4f}] {unit.contextual_text}: {unit.original_chunk[:50]}..."
                )
        except Exception as e:
            print(f"  Error in hybrid search: {e}")

    # Clean up
    print("\n6. Cleaning up resources...")
    for path in [database_path, bm25_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  - Removed file: {path}")

    print("\n=== Example completed ===")


if __name__ == "__main__":
    asyncio.run(main())
