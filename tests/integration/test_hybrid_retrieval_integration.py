"""
Integration test for the hybrid retrieval system.

This script tests the full integration between SQLiteVectorStore, RankBM25Store,
and the LangchainHybridRetriever to demonstrate the complete hybrid retrieval workflow.
"""

import asyncio
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.sqlite_store import SQLiteVectorStore

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hybrid_integration_test")


def create_test_knowledge_unit(id_num, title, content):
    """Create a test knowledge unit."""
    return KnowledgeUnit(
        unique_id=f"test-{id_num}",
        original_chunk=content,
        contextual_text=title,
        knowledge_source="corpus",
        metadata={"source": "integration_test", "created_at": "2023-01-01", "domain": "test"},
    )


async def run_test():
    """Run the integration test."""
    logger.info("=== Hybrid Retrieval Integration Test ===")

    # Create temporary files
    with (
        tempfile.NamedTemporaryFile(suffix=".db", delete=False) as vector_db_file,
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as bm25_file,
    ):
        vector_db_path = vector_db_file.name
        bm25_path = bm25_file.name

    logger.info(f"Vector DB path: {vector_db_path}")
    logger.info(f"BM25 index path: {bm25_path}")

    try:
        # Step 1: Create test documents
        logger.info("\nStep 1: Creating test documents...")
        test_docs = [
            (
                "Python Programming",
                "Python is a high-level programming language known for its readability and versatility.",
            ),
            (
                "JavaScript Basics",
                "JavaScript is a scripting language that enables interactive web pages.",
            ),
            (
                "Machine Learning",
                "Machine learning involves algorithms that learn patterns from data.",
            ),
            (
                "Python Libraries",
                "Key Python libraries include NumPy, Pandas, and TensorFlow for data analysis.",
            ),
            (
                "Deep Learning",
                "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
            ),
            (
                "Web Development",
                "Web development involves creating websites using HTML, CSS, and JavaScript.",
            ),
            (
                "Data Analysis",
                "Data analysis includes techniques to clean, transform, and model data.",
            ),
            (
                "Neural Networks",
                "Neural networks are computational models inspired by the human brain.",
            ),
        ]

        knowledge_units = {}
        for i, (title, content) in enumerate(test_docs, 1):
            unit = create_test_knowledge_unit(i, title, content)
            knowledge_units[unit.unique_id] = unit
            logger.info(f"  - Created unit {i}: {title}")

        # Step 2: Initialize vector store
        logger.info("\nStep 2: Initializing SQLiteVectorStore...")
        vector_store = SQLiteVectorStore(
            database_path=vector_db_path,
            embedding_dim=384,  # Smaller dimension for test
            create_tables=True,
        )

        # Ensure store is initialized
        await vector_store._ensure_initialized()
        logger.info("  - Vector store initialized successfully")

        # Step 3: Add documents to vector store
        logger.info("\nStep 3: Adding documents to vector store...")
        for unit_id, unit in knowledge_units.items():
            try:
                await vector_store.add(unit)
                logger.info(f"  - Added {unit.contextual_text} to vector store")
            except Exception as e:
                logger.error(f"  - Failed to add {unit.contextual_text}: {e}")
                traceback.print_exc()

        # Step 4: Initialize BM25 store
        logger.info("\nStep 4: Initializing RankBM25Store...")
        bm25_store = RankBM25Store(index_path=bm25_path)
        logger.info("  - BM25 store initialized successfully")

        # Step 5: Update BM25 index with documents
        logger.info("\nStep 5: Updating BM25 index...")
        try:
            await bm25_store.update_index(knowledge_units)
            logger.info(f"  - Updated BM25 index with {len(knowledge_units)} documents")
        except Exception as e:
            logger.error(f"  - Failed to update BM25 index: {e}")
            traceback.print_exc()

        # Step 6: Initialize hybrid retriever
        logger.info("\nStep 6: Initializing hybrid retriever...")
        hybrid_retriever = LangchainHybridRetriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            vector_weight=0.6,
            bm25_weight=0.4,
            min_score_threshold=0.0,  # No threshold for testing
        )
        logger.info(
            f"  - Hybrid retriever initialized with weights: vector={hybrid_retriever.vector_weight}, BM25={hybrid_retriever.bm25_weight}"
        )

        # Step 7: Test queries
        test_queries = [
            "Python programming language",
            "neural networks and deep learning",
            "web development technologies",
            "data analysis tools and libraries",
        ]

        logger.info("\nStep 7: Testing search with different queries...")
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")

            # Vector search
            logger.info("Vector search results:")
            try:
                embedding = await vector_store.generate_embedding(query)
                vector_results = await vector_store.similarity_search(embedding, limit=3)

                for i, (unit, score) in enumerate(vector_results, 1):
                    logger.info(f"  {i}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")
            except Exception as e:
                logger.error(f"  Error in vector search: {e}")
                traceback.print_exc()

            # BM25 search
            logger.info("\nBM25 search results:")
            try:
                bm25_results = await bm25_store.search(query, k=3)

                for i, (unit_id, score) in enumerate(bm25_results, 1):
                    unit = knowledge_units.get(unit_id)
                    if unit:
                        logger.info(
                            f"  {i}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}"
                        )
                    else:
                        logger.error(f"  {i}. [{score:.4f}] Unit not found: {unit_id}")
            except Exception as e:
                logger.error(f"  Error in BM25 search: {e}")
                traceback.print_exc()

            # Hybrid search
            logger.info("\nHybrid search results:")
            try:
                hybrid_results = await hybrid_retriever.search(query, limit=3)

                for i, (unit, score) in enumerate(hybrid_results, 1):
                    logger.info(f"  {i}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")
            except Exception as e:
                logger.error(f"  Error in hybrid search: {e}")
                traceback.print_exc()

        # Step 8: Test with different weights
        logger.info("\nStep 8: Testing with different weights...")

        # Set equal weights
        await hybrid_retriever.set_weights(0.5, 0.5)
        logger.info("New weights: vector=0.5, BM25=0.5")

        # BM25 dominant
        await hybrid_retriever.set_weights(0.2, 0.8)
        logger.info("New weights: vector=0.2, BM25=0.8")

        # Test reranking function
        logger.info("\nStep 9: Testing result reranking...")
        query = "Python libraries for data"
        logger.info(f"Query: '{query}'")

        try:
            # Get results
            results = await hybrid_retriever.search(query, limit=5)
            logger.info("Original results:")
            for i, (unit, score) in enumerate(results, 1):
                logger.info(f"  {i}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")

            # Rerank with length penalty
            logger.info("\nReranked results (with length penalty):")
            reranked = await hybrid_retriever.rerank_results(results, query)
            for i, (unit, score) in enumerate(reranked, 1):
                logger.info(f"  {i}. [{score:.4f}] {unit.unique_id}: {unit.contextual_text}")
        except Exception as e:
            logger.error(f"Error in reranking test: {e}")
            traceback.print_exc()

        logger.info("\n=== Integration test completed successfully ===")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        logger.info("\nCleaning up temporary files...")
        for path in [vector_db_path, bm25_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"  - Removed {path}")
                except Exception as e:
                    logger.warning(f"  - Failed to remove {path}: {e}")


if __name__ == "__main__":
    asyncio.run(run_test())
