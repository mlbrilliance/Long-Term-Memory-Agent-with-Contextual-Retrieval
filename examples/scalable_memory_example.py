"""
Example script demonstrating the scalable memory system with efficient performance.

This script shows how to configure and use:
1. Persistent vector stores (Chroma, Pinecone)
2. Batch processing for efficient operations
3. Memory pruning for large knowledge bases
4. Knowledge graph integration
"""

import argparse
import json
import logging
import os
import time

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.memory.batch_processor import BatchedVectorStore
from ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.knowledge_graph import KnowledgeGraph
from ltm_agent.memory.persistent_store import ChromaVectorStore, PineconeVectorStore
from ltm_agent.memory.pruning import MemoryPruner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_embedding_function():
    """
    Create an embedding function based on available models.

    Returns:
        Function that converts text to embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embedding_function(text: str) -> list[float]:
            return model.encode(text).tolist()

        logger.info("Using SentenceTransformer for embeddings")
        return embedding_function

    except ImportError:
        logger.warning("SentenceTransformer not available, using random embeddings")

        import numpy as np

        def random_embedding(text: str) -> list[float]:
            # Random embedding for demonstration
            np.random.seed(hash(text) % 2**32)
            return np.random.normal(0, 1, 384).tolist()

        return random_embedding


def create_vector_store(store_type: str, embedding_dim: int = 384, **kwargs) -> BatchedVectorStore:
    """
    Create a vector store based on the specified type.

    Args:
        store_type: Type of vector store ('memory', 'chroma', or 'pinecone')
        embedding_dim: Embedding dimensions
        **kwargs: Additional parameters for specific store types

    Returns:
        Batched vector store instance
    """
    embedding_function = get_embedding_function()

    if store_type == "memory":
        # In-memory vector store
        base_store = InMemoryVectorStore()
        logger.info("Created in-memory vector store")

    elif store_type == "chroma":
        # ChromaDB persistent store
        persist_dir = kwargs.get("persist_dir", "./chroma_db")
        collection_name = kwargs.get("collection_name", "ltm_agent_memory")

        base_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_dimensions=embedding_dim,
        )
        logger.info(f"Created ChromaDB vector store in {persist_dir}")

    elif store_type == "pinecone":
        # Pinecone cloud store
        api_key = kwargs.get("api_key") or os.environ.get("PINECONE_API_KEY")
        environment = kwargs.get("environment") or os.environ.get("PINECONE_ENVIRONMENT")
        index_name = kwargs.get("index_name", "ltm-agent-memory")

        if not api_key or not environment:
            raise ValueError(
                "Pinecone API key and environment required. "
                "Provide as arguments or set PINECONE_API_KEY and PINECONE_ENVIRONMENT"
            )

        base_store = PineconeVectorStore(
            api_key=api_key,
            environment=environment,
            index_name=index_name,
            namespace=kwargs.get("namespace", "default"),
            embedding_dimensions=embedding_dim,
        )
        logger.info(f"Created Pinecone vector store with index {index_name}")

    else:
        raise ValueError(f"Unknown store type: {store_type}")

    # Wrap with batch processor
    batch_size = kwargs.get("batch_size", 5)
    use_async = kwargs.get("async_processing", False)

    logger.info(f"Wrapping store with batch processor (batch_size={batch_size}, async={use_async})")
    return BatchedVectorStore(
        base_store,
        batch_size=batch_size,
        embedding_function=embedding_function,
        use_async=use_async,
    )


def create_agent(store_type: str, **kwargs):
    """
    Create an agent with enhanced memory capabilities.

    Args:
        store_type: Type of vector store to use
        **kwargs: Additional parameters

    Returns:
        Tuple of (agent, memory_manager, memory_pruner)
    """
    # Create vector store
    vector_store = create_vector_store(
        store_type=store_type,
        batch_size=kwargs.get("batch_size", 5),
        async_processing=kwargs.get("async_processing", False),
        persist_dir=kwargs.get("persist_dir"),
        api_key=kwargs.get("api_key"),
        environment=kwargs.get("environment"),
    )

    # Create memory components
    knowledge_graph = KnowledgeGraph()
    contextualizer = EnhancedContextualizer(use_graph=True)

    # Create memory manager
    memory_manager = EnhancedMemoryManager(
        memory_store=vector_store,
        contextualizer=contextualizer,
        knowledge_graph=knowledge_graph,
    )

    # Memory pruner (for cleaning up less relevant or redundant memories)
    pruner = MemoryPruner(memory_manager)

    # Create the agent
    agent = LongTermMemoryAgent(memory_manager=memory_manager)

    logger.info("Created agent with enhanced memory system")
    return agent, memory_manager, pruner


def load_test_data(file_path: str) -> list[str]:
    """
    Load test data from a file.

    Args:
        file_path: Path to the test data file

    Returns:
        List of text items
    """
    try:
        if os.path.exists(file_path):
            with open(file_path) as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} items from {file_path}")
                return data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")

    # If file not found or error, generate sample data
    logger.info("Generating sample test data")
    test_data = [
        "Python is a high-level, interpreted programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Vector databases like Pinecone and ChromaDB are optimized for similarity search with embeddings.",
        "Knowledge graphs represent information as nodes and edges, allowing for semantic connections between concepts.",
        "Memory management is critical for large-scale AI systems to maintain efficiency and relevance.",
        "The LangChain framework provides tools for building applications with large language models.",
        "Embeddings are vector representations that capture semantic meaning in a numerical form.",
        "Batch processing improves throughput by handling multiple operations together.",
        "Async operations allow systems to continue work while waiting for I/O or other tasks.",
        "Pruning helps memory systems remain efficient by removing redundant or irrelevant information.",
        "Context retrieval is the process of finding relevant information based on a query.",
        "Hybrid search combines multiple retrieval methods for improved accuracy.",
        "Semantic search goes beyond keyword matching to understand the meaning behind queries.",
        "Efficient memory systems balance speed, accuracy, and resource utilization.",
        "Knowledge graphs enable multi-hop reasoning by traversing connected concepts.",
    ]

    # Save the generated data
    try:
        with open(file_path, "w") as f:
            json.dump(test_data, f)
        logger.info(f"Saved {len(test_data)} sample items to {file_path}")
    except Exception as e:
        logger.warning(f"Could not save sample data: {e}")

    return test_data


def run_performance_test(
    agent: LongTermMemoryAgent,
    memory_manager: EnhancedMemoryManager,
    pruner: MemoryPruner,
    test_data: list[str],
    batch_size: int = 5,
):
    """
    Run a performance test with the agent and test data.

    Args:
        agent: The LTM agent
        memory_manager: The memory manager
        pruner: The memory pruner
        test_data: List of text items to use
        batch_size: Number of items to process in each batch
    """
    logger.info(f"Starting performance test with {len(test_data)} items")
    start_time = time.time()

    # First, add all items to memory
    total_items = len(test_data)
    for i in range(0, total_items, batch_size):
        batch = test_data[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(total_items - 1) // batch_size + 1}")

        for item in batch:
            memory_manager.add(item, metadata={"source": "performance_test"})

    add_end_time = time.time()

    # Create query data - we'll use a subset of original data and some new queries
    query_data = [
        "Tell me about Python",
        "What is machine learning?",
        "How do vector databases work?",
        "What's the connection between knowledge graphs and memory systems?",
        "Explain batch processing efficiency",
    ]

    # Run queries
    logger.info(f"Running {len(query_data)} queries")
    query_start = time.time()

    for query in query_data:
        logger.info(f"Query: {query}")
        result = memory_manager.retrieve(query, top_k=3)
        logger.info(f"Retrieved {len(result)} items")

    query_end = time.time()

    # Run memory pruning
    logger.info("Running memory pruning")

    pruning_start = time.time()
    pruning_stats = pruner.prune_memory()
    pruning_end = time.time()

    # Get final stats
    final_count = memory_manager.memory_store.count()

    # Calculate times
    add_time = add_end_time - start_time
    query_time = query_end - query_start
    pruning_time = pruning_end - pruning_start
    total_time = time.time() - start_time

    # Report results
    logger.info("\n========== PERFORMANCE TEST RESULTS ==========")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Add time: {add_time:.2f} seconds ({total_items / add_time:.2f} items/sec)")
    logger.info(
        f"Query time: {query_time:.2f} seconds ({len(query_data) / query_time:.2f} queries/sec)"
    )
    logger.info(f"Pruning time: {pruning_time:.2f} seconds")
    logger.info(f"Items added: {total_items}")
    logger.info(f"Final item count: {final_count}")
    logger.info(f"Items pruned: {pruning_stats['total_pruned']}")
    logger.info(f"Redundant items pruned: {pruning_stats['redundant_pruned']}")
    logger.info(f"Outdated items pruned: {pruning_stats['outdated_pruned']}")
    logger.info(f"Low relevance items pruned: {pruning_stats['low_relevance_pruned']}")

    # Get knowledge graph stats
    graph_stats = memory_manager.get_knowledge_graph_stats()
    logger.info("\n========== KNOWLEDGE GRAPH STATS ==========")
    for key, value in graph_stats.items():
        logger.info(f"{key}: {value}")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Scalable Memory Example")

    parser.add_argument(
        "--store-type",
        type=str,
        choices=["memory", "chroma", "pinecone"],
        default="memory",
        help="Type of vector store to use",
    )
    parser.add_argument(
        "--persist-dir", type=str, default="./chroma_db", help="Directory for ChromaDB persistence"
    )
    parser.add_argument("--pinecone-key", type=str, help="Pinecone API key")
    parser.add_argument("--pinecone-env", type=str, help="Pinecone environment")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument(
        "--data-file", type=str, default="test_data.json", help="Data file for test items"
    )
    parser.add_argument("--async-processing", action="store_true", help="Use async processing")

    args = parser.parse_args()

    try:
        # Create agent with specified configuration
        logger.info(f"Creating agent with {args.store_type} vector store")
        agent, memory_manager, pruner = create_agent(
            store_type=args.store_type,
            persist_dir=args.persist_dir,
            api_key=args.pinecone_key,
            environment=args.pinecone_env,
            batch_size=args.batch_size,
            async_processing=args.async_processing,
        )

        # Load test data
        logger.info(f"Loading test data from {args.data_file}")
        test_data = load_test_data(args.data_file)

        # Run performance test
        run_performance_test(
            agent=agent,
            memory_manager=memory_manager,
            pruner=pruner,
            test_data=test_data,
            batch_size=args.batch_size,
        )

        # Close resources
        if hasattr(memory_manager.memory_store, "close"):
            memory_manager.memory_store.close()

        return 0
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info(
            "Please install required packages: pip install sentence-transformers chromadb pinecone-client"
        )
        return 1
    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
