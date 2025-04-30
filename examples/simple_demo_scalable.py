"""
Simple demonstration of the scalable memory system.

This script shows how to use:
1. Batched processing for better performance
2. Enhanced memory connections with the knowledge graph
3. Memory pruning for managing large knowledge bases
"""

import logging
import time

import numpy as np

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.knowledge_graph import KnowledgeGraph
from ltm_agent.memory.pruning import MemoryPruner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simple_embedding_function(text: str) -> list[float]:
    """
    Simple deterministic embedding function based on text hash.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    # Use text hash as seed for reproducibility
    np.random.seed(hash(text) % 2**32)
    # Generate a 384-dimensional embedding
    return np.random.normal(0, 1, 384).tolist()


def main():
    """Run a simple demonstration of the scalable memory system."""

    logger.info("Creating enhanced memory system...")

    # Create memory components
    vector_store = InMemoryVectorStore()
    contextualizer = EnhancedContextualizer()
    knowledge_graph = KnowledgeGraph()

    # Create enhanced memory manager
    memory_manager = EnhancedMemoryManager(
        memory_store=vector_store,
        contextualizer=contextualizer,
        config={"contradiction_resolution": "flag", "auto_consolidate": True},
    )

    # Add the knowledge graph to the memory manager
    memory_manager.knowledge_graph = knowledge_graph

    # Create memory pruner
    pruner = MemoryPruner(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        config={"similarity_threshold": 0.9, "max_knowledge_units": 1000},
    )

    # Create agent
    agent = LongTermMemoryAgent(memory_manager=memory_manager)

    # Test data - demonstrating connections between concepts
    test_data = [
        "Python is a high-level programming language known for its readability.",
        "Python was created by Guido van Rossum in the late 1980s.",
        "Python 3 was released in 2008 as a major revision.",
        "Machine learning requires significant computational resources.",
        "GPUs are specialized hardware that excel at parallel computations.",
        "NVIDIA is a leading manufacturer of high-performance GPUs.",
        "TensorFlow is an open-source machine learning framework developed by Google.",
        "Keras is a high-level neural networks API that can run on top of other frameworks.",
        "In 2019, TensorFlow 2.0 integrated Keras as its official high-level API.",
        "Neural networks consist of layers of interconnected nodes or neurons.",
        "Deep learning refers to neural networks with many hidden layers.",
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data.",
        "Recurrent Neural Networks (RNNs) are designed to work with sequential data.",
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris, France.",
        "Paris hosts the Louvre Museum, home to the Mona Lisa painting.",
        "Leonardo da Vinci painted the Mona Lisa in the early 16th century.",
        "The Mona Lisa is one of the most famous paintings in the world.",
        "Shakespeare wrote Romeo and Juliet.",
        "Hamlet is considered one of Shakespeare's greatest works.",
    ]

    # Add deliberately redundant knowledge to demonstrate pruning
    redundant_data = [
        "Python programming language is known for its readability and simplicity.",
        "NVIDIA manufactures high-performance GPUs for computing.",
        "The Mona Lisa, painted by Leonardo da Vinci, is in the Louvre Museum.",
    ]

    # Add contradictory knowledge to demonstrate detection
    contradictory_data = [
        "Python was created by John Smith in the early 1990s.",
        "The capital of France is Lyon, not Paris.",
    ]

    # Combine all test data
    all_data = test_data + redundant_data + contradictory_data

    # Add knowledge in batches
    logger.info(f"Adding {len(all_data)} knowledge units...")
    start_time = time.time()

    for i, text in enumerate(all_data):
        # Create and add knowledge unit
        memory_manager.add_knowledge(
            content=text, source=f"test_data_{i}", metadata={"test_index": i}
        )

    add_time = time.time() - start_time
    logger.info(
        f"Added {len(all_data)} units in {add_time:.2f} seconds "
        f"({len(all_data) / add_time:.2f} units/sec)"
    )

    # Get knowledge graph stats
    logger.info("\n========== KNOWLEDGE GRAPH STATS ==========")
    graph_stats = memory_manager.get_knowledge_graph_stats()
    for key, value in graph_stats.items():
        logger.info(f"{key}: {value}")

    # Run some test queries to show connections
    logger.info("\n========== TESTING CONNECTIONS ==========")
    test_queries = [
        "Tell me about Python programming language.",
        "What is the relationship between TensorFlow and Keras?",
        "How do different neural network architectures relate to each other?",
        "What do you know about Paris and its landmarks?",
        "Tell me about Shakespeare's works.",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")

        # Get relevant knowledge
        start_time = time.time()
        units = memory_manager.retrieve_relevant(query, limit=3)
        query_time = time.time() - start_time

        logger.info(f"Found {len(units)} relevant units in {query_time:.4f} seconds")

        for i, unit in enumerate(units):
            logger.info(f"  {i + 1}. {unit.original_chunk[:100]}...")

            # Show connections for this unit
            connections = memory_manager.find_connections(unit.unique_id, max_connections=2)
            if connections:
                logger.info("    Connected to:")
                for conn in connections:
                    logger.info(f"    - {conn['content'][:50]}...")

    # Test contradiction detection
    logger.info("\n========== TESTING CONTRADICTION DETECTION ==========")
    contradictions = memory_manager.detect_contradictions(
        memory_manager.retrieve_relevant("Who created Python?")[0].unique_id
    )

    logger.info(f"Found {len(contradictions)} contradictions:")
    for i, contradiction in enumerate(contradictions):
        logger.info(f"  {i + 1}. {contradiction['content']}")
        logger.info(f"     Details: {contradiction['details']}")

    # Run memory pruning
    logger.info("\n========== TESTING MEMORY PRUNING ==========")
    start_time = time.time()
    pruning_stats = pruner.prune_memory()
    pruning_time = time.time() - start_time

    logger.info(f"Pruning completed in {pruning_time:.2f} seconds")
    logger.info(f"Items before pruning: {pruning_stats['total_size_before']}")
    logger.info(f"Items after pruning: {pruning_stats['total_size_after']}")
    logger.info(f"Redundant items pruned: {pruning_stats['redundant_pruned']}")
    logger.info(f"Low relevance items pruned: {pruning_stats['low_relevance_pruned']}")

    # Demonstrate agent responses after pruning
    logger.info("\n========== AGENT RESPONSES AFTER PRUNING ==========")
    for query in test_queries[:2]:  # Just test a couple
        logger.info(f"\nQuery: {query}")

        start_time = time.time()
        response = agent.generate_response(query)
        response_time = time.time() - start_time

        logger.info(f"Response generated in {response_time:.2f} seconds:")
        logger.info(f"{response}")

    logger.info("\nScalable memory demonstration completed successfully!")


if __name__ == "__main__":
    main()
