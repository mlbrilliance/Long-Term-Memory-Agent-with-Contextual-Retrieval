"""
Minimal demonstration of the enhanced memory system.

This script demonstrates:
1. Knowledge graph-based connections
2. Memory pruning capabilities
3. Multi-step reasoning
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import basic models first (these should be available)
from ltm_agent.core.models import KnowledgeUnit


class SimpleDemoGraph:
    """Simplified knowledge graph for demonstration purposes."""

    def __init__(self):
        """Initialize an empty graph."""
        self.nodes = {}  # id -> content
        self.edges = {}  # id -> list of connected ids

    def add_node(self, node_id: str, content: str):
        """Add a node to the graph."""
        self.nodes[node_id] = content
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, from_id: str, to_id: str):
        """Add an edge between nodes."""
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.edges:
            self.edges[to_id] = []

        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)

    def get_connected(self, node_id: str) -> list[str]:
        """Get IDs of nodes connected to the given node."""
        return self.edges.get(node_id, [])

    def find_path(self, from_id: str, to_id: str, max_depth: int = 3) -> list[list[str]]:
        """Find paths between two nodes."""
        paths = []

        def dfs(current: str, path: list[str], depth: int):
            if depth > max_depth:
                return
            if current == to_id:
                paths.append(path.copy())
                return

            for neighbor in self.edges.get(current, []):
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()

        dfs(from_id, [from_id], 1)
        return paths


class SimpleMemoryPruner:
    """Simple memory pruner for demonstration purposes."""

    def __init__(self):
        """Initialize the pruner."""
        self.redundancy_threshold = 0.8

    def is_redundant(self, content1: str, content2: str) -> bool:
        """Check if two content items are redundant."""
        # Simple redundancy check based on word overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))

        return similarity >= self.redundancy_threshold

    def find_redundant(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find redundant items in a list."""
        redundant = []

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if self.is_redundant(items[i]["content"], items[j]["content"]):
                    # Mark the later item as redundant
                    redundant.append(items[j])
                    break

        return redundant


def main():
    """Run the minimal scalable demo."""

    logger.info("Starting minimal scalable memory demo")

    # Test data demonstrating connections
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
    ]

    # Create knowledge units
    units = []

    for text in test_data:
        unit_id = str(uuid.uuid4())
        unit = KnowledgeUnit(
            unique_id=unit_id,
            original_chunk=text,
            processed_chunk=text,
            created_at=datetime.now().isoformat(),
            source="demo",
        )
        units.append(unit)

    logger.info(f"Created {len(units)} knowledge units")

    # Create a simple knowledge graph
    graph = SimpleDemoGraph()

    # Add nodes to graph
    for unit in units:
        graph.add_node(unit.unique_id, unit.original_chunk)

    # Create connections based on content similarity
    logger.info("Creating knowledge connections")
    connection_count = 0

    for i in range(len(units)):
        for j in range(len(units)):
            if i != j:
                # Connect related content
                if (
                    ("Python" in units[i].original_chunk and "Python" in units[j].original_chunk)
                    or (
                        "TensorFlow" in units[i].original_chunk
                        and "TensorFlow" in units[j].original_chunk
                    )
                    or ("Keras" in units[i].original_chunk and "Keras" in units[j].original_chunk)
                    or (
                        "neural" in units[i].original_chunk.lower()
                        and "neural" in units[j].original_chunk.lower()
                    )
                ):
                    graph.add_edge(units[i].unique_id, units[j].unique_id)
                    connection_count += 1

    logger.info(f"Created {connection_count} connections in the knowledge graph")

    # Demonstrate multi-hop reasoning
    logger.info("\n========== MULTI-HOP REASONING DEMO ==========")

    # Find paths between Python and TensorFlow
    python_unit = next((u for u in units if "Python" in u.original_chunk), None)
    tf_unit = next((u for u in units if "TensorFlow" in u.original_chunk), None)

    if python_unit and tf_unit:
        logger.info("Finding paths between Python and TensorFlow knowledge")
        paths = graph.find_path(python_unit.unique_id, tf_unit.unique_id)

        logger.info(f"Found {len(paths)} possible connection paths")

        if paths:
            logger.info("Example path of connections:")
            for path in paths[:1]:  # Show just the first path
                logger.info("  Path steps:")
                for node_id in path:
                    logger.info(f"  - {graph.nodes[node_id][:80]}...")

    # Demonstrate memory pruning
    logger.info("\n========== MEMORY PRUNING DEMO ==========")

    # Add redundant items
    redundant_data = [
        "Python is a programming language that is known for being highly readable.",
        "GPUs are hardware designed for parallel computation tasks.",
        "TensorFlow, developed by Google, is an open-source machine learning framework.",
    ]

    redundant_units = []
    for text in redundant_data:
        unit_id = str(uuid.uuid4())
        unit = KnowledgeUnit(
            unique_id=unit_id,
            original_chunk=text,
            processed_chunk=text,
            created_at=datetime.now().isoformat(),
            source="demo_redundant",
        )
        redundant_units.append(unit)
        units.append(unit)
        graph.add_node(unit.unique_id, unit.original_chunk)

    # Run pruning
    logger.info("Running memory pruning to detect redundancies")
    pruner = SimpleMemoryPruner()

    items_to_check = [{"id": unit.unique_id, "content": unit.original_chunk} for unit in units]

    redundant_items = pruner.find_redundant(items_to_check)

    logger.info(f"Found {len(redundant_items)} redundant items that could be pruned")
    for item in redundant_items:
        logger.info(f"  Redundant: {item['content'][:80]}...")

    # Demonstrate batched processing performance
    logger.info("\n========== BATCH PROCESSING DEMO ==========")

    # Generate a larger dataset
    batch_sizes = [1, 10, 50, 100]
    large_dataset = []

    for i in range(500):
        large_dataset.append(
            f"Test knowledge item {i}: This is a sample text for batch processing demonstration."
        )

    logger.info("Testing processing performance with different batch sizes")

    for batch_size in batch_sizes:
        start_time = time.time()

        # Process in batches
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i : i + batch_size]
            # Simulate processing (just measure time)
            _ = [text.lower() for text in batch]

        elapsed = time.time() - start_time
        items_per_second = len(large_dataset) / elapsed

        logger.info(
            f"Batch size {batch_size}: Processed {len(large_dataset)} items in {elapsed:.4f} seconds "
            f"({items_per_second:.2f} items/sec)"
        )

    logger.info("\nMinimal scalable memory demo completed successfully!")


if __name__ == "__main__":
    main()
