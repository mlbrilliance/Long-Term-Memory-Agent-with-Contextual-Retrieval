"""
Minimal demonstration of the Long-Term Memory Agent capabilities.
This script avoids problematic fields and focuses on successfully
demonstrating core memory features.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


async def run_basic_memory_demo():
    """Execute a basic demonstration of memory features."""
    print("=== Simplified Memory Features Demonstration ===\n")

    # Create memory components
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(memory_store=vector_store, contextualizer=SimpleContextualizer())

    # Initialize
    await memory_manager.initialize()

    # 1. Add some knowledge units
    print("1. Adding knowledge units...")

    ai_id = await memory_manager.add_knowledge(
        content="Artificial intelligence is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence.",
        source="corpus",
        metadata={"domain": "AI", "topic": "artificial intelligence"},
    )
    print(f"Added AI knowledge unit: {ai_id}")

    ml_id = await memory_manager.add_knowledge(
        content="Machine learning is a subset of AI that focuses on algorithms that can learn from and make predictions based on data.",
        source="corpus",
        metadata={"domain": "AI", "topic": "machine learning"},
    )
    print(f"Added ML knowledge unit: {ml_id}")

    # 2. Retrieve related knowledge
    print("\n2. Retrieving related knowledge...")
    query = "How do computers learn from data?"

    related_units = await memory_manager.get_related_knowledge(query, limit=5)
    print(f"Query: '{query}'")
    print(f"Found {len(related_units)} related knowledge units:")

    for i, (unit, score) in enumerate(related_units):
        print(f"{i + 1}. {unit.original_chunk} (relevance: {score:.2f})")

    # 3. Add new knowledge with metadata that references existing knowledge
    print("\n3. Adding knowledge with references to existing knowledge...")

    reinforcement_id = await memory_manager.add_knowledge(
        content="Reinforcement learning is a type of machine learning where agents learn to make decisions by receiving rewards or penalties.",
        source="corpus",
        metadata={
            "domain": "AI",
            "topic": "reinforcement learning",
            "related_to": [ml_id],  # Reference to machine learning
        },
    )
    print(f"Added RL knowledge unit with reference to ML: {reinforcement_id}")

    # 4. Add knowledge with more advanced metadata
    print("\n4. Adding knowledge with temporal and organizational metadata...")

    deep_learning_id = await memory_manager.add_knowledge(
        content="Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        source="corpus",
        metadata={
            "domain": "AI",
            "topic": "deep learning",
            "subtopic_of": ml_id,  # Deep learning is a subtopic of machine learning
            "importance": "high",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Added deep learning unit with advanced metadata: {deep_learning_id}")

    # 5. Demonstrate context-aware retrieval
    print("\n5. Demonstrating context-aware retrieval...")

    contextual_query = "neural networks in AI"

    related_units = await memory_manager.get_related_knowledge(contextual_query, limit=5)
    print(f"Contextual query: '{contextual_query}'")
    print(f"Found {len(related_units)} related knowledge units:")

    for i, (unit, score) in enumerate(related_units):
        print(f"{i + 1}. {unit.original_chunk} (relevance: {score:.2f})")
        # Display metadata to show the connections
        if unit.metadata:
            print(
                f"   - Metadata: {', '.join(f'{k}: {v}' for k, v in unit.metadata.items() if k != 'related_to')}"
            )

    print("\n=== Demonstration Completed ===")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "logs" / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run and capture output to a file
    output_file = output_dir / "memory_demo_output.txt"

    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_file, "w") as f:
        sys.stdout = f
        asyncio.run(run_basic_memory_demo())
        sys.stdout = original_stdout

    print(f"Demo completed. Output saved to: {output_file}")
    # Also print the output to console
    with open(output_file) as f:
        print(f.read())
