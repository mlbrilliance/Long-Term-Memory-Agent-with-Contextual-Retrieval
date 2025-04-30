import asyncio
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


async def demonstrate_memory_features():
    """Simple demonstration of the enhanced memory features."""
    print("=== Memory Features Demonstration ===\n")

    # Create a vector store and memory manager
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(memory_store=vector_store, contextualizer=SimpleContextualizer())

    # Initialize the memory manager
    await memory_manager.initialize()

    # 1. Add knowledge units on related topics
    print("1. Adding knowledge units on related topics...")

    ml_id1 = await memory_manager.add_knowledge(
        content="Machine learning enables systems to learn from data without explicit programming.",
        source="corpus",
        metadata={"domain": "AI", "topic": "machine learning"},
    )
    print(f"Added machine learning unit: {ml_id1}")

    py_id = await memory_manager.add_knowledge(
        content="Python is the most popular programming language for machine learning.",
        source="corpus",
        metadata={"domain": "programming", "topic": "Python"},
    )
    print(f"Added Python unit: {py_id}")

    # 2. Find related knowledge
    print("\n2. Finding related knowledge based on content...")
    query = "How is Python used in AI?"
    related_units = await memory_manager.get_related_knowledge(query, limit=5)

    print(f"Query: '{query}'")
    print(f"Found {len(related_units)} related knowledge units:")
    for unit, score in related_units:
        print(f"- {unit.original_chunk} (relevance: {score:.2f})")

    # 3. Demonstrate content merging with feedback
    print("\n3. Demonstrating content merging with feedback...")

    # First, let's retrieve the machine learning knowledge unit
    ml_unit = await memory_manager.retrieve_knowledge(ml_id1)
    print(f"Original content: {ml_unit.original_chunk}")

    # Now add feedback that extends this knowledge
    feedback = (
        "Machine learning includes supervised, unsupervised, and reinforcement learning approaches."
    )
    print(f"Adding feedback: {feedback}")

    # Process the feedback by adding it as new knowledge
    try:
        feedback_id = await memory_manager.add_knowledge(
            content=feedback, source="feedback", metadata={"enhances": ml_id1}
        )
        print(f"Added feedback: {feedback_id}")
    except Exception as e:
        print(f"Error adding feedback: {str(e)}")

    # Try to update the existing knowledge
    print("Updating original knowledge with feedback...")
    enhanced_content = f"{ml_unit.original_chunk} {feedback}"

    try:
        # Update the knowledge unit directly
        await memory_manager.update_knowledge(
            ml_id1, content=enhanced_content, metadata=ml_unit.metadata
        )
        print(f"Updated knowledge unit: {ml_id1}")

        # Retrieve the updated knowledge
        updated_unit = await memory_manager.retrieve_knowledge(ml_id1)
        print(f"Updated content: {updated_unit.original_chunk}")
    except Exception as e:
        print(f"Error updating knowledge: {str(e)}")
        # Fall back to adding as new knowledge
        try:
            new_id = await memory_manager.add_knowledge(
                content=enhanced_content, source="corpus", metadata=ml_unit.metadata
            )
            print(f"Added as new knowledge unit instead: {new_id}")
        except Exception as e2:
            print(f"Error adding new knowledge: {str(e2)}")

    # 4. Demonstrate knowledge retrieval with improved context formatting
    print("\n4. Demonstrating knowledge retrieval with improved context formatting...")
    new_query = "What are the different types of machine learning?"

    related_units = await memory_manager.get_related_knowledge(new_query, limit=5)
    print(f"Query: '{new_query}'")
    print(f"Found {len(related_units)} related knowledge units:")

    # Sort by relevance
    related_units.sort(key=lambda x: x[1], reverse=True)

    for unit, score in related_units:
        relevance_marker = (
            "HIGH RELEVANCE"
            if score > 0.7
            else "MEDIUM RELEVANCE" if score > 0.5 else "LOW RELEVANCE"
        )
        print(f"- [{relevance_marker}] {unit.original_chunk} (score: {score:.2f})")

    print("\n=== Demonstration Completed ===")


# Run the demonstration
if __name__ == "__main__":
    asyncio.run(demonstrate_memory_features())
