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


async def demonstrate_memory_consolidation():
    """Simple demonstration of memory management and consolidation."""
    print("=== Memory Consolidation Demonstration ===\n")

    # Create a vector store and memory manager
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(memory_store=vector_store, contextualizer=SimpleContextualizer())

    # Initialize the memory manager
    await memory_manager.initialize()

    # Add similar knowledge units to demonstrate consolidation
    print("Adding sample knowledge units...")

    ml_id1 = await memory_manager.add_knowledge(
        content="Machine learning enables computers to learn from data and improve over time.",
        source="corpus",
        metadata={"domain": "AI", "importance": "high"},
    )
    print(f"Added machine learning unit 1: {ml_id1}")

    ml_id2 = await memory_manager.add_knowledge(
        content="Machine learning algorithms can learn from and make predictions on data.",
        source="corpus",
        metadata={"domain": "AI", "application": "prediction"},
    )
    print(f"Added machine learning unit 2: {ml_id2}")

    py_id = await memory_manager.add_knowledge(
        content="Python is widely used for implementing machine learning algorithms.",
        source="corpus",
        metadata={"domain": "programming", "related_to": "machine learning"},
    )
    print(f"Added Python unit: {py_id}")

    # List all knowledge units
    units = await memory_manager.list_knowledge()
    print(f"\nInitial knowledge base has {len(units)} units")

    # Retrieve the units to check them
    ml_unit1 = await memory_manager.retrieve_knowledge(ml_id1)
    ml_unit2 = await memory_manager.retrieve_knowledge(ml_id2)
    py_unit = await memory_manager.retrieve_knowledge(py_id)

    print("\n--- Initial Knowledge State ---")
    print(f"ML Unit 1: {ml_unit1.original_chunk}")
    print(f"ML Unit 2: {ml_unit2.original_chunk}")
    print(f"Python Unit: {py_unit.original_chunk}")

    # Create a reference between the units
    print("\nCreating cross-references between units...")

    # Update ML unit 1 to reference ML unit 2
    ml1_metadata = ml_unit1.metadata or {}
    if "related_units" not in ml1_metadata:
        ml1_metadata["related_units"] = []
    ml1_metadata["related_units"].append(ml_id2)
    await memory_manager.update_knowledge(ml_id1, metadata=ml1_metadata)

    # Update ML unit 2 to reference ML unit 1 and Python unit
    ml2_metadata = ml_unit2.metadata or {}
    if "related_units" not in ml2_metadata:
        ml2_metadata["related_units"] = []
    ml2_metadata["related_units"].append(ml_id1)
    ml2_metadata["related_units"].append(py_id)
    await memory_manager.update_knowledge(ml_id2, metadata=ml2_metadata)

    # Update Python unit to reference ML unit 2
    py_metadata = py_unit.metadata or {}
    if "related_units" not in py_metadata:
        py_metadata["related_units"] = []
    py_metadata["related_units"].append(ml_id2)
    await memory_manager.update_knowledge(py_id, metadata=py_metadata)

    # Verify the references were created
    ml_unit1 = await memory_manager.retrieve_knowledge(ml_id1)
    ml_unit2 = await memory_manager.retrieve_knowledge(ml_id2)
    py_unit = await memory_manager.retrieve_knowledge(py_id)

    print("\n--- Knowledge State After Cross-Referencing ---")
    print(f"ML Unit 1 references: {ml_unit1.metadata.get('related_units', [])}")
    print(f"ML Unit 2 references: {ml_unit2.metadata.get('related_units', [])}")
    print(f"Python Unit references: {py_unit.metadata.get('related_units', [])}")

    # Add feedback that enhances the machine learning knowledge
    print("\nSimulating feedback that enhances knowledge...")
    feedback_id = await memory_manager.add_knowledge(
        content="Machine learning systems improve through both supervised and unsupervised learning approaches.",
        source="feedback",
        metadata={"enhances": ml_id1, "importance": "high"},
    )
    print(f"Added feedback: {feedback_id}")

    # Merge the feedback with the existing knowledge
    print("\nMerging feedback with existing knowledge...")
    ml_unit1 = await memory_manager.retrieve_knowledge(ml_id1)
    feedback = await memory_manager.retrieve_knowledge(feedback_id)

    # Manually merge the content
    merged_content = f"{ml_unit1.original_chunk} {feedback.original_chunk}"

    # Update the ML unit 1 with the merged content
    ml1_metadata = ml_unit1.metadata or {}
    if "update_history" not in ml1_metadata:
        ml1_metadata["update_history"] = []

    ml1_metadata["update_history"].append(
        {
            "previous_content": ml_unit1.original_chunk,
            "update_source": feedback_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    await memory_manager.update_knowledge(ml_id1, content=merged_content, metadata=ml1_metadata)

    # Verify the update
    updated_ml_unit1 = await memory_manager.retrieve_knowledge(ml_id1)

    print("\n--- Final Knowledge State After Update ---")
    print(f"Updated ML Unit 1: {updated_ml_unit1.original_chunk}")
    print(f"Update history: {updated_ml_unit1.metadata.get('update_history', [])}")
    print(f"References preserved: {updated_ml_unit1.metadata.get('related_units', [])}")

    # Demonstrate finding related knowledge
    print("\nDemonstrating knowledge retrieval with cross-references...")
    query = "How is Python used in machine learning?"

    related_units = await memory_manager.get_related_knowledge(query, limit=5)
    print(f"\nQuery: {query}")
    print(f"Found {len(related_units)} related knowledge units:")

    for unit, score in related_units:
        print(f"- {unit.original_chunk} (relevance: {score:.2f})")

        # Show cross-references
        references = unit.metadata.get("related_units", [])
        if references:
            print(f"  Has {len(references)} references to other knowledge units")

    print("\n=== Demonstration Completed ===")


# Run the demonstration
if __name__ == "__main__":
    asyncio.run(demonstrate_memory_consolidation())
