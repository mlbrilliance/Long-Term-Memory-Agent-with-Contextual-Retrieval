import asyncio
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from ltm_agent.memory.consolidator import MemoryConsolidator
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager


async def test_memory_improvements():
    """Test the memory consolidation improvements with similar content."""
    print("Testing Knowledge Cross-Referencer and Memory Consolidator")

    # Create a vector store and memory manager
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(memory_store=vector_store, contextualizer=SimpleContextualizer())

    # Initialize the memory manager
    await memory_manager.initialize()

    # Create sample knowledge units with similar content
    print("\nAdding sample knowledge units with similar content...")

    # First pair of similar units about machine learning
    unit1_id = await memory_manager.add_knowledge(
        content="Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
        source="corpus",
        metadata={"type": "technology", "field": "AI"},
    )
    print(f"Added unit 1: {unit1_id}")

    unit2_id = await memory_manager.add_knowledge(
        content="Machine learning is a field of AI that enables systems to learn from data without explicit programming.",
        source="corpus",
        metadata={"type": "technology", "field": "AI"},
    )
    print(f"Added unit 2 (similar to unit 1): {unit2_id}")

    # Second pair of similar units about Python
    unit3_id = await memory_manager.add_knowledge(
        content="Python is a high-level programming language known for its readability and simplicity.",
        source="corpus",
        metadata={"type": "technology", "field": "programming"},
    )
    print(f"Added unit 3: {unit3_id}")

    unit4_id = await memory_manager.add_knowledge(
        content="Python programming language is widely used due to its simplicity and readable syntax.",
        source="corpus",
        metadata={"type": "technology", "field": "programming"},
    )
    print(f"Added unit 4 (similar to unit 3): {unit4_id}")

    # Cross-domain unit linking ML and Python
    unit5_id = await memory_manager.add_knowledge(
        content="Python is the most popular language for implementing machine learning algorithms.",
        source="corpus",
        metadata={"type": "technology", "field": "programming"},
    )
    print(f"Added unit 5 (links Python and ML): {unit5_id}")

    print("\nManually creating cross-references between similar units...")

    # Directly modify metadata to create cross-references (to avoid async issues)
    ml_unit1 = await memory_manager.retrieve_knowledge(unit1_id)
    ml_unit2 = await memory_manager.retrieve_knowledge(unit2_id)

    # Add cross-references between machine learning units
    if not ml_unit1.metadata:
        ml_unit1.metadata = {}
    if "related_units" not in ml_unit1.metadata:
        ml_unit1.metadata["related_units"] = []
    ml_unit1.metadata["related_units"].append(unit2_id)

    if not ml_unit2.metadata:
        ml_unit2.metadata = {}
    if "related_units" not in ml_unit2.metadata:
        ml_unit2.metadata["related_units"] = []
    ml_unit2.metadata["related_units"].append(unit1_id)

    # Update units with cross-references
    await memory_manager.update_knowledge(unit1_id, metadata=ml_unit1.metadata)
    await memory_manager.update_knowledge(unit2_id, metadata=ml_unit2.metadata)

    # Do the same for Python units
    py_unit1 = await memory_manager.retrieve_knowledge(unit3_id)
    py_unit2 = await memory_manager.retrieve_knowledge(unit4_id)
    connector = await memory_manager.retrieve_knowledge(unit5_id)

    # Link Python units
    if not py_unit1.metadata:
        py_unit1.metadata = {}
    if "related_units" not in py_unit1.metadata:
        py_unit1.metadata["related_units"] = []
    py_unit1.metadata["related_units"].append(unit4_id)
    py_unit1.metadata["related_units"].append(unit5_id)  # Link to connector

    if not py_unit2.metadata:
        py_unit2.metadata = {}
    if "related_units" not in py_unit2.metadata:
        py_unit2.metadata["related_units"] = []
    py_unit2.metadata["related_units"].append(unit3_id)
    py_unit2.metadata["related_units"].append(unit5_id)  # Link to connector

    # Link connector to both domains
    if not connector.metadata:
        connector.metadata = {}
    if "related_units" not in connector.metadata:
        connector.metadata["related_units"] = []
    connector.metadata["related_units"].append(unit1_id)  # Link to ML
    connector.metadata["related_units"].append(unit3_id)  # Link to Python

    # Update units with cross-references
    await memory_manager.update_knowledge(unit3_id, metadata=py_unit1.metadata)
    await memory_manager.update_knowledge(unit4_id, metadata=py_unit2.metadata)
    await memory_manager.update_knowledge(unit5_id, metadata=connector.metadata)

    # Check references
    print("\nVerifying cross-references were added:")
    ml_unit1 = await memory_manager.retrieve_knowledge(unit1_id)
    py_unit1 = await memory_manager.retrieve_knowledge(unit3_id)
    connector = await memory_manager.retrieve_knowledge(unit5_id)

    print(f"Machine Learning unit references: {ml_unit1.metadata.get('related_units', [])}")
    print(f"Python unit references: {py_unit1.metadata.get('related_units', [])}")
    print(f"Connector unit references: {connector.metadata.get('related_units', [])}")

    # Initialize and run the memory consolidator
    print("\nRunning memory consolidation...")
    try:
        # Create a consolidator
        consolidator = MemoryConsolidator(memory_manager)

        # Run consolidation
        await consolidator.consolidate_memory()

        # Check for changes after consolidation
        final_units = await memory_manager.list_knowledge()
        print(f"\nTotal knowledge units after consolidation: {len(final_units)}")

        # Print all units after consolidation
        print("\nKnowledge units after consolidation:")
        for i, unit in enumerate(final_units):
            print(f"\nUnit {i + 1}:")
            print(f"ID: {unit.unique_id}")
            print(f"Content: {unit.original_chunk}")
            print(f"References: {unit.metadata.get('related_units', [])}")

            # Print relationship details if they exist
            relationships = unit.metadata.get("relationships", {})
            if relationships:
                print("Relationships:")
                for rel_id, rel_info in relationships.items():
                    sim = rel_info.get("similarity", "N/A")
                    if not isinstance(sim, str):
                        sim = f"{sim:.2f}"
                    print(f"  - {rel_id}: similarity={sim}")

            # Show domain information if it exists
            domains = unit.metadata.get("domain_links", [])
            if domains:
                print(f"Domains: {domains}")
    except Exception as e:
        print(f"Error during consolidation: {str(e)}\n")
        import traceback

        traceback.print_exc()

    print("\nTest completed.")


# Run the async test
if __name__ == "__main__":
    asyncio.run(test_memory_improvements())
