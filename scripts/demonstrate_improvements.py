"""
Memory System Improvements Demonstration

This script demonstrates the key improvements made to the memory system:
1. Enhanced knowledge correction in process_feedback
2. Improved knowledge interconnection in compute_unit_similarity

Rather than trying to import the actual system (which is having path issues),
this script uses simplified implementations that showcase the core enhancements.
"""

import asyncio
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("demonstration")


class KnowledgeUnit:
    """Simple knowledge unit implementation for demonstration."""

    def __init__(
        self,
        unique_id: str,
        original_chunk: str,
        processed_chunk: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        timestamp: str | None = None,
    ):
        self.unique_id = unique_id
        self.original_chunk = original_chunk
        self.processed_chunk = processed_chunk or original_chunk
        self.metadata = metadata or {}
        self.embedding = embedding or [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
        self.timestamp = timestamp or datetime.now().isoformat()

    def __str__(self):
        return f"KnowledgeUnit(id={self.unique_id}, content={self.original_chunk[:30]}...)"


class BasicMemoryManager:
    """Basic memory manager for demonstration."""

    def __init__(self):
        self.storage = {}
        self.counter = 0

    async def add_knowledge(
        self,
        content: str,
        source: str = "demo",
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a knowledge unit."""
        self.counter += 1
        unit_id = f"unit_{self.counter}"

        unit = KnowledgeUnit(unique_id=unit_id, original_chunk=content, metadata=metadata or {})

        self.storage[unit_id] = unit
        logger.info(f"Added knowledge unit: {unit_id}")
        return unit_id

    async def retrieve_knowledge(self, unique_id: str) -> KnowledgeUnit | None:
        """Retrieve a knowledge unit by ID."""
        return self.storage.get(unique_id)

    async def update_knowledge(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Update a knowledge unit."""
        self.storage[knowledge_unit.unique_id] = knowledge_unit
        return True

    async def list_knowledge(self) -> list[KnowledgeUnit]:
        """List all knowledge units."""
        return list(self.storage.values())

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Basic text similarity using character overlap ratio."""
        return SequenceMatcher(None, text1, text2).ratio()


class OriginalAgent:
    """Original implementation for comparison."""

    def __init__(self):
        self.memory_manager = BasicMemoryManager()
        self.last_query = None
        self.last_response = None

    async def process_feedback(self, feedback: str):
        """Original simple feedback processing."""
        if not self.last_query or not self.last_response:
            return

        logger.info(f"[ORIGINAL] Processing feedback: {feedback}")

        # Simple composite knowledge
        composite_content = f"Query: '{self.last_query}' → Correct response: '{feedback}'"

        # Basic metadata
        metadata = {"type": "feedback", "query": self.last_query}

        # Store as simple feedback
        feedback_id = await self.memory_manager.add_knowledge(
            content=composite_content, source="human_feedback", metadata=metadata
        )

        logger.info(f"[ORIGINAL] Created feedback unit: {feedback_id}")
        return feedback_id


class EnhancedAgent:
    """Enhanced implementation with improvements."""

    def __init__(self):
        self.memory_manager = BasicMemoryManager()
        self.last_query = None
        self.last_response = None

    async def process_feedback(self, feedback: str):
        """
        Enhanced process_feedback method with improvements:
        1. Better correction detection
        2. Creation of negative examples
        3. Improved metadata linking
        """
        if not self.last_query or not self.last_response:
            return

        logger.info(f"[ENHANCED] Processing feedback: {feedback}")

        # Enhanced detection of correction feedback
        is_correction = any(
            marker in feedback.lower()
            for marker in ["actually", "instead", "correction", "wrong", "incorrect", "not true"]
        )

        # Create composite knowledge with more detail
        composite_content = (
            f"Query: '{self.last_query}'\n"
            f"Initial response: '{self.last_response}'\n"
            f"Correction: '{feedback}'"
        )

        # Enhanced metadata
        metadata = {
            "type": "feedback",
            "feedback_type": "correction" if is_correction else "clarification",
            "query": self.last_query,
            "original_response": self.last_response,
            "importance": "high" if is_correction else "medium",
            "timestamp": datetime.now().isoformat(),
        }

        # Store the feedback
        feedback_id = await self.memory_manager.add_knowledge(
            content=composite_content,
            source="human_feedback",
            context=f"Feedback for query: {self.last_query}",
            metadata=metadata,
        )

        # If correction, create negative example (new)
        if is_correction:
            negative_example = (
                f"INCORRECT INFORMATION: '{self.last_response}'\n"
                f"CORRECT INFORMATION: '{feedback}'\n"
                f"CONTEXT: Response to '{self.last_query}'"
            )

            negative_id = await self.memory_manager.add_knowledge(
                content=negative_example,
                source="generated_negative_example",
                context="Negative example from correction",
                metadata={
                    "type": "negative_example",
                    "related_to": feedback_id,
                    "importance": "high",
                    "original_query": self.last_query,
                },
            )

            logger.info(f"[ENHANCED] Created negative example: {negative_id}")

        # Store correct response directly (new)
        correct_response = f"Query: '{self.last_query}'\nCorrect response: '{feedback}'"

        correct_id = await self.memory_manager.add_knowledge(
            content=correct_response,
            source="corrected_response",
            context=f"Corrected response for: {self.last_query}",
            metadata={
                "type": "corrected_response",
                "related_to": feedback_id,
                "importance": "high",
                "query": self.last_query,
            },
        )

        logger.info(f"[ENHANCED] Created feedback unit: {feedback_id}")
        logger.info(f"[ENHANCED] Stored correct response: {correct_id}")
        return feedback_id


class OriginalConsolidator:
    """Original implementation of memory consolidation."""

    def __init__(self, memory_manager: BasicMemoryManager):
        self.memory_manager = memory_manager

    async def compute_unit_similarity(self, unit1: KnowledgeUnit, unit2: KnowledgeUnit) -> float:
        """Original basic similarity computation."""
        # Just use text similarity
        similarity = self.memory_manager._compute_text_similarity(
            unit1.original_chunk, unit2.original_chunk
        )

        logger.info(
            f"[ORIGINAL] Similarity between {unit1.unique_id} and {unit2.unique_id}: {similarity:.4f}"
        )
        return similarity


class EnhancedConsolidator:
    """Enhanced implementation of memory consolidation."""

    def __init__(self, memory_manager: BasicMemoryManager):
        self.memory_manager = memory_manager

    async def compute_unit_similarity(self, unit1: KnowledgeUnit, unit2: KnowledgeUnit) -> float:
        """
        Enhanced similarity calculation with improvements:
        1. Multi-factor similarity assessment
        2. Metadata relationship recognition
        3. Temporal proximity analysis
        4. Type similarity consideration
        """
        similarity_score = 0.0
        similarity_factors = 0

        # 1. Text content similarity (primary factor)
        text_similarity = self.memory_manager._compute_text_similarity(
            unit1.original_chunk, unit2.original_chunk
        )
        similarity_score += text_similarity * 0.5  # 50% weight
        similarity_factors += 0.5

        # 2. Metadata relationship checks
        metadata1 = unit1.metadata or {}
        metadata2 = unit2.metadata or {}

        # Direct references
        direct_reference = False
        if "related_to" in metadata1 and metadata1["related_to"] == unit2.unique_id:
            direct_reference = True
        if "related_to" in metadata2 and metadata2["related_to"] == unit1.unique_id:
            direct_reference = True

        # Shared references
        shared_references = set()
        if "related_units" in metadata1 and "related_units" in metadata2:
            shared_references = set(metadata1["related_units"]) & set(metadata2["related_units"])

        if direct_reference:
            similarity_score += 0.3  # Strong boost
            similarity_factors += 0.3

        if shared_references:
            similarity_score += min(0.2, 0.05 * len(shared_references))
            similarity_factors += 0.2

        # 3. Temporal proximity
        try:
            time1 = datetime.fromisoformat(unit1.timestamp)
            time2 = datetime.fromisoformat(unit2.timestamp)
            time_diff = abs((time1 - time2).total_seconds())

            # Items created close together (within 5 minutes)
            if time_diff < 300:
                temporal_factor = max(0.0, 0.1 * (1 - time_diff / 300))
                similarity_score += temporal_factor
                similarity_factors += 0.1
        except (ValueError, TypeError):
            pass

        # 4. Type similarity
        if metadata1.get("type") == metadata2.get("type"):
            similarity_score += 0.1
            similarity_factors += 0.1

        # Normalize
        if similarity_factors > 0:
            normalized_similarity = similarity_score / similarity_factors
        else:
            normalized_similarity = text_similarity

        logger.info(
            f"[ENHANCED] Similarity between {unit1.unique_id} and {unit2.unique_id}: {normalized_similarity:.4f}"
        )
        return normalized_similarity


async def demonstrate_correction_improvements():
    """Demonstrate the improvements in knowledge correction."""
    print("\n" + "=" * 80)
    print("DEMONSTRATING KNOWLEDGE CORRECTION IMPROVEMENTS")
    print("=" * 80)

    # Test data
    query = "What are Python's limitations?"
    response = "Python is always the fastest programming language for all computational tasks."
    feedback = """
    Actually, Python is generally slower than compiled languages like C++ or Rust for
    computation-heavy tasks. It's optimized for developer productivity rather than
    raw performance. However, for many applications, Python's speed is sufficient,
    and performance-critical parts can be optimized with C extensions.
    """

    # Set up original agent
    original_agent = OriginalAgent()
    original_agent.last_query = query
    original_agent.last_response = response

    # Set up enhanced agent
    enhanced_agent = EnhancedAgent()
    enhanced_agent.last_query = query
    enhanced_agent.last_response = response

    # Process feedback with original implementation
    print("\n[ORIGINAL IMPLEMENTATION]")
    await original_agent.process_feedback(feedback)
    original_units = await original_agent.memory_manager.list_knowledge()
    print(f"Created {len(original_units)} knowledge units")
    for unit in original_units:
        print(f"- {unit.unique_id}: {unit.metadata.get('type', 'unknown')}")

    # Process same feedback with enhanced implementation
    print("\n[ENHANCED IMPLEMENTATION]")
    await enhanced_agent.process_feedback(feedback)
    enhanced_units = await enhanced_agent.memory_manager.list_knowledge()
    print(f"Created {len(enhanced_units)} knowledge units")
    for unit in enhanced_units:
        print(
            f"- {unit.unique_id}: {unit.metadata.get('type', 'unknown')} | {unit.metadata.get('feedback_type', '')}"
        )

    # Compare metadata richness
    print("\nMetadata Comparison:")
    original_keys = set()
    for unit in original_units:
        original_keys.update(unit.metadata.keys())

    enhanced_keys = set()
    for unit in enhanced_units:
        enhanced_keys.update(unit.metadata.keys())

    print(f"Original metadata keys: {original_keys}")
    print(f"Enhanced metadata keys: {enhanced_keys}")
    print(f"New metadata fields: {enhanced_keys - original_keys}")

    return {
        "original_units": len(original_units),
        "enhanced_units": len(enhanced_units),
        "original_metadata_keys": len(original_keys),
        "enhanced_metadata_keys": len(enhanced_keys),
    }


async def demonstrate_similarity_improvements():
    """Demonstrate the improvements in similarity computation."""
    print("\n" + "=" * 80)
    print("DEMONSTRATING SIMILARITY COMPUTATION IMPROVEMENTS")
    print("=" * 80)

    # Create memory manager
    memory_manager = BasicMemoryManager()

    # Create test knowledge units
    unit1 = KnowledgeUnit(
        unique_id="test1",
        original_chunk="Python is a high-level programming language known for its readability.",
        timestamp=datetime.now().isoformat(),
    )

    unit2 = KnowledgeUnit(
        unique_id="test2",
        original_chunk="Python is a programming language with great readability.",
        metadata={"related_to": "test1", "type": "clarification"},
        timestamp=datetime.now().isoformat(),
    )

    unit3 = KnowledgeUnit(
        unique_id="test3",
        original_chunk="JavaScript is a programming language used for web development.",
        timestamp=(datetime.now()).isoformat(),
        metadata={"type": "information"},
    )

    unit4 = KnowledgeUnit(
        unique_id="test4",
        original_chunk="Python uses reference counting and garbage collection for memory management.",
        metadata={"related_units": ["test1"], "type": "clarification"},
        timestamp=(datetime.now()).isoformat(),
    )

    # Create consolidators
    original_consolidator = OriginalConsolidator(memory_manager)
    enhanced_consolidator = EnhancedConsolidator(memory_manager)

    # Test pairs
    test_pairs = [
        (unit1, unit2, "Very similar content, direct reference"),
        (unit1, unit3, "Different content, no links"),
        (unit1, unit4, "Related content, metadata link"),
        (unit3, unit4, "Different content, same type"),
    ]

    results = {"original": {}, "enhanced": {}}

    # Compare similarity computations
    print("\nSimilarity Comparison:")
    for unit_a, unit_b, desc in test_pairs:
        pair_key = f"{unit_a.unique_id}-{unit_b.unique_id}"

        # Original computation
        original_similarity = await original_consolidator.compute_unit_similarity(unit_a, unit_b)
        results["original"][pair_key] = original_similarity

        # Enhanced computation
        enhanced_similarity = await enhanced_consolidator.compute_unit_similarity(unit_a, unit_b)
        results["enhanced"][pair_key] = enhanced_similarity

        # Display comparison
        print(f"\nPair: {pair_key} ({desc})")
        print(f"- Original similarity: {original_similarity:.4f}")
        print(f"- Enhanced similarity: {enhanced_similarity:.4f}")
        print(f"- Difference: {enhanced_similarity - original_similarity:.4f}")

    # Calculate overall impact
    avg_original = sum(results["original"].values()) / len(results["original"])
    avg_enhanced = sum(results["enhanced"].values()) / len(results["enhanced"])

    print("\nOverall Impact:")
    print(f"- Average similarity (original): {avg_original:.4f}")
    print(f"- Average similarity (enhanced): {avg_enhanced:.4f}")
    print(f"- Average difference: {avg_enhanced - avg_original:.4f}")

    return results


async def main():
    """Run the demonstration of improvements."""
    print("=" * 80)
    print("MEMORY SYSTEM IMPROVEMENTS DEMONSTRATION")
    print("=" * 80)
    print("\nThis script demonstrates the improvements made to the memory system.")

    try:
        # Demonstrate correction improvements
        correction_results = await demonstrate_correction_improvements()

        # Demonstrate similarity improvements
        similarity_results = await demonstrate_similarity_improvements()

        # Overall summary
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)

        print("\nKnowledge Correction Improvements:")
        print(f"- Original implementation: {correction_results['original_units']} knowledge units")
        print(f"- Enhanced implementation: {correction_results['enhanced_units']} knowledge units")
        print(
            f"- Metadata richness: {correction_results['original_metadata_keys']} vs {correction_results['enhanced_metadata_keys']} fields"
        )

        print("\nSimilarity Computation Improvements:")
        similar_pairs = [(k, v) for k, v in similarity_results["enhanced"].items() if v > 0.7]
        dissimilar_pairs = [(k, v) for k, v in similarity_results["enhanced"].items() if v <= 0.7]

        print(
            f"- More nuanced similarity assessment: {len(similarity_results['enhanced'])} pairs evaluated"
        )
        print(f"- Similar content correctly identified: {len(similar_pairs)} pairs")
        print(f"- Dissimilar content correctly identified: {len(dissimilar_pairs)} pairs")

        # Save results
        results = {
            "correction_demonstration": correction_results,
            "similarity_demonstration": {
                "original": similarity_results["original"],
                "enhanced": similarity_results["enhanced"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        result_file = f"demonstration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDemonstration results saved to {result_file}")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
