#!/usr/bin/env python
"""
Interactive Demo for Enhanced Memory System

This script provides an interactive demo where you can test the key features
of the enhanced memory system:
1. Progressive learning across multiple interactions
2. Knowledge correction and updating
3. Cross-referencing between related information

Usage:
    python scripts/interactive_demo.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("interactive_demo.log")],
)
logger = logging.getLogger("interactive_demo")

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


# Import simplified components for demo
class KnowledgeUnit:
    """Simple knowledge unit for demonstration."""

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


class MemoryStore:
    """Simple memory store for demonstration."""

    def __init__(self):
        self.storage = {}
        self.counter = 0

    async def add(self, knowledge_unit: KnowledgeUnit) -> str:
        """Add a knowledge unit."""
        self.storage[knowledge_unit.unique_id] = knowledge_unit
        return knowledge_unit.unique_id

    async def get(self, unique_id: str) -> KnowledgeUnit | None:
        """Get a knowledge unit by ID."""
        return self.storage.get(unique_id)

    async def update(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Update a knowledge unit."""
        self.storage[knowledge_unit.unique_id] = knowledge_unit
        return True

    async def search(self, query: str, k: int = 5) -> list[tuple]:
        """Search for knowledge units by query."""
        results = []
        for unit in self.storage.values():
            # Simple text similarity
            query_words = set(query.lower().split())
            content_words = set(unit.original_chunk.lower().split())
            common_words = query_words.intersection(content_words)

            if common_words:  # If there are common words
                score = len(common_words) / max(len(query_words), 1)

                # Boost correction units
                if (
                    unit.metadata.get("type") == "correction"
                    or unit.metadata.get("feedback_type") == "correction"
                ):
                    score *= 1.5

                results.append((unit, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    async def list(self) -> list[KnowledgeUnit]:
        """List all knowledge units."""
        return list(self.storage.values())


class SimpleMemoryManager:
    """Simple memory manager for demonstration."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
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
        unique_id = f"unit_{self.counter}"

        unit = KnowledgeUnit(unique_id=unique_id, original_chunk=content, metadata=metadata or {})

        await self.memory_store.add(unit)
        logger.info(f"Added knowledge unit: {unique_id}")
        return unique_id

    async def retrieve_knowledge(self, unique_id: str) -> KnowledgeUnit | None:
        """Retrieve a knowledge unit by ID."""
        return await self.memory_store.get(unique_id)

    async def get_related_knowledge(self, query: str, k: int = 5) -> list[tuple]:
        """Get knowledge related to a query."""
        results = await self.memory_store.search(query, k)

        # Print retrieval for debugging
        logger.info(f"Retrieved {len(results)} units for query: {query}")
        for unit, score in results:
            unit_type = unit.metadata.get("type", "standard")
            logger.info(f"- {unit.unique_id} ({unit_type}): {score:.2f}")

        return results

    async def list_knowledge(self) -> list[KnowledgeUnit]:
        """List all knowledge units."""
        return await self.memory_store.list()

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common_words = words1.intersection(words2)

        if not words1 or not words2:
            return 0.0

        return len(common_words) / max(len(words1), len(words2))


class InteractiveAgent:
    """Interactive agent for demonstration."""

    def __init__(self):
        self.memory_store = MemoryStore()
        self.memory_manager = SimpleMemoryManager(self.memory_store)
        self.last_query = None
        self.last_response = None
        self.debug_mode = False

    async def process_query(self, query: str) -> str:
        """Process a user query."""
        self.last_query = query
        logger.info(f"Processing query: {query}")

        # Get related knowledge
        knowledge_units = await self.memory_manager.get_related_knowledge(query, k=5)

        # Generate response based on knowledge
        response = await self._generate_response(query, knowledge_units)
        self.last_response = response

        # Store the interaction
        await self.memory_manager.add_knowledge(
            content=f"Query: {query} → Response: {response}",
            source="interaction",
            metadata={
                "type": "interaction",
                "query": query,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return response

    async def _generate_response(self, query: str, knowledge_units: list[tuple]) -> str:
        """Generate a response based on knowledge units."""
        if not knowledge_units:
            return "I don't have specific information about that query yet."

        logger.info(f"Generating response from {len(knowledge_units)} knowledge units")

        # Check if we have correction units
        correction_units = [
            unit
            for unit, _ in knowledge_units
            if unit.metadata.get("type") == "correction"
            or unit.metadata.get("feedback_type") == "correction"
            or unit.metadata.get("type") == "corrected_response"
        ]

        # First priority: use corrections if available
        if correction_units:
            # Check if correction is relevant to the query
            for correction in correction_units:
                original_query = correction.metadata.get("query", "")
                if (
                    original_query
                    and self.memory_manager.compute_text_similarity(query, original_query) > 0.5
                ):
                    logger.info(f"Using relevant correction unit: {correction.unique_id}")
                    return self._extract_corrected_content(correction.original_chunk)

        # Collect information from knowledge units based on their types and relevance
        direct_knowledge = []
        relevant_interaction_responses = []

        # First, compute the relevance of each unit to the query
        for unit, score in knowledge_units:
            unit_type = unit.metadata.get("type", "")
            content = unit.original_chunk

            # Calculate direct relevance of the content to the query
            content_relevance = self.memory_manager.compute_text_similarity(query, content)
            logger.info(f"Unit {unit.unique_id} relevance: {content_relevance:.2f}")

            # Direct knowledge (facts, concepts, etc.)
            if unit_type in ["fact", "concept", "rule", "direct_input"]:
                # Skip question-like knowledge for question queries
                if unit.metadata.get("is_question_like", False) and (
                    query.endswith("?")
                    or any(
                        query.lower().startswith(q)
                        for q in ["what", "how", "why", "when", "where", "who", "which"]
                    )
                ):
                    logger.info(
                        f"Skipping question-like knowledge for question query: {unit.unique_id}"
                    )
                    continue

                # Only include if the content is relevant to the query
                if content_relevance > 0.3:  # Threshold for relevance
                    logger.info(
                        f"Found relevant direct knowledge: {unit.unique_id} (score: {score:.2f}, relevance: {content_relevance:.2f})"
                    )

                    # Process the knowledge to extract the actual information
                    processed_content = self._process_knowledge_content(content, query)
                    direct_knowledge.append((processed_content, score * content_relevance))

            # Extract previous answers from interaction units
            elif unit_type == "interaction" and "Response:" in unit.original_chunk:
                query_part = ""
                response_part = ""

                if "Query:" in unit.original_chunk and "Response:" in unit.original_chunk:
                    # Split into query and response
                    parts = unit.original_chunk.split("Response:")
                    query_part = parts[0].replace("Query:", "").strip().strip("'\"")
                    response_part = parts[1].strip().strip("'\"")

                    # Skip unhelpful default responses
                    if "don't have specific information" in response_part:
                        continue

                    # Only include if the previous query is similar to current query
                    similarity = self.memory_manager.compute_text_similarity(query, query_part)
                    if similarity > 0.5:  # Threshold for considering it relevant
                        logger.info(
                            f"Found relevant interaction: {unit.unique_id} (similarity: {similarity:.2f})"
                        )
                        relevant_interaction_responses.append((response_part, similarity * score))

        # Second priority: use relevant direct knowledge if available
        if direct_knowledge:
            # Sort by score and use the most relevant knowledge
            direct_knowledge.sort(key=lambda x: x[1], reverse=True)
            best_knowledge, best_score = direct_knowledge[0]

            # Only use if it's truly relevant
            if best_score > 0.25:
                logger.info(
                    f"Using direct knowledge with score {best_score:.2f}: {best_knowledge[:50]}..."
                )
                return best_knowledge

        # Third priority: use relevant interaction responses
        if relevant_interaction_responses:
            # Sort by score and use the most relevant response
            relevant_interaction_responses.sort(key=lambda x: x[1], reverse=True)
            best_response, best_score = relevant_interaction_responses[0]

            # Only use if it's truly relevant
            if best_score > 0.25:
                logger.info(
                    f"Using relevant interaction response with score {best_score:.2f}: {best_response[:50]}..."
                )
                return best_response

        # If we reach here, we don't have specific relevant knowledge
        return "I don't have specific information about that query yet."

    def _process_knowledge_content(self, content: str, query: str) -> str:
        """
        Process knowledge content to extract useful information.
        This handles cases where the content might be a question
        or might need to be reformatted to be more informative.
        """
        # Remove question phrasing
        content = content.strip()

        # Handle direct statements (already in the right form)
        if not any(
            content.lower().startswith(q)
            for q in ["what", "how", "why", "when", "where", "who", "which"]
        ) and not content.endswith("?"):
            return content

        # Handle "what is X" style content when matched with "what is X" style queries
        if content.lower().startswith("what is") and query.lower().startswith("what is"):
            # Extract the subject
            subject = content[8:].strip().rstrip("?")

            # Check if we have additional knowledge about this subject
            # This is a simple placeholder - in a real system, you'd do a proper lookup
            return f"I know the question about {subject}, but I don't have specific details beyond that."

        # For other question-like knowledge, make it clear it's a placeholder
        return f"This is related to your query, but I need more information to provide details about: {content}"

    def _extract_corrected_content(self, content: str) -> str:
        """Extract corrected content from a correction unit."""
        if "Correct response:" in content:
            parts = content.split("Correct response:")
            if len(parts) > 1:
                return parts[1].strip().strip("'\"")

        if "CORRECT INFORMATION:" in content:
            parts = content.split("CORRECT INFORMATION:")
            if len(parts) > 1:
                return parts[1].strip().strip("'\"")

        if "Correction:" in content:
            parts = content.split("Correction:")
            if len(parts) > 1:
                return parts[1].strip().strip("'\"")

        return content

    async def process_feedback(self, feedback: str):
        """
        Process feedback to update knowledge.

        This implements the enhanced correction detection and handling.
        """
        if not self.last_query or not self.last_response:
            print("No previous interaction to correct.")
            return

        logger.info(f"Processing feedback: {feedback}")

        # Enhanced detection of correction feedback
        is_correction = any(
            marker in feedback.lower()
            for marker in [
                "actually",
                "instead",
                "correction",
                "wrong",
                "incorrect",
                "not true",
                "inaccurate",
            ]
        )

        # Create composite knowledge
        composite_content = (
            f"Query: '{self.last_query}'\n"
            f"Initial response: '{self.last_response}'\n"
            f"{'Correction' if is_correction else 'Feedback'}: '{feedback}'"
        )

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
            content=composite_content, source="human_feedback", metadata=metadata
        )

        # If this is a correction, create a negative example
        if is_correction:
            # Create negative example
            negative_example = (
                f"INCORRECT INFORMATION: '{self.last_response}'\n"
                f"CORRECT INFORMATION: '{feedback}'\n"
                f"CONTEXT: Response to '{self.last_query}'"
            )

            neg_id = await self.memory_manager.add_knowledge(
                content=negative_example,
                source="negative_example",
                metadata={
                    "type": "negative_example",
                    "related_to": feedback_id,
                    "importance": "high",
                    "original_query": self.last_query,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            print("✓ Created negative example to prevent repeating incorrect information")

            # Store direct corrected response for easier retrieval
            correct_response = f"Query: '{self.last_query}'\nCorrect response: '{feedback}'"

            correct_id = await self.memory_manager.add_knowledge(
                content=correct_response,
                source="corrected_response",
                metadata={
                    "type": "corrected_response",
                    "related_to": feedback_id,
                    "importance": "high",
                    "query": self.last_query,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            print("✓ Created correction unit with priority boosting")
        else:
            print("✓ Stored feedback as clarification")

        print("✓ Feedback processed successfully")

    async def add_direct_knowledge(self, content: str, knowledge_type: str = "fact"):
        """
        Add knowledge directly to the memory system.

        Args:
            content: The knowledge content
            knowledge_type: The type of knowledge (fact, rule, concept, etc.)
        """
        # Check if the content appears to be a question rather than a statement
        is_question = any(
            content.strip().lower().startswith(q)
            for q in [
                "what",
                "how",
                "why",
                "when",
                "where",
                "who",
                "which",
                "is",
                "are",
                "can",
                "does",
                "do",
            ]
        ) or content.strip().endswith("?")

        if is_question:
            print(
                "⚠️ Warning: Your input looks like a question rather than a fact. For best results, add statements like:"
            )
            print("   'JavaScript is a programming language' instead of 'what is JavaScript'")
            print("   To continue anyway, type 'y', or any other key to modify your input:")
            confirm = input("> ")
            if confirm.lower() != "y":
                return None

            # If they confirm, store it but tag it as potentially problematic
            logger.info(f"Adding question-like knowledge (with warning): {content}")
            metadata = {
                "type": knowledge_type,
                "source": "direct_input",
                "importance": "medium",  # Lower importance for potentially problematic input
                "is_question_like": True,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            logger.info(f"Adding direct knowledge of type '{knowledge_type}': {content}")
            metadata = {
                "type": knowledge_type,
                "source": "direct_input",
                "importance": "high",
                "timestamp": datetime.now().isoformat(),
            }

        knowledge_id = await self.memory_manager.add_knowledge(
            content=content, source="direct_input", metadata=metadata
        )

        print(f"✓ Added new knowledge unit with ID: {knowledge_id}")
        return knowledge_id

    async def show_knowledge(self):
        """Display all knowledge units."""
        units = await self.memory_manager.list_knowledge()

        if not units:
            print("\nNo knowledge units available yet. Try asking some questions first.")
            return

        print(f"\n=== KNOWLEDGE BASE ({len(units)} units) ===\n")

        for unit in units:
            unit_type = unit.metadata.get("type", "standard")
            query = unit.metadata.get("query", "")

            # Format based on type
            if unit_type == "interaction":
                print(f"[INTERACTION] {unit.unique_id}: Q: {query}")
                if self.debug_mode:
                    print(f"  Content: {unit.original_chunk}")

            elif (
                unit_type == "correction"
                or unit_type == "corrected_response"
                or unit.metadata.get("feedback_type") == "correction"
            ):
                print(f"[CORRECTION] {unit.unique_id}: Q: {query}")
                if self.debug_mode:
                    print(f"  Content: {unit.original_chunk}")

            elif unit_type == "negative_example":
                print(f"[NEGATIVE EXAMPLE] {unit.unique_id}")
                if self.debug_mode:
                    print(f"  Content: {unit.original_chunk}")

            elif unit_type == "feedback":
                feedback_type = unit.metadata.get("feedback_type", "general")
                print(f"[FEEDBACK/{feedback_type.upper()}] {unit.unique_id}: Q: {query}")
                if self.debug_mode:
                    print(f"  Content: {unit.original_chunk}")

            else:
                print(f"[{unit_type.upper()}] {unit.unique_id}")
                if self.debug_mode:
                    print(f"  Content: {unit.original_chunk}")

    def toggle_debug(self):
        """Toggle debug mode to show full content."""
        self.debug_mode = not self.debug_mode
        print(f"\nDebug mode {'enabled' if self.debug_mode else 'disabled'}")


async def main():
    """Run the interactive demo."""
    print("\n" + "=" * 80)
    print(" " * 20 + "ENHANCED MEMORY SYSTEM INTERACTIVE DEMO")
    print("=" * 80)

    print("\nThis demo showcases the key features of the enhanced memory system:")
    print("1. Progressive learning - Building knowledge across multiple interactions")
    print("2. Knowledge correction - Updating understanding when given corrections")
    print("3. Cross-referencing - Making connections between related information")

    print("\nCommands:")
    print("- Type 'exit' or 'quit' to end the demo")
    print("- Type 'knowledge' to see the current knowledge base")
    print("- Type 'debug' to toggle showing full knowledge content")
    print("- Type 'correct: [your correction]' to provide a correction to the last response")
    print("- Type 'add: [your knowledge]' to add direct knowledge to the system")
    print("- Type anything else to ask a question")

    agent = InteractiveAgent()

    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter your query or command: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for using the interactive demo!")
            break

        elif user_input.lower() == "knowledge":
            await agent.show_knowledge()

        elif user_input.lower() == "debug":
            agent.toggle_debug()

        elif user_input.lower().startswith("correct:") or user_input.lower().startswith(
            "correction:"
        ):
            correction = user_input.split(":", 1)[1].strip()
            if correction:
                await agent.process_feedback(correction)
            else:
                print("Please provide the correction content after 'correct:'")

        elif user_input.lower().startswith("add:"):
            knowledge = user_input.split(":", 1)[1].strip()
            if knowledge:
                await agent.add_direct_knowledge(knowledge)
            else:
                print("Please provide the knowledge content after 'add:'")

        else:
            # Process as a query
            response = await agent.process_query(user_input)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())
