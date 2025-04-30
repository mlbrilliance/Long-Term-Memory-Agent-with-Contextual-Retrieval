#!/usr/bin/env python
"""
Simple Interactive Demo for Enhanced Memory System

This script provides a lightweight interactive demo where you can test the key features
of the enhanced memory system without the complexity of the full implementation.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("simple_demo")


class MemoryUnit:
    """Simple memory unit for demonstration."""

    def __init__(self, id: str, content: str, type: str = "fact"):
        self.id = id
        self.content = content
        self.type = type
        self.timestamp = datetime.now().isoformat()


class SimpleMemorySystem:
    """Simple memory system for demonstration."""

    def __init__(self):
        self.knowledge_units = []
        self.counter = 0

    async def add_knowledge(self, content: str, type: str = "fact") -> str:
        """Add knowledge to the system."""
        self.counter += 1
        unit_id = f"unit_{self.counter}"

        unit = MemoryUnit(id=unit_id, content=content, type=type)
        self.knowledge_units.append(unit)

        logger.info(f"Added '{type}' knowledge: {content}")
        return unit_id

    async def search(self, query: str) -> list[tuple[MemoryUnit, float]]:
        """Search for knowledge related to query."""
        results = []

        for unit in self.knowledge_units:
            # Simple keyword matching
            relevance = self._compute_relevance(query, unit.content)
            if relevance > 0:
                results.append((unit, relevance))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:5]  # Top 5 results

    def _compute_relevance(self, query: str, content: str) -> float:
        """Compute relevance between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        # Count matching words
        common_words = query_words.intersection(content_words)

        if not common_words:
            return 0.0

        # Calculate relevance score
        return len(common_words) / max(len(query_words), 1)


class InteractiveDemo:
    """Interactive demo for testing the memory system."""

    def __init__(self):
        self.memory = SimpleMemorySystem()
        self.last_query = None

    async def process_query(self, query: str) -> str:
        """Process a user query."""
        self.last_query = query

        # Search for relevant knowledge
        results = await self.memory.search(query)

        if not results:
            return "I don't have specific information about that query yet."

        # Log search results
        logger.info(f"Found {len(results)} relevant units for query: '{query}'")
        for unit, score in results:
            logger.info(f"- {unit.id} ({unit.type}): {score:.2f} - {unit.content}")

        # Find the best match
        best_unit, best_score = results[0]

        # Only use if it's relevant enough
        if best_score < 0.3:
            return "I don't have specific information about that query yet."

        return best_unit.content

    async def add_knowledge(self, content: str, type: str = "fact") -> str:
        """Add knowledge directly."""
        # Check if content is a question
        is_question = content.strip().endswith("?") or any(
            content.lower().startswith(q)
            for q in ["what", "how", "why", "when", "where", "who", "which"]
        )

        if is_question:
            print("Warning: Your input looks like a question. Consider adding statements instead.")
            print(
                "For example, instead of 'what is Python?', add 'Python is a programming language'"
            )

        # Add the knowledge
        unit_id = await self.memory.add_knowledge(content, type)
        print(f"✓ Added new knowledge with ID: {unit_id}")
        return unit_id

    async def show_knowledge(self):
        """Display all knowledge units."""
        if not self.memory.knowledge_units:
            print("\nNo knowledge units available yet.")
            return

        print(f"\n=== KNOWLEDGE BASE ({len(self.memory.knowledge_units)} units) ===\n")

        for unit in self.memory.knowledge_units:
            print(f"[{unit.type.upper()}] {unit.id}: {unit.content}")


async def main():
    """Run the interactive demo."""
    print("\n" + "=" * 80)
    print(" " * 20 + "SIMPLE MEMORY SYSTEM DEMO")
    print("=" * 80)

    print("\nCommands:")
    print("- Type 'exit' or 'quit' to end the demo")
    print("- Type 'knowledge' to see the current knowledge base")
    print("- Type 'add: [your knowledge]' to add direct knowledge to the system")
    print("- Type anything else to ask a question")

    demo = InteractiveDemo()

    # Add some initial knowledge
    await demo.add_knowledge(
        "JavaScript is a programming language used primarily for web development"
    )
    await demo.add_knowledge(
        "Python is a high-level programming language known for its readability"
    )
    await demo.add_knowledge(
        "The meaning of life according to The Hitchhiker's Guide to the Galaxy is 42"
    )

    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter your query or command: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for using the demo!")
            break

        elif user_input.lower() == "knowledge":
            await demo.show_knowledge()

        elif user_input.lower().startswith("add:"):
            knowledge = user_input.split(":", 1)[1].strip()
            if knowledge:
                await demo.add_knowledge(knowledge)
            else:
                print("Please provide the knowledge content after 'add:'")

        else:
            # Process as a query
            response = await demo.process_query(user_input)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())
