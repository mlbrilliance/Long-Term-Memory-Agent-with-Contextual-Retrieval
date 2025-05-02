#!/usr/bin/env python
"""
Interactive Memory Agent Runner

This script provides a simple CLI for interacting with the Long-Term Memory Agent.
It handles all the import issues by setting up the paths correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add the project root directory to Python's module search path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
load_dotenv()

# Now import the script components after setting up the path
try:
    from memory.simple_enhanced_memory import SimpleEnhancedMemory
except ImportError:
    logger.error("Could not import SimpleEnhancedMemory. Using built-in memory system.")

    # Simple memory implementation if the import fails
    class SimpleEnhancedMemory:
        def __init__(self):
            self.memory = {}

        def add_knowledge(self, content, unit_type="fact", metadata=None):
            key = len(self.memory)
            self.memory[key] = {"content": content, "type": unit_type, "metadata": metadata or {}}
            return key

        def answer(self, query):
            # Simple matching on exact query or any content containing query words
            query_words = set(query.lower().split())
            for item in self.memory.values():
                if query.lower() in item["content"].lower():
                    return item["content"]

                # Check word overlap
                content_words = set(item["content"].lower().split())
                if query_words.intersection(content_words):
                    return item["content"]

            return "I don't have information about that."


class InteractiveAgent:
    """Interactive agent with memory capabilities."""

    def __init__(self):
        """Initialize the agent."""
        self.memory_system = SimpleEnhancedMemory()
        self.last_query = None

    async def process_query(self, query):
        """Process a user query and generate a response."""
        self.last_query = query
        logger.info(f"Processing query: {query}")

        # Generate a response using the memory system
        response = self.memory_system.answer(query)
        logger.info(f"Generated response: {response}")

        return response

    async def add_knowledge(self, content, knowledge_type="fact"):
        """Add knowledge to the memory system."""
        result = self.memory_system.add_knowledge(content, knowledge_type)
        return result

    async def process_feedback(self, correction):
        """Process user feedback or correction."""
        if not self.last_query:
            print("No previous query to correct.")
            return

        print(f"Processing correction for query: {self.last_query}")

        # Add the correction as a special knowledge type
        await self.add_knowledge(correction, "correction")

        print("Correction added successfully! Future queries will use this updated knowledge.")


async def main():
    """Run the interactive agent."""
    print("\n" + "=" * 80)
    print(" " * 20 + "MEMORY AGENT INTERACTIVE CLI")
    print("=" * 80)

    print("\nCommands:")
    print("- Type 'exit' or 'quit' to end the session")
    print("- Type 'correct: [your correction]' to provide a correction to the last response")
    print("- Type 'add: [your knowledge]' to add direct knowledge to the system")
    print("- Type anything else to ask a question")

    agent = InteractiveAgent()

    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter your query or command: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for using the Memory Agent CLI. Goodbye!")
            break

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
                await agent.add_knowledge(knowledge)
                print("\nKnowledge added successfully as 'fact'!")
                print(f"Content: {knowledge}")
            else:
                print("Please provide the knowledge content after 'add:'")

        else:
            # Process as a query
            response = await agent.process_query(user_input)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    # Run the main asyncio event loop
    asyncio.run(main())
