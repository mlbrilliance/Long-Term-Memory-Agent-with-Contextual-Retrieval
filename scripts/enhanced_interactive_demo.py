#!/usr/bin/env python
"""
Enhanced Interactive Demo

This script provides an interactive demo where you can test the enhanced memory system
with all five core capabilities:

1. Progressive learning - Building knowledge across multiple interactions
2. Knowledge correction - Updating understanding when given corrections
3. Multi-hop reasoning - Making connections between different knowledge pieces
4. Cross-referencing - Establishing relationships between related concepts
5. Memory consolidation - Consolidating and organizing related information

Usage:
    python scripts/enhanced_interactive_demo.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("enhanced_demo.log")],
)
logger = logging.getLogger("enhanced_demo")

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import the enhanced memory system components
from memory.simple_enhanced_memory import SimpleEnhancedMemory


class EnhancedInteractiveAgent:
    """
    Interactive agent that uses the enhanced memory system.
    """

    def __init__(self):
        """Initialize the agent."""
        self.memory_system = SimpleEnhancedMemory()
        self.last_query = None
        self.debug_mode = False
        self.example_scenarios = self._initialize_examples()

    def _initialize_examples(self) -> dict[str, list[dict[str, str]]]:
        """Initialize example scenarios for demonstration."""
        return {
            "progressive_learning": [
                {"query": "What is Python?", "expected": "Python is a programming language."},
                {
                    "query": "Who created Python?",
                    "expected": "Python was created by Guido van Rossum in 1991.",
                },
            ],
            "knowledge_correction": [
                {
                    "query": "When was JavaScript created?",
                    "expected": "JavaScript was created in 2000.",
                },
                {"correction": "JavaScript was actually created in 1995 by Brendan Eich."},
            ],
            "multi_hop_reasoning": [
                {"add": "Alice is Bob's sister."},
                {"add": "Bob is Carol's father."},
                {
                    "query": "What is Alice's relationship to Carol?",
                    "expected": "Alice is Carol's aunt...",
                },
            ],
            "cross_referencing": [
                {"add": "Machine learning is a subset of artificial intelligence."},
                {"add": "Neural networks are used in machine learning."},
                {
                    "query": "How are neural networks related to AI?",
                    "expected": "Neural networks are related to AI...",
                },
            ],
            "memory_consolidation": [
                {"add": "The Earth is the third planet from the Sun."},
                {"add": "Our solar system has eight planets."},
                {
                    "add": "The Earth is the only known planet with abundant liquid water.",
                    "type": "consolidation",
                },
                {"query": "What unique feature does Earth have?", "expected": "...water..."},
            ],
        }

    async def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.

        Args:
            query: The user query

        Returns:
            The response to the query
        """
        self.last_query = query
        logger.info(f"Processing query: {query}")

        # Add the query as interaction knowledge
        await self.add_interaction(query)

        # Generate a response
        response = self.memory_system.answer(query)
        logger.info(f"Generated response: {response}")

        # Add the response as knowledge
        await self.add_response(query, response)

        return response

    async def add_interaction(self, query: str) -> None:
        """
        Add an interaction to the memory system.

        Args:
            query: The user query
        """
        self.memory_system.add_knowledge(
            query,
            unit_type="interaction",
            metadata={"query": query, "timestamp": datetime.now().isoformat()},
        )

    async def add_response(self, query: str, response: str) -> None:
        """
        Add a response to the memory system.

        Args:
            query: The user query
            response: The system response
        """
        self.memory_system.add_knowledge(
            response,
            unit_type="response",
            metadata={"query": query, "timestamp": datetime.now().isoformat()},
        )

    async def process_feedback(self, correction: str) -> None:
        """
        Process user feedback or correction.

        Args:
            correction: The correction provided by the user
        """
        if not self.last_query:
            print("No previous query to correct.")
            return

        print(f"\nProcessing correction for query: {self.last_query}")

        # Add the correction to the memory system
        self.memory_system.add_knowledge(
            correction,
            unit_type="correction",
            metadata={"original_query": self.last_query, "timestamp": datetime.now().isoformat()},
        )

        print("Correction added successfully! Future queries will use this updated knowledge.")

    async def add_direct_knowledge(self, content: str, knowledge_type: str = "fact") -> None:
        """
        Add direct knowledge to the memory system.

        Args:
            content: The knowledge content
            knowledge_type: The type of knowledge (fact, correction, consolidation, etc.)
        """
        # If the content is a question, treat it as a query instead
        if content.strip().endswith("?"):
            print("\nThis looks like a question. Processing as a query instead...")
            response = await self.process_query(content)
            print(f"\nResponse: {response}")
            return

        # Add to memory system
        self.memory_system.add_knowledge(
            content,
            unit_type=knowledge_type,
            metadata={"source": "direct_input", "timestamp": datetime.now().isoformat()},
        )

        print(f"\nKnowledge added successfully as '{knowledge_type}'!")
        print(f"Content: {content}")

    async def show_knowledge(self) -> None:
        """Display all knowledge units in the memory system."""
        # In a real implementation, this would extract and display all knowledge units
        # For the SimpleEnhancedMemory, we'll show a summary
        if not self.memory_system.units:
            print("\nNo knowledge units in memory yet.")
            return

        print(f"\n=== KNOWLEDGE BASE ({len(self.memory_system.units)} units) ===\n")

        for i, unit in enumerate(self.memory_system.units):
            unit_type = unit.type

            # Format based on type
            if unit_type == "interaction":
                print(f"[INTERACTION] Unit {i + 1}: {unit.content[:50]}...")
                if self.debug_mode:
                    print(f"  Full content: {unit.content}")

            elif unit_type == "correction":
                original_query = unit.metadata.get("original_query", "")
                print(f"[CORRECTION] Unit {i + 1} for '{original_query}'")
                if self.debug_mode:
                    print(f"  Content: {unit.content}")

            elif unit_type == "consolidation":
                print(f"[CONSOLIDATION] Unit {i + 1}")
                if self.debug_mode:
                    print(f"  Content: {unit.content}")

            else:
                print(f"[{unit_type.upper()}] Unit {i + 1}")
                if self.debug_mode:
                    print(f"  Content: {unit.content}")

    def toggle_debug(self) -> None:
        """Toggle debug mode to show full content."""
        self.debug_mode = not self.debug_mode
        print(f"\nDebug mode {'enabled' if self.debug_mode else 'disabled'}")

    async def load_example_scenario(self, scenario: str) -> None:
        """
        Load an example scenario to demonstrate a specific capability.

        Args:
            scenario: The name of the scenario to load
        """
        if scenario not in self.example_scenarios:
            print(f"\nUnknown scenario: {scenario}")
            print(f"Available scenarios: {', '.join(self.example_scenarios.keys())}")
            return

        print(f"\n=== LOADING SCENARIO: {scenario.upper()} ===\n")

        steps = self.example_scenarios[scenario]

        for i, step in enumerate(steps):
            print(f"Step {i + 1}:")

            if "add" in step:
                content = step["add"]
                unit_type = step.get("type", "fact")
                print(f"  Adding knowledge: {content}")
                await self.add_direct_knowledge(content, unit_type)

            elif "query" in step:
                query = step["query"]
                expected = step.get("expected", "")
                print(f"  Query: {query}")
                response = await self.process_query(query)
                print(f"  Response: {response}")
                if expected:
                    print(f"  Expected to contain: {expected}")

            elif "correction" in step:
                correction = step["correction"]
                print(f"  Adding correction: {correction}")
                await self.process_feedback(correction)

            # Small pause between steps
            await asyncio.sleep(0.5)

        print(f"\n=== SCENARIO COMPLETED: {scenario.upper()} ===")


async def main():
    """Run the enhanced interactive demo."""
    print("\n" + "=" * 80)
    print(" " * 20 + "ENHANCED MEMORY SYSTEM INTERACTIVE DEMO")
    print("=" * 80)

    print("\nThis demo showcases all five core capabilities of the enhanced memory system:")
    print("1. Progressive learning - Building knowledge across multiple interactions")
    print("2. Knowledge correction - Updating understanding when given corrections")
    print("3. Multi-hop reasoning - Making connections between different knowledge pieces")
    print("4. Cross-referencing - Establishing relationships between related concepts")
    print("5. Memory consolidation - Consolidating and organizing related information")

    print("\nCommands:")
    print("- Type 'exit' or 'quit' to end the demo")
    print("- Type 'knowledge' to see the current knowledge base")
    print("- Type 'debug' to toggle showing full knowledge content")
    print("- Type 'correct: [your correction]' to provide a correction to the last response")
    print("- Type 'add: [your knowledge]' to add direct knowledge to the system")
    print("- Type 'add as consolidation: [your knowledge]' to add consolidated knowledge")
    print("- Type 'demo: [capability]' to run an example scenario demonstrating a capability")
    print("- Type anything else to ask a question")

    print("\nAvailable demo scenarios:")
    print("- demo: progressive_learning - Shows building knowledge over time")
    print("- demo: knowledge_correction - Shows correcting outdated information")
    print("- demo: multi_hop_reasoning - Shows connecting facts (Alice-Bob-Carol example)")
    print("- demo: cross_referencing - Shows relating concepts (neural networks-ML-AI example)")
    print(
        "- demo: memory_consolidation - Shows prioritizing consolidated knowledge (Earth example)"
    )

    agent = EnhancedInteractiveAgent()

    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter your query or command: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for using the enhanced interactive demo!")
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

        elif user_input.lower().startswith("add as consolidation:"):
            knowledge = user_input.split(":", 1)[1].strip()
            if knowledge:
                await agent.add_direct_knowledge(knowledge, "consolidation")
            else:
                print("Please provide the knowledge content after 'add as consolidation:'")

        elif user_input.lower().startswith("add:"):
            knowledge = user_input.split(":", 1)[1].strip()
            if knowledge:
                await agent.add_direct_knowledge(knowledge)
            else:
                print("Please provide the knowledge content after 'add:'")

        elif user_input.lower().startswith("demo:"):
            scenario = user_input.split(":", 1)[1].strip()
            if scenario:
                await agent.load_example_scenario(scenario)
            else:
                print("Please specify which scenario to demo after 'demo:'")
                print(
                    "Available scenarios: progressive_learning, knowledge_correction, multi_hop_reasoning, cross_referencing, memory_consolidation"
                )

        else:
            # Process as a query
            response = await agent.process_query(user_input)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())
