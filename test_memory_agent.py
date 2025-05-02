"""
Test script for the Long-Term Memory Agent.

This script runs a series of example interactions with the agent to demonstrate its capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to Python's module search path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import the necessary modules
from ltm_agent.main import initialize_components

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_example_interactions():
    """Run a series of example interactions with the memory agent."""
    print("\n===== MEMORY AGENT TEST =====\n")
    print("Initializing components...")

    # Initialize the agent and other components
    components = await initialize_components()
    agent = components["agent"]

    # Example interactions
    examples = [
        # Add knowledge about the meaning of life
        {
            "input": "feedback: The meaning of life is 42 according to The Hitchhiker's Guide to the Galaxy.",
            "type": "feedback",
            "description": "Teaching about the meaning of life",
        },
        # Query about the meaning of life
        {
            "input": "What is the meaning of life?",
            "type": "query",
            "description": "Asking about the meaning of life",
        },
        # Add more knowledge about Hitchhiker's Guide
        {
            "input": "feedback: Douglas Adams wrote The Hitchhiker's Guide to the Galaxy as a radio play before it became a book.",
            "type": "feedback",
            "description": "Teaching about Douglas Adams",
        },
        # Query about Hitchhiker's Guide
        {
            "input": "Tell me about The Hitchhiker's Guide to the Galaxy.",
            "type": "query",
            "description": "Asking about Hitchhiker's Guide",
        },
        # Add knowledge about Python
        {
            "input": "feedback: Python is a popular programming language created by Guido van Rossum.",
            "type": "feedback",
            "description": "Teaching about Python",
        },
        # Query about Python
        {
            "input": "Who created Python?",
            "type": "query",
            "description": "Asking about Python's creator",
        },
    ]

    # Process each example
    for i, example in enumerate(examples):
        print(f"\n[Example {i + 1}] {example['description']}")
        print(f"Input: {example['input']}")

        try:
            if example["type"] == "query":
                # Process a query
                response = await agent.async_invoke(example["input"])
                print(f"Response: {response}")
            elif example["type"] == "feedback":
                # Process feedback (remove the prefix)
                feedback = example["input"].replace("feedback: ", "")
                await agent.async_process_feedback(feedback)
                print("Feedback processed successfully.")

            # Add a small delay between operations to avoid overloading
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error processing example: {e}")
            print(f"Error: {e}")

    print("\n===== TEST COMPLETE =====")
    print("All examples have been processed.")
    print("The agent's memory now contains this information in the persistent SQLite database.")
    print(
        "You can run 'python run_agent.py' to interact with the agent and see that it remembers everything."
    )


if __name__ == "__main__":
    asyncio.run(run_example_interactions())
