"""
Simplified test to validate the feedback processing in the Memory Agent.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python's module search path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import the necessary modules
from ltm_agent.main import initialize_components


async def main():
    print("\n===== Testing Memory Agent Feedback =====\n")

    # Initialize the agent
    components = await initialize_components()
    agent = components["agent"]

    # Add simple feedback
    print("Adding feedback: 'The capital of France is Paris.'")
    await agent.async_process_feedback("The capital of France is Paris.")
    print("Feedback processed successfully!")

    # Query the knowledge
    print("\nQuerying: 'What is the capital of France?'")
    response = await agent.async_invoke("What is the capital of France?")
    print(f"Response: {response}")

    print("\n===== Test Complete =====")


if __name__ == "__main__":
    asyncio.run(main())
