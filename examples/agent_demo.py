"""
Demo of the Long-Term Memory Agent.

This script demonstrates the Long-Term Memory Agent in action with
a simple interaction flow, showing how the agent:
1. Retrieves relevant knowledge from memory
2. Incorporates that knowledge in responses
3. Learns from interactions and feedback
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Add the project root to the path to enable imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add the src directory to the Python path
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from src.ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from src.ltm_agent.memory.contextualizer import SimpleContextualizer
from src.ltm_agent.memory.in_memory_store import InMemoryVectorStore
from src.ltm_agent.memory.manager import MemoryManager

# Setup logging and load environment variables
load_dotenv()


async def seed_memory(memory_manager):
    """Seed the memory with some initial knowledge."""
    print("Seeding memory with initial knowledge...")

    # Create some knowledge units about major cities
    knowledge_units = [
        {
            "original_chunk": "Paris is the capital of France and known as the City of Light.",
            "knowledge_source": "corpus",
        },
        {
            "original_chunk": "Tokyo is the capital of Japan and the world's most populous metropolitan area.",
            "knowledge_source": "corpus",
        },
        {
            "original_chunk": "New York City is known as the Big Apple and is the most populous city in the United States.",
            "knowledge_source": "corpus",
        },
    ]

    # Add each knowledge unit to memory
    for ku_data in knowledge_units:
        await memory_manager.process_potential_knowledge(
            content=ku_data["original_chunk"],
            source=ku_data["knowledge_source"],
            metadata={"type": "seed"},
            similarity_threshold=0.7,  # Pass similarity_threshold here instead
        )

    print(f"Added {len(knowledge_units)} knowledge units to memory.")


async def simulate_interaction(agent):
    """Simulate an interaction with the agent."""
    queries = [
        "What is Paris known for?",
        "Tell me about Tokyo.",
        "What's the most populous city in the US?",
    ]

    print("\n===== SIMULATING AGENT INTERACTIONS =====")

    for query in queries:
        print(f"\n>> USER QUERY: {query}")

        # Use the async methods directly
        response = await agent.async_invoke(query)

        print(f"<< AGENT RESPONSE: {response}")

        # Simulate human feedback
        if "Paris" in query:
            feedback = "Paris is also famous for the Eiffel Tower and its cuisine."
            print(f"\n>> USER FEEDBACK: {feedback}")
            await agent.async_process_feedback(feedback)

    # Try a follow-up query that should leverage learned information
    follow_up = "What are some famous landmarks in Paris?"
    print(f"\n>> FOLLOW-UP QUERY: {follow_up}")
    response = await agent.async_invoke(follow_up)
    print(f"<< AGENT RESPONSE: {response}")


async def main():
    """Run the agent demo."""
    print("Initializing Long-Term Memory Agent components...")

    # Initialize vector store
    vector_store = InMemoryVectorStore(embedding_dim=1536)

    # Initialize contextualizer
    contextualizer = SimpleContextualizer()

    # Initialize memory manager - set bm25_store to None to avoid incompatibility
    memory_manager = MemoryManager(
        memory_store=vector_store,
        bm25_store=None,  # Set to None to avoid using BM25Store
        contextualizer=contextualizer,
    )

    # Initialize LLM - use a mock LLM if no API key is available
    try:
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        llm = ChatAnthropic(api_key=api_key, model="claude-3-haiku-20240307", temperature=0.7)
        print("Using Claude LLM")
    except (ImportError, ValueError) as e:
        print(f"Warning: {str(e)}. Using Mock LLM instead.")

        class MockLLM:
            def invoke(self, prompt):
                if "Paris" in prompt:
                    return "Paris is the capital of France and known as the City of Light. It's famous for its art, culture, and landmarks."
                elif "Tokyo" in prompt:
                    return "Tokyo is the capital of Japan and the largest metropolitan area in the world. It's known for its technology, food, and unique culture."
                elif "New York" in prompt or "US" in prompt:
                    return "New York City is the most populous city in the United States. It's known as the Big Apple and is a global center for finance, culture, and entertainment."
                elif "Eiffel Tower" in prompt or "landmarks" in prompt:
                    return "The Eiffel Tower is the most famous landmark in Paris. Other notable landmarks include the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe."
                else:
                    return "I don't have specific information about that in my knowledge base."

        llm = MockLLM()

    # Initialize the agent
    agent = LongTermMemoryAgent(
        memory_manager=memory_manager, contextualizer=contextualizer, llm=llm, knowledge_limit=5
    )

    # Seed the memory with initial knowledge
    await seed_memory(memory_manager)

    # Simulate interactions with the agent
    await simulate_interaction(agent)

    print("\nDemo completed.")


if __name__ == "__main__":
    asyncio.run(main())
