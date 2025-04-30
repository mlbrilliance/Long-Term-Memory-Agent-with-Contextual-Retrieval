"""
Main entry point for the Long-Term Memory Agent.

This module assembles all the components and provides the main agent loop
for interaction with the system.
"""

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.core.config import Settings
from ltm_agent.memory.bm25_store import BM25Store
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def initialize_components(settings: Settings | None = None) -> dict[str, Any]:
    """
    Initialize all the components needed for the agent.

    Args:
        settings: Optional settings to override defaults

    Returns:
        Dict containing all the initialized components
    """
    if settings is None:
        settings = Settings()

    logger.info("Initializing components with settings: %s", settings)

    # Initialize the vector store
    vector_store = InMemoryVectorStore(embedding_dim=settings.embedding_dim)

    # Initialize the BM25 store
    bm25_store = BM25Store()

    # Initialize the contextualizer
    contextualizer = SimpleContextualizer()

    # Initialize the memory manager
    memory_manager = MemoryManager(
        memory_store=vector_store,
        bm25_store=bm25_store,
        similarity_threshold=settings.similarity_threshold,
        contextualizer=contextualizer,
    )

    # Initialize the language model
    # For this project, we'll use Anthropic's Claude API
    try:
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        llm = ChatAnthropic(
            api_key=api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        logger.info(f"Initialized LLM model: {settings.llm_model}")
    except ImportError:
        logger.warning("langchain_anthropic not installed. Using a mock LLM for development.")

        class MockLLM:
            def invoke(self, prompt):
                return f"Mock response to: {prompt[:30]}..."

        llm = MockLLM()

    # Initialize the agent
    agent = LongTermMemoryAgent(
        memory_manager=memory_manager,
        contextualizer=contextualizer,
        llm=llm,
        knowledge_limit=settings.knowledge_limit,
    )

    return {
        "vector_store": vector_store,
        "bm25_store": bm25_store,
        "memory_manager": memory_manager,
        "contextualizer": contextualizer,
        "llm": llm,
        "agent": agent,
        "settings": settings,
    }


async def agent_loop(components: dict[str, Any]):
    """
    Main agent interaction loop.

    Args:
        components: Dictionary of initialized components
    """
    agent = components["agent"]
    print("\n===== Long-Term Memory Agent =====")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'feedback: <your feedback>' to provide feedback to the agent")
    print("=====================================\n")

    while True:
        try:
            user_input = input("\nQuery: ")

            if user_input.lower() in ("exit", "quit"):
                print("Exiting agent loop. Goodbye!")
                break

            if user_input.lower().startswith("feedback:"):
                feedback = user_input[len("feedback:") :].strip()
                if feedback:
                    print(f"Processing feedback: {feedback}")
                    await asyncio.to_thread(agent.process_feedback, feedback)
                    print("Feedback processed and stored in memory.")
                else:
                    print("Empty feedback. Please provide some content after 'feedback:'")
                continue

            print("\nProcessing your query...")
            response = await asyncio.to_thread(agent.invoke, user_input)

            print(f"\nResponse: {response}")

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Exiting.")
            break
        except Exception as e:
            logger.exception("Error in agent loop")
            print(f"\nAn error occurred: {str(e)}")


async def main():
    """Main entry point for the agent system."""
    try:
        # Initialize all components
        components = await initialize_components()

        # Run the agent loop
        await agent_loop(components)

    except Exception as e:
        logger.exception("Error in main")
        print(f"Critical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main asyncio event loop
    asyncio.run(main())
