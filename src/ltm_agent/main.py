"""
Main entry point for the Long-Term Memory Agent.

This module assembles all the components and provides the main agent loop
for interaction with the system.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.core.config import Settings
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.manager import MemoryManager
from ltm_agent.memory.sqlite_store import SQLiteVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def initialize_components(settings: Settings | None = None) -> dict[str, Any]:
    """
    Initialize all the components needed for the agent.

    Args:
        settings: Optional settings to override defaults

    Returns:
        Dict containing all the initialized components
    """
    # Explicitly load the .env file from the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loading environment variables from {env_path}")

    if settings is None:
        settings = Settings.from_env()

    # Create a sanitized version of settings for logging (hide API keys)
    sanitized_settings = {
        "vector_db_path": settings.vector_db_path,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model_name": settings.embedding_model_name,
        "log_level": settings.log_level,
        "anthropic_api_key": "[REDACTED]" if settings.anthropic_api_key else None,
        "perplexity_api_key": "[REDACTED]" if settings.perplexity_api_key else None,
        "voyage_api_key": "[REDACTED]" if settings.voyage_api_key else None,
        "cohere_api_key": "[REDACTED]" if settings.cohere_api_key else None,
    }

    logger.info(f"Initializing components with settings: {sanitized_settings}")

    # Define the path for the persistent database file
    db_path = "persistent_agent_memory.db"  # You can change the filename
    logger.info(f"Using persistent SQLite database at: {db_path}")

    # Initialize the vector store with SQLite for persistence
    # Use a default embedding dimension of 768 (typical for embedding models)
    embedding_dim = 768
    vector_store = SQLiteVectorStore(database_path=db_path, embedding_dim=embedding_dim)

    # Initialize the persistent store (important!)
    await vector_store.initialize()

    # Initialize the BM25 store
    bm25_store = RankBM25Store()

    # Initialize the contextualizer
    contextualizer = SimpleContextualizer()

    # Initialize the memory manager with the correct parameters
    memory_manager = MemoryManager(
        memory_store=vector_store,
        bm25_store=bm25_store,
        contextualizer=contextualizer,
    )

    # Initialize the language model
    # For this project, we'll use Anthropic's Claude API
    try:
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Use default values for missing settings
        llm_model = "claude-3-haiku-20240307"  # Using a verified working model
        temperature = 0.7  # Default temperature
        max_tokens = 4000  # Default max tokens

        llm = ChatAnthropic(
            api_key=api_key,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Initialized LLM model: {llm_model}")
    except ImportError:
        logger.warning("langchain_anthropic not installed. Using a mock LLM for development.")

        class MockLLM:
            def invoke(self, prompt):
                return f"Mock response to: {prompt[:30]}..."

        llm = MockLLM()

    # Initialize the agent
    # Use a default value for knowledge_limit
    knowledge_limit = 10  # Default value for maximum number of knowledge units to return
    agent = LongTermMemoryAgent(
        memory_manager=memory_manager,
        contextualizer=contextualizer,
        llm=llm,
        knowledge_limit=knowledge_limit,
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


# Entry point for console script
def run_main():
    """Entry point for console script."""
    asyncio.run(main())
