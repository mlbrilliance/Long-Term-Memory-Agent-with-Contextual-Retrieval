"""
Simplified Memory Agent with Persistent Storage

This script provides a simplified interface to the Long-Term Memory Agent
that works around some of the more complex cross-referencing features
while still providing persistent storage of knowledge.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add the src directory to Python's module search path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the necessary components
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.sqlite_store import SQLiteVectorStore

# Load environment variables
load_dotenv()


class SimpleMemoryAgent:
    """A simplified memory agent with persistent storage."""

    def __init__(self, db_path="persistent_agent_memory.db"):
        """Initialize the simplified memory agent."""
        self.db_path = db_path
        self.vector_store = None
        self.contextualizer = SimpleContextualizer()
        self.llm = None

        # Initialize Claude API
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=1000,
        )

        logger.info(f"Initialized SimpleMemoryAgent with database at {db_path}")

    async def initialize(self):
        """Initialize the vector store."""
        self.vector_store = SQLiteVectorStore(database_path=self.db_path)
        await self.vector_store.initialize()
        logger.info("Vector store initialized")

    async def add_knowledge(self, content):
        """Add knowledge to the memory store."""
        if not self.vector_store:
            await self.initialize()

        # Generate a unique ID
        unique_id = str(uuid.uuid4())

        # Create a simple knowledge unit
        knowledge_unit = {
            "unique_id": unique_id,
            "original_chunk": content,
            "contextual_text": content,  # Simple implementation without contextualizing
            "knowledge_source": "feedback",
            "timestamp": datetime.now().isoformat(),
            "embedding_vector": await self.vector_store._generate_embedding_implementation(content),
        }

        # Add to the store
        await self.vector_store._add_implementation(knowledge_unit)
        logger.info(f"Added knowledge: {content[:50]}...")
        return unique_id

    async def search_knowledge(self, query, limit=5):
        """Search for knowledge related to the query."""
        if not self.vector_store:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self.vector_store._generate_embedding_implementation(query)

        # Search for related knowledge
        try:
            results = await self.vector_store._search_implementation(query, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    async def answer_question(self, question):
        """Answer a question using the stored knowledge."""
        if not self.vector_store:
            await self.initialize()

        # Get related knowledge
        related_knowledge = await self.search_knowledge(question)

        # Format the knowledge as context
        context = ""
        if related_knowledge:
            context = "Here is some relevant information:\n\n"
            for i, (unit, score) in enumerate(related_knowledge):
                context += f"{i + 1}. {unit.original_chunk}\n\n"

        # Prepare the prompt
        prompt = f"""You are a knowledgeable assistant with access to information stored in a memory system.

QUESTION: {question}

{context if context else "I don't have specific information about that topic in my memory yet."}

Based on the information available (if any), please provide a helpful and accurate answer to the question.
If the information in the memory is incomplete or not relevant, please state that clearly and provide general knowledge as appropriate.
"""

        # Generate a response
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        return response.content


async def main():
    """Run the simplified memory agent interactive loop."""
    print("\n===== Simple Memory Agent with Persistent Storage =====")
    print("This agent stores knowledge in a SQLite database for long-term memory.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'add: <information>' to add new knowledge.")
    print("Type anything else to ask a question.")
    print("=" * 60 + "\n")

    agent = SimpleMemoryAgent()
    await agent.initialize()

    while True:
        user_input = input("\nInput: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Your knowledge has been saved.")
            break

        if user_input.lower().startswith("add:"):
            # Add knowledge
            knowledge = user_input[4:].strip()
            await agent.add_knowledge(knowledge)
            print(f"Knowledge added: {knowledge}")
        else:
            # Answer a question
            print("Generating answer...")
            answer = await agent.answer_question(user_input)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
