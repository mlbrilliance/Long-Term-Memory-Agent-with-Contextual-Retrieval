import logging
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_dir)

# Import required modules
from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_evaluate")


class SimpleMockLLM:
    """A simple mock LLM for testing purposes."""

    def invoke(self, prompt: str) -> str:
        """Generate a response based on keywords in the prompt."""
        if "Paris" in prompt:
            return "Paris is the capital of France and known for the Eiffel Tower."
        elif "Tokyo" in prompt:
            return "Tokyo is the capital of Japan and the most populous metropolitan area in the world."
        elif "New York" in prompt:
            return "New York City is the most populous city in the United States."
        elif "machine learning" in prompt.lower():
            return "Machine learning is a subset of artificial intelligence."
        elif "quantum computing" in prompt.lower():
            return "Quantum computing uses quantum phenomena like superposition and entanglement."
        else:
            return "I don't have specific information about that."


def test_agent_improvements():
    """Test and demonstrate the improvements to the agent."""
    logger.info("Starting simplified agent evaluation")

    # Create a vector store
    vector_store = InMemoryVectorStore()

    # Create a memory manager
    memory_manager = MemoryManager(vector_store=vector_store, contextualizer=SimpleContextualizer())

    # Add sample knowledge
    knowledge_items = [
        {
            "content": "Paris is the capital of France and known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            "source": "corpus",
            "metadata": {"type": "city", "country": "France"},
        },
        {
            "content": "Tokyo is the capital of Japan and the most populous metropolitan area in the world.",
            "source": "corpus",
            "metadata": {"type": "city", "country": "Japan"},
        },
        {
            "content": "New York City is the most populous city in the United States and known for landmarks such as the Statue of Liberty and Empire State Building.",
            "source": "corpus",
            "metadata": {"type": "city", "country": "USA"},
        },
    ]

    for item in knowledge_items:
        memory_manager.add_knowledge(
            content=item["content"], source=item["source"], metadata=item["metadata"]
        )

    # Create a mock LLM
    llm = SimpleMockLLM()

    # Create the agent with consolidation disabled
    agent = LongTermMemoryAgent(memory_manager=memory_manager, llm=llm, enable_consolidation=False)

    # Test standard queries
    queries = ["What is Paris known for?", "Tell me about Tokyo.", "What's in New York City?"]

    for query in queries:
        logger.info(f"Query: {query}")
        response = agent.invoke(query)
        logger.info(f"Response: {response}")
        logger.info("---")

    # Test knowledge update through feedback
    logger.info("Testing knowledge update:")

    # Initial query about Paris
    query = "What is Paris known for?"
    logger.info(f"Initial Query: {query}")
    response = agent.invoke(query)
    logger.info(f"Initial Response: {response}")

    # Provide feedback to enrich knowledge
    feedback = "Paris is also known as the 'City of Light' and is famous for its cuisine, fashion, and art scene."
    logger.info(f"Providing feedback: {feedback}")
    agent.process_feedback(feedback)

    # Query again to see if knowledge has been updated
    logger.info(f"Follow-up Query: {query}")
    updated_response = agent.invoke(query)
    logger.info(f"Updated Response: {updated_response}")

    # Check if the response has been enriched with the feedback
    if "City of Light" in updated_response:
        logger.info("SUCCESS: Knowledge was successfully updated with feedback")
    else:
        logger.info("FAILURE: Knowledge was not updated properly")

    # List all knowledge units to see what's been stored
    knowledge_units = memory_manager.list_knowledge()
    logger.info(f"Total knowledge units: {len(knowledge_units)}")

    for unit in knowledge_units:
        logger.info(f"ID: {unit.unique_id}")
        logger.info(f"Content: {unit.original_chunk}")
        logger.info(f"Source: {unit.source}")
        logger.info(f"Metadata: {unit.metadata}")
        logger.info("---")

    logger.info("Simplified evaluation complete")


if __name__ == "__main__":
    test_agent_improvements()
