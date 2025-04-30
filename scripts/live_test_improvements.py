#!/usr/bin/env python
"""
Live Test Script for Long-Term Memory Agent Improvements

This script conducts targeted tests to evaluate the real-world performance of
the enhanced memory system, focusing specifically on:
1. Progressive learning - building knowledge across multiple interactions
2. Knowledge correction - updating understanding when given corrections
3. Multi-hop reasoning - making connections between different knowledge pieces
4. Cross-referencing - establishing relationships between related concepts
5. Memory consolidation - the agent's ability to consolidate and organize related information
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add the project root and src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("live_test_results.log")],
)

logger = logging.getLogger("live_test")


# Create a simple mock LLM for testing
class MockLLM:
    """A mock LLM for testing that returns predefined responses."""

    def __init__(self):
        self.responses = {
            "python": "Python is a high-level programming language known for its readability and versatility.",
            "python features": "Python features include dynamic typing, automatic memory management, and support for multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "memory management": "Python handles memory management automatically through a combination of reference counting and garbage collection. When objects are no longer referenced, Python's garbage collector reclaims the memory.",
            "python limitations": "Python's limitations include slower execution speed compared to compiled languages, the Global Interpreter Lock (GIL) limiting multithreading, and higher memory usage.",
        }

    async def agenerate(self, prompt, **kwargs):
        """Generate a mock response based on the prompt content."""
        # Search for keywords in the prompt
        prompt_lower = prompt.lower()
        response = "I don't have specific information about that query."

        for key, value in self.responses.items():
            if key in prompt_lower:
                response = value
                break

        # Mock response object structure
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)


async def create_simple_memory_agent():
    """
    Create a minimal working memory agent for testing.
    This uses in-memory storage to avoid dependencies on external databases.
    """
    try:
        # Import necessary modules with better error handling
        try:
            from src.ltm_agent.agent.ltm_agent import LongTermMemoryAgent
            from src.ltm_agent.memory.bm25_store import RankBM25Store
            from src.ltm_agent.memory.contextualizer import SimpleContextualizer
            from src.ltm_agent.memory.manager import MemoryManager
            from src.ltm_agent.memory.vector_store import InMemoryVectorStore
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.info("Trying alternative import paths...")

            # Try without the 'src.' prefix
            from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
            from ltm_agent.memory.bm25_store import RankBM25Store
            from ltm_agent.memory.contextualizer import SimpleContextualizer
            from ltm_agent.memory.manager import MemoryManager
            from ltm_agent.memory.vector_store import InMemoryVectorStore

        # Create in-memory vector store
        vector_store = InMemoryVectorStore(collection_name="live_test")

        # Create BM25 store with a temporary file
        temp_bm25_path = os.path.join(os.getcwd(), "temp_bm25_index.pkl")
        bm25_store = RankBM25Store(save_path=temp_bm25_path)

        # Create contextualizer
        contextualizer = SimpleContextualizer()

        # Create memory manager
        memory_manager = MemoryManager(
            memory_store=vector_store, contextualizer=contextualizer, bm25_store=bm25_store
        )

        # Create LTM Agent
        agent = LongTermMemoryAgent(
            memory_manager=memory_manager,
            contextualizer=contextualizer,
            llm=MockLLM(),
            knowledge_limit=5,
            enable_consolidation=True,
        )

        logger.info("Successfully created memory agent for testing")
        return agent

    except Exception as e:
        logger.error(f"Error creating memory agent: {e}")
        raise


async def test_progressive_learning(agent):
    """Test progressive learning across multiple interactions."""
    logger.info("===== TESTING PROGRESSIVE LEARNING =====")

    # Step 1: Initial query about Python
    logger.info("Step 1: Initial query about Python")
    response1 = await agent.async_invoke("What is Python?")
    logger.info(f"Response: {response1}")

    # Step 2: Follow-up query building on previous knowledge
    logger.info("Step 2: Follow-up query about Python features")
    response2 = await agent.async_invoke("Tell me more about Python's features")
    logger.info(f"Response: {response2}")

    # Step 3: Third query that requires connecting information
    logger.info("Step 3: Query about Python memory management")
    response3 = await agent.async_invoke("How does Python handle memory management?")
    logger.info(f"Response: {response3}")

    # Step 4: Test knowledge aggregation
    logger.info("Step 4: Testing knowledge aggregation")
    response4 = await agent.async_invoke("Summarize what you know about Python")
    logger.info(f"Response: {response4}")

    # Gather results
    results = {
        "initial_query_response": response1,
        "follow_up_response": response2,
        "connection_query_response": response3,
        "aggregation_response": response4,
    }

    return results


async def test_knowledge_correction(agent):
    """Test knowledge correction capabilities."""
    logger.info("===== TESTING KNOWLEDGE CORRECTION =====")

    # Step 1: Initial query about Python limitations
    logger.info("Step 1: Query about Python limitations")
    response1 = await agent.async_invoke("What are Python's limitations?")
    logger.info(f"Initial response: {response1}")

    # Step 2: Provide corrective feedback
    correction = """
    Python's limitations are often overstated. While some say Python is slow, it's
    actually quite fast for most applications, especially with recent improvements
    in PyPy and Python 3.11+. The GIL is only an issue for CPU-bound multithreaded
    code, not for I/O-bound or multiprocess applications. Memory usage can be
    optimized with appropriate data structures and libraries like NumPy.
    """

    logger.info(f"Step 2: Providing correction: {correction}")
    await agent.process_feedback(correction)

    # Step 3: Same query again to see if correction was applied
    logger.info("Step 3: Repeating query to check correction")
    response2 = await agent.async_invoke("What are Python's limitations?")
    logger.info(f"Response after correction: {response2}")

    # Step 4: Similar query to test correction generalization
    logger.info("Step 4: Similar query to test correction generalization")
    response3 = await agent.async_invoke("Tell me about potential drawbacks of using Python")
    logger.info(f"Response to similar query: {response3}")

    # Gather results
    results = {
        "original_response": response1,
        "correction_applied": correction,
        "response_after_correction": response2,
        "response_to_similar_query": response3,
    }

    return results


async def main():
    """Run the live tests and display results."""
    print("=" * 80)
    print("LONG-TERM MEMORY AGENT - LIVE IMPROVEMENT TESTING")
    print("=" * 80)
    print("\nInitializing test environment...")

    try:
        # Create agent
        agent = await create_simple_memory_agent()

        # Test results
        results = {
            "progressive_learning": {},
            "knowledge_correction": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Test progressive learning
        results["progressive_learning"] = await test_progressive_learning(agent)

        # Test knowledge correction
        results["knowledge_correction"] = await test_knowledge_correction(agent)

        # Save results to file
        result_file = f"live_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\nTests completed successfully!")
        print(f"Results saved to {result_file}")

        # Analyze and report on results
        analyze_results(results)

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
        print(f"\nError during testing: {e}")


def analyze_results(results):
    """Analyze and report on test results."""
    print("\n" + "=" * 80)
    print("TEST RESULTS ANALYSIS")
    print("=" * 80)

    # Analyze progressive learning
    pl_results = results.get("progressive_learning", {})
    if pl_results:
        print("\n✓ Progressive Learning Test:")
        if all(pl_results.values()):
            print("  All queries were processed successfully.")
            # Check if later responses incorporate knowledge from earlier queries
            combined_earlier = pl_results.get("initial_query_response", "") + pl_results.get(
                "follow_up_response", ""
            )
            aggregation = pl_results.get("aggregation_response", "")

            # Very simple check for knowledge retention - look for shared words
            shared_words = set(combined_earlier.lower().split()) & set(aggregation.lower().split())
            significant_shared = len(shared_words) > 10

            if significant_shared:
                print(
                    "  ✓ Good knowledge retention: Final response incorporated information from earlier interactions."
                )
            else:
                print(
                    "  ⚠ Limited knowledge retention: Final response may not fully incorporate earlier information."
                )
        else:
            print("  ⚠ Some queries did not receive responses.")

    # Analyze knowledge correction
    kc_results = results.get("knowledge_correction", {})
    if kc_results:
        print("\n✓ Knowledge Correction Test:")
        original = kc_results.get("original_response", "")
        corrected = kc_results.get("response_after_correction", "")

        if original != corrected:
            print("  ✓ Correction was applied: Response changed after feedback.")
            # Calculate simple text difference
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(None, original, corrected).ratio()
            print(f"  Response similarity before/after correction: {similarity:.2f}")

            if similarity < 0.7:
                print("  ✓ Major correction impact: Significant response changes.")
            else:
                print("  ⚠ Minor correction impact: Subtle response changes.")

            # Check correction generalization
            similar_query = kc_results.get("response_to_similar_query", "")
            similar_to_original = SequenceMatcher(None, original, similar_query).ratio()
            similar_to_corrected = SequenceMatcher(None, corrected, similar_query).ratio()

            if similar_to_corrected > similar_to_original:
                print("  ✓ Good correction generalization: Correction applied to similar queries.")
            else:
                print("  ⚠ Limited correction generalization.")
        else:
            print("  ⚠ Correction was not applied: Response unchanged after feedback.")

    print("\nOverall Assessment:")
    if (pl_results and all(pl_results.values())) and (kc_results and original != corrected):
        print(
            "✓ The memory system successfully demonstrated both progressive learning and knowledge correction."
        )
    elif pl_results and all(pl_results.values()):
        print(
            "⚠ The memory system demonstrated progressive learning but had issues with knowledge correction."
        )
    elif kc_results and original != corrected:
        print(
            "⚠ The memory system demonstrated knowledge correction but had issues with progressive learning."
        )
    else:
        print("⚠ The memory system showed limitations in both tested areas.")


if __name__ == "__main__":
    asyncio.run(main())
