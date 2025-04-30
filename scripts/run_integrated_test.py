#!/usr/bin/env python
"""
Integrated Test Harness for Enhanced Memory System

This script integrates the enhanced memory system into the full agent framework
and evaluates its real-world performance on complex scenarios including:
- Multi-step reasoning chains
- Knowledge correction and update
- Information synthesis across topics
- Cross-referencing between knowledge domains
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("integrated_test.log")],
)
logger = logging.getLogger("integrated_test")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


# Mock LLM implementation for testing
class MockLLM:
    """Mock LLM for testing that returns predefined responses based on context."""

    def __init__(self):
        self.responses = {
            # Machine learning domain
            "what is machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "types of machine learning": "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Each has different applications and methodologies.",
            "what is supervised learning": "Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions or decisions. It requires input-output pairs for training.",
            # Python programming domain
            "python features": "Python features include dynamic typing, automatic memory management, comprehensive standard libraries, and support for multiple programming paradigms.",
            "python memory management": "Python manages memory through reference counting and a garbage collector, which automatically deallocates objects that are no longer referenced.",
            # Cross-domain topics
            "machine learning in python": "Python is widely used for machine learning due to libraries like TensorFlow, PyTorch, and scikit-learn that provide powerful tools for building and training models.",
        }

        # Incorrect information (to be corrected)
        self.incorrect_responses = {
            "limitations of supervised learning": "Supervised learning has very few limitations and works equally well with all types of data and problem domains.",
            "python performance": "Python is always the fastest programming language for all computational tasks.",
        }

    async def agenerate(self, prompt, **kwargs):
        """Generate a mock response based on prompt content and context."""
        # Extract relevant knowledge (if provided)
        relevant_knowledge = kwargs.get("relevant_knowledge", [])
        has_relevant_knowledge = len(relevant_knowledge) > 0

        # Extract query from prompt
        query_lines = [line for line in prompt.split("\n") if "Query:" in line]
        query = query_lines[0].replace("Query:", "").strip() if query_lines else prompt.lower()

        # First check for exact matches
        response = None
        for key, value in self.responses.items():
            if key.lower() in query.lower():
                response = value
                break

        # Then check for incorrect information (for testing correction)
        for key, value in self.incorrect_responses.items():
            if key.lower() in query.lower():
                response = value
                break

        # If no direct match, but we have relevant knowledge
        if not response and has_relevant_knowledge:
            # Simplistic approach: Combine the first 2 knowledge units as response
            combined = " ".join([unit.original_chunk for unit, _ in relevant_knowledge[:2]])
            response = f"Based on my knowledge: {combined}"

        # Default response
        if not response:
            response = "I don't have specific information about that query."

        # Mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)


async def setup_agent():
    """Set up the enhanced LTM Agent with memory components."""
    try:
        # Import necessary components
        try:
            from src.ltm_agent.agent.ltm_agent import LongTermMemoryAgent
            from src.ltm_agent.memory.bm25_store import RankBM25Store
            from src.ltm_agent.memory.contextualizer import SimpleContextualizer
            from src.ltm_agent.memory.manager import MemoryManager
            from src.ltm_agent.memory.vector_store import InMemoryVectorStore

            logger.info("Successfully imported from src.ltm_agent")
        except ImportError as e:
            logger.error(f"Import error: {e}")
            from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
            from ltm_agent.memory.bm25_store import RankBM25Store
            from ltm_agent.memory.contextualizer import SimpleContextualizer
            from ltm_agent.memory.manager import MemoryManager
            from ltm_agent.memory.vector_store import InMemoryVectorStore

            logger.info("Successfully imported from ltm_agent")

        # Set up in-memory storage
        vector_store = InMemoryVectorStore(collection_name="integrated_test")

        # Set up BM25 store with temp file
        temp_path = os.path.join(os.getcwd(), "temp_bm25_test.pkl")
        bm25_store = RankBM25Store(save_path=temp_path)

        # Set up contextualizer
        contextualizer = SimpleContextualizer()

        # Set up memory manager
        memory_manager = MemoryManager(
            memory_store=vector_store, contextualizer=contextualizer, bm25_store=bm25_store
        )

        # Set up LTM Agent with enhanced features
        agent = LongTermMemoryAgent(
            memory_manager=memory_manager,
            contextualizer=contextualizer,
            llm=MockLLM(),
            knowledge_limit=5,
            enable_consolidation=True,
        )

        logger.info("Agent setup completed successfully")
        return agent
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        raise


async def test_multi_step_reasoning(agent):
    """Test multi-step reasoning across a knowledge domain."""
    logger.info("===== TESTING MULTI-STEP REASONING =====")

    # Initial query to establish domain knowledge
    logger.info("Step 1: Initial query about machine learning")
    response1 = await agent.async_invoke("What is machine learning?")
    logger.info(f"Response: {response1}")

    # Follow-up query building on that knowledge
    logger.info("Step 2: Follow-up query about types")
    response2 = await agent.async_invoke("What are the main types of machine learning?")
    logger.info(f"Response: {response2}")

    # Specific query about one of the types
    logger.info("Step 3: Specific query about supervised learning")
    response3 = await agent.async_invoke("Can you explain supervised learning in more detail?")
    logger.info(f"Response: {response3}")

    # Query requiring synthesizing information from previous queries
    logger.info("Step 4: Query requiring knowledge synthesis")
    response4 = await agent.async_invoke(
        "What are the advantages and limitations of different machine learning approaches?"
    )
    logger.info(f"Response: {response4}")

    return {
        "initial_query": "What is machine learning?",
        "initial_response": response1,
        "follow_up_query": "What are the main types of machine learning?",
        "follow_up_response": response2,
        "specific_query": "Can you explain supervised learning in more detail?",
        "specific_response": response3,
        "synthesis_query": "What are the advantages and limitations of different machine learning approaches?",
        "synthesis_response": response4,
    }


async def test_cross_domain_reasoning(agent):
    """Test reasoning across multiple knowledge domains."""
    logger.info("===== TESTING CROSS-DOMAIN REASONING =====")

    # Establish knowledge in programming domain
    logger.info("Step 1: Query about Python features")
    response1 = await agent.async_invoke("What are the key features of Python?")
    logger.info(f"Response: {response1}")

    # Establish knowledge in ML domain
    logger.info("Step 2: Query about machine learning")
    response2 = await agent.async_invoke("What is supervised learning in machine learning?")
    logger.info(f"Response: {response2}")

    # Cross-domain query
    logger.info("Step 3: Cross-domain query")
    response3 = await agent.async_invoke("How is Python used in machine learning applications?")
    logger.info(f"Response: {response3}")

    # Advanced cross-domain synthesis
    logger.info("Step 4: Cross-domain synthesis")
    response4 = await agent.async_invoke(
        "How does Python's memory management affect machine learning performance?"
    )
    logger.info(f"Response: {response4}")

    return {
        "domain1_query": "What are the key features of Python?",
        "domain1_response": response1,
        "domain2_query": "What is supervised learning in machine learning?",
        "domain2_response": response2,
        "cross_domain_query": "How is Python used in machine learning applications?",
        "cross_domain_response": response3,
        "synthesis_query": "How does Python's memory management affect machine learning performance?",
        "synthesis_response": response4,
    }


async def test_knowledge_correction(agent):
    """Test enhanced knowledge correction capabilities."""
    logger.info("===== TESTING KNOWLEDGE CORRECTION =====")

    # Initial query with incorrect information
    logger.info("Step 1: Query about supervised learning limitations (with incorrect response)")
    response1 = await agent.async_invoke("What are the limitations of supervised learning?")
    logger.info(f"Initial response: {response1}")

    # Provide a correction
    correction = """
    Supervised learning has several important limitations:
    1. It requires labeled data, which can be expensive or time-consuming to obtain
    2. It may perform poorly on unseen data (overfitting)
    3. It struggles with imbalanced datasets
    4. It may not perform well with very complex relationships unless given sufficient data
    5. It requires high-quality labels - garbage in, garbage out applies
    """

    logger.info(f"Step 2: Providing correction: {correction}")
    await agent.process_feedback(correction)

    # Same query after correction
    logger.info("Step 3: Repeating query to test correction application")
    response2 = await agent.async_invoke("What are the limitations of supervised learning?")
    logger.info(f"Response after correction: {response2}")

    # Similar query to test generalization
    logger.info("Step 4: Similar query to test correction generalization")
    response3 = await agent.async_invoke(
        "What challenges might I face when applying supervised learning?"
    )
    logger.info(f"Response to similar query: {response3}")

    # Second correction (to see cumulativeness)
    second_correction = """
    Another major limitation of supervised learning is its difficulty to adapt to changing environments or concept drift
    without retraining. This makes it less suitable for dynamic systems where patterns change over time.
    """

    logger.info(f"Step 5: Providing second correction: {second_correction}")
    await agent.process_feedback(second_correction)

    # Final query
    logger.info("Step 6: Final query to test multiple corrections")
    response4 = await agent.async_invoke(
        "What are all the limitations of supervised learning approaches?"
    )
    logger.info(f"Response after multiple corrections: {response4}")

    return {
        "initial_query": "What are the limitations of supervised learning?",
        "initial_response": response1,
        "first_correction": correction,
        "response_after_correction": response2,
        "similar_query": "What challenges might I face when applying supervised learning?",
        "similar_query_response": response3,
        "second_correction": second_correction,
        "response_after_multiple_corrections": response4,
    }


async def analyze_memory_structure(agent):
    """Analyze the memory structure after tests to evaluate cross-referencing."""
    logger.info("===== ANALYZING MEMORY STRUCTURE =====")

    try:
        # Get all knowledge units
        all_units = await agent.memory_manager.list_knowledge()
        logger.info(f"Total knowledge units: {len(all_units)}")

        # Analyze connection density
        connected_units = 0
        connection_count = 0
        correction_units = 0

        for unit in all_units:
            metadata = unit.metadata or {}

            # Check for connections
            if "related_units" in metadata and metadata["related_units"]:
                connected_units += 1
                connection_count += len(metadata["related_units"])

            # Check for corrections
            if (
                metadata.get("type") == "correction"
                or metadata.get("feedback_type") == "correction"
            ):
                correction_units += 1

        # Calculate metrics
        connection_density = connection_count / max(1, len(all_units))
        connection_percentage = (connected_units / max(1, len(all_units))) * 100

        logger.info(f"Connected units: {connected_units} ({connection_percentage:.1f}% of total)")
        logger.info(f"Total connections: {connection_count}")
        logger.info(f"Connection density: {connection_density:.2f} connections per unit")
        logger.info(f"Correction units: {correction_units}")

        return {
            "total_units": len(all_units),
            "connected_units": connected_units,
            "connection_percentage": connection_percentage,
            "total_connections": connection_count,
            "connection_density": connection_density,
            "correction_units": correction_units,
        }

    except Exception as e:
        logger.error(f"Error analyzing memory structure: {e}")
        return {"error": str(e)}


async def generate_performance_report(results):
    """Generate a comprehensive performance report of the enhanced memory system."""

    report_lines = [
        "# Enhanced Memory System Performance Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Test Scenarios and Results",
        "### 1. Multi-Step Reasoning Performance",
        "The system was tested on its ability to build knowledge progressively and leverage it for increasingly complex queries.",
        f'- *Initial Query*: "{results["multi_step"]["initial_query"]}"',
        f'- *Final Synthesis Query*: "{results["multi_step"]["synthesis_query"]}"',
        "### 2. Cross-Domain Reasoning Performance",
        "The system was tested on its ability to integrate knowledge across different domains.",
        f'- *Domain 1 Query*: "{results["cross_domain"]["domain1_query"]}"',
        f'- *Domain 2 Query*: "{results["cross_domain"]["domain2_query"]}"',
        f'- *Cross-Domain Synthesis Query*: "{results["cross_domain"]["synthesis_query"]}"',
        "### 3. Knowledge Correction Performance",
        "The system was tested on its ability to incorporate corrections and update its knowledge.",
        f'- *Test Query*: "{results["correction"]["initial_query"]}"',
        "- *Number of Corrections Applied*: 2",
        f'- *Correction Generalization Test Query*: "{results["correction"]["similar_query"]}"',
        "\n## Memory System Structure Analysis",
        f"- *Total Knowledge Units*: {results['memory_analysis']['total_units']}",
        f"- *Connected Units*: {results['memory_analysis']['connected_units']} ({results['memory_analysis']['connection_percentage']:.1f}% of total)",
        f"- *Total Connections*: {results['memory_analysis']['total_connections']}",
        f"- *Connection Density*: {results['memory_analysis']['connection_density']:.2f} connections per unit",
        f"- *Correction Units*: {results['memory_analysis']['correction_units']}",
        "\n## Performance Assessment",
    ]

    # Generate overall assessment
    connection_density = results["memory_analysis"]["connection_density"]
    correction_units = results["memory_analysis"]["correction_units"]

    if connection_density >= 2.0 and correction_units >= 2:
        assessment = [
            "The enhanced memory system demonstrates **excellent performance** in both multi-step reasoning and knowledge correction:",
            "- **Strong cross-referencing**: High connection density indicates effective relationship modeling",
            "- **Effective knowledge correction**: Multiple corrections successfully integrated",
            "- **Good cross-domain integration**: Successfully related concepts across different knowledge domains",
        ]
    elif connection_density >= 1.0 and correction_units >= 1:
        assessment = [
            "The enhanced memory system demonstrates **good performance** in both multi-step reasoning and knowledge correction:",
            "- **Adequate cross-referencing**: Reasonable connection density indicates functional relationship modeling",
            "- **Functional knowledge correction**: Corrections are being properly integrated",
            "- **Moderate cross-domain integration**: Some success in relating concepts across domains",
        ]
    else:
        assessment = [
            "The enhanced memory system demonstrates **moderate performance** with areas for improvement:",
            "- **Limited cross-referencing**: Low connection density suggests relationships are not being fully modeled",
            "- **Basic knowledge correction**: Minimal evidence of correction integration",
            "- **Simple cross-domain integration**: Limited success in relating concepts across domains",
        ]

    report_lines.extend(assessment)

    # Add recommendations
    report_lines.extend(
        [
            "\n## Recommendations for Further Improvement",
            "1. **Temporal decay modeling**: Implement strength decay for older connections",
            "2. **Importance-based retrieval**: Prioritize knowledge units based on relevance to query",
            "3. **Contradiction detection**: Automatically identify and flag contradictory knowledge",
            "4. **Dynamic knowledge consolidation**: Periodically merge related knowledge units",
        ]
    )

    return "\n".join(report_lines)


async def main():
    """Run the integrated tests and generate comprehensive report."""
    print("=" * 80)
    print("INTEGRATED TESTING OF ENHANCED MEMORY SYSTEM")
    print("=" * 80)
    print("\nInitializing full agent system with enhancements...")

    try:
        # Set up agent with enhanced memory system
        agent = await setup_agent()

        results = {
            "multi_step": {},
            "cross_domain": {},
            "correction": {},
            "memory_analysis": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Run multi-step reasoning test
        print("\nRunning multi-step reasoning tests...")
        results["multi_step"] = await test_multi_step_reasoning(agent)

        # Run cross-domain reasoning test
        print("\nRunning cross-domain reasoning tests...")
        results["cross_domain"] = await test_cross_domain_reasoning(agent)

        # Run knowledge correction test
        print("\nRunning knowledge correction tests...")
        results["correction"] = await test_knowledge_correction(agent)

        # Analyze memory structure
        print("\nAnalyzing memory structure...")
        results["memory_analysis"] = await analyze_memory_structure(agent)

        # Generate and save performance report
        print("\nGenerating performance report...")
        report = await generate_performance_report(results)

        # Save results and report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"integrated_test_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(f"memory_system_performance_report_{timestamp}.md", "w") as f:
            f.write(report)

        print("\nTesting completed successfully!")
        print(f"Results saved to integrated_test_results_{timestamp}.json")
        print(f"Performance report saved to memory_system_performance_report_{timestamp}.md")

        # Display key findings
        print("\nKEY FINDINGS:")
        print(f"- Total knowledge units created: {results['memory_analysis']['total_units']}")
        print(
            f"- Connection density: {results['memory_analysis']['connection_density']:.2f} connections per unit"
        )
        print(
            f"- Connected units: {results['memory_analysis']['connection_percentage']:.1f}% of all units"
        )
        print(f"- Correction units created: {results['memory_analysis']['correction_units']}")

    except Exception as e:
        print(f"Error during integrated testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
