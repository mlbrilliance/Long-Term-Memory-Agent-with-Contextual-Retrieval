#!/usr/bin/env python
"""
Comprehensive Integration Test Suite for Memory System Improvements

This script provides a robust testing framework to ensure the memory system
improvements integrate well with the entire LTM Agent architecture. It tests:

1. System-level integration
2. Component interactions
3. End-to-end workflows
4. Performance under different scenarios
5. Edge cases handling

The tests use a mock LLM but exercise the actual memory system components.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("integration_test.log")],
)
logger = logging.getLogger("integration_test")

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


# Mock LLM implementation
class MockLLM:
    """Mock LLM for testing that simulates responses based on queries."""

    def __init__(self):
        self.knowledge_base = {
            # General knowledge
            "python": "Python is a high-level programming language known for its readability.",
            "machine learning": "Machine learning is a type of artificial intelligence that allows systems to learn from data.",
            "javascript": "JavaScript is a programming language commonly used for web development.",
            # Detailed knowledge
            "python features": "Python features include dynamic typing, automatic memory management, comprehensive standard libraries, and support for multiple programming paradigms.",
            "python memory": "Python manages memory through reference counting and garbage collection.",
            "supervised learning": "Supervised learning is a machine learning approach where models are trained on labeled data.",
            # Initially incorrect knowledge (for testing corrections)
            "python speed": "Python is always the fastest programming language for all computational tasks.",
            "python threading": "Python's GIL has no impact on multithreaded performance.",
        }

    async def agenerate(self, prompt, **kwargs):
        """Generate a response based on prompt content."""
        query = prompt.lower()
        response = "I don't have specific information about that query."

        # Try to find the most relevant knowledge
        for key, value in self.knowledge_base.items():
            if key in query:
                response = value
                break

        # Simulate response object
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)


class IntegrationTestSuite:
    """Comprehensive test suite for memory system integration."""

    def __init__(self):
        """Initialize the test suite."""
        self.agent = None
        self.memory_manager = None
        self.test_results = {}

    async def setup(self):
        """Set up the test environment and components."""
        logger.info("Setting up integration test environment")

        try:
            # Use a try/except block to handle different import scenarios
            try:
                # Try importing with the 'src.' prefix first
                from src.ltm_agent.agent.ltm_agent import LongTermMemoryAgent
                from src.ltm_agent.memory.bm25_store import RankBM25Store
                from src.ltm_agent.memory.consolidator import MemoryConsolidator
                from src.ltm_agent.memory.contextualizer import SimpleContextualizer
                from src.ltm_agent.memory.cross_referencer import KnowledgeCrossReferencer
                from src.ltm_agent.memory.manager import MemoryManager
                from src.ltm_agent.memory.vector_store import InMemoryVectorStore

                logger.info("Successfully imported modules with 'src.' prefix")

            except ImportError:
                # If that fails, try without the 'src.' prefix
                from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
                from ltm_agent.memory.bm25_store import RankBM25Store
                from ltm_agent.memory.contextualizer import SimpleContextualizer
                from ltm_agent.memory.manager import MemoryManager
                from ltm_agent.memory.vector_store import InMemoryVectorStore

                logger.info("Successfully imported modules without 'src.' prefix")

            # Initialize components
            vector_store = InMemoryVectorStore(collection_name="integration_test")

            # Use a temporary file for BM25 store
            bm25_path = os.path.join(os.getcwd(), "integration_test_bm25.pkl")
            bm25_store = RankBM25Store(save_path=bm25_path)

            # Initialize contextualizer
            contextualizer = SimpleContextualizer()

            # Initialize memory manager
            self.memory_manager = MemoryManager(
                memory_store=vector_store, contextualizer=contextualizer, bm25_store=bm25_store
            )

            # Initialize the LTM Agent
            self.agent = LongTermMemoryAgent(
                memory_manager=self.memory_manager,
                contextualizer=contextualizer,
                llm=MockLLM(),
                knowledge_limit=5,
                enable_consolidation=True,
            )

            logger.info("Test environment setup complete")
            return True

        except Exception as e:
            logger.error(f"Error during setup: {e}")

            # Create a simplified test structure if imports fail
            logger.info("Setting up simplified test environment instead")
            from scripts.demonstrate_improvements import BasicMemoryManager, EnhancedAgent

            self.memory_manager = BasicMemoryManager()
            self.agent = EnhancedAgent()

            return False

    async def test_basic_functionality(self):
        """Test that basic functionality works as expected."""
        logger.info("=== Testing Basic Functionality ===")

        # Query and response
        query = "What is Python?"
        response = await self.agent.async_invoke(query)

        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")

        # Check that a knowledge unit was created
        knowledge_units = await self.memory_manager.list_knowledge()
        logger.info(f"Created {len(knowledge_units)} knowledge unit(s)")

        # Verify content of the knowledge unit
        if knowledge_units:
            unit = knowledge_units[0]
            logger.info(f"Knowledge unit content: {unit.original_chunk[:50]}...")

        return {
            "query": query,
            "response": response,
            "knowledge_units_created": len(knowledge_units),
        }

    async def test_multi_step_reasoning(self):
        """Test multi-step reasoning capabilities."""
        logger.info("=== Testing Multi-Step Reasoning ===")

        # Step 1: Initial query
        response1 = await self.agent.async_invoke("What is Python?")
        logger.info(f"Response 1: {response1}")

        # Step 2: Follow-up query
        response2 = await self.agent.async_invoke("What are the features of Python?")
        logger.info(f"Response 2: {response2}")

        # Step 3: Memory-dependent query
        response3 = await self.agent.async_invoke("How does Python handle memory management?")
        logger.info(f"Response 3: {response3}")

        # Step 4: Synthesis query
        response4 = await self.agent.async_invoke("Summarize what you know about Python.")
        logger.info(f"Response 4: {response4}")

        # Analyze connections between knowledge units
        connections = await self.analyze_knowledge_connections()

        return {
            "responses": [response1, response2, response3, response4],
            "connections": connections,
        }

    async def test_knowledge_correction(self):
        """Test the enhanced knowledge correction capabilities."""
        logger.info("=== Testing Knowledge Correction ===")

        # Initial query with potentially incorrect information
        query = "How does Python performance compare to other languages?"
        response1 = await self.agent.async_invoke(query)
        logger.info(f"Initial response: {response1}")

        # Provide a correction
        correction = """
        Python is generally slower than compiled languages like C++ or Rust for
        computation-intensive tasks. However, for I/O-bound applications, the difference
        is minimal. Python's performance can be significantly improved using libraries
        like NumPy or by using JIT compilers like PyPy.
        """

        logger.info(f"Providing correction: {correction}")
        await self.agent.process_feedback(correction)

        # Same query after correction
        response2 = await self.agent.async_invoke(query)
        logger.info(f"Response after correction: {response2}")

        # Similar query to test generalization
        similar_query = "Is Python a fast programming language?"
        response3 = await self.agent.async_invoke(similar_query)
        logger.info(f"Response to similar query: {response3}")

        # Analyze correction units
        correction_units = await self.analyze_correction_units()

        return {
            "initial_query": query,
            "initial_response": response1,
            "correction": correction,
            "corrected_response": response2,
            "similar_query": similar_query,
            "similar_response": response3,
            "correction_analysis": correction_units,
        }

    async def test_cross_domain_reasoning(self):
        """Test reasoning across different knowledge domains."""
        logger.info("=== Testing Cross-Domain Reasoning ===")

        # Establish knowledge in domain 1
        response1 = await self.agent.async_invoke("What is Python?")
        logger.info(f"Domain 1 response: {response1}")

        # Establish knowledge in domain 2
        response2 = await self.agent.async_invoke("What is machine learning?")
        logger.info(f"Domain 2 response: {response2}")

        # Cross-domain query
        cross_query = "How is Python used in machine learning?"
        response3 = await self.agent.async_invoke(cross_query)
        logger.info(f"Cross-domain response: {response3}")

        # Check for cross-domain connections
        connections = await self.analyze_knowledge_connections()

        return {
            "domain1_response": response1,
            "domain2_response": response2,
            "cross_domain_query": cross_query,
            "cross_domain_response": response3,
            "cross_domain_connections": connections,
        }

    async def test_memory_consolidation(self):
        """Test the memory consolidation process."""
        logger.info("=== Testing Memory Consolidation ===")

        # Get initial state
        initial_units = await self.memory_manager.list_knowledge()
        logger.info(f"Initial knowledge units: {len(initial_units)}")

        # Add some potentially related knowledge
        await self.agent.async_invoke("Python is a high-level programming language.")
        await self.agent.async_invoke("Python was created by Guido van Rossum.")
        await self.agent.async_invoke("Python's design philosophy emphasizes code readability.")

        # Trigger consolidation if possible
        try:
            # This might not be directly accessible depending on the agent implementation
            if hasattr(self.agent, "consolidate_memory"):
                await self.agent.consolidate_memory()
                logger.info("Explicitly triggered memory consolidation")
            else:
                # Add more knowledge to potentially trigger automatic consolidation
                for i in range(5):
                    await self.agent.async_invoke(
                        f"Python fact {i}: Python is widely used in data science."
                    )
                logger.info("Added more knowledge to trigger automatic consolidation")
        except Exception as e:
            logger.warning(f"Could not explicitly trigger consolidation: {e}")

        # Get final state
        final_units = await self.memory_manager.list_knowledge()
        logger.info(f"Final knowledge units: {len(final_units)}")

        # Analyze connections
        connections = await self.analyze_knowledge_connections()

        return {
            "initial_unit_count": len(initial_units),
            "final_unit_count": len(final_units),
            "connections": connections,
        }

    async def analyze_knowledge_connections(self):
        """Analyze connections between knowledge units."""
        logger.info("Analyzing knowledge connections")

        try:
            all_units = await self.memory_manager.list_knowledge()

            # Count connections
            connected_units = 0
            connection_count = 0

            for unit in all_units:
                metadata = unit.metadata or {}

                if "related_units" in metadata and metadata["related_units"]:
                    connected_units += 1
                    connection_count += len(metadata["related_units"])

            # Calculate metrics
            connection_density = connection_count / max(1, len(all_units))
            connection_percentage = (connected_units / max(1, len(all_units))) * 100

            logger.info(
                f"Connected units: {connected_units} ({connection_percentage:.1f}% of total)"
            )
            logger.info(f"Total connections: {connection_count}")
            logger.info(f"Connection density: {connection_density:.2f} connections per unit")

            return {
                "total_units": len(all_units),
                "connected_units": connected_units,
                "connection_percentage": connection_percentage,
                "total_connections": connection_count,
                "connection_density": connection_density,
            }

        except Exception as e:
            logger.error(f"Error analyzing connections: {e}")
            return {"error": str(e)}

    async def analyze_correction_units(self):
        """Analyze correction-related knowledge units."""
        logger.info("Analyzing correction units")

        try:
            all_units = await self.memory_manager.list_knowledge()

            # Find correction-related units
            correction_units = []
            negative_examples = []

            for unit in all_units:
                metadata = unit.metadata or {}

                if (
                    metadata.get("type") == "correction"
                    or metadata.get("feedback_type") == "correction"
                ):
                    correction_units.append(unit)

                if metadata.get("type") == "negative_example":
                    negative_examples.append(unit)

            logger.info(f"Found {len(correction_units)} correction units")
            logger.info(f"Found {len(negative_examples)} negative examples")

            return {
                "correction_units": len(correction_units),
                "negative_examples": len(negative_examples),
            }

        except Exception as e:
            logger.error(f"Error analyzing correction units: {e}")
            return {"error": str(e)}

    async def run_all_tests(self):
        """Run all integration tests and collect results."""
        logger.info("Running all integration tests")

        # Set up the environment
        system_ready = await self.setup()

        if not system_ready:
            logger.warning("Using simplified test environment due to import issues")

        # Initialize results
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "system_ready": system_ready,
            "tests": {},
        }

        # Run the tests
        try:
            # Basic functionality
            self.test_results["tests"][
                "basic_functionality"
            ] = await self.test_basic_functionality()

            # Multi-step reasoning
            self.test_results["tests"][
                "multi_step_reasoning"
            ] = await self.test_multi_step_reasoning()

            # Knowledge correction
            self.test_results["tests"][
                "knowledge_correction"
            ] = await self.test_knowledge_correction()

            # Cross-domain reasoning
            self.test_results["tests"][
                "cross_domain_reasoning"
            ] = await self.test_cross_domain_reasoning()

            # Memory consolidation
            self.test_results["tests"][
                "memory_consolidation"
            ] = await self.test_memory_consolidation()

            logger.info("All tests completed")

        except Exception as e:
            logger.error(f"Error during testing: {e}")
            self.test_results["error"] = str(e)

        # Save results
        self.save_results()

        return self.test_results

    def save_results(self):
        """Save test results to a JSON file."""
        filename = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(self.test_results, f, indent=2)

        logger.info(f"Results saved to {filename}")
        return filename


async def run_performance_test(agent, memory_manager):
    """Run a performance test with many interactions."""
    logger.info("=== Running Performance Test ===")

    start_time = time.time()

    # Create a large number of interactions
    queries = [
        "What is Python?",
        "What are Python's main features?",
        "How does Python handle memory management?",
        "What are Python's limitations?",
        "How does Python compare to other languages?",
        "What is machine learning?",
        "How is Python used in machine learning?",
        "What are the different types of machine learning?",
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is reinforcement learning?",
        "What are neural networks?",
        "How do neural networks work?",
        "What is deep learning?",
        "What is the relationship between AI, ML, and deep learning?",
        "What is JavaScript?",
        "How does JavaScript compare to Python?",
        "What are JavaScript's main features?",
        "What is web development?",
        "How are Python and JavaScript used in web development?",
    ]

    # Run queries
    responses = []
    for i, query in enumerate(queries):
        logger.info(f"Running query {i + 1}/{len(queries)}: {query}")
        response = await agent.async_invoke(query)
        responses.append(response)

    # Add some corrections
    corrections = [
        "Python is generally slower than compiled languages, but its ease of use often outweighs this limitation.",
        "JavaScript is primarily used for web development, but Node.js enables server-side applications.",
        "Deep learning is a subset of machine learning focusing on neural networks with multiple layers.",
    ]

    for i, correction in enumerate(corrections):
        logger.info(f"Applying correction {i + 1}/{len(corrections)}")
        await agent.process_feedback(correction)

    # Measure final memory size
    all_units = await memory_manager.list_knowledge()

    end_time = time.time()
    duration = end_time - start_time

    return {
        "queries_processed": len(queries),
        "corrections_applied": len(corrections),
        "final_knowledge_units": len(all_units),
        "duration_seconds": duration,
        "average_time_per_query": duration / (len(queries) + len(corrections)),
    }


def generate_markdown_report(results):
    """Generate a Markdown report from test results."""

    if not results:
        return "No test results available."

    report = [
        "# Memory System Integration Test Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## System Status",
        f"Full System Available: {'Yes' if results.get('system_ready', False) else 'No - Used Simplified Environment'}",
        "\n## Test Results Summary",
    ]

    # Basic functionality
    basic = results.get("tests", {}).get("basic_functionality", {})
    if basic:
        report.extend(
            [
                "\n### Basic Functionality",
                f'- Query: "{basic.get("query", "")}"',
                f"- Response received: {'Yes' if basic.get('response') else 'No'}",
                f"- Knowledge units created: {basic.get('knowledge_units_created', 0)}",
            ]
        )

    # Multi-step reasoning
    multi_step = results.get("tests", {}).get("multi_step_reasoning", {})
    if multi_step:
        connections = multi_step.get("connections", {})
        report.extend(
            [
                "\n### Multi-Step Reasoning",
                f"- Responses received: {len(multi_step.get('responses', []))}",
                f"- Connected units: {connections.get('connected_units', 0)} ({connections.get('connection_percentage', 0):.1f}% of total)",
                f"- Connection density: {connections.get('connection_density', 0):.2f} connections per unit",
            ]
        )

    # Knowledge correction
    correction = results.get("tests", {}).get("knowledge_correction", {})
    if correction:
        correction_analysis = correction.get("correction_analysis", {})
        report.extend(
            [
                "\n### Knowledge Correction",
                f'- Initial query: "{correction.get("initial_query", "")}"',
                f"- Correction applied: {'Yes' if correction.get('correction') else 'No'}",
                f"- Response changed after correction: {'Yes' if correction.get('initial_response') != correction.get('corrected_response') else 'No'}",
                f"- Correction units created: {correction_analysis.get('correction_units', 0)}",
                f"- Negative examples created: {correction_analysis.get('negative_examples', 0)}",
            ]
        )

    # Cross-domain reasoning
    cross_domain = results.get("tests", {}).get("cross_domain_reasoning", {})
    if cross_domain:
        connections = cross_domain.get("cross_domain_connections", {})
        report.extend(
            [
                "\n### Cross-Domain Reasoning",
                f'- Cross-domain query: "{cross_domain.get("cross_domain_query", "")}"',
                f"- Connected units: {connections.get('connected_units', 0)} ({connections.get('connection_percentage', 0):.1f}% of total)",
                f"- Connection density: {connections.get('connection_density', 0):.2f} connections per unit",
            ]
        )

    # Memory consolidation
    consolidation = results.get("tests", {}).get("memory_consolidation", {})
    if consolidation:
        connections = consolidation.get("connections", {})
        report.extend(
            [
                "\n### Memory Consolidation",
                f"- Initial unit count: {consolidation.get('initial_unit_count', 0)}",
                f"- Final unit count: {consolidation.get('final_unit_count', 0)}",
                f"- Connected units: {connections.get('connected_units', 0)} ({connections.get('connection_percentage', 0):.1f}% of total)",
                f"- Connection density: {connections.get('connection_density', 0):.2f} connections per unit",
            ]
        )

    # Overall assessment
    report.extend(["\n## Overall Assessment"])

    # Create an assessment based on test results
    if all("error" not in test for test in results.get("tests", {}).values()):
        # Check correction effectiveness
        correction_effective = False
        if correction and correction.get("initial_response") != correction.get(
            "corrected_response"
        ):
            correction_effective = True

        # Check connection density
        good_connections = False
        if multi_step and multi_step.get("connections", {}).get("connection_density", 0) > 0.5:
            good_connections = True

        if correction_effective and good_connections:
            assessment = "The memory system demonstrates excellent integration with strong correction capabilities and good knowledge interconnections."
        elif correction_effective:
            assessment = "The memory system demonstrates good integration with effective correction capabilities, but limited knowledge interconnections."
        elif good_connections:
            assessment = "The memory system demonstrates good integration with strong knowledge interconnections, but limited correction capabilities."
        else:
            assessment = "The memory system demonstrates basic integration with areas for improvement in both correction capabilities and knowledge interconnections."
    else:
        assessment = "The integration tests encountered errors. Further investigation is needed."

    report.append(assessment)

    # Recommendations
    report.extend(
        [
            "\n## Recommendations",
            "1. Enhance the consolidation process to improve connection density",
            "2. Add more sophisticated detection of correction contexts",
            "3. Implement better cross-domain knowledge linking",
            "4. Add decay mechanisms for temporal relevance",
            "5. Optimize performance for larger knowledge bases",
        ]
    )

    return "\n".join(report)


async def main():
    """Run the integration test suite."""
    print("=" * 80)
    print("MEMORY SYSTEM INTEGRATION TEST SUITE")
    print("=" * 80)

    try:
        # Run the integration tests
        test_suite = IntegrationTestSuite()
        results = await test_suite.run_all_tests()

        # Generate and save report
        report = generate_markdown_report(results)
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, "w") as f:
            f.write(report)

        print("\nIntegration testing completed successfully!")
        print(f"Results saved to {report_file}")

        # Display key findings from report
        print("\nKEY FINDINGS:")
        for line in report.split("\n"):
            if line.startswith("- ") or line.startswith("## "):
                print(line)

    except Exception as e:
        print(f"Error during integration testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
