#!/usr/bin/env python
"""
Multi-Step Learning Evaluation for the Long-Term Memory Agent.

This script specifically focuses on evaluating the agent's ability to:
1. Learn information over multiple interactions
2. Consolidate related information from different interactions
3. Apply knowledge learned in earlier steps to later questions
4. Update its understanding when given corrections or additional context
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add the src directory to the Python path to make ltm_agent modules importable
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_dir)

# Now import the ltm_agent modules
from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="logs/evaluation/multi_step_eval.log",
    filemode="w",
)
logger = logging.getLogger("multi_step_evaluator")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


class MockLLM:
    """Mock LLM for testing that returns predefined responses based on query content."""

    def __init__(self):
        self.memory_manager = None
        self.responses = {
            "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            "machine learning": "Machine learning is a subset of AI that enables systems to learn from data and improve over time.",
            "neural network": "A neural network is a computational model inspired by the human brain's structure and function.",
            "transformer": "Transformer is a deep learning architecture that uses self-attention mechanisms and has revolutionized NLP.",
            "language model": "A language model is a statistical model that predicts the probability of a sequence of words.",
            "reinforcement learning": "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment through trial and error.",
        }

    def set_memory_manager(self, memory_manager):
        """Set the memory manager for context retrieval."""
        self.memory_manager = memory_manager

    async def agenerate(self, prompt, context=None):
        """Generate a response based on the prompt and context."""
        # First check if we have any knowledge units that match keywords
        if context:
            # Extract knowledge units from context
            context_lower = context.lower()

            # Use our predefined responses if keywords match
            for keyword, response in self.responses.items():
                if keyword in prompt.lower() or keyword in context_lower:
                    return response

        # Fallback response
        return "I don't have enough information to answer that question."


class MultiStepLearningEvaluation:
    """Evaluates the Long-Term Memory Agent's multi-step learning capabilities."""

    def __init__(
        self,
        results_path: str = "logs/evaluation/multi_step_results.json",
        enable_consolidation: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            results_path: Path to save evaluation results
            enable_consolidation: Whether to enable memory consolidation
        """
        self.results_path = results_path
        self.enable_consolidation = enable_consolidation
        self.agent = None
        self.memory_manager = None
        self.test_scenarios = self._define_test_scenarios()

    def _define_test_scenarios(self) -> list[dict[str, Any]]:
        """
        Define test scenarios for multi-step learning evaluation.

        Returns:
            List[Dict[str, Any]]: List of test scenarios
        """
        return [
            {
                "name": "progressive_learning_python",
                "description": "Learning about Python features across multiple interactions",
                "interactions": [
                    {
                        "step": 1,
                        "query": "What is Python used for?",
                        "expected_knowledge": ["Python", "programming language"],
                        "expected_response_contains": ["Python", "programming"],
                    },
                    {
                        "step": 2,
                        "query": "What are some popular Python libraries for data science?",
                        "expected_knowledge": ["Python", "library", "data science"],
                        "expected_response_contains": ["data science", "library"],
                        "feedback": "Python libraries like NumPy, Pandas, and SciPy are widely used for data science tasks. NumPy provides scientific computing capabilities, Pandas is used for data manipulation, and SciPy offers scientific and technical computing functionality.",
                    },
                    {
                        "step": 3,
                        "query": "How does NumPy relate to Python?",
                        "expected_knowledge": ["NumPy", "Python", "library"],
                        "expected_response_contains": ["NumPy", "Python", "library"],
                        "knowledge_connection": {
                            "source": "Python",
                            "target": "NumPy",
                            "relationship": "has_library",
                        },
                    },
                    {
                        "step": 4,
                        "query": "What are the key Python data science libraries we've discussed?",
                        "expected_knowledge": [
                            "NumPy",
                            "Pandas",
                            "SciPy",
                            "Python",
                            "data science",
                        ],
                        "expected_response_contains": ["NumPy", "Pandas", "SciPy"],
                        "tests_memory_consolidation": True,
                    },
                ],
            },
            {
                "name": "knowledge_correction",
                "description": "Learning and correcting information about machine learning",
                "interactions": [
                    {
                        "step": 1,
                        "query": "What is machine learning?",
                        "expected_knowledge": ["machine learning", "AI"],
                        "expected_response_contains": ["machine learning"],
                    },
                    {
                        "step": 2,
                        "query": "Is deep learning the same as machine learning?",
                        "expected_knowledge": ["deep learning", "machine learning"],
                        "expected_response_contains": ["deep learning", "machine learning"],
                        "feedback": "There's a misconception I want to clarify. Deep learning is actually a subset of machine learning that utilizes neural networks with multiple layers. It's a more advanced approach compared to traditional machine learning algorithms like decision trees or linear regression.",
                    },
                    {
                        "step": 3,
                        "query": "Explain the relationship between deep learning and machine learning.",
                        "expected_knowledge": [
                            "deep learning",
                            "subset",
                            "machine learning",
                            "neural networks",
                        ],
                        "expected_response_contains": ["subset", "neural networks"],
                        "tests_knowledge_correction": True,
                    },
                ],
            },
            {
                "name": "multi_hop_reasoning",
                "description": "Building connections across multiple knowledge pieces",
                "interactions": [
                    {
                        "step": 1,
                        "query": "What are transformers in machine learning?",
                        "expected_knowledge": ["transformer", "machine learning", "self-attention"],
                        "expected_response_contains": ["transformer", "self-attention"],
                    },
                    {
                        "step": 2,
                        "query": "How do language models work?",
                        "expected_knowledge": ["language model", "predict", "words"],
                        "expected_response_contains": ["language model", "predict"],
                    },
                    {
                        "step": 3,
                        "query": "What's the relationship between transformers and language models?",
                        "expected_knowledge": ["transformer", "language model"],
                        "expected_response_contains": ["transformer", "language model"],
                        "feedback": "Modern language models like GPT and BERT are based on transformer architectures. Transformers revolutionized NLP by enabling models to process text in parallel rather than sequentially, and their self-attention mechanism allows them to capture long-range dependencies in text.",
                    },
                    {
                        "step": 4,
                        "query": "Explain how transformers are used in modern language models.",
                        "expected_knowledge": [
                            "transformer",
                            "language model",
                            "GPT",
                            "BERT",
                            "self-attention",
                        ],
                        "expected_response_contains": ["GPT", "BERT", "self-attention"],
                        "tests_multi_hop_reasoning": True,
                    },
                ],
            },
        ]

    async def setup_agent(self) -> None:
        """
        Set up the memory manager and agent.
        """
        # Set up in-memory vector store
        vector_store = InMemoryVectorStore()

        # Set up memory manager with simple contextualizer
        self.memory_manager = MemoryManager(
            memory_store=vector_store, contextualizer=SimpleContextualizer()
        )

        # Initialize the memory manager
        await self.memory_manager.initialize()

        # Initialize a mock LLM
        llm = MockLLM()
        llm.set_memory_manager(self.memory_manager)

        # Initialize the agent
        self.agent = LongTermMemoryAgent(
            memory_manager=self.memory_manager,
            llm=llm,
            enable_consolidation=self.enable_consolidation,
            consolidation_interval=60,  # Set to 1 minute for testing
        )

    async def run_evaluation(self) -> dict[str, Any]:
        """
        Run the multi-step learning evaluation and calculate metrics.

        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting multi-step learning evaluation")

        # Set up the agent
        await self.setup_agent()

        # Results structure
        results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "consolidation_enabled": self.enable_consolidation,
            },
            "scenarios": {},
            "overall": {
                "progressive_learning_score": 0.0,
                "knowledge_correction_score": 0.0,
                "multi_hop_reasoning_score": 0.0,
                "cross_referencing_score": 0.0,
            },
        }

        # Run each test scenario
        for scenario in self.test_scenarios:
            scenario_name = scenario["name"]
            logger.info(f"Running scenario: {scenario_name}")

            scenario_results = await self._run_scenario(scenario)
            results["scenarios"][scenario_name] = scenario_results

        # Calculate overall metrics
        scenario_count = len(self.test_scenarios)
        results["overall"]["progressive_learning_score"] = (
            sum(
                results["scenarios"][s["name"]].get("progressive_learning_score", 0.0)
                for s in self.test_scenarios
            )
            / scenario_count
        )

        results["overall"]["knowledge_correction_score"] = sum(
            results["scenarios"][s["name"]].get("knowledge_correction_score", 0.0)
            for s in self.test_scenarios
            if any(i.get("tests_knowledge_correction") for i in s["interactions"])
        ) / len(
            [
                s
                for s in self.test_scenarios
                if any(i.get("tests_knowledge_correction") for i in s["interactions"])
            ]
        )

        results["overall"]["multi_hop_reasoning_score"] = sum(
            results["scenarios"][s["name"]].get("multi_hop_reasoning_score", 0.0)
            for s in self.test_scenarios
            if any(i.get("tests_multi_hop_reasoning") for i in s["interactions"])
        ) / len(
            [
                s
                for s in self.test_scenarios
                if any(i.get("tests_multi_hop_reasoning") for i in s["interactions"])
            ]
        )

        results["overall"]["cross_referencing_score"] = sum(
            results["scenarios"][s["name"]].get("cross_referencing_score", 0.0)
            for s in self.test_scenarios
            if any(i.get("knowledge_connection") for i in s["interactions"])
        ) / len(
            [
                s
                for s in self.test_scenarios
                if any(i.get("knowledge_connection") for i in s["interactions"])
            ]
        )

        # Save results
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Multi-step learning evaluation complete. Results saved to {self.results_path}"
        )

        return results

    async def _run_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """
        Run a single evaluation scenario.

        Args:
            scenario: The scenario configuration

        Returns:
            Dict[str, Any]: The scenario results
        """
        scenario_results = {
            "name": scenario["name"],
            "description": scenario["description"],
            "interactions": [],
            "progressive_learning_score": 0.0,
            "knowledge_correction_score": 0.0,
            "multi_hop_reasoning_score": 0.0,
            "cross_referencing_score": 0.0,
        }

        knowledge_units_before = {}

        # Process each interaction step
        for interaction in scenario["interactions"]:
            step = interaction["step"]
            query = interaction["query"]
            expected_knowledge = interaction.get("expected_knowledge", [])
            expected_response_contains = interaction.get("expected_response_contains", [])

            logger.info(f"Step {step}: Processing query: {query}")

            # Get knowledge units before this interaction
            units_before = await self.memory_manager.list_knowledge()

            # Get response from agent
            response = await self.agent.async_invoke(query)

            # Process feedback if provided
            if "feedback" in interaction:
                feedback = interaction["feedback"]
                logger.info(f"Step {step}: Processing feedback: {feedback}")
                await self.agent.async_process_feedback(feedback, query, response)

            # If there's a knowledge connection to test
            cross_referencing_score = 0.0
            if "knowledge_connection" in interaction:
                connection = interaction["knowledge_connection"]
                source_term = connection["source"]
                target_term = connection["target"]
                relationship = connection["relationship"]

                # Get knowledge units that might have this connection
                units_after = await self.memory_manager.list_knowledge()

                # Look for evidence of the connection
                cross_referencing_score = await self._evaluate_knowledge_connection(
                    source_term, target_term, relationship, units_after
                )

            # Evaluate knowledge correction if needed
            knowledge_correction_score = 0.0
            if interaction.get("tests_knowledge_correction", False):
                units_after = await self.memory_manager.list_knowledge()
                knowledge_correction_score = await self._evaluate_knowledge_correction(units_after)

            # Evaluate multi-hop reasoning if needed
            multi_hop_reasoning_score = 0.0
            if interaction.get("tests_multi_hop_reasoning", False):
                units_after = await self.memory_manager.list_knowledge()
                multi_hop_reasoning_score = await self._evaluate_multi_hop_reasoning(units_after)

            # Evaluate memory consolidation if needed
            memory_consolidation_score = 0.0
            if interaction.get("tests_memory_consolidation", False):
                # Wait a bit for consolidation to happen if enabled
                if self.enable_consolidation:
                    logger.info("Waiting for memory consolidation to occur...")
                    await asyncio.sleep(65)  # Just over consolidation_interval

                units_after = await self.memory_manager.list_knowledge()
                memory_consolidation_score = await self._evaluate_memory_consolidation(units_after)

            # Evaluate response content
            response_score = await self._evaluate_response_content(
                response, expected_response_contains
            )

            # Store interaction results
            interaction_result = {
                "step": step,
                "query": query,
                "response": response,
                "response_score": response_score,
                "expected_knowledge": expected_knowledge,
                "expected_response_contains": expected_response_contains,
            }

            # Add specific evaluation scores if applicable
            if "knowledge_connection" in interaction:
                interaction_result["cross_referencing_score"] = cross_referencing_score
                scenario_results["cross_referencing_score"] = cross_referencing_score

            if interaction.get("tests_knowledge_correction", False):
                interaction_result["knowledge_correction_score"] = knowledge_correction_score
                scenario_results["knowledge_correction_score"] = knowledge_correction_score

            if interaction.get("tests_multi_hop_reasoning", False):
                interaction_result["multi_hop_reasoning_score"] = multi_hop_reasoning_score
                scenario_results["multi_hop_reasoning_score"] = multi_hop_reasoning_score

            if interaction.get("tests_memory_consolidation", False):
                interaction_result["memory_consolidation_score"] = memory_consolidation_score

            scenario_results["interactions"].append(interaction_result)

        # Calculate progressive learning score based on all interactions
        response_scores = [i["response_score"] for i in scenario_results["interactions"]]
        scenario_results["progressive_learning_score"] = (
            sum(response_scores) / len(response_scores) if response_scores else 0.0
        )

        return scenario_results

    async def _evaluate_response_content(
        self, response: str, expected_elements: list[str]
    ) -> float:
        """
        Evaluate if the response contains expected elements.

        Args:
            response: The agent's response
            expected_elements: Elements expected in the response

        Returns:
            float: Score from 0.0 to 1.0
        """
        if not expected_elements:
            return 1.0  # No expectations means perfect score

        response_lower = response.lower()
        matches = sum(1 for element in expected_elements if element.lower() in response_lower)
        return matches / len(expected_elements)

    async def _evaluate_knowledge_connection(
        self,
        source_term: str,
        target_term: str,
        relationship: str,
        knowledge_units: list[KnowledgeUnit],
    ) -> float:
        """
        Evaluate if a knowledge connection exists between two terms.

        Args:
            source_term: Source term
            target_term: Target term
            relationship: Expected relationship
            knowledge_units: List of knowledge units

        Returns:
            float: Score from 0.0 to 1.0 indicating connection strength
        """
        # Check if any knowledge unit contains both terms
        connection_strength = 0.0
        max_strength = 0.0

        for unit in knowledge_units:
            content = unit.original_chunk.lower()
            metadata = unit.metadata or {}

            # Check if both terms are in the content
            contains_source = source_term.lower() in content
            contains_target = target_term.lower() in content

            # Check if there's metadata indicating a relationship
            has_relationship = False
            if metadata.get("related_units", []) or metadata.get("relationships", []):
                has_relationship = True

            # Calculate strength score
            if contains_source and contains_target:
                strength = 0.5  # Base score for containing both terms
                if has_relationship:
                    strength = 1.0  # Full score if relationship is also indicated

                if strength > max_strength:
                    max_strength = strength

        return max_strength

    async def _evaluate_knowledge_correction(self, knowledge_units: list[KnowledgeUnit]) -> float:
        """
        Evaluate if knowledge has been corrected appropriately.

        Args:
            knowledge_units: List of knowledge units

        Returns:
            float: Score from 0.0 to 1.0
        """
        # Look for knowledge units that mention deep learning as a subset of machine learning
        correction_strength = 0.0

        for unit in knowledge_units:
            content = unit.original_chunk.lower()

            # Check for correction of deep learning being a subset of machine learning
            if "deep learning" in content and "machine learning" in content and "subset" in content:
                correction_strength = 1.0
                break

        return correction_strength

    async def _evaluate_multi_hop_reasoning(self, knowledge_units: list[KnowledgeUnit]) -> float:
        """
        Evaluate if multi-hop reasoning connections are established.

        Args:
            knowledge_units: List of knowledge units

        Returns:
            float: Score from 0.0 to 1.0
        """
        # Look for knowledge units that connect transformers, GPT/BERT, and language models
        reasoning_strength = 0.0

        for unit in knowledge_units:
            content = unit.original_chunk.lower()

            # Check for connections between transformer, language models, and specific models
            has_transformer = "transformer" in content
            has_language_model = "language model" in content
            has_specific_model = "gpt" in content or "bert" in content

            if has_transformer and has_language_model and has_specific_model:
                reasoning_strength = 1.0
                break

        return reasoning_strength

    async def _evaluate_memory_consolidation(self, knowledge_units: list[KnowledgeUnit]) -> float:
        """
        Evaluate if memory consolidation has occurred appropriately.

        Args:
            knowledge_units: List of knowledge units

        Returns:
            float: Score from 0.0 to 1.0
        """
        # Check if knowledge about Python data science libraries has been consolidated
        python_units_with_libraries = []

        for unit in knowledge_units:
            content = unit.original_chunk.lower()

            # Check for Python and multiple libraries in the same unit
            has_python = "python" in content
            library_count = sum(1 for lib in ["numpy", "pandas", "scipy"] if lib in content)

            if has_python and library_count >= 2:
                python_units_with_libraries.append(unit)

        # Scoring based on consolidation evidence
        if python_units_with_libraries:
            return 1.0
        else:
            return 0.0


# Main execution
async def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-step learning capabilities")
    parser.add_argument("--consolidation", action="store_true", help="Enable memory consolidation")
    parser.add_argument(
        "--results",
        type=str,
        default="logs/evaluation/multi_step_results.json",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    evaluator = MultiStepLearningEvaluation(
        results_path=args.results, enable_consolidation=args.consolidation
    )
    results = await evaluator.run_evaluation()

    print("\nEvaluation Summary:")
    print(f"Progressive Learning Score: {results['overall']['progressive_learning_score']:.2f}")
    print(f"Knowledge Correction Score: {results['overall']['knowledge_correction_score']:.2f}")
    print(f"Multi-Hop Reasoning Score: {results['overall']['multi_hop_reasoning_score']:.2f}")
    print(f"Cross-Referencing Score: {results['overall']['cross_referencing_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
