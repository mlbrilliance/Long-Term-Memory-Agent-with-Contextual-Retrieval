"""
Conversation Simulator for LTM Agent Evaluation.

This script simulates realistic multi-turn conversations to evaluate
how the agent's memory evolves and improves through natural dialogue patterns.
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.manager import MemoryManager

# Configure logging
log_dir = Path(__file__).parent.parent / "logs" / "evaluation"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "conversation_simulator.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("conversation_simulator")


class MockLLM:
    """Mock LLM for testing that provides simulated responses."""

    def __init__(self):
        """Initialize the mock LLM with basic domain knowledge."""
        self.memory_manager = None
        self.domain_knowledge = {
            "programming": {
                "python": "Python is a high-level programming language known for its simplicity and readability. It has a large ecosystem of libraries.",
                "javascript": "JavaScript is a scripting language primarily used for web development. It enables interactive web pages and is an essential part of web applications.",
                "libraries": "Programming languages often have libraries of pre-written code that developers can use to simplify common tasks.",
                "frameworks": "Frameworks provide structure and functionality for software development, often tailored to specific languages or purposes.",
            },
            "machine_learning": {
                "basics": "Machine learning is a field of AI focused on creating systems that learn from data rather than following explicit instructions.",
                "algorithms": "Machine learning algorithms include supervised learning (using labeled training data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).",
                "neural_networks": "Neural networks are machine learning systems inspired by the human brain, consisting of layers of interconnected nodes that process data.",
                "training": "Training a machine learning model involves feeding it data so it can learn patterns and make predictions or decisions.",
            },
            "data_science": {
                "basics": "Data science combines statistics, math, programming, and domain expertise to extract insights from data.",
                "analysis": "Data analysis involves examining, cleaning, transforming, and modeling data to discover useful information.",
                "visualization": "Data visualization uses graphical representations to communicate patterns, trends, and insights in data.",
                "big_data": "Big data refers to extremely large datasets that may be analyzed to reveal patterns and trends, especially relating to human behavior and interactions.",
            },
        }

    def set_memory_manager(self, memory_manager):
        """Set the memory manager to access context."""
        self.memory_manager = memory_manager

    async def agenerate(self, prompt, context=None):
        """Generate a response based on prompt and available context."""
        prompt_lower = prompt.lower()

        # First try to use knowledge from the memory context if available
        if context and len(context) > 20:
            # Prioritize context if it seems relevant
            relevance_score = self._calculate_relevance(prompt_lower, context.lower())
            if relevance_score > 0.5:
                return f"Based on what I know: {context[:300]}..."

        # If no relevant context, use domain knowledge
        for domain, topics in self.domain_knowledge.items():
            if domain.lower() in prompt_lower:
                for topic, knowledge in topics.items():
                    if topic.lower() in prompt_lower:
                        return knowledge

        # For cross-domain questions, try to combine knowledge
        domains_in_prompt = [
            domain for domain in self.domain_knowledge.keys() if domain.lower() in prompt_lower
        ]

        if len(domains_in_prompt) > 1:
            combined_response = "This question spans multiple domains. "
            for domain in domains_in_prompt[:2]:  # Just use first two domains to keep it simple
                relevant_topics = [
                    topic
                    for topic in self.domain_knowledge[domain].keys()
                    if topic.lower() in prompt_lower
                ]
                if relevant_topics:
                    topic = relevant_topics[0]
                    combined_response += f"Regarding {domain}/{topic}: {self.domain_knowledge[domain][topic][:100]}... "

            return combined_response

        # Default response if nothing specific matches
        return "I don't have specific information about that topic yet."

    def _calculate_relevance(self, query, context):
        """
        Calculate simple relevance between query and context based on keyword matching.
        Returns a score between 0.0 and 1.0.
        """
        query_words = set(query.split())
        context_words = set(context.split())

        # Count matching words
        matches = query_words.intersection(context_words)

        # Calculate Jaccard similarity
        if not query_words or not context_words:
            return 0.0

        return len(matches) / len(query_words)


class ConversationScenario:
    """Defines a multi-turn conversation scenario for evaluation."""

    def __init__(self, name: str, description: str, domain: str):
        """Initialize a conversation scenario."""
        self.name = name
        self.description = description
        self.domain = domain
        self.turns = []
        self.expected_outcomes = {}

    def add_turn(self, role: str, content: str, turn_id: str | None = None):
        """
        Add a conversation turn.

        Args:
            role: Either 'user' or 'agent'
            content: The message content
            turn_id: Optional unique ID for the turn (will be generated if not provided)
        """
        if turn_id is None:
            turn_id = str(uuid.uuid4())

        self.turns.append({"id": turn_id, "role": role, "content": content})

        return turn_id

    def add_expected_outcome(
        self, outcome_type: str, description: str, evaluation_criteria: dict[str, Any]
    ):
        """
        Add an expected outcome for the conversation.

        Args:
            outcome_type: Type of outcome (e.g., 'knowledge_acquisition', 'connections')
            description: Description of the expected outcome
            evaluation_criteria: Criteria to evaluate if the outcome was achieved
        """
        self.expected_outcomes[outcome_type] = {
            "description": description,
            "criteria": evaluation_criteria,
        }

    def to_dict(self):
        """Convert the scenario to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "turns": self.turns,
            "expected_outcomes": self.expected_outcomes,
        }


class ConversationSimulator:
    """Simulates realistic conversations to test the LTM agent."""

    def __init__(self, enable_consolidation: bool = True, results_dir: str | None = None):
        """Initialize the conversation simulator."""
        self.enable_consolidation = enable_consolidation

        if results_dir is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.results_dir = (
                Path(__file__).parent.parent
                / "logs"
                / "evaluation"
                / f"conversation_results_{timestamp}"
            )
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios = self._define_scenarios()
        self.memory_manager = None
        self.agent = None

    def _define_scenarios(self) -> dict[str, ConversationScenario]:
        """Define conversation scenarios for testing."""
        scenarios = {}

        # Scenario 1: Programming Language Discussion
        prog_scenario = ConversationScenario(
            name="programming_languages",
            description="A conversation about programming languages focusing on Python",
            domain="programming",
        )

        # Add turns for the programming scenario
        prog_scenario.add_turn("user", "What can you tell me about Python?")
        prog_scenario.add_turn("agent", "<will be filled by agent>")
        prog_scenario.add_turn("user", "What are some popular Python libraries?")
        prog_scenario.add_turn("agent", "<will be filled by agent>")
        prog_scenario.add_turn(
            "user", "I'm specifically interested in data science libraries in Python."
        )
        prog_scenario.add_turn("agent", "<will be filled by agent>")
        prog_scenario.add_turn(
            "user", "How do these libraries compare to libraries in other languages like R?"
        )
        prog_scenario.add_turn("agent", "<will be filled by agent>")
        prog_scenario.add_turn(
            "user",
            "Thanks for that information. Now, could you suggest a learning path for someone new to Python data science?",
        )
        prog_scenario.add_turn("agent", "<will be filled by agent>")

        # Expected outcomes
        prog_scenario.add_expected_outcome(
            "knowledge_acquisition",
            "The agent should acquire and connect knowledge about Python and its libraries",
            {
                "topics": ["python", "libraries", "data science"],
                "connections": ["python-libraries", "python-data-science"],
            },
        )
        prog_scenario.add_expected_outcome(
            "conversation_coherence",
            "The agent should maintain context across the conversation",
            {"maintains_topic": True, "references_previous_turns": True},
        )

        scenarios["programming_languages"] = prog_scenario

        # Scenario 2: Machine Learning Concepts
        ml_scenario = ConversationScenario(
            name="machine_learning_concepts",
            description="A conversation introducing and exploring machine learning concepts",
            domain="machine_learning",
        )

        # Add turns for the machine learning scenario
        ml_scenario.add_turn("user", "I'm new to machine learning. Can you explain the basics?")
        ml_scenario.add_turn("agent", "<will be filled by agent>")
        ml_scenario.add_turn("user", "What are the main types of machine learning algorithms?")
        ml_scenario.add_turn("agent", "<will be filled by agent>")
        ml_scenario.add_turn("user", "Tell me more about neural networks.")
        ml_scenario.add_turn("agent", "<will be filled by agent>")
        ml_scenario.add_turn(
            "user", "What's the difference between deep learning and traditional machine learning?"
        )
        ml_scenario.add_turn("agent", "<will be filled by agent>")
        ml_scenario.add_turn("user", "How would you explain backpropagation in simple terms?")
        ml_scenario.add_turn("agent", "<will be filled by agent>")
        ml_scenario.add_turn(
            "user",
            "Based on what you've told me, would you recommend starting with simple algorithms or diving into deep learning?",
        )
        ml_scenario.add_turn("agent", "<will be filled by agent>")

        # Expected outcomes
        ml_scenario.add_expected_outcome(
            "progressive_knowledge",
            "The agent should build knowledge progressively from basic to advanced topics",
            {
                "knowledge_progression": [
                    "basics",
                    "algorithms",
                    "neural_networks",
                    "deep_learning",
                ],
                "maintains_continuity": True,
            },
        )
        ml_scenario.add_expected_outcome(
            "knowledge_application",
            "The agent should apply earlier knowledge to answer later questions",
            {"references_previous_knowledge": True, "connects_concepts": True},
        )

        scenarios["machine_learning_concepts"] = ml_scenario

        # Scenario 3: Cross-Domain Conversation
        cross_domain_scenario = ConversationScenario(
            name="cross_domain_learning",
            description="A conversation spanning multiple domains to test knowledge integration",
            domain="cross-domain",
        )

        # Add turns for the cross-domain scenario
        cross_domain_scenario.add_turn("user", "Can you explain what data science is?")
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")
        cross_domain_scenario.add_turn(
            "user", "What programming languages are commonly used in data science?"
        )
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")
        cross_domain_scenario.add_turn("user", "How is machine learning used in data science?")
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")
        cross_domain_scenario.add_turn(
            "user", "Can you give examples of machine learning algorithms used for data analysis?"
        )
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")
        cross_domain_scenario.add_turn(
            "user",
            "I'm working on a project involving financial data. What approach would you recommend?",
        )
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")
        cross_domain_scenario.add_turn(
            "user",
            "Would Python be a good choice for this project, and which specific libraries might help?",
        )
        cross_domain_scenario.add_turn("agent", "<will be filled by agent>")

        # Expected outcomes
        cross_domain_scenario.add_expected_outcome(
            "cross_domain_integration",
            "The agent should integrate knowledge across domains",
            {
                "domains_connected": ["data_science", "programming", "machine_learning"],
                "applies_integrated_knowledge": True,
            },
        )
        cross_domain_scenario.add_expected_outcome(
            "specific_application",
            "The agent should apply general knowledge to specific use case",
            {"contextualizes_general_knowledge": True, "provides_specific_recommendations": True},
        )

        scenarios["cross_domain_learning"] = cross_domain_scenario

        return scenarios

    async def setup_agent(self):
        """Set up the LTM agent for testing."""
        # Initialize memory store
        vector_store = InMemoryVectorStore()

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            memory_store=vector_store, contextualizer=SimpleContextualizer()
        )

        # Initialize memory manager
        await self.memory_manager.initialize()

        # Create mock LLM
        llm = MockLLM()
        llm.set_memory_manager(self.memory_manager)

        # Initialize the agent
        self.agent = LongTermMemoryAgent(
            memory_manager=self.memory_manager,
            llm=llm,
            enable_consolidation=self.enable_consolidation,
            consolidation_interval=60,  # Set to 1 minute for testing
        )

        logger.info("Agent setup complete")

    async def run_scenario(self, scenario_name: str) -> dict[str, Any]:
        """
        Run a single conversation scenario.

        Args:
            scenario_name: Name of the scenario to run

        Returns:
            Dict with conversation results and evaluation
        """
        if not self.agent:
            await self.setup_agent()

        scenario = self.scenarios[scenario_name]
        logger.info(f"Running scenario: {scenario.name}")
        print(f"\n===== Running Scenario: {scenario.name} =====")
        print(f"Description: {scenario.description}\n")

        # Results will track the full conversation with agent responses
        results = {
            "scenario": scenario.to_dict(),
            "conversation": [],
            "knowledge_snapshots": [],
            "evaluation": {},
        }

        # Take a knowledge snapshot before the conversation
        initial_knowledge = await self._take_knowledge_snapshot("pre_conversation")
        results["knowledge_snapshots"].append(
            {"stage": "pre_conversation", "knowledge": initial_knowledge}
        )

        # Process each conversation turn
        user_turns = [turn for turn in scenario.turns if turn["role"] == "user"]

        for i, user_turn in enumerate(user_turns):
            query = user_turn["content"]
            turn_id = user_turn["id"]

            print(f"User: {query}")

            # Get agent response
            response = await self.agent.async_invoke(query)

            print(f"Agent: {response}\n")

            # Record the conversation turn
            results["conversation"].append({"turn_id": turn_id, "role": "user", "content": query})

            results["conversation"].append(
                {"turn_id": f"response_{turn_id}", "role": "agent", "content": response}
            )

            # Take knowledge snapshot after significant turns
            if i == len(user_turns) - 1 or i % 3 == 2:
                snapshot_stage = f"turn_{i + 1}"
                knowledge = await self._take_knowledge_snapshot(snapshot_stage)
                results["knowledge_snapshots"].append(
                    {"stage": snapshot_stage, "knowledge": knowledge}
                )

        # If consolidation is enabled, wait for it to potentially happen
        if self.enable_consolidation:
            print("Waiting for potential memory consolidation (1 minute)...")
            await asyncio.sleep(65)  # Just over consolidation interval

            # Take a final knowledge snapshot
            final_knowledge = await self._take_knowledge_snapshot("post_consolidation")
            results["knowledge_snapshots"].append(
                {"stage": "post_consolidation", "knowledge": final_knowledge}
            )

        # Evaluate the results
        results["evaluation"] = await self._evaluate_scenario(scenario, results)

        # Save results to file
        results_file = self.results_dir / f"{scenario.name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Scenario {scenario.name} completed. Results saved to {results_file}")
        print(f"Scenario completed. Results saved to {results_file}")

        return results

    async def _take_knowledge_snapshot(self, stage: str) -> list[dict[str, Any]]:
        """
        Take a snapshot of the current knowledge state.

        Args:
            stage: Label for the snapshot stage

        Returns:
            List of knowledge units
        """
        units = await self.memory_manager.list_knowledge()

        # Format for easier analysis
        knowledge = []
        for unit in units:
            knowledge.append(
                {"id": unit.unique_id, "content": unit.original_chunk, "metadata": unit.metadata}
            )

        return knowledge

    async def _evaluate_scenario(
        self, scenario: ConversationScenario, results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate the scenario results against expected outcomes.

        Args:
            scenario: The conversation scenario
            results: The conversation results

        Returns:
            Dict with evaluation results
        """
        evaluation = {}

        # Get the initial and final knowledge snapshots
        initial_knowledge = next(
            (
                s["knowledge"]
                for s in results["knowledge_snapshots"]
                if s["stage"] == "pre_conversation"
            ),
            [],
        )

        final_knowledge = []
        if self.enable_consolidation:
            final_knowledge = next(
                (
                    s["knowledge"]
                    for s in results["knowledge_snapshots"]
                    if s["stage"] == "post_consolidation"
                ),
                [],
            )
        else:
            # Get the last snapshot if consolidation not enabled
            if results["knowledge_snapshots"]:
                final_knowledge = results["knowledge_snapshots"][-1]["knowledge"]

        # Calculate knowledge growth
        knowledge_growth = len(final_knowledge) - len(initial_knowledge)
        evaluation["knowledge_growth"] = knowledge_growth

        # Evaluate each expected outcome
        for outcome_type, outcome in scenario.expected_outcomes.items():
            evaluation[outcome_type] = self._evaluate_outcome(
                outcome_type, outcome, results, initial_knowledge, final_knowledge
            )

        # Calculate overall score
        scores = []
        for outcome_evaluation in evaluation.values():
            if isinstance(outcome_evaluation, dict) and "score" in outcome_evaluation:
                scores.append(outcome_evaluation["score"])

        if scores:
            evaluation["overall_score"] = sum(scores) / len(scores)
        else:
            evaluation["overall_score"] = 0.0

        return evaluation

    def _evaluate_outcome(
        self,
        outcome_type: str,
        outcome: dict[str, Any],
        results: dict[str, Any],
        initial_knowledge: list[dict[str, Any]],
        final_knowledge: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate a specific expected outcome.

        Args:
            outcome_type: The type of outcome
            outcome: The expected outcome definition
            results: The conversation results
            initial_knowledge: Knowledge snapshot from before the conversation
            final_knowledge: Knowledge snapshot from after the conversation

        Returns:
            Dict with evaluation results for this outcome
        """
        evaluation = {
            "description": outcome["description"],
            "success": False,
            "score": 0.0,
            "details": {},
        }

        criteria = outcome["criteria"]

        if outcome_type == "knowledge_acquisition":
            # Check if expected topics are in the knowledge
            topics = criteria.get("topics", [])
            topic_matches = 0

            for topic in topics:
                for unit in final_knowledge:
                    if topic.lower() in unit["content"].lower():
                        topic_matches += 1
                        break

            topic_score = topic_matches / len(topics) if topics else 0.0
            evaluation["details"]["topic_coverage"] = {
                "score": topic_score,
                "matches": topic_matches,
                "expected": len(topics),
            }

            # Check for expected connections
            connections = criteria.get("connections", [])
            connection_matches = 0

            for connection in connections:
                parts = connection.split("-")
                if len(parts) == 2:
                    part1, part2 = parts

                    # Look for units that mention both parts
                    for unit in final_knowledge:
                        if (
                            part1.lower() in unit["content"].lower()
                            and part2.lower() in unit["content"].lower()
                        ):
                            connection_matches += 1
                            break

            connection_score = connection_matches / len(connections) if connections else 0.0
            evaluation["details"]["connection_coverage"] = {
                "score": connection_score,
                "matches": connection_matches,
                "expected": len(connections),
            }

            # Overall score for knowledge acquisition
            evaluation["score"] = (topic_score + connection_score) / 2
            evaluation["success"] = evaluation["score"] >= 0.5

        elif outcome_type == "conversation_coherence":
            # Simplified coherence evaluation
            maintains_topic = criteria.get("maintains_topic", False)
            references_previous = criteria.get("references_previous_turns", False)

            # For demonstration, just assume coherence if knowledge grew
            coherence_score = min(1.0, max(0.0, knowledge_growth / 5))
            evaluation["details"]["coherence"] = {
                "score": coherence_score,
                "knowledge_growth": knowledge_growth,
            }

            evaluation["score"] = coherence_score
            evaluation["success"] = evaluation["score"] >= 0.5

        elif outcome_type == "progressive_knowledge":
            # Check if knowledge progresses through expected stages
            progression = criteria.get("knowledge_progression", [])
            progression_matches = 0

            # Get knowledge snapshots in order
            snapshots = sorted(
                results["knowledge_snapshots"],
                key=lambda s: (
                    int(s["stage"].split("_")[1]) if s["stage"].startswith("turn_") else 0
                ),
            )

            # Check if later snapshots include more advanced topics
            for i, topic in enumerate(progression):
                if i < len(snapshots) - 1:
                    earlier = snapshots[i]["knowledge"]
                    later = snapshots[i + 1]["knowledge"]

                    # Check if topic appears in later but not earlier knowledge
                    topic_in_earlier = any(
                        topic.lower() in unit["content"].lower() for unit in earlier
                    )
                    topic_in_later = any(topic.lower() in unit["content"].lower() for unit in later)

                    if not topic_in_earlier and topic_in_later:
                        progression_matches += 1

            progression_score = progression_matches / len(progression) if progression else 0.0
            evaluation["details"]["knowledge_progression"] = {
                "score": progression_score,
                "matches": progression_matches,
                "expected": len(progression),
            }

            evaluation["score"] = progression_score
            evaluation["success"] = evaluation["score"] >= 0.4  # Lower threshold for progression

        elif outcome_type == "cross_domain_integration":
            # Check if domains are connected in knowledge
            domains = criteria.get("domains_connected", [])
            domain_pairs = []

            # Create pairs of domains
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    domain_pairs.append((domains[i], domains[j]))

            # Count connected domain pairs
            pair_matches = 0
            for domain1, domain2 in domain_pairs:
                for unit in final_knowledge:
                    if (
                        domain1.lower() in unit["content"].lower()
                        and domain2.lower() in unit["content"].lower()
                    ):
                        pair_matches += 1
                        break

            integration_score = pair_matches / len(domain_pairs) if domain_pairs else 0.0
            evaluation["details"]["domain_integration"] = {
                "score": integration_score,
                "matches": pair_matches,
                "possible_pairs": len(domain_pairs),
            }

            evaluation["score"] = integration_score
            evaluation["success"] = evaluation["score"] >= 0.5

        else:
            # Default evaluation for other outcome types
            evaluation["details"][
                "note"
            ] = f"No specific evaluation for outcome type: {outcome_type}"

            # Simple measure: did knowledge grow?
            evaluation["score"] = min(1.0, max(0.0, knowledge_growth / 5))
            evaluation["success"] = evaluation["score"] >= 0.5

        return evaluation

    async def run_all_scenarios(self) -> dict[str, Any]:
        """
        Run all conversation scenarios and collect results.

        Returns:
            Dict with results for all scenarios
        """
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios_run": len(self.scenarios),
            "results_by_scenario": {},
            "summary": {"overall_score": 0.0, "successful_scenarios": 0},
        }

        for scenario_name in self.scenarios:
            results = await self.run_scenario(scenario_name)
            all_results["results_by_scenario"][scenario_name] = {
                "overall_score": results["evaluation"]["overall_score"],
                "details": results["evaluation"],
            }

        # Calculate overall summary
        scenario_scores = [
            results["overall_score"]
            for scenario, results in all_results["results_by_scenario"].items()
        ]

        successful_scenarios = sum(
            1
            for scenario, results in all_results["results_by_scenario"].items()
            if results["overall_score"] >= 0.6
        )

        all_results["summary"]["overall_score"] = (
            sum(scenario_scores) / len(scenario_scores) if scenario_scores else 0.0
        )
        all_results["summary"]["successful_scenarios"] = successful_scenarios

        # Save summary results
        summary_file = self.results_dir / "all_scenarios_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"All scenarios completed. Summary saved to {summary_file}")
        print(f"\nAll scenarios completed. Summary saved to {summary_file}")

        return all_results


async def main():
    """Run the conversation simulator."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversation simulator for LTM agent evaluation")
    parser.add_argument("--scenario", type=str, help="Run a specific scenario (by name)")
    parser.add_argument(
        "--no-consolidation", action="store_true", help="Disable memory consolidation"
    )
    parser.add_argument("--results-dir", type=str, help="Directory to save results")
    args = parser.parse_args()

    simulator = ConversationSimulator(
        enable_consolidation=not args.no_consolidation, results_dir=args.results_dir
    )

    if args.scenario:
        if args.scenario in simulator.scenarios:
            await simulator.run_scenario(args.scenario)
        else:
            print(f"Unknown scenario: {args.scenario}")
            print("Available scenarios:", list(simulator.scenarios.keys()))
    else:
        await simulator.run_all_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
