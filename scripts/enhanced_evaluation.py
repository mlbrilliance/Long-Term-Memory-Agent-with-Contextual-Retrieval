"""
Enhanced Evaluation System - Integrating conversations, benchmarks, and visualization.

This script ties together the conversation simulator, benchmark runner, and
memory visualizer to provide a comprehensive evaluation of the Long-Term
Memory Agent's capabilities and performance over time.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import benchmark components
from benchmark_runner import (
    Benchmark,
    BenchmarkRunner,
    BenchmarkSuite,
    ConversationBenchmark,
    CoreMemoryBenchmark,
    MemoryConnectionsBenchmark,
)

# Import conversation simulator
from conversation_simulator import ConversationScenario, ConversationSimulator, MockLLM

# Import memory visualizer
from memory_visualizer import MemoryVisualizer

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.memory.contextualizer import SimpleContextualizer
from ltm_agent.memory.in_memory_store import InMemoryVectorStore

# Import agent components
from ltm_agent.memory.manager import MemoryManager

# Configure directories
eval_dir = Path(__file__).parent.parent / "logs" / "evaluations"
eval_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(eval_dir / "enhanced_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("enhanced_evaluation")


class EnhancedEvaluationSystem:
    """
    Integrates conversation simulation, benchmarking, and visualization
    to provide comprehensive evaluation of the LTM agent.
    """

    def __init__(self, output_dir: str | None = None):
        """
        Initialize the enhanced evaluation system.

        Args:
            output_dir: Optional directory to save evaluation results
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = eval_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.benchmark_dir = self.output_dir / "benchmarks"
        self.conversation_dir = self.output_dir / "conversations"
        self.visualization_dir = self.output_dir / "visualizations"

        self.benchmark_dir.mkdir(exist_ok=True)
        self.conversation_dir.mkdir(exist_ok=True)
        self.visualization_dir.mkdir(exist_ok=True)

        # Initialize components
        self.benchmark_runner = BenchmarkRunner(storage_path=str(self.benchmark_dir))
        self.visualizer = MemoryVisualizer(output_dir=str(self.visualization_dir))

    async def _create_agent(self) -> LongTermMemoryAgent:
        """
        Create a Long-Term Memory Agent instance for evaluation.

        Returns:
            LongTermMemoryAgent instance
        """
        # Create vector store
        vector_store = InMemoryVectorStore()

        # Create contextualizer
        contextualizer = SimpleContextualizer()

        # Create memory manager
        memory_manager = MemoryManager(memory_store=vector_store, contextualizer=contextualizer)

        # Create agent
        agent = LongTermMemoryAgent(memory_manager=memory_manager)

        return agent

    async def _capture_knowledge_snapshot(
        self, agent: LongTermMemoryAgent, stage: str
    ) -> dict[str, Any]:
        """
        Capture a snapshot of the agent's knowledge at a specific stage.

        Args:
            agent: The LongTermMemoryAgent instance
            stage: A string descriptor of the current evaluation stage

        Returns:
            Dict containing the knowledge snapshot
        """
        # Get all knowledge units
        units = await agent.memory_manager.list_knowledge()

        # Format for storage
        knowledge_units = []
        for unit in units:
            knowledge_units.append(
                {
                    "id": unit.unique_id,
                    "content": unit.original_chunk,
                    "metadata": unit.metadata or {},
                }
            )

        # Create snapshot
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "knowledge": knowledge_units,
        }

        return snapshot

    async def run_conversation_with_visualization(
        self,
        scenario: ConversationScenario,
        capture_stages: list[str] | None = None,
        output_prefix: str = "conversation",
    ) -> dict[str, Any]:
        """
        Run a conversation scenario with the agent and generate visualizations.

        Args:
            scenario: The conversation scenario to run
            capture_stages: Optional list of stage names to capture snapshots at
            output_prefix: Prefix for output files

        Returns:
            Dict with paths to generated outputs and evaluation results
        """
        # Create agent
        agent = await self._create_agent()

        # Create mock LLM
        mock_llm = MockLLM()

        # Create conversation simulator
        simulator = ConversationSimulator(agent=agent, llm=mock_llm)

        # Prepare to capture knowledge snapshots
        knowledge_snapshots = []
        conversation_turns = []

        # Initial snapshot
        initial_snapshot = await self._capture_knowledge_snapshot(agent=agent, stage="Initial")
        knowledge_snapshots.append(initial_snapshot)

        # If no capture stages specified, create default stages
        if not capture_stages:
            total_turns = len(scenario.turns)
            capture_stages = [f"Turn {i + 1}/{total_turns}" for i in range(total_turns)]

        # Run conversation
        evaluation_results = await simulator.run_scenario(
            scenario=scenario,
            callbacks={
                "on_turn_complete": lambda turn_idx, user_input, agent_response: (
                    conversation_turns.append(
                        {"turn": turn_idx, "role": "user", "content": user_input}
                    ),
                    conversation_turns.append(
                        {"turn": turn_idx, "role": "assistant", "content": agent_response}
                    ),
                )
            },
        )

        # Capture snapshot after each turn
        for i, stage in enumerate(capture_stages):
            snapshot = await self._capture_knowledge_snapshot(agent=agent, stage=stage)
            knowledge_snapshots.append(snapshot)

        # Generate timestamp for filenames
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save conversation turns
        conversation_file = f"{output_prefix}_turns_{timestamp}.json"
        conversation_path = self.conversation_dir / conversation_file
        with open(conversation_path, "w") as f:
            json.dump(conversation_turns, f, indent=2)

        # Save knowledge snapshots
        snapshots_file = f"{output_prefix}_snapshots_{timestamp}.json"
        snapshots_path = self.conversation_dir / snapshots_file
        with open(snapshots_path, "w") as f:
            json.dump(knowledge_snapshots, f, indent=2)

        # Generate visualizations
        viz_paths = {}

        # Knowledge graph of final state
        final_units = knowledge_snapshots[-1]["knowledge"]
        kg_file = f"{output_prefix}_knowledge_graph_{timestamp}.png"
        kg_path = self.visualizer.visualize_knowledge_graph(
            knowledge_units=final_units,
            title=f"Knowledge Graph After {scenario.name}",
            output_file=kg_file,
        )
        viz_paths["knowledge_graph"] = kg_path

        # Topic network of final state
        tn_file = f"{output_prefix}_topic_network_{timestamp}.png"
        tn_path = self.visualizer.visualize_topic_network(
            knowledge_units=final_units,
            title=f"Topic Network After {scenario.name}",
            output_file=tn_file,
        )
        viz_paths["topic_network"] = tn_path

        # Knowledge growth over time
        kg_file = f"{output_prefix}_knowledge_growth_{timestamp}.png"
        kg_path = self.visualizer.visualize_knowledge_growth(
            knowledge_snapshots=knowledge_snapshots,
            title=f"Knowledge Growth During {scenario.name}",
            output_file=kg_file,
        )
        viz_paths["knowledge_growth"] = kg_path

        # Conversation knowledge evolution
        ck_file = f"{output_prefix}_conversation_knowledge_{timestamp}.png"
        ck_path = self.visualizer.visualize_conversation_knowledge(
            conversation=conversation_turns,
            knowledge_snapshots=knowledge_snapshots,
            title=f"Knowledge Evolution During {scenario.name}",
            output_file=ck_file,
        )
        viz_paths["conversation_knowledge"] = ck_path

        # Compile results
        results = {
            "timestamp": timestamp,
            "scenario": scenario.name,
            "evaluation_results": evaluation_results,
            "conversation_path": str(conversation_path),
            "snapshots_path": str(snapshots_path),
            "visualizations": viz_paths,
        }

        # Save combined results
        results_file = f"{output_prefix}_results_{timestamp}.json"
        results_path = self.output_dir / results_file
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Conversation evaluation completed for {scenario.name}")
        return results

    async def run_benchmark_with_visualization(
        self, benchmark_suite: BenchmarkSuite, output_prefix: str = "benchmark"
    ) -> dict[str, Any]:
        """
        Run a benchmark suite and generate visualizations.

        Args:
            benchmark_suite: The benchmark suite to run
            output_prefix: Prefix for output files

        Returns:
            Dict with benchmark results and paths to visualizations
        """
        # Run benchmark
        benchmark_results = await self.benchmark_runner.run_suite(suite=benchmark_suite)

        # Generate timestamp for filenames
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Generate visualizations based on benchmark results
        viz_paths = {}

        # Extract knowledge units from benchmark results if available
        final_knowledge = []
        for benchmark_name, result in benchmark_results["benchmark_results"].items():
            if "final_knowledge" in result:
                final_knowledge.extend(result["final_knowledge"])

        # Generate knowledge graph if knowledge units available
        if final_knowledge:
            kg_file = f"{output_prefix}_knowledge_graph_{timestamp}.png"
            kg_path = self.visualizer.visualize_knowledge_graph(
                knowledge_units=final_knowledge,
                title=f"Knowledge Graph After {benchmark_suite.name}",
                output_file=kg_file,
            )
            viz_paths["knowledge_graph"] = kg_path

            # Topic network
            tn_file = f"{output_prefix}_topic_network_{timestamp}.png"
            tn_path = self.visualizer.visualize_topic_network(
                knowledge_units=final_knowledge,
                title=f"Topic Network After {benchmark_suite.name}",
                output_file=tn_file,
            )
            viz_paths["topic_network"] = tn_path

        # Add visualization paths to results
        benchmark_results["visualizations"] = viz_paths

        # Save combined results
        results_file = f"{output_prefix}_results_{timestamp}.json"
        results_path = self.output_dir / results_file
        with open(results_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        logger.info(f"Benchmark evaluation completed for {benchmark_suite.name}")
        return benchmark_results

    async def run_comprehensive_evaluation(
        self, scenarios: list[ConversationScenario], benchmarks: list[Benchmark]
    ) -> dict[str, Any]:
        """
        Run a comprehensive evaluation using both conversation scenarios
        and benchmarks, with visualizations.

        Args:
            scenarios: List of conversation scenarios to run
            benchmarks: List of benchmarks to run

        Returns:
            Dict with paths to results and summary metrics
        """
        # Generate timestamp for this evaluation run
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Create evaluation directory
        eval_run_dir = self.output_dir / f"evaluation_run_{timestamp}"
        eval_run_dir.mkdir(exist_ok=True)

        # Results storage
        conversation_results = []
        benchmark_results = {}

        # Run conversation scenarios
        for i, scenario in enumerate(scenarios):
            logger.info(f"Running conversation scenario {i + 1}/{len(scenarios)}: {scenario.name}")

            result = await self.run_conversation_with_visualization(
                scenario=scenario,
                output_prefix=f"scenario_{i + 1}_{scenario.name.lower().replace(' ', '_')}",
            )
            conversation_results.append(result)

        # Create benchmark suite and run
        if benchmarks:
            suite = BenchmarkSuite(name="Comprehensive Evaluation Suite", benchmarks=benchmarks)

            logger.info(f"Running benchmark suite with {len(benchmarks)} benchmarks")

            suite_results = await self.run_benchmark_with_visualization(
                benchmark_suite=suite, output_prefix="comprehensive_suite"
            )
            benchmark_results = suite_results

        # Compile all results
        comprehensive_results = {
            "timestamp": timestamp,
            "conversation_results": conversation_results,
            "benchmark_results": benchmark_results,
        }

        # Save compiled results
        results_file = f"comprehensive_evaluation_{timestamp}.json"
        results_path = eval_run_dir / results_file
        with open(results_path, "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        # Generate summary report
        summary_report = self._generate_summary_report(comprehensive_results, timestamp)

        # Save summary report
        summary_file = f"evaluation_summary_{timestamp}.md"
        summary_path = eval_run_dir / summary_file
        with open(summary_path, "w") as f:
            f.write(summary_report)

        logger.info(f"Comprehensive evaluation completed. Summary saved to {summary_path}")

        return {"results_path": str(results_path), "summary_path": str(summary_path)}

    def _generate_summary_report(self, results: dict[str, Any], timestamp: str) -> str:
        """
        Generate a markdown summary report from evaluation results.

        Args:
            results: Comprehensive evaluation results
            timestamp: Timestamp for this evaluation run

        Returns:
            Markdown formatted summary report
        """
        # Format timestamp for display
        display_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

        # Start building the report
        report = f"""# Long-Term Memory Agent Evaluation Summary

## Evaluation Run: {display_time}

### Overview
This report summarizes the results of a comprehensive evaluation of the Long-Term Memory Agent's
capabilities, including conversation handling and benchmark performance.

"""

        # Add conversation results
        if results.get("conversation_results"):
            report += "## Conversation Scenarios\n\n"

            for i, result in enumerate(results["conversation_results"]):
                scenario_name = result.get("scenario", f"Scenario {i + 1}")
                report += f"### {scenario_name}\n\n"

                # Add evaluation metrics
                if "evaluation_results" in result:
                    metrics = result["evaluation_results"]
                    report += "#### Metrics\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"

                    for metric, value in metrics.items():
                        report += f"| {metric} | {value} |\n"

                    report += "\n"

                # Add visualization links
                if "visualizations" in result:
                    report += "#### Visualizations\n\n"

                    for viz_type, viz_path in result["visualizations"].items():
                        if viz_path:
                            report += f"- [{viz_type.replace('_', ' ').title()}]({Path(viz_path).relative_to(self.output_dir)})\n"

                    report += "\n"

                report += "\n"

        # Add benchmark results
        if results.get("benchmark_results"):
            benchmark_data = results["benchmark_results"]
            report += "## Benchmark Results\n\n"

            if "suite_name" in benchmark_data:
                report += f"### {benchmark_data['suite_name']}\n\n"

            if "benchmark_results" in benchmark_data:
                report += "#### Individual Benchmark Results\n\n"
                report += "| Benchmark | Score | Status |\n"
                report += "|-----------|-------|--------|\n"

                for name, result in benchmark_data["benchmark_results"].items():
                    score = result.get("score", "N/A")
                    status = result.get("status", "Unknown")
                    report += f"| {name} | {score} | {status} |\n"

                report += "\n"

            if "suite_score" in benchmark_data:
                report += f"#### Overall Suite Score: {benchmark_data['suite_score']}\n\n"

            # Add visualization links
            if "visualizations" in benchmark_data:
                report += "#### Visualizations\n\n"

                for viz_type, viz_path in benchmark_data["visualizations"].items():
                    if viz_path:
                        report += f"- [{viz_type.replace('_', ' ').title()}]({Path(viz_path).relative_to(self.output_dir)})\n"

                report += "\n"

        # Add conclusion
        report += """## Conclusion

This evaluation provides insights into the Long-Term Memory Agent's performance in conversation handling
and benchmark tests. Review the detailed results and visualizations for a comprehensive understanding
of the agent's strengths and areas for improvement.
"""

        return report


# Example scenarios for testing
def create_test_scenarios() -> list[ConversationScenario]:
    """Create test conversation scenarios."""
    # Progressive learning scenario
    progressive_scenario = ConversationScenario(
        name="Progressive Learning",
        description="Tests the agent's ability to build knowledge progressively",
        domain="personal_info",
    )

    # Add turns
    progressive_scenario.add_turn("user", "My name is John and I work as a software engineer.")
    progressive_scenario.add_turn("user", "I specialize in Python and JavaScript development.")
    progressive_scenario.add_turn(
        "user", "I've been working on a machine learning project recently."
    )
    progressive_scenario.add_turn("user", "Can you tell me what you know about me?")

    # Add expected outcome
    progressive_scenario.add_expected_outcome(
        outcome_type="knowledge_acquisition",
        description="Agent should learn basic personal information",
        evaluation_criteria={
            "expected_topics": [
                "John",
                "software engineer",
                "Python",
                "JavaScript",
                "machine learning",
            ]
        },
    )

    # Knowledge correction scenario
    correction_scenario = ConversationScenario(
        name="Knowledge Correction",
        description="Tests the agent's ability to correct knowledge",
        domain="geography",
    )

    # Add turns
    correction_scenario.add_turn("user", "The capital of France is London.")
    correction_scenario.add_turn(
        "user", "Actually, I made a mistake. The capital of France is Paris."
    )
    correction_scenario.add_turn("user", "What is the capital of France?")

    # Add expected outcome
    correction_scenario.add_expected_outcome(
        outcome_type="knowledge_correction",
        description="Agent should correct previously stored incorrect information",
        evaluation_criteria={
            "expected_topics": ["Paris", "capital", "France"],
            "unexpected_topics": ["London"],
        },
    )

    # Multi-hop reasoning scenario
    multihop_scenario = ConversationScenario(
        name="Multi-hop Reasoning",
        description="Tests the agent's ability to connect separate pieces of information",
        domain="relationships",
    )

    # Add turns
    multihop_scenario.add_turn("user", "Alice is Bob's sister.")
    multihop_scenario.add_turn("user", "Bob is Charlie's father.")
    multihop_scenario.add_turn("user", "What is the relationship between Alice and Charlie?")

    # Add expected outcome
    multihop_scenario.add_expected_outcome(
        outcome_type="relationship_inference",
        description="Agent should infer relationships between entities",
        evaluation_criteria={"expected_topics": ["Alice", "Charlie", "Bob", "aunt"]},
    )

    return [progressive_scenario, correction_scenario, multihop_scenario]


# Example benchmarks
def create_test_benchmarks() -> list[Benchmark]:
    """Create test benchmarks."""

    # Memory connections benchmark
    memory_connections = MemoryConnectionsBenchmark(
        name="Memory Connections Test",
        description="Tests the agent's ability to form connections between related memories",
    )

    # Core memory benchmark
    core_memory = CoreMemoryBenchmark(
        name="Core Memory Test", description="Tests the agent's fundamental memory capabilities"
    )

    # Conversation benchmark
    conversation = ConversationBenchmark(
        name="Conversation Handling Test",
        description="Tests the agent's ability to maintain context in a conversation",
    )

    return [memory_connections, core_memory, conversation]


async def main():
    """Run the enhanced evaluation system."""
    parser = argparse.ArgumentParser(description="Enhanced evaluation system for LTM agent")
    parser.add_argument("--output-dir", type=str, help="Directory to save evaluation results")
    parser.add_argument(
        "--scenario-only", action="store_true", help="Run only conversation scenarios"
    )
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmarks")
    args = parser.parse_args()

    # Create evaluation system
    evaluator = EnhancedEvaluationSystem(output_dir=args.output_dir)

    # Create test scenarios and benchmarks
    scenarios = create_test_scenarios()
    benchmarks = create_test_benchmarks()

    # Run evaluations based on arguments
    if args.scenario_only:
        for scenario in scenarios:
            await evaluator.run_conversation_with_visualization(scenario)
    elif args.benchmark_only:
        suite = BenchmarkSuite(name="Test Benchmark Suite", benchmarks=benchmarks)
        await evaluator.run_benchmark_with_visualization(suite)
    else:
        # Run comprehensive evaluation
        results = await evaluator.run_comprehensive_evaluation(
            scenarios=scenarios, benchmarks=benchmarks
        )

        print(f"\nEvaluation complete. Summary report saved to: {results['summary_path']}")


if __name__ == "__main__":
    asyncio.run(main())
