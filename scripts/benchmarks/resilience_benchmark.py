"""
Benchmark module for measuring resilience and recovery capabilities.

This module evaluates:
1. Recovery from simulated failures
2. Graceful degradation under adverse conditions
3. Data consistency after recovery
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.ltm_agent.core.models import KnowledgeUnit
from src.ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from src.ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from src.ltm_agent.memory.in_memory_store import InMemoryVectorStore
from src.ltm_agent.memory.knowledge_graph import KnowledgeGraph
from src.ltm_agent.memory.monitoring import get_monitor
from src.ltm_agent.memory.resilience import (
    FailureType,
    RecoveryStrategy,
    ResilientOperation,
    get_resilience_manager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResilienceBenchmark:
    """Benchmark for measuring resilience and recovery capabilities."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the resilience benchmark.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.results = {}
        self.memory_manager = None
        self.resilience_manager = None
        self.monitor = None
        self.output_dir = self.config.get("output_dir", "benchmark_results")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self) -> None:
        """Set up benchmark environment with resilient memory manager."""
        logger.info("Setting up resilience benchmark environment...")

        # Get monitor
        self.monitor = get_monitor({"enable_telemetry": True, "telemetry_interval_seconds": 60})

        # Get resilience manager
        self.resilience_manager = get_resilience_manager(
            {
                "backup_dir": os.path.join(self.output_dir, "backups"),
                "backup_interval_hours": 0.1,  # Short interval for testing
                "max_backups": 5,
            }
        )

        # Create memory components
        vector_store = InMemoryVectorStore()
        knowledge_graph = KnowledgeGraph()
        contextualizer = EnhancedContextualizer()

        # Create enhanced memory manager
        self.memory_manager = EnhancedMemoryManager(
            memory_store=vector_store,
            contextualizer=contextualizer,
            knowledge_graph=knowledge_graph,
        )

        logger.info("Benchmark environment setup complete")

    def _generate_test_data(self, count: int = 100) -> list[KnowledgeUnit]:
        """
        Generate test knowledge units.

        Args:
            count: Number of units to generate

        Returns:
            List of knowledge units
        """
        knowledge_units = []

        # Generate a set of knowledge chunks
        topics = [
            "Python programming",
            "Machine learning",
            "Data science",
            "Artificial intelligence",
            "Natural language processing",
        ]

        for i in range(count):
            topic_idx = i % len(topics)
            topic = topics[topic_idx]

            # Create knowledge unit with content related to the topic
            unit = KnowledgeUnit(
                id=str(uuid.uuid4()),
                text=f"This is knowledge about {topic}: fact #{i + 1}. "
                f"This information is related to {topic} and contains specific details "
                f"that might be useful for retrieval testing.",
                created_at=datetime.now(),
                metadata={"topic": topic, "index": i},
            )

            knowledge_units.append(unit)

        return knowledge_units

    def _simulate_failure(self, failure_type: FailureType) -> Exception:
        """
        Simulate a specific type of failure.

        Args:
            failure_type: Type of failure to simulate

        Returns:
            Simulated exception
        """
        # Map failure types to exception types
        failure_exceptions = {
            FailureType.DATA_CORRUPTION: ValueError("Simulated data corruption"),
            FailureType.DATA_INCONSISTENCY: ValueError("Simulated data inconsistency"),
            FailureType.DATA_LOSS: KeyError("Simulated data loss"),
            FailureType.DATABASE_UNAVAILABLE: ConnectionError("Simulated database unavailable"),
            FailureType.EMBEDDING_SERVICE_ERROR: RuntimeError("Simulated embedding service error"),
            FailureType.TIMEOUT: TimeoutError("Simulated timeout"),
            FailureType.RATE_LIMIT_EXCEEDED: ValueError("Simulated rate limit exceeded"),
            FailureType.API_ERROR: Exception("Simulated API error"),
            FailureType.UNKNOWN: Exception("Simulated unknown failure"),
        }

        # Return the corresponding exception
        return failure_exceptions.get(failure_type, Exception("Generic simulation failure"))

    def test_recovery_from_failure(
        self,
        failure_type: FailureType,
        recovery_strategy: RecoveryStrategy | None = None,
        retry_count: int = 3,
    ) -> dict[str, Any]:
        """
        Test recovery from a specific type of failure.

        Args:
            failure_type: Type of failure to simulate
            recovery_strategy: Optional specific recovery strategy to use
            retry_count: Number of retry attempts

        Returns:
            Test results
        """
        logger.info(f"Testing recovery from {failure_type.value} failure...")

        # Initialize result structure
        result = {
            "failure_type": failure_type.value,
            "recovery_strategy": recovery_strategy.value if recovery_strategy else None,
            "success": False,
            "recovery_time": 0,
            "data_consistency": 0,
        }

        # Generate test data
        test_data = self._generate_test_data(50)

        # Add data to memory manager
        for unit in test_data:
            self.memory_manager.add_knowledge_unit(unit)

        # Create a checkpoint before simulating failure
        logger.info("Creating checkpoint before simulating failure...")
        data_to_checkpoint = {
            "knowledge_units": self.memory_manager.get_all_knowledge_units(),
            "metadata": {"timestamp": datetime.now().isoformat()},
        }
        checkpoint_file = self.resilience_manager.create_checkpoint(
            data_to_checkpoint, "memory_manager"
        )
        logger.info(f"Created checkpoint: {checkpoint_file}")

        # Get all units to verify data consistency later
        original_units = {unit.id: unit for unit in self.memory_manager.get_all_knowledge_units()}

        # Define a test operation that will fail
        def failing_operation():
            # Simulate the failure
            raise self._simulate_failure(failure_type)

        # Record failure and attempt recovery
        try:
            start_time = time.time()

            # Execute failing operation with resilience
            with ResilientOperation(
                operation_name="test_operation",
                component_name="memory_manager",
                resilience_manager=self.resilience_manager,
                failure_mapping={Exception: failure_type},
                context={
                    "component_name": "memory_manager",
                    "retry_operation": failing_operation,
                    "max_retries": retry_count,
                    "base_delay": 0.1,  # Short delay for testing
                    "max_delay": 1.0,
                },
            ):
                failing_operation()

        except Exception as e:
            # Recovery failed
            logger.warning(f"Recovery failed: {e}")
            result["success"] = False

        else:
            # Recovery succeeded
            recovery_time = time.time() - start_time
            result["success"] = True
            result["recovery_time"] = recovery_time

            # Check data consistency (if applicable)
            if failure_type in [
                FailureType.DATA_CORRUPTION,
                FailureType.DATA_LOSS,
                FailureType.DATA_INCONSISTENCY,
            ]:
                # Restore from checkpoint for data integrity failures
                restored_data = self.resilience_manager.restore_from_checkpoint("memory_manager")

                if restored_data:
                    # Compare restored units with original
                    restored_units = {unit.id: unit for unit in restored_data["knowledge_units"]}

                    # Calculate consistency percentage
                    consistent_count = 0
                    for unit_id, unit in original_units.items():
                        if unit_id in restored_units and unit.text == restored_units[unit_id].text:
                            consistent_count += 1

                    consistency = consistent_count / max(1, len(original_units)) * 100
                    result["data_consistency"] = consistency

        logger.info(
            f"Recovery test result: success={result['success']}, "
            f"recovery_time={result.get('recovery_time', 'N/A')}, "
            f"data_consistency={result.get('data_consistency', 'N/A')}%"
        )

        return result

    def run_benchmark(
        self, test_failures: list[FailureType] | None = None, trials_per_failure: int = 5
    ) -> dict[str, Any]:
        """
        Run the resilience benchmark with various failure scenarios.

        Args:
            test_failures: List of failure types to test
            trials_per_failure: Number of trials for each failure type

        Returns:
            Benchmark results
        """
        logger.info("Running resilience benchmark...")

        # Setup if not already done
        if not self.memory_manager or not self.resilience_manager:
            self.setup()

        # Use default failure types if not specified
        if test_failures is None:
            test_failures = [
                FailureType.DATA_CORRUPTION,
                FailureType.DATA_LOSS,
                FailureType.DATABASE_UNAVAILABLE,
                FailureType.EMBEDDING_SERVICE_ERROR,
                FailureType.TIMEOUT,
                FailureType.RATE_LIMIT_EXCEEDED,
            ]

        # Initialize results structure
        results = {
            "failure_types": [f.value for f in test_failures],
            "trials_per_failure": trials_per_failure,
            "success_rates": {},
            "avg_recovery_times": {},
            "data_consistency": {},
            "detailed_results": {},
        }

        # Test each failure type
        for failure_type in test_failures:
            logger.info(f"Benchmarking recovery from {failure_type.value}...")

            trial_results = []

            # Run multiple trials for statistical significance
            for trial in range(trials_per_failure):
                logger.info(f"Trial {trial + 1}/{trials_per_failure} for {failure_type.value}")
                result = self.test_recovery_from_failure(failure_type)
                trial_results.append(result)

            # Calculate statistics
            success_count = sum(1 for r in trial_results if r["success"])
            success_rate = success_count / trials_per_failure * 100

            recovery_times = [
                r["recovery_time"] for r in trial_results if r["success"] and r["recovery_time"] > 0
            ]
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0

            consistency_values = [
                r["data_consistency"]
                for r in trial_results
                if r["success"] and r["data_consistency"] > 0
            ]
            avg_consistency = np.mean(consistency_values) if consistency_values else 0

            # Store results
            results["success_rates"][failure_type.value] = success_rate
            results["avg_recovery_times"][failure_type.value] = avg_recovery_time
            results["data_consistency"][failure_type.value] = avg_consistency
            results["detailed_results"][failure_type.value] = trial_results

        logger.info("Resilience benchmark completed")
        self.results = results
        return results

    def plot_results(self) -> None:
        """Plot the resilience benchmark results."""
        if not self.results:
            logger.warning("No benchmark results to plot. Run benchmark first.")
            return

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Get data for plotting
        failure_types = self.results["failure_types"]
        success_rates = [self.results["success_rates"].get(ft, 0) for ft in failure_types]
        recovery_times = [self.results["avg_recovery_times"].get(ft, 0) for ft in failure_types]

        # Plot success rates
        bar_width = 0.35
        x = np.arange(len(failure_types))

        axes[0].bar(x, success_rates, bar_width, label="Success Rate")
        axes[0].set_title("Recovery Success Rate by Failure Type")
        axes[0].set_xlabel("Failure Type")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(failure_types, rotation=45, ha="right")
        axes[0].set_ylim(0, 105)  # 0-105% for better visibility
        axes[0].grid(axis="y")

        # Add success rate values on top of bars
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 2, f"{v:.1f}%", ha="center")

        # Plot recovery times
        axes[1].bar(x, recovery_times, bar_width, label="Recovery Time", color="orange")
        axes[1].set_title("Average Recovery Time by Failure Type")
        axes[1].set_xlabel("Failure Type")
        axes[1].set_ylabel("Recovery Time (seconds)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(failure_types, rotation=45, ha="right")
        axes[1].grid(axis="y")

        # Add recovery time values on top of bars
        for i, v in enumerate(recovery_times):
            axes[1].text(i, v + 0.05, f"{v:.3f}s", ha="center")

        # Adjust layout and save figure
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(self.output_dir, "resilience_benchmark.png")
        plt.savefig(output_path)
        logger.info(f"Benchmark plot saved to {output_path}")

        # Show the plot (if in interactive environment)
        plt.close()

    def print_results(self) -> None:
        """Print the benchmark results in a readable format."""
        if not self.results:
            logger.warning("No benchmark results to print. Run benchmark first.")
            return

        print("\n" + "=" * 60)
        print("RESILIENCE BENCHMARK RESULTS")
        print("=" * 60)

        print("\nFailure Type | Success Rate | Recovery Time | Data Consistency")
        print("-" * 75)

        for failure_type in self.results["failure_types"]:
            success_rate = self.results["success_rates"].get(failure_type, 0)
            recovery_time = self.results["avg_recovery_times"].get(failure_type, 0)
            consistency = self.results["data_consistency"].get(failure_type, 0)

            print(
                f"{failure_type:15s} | {success_rate:>11.1f}% | {recovery_time:>12.3f}s | {consistency:>15.1f}%"
            )

        print("-" * 75)

        # Calculate overall statistics
        avg_success_rate = np.mean(list(self.results["success_rates"].values()))
        avg_recovery_time = np.mean(list(self.results["avg_recovery_times"].values()))
        avg_consistency = np.mean([v for v in self.results["data_consistency"].values() if v > 0])

        print(
            f"AVERAGE        | {avg_success_rate:>11.1f}% | {avg_recovery_time:>12.3f}s | {avg_consistency:>15.1f}%"
        )

        print("=" * 75 + "\n")

        # Save results to JSON
        output_path = os.path.join(self.output_dir, "resilience_benchmark.json")
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    benchmark = ResilienceBenchmark({"output_dir": "benchmark_results"})
    benchmark.setup()
    results = benchmark.run_benchmark(
        test_failures=[
            FailureType.DATA_CORRUPTION,
            FailureType.DATABASE_UNAVAILABLE,
            FailureType.EMBEDDING_SERVICE_ERROR,
            FailureType.TIMEOUT,
        ],
        trials_per_failure=3,  # Lower for quicker testing
    )
    benchmark.print_results()
    benchmark.plot_results()
