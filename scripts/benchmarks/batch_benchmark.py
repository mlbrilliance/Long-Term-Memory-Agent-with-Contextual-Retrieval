"""
Benchmark module for measuring batch processing performance.

This module evaluates:
1. Performance improvement with batch processing
2. Optimal batch sizes for different operations
3. Throughput for various batch configurations
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
from src.ltm_agent.memory.batch_processor import BatchProcessor
from src.ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from src.ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from src.ltm_agent.memory.in_memory_store import InMemoryVectorStore
from src.ltm_agent.memory.knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BatchBenchmark:
    """Benchmark for measuring batch processing performance."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the batch processing benchmark.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.results = {}
        self.output_dir = self.config.get("output_dir", "benchmark_results")
        self.batch_sizes = self.config.get("batch_sizes", [1, 5, 10, 20, 50, 100, 200])

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_test_data(self, count: int) -> list[KnowledgeUnit]:
        """
        Generate test knowledge units.

        Args:
            count: Number of units to generate

        Returns:
            List of knowledge units
        """
        knowledge_units = []

        # Generate varied content to better simulate real-world data
        topics = [
            "Python programming",
            "Machine learning",
            "Data science",
            "Artificial intelligence",
            "Natural language processing",
            "Computer vision",
            "Robotics",
            "Deep learning",
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

    def _create_memory_manager(self, batch_size: int) -> EnhancedMemoryManager:
        """
        Create a memory manager with the specified batch size.

        Args:
            batch_size: Batch size to use

        Returns:
            Configured EnhancedMemoryManager
        """
        # Create components
        vector_store = InMemoryVectorStore()
        knowledge_graph = KnowledgeGraph()
        contextualizer = EnhancedContextualizer()

        # Create batch processor with specified batch size
        batch_processor = BatchProcessor(batch_size=batch_size) if batch_size > 1 else None

        # Create memory manager
        memory_manager = EnhancedMemoryManager(
            memory_store=vector_store,
            contextualizer=contextualizer,
            knowledge_graph=knowledge_graph,
            batch_processor=batch_processor,
        )

        return memory_manager

    def benchmark_add_operations(self, data_count: int = 1000, trials: int = 3) -> dict[str, Any]:
        """
        Benchmark adding knowledge units with different batch sizes.

        Args:
            data_count: Number of knowledge units to add
            trials: Number of trials for each batch size

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking add operations with {data_count} units...")

        results = {
            "batch_sizes": self.batch_sizes,
            "throughput": [],
            "time_per_unit": [],
            "total_time": [],
        }

        # Generate test data
        test_data = self._generate_test_data(data_count)

        # Test each batch size
        for batch_size in self.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            batch_times = []

            # Run multiple trials
            for trial in range(trials):
                # Create a fresh memory manager for each trial
                memory_manager = self._create_memory_manager(batch_size)

                # Measure time to add all units
                start_time = time.time()

                for unit in test_data:
                    memory_manager.add_knowledge_unit(unit)

                elapsed_time = time.time() - start_time
                batch_times.append(elapsed_time)

            # Calculate average time across trials
            avg_time = np.mean(batch_times)
            time_per_unit = avg_time / data_count
            throughput = data_count / avg_time

            # Store results
            results["total_time"].append(avg_time)
            results["time_per_unit"].append(time_per_unit)
            results["throughput"].append(throughput)

            logger.info(
                f"Batch size {batch_size}: {avg_time:.4f}s total, "
                f"{time_per_unit * 1000:.4f}ms per unit, "
                f"{throughput:.1f} units/second"
            )

        return results

    def benchmark_search_operations(
        self, data_count: int = 1000, query_count: int = 50, trials: int = 3
    ) -> dict[str, Any]:
        """
        Benchmark search operations with different batch sizes.

        Args:
            data_count: Number of knowledge units to add
            query_count: Number of queries to perform
            trials: Number of trials for each batch size

        Returns:
            Benchmark results
        """
        logger.info(
            f"Benchmarking search operations with {data_count} units and {query_count} queries..."
        )

        results = {
            "batch_sizes": self.batch_sizes,
            "throughput": [],
            "time_per_query": [],
            "total_time": [],
        }

        # Generate test data
        test_data = self._generate_test_data(data_count)

        # Generate test queries
        topics = [
            "Python",
            "machine learning",
            "data",
            "artificial intelligence",
            "NLP",
            "robotics",
            "vision",
            "deep learning",
        ]
        queries = [f"Tell me about {topic}" for topic in topics] * (query_count // len(topics) + 1)
        queries = queries[:query_count]

        # Test each batch size
        for batch_size in self.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            query_times = []

            # Run multiple trials
            for trial in range(trials):
                # Create a fresh memory manager for each trial
                memory_manager = self._create_memory_manager(batch_size)

                # Add all units first
                for unit in test_data:
                    memory_manager.add_knowledge_unit(unit)

                # Measure time to perform all queries
                start_time = time.time()

                for query in queries:
                    memory_manager.search(query, top_k=5)

                elapsed_time = time.time() - start_time
                query_times.append(elapsed_time)

            # Calculate average time across trials
            avg_time = np.mean(query_times)
            time_per_query = avg_time / query_count
            throughput = query_count / avg_time

            # Store results
            results["total_time"].append(avg_time)
            results["time_per_query"].append(time_per_query)
            results["throughput"].append(throughput)

            logger.info(
                f"Batch size {batch_size}: {avg_time:.4f}s total, "
                f"{time_per_query * 1000:.4f}ms per query, "
                f"{throughput:.1f} queries/second"
            )

        return results

    def run_benchmark(
        self, data_count: int = 1000, query_count: int = 50, trials: int = 3
    ) -> dict[str, Any]:
        """
        Run comprehensive batch processing benchmarks.

        Args:
            data_count: Number of knowledge units to use
            query_count: Number of queries to perform
            trials: Number of trials for each configuration

        Returns:
            Comprehensive benchmark results
        """
        logger.info("Running batch processing benchmark...")

        # Benchmark add operations
        add_results = self.benchmark_add_operations(data_count, trials)

        # Benchmark search operations
        search_results = self.benchmark_search_operations(data_count, query_count, trials)

        # Combine results
        results = {
            "batch_sizes": self.batch_sizes,
            "add_operations": add_results,
            "search_operations": search_results,
            "configuration": {
                "data_count": data_count,
                "query_count": query_count,
                "trials": trials,
            },
        }

        logger.info("Batch processing benchmark completed")

        # Save results
        self.results = results
        output_path = os.path.join(self.output_dir, "batch_benchmark.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {output_path}")

        return results

    def plot_results(self) -> None:
        """Plot the batch processing benchmark results."""
        if not self.results:
            logger.warning("No benchmark results to plot. Run benchmark first.")
            return

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data
        batch_sizes = self.results["batch_sizes"]

        # Plot add operation throughput
        add_throughput = self.results["add_operations"]["throughput"]
        axes[0, 0].plot(batch_sizes, add_throughput, "bo-")
        axes[0, 0].set_title("Add Operation Throughput")
        axes[0, 0].set_xlabel("Batch Size")
        axes[0, 0].set_ylabel("Throughput (units/second)")
        axes[0, 0].grid(True)

        # Plot add operation time per unit
        add_time_per_unit = self.results["add_operations"]["time_per_unit"]
        axes[0, 1].plot(batch_sizes, np.array(add_time_per_unit) * 1000, "bo-")  # Convert to ms
        axes[0, 1].set_title("Add Operation Time Per Unit")
        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Time (ms/unit)")
        axes[0, 1].grid(True)

        # Plot search operation throughput
        search_throughput = self.results["search_operations"]["throughput"]
        axes[1, 0].plot(batch_sizes, search_throughput, "ro-")
        axes[1, 0].set_title("Search Operation Throughput")
        axes[1, 0].set_xlabel("Batch Size")
        axes[1, 0].set_ylabel("Throughput (queries/second)")
        axes[1, 0].grid(True)

        # Plot search operation time per query
        search_time_per_query = self.results["search_operations"]["time_per_query"]
        axes[1, 1].plot(batch_sizes, np.array(search_time_per_query) * 1000, "ro-")  # Convert to ms
        axes[1, 1].set_title("Search Operation Time Per Query")
        axes[1, 1].set_xlabel("Batch Size")
        axes[1, 1].set_ylabel("Time (ms/query)")
        axes[1, 1].grid(True)

        # Adjust layout and save figure
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(self.output_dir, "batch_benchmark.png")
        plt.savefig(output_path)
        logger.info(f"Benchmark plot saved to {output_path}")

        # Show the plot (if in interactive environment)
        plt.close()

    def find_optimal_batch_size(self) -> dict[str, Any]:
        """
        Find the optimal batch size based on benchmark results.

        Returns:
            Dictionary with optimal batch sizes for different operations
        """
        if not self.results:
            logger.warning("No benchmark results to analyze. Run benchmark first.")
            return {}

        # Find optimal batch size for add operations (highest throughput)
        add_throughput = self.results["add_operations"]["throughput"]
        optimal_add_idx = np.argmax(add_throughput)
        optimal_add_batch = self.results["batch_sizes"][optimal_add_idx]
        optimal_add_throughput = add_throughput[optimal_add_idx]

        # Find optimal batch size for search operations (highest throughput)
        search_throughput = self.results["search_operations"]["throughput"]
        optimal_search_idx = np.argmax(search_throughput)
        optimal_search_batch = self.results["batch_sizes"][optimal_search_idx]
        optimal_search_throughput = search_throughput[optimal_search_idx]

        # Calculate improvement over no batching (batch size = 1)
        baseline_add_throughput = self.results["add_operations"]["throughput"][0]
        baseline_search_throughput = self.results["search_operations"]["throughput"][0]

        add_improvement = (
            (optimal_add_throughput - baseline_add_throughput) / baseline_add_throughput
        ) * 100
        search_improvement = (
            (optimal_search_throughput - baseline_search_throughput) / baseline_search_throughput
        ) * 100

        return {
            "optimal_add_batch_size": optimal_add_batch,
            "optimal_add_throughput": optimal_add_throughput,
            "add_improvement_percent": add_improvement,
            "optimal_search_batch_size": optimal_search_batch,
            "optimal_search_throughput": optimal_search_throughput,
            "search_improvement_percent": search_improvement,
        }

    def print_results(self) -> None:
        """Print the benchmark results in a readable format."""
        if not self.results:
            logger.warning("No benchmark results to print. Run benchmark first.")
            return

        print("\n" + "=" * 60)
        print("BATCH PROCESSING BENCHMARK RESULTS")
        print("=" * 60)

        print("\nADD OPERATIONS:")
        print("Batch Size | Throughput (units/s) | Time Per Unit (ms)")
        print("-" * 60)

        for i, batch_size in enumerate(self.results["batch_sizes"]):
            throughput = self.results["add_operations"]["throughput"][i]
            time_per_unit = (
                self.results["add_operations"]["time_per_unit"][i] * 1000
            )  # Convert to ms

            print(f"{batch_size:>10d} | {throughput:>20.2f} | {time_per_unit:>17.4f}")

        print("\nSEARCH OPERATIONS:")
        print("Batch Size | Throughput (queries/s) | Time Per Query (ms)")
        print("-" * 60)

        for i, batch_size in enumerate(self.results["batch_sizes"]):
            throughput = self.results["search_operations"]["throughput"][i]
            time_per_query = (
                self.results["search_operations"]["time_per_query"][i] * 1000
            )  # Convert to ms

            print(f"{batch_size:>10d} | {throughput:>22.2f} | {time_per_query:>18.4f}")

        # Print optimal batch sizes
        optimal = self.find_optimal_batch_size()

        print("\nOPTIMAL BATCH SIZES:")
        print(
            f"Add Operations:    {optimal['optimal_add_batch_size']} "
            f"(+{optimal['add_improvement_percent']:.1f}% throughput)"
        )
        print(
            f"Search Operations: {optimal['optimal_search_batch_size']} "
            f"(+{optimal['search_improvement_percent']:.1f}% throughput)"
        )

        print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    benchmark = BatchBenchmark(
        {
            "batch_sizes": [1, 5, 10, 20, 50, 100],  # Example batch sizes
            "output_dir": "benchmark_results",
        }
    )
    results = benchmark.run_benchmark(
        data_count=500,
        query_count=20,
        trials=2,  # Smaller for quick testing
    )
    benchmark.print_results()
    benchmark.plot_results()
