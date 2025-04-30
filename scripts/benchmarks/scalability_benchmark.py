"""
Benchmark module for measuring scalability performance.

This module evaluates:
1. How performance scales with increasing knowledge base size
2. Memory usage patterns with growing data
3. Query latency as a function of data size
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil

from src.ltm_agent.core.models import KnowledgeUnit
from src.ltm_agent.memory.batch_processor import BatchProcessor
from src.ltm_agent.memory.contextualizer import SimpleContextualizer
from src.ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from src.ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from src.ltm_agent.memory.in_memory_store import InMemoryVectorStore
from src.ltm_agent.memory.knowledge_graph import KnowledgeGraph
from src.ltm_agent.memory.manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScalabilityBenchmark:
    """Benchmark for measuring scalability performance."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the scalability benchmark.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.results = {}
        self.baseline_memory_manager = None
        self.enhanced_memory_manager = None
        self.data_sizes = self.config.get("data_sizes", [10, 100, 500, 1000, 2000, 5000])
        self.output_dir = self.config.get("output_dir", "benchmark_results")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self) -> None:
        """
        Set up benchmark environment with both baseline and enhanced memory managers.
        """
        logger.info("Setting up scalability benchmark environment...")

        # Create baseline memory manager (simple in-memory implementation)
        baseline_store = InMemoryVectorStore()
        baseline_contextualizer = SimpleContextualizer()
        self.baseline_memory_manager = MemoryManager(
            memory_store=baseline_store, contextualizer=baseline_contextualizer
        )

        # Create enhanced memory manager (with knowledge graph and batch processing)
        enhanced_store = InMemoryVectorStore()
        knowledge_graph = KnowledgeGraph()
        enhanced_contextualizer = EnhancedContextualizer()
        batch_processor = BatchProcessor(batch_size=50)

        self.enhanced_memory_manager = EnhancedMemoryManager(
            memory_store=enhanced_store,
            contextualizer=enhanced_contextualizer,
            knowledge_graph=knowledge_graph,
            batch_processor=batch_processor,
        )

        logger.info("Benchmark environment setup complete")

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
            "Reinforcement learning",
            "Cognitive science",
        ]

        for i in range(count):
            topic_idx = i % len(topics)
            topic = topics[topic_idx]

            # Create knowledge unit with content related to the topic
            # Vary the length to better simulate real data
            content_length = np.random.randint(50, 300)
            content = f"This is knowledge about {topic}: fact #{i + 1}. " + "X" * content_length

            unit = KnowledgeUnit(
                id=str(uuid.uuid4()),
                text=content,
                created_at=datetime.now(),
                metadata={"topic": topic, "index": i, "length": len(content)},
            )

            knowledge_units.append(unit)

        return knowledge_units

    def _measure_memory_usage(self) -> float:
        """
        Measure current memory usage.

        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def run_benchmark(self, query_count: int = 10) -> dict[str, Any]:
        """
        Run the scalability benchmark comparing baseline and enhanced systems.

        Args:
            query_count: Number of queries to perform at each data size

        Returns:
            Benchmark results
        """
        logger.info(f"Running scalability benchmark with data sizes: {self.data_sizes}")

        # Setup if not already done
        if not self.baseline_memory_manager or not self.enhanced_memory_manager:
            self.setup()

        # Prepare queries
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
        queries = [f"Tell me about {topic}" for topic in topics]
        queries = queries * (query_count // len(queries) + 1)
        queries = queries[:query_count]

        # Initialize results structure
        results = {
            "baseline": {
                "data_sizes": self.data_sizes,
                "load_times": [],
                "memory_usage": [],
                "avg_query_times": [],
            },
            "enhanced": {
                "data_sizes": self.data_sizes,
                "load_times": [],
                "memory_usage": [],
                "avg_query_times": [],
            },
        }

        # Test each data size
        for size in self.data_sizes:
            logger.info(f"Testing with data size: {size}")

            # Generate test data for this size
            test_data = self._generate_test_data(size)

            # Clear previous data
            if hasattr(self.baseline_memory_manager, "clear"):
                self.baseline_memory_manager.clear()
            if hasattr(self.enhanced_memory_manager, "clear"):
                self.enhanced_memory_manager.clear()

            # Reset memory measurement
            _ = self._measure_memory_usage()

            # Benchmark baseline system
            logger.info(f"Benchmarking baseline memory system with {size} knowledge units...")

            # Load data into baseline system
            baseline_start_time = time.time()
            for unit in test_data:
                self.baseline_memory_manager.add_knowledge_unit(unit)
            baseline_load_time = time.time() - baseline_start_time
            results["baseline"]["load_times"].append(baseline_load_time)

            # Measure memory usage
            baseline_memory = self._measure_memory_usage()
            results["baseline"]["memory_usage"].append(baseline_memory)

            # Run queries
            baseline_query_times = []
            for query in queries:
                start_time = time.time()
                _ = self.baseline_memory_manager.search(query, top_k=5)
                query_time = time.time() - start_time
                baseline_query_times.append(query_time)

            # Record average query time
            avg_baseline_query_time = np.mean(baseline_query_times)
            results["baseline"]["avg_query_times"].append(avg_baseline_query_time)

            # Reset memory measurement
            _ = self._measure_memory_usage()

            # Benchmark enhanced system
            logger.info(f"Benchmarking enhanced memory system with {size} knowledge units...")

            # Load data into enhanced system
            enhanced_start_time = time.time()
            for unit in test_data:
                self.enhanced_memory_manager.add_knowledge_unit(unit)
            enhanced_load_time = time.time() - enhanced_start_time
            results["enhanced"]["load_times"].append(enhanced_load_time)

            # Measure memory usage
            enhanced_memory = self._measure_memory_usage()
            results["enhanced"]["memory_usage"].append(enhanced_memory)

            # Run queries
            enhanced_query_times = []
            for query in queries:
                start_time = time.time()
                _ = self.enhanced_memory_manager.search(query, top_k=5)
                query_time = time.time() - start_time
                enhanced_query_times.append(query_time)

            # Record average query time
            avg_enhanced_query_time = np.mean(enhanced_query_times)
            results["enhanced"]["avg_query_times"].append(avg_enhanced_query_time)

            logger.info(f"Completed testing with data size: {size}")

        logger.info("Scalability benchmark completed")
        self.results = results
        return results

    def plot_results(self) -> None:
        """Plot the scalability benchmark results."""
        if not self.results:
            logger.warning("No benchmark results to plot. Run benchmark first.")
            return

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Plot load times
        axes[0].plot(
            self.results["baseline"]["data_sizes"],
            self.results["baseline"]["load_times"],
            "bo-",
            label="Baseline",
        )
        axes[0].plot(
            self.results["enhanced"]["data_sizes"],
            self.results["enhanced"]["load_times"],
            "ro-",
            label="Enhanced",
        )
        axes[0].set_title("Load Time vs. Data Size")
        axes[0].set_xlabel("Number of Knowledge Units")
        axes[0].set_ylabel("Load Time (seconds)")
        axes[0].grid(True)
        axes[0].legend()

        # Plot memory usage
        axes[1].plot(
            self.results["baseline"]["data_sizes"],
            self.results["baseline"]["memory_usage"],
            "bo-",
            label="Baseline",
        )
        axes[1].plot(
            self.results["enhanced"]["data_sizes"],
            self.results["enhanced"]["memory_usage"],
            "ro-",
            label="Enhanced",
        )
        axes[1].set_title("Memory Usage vs. Data Size")
        axes[1].set_xlabel("Number of Knowledge Units")
        axes[1].set_ylabel("Memory Usage (MB)")
        axes[1].grid(True)
        axes[1].legend()

        # Plot query times
        axes[2].plot(
            self.results["baseline"]["data_sizes"],
            self.results["baseline"]["avg_query_times"],
            "bo-",
            label="Baseline",
        )
        axes[2].plot(
            self.results["enhanced"]["data_sizes"],
            self.results["enhanced"]["avg_query_times"],
            "ro-",
            label="Enhanced",
        )
        axes[2].set_title("Query Time vs. Data Size")
        axes[2].set_xlabel("Number of Knowledge Units")
        axes[2].set_ylabel("Average Query Time (seconds)")
        axes[2].grid(True)
        axes[2].legend()

        # Adjust layout and save figure
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(self.output_dir, "scalability_benchmark.png")
        plt.savefig(output_path)
        logger.info(f"Benchmark plot saved to {output_path}")

        # Show the plot (if in interactive environment)
        plt.close()

    def calculate_improvement_ratios(self) -> dict[str, Any]:
        """
        Calculate performance improvement ratios between enhanced and baseline.

        Returns:
            Dictionary of improvement metrics
        """
        if not self.results:
            logger.warning("No benchmark results to analyze. Run benchmark first.")
            return {}

        improvements = {
            "data_sizes": self.results["baseline"]["data_sizes"],
            "load_time_ratios": [],
            "memory_usage_ratios": [],
            "query_time_ratios": [],
        }

        # Calculate ratios (baseline / enhanced, so >1 means enhanced is better)
        for i in range(len(self.results["baseline"]["data_sizes"])):
            # Load time ratio
            if self.results["enhanced"]["load_times"][i] > 0:
                load_ratio = (
                    self.results["baseline"]["load_times"][i]
                    / self.results["enhanced"]["load_times"][i]
                )
            else:
                load_ratio = 1.0
            improvements["load_time_ratios"].append(load_ratio)

            # Memory usage ratio
            if self.results["enhanced"]["memory_usage"][i] > 0:
                memory_ratio = (
                    self.results["baseline"]["memory_usage"][i]
                    / self.results["enhanced"]["memory_usage"][i]
                )
            else:
                memory_ratio = 1.0
            improvements["memory_usage_ratios"].append(memory_ratio)

            # Query time ratio
            if self.results["enhanced"]["avg_query_times"][i] > 0:
                query_ratio = (
                    self.results["baseline"]["avg_query_times"][i]
                    / self.results["enhanced"]["avg_query_times"][i]
                )
            else:
                query_ratio = 1.0
            improvements["query_time_ratios"].append(query_ratio)

        return improvements

    def print_results(self) -> None:
        """Print the benchmark results in a readable format."""
        if not self.results:
            logger.warning("No benchmark results to print. Run benchmark first.")
            return

        print("\n" + "=" * 60)
        print("SCALABILITY BENCHMARK RESULTS")
        print("=" * 60)

        print("\nData Size | Load Time (s) | Memory (MB) | Query Time (s)")
        print("-" * 60)

        for i, size in enumerate(self.results["baseline"]["data_sizes"]):
            baseline_load = self.results["baseline"]["load_times"][i]
            baseline_memory = self.results["baseline"]["memory_usage"][i]
            baseline_query = self.results["baseline"]["avg_query_times"][i]

            enhanced_load = self.results["enhanced"]["load_times"][i]
            enhanced_memory = self.results["enhanced"]["memory_usage"][i]
            enhanced_query = self.results["enhanced"]["avg_query_times"][i]

            print(f"Size: {size:5d} | ", end="")
            print(f"BL: {baseline_load:.4f} | {baseline_memory:.2f} | {baseline_query:.4f}")
            print("         | ", end="")
            print(f"EN: {enhanced_load:.4f} | {enhanced_memory:.2f} | {enhanced_query:.4f}")

            # Calculate improvement percentages
            load_imp = (
                ((baseline_load - enhanced_load) / baseline_load) * 100 if baseline_load > 0 else 0
            )
            mem_imp = (
                ((baseline_memory - enhanced_memory) / baseline_memory) * 100
                if baseline_memory > 0
                else 0
            )
            query_imp = (
                ((baseline_query - enhanced_query) / baseline_query) * 100
                if baseline_query > 0
                else 0
            )

            print("         | ", end="")
            print(f"Imp: {load_imp:+.1f}% | {mem_imp:+.1f}% | {query_imp:+.1f}%")
            print("-" * 60)

        # Calculate average improvements
        improvements = self.calculate_improvement_ratios()
        avg_load_ratio = np.mean(improvements["load_time_ratios"])
        avg_memory_ratio = np.mean(improvements["memory_usage_ratios"])
        avg_query_ratio = np.mean(improvements["query_time_ratios"])

        print("\nAVERAGE IMPROVEMENT RATIOS (Baseline/Enhanced, >1 means Enhanced is better)")
        print(f"  Load Time Ratio: {avg_load_ratio:.2f}x")
        print(f"  Memory Usage Ratio: {avg_memory_ratio:.2f}x")
        print(f"  Query Time Ratio: {avg_query_ratio:.2f}x")

        print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    benchmark = ScalabilityBenchmark(
        {
            "data_sizes": [10, 100, 500, 1000],  # Smaller sizes for quick testing
            "output_dir": "benchmark_results",
        }
    )
    benchmark.setup()
    results = benchmark.run_benchmark(query_count=5)
    benchmark.print_results()
    benchmark.plot_results()
