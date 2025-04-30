"""
Benchmark module for measuring retrieval performance.

This module evaluates:
1. Retrieval speed (latency)
2. Retrieval accuracy (relevance of results)
3. Memory efficiency during retrieval operations
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import psutil

from src.ltm_agent.core.models import KnowledgeUnit
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


class RetrievalBenchmark:
    """Benchmark for measuring retrieval performance."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the retrieval benchmark.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.results = {}
        self.baseline_memory_manager = None
        self.enhanced_memory_manager = None

    def setup(self) -> None:
        """
        Set up benchmark environment with both baseline and enhanced memory managers.
        """
        logger.info("Setting up retrieval benchmark environment...")

        # Create baseline memory manager (simple in-memory implementation)
        baseline_store = InMemoryVectorStore()
        baseline_contextualizer = SimpleContextualizer()
        self.baseline_memory_manager = MemoryManager(
            memory_store=baseline_store, contextualizer=baseline_contextualizer
        )

        # Create enhanced memory manager (with knowledge graph and enhanced contextualizer)
        enhanced_store = InMemoryVectorStore()
        knowledge_graph = KnowledgeGraph()
        enhanced_contextualizer = EnhancedContextualizer()
        self.enhanced_memory_manager = EnhancedMemoryManager(
            memory_store=enhanced_store,
            contextualizer=enhanced_contextualizer,
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

        # Generate a set of related knowledge chunks
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

    def _measure_memory_usage(self) -> float:
        """
        Measure current memory usage.

        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def run_benchmark(
        self, knowledge_count: int = 100, query_count: int = 20, top_k: int = 5
    ) -> dict[str, Any]:
        """
        Run the retrieval benchmark comparing baseline and enhanced systems.

        Args:
            knowledge_count: Number of knowledge units to use
            query_count: Number of queries to perform
            top_k: Number of results to retrieve

        Returns:
            Benchmark results
        """
        logger.info(f"Running retrieval benchmark with {knowledge_count} knowledge units...")

        # Setup if not already done
        if not self.baseline_memory_manager or not self.enhanced_memory_manager:
            self.setup()

        # Generate test data
        knowledge_units = self._generate_test_data(knowledge_count)

        # Generate test queries
        topics = ["Python", "machine learning", "data", "artificial intelligence", "NLP"]
        queries = [f"Tell me about {topic}" for topic in topics] * (query_count // len(topics) + 1)
        queries = queries[:query_count]

        # Benchmark results
        results = {
            "baseline": {
                "load_time": 0,
                "retrieval_times": [],
                "memory_usage": 0,
                "total_results": 0,
            },
            "enhanced": {
                "load_time": 0,
                "retrieval_times": [],
                "memory_usage": 0,
                "total_results": 0,
            },
        }

        # Benchmark baseline system
        logger.info("Benchmarking baseline memory system...")

        # Load data into baseline system
        baseline_start_time = time.time()
        for unit in knowledge_units:
            self.baseline_memory_manager.add_knowledge_unit(unit)
        baseline_load_time = time.time() - baseline_start_time
        results["baseline"]["load_time"] = baseline_load_time

        # Measure memory usage
        results["baseline"]["memory_usage"] = self._measure_memory_usage()

        # Run queries
        baseline_retrieval_times = []
        baseline_total_results = 0

        for query in queries:
            start_time = time.time()
            retrieved = self.baseline_memory_manager.search(query, top_k=top_k)
            retrieval_time = time.time() - start_time

            baseline_retrieval_times.append(retrieval_time)
            baseline_total_results += len(retrieved)

        results["baseline"]["retrieval_times"] = baseline_retrieval_times
        results["baseline"]["total_results"] = baseline_total_results

        # Reset memory measurement
        _ = self._measure_memory_usage()

        # Benchmark enhanced system
        logger.info("Benchmarking enhanced memory system...")

        # Load data into enhanced system
        enhanced_start_time = time.time()
        for unit in knowledge_units:
            self.enhanced_memory_manager.add_knowledge_unit(unit)
        enhanced_load_time = time.time() - enhanced_start_time
        results["enhanced"]["load_time"] = enhanced_load_time

        # Measure memory usage
        results["enhanced"]["memory_usage"] = self._measure_memory_usage()

        # Run queries
        enhanced_retrieval_times = []
        enhanced_total_results = 0

        for query in queries:
            start_time = time.time()
            retrieved = self.enhanced_memory_manager.search(query, top_k=top_k)
            retrieval_time = time.time() - start_time

            enhanced_retrieval_times.append(retrieval_time)
            enhanced_total_results += len(retrieved)

        results["enhanced"]["retrieval_times"] = enhanced_retrieval_times
        results["enhanced"]["total_results"] = enhanced_total_results

        # Calculate summary statistics
        results["baseline"]["avg_retrieval_time"] = np.mean(baseline_retrieval_times)
        results["baseline"]["min_retrieval_time"] = np.min(baseline_retrieval_times)
        results["baseline"]["max_retrieval_time"] = np.max(baseline_retrieval_times)

        results["enhanced"]["avg_retrieval_time"] = np.mean(enhanced_retrieval_times)
        results["enhanced"]["min_retrieval_time"] = np.min(enhanced_retrieval_times)
        results["enhanced"]["max_retrieval_time"] = np.max(enhanced_retrieval_times)

        # Calculate improvement percentages
        if results["baseline"]["avg_retrieval_time"] > 0:
            speed_improvement = (
                (
                    results["baseline"]["avg_retrieval_time"]
                    - results["enhanced"]["avg_retrieval_time"]
                )
                / results["baseline"]["avg_retrieval_time"]
            ) * 100
            results["speed_improvement_percent"] = speed_improvement

        if results["baseline"]["memory_usage"] > 0:
            memory_efficiency = (
                (results["baseline"]["memory_usage"] - results["enhanced"]["memory_usage"])
                / results["baseline"]["memory_usage"]
            ) * 100
            results["memory_efficiency_percent"] = memory_efficiency

        logger.info("Retrieval benchmark completed")
        self.results = results
        return results

    def evaluate_relevance(self, sample_size: int = 20) -> dict[str, float]:
        """
        Evaluate relevance of retrieved results (requires ground truth).

        This is a simplified simulation of relevance evaluation.

        Args:
            sample_size: Number of queries to evaluate

        Returns:
            Relevance metrics
        """
        logger.info("Evaluating retrieval relevance...")

        # Generate test data
        knowledge_units = self._generate_test_data(100)

        # Create ground truth by manually associating queries with relevant unit IDs
        topics = [
            "Python programming",
            "machine learning",
            "data science",
            "artificial intelligence",
            "natural language processing",
        ]

        ground_truth = {}

        # Organize units by topic for ground truth creation
        units_by_topic = {topic: [] for topic in topics}
        for unit in knowledge_units:
            topic = unit.metadata.get("topic")
            if topic in units_by_topic:
                units_by_topic[topic].append(unit.id)

        # Create ground truth mapping (query -> relevant unit IDs)
        for topic in topics:
            query = f"Tell me about {topic.lower()}"
            ground_truth[query] = units_by_topic[topic]

        # Add units to both systems
        for unit in knowledge_units:
            self.baseline_memory_manager.add_knowledge_unit(unit)
            self.enhanced_memory_manager.add_knowledge_unit(unit)

        # Run queries and calculate relevance
        baseline_precision = []
        enhanced_precision = []

        for query, relevant_ids in list(ground_truth.items())[:sample_size]:
            # Baseline retrieval
            baseline_results = self.baseline_memory_manager.search(query, top_k=10)
            baseline_retrieved_ids = [unit.id for unit in baseline_results]

            # Enhanced retrieval
            enhanced_results = self.enhanced_memory_manager.search(query, top_k=10)
            enhanced_retrieved_ids = [unit.id for unit in enhanced_results]

            # Calculate precision@k (proportion of retrieved items that are relevant)
            baseline_relevant_count = sum(1 for id in baseline_retrieved_ids if id in relevant_ids)
            enhanced_relevant_count = sum(1 for id in enhanced_retrieved_ids if id in relevant_ids)

            baseline_precision.append(baseline_relevant_count / max(1, len(baseline_retrieved_ids)))
            enhanced_precision.append(enhanced_relevant_count / max(1, len(enhanced_retrieved_ids)))

        # Calculate average precision
        avg_baseline_precision = np.mean(baseline_precision) if baseline_precision else 0
        avg_enhanced_precision = np.mean(enhanced_precision) if enhanced_precision else 0

        # Calculate improvement
        precision_improvement = 0
        if avg_baseline_precision > 0:
            precision_improvement = (
                (avg_enhanced_precision - avg_baseline_precision) / avg_baseline_precision
            ) * 100

        return {
            "baseline_precision": avg_baseline_precision,
            "enhanced_precision": avg_enhanced_precision,
            "precision_improvement_percent": precision_improvement,
        }

    def print_results(self) -> None:
        """Print the benchmark results in a readable format."""
        if not self.results:
            logger.warning("No benchmark results to print. Run benchmark first.")
            return

        print("\n" + "=" * 50)
        print("RETRIEVAL BENCHMARK RESULTS")
        print("=" * 50)

        print("\nBASELINE SYSTEM:")
        print(f"  Load Time: {self.results['baseline']['load_time']:.4f} seconds")
        print(f"  Memory Usage: {self.results['baseline']['memory_usage']:.2f} MB")
        print(f"  Avg Retrieval Time: {self.results['baseline']['avg_retrieval_time']:.4f} seconds")
        print(f"  Min Retrieval Time: {self.results['baseline']['min_retrieval_time']:.4f} seconds")
        print(f"  Max Retrieval Time: {self.results['baseline']['max_retrieval_time']:.4f} seconds")

        print("\nENHANCED SYSTEM:")
        print(f"  Load Time: {self.results['enhanced']['load_time']:.4f} seconds")
        print(f"  Memory Usage: {self.results['enhanced']['memory_usage']:.2f} MB")
        print(f"  Avg Retrieval Time: {self.results['enhanced']['avg_retrieval_time']:.4f} seconds")
        print(f"  Min Retrieval Time: {self.results['enhanced']['min_retrieval_time']:.4f} seconds")
        print(f"  Max Retrieval Time: {self.results['enhanced']['max_retrieval_time']:.4f} seconds")

        print("\nIMPROVEMENTS:")
        if "speed_improvement_percent" in self.results:
            improvement = self.results["speed_improvement_percent"]
            if improvement > 0:
                print(f"  Speed: {improvement:.2f}% faster")
            else:
                print(f"  Speed: {-improvement:.2f}% slower")

        if "memory_efficiency_percent" in self.results:
            efficiency = self.results["memory_efficiency_percent"]
            if efficiency > 0:
                print(f"  Memory: {efficiency:.2f}% more efficient")
            else:
                print(f"  Memory: {-efficiency:.2f}% less efficient")

        print("=" * 50 + "\n")


# Example usage
if __name__ == "__main__":
    benchmark = RetrievalBenchmark()
    benchmark.setup()
    results = benchmark.run_benchmark(knowledge_count=100, query_count=10)
    benchmark.print_results()

    # Evaluate relevance
    relevance_results = benchmark.evaluate_relevance()
    print("\nRELEVANCE EVALUATION:")
    print(f"  Baseline Precision: {relevance_results['baseline_precision']:.4f}")
    print(f"  Enhanced Precision: {relevance_results['enhanced_precision']:.4f}")
    print(f"  Improvement: {relevance_results['precision_improvement_percent']:.2f}%")
