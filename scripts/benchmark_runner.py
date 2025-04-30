"""
Benchmark Runner for LTM Agent.

This script provides benchmarking capabilities to track performance
improvements over time by running standardized tests and storing results.
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure directories
log_dir = Path(__file__).parent.parent / "logs" / "benchmarks"
log_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path(__file__).parent.parent / "logs" / "benchmarks" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
reports_dir = Path(__file__).parent.parent / "logs" / "benchmarks" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "benchmark_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger("benchmark_runner")


class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, description: str):
        """Initialize the benchmark."""
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        self.metrics = {}

    async def run(self) -> dict[str, Any]:
        """
        Run the benchmark and return metrics.

        This method should be overridden by specific benchmark implementations.

        Returns:
            Dict with benchmark metrics
        """
        raise NotImplementedError("Benchmark implementations must override the run method")

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark metadata to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
        }


class MemoryConnectionsBenchmark(Benchmark):
    """Benchmark for testing memory connections capabilities."""

    def __init__(self):
        """Initialize the memory connections benchmark."""
        super().__init__(
            name="memory_connections",
            description="Tests the agent's ability to create and retrieve knowledge connections",
        )

    async def run(self) -> dict[str, Any]:
        """Run the memory connections benchmark."""
        # Import the memory connections test
        from test_memory_connections import test_memory_connections

        # Run the test
        logger.info("Running memory connections benchmark")
        results = await test_memory_connections()

        # Extract metrics
        self.metrics = {
            "success": results.get("success", False),
            "connection_quality": results.get("connection_quality", 0.0),
            "knowledge_units": results.get("knowledge_units", 0),
            "connected_retrievals": results.get("connected_retrievals", 0),
        }

        return self.metrics


class CoreMemoryBenchmark(Benchmark):
    """Benchmark for testing core memory capabilities."""

    def __init__(self):
        """Initialize the core memory benchmark."""
        super().__init__(
            name="core_memory",
            description="Tests the fundamental mechanisms of knowledge storage and retrieval",
        )

    async def run(self) -> dict[str, Any]:
        """Run the core memory benchmark."""
        # Import the core memory test
        from core_memory_test import test_core_memory_capabilities

        # Run the test
        logger.info("Running core memory benchmark")
        results = await test_core_memory_capabilities()

        # Extract metrics
        success_rate = results.get("success_rate", "0/0")
        stage_success = results.get("stage_success", "0/0")

        # Parse success rates
        try:
            success_steps, total_steps = map(int, success_rate.split("/"))
            success_stages, total_stages = map(int, stage_success.split("/"))

            step_success_rate = success_steps / total_steps if total_steps > 0 else 0.0
            stage_success_rate = success_stages / total_stages if total_stages > 0 else 0.0
        except:
            step_success_rate = 0.0
            stage_success_rate = 0.0

        self.metrics = {
            "step_success_rate": step_success_rate,
            "stage_success_rate": stage_success_rate,
            "error_count": len(results.get("errors", [])),
        }

        return self.metrics


class ConversationBenchmark(Benchmark):
    """Benchmark for testing conversation capabilities."""

    def __init__(self, scenario: str | None = None):
        """
        Initialize the conversation benchmark.

        Args:
            scenario: Optional specific scenario to test (None for all scenarios)
        """
        name = f"conversation_{scenario}" if scenario else "conversation_all"
        super().__init__(
            name=name,
            description=f"Tests the agent's performance in {'a specific' if scenario else 'all'} conversation scenario(s)",
        )
        self.scenario = scenario

    async def run(self) -> dict[str, Any]:
        """Run the conversation benchmark."""
        # Import the conversation simulator
        from conversation_simulator import ConversationSimulator

        # Initialize simulator
        simulator = ConversationSimulator(
            enable_consolidation=True, results_dir=str(results_dir / f"conversation_{self.id}")
        )

        # Run scenario(s)
        logger.info(f"Running conversation benchmark: {self.name}")

        if self.scenario:
            results = await simulator.run_scenario(self.scenario)
            overall_score = results["evaluation"].get("overall_score", 0.0)
            successful_outcomes = sum(
                1
                for outcome in results["evaluation"].values()
                if isinstance(outcome, dict) and outcome.get("success", False)
            )
            total_outcomes = sum(
                1
                for outcome in results["evaluation"].values()
                if isinstance(outcome, dict) and "success" in outcome
            )

            self.metrics = {
                "overall_score": overall_score,
                "successful_outcomes": successful_outcomes,
                "total_outcomes": total_outcomes,
                "success_rate": successful_outcomes / total_outcomes if total_outcomes > 0 else 0.0,
            }
        else:
            results = await simulator.run_all_scenarios()
            self.metrics = {
                "overall_score": results["summary"].get("overall_score", 0.0),
                "successful_scenarios": results["summary"].get("successful_scenarios", 0),
                "total_scenarios": results.get("scenarios_run", 0),
                "success_rate": (
                    results["summary"].get("successful_scenarios", 0)
                    / results.get("scenarios_run", 1)
                ),
            }

        return self.metrics


class BenchmarkSuite:
    """A suite of benchmarks to run together."""

    def __init__(self, name: str, description: str):
        """Initialize the benchmark suite."""
        self.name = name
        self.description = description
        self.benchmarks = []
        self.id = str(uuid.uuid4())

    def add_benchmark(self, benchmark: Benchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    async def run(self) -> dict[str, Any]:
        """
        Run all benchmarks in the suite.

        Returns:
            Dict with results from all benchmarks
        """
        results = {
            "suite_id": self.id,
            "suite_name": self.name,
            "description": self.description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": {},
            "summary": {
                "total_benchmarks": len(self.benchmarks),
                "successful_benchmarks": 0,
                "average_score": 0.0,
            },
        }

        successful_benchmarks = 0
        scores = []

        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            try:
                metrics = await benchmark.run()

                # Determine if benchmark was successful
                success = False
                if "success" in metrics:
                    success = metrics["success"]
                elif "success_rate" in metrics:
                    success = metrics["success_rate"] >= 0.6
                elif "overall_score" in metrics:
                    success = metrics["overall_score"] >= 0.6

                if success:
                    successful_benchmarks += 1

                # Calculate score for averaging
                score = 0.0
                if "overall_score" in metrics:
                    score = metrics["overall_score"]
                elif "success_rate" in metrics:
                    score = metrics["success_rate"]
                elif "connection_quality" in metrics:
                    score = metrics["connection_quality"] / 100.0

                scores.append(score)

                # Store benchmark results
                results["benchmarks"][benchmark.name] = {
                    "id": benchmark.id,
                    "description": benchmark.description,
                    "success": success,
                    "metrics": metrics,
                }

                logger.info(f"Benchmark {benchmark.name} completed with success={success}")

            except Exception as e:
                logger.error(f"Error running benchmark {benchmark.name}: {str(e)}")
                results["benchmarks"][benchmark.name] = {
                    "id": benchmark.id,
                    "description": benchmark.description,
                    "success": False,
                    "error": str(e),
                }

        # Update summary
        results["summary"]["successful_benchmarks"] = successful_benchmarks
        results["summary"]["average_score"] = sum(scores) / len(scores) if scores else 0.0

        return results


class BenchmarkRunner:
    """Manages running benchmarks and tracking results over time."""

    def __init__(self, storage_path: str | None = None):
        """
        Initialize the benchmark runner.

        Args:
            storage_path: Optional path to store benchmark results
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = results_dir

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "benchmark_history.json"
        self.history = self._load_history()

    def _load_history(self) -> dict[str, Any]:
        """Load benchmark history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except:
                logger.warning("Could not load benchmark history, creating new history")

        # Create new history if file doesn't exist or couldn't be loaded
        return {"runs": [], "latest_run_id": None, "total_runs": 0}

    def _save_history(self):
        """Save benchmark history to file."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _save_run_results(self, run_id: str, results: dict[str, Any]):
        """Save benchmark run results to file."""
        run_file = self.storage_path / f"run_{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(results, f, indent=2)

    async def run_suite(self, suite: BenchmarkSuite) -> dict[str, Any]:
        """
        Run a benchmark suite and store results.

        Args:
            suite: The benchmark suite to run

        Returns:
            Dict with benchmark results
        """
        # Run the suite
        logger.info(f"Running benchmark suite: {suite.name}")
        results = await suite.run()

        # Generate run ID
        run_id = str(uuid.uuid4())
        results["run_id"] = run_id

        # Save detailed results
        self._save_run_results(run_id, results)

        # Update history
        self.history["runs"].append(
            {
                "run_id": run_id,
                "suite_name": suite.name,
                "timestamp": results["timestamp"],
                "successful_benchmarks": results["summary"]["successful_benchmarks"],
                "total_benchmarks": results["summary"]["total_benchmarks"],
                "average_score": results["summary"]["average_score"],
            }
        )
        self.history["latest_run_id"] = run_id
        self.history["total_runs"] += 1

        # Save updated history
        self._save_history()

        logger.info(f"Benchmark suite {suite.name} completed. Run ID: {run_id}")
        return results

    def get_run_results(self, run_id: str) -> dict[str, Any] | None:
        """
        Get results for a specific benchmark run.

        Args:
            run_id: The run ID

        Returns:
            Dict with benchmark results or None if not found
        """
        run_file = self.storage_path / f"run_{run_id}.json"
        if run_file.exists():
            with open(run_file) as f:
                return json.load(f)
        return None

    def get_latest_run(self) -> dict[str, Any] | None:
        """
        Get the latest benchmark run results.

        Returns:
            Dict with benchmark results or None if no runs exist
        """
        if self.history["latest_run_id"]:
            return self.get_run_results(self.history["latest_run_id"])
        return None

    def get_benchmark_history(self, benchmark_name: str) -> list[dict[str, Any]]:
        """
        Get history of a specific benchmark across runs.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            List of benchmark results over time
        """
        history = []

        for run_entry in self.history["runs"]:
            run_id = run_entry["run_id"]
            run_results = self.get_run_results(run_id)

            if run_results and benchmark_name in run_results["benchmarks"]:
                benchmark_result = run_results["benchmarks"][benchmark_name]
                history.append(
                    {
                        "run_id": run_id,
                        "timestamp": run_results["timestamp"],
                        "success": benchmark_result["success"],
                        "metrics": benchmark_result["metrics"],
                    }
                )

        return history

    def generate_benchmark_report(self, benchmark_name: str) -> str:
        """
        Generate a report for a specific benchmark's performance over time.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Path to the generated report
        """
        history = self.get_benchmark_history(benchmark_name)

        if not history:
            logger.warning(f"No history found for benchmark: {benchmark_name}")
            return None

        # Prepare data for plotting
        timestamps = []
        scores = []

        for entry in history:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            timestamps.append(timestamp)

            # Extract score based on available metrics
            metrics = entry["metrics"]
            score = 0.0

            if "overall_score" in metrics:
                score = metrics["overall_score"]
            elif "success_rate" in metrics:
                score = metrics["success_rate"]
            elif "connection_quality" in metrics:
                score = metrics["connection_quality"] / 100.0

            scores.append(score)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, scores, marker="o", linestyle="-")
        plt.title(f"{benchmark_name} Performance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.grid(True)

        # Format dates on x-axis
        date_formatter = DateFormatter("%Y-%m-%d %H:%M")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.gcf().autofmt_xdate()

        # Save plot
        report_path = reports_dir / f"{benchmark_name}_history.png"
        plt.savefig(report_path)
        plt.close()

        # Generate text report
        text_report_path = reports_dir / f"{benchmark_name}_report.txt"
        with open(text_report_path, "w") as f:
            f.write(f"Benchmark Report: {benchmark_name}\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write(f"Total Runs: {len(history)}\n")
            f.write(f"First Run: {timestamps[0] if timestamps else 'N/A'}\n")
            f.write(f"Latest Run: {timestamps[-1] if timestamps else 'N/A'}\n\n")

            f.write("Performance Trend:\n")
            if len(scores) >= 2:
                initial_score = scores[0]
                latest_score = scores[-1]
                change = latest_score - initial_score
                change_percent = (change / initial_score) * 100 if initial_score > 0 else 0

                f.write(f"Initial Score: {initial_score:.2f}\n")
                f.write(f"Latest Score: {latest_score:.2f}\n")
                f.write(f"Change: {change:.2f} ({change_percent:+.1f}%)\n\n")
            else:
                f.write(f"Only one run available, score: {scores[0] if scores else 'N/A'}\n\n")

            f.write("Detailed History:\n")
            for i, entry in enumerate(history):
                f.write(f"Run {i + 1}: {timestamps[i].strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"  Score: {scores[i]:.2f}\n")
                f.write(f"  Success: {entry['success']}\n")
                f.write("  Metrics:\n")
                for metric, value in entry["metrics"].items():
                    f.write(f"    {metric}: {value}\n")
                f.write("\n")

        logger.info(f"Generated benchmark report: {text_report_path}")
        return str(text_report_path)

    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all benchmarks over time.

        Returns:
            Path to the generated report
        """
        if not self.history["runs"]:
            logger.warning("No benchmark runs found")
            return None

        # Prepare data for plotting
        timestamps = []
        average_scores = []
        success_rates = []

        for run_entry in self.history["runs"]:
            timestamp = datetime.fromisoformat(run_entry["timestamp"])
            timestamps.append(timestamp)

            average_scores.append(run_entry["average_score"])
            success_rate = (
                run_entry["successful_benchmarks"] / run_entry["total_benchmarks"]
                if run_entry["total_benchmarks"] > 0
                else 0
            )
            success_rates.append(success_rate)

        # Create plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, average_scores, marker="o", linestyle="-", color="blue")
        plt.title("Average Benchmark Score Over Time")
        plt.ylabel("Average Score")
        plt.ylim(0, 1.1)
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, success_rates, marker="s", linestyle="-", color="green")
        plt.title("Benchmark Success Rate Over Time")
        plt.xlabel("Date")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1.1)
        plt.grid(True)

        # Format dates on x-axis
        date_formatter = DateFormatter("%Y-%m-%d %H:%M")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.gcf().autofmt_xdate()

        plt.tight_layout()

        # Save plot
        report_path = reports_dir / "benchmark_summary.png"
        plt.savefig(report_path)
        plt.close()

        # Generate text report
        text_report_path = reports_dir / "benchmark_summary_report.txt"
        with open(text_report_path, "w") as f:
            f.write("Benchmark Summary Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write(f"Total Runs: {len(self.history['runs'])}\n")
            f.write(f"First Run: {timestamps[0] if timestamps else 'N/A'}\n")
            f.write(f"Latest Run: {timestamps[-1] if timestamps else 'N/A'}\n\n")

            f.write("Performance Trend:\n")
            if len(average_scores) >= 2:
                initial_score = average_scores[0]
                latest_score = average_scores[-1]
                change = latest_score - initial_score
                change_percent = (change / initial_score) * 100 if initial_score > 0 else 0

                f.write(f"Initial Average Score: {initial_score:.2f}\n")
                f.write(f"Latest Average Score: {latest_score:.2f}\n")
                f.write(f"Change: {change:.2f} ({change_percent:+.1f}%)\n\n")

                initial_success = success_rates[0]
                latest_success = success_rates[-1]
                success_change = latest_success - initial_success
                success_change_percent = (
                    (success_change / initial_success) * 100 if initial_success > 0 else 0
                )

                f.write(f"Initial Success Rate: {initial_success:.2f}\n")
                f.write(f"Latest Success Rate: {latest_success:.2f}\n")
                f.write(f"Change: {success_change:.2f} ({success_change_percent:+.1f}%)\n\n")
            else:
                f.write("Only one run available:\n")
                f.write(f"Average Score: {average_scores[0] if average_scores else 'N/A'}\n")
                f.write(f"Success Rate: {success_rates[0] if success_rates else 'N/A'}\n\n")

            f.write("Individual Benchmark Trends:\n")
            latest_run = self.get_latest_run()
            if latest_run:
                for benchmark_name in latest_run["benchmarks"]:
                    f.write(f"\n{benchmark_name}:\n")
                    history = self.get_benchmark_history(benchmark_name)
                    if len(history) >= 2:
                        first_metrics = history[0]["metrics"]
                        last_metrics = history[-1]["metrics"]

                        # Try to find a common metric to compare
                        for metric in ["overall_score", "success_rate", "connection_quality"]:
                            if metric in first_metrics and metric in last_metrics:
                                first_value = first_metrics[metric]
                                last_value = last_metrics[metric]

                                # Adjust connection_quality to 0-1 scale
                                if metric == "connection_quality":
                                    first_value /= 100.0
                                    last_value /= 100.0

                                change = last_value - first_value
                                change_percent = (
                                    (change / first_value) * 100 if first_value > 0 else 0
                                )

                                f.write(f"  {metric}: {first_value:.2f} -> {last_value:.2f} ")
                                f.write(f"({change:+.2f}, {change_percent:+.1f}%)\n")
                                break
                        else:
                            f.write("  No comparable metrics found\n")
                    else:
                        f.write("  Insufficient history for trend analysis\n")

        logger.info(f"Generated summary report: {text_report_path}")
        return str(text_report_path)


def create_standard_suite() -> BenchmarkSuite:
    """Create the standard benchmark suite."""
    suite = BenchmarkSuite(
        name="standard_suite", description="Standard suite of benchmarks for the LTM agent"
    )

    # Add benchmarks
    suite.add_benchmark(CoreMemoryBenchmark())
    suite.add_benchmark(MemoryConnectionsBenchmark())
    suite.add_benchmark(ConversationBenchmark("programming_languages"))

    return suite


async def main():
    """Run the benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark runner for LTM agent")
    parser.add_argument("--run", action="store_true", help="Run the standard benchmark suite")
    parser.add_argument("--report", type=str, help="Generate report for a specific benchmark")
    parser.add_argument("--summary", action="store_true", help="Generate summary report")
    parser.add_argument("--storage", type=str, help="Path to store benchmark results")
    args = parser.parse_args()

    runner = BenchmarkRunner(storage_path=args.storage)

    if args.run:
        suite = create_standard_suite()
        results = await runner.run_suite(suite)
        print("\nBenchmark suite completed.")
        print(f"Average Score: {results['summary']['average_score']:.2f}")
        print(
            f"Successful Benchmarks: {results['summary']['successful_benchmarks']}/{results['summary']['total_benchmarks']}"
        )

    if args.report:
        report_path = runner.generate_benchmark_report(args.report)
        if report_path:
            print(f"\nBenchmark report generated: {report_path}")
        else:
            print(f"\nNo data available for benchmark: {args.report}")

    if args.summary:
        summary_path = runner.generate_summary_report()
        if summary_path:
            print(f"\nSummary report generated: {summary_path}")
        else:
            print("\nNo benchmark data available for summary report")

    # If no arguments provided, show usage
    if not (args.run or args.report or args.summary):
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
