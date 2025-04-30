"""
Evaluation Analyzer - Analyzes existing test results and benchmark data.

This script examines test results, benchmark data, and memory performance to
identify areas for improvement in the Long-Term Memory Agent.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure directories
analysis_dir = Path(__file__).parent.parent / "logs" / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(analysis_dir / "evaluation_analyzer.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("evaluation_analyzer")


class EvaluationAnalyzer:
    """Analyzes evaluation results and identifies areas for improvement."""

    def __init__(self, output_dir: str | None = None):
        """
        Initialize the evaluation analyzer.

        Args:
            output_dir: Optional directory to save analysis results
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = analysis_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_sources = {}

    def load_test_results(self, test_type: str, results_file: str) -> bool:
        """
        Load test results from a file.

        Args:
            test_type: Type of test (e.g., 'memory_connections', 'core_memory')
            results_file: Path to the results file

        Returns:
            bool: True if loading was successful
        """
        try:
            file_path = Path(results_file)
            if not file_path.exists():
                logger.error(f"Results file not found: {file_path}")
                return False

            with open(file_path) as f:
                try:
                    data = json.load(f)
                    self.data_sources[test_type] = {
                        "file": str(file_path),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": data,
                    }
                    logger.info(f"Loaded {test_type} test results from {file_path}")
                    return True
                except json.JSONDecodeError:
                    # Try loading as plain text if not JSON
                    f.seek(0)
                    content = f.read()
                    self.data_sources[test_type] = {
                        "file": str(file_path),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {"raw_content": content},
                    }
                    logger.info(f"Loaded {test_type} test results as plain text from {file_path}")
                    return True
        except Exception as e:
            logger.error(f"Error loading {test_type} test results: {str(e)}")
            return False

    def load_benchmark_history(self, benchmark_history_file: str) -> bool:
        """
        Load benchmark history from file.

        Args:
            benchmark_history_file: Path to the benchmark history file

        Returns:
            bool: True if loading was successful
        """
        try:
            file_path = Path(benchmark_history_file)
            if not file_path.exists():
                logger.error(f"Benchmark history file not found: {file_path}")
                return False

            with open(file_path) as f:
                data = json.load(f)
                self.data_sources["benchmark_history"] = {
                    "file": str(file_path),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": data,
                }
                logger.info(f"Loaded benchmark history from {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error loading benchmark history: {str(e)}")
            return False

    def scan_logs_directory(self, logs_dir: str) -> dict[str, list[str]]:
        """
        Scan logs directory for relevant test and benchmark results.

        Args:
            logs_dir: Path to logs directory

        Returns:
            Dict mapping data types to lists of file paths
        """
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            logger.error(f"Logs directory not found: {logs_path}")
            return {}

        found_files = {
            "core_memory": [],
            "memory_connections": [],
            "conversation": [],
            "benchmark": [],
        }

        # Walk through the logs directory
        for root, dirs, files in os.walk(logs_path):
            for file in files:
                file_path = Path(root) / file

                # Check file type based on name
                if "core_memory" in file.lower() and file.endswith((".json", ".txt")):
                    found_files["core_memory"].append(str(file_path))
                elif "memory_connections" in file.lower() and file.endswith((".json", ".txt")):
                    found_files["memory_connections"].append(str(file_path))
                elif "conversation" in file.lower() and file.endswith((".json", ".txt")):
                    found_files["conversation"].append(str(file_path))
                elif "benchmark" in file.lower() and file.endswith((".json", ".txt")):
                    found_files["benchmark"].append(str(file_path))

        return found_files

    def load_latest_results(self, logs_dir: str) -> int:
        """
        Load the latest results for each test and benchmark type.

        Args:
            logs_dir: Path to logs directory

        Returns:
            int: Number of data sources loaded
        """
        found_files = self.scan_logs_directory(logs_dir)
        loaded = 0

        # Load the most recent file of each type (based on modification time)
        for test_type, files in found_files.items():
            if files:
                # Sort by modification time (newest first)
                sorted_files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)
                if self.load_test_results(test_type, sorted_files[0]):
                    loaded += 1

        logger.info(f"Loaded {loaded} data sources from logs directory")
        return loaded

    def analyze(self) -> dict[str, Any]:
        """
        Analyze loaded data and identify improvement areas.

        Returns:
            Dict with analysis results
        """
        if not self.data_sources:
            logger.error("No data sources loaded for analysis")
            return {"error": "No data sources loaded"}

        analysis_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_sources": {k: v["file"] for k, v in self.data_sources.items()},
            "findings": [],
            "improvement_areas": [],
            "metrics": {},
        }

        # Extract metrics
        metrics = {}

        # Extract core memory metrics
        if "core_memory" in self.data_sources:
            core_data = self.data_sources["core_memory"]["data"]
            if isinstance(core_data, dict):
                if "success_rate" in core_data:
                    metrics["core_memory_success_rate"] = core_data["success_rate"]
                    # Parse x/y format if needed
                    if (
                        isinstance(metrics["core_memory_success_rate"], str)
                        and "/" in metrics["core_memory_success_rate"]
                    ):
                        x, y = metrics["core_memory_success_rate"].split("/")
                        if y != "0":
                            metrics["core_memory_success_rate"] = float(x) / float(y)
                        else:
                            metrics["core_memory_success_rate"] = 0.0

                if "stage_success" in core_data:
                    metrics["core_memory_stage_success"] = core_data["stage_success"]
                    # Parse x/y format if needed
                    if (
                        isinstance(metrics["core_memory_stage_success"], str)
                        and "/" in metrics["core_memory_stage_success"]
                    ):
                        x, y = metrics["core_memory_stage_success"].split("/")
                        if y != "0":
                            metrics["core_memory_stage_success"] = float(x) / float(y)
                        else:
                            metrics["core_memory_stage_success"] = 0.0

                # Analyze stages
                if "stages" in core_data:
                    failed_stages = [
                        stage
                        for stage, data in core_data["stages"].items()
                        if not data.get("success", False)
                    ]

                    if failed_stages:
                        analysis_results["findings"].append(
                            {
                                "type": "core_memory_failed_stages",
                                "description": f"Failed stages in core memory test: {', '.join(failed_stages)}",
                                "severity": "medium",
                            }
                        )

                        for stage in failed_stages:
                            analysis_results["improvement_areas"].append(
                                {
                                    "area": f"core_memory_{stage.lower().replace(' ', '_')}",
                                    "description": f"Improve {stage} functionality",
                                    "priority": "high",
                                }
                            )

        # Extract memory connections metrics
        if "memory_connections" in self.data_sources:
            conn_data = self.data_sources["memory_connections"]["data"]
            if isinstance(conn_data, dict):
                if "connection_quality" in conn_data:
                    metrics["memory_connection_quality"] = conn_data["connection_quality"]

                if "connected_retrievals" in conn_data and "knowledge_units" in conn_data:
                    connected = conn_data["connected_retrievals"]
                    total = conn_data["knowledge_units"]
                    if total > 0:
                        metrics["memory_connection_ratio"] = connected / total
                    else:
                        metrics["memory_connection_ratio"] = 0.0

                # Check success
                if not conn_data.get("success", True):
                    analysis_results["findings"].append(
                        {
                            "type": "memory_connections_failure",
                            "description": "Memory connections test failed",
                            "severity": "high",
                        }
                    )

                    analysis_results["improvement_areas"].append(
                        {
                            "area": "memory_relation_tracking",
                            "description": "Improve tracking and retrieval of related knowledge units",
                            "priority": "high",
                        }
                    )

        # Benchmark history analysis
        if "benchmark_history" in self.data_sources:
            history_data = self.data_sources["benchmark_history"]["data"]
            if "runs" in history_data and history_data["runs"]:
                # Compare first and last runs
                first_run = history_data["runs"][0]
                last_run = history_data["runs"][-1]

                metrics["benchmark_runs"] = len(history_data["runs"])
                metrics["benchmark_first_score"] = first_run.get("average_score", 0.0)
                metrics["benchmark_latest_score"] = last_run.get("average_score", 0.0)

                # Calculate improvement
                improvement = metrics["benchmark_latest_score"] - metrics["benchmark_first_score"]
                metrics["benchmark_improvement"] = improvement

                if improvement <= 0:
                    analysis_results["findings"].append(
                        {
                            "type": "benchmark_regression",
                            "description": f"Benchmark performance has not improved (change: {improvement:.2f})",
                            "severity": "high",
                        }
                    )

                    analysis_results["improvement_areas"].append(
                        {
                            "area": "overall_performance",
                            "description": "Investigate performance regression in benchmarks",
                            "priority": "critical",
                        }
                    )

        # Store metrics in results
        analysis_results["metrics"] = metrics

        # Add generic findings based on combined analysis
        if "memory_connection_quality" in metrics and metrics["memory_connection_quality"] < 50:
            analysis_results["findings"].append(
                {
                    "type": "poor_connection_quality",
                    "description": f"Memory connection quality is low ({metrics['memory_connection_quality']:.1f}%)",
                    "severity": "high",
                }
            )

            analysis_results["improvement_areas"].append(
                {
                    "area": "connection_quality",
                    "description": "Improve quality of connections between knowledge units",
                    "priority": "high",
                }
            )

        # Memory utilization findings
        if "core_memory_success_rate" in metrics and metrics["core_memory_success_rate"] < 0.7:
            analysis_results["findings"].append(
                {
                    "type": "core_memory_issues",
                    "description": f"Core memory capabilities need improvement (success rate: {metrics['core_memory_success_rate']:.2f})",
                    "severity": "medium",
                }
            )

            analysis_results["improvement_areas"].append(
                {
                    "area": "core_memory_robustness",
                    "description": "Strengthen basic memory storage and retrieval operations",
                    "priority": "medium",
                }
            )

        # Prioritize improvement areas
        analysis_results["improvement_areas"] = sorted(
            analysis_results["improvement_areas"],
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["priority"], 4),
        )

        # Generate visualizations
        self._generate_visualizations(metrics, analysis_results["findings"])

        return analysis_results

    def _generate_visualizations(self, metrics: dict[str, Any], findings: list[dict[str, Any]]):
        """
        Generate visualizations for analysis results.

        Args:
            metrics: Collected metrics
            findings: Analysis findings
        """
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Performance metrics visualization
        if metrics:
            plt.figure(figsize=(10, 6))

            # Filter metrics to numerical values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            if numeric_metrics:
                # Create bar chart of metrics
                plt.bar(numeric_metrics.keys(), numeric_metrics.values())
                plt.xlabel("Metric")
                plt.ylabel("Value")
                plt.title("Performance Metrics Summary")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the chart
                metrics_chart_path = self.output_dir / f"metrics_summary_{timestamp}.png"
                plt.savefig(metrics_chart_path)
                plt.close()
                logger.info(f"Generated metrics visualization: {metrics_chart_path}")

        # 2. Findings by severity
        if findings:
            severity_counts = {}
            for finding in findings:
                severity = finding.get("severity", "unknown")
                if severity not in severity_counts:
                    severity_counts[severity] = 0
                severity_counts[severity] += 1

            if severity_counts:
                plt.figure(figsize=(8, 6))
                plt.pie(
                    severity_counts.values(),
                    labels=severity_counts.keys(),
                    autopct="%1.1f%%",
                    colors=(
                        ["red", "orange", "yellow", "green"]
                        if "critical" in severity_counts
                        else ["orange", "yellow", "green"]
                    ),
                )
                plt.title("Findings by Severity")
                plt.axis("equal")

                # Save the chart
                findings_chart_path = self.output_dir / f"findings_by_severity_{timestamp}.png"
                plt.savefig(findings_chart_path)
                plt.close()
                logger.info(f"Generated findings visualization: {findings_chart_path}")

    def generate_report(self, analysis_results: dict[str, Any]) -> str:
        """
        Generate a detailed analysis report.

        Args:
            analysis_results: Results from the analyze method

        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"analysis_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("# Memory Agent Performance Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data sources
            f.write("## Data Sources Analyzed\n\n")
            for source_type, source_path in analysis_results["data_sources"].items():
                f.write(f"* **{source_type}**: {source_path}\n")
            f.write("\n")

            # Metrics summary
            if analysis_results.get("metrics"):
                f.write("## Performance Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric, value in analysis_results["metrics"].items():
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                    f.write(f"| {metric.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")

            # Key findings
            if analysis_results.get("findings"):
                f.write("## Key Findings\n\n")

                for finding in analysis_results["findings"]:
                    severity = finding.get("severity", "unknown").upper()
                    f.write(f"* **[{severity}]** {finding['description']}\n")
                f.write("\n")

            # Improvement areas
            if analysis_results.get("improvement_areas"):
                f.write("## Recommended Improvement Areas\n\n")

                for i, area in enumerate(analysis_results["improvement_areas"], 1):
                    priority = area.get("priority", "unknown").upper()
                    f.write(f"### {i}. {area['area'].replace('_', ' ').title()} ({priority})\n\n")
                    f.write(f"{area['description']}\n\n")

            # Conclusion
            f.write("## Conclusion\n\n")

            # Generate appropriate conclusion based on findings
            if not analysis_results.get("improvement_areas"):
                f.write(
                    "No significant improvement areas were identified. The memory agent appears to be functioning well based on the analyzed data.\n"
                )
            else:
                num_critical = sum(
                    1
                    for area in analysis_results.get("improvement_areas", [])
                    if area.get("priority") == "critical"
                )
                num_high = sum(
                    1
                    for area in analysis_results.get("improvement_areas", [])
                    if area.get("priority") == "high"
                )

                if num_critical > 0:
                    f.write(
                        f"The analysis identified {num_critical} critical and {num_high} high-priority areas for improvement. Addressing these issues should be the immediate focus to enhance the memory agent's performance.\n"
                    )
                elif num_high > 0:
                    f.write(
                        f"The analysis identified {num_high} high-priority areas for improvement. Addressing these issues will significantly enhance the memory agent's performance.\n"
                    )
                else:
                    f.write(
                        "The analysis identified several areas for improvement. Addressing these will incrementally enhance the memory agent's performance.\n"
                    )

        logger.info(f"Generated analysis report: {report_file}")
        return str(report_file)


async def main():
    """Run the evaluation analyzer."""
    parser = argparse.ArgumentParser(description="Analyze LTM agent evaluation results")
    parser.add_argument("--logs-dir", type=str, help="Path to logs directory")
    parser.add_argument("--output-dir", type=str, help="Directory to save analysis results")
    parser.add_argument("--core-memory", type=str, help="Path to core memory test results")
    parser.add_argument(
        "--memory-connections", type=str, help="Path to memory connections test results"
    )
    parser.add_argument("--benchmark-history", type=str, help="Path to benchmark history file")
    args = parser.parse_args()

    analyzer = EvaluationAnalyzer(output_dir=args.output_dir)

    # Load data from specified sources
    data_loaded = False

    if args.logs_dir:
        num_loaded = analyzer.load_latest_results(args.logs_dir)
        data_loaded = num_loaded > 0

    if args.core_memory:
        if analyzer.load_test_results("core_memory", args.core_memory):
            data_loaded = True

    if args.memory_connections:
        if analyzer.load_test_results("memory_connections", args.memory_connections):
            data_loaded = True

    if args.benchmark_history:
        if analyzer.load_benchmark_history(args.benchmark_history):
            data_loaded = True

    if not data_loaded:
        # If no specific sources provided, look in default locations
        logs_dir = Path(__file__).parent.parent / "logs"
        analyzer.load_latest_results(str(logs_dir))

    # Perform analysis
    analysis_results = analyzer.analyze()

    # Generate report
    report_path = analyzer.generate_report(analysis_results)

    print(f"\nAnalysis complete. Report saved to: {report_path}")

    # Print key improvement areas to console
    if analysis_results.get("improvement_areas"):
        print("\nKey Improvement Areas:")
        for i, area in enumerate(analysis_results["improvement_areas"][:3], 1):
            priority = area.get("priority", "unknown").upper()
            print(f"{i}. [{priority}] {area['description']}")


if __name__ == "__main__":
    asyncio.run(main())
