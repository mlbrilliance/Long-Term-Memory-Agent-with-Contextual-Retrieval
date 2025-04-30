#!/usr/bin/env python
"""
Correction System Monitoring Dashboard

This script provides a simple dashboard to visualize the performance of
the memory system's correction capabilities over time. It reads metrics
from the correction_metrics.json file and displays key performance indicators.

Usage:
    python scripts/correction_monitor.py [--metrics-file path/to/metrics.json]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Matplotlib not found. Install with 'pip install matplotlib' for visualization features.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("correction_monitor")

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


class CorrectionMetrics:
    """Class to load, analyze and visualize correction metrics."""

    def __init__(self, metrics_file: str = "correction_metrics.json"):
        """Initialize with path to metrics file."""
        self.metrics_file = metrics_file
        self.metrics = {}
        self.timestamps = []
        self.correction_rates = []
        self.application_rates = []
        self.similarity_scores = []
        self.decay_values = []

    def load_metrics(self) -> bool:
        """Load metrics from file."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file) as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {self.metrics_file}")
                self._process_metrics()
                return True
            else:
                logger.warning(f"Metrics file {self.metrics_file} not found")

                # Create sample data for demonstration
                self._create_sample_data()
                return True
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return False

    def _create_sample_data(self):
        """Create sample metrics data for demonstration."""
        logger.info("Creating sample metrics data for demonstration")

        # Create 30 days of sample data
        base_date = datetime.now() - timedelta(days=30)

        self.metrics = {
            "timestamps": [],
            "daily_metrics": {},
            "cumulative_metrics": {
                "total_interactions": 0,
                "total_corrections": 0,
                "total_correction_applications": 0,
            },
        }

        # Generate simulated daily metrics with improvement trend
        for day in range(31):
            date_str = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")

            # Simulate metrics with improving trend
            correction_rate = min(0.95, 0.65 + (day * 0.01))  # Improves from 65% to 95%
            application_rate = min(0.90, 0.50 + (day * 0.015))  # Improves from 50% to 90%
            similarity = min(0.85, 0.60 + (day * 0.008))  # Improves from 60% to 85%
            decay = max(0.05, 0.30 - (day * 0.008))  # Improves from 30% to 5%

            # Add random fluctuation
            import random

            correction_rate += random.uniform(-0.05, 0.05)
            application_rate += random.uniform(-0.07, 0.07)
            similarity += random.uniform(-0.05, 0.05)
            decay += random.uniform(-0.03, 0.03)

            # Ensure values are in valid ranges
            correction_rate = max(0.1, min(1.0, correction_rate))
            application_rate = max(0.1, min(1.0, application_rate))
            similarity = max(0.1, min(1.0, similarity))
            decay = max(0.0, min(0.5, decay))

            # Create daily metrics
            interactions = random.randint(80, 150)
            corrections = int(interactions * random.uniform(0.05, 0.15))
            applications = int(corrections * application_rate)

            daily_metrics = {
                "interactions": interactions,
                "corrections": corrections,
                "correction_applications": applications,
                "correction_rate": correction_rate,
                "application_rate": application_rate,
                "avg_similarity": similarity,
                "decay_factor": decay,
            }

            # Update cumulative metrics
            self.metrics["cumulative_metrics"]["total_interactions"] += interactions
            self.metrics["cumulative_metrics"]["total_corrections"] += corrections
            self.metrics["cumulative_metrics"]["total_correction_applications"] += applications

            # Add to daily metrics
            self.metrics["timestamps"].append(date_str)
            self.metrics["daily_metrics"][date_str] = daily_metrics

        # Process the sample data
        self._process_metrics()

    def _process_metrics(self):
        """Process loaded metrics into series for analysis and visualization."""
        if not self.metrics:
            return

        self.timestamps = self.metrics.get("timestamps", [])

        # Extract metric series
        daily_metrics = self.metrics.get("daily_metrics", {})

        for timestamp in self.timestamps:
            if timestamp in daily_metrics:
                day_data = daily_metrics[timestamp]
                self.correction_rates.append(day_data.get("correction_rate", 0))
                self.application_rates.append(day_data.get("application_rate", 0))
                self.similarity_scores.append(day_data.get("avg_similarity", 0))
                self.decay_values.append(day_data.get("decay_factor", 0))

    def get_summary_stats(self) -> dict[str, Any]:
        """Calculate summary statistics from the metrics."""
        if not self.metrics:
            return {"error": "No metrics data available"}

        cumulative = self.metrics.get("cumulative_metrics", {})

        # Calculate averages from all daily metrics
        avg_correction_rate = sum(self.correction_rates) / max(1, len(self.correction_rates))
        avg_application_rate = sum(self.application_rates) / max(1, len(self.application_rates))
        avg_similarity = sum(self.similarity_scores) / max(1, len(self.similarity_scores))
        avg_decay = sum(self.decay_values) / max(1, len(self.decay_values))

        # Calculate trend (improvement over time)
        if len(self.application_rates) >= 7:
            recent_application = sum(self.application_rates[-7:]) / 7
            earlier_application = sum(self.application_rates[:-7]) / max(
                1, len(self.application_rates[:-7])
            )
            application_trend = recent_application - earlier_application
        else:
            application_trend = 0

        return {
            "total_interactions": cumulative.get("total_interactions", 0),
            "total_corrections": cumulative.get("total_corrections", 0),
            "total_applications": cumulative.get("total_correction_applications", 0),
            "avg_correction_rate": avg_correction_rate,
            "avg_application_rate": avg_application_rate,
            "avg_similarity": avg_similarity,
            "avg_decay": avg_decay,
            "application_trend": application_trend,
            "data_points": len(self.timestamps),
        }

    def visualize_metrics(self, output_file: str | None = None):
        """Visualize the metrics using matplotlib."""
        if not HAS_VISUALIZATION:
            print("Visualization requires matplotlib. Install with 'pip install matplotlib'")
            return

        if not self.timestamps:
            print("No data to visualize")
            return

        # Create a grid of plots
        plt.figure(figsize=(12, 10))
        plt.suptitle("Memory System Correction Performance", fontsize=16)

        # Use a subset of timestamps for better readability
        num_ticks = min(10, len(self.timestamps))
        tick_indices = np.linspace(0, len(self.timestamps) - 1, num_ticks, dtype=int)
        tick_labels = [self.timestamps[i] for i in tick_indices]

        # Plot 1: Correction and Application Rates
        plt.subplot(2, 2, 1)
        plt.plot(self.correction_rates, "b-", label="Correction Rate")
        plt.plot(self.application_rates, "g-", label="Application Rate")
        plt.title("Correction Effectiveness")
        plt.ylabel("Rate (0-1)")
        plt.grid(True, alpha=0.3)
        plt.xticks(tick_indices, tick_labels, rotation=45)
        plt.legend()

        # Plot 2: Similarity Scores
        plt.subplot(2, 2, 2)
        plt.plot(self.similarity_scores, "r-", label="Similarity")
        plt.title("Correction-Response Similarity")
        plt.ylabel("Similarity (0-1)")
        plt.grid(True, alpha=0.3)
        plt.xticks(tick_indices, tick_labels, rotation=45)

        # Plot 3: Decay Values
        plt.subplot(2, 2, 3)
        plt.plot(self.decay_values, "m-", label="Decay")
        plt.title("Correction Influence Decay")
        plt.xlabel("Date")
        plt.ylabel("Decay Factor (0-1)")
        plt.grid(True, alpha=0.3)
        plt.xticks(tick_indices, tick_labels, rotation=45)

        # Plot 4: 7-day Moving Average of Application Rate
        plt.subplot(2, 2, 4)

        # Calculate moving average
        window_size = min(7, len(self.application_rates))
        moving_avg = []
        for i in range(len(self.application_rates) - window_size + 1):
            window_avg = sum(self.application_rates[i : i + window_size]) / window_size
            moving_avg.append(window_avg)

        # Pad start of moving average to match original data length
        padding = [moving_avg[0]] * (len(self.application_rates) - len(moving_avg))
        padded_avg = padding + moving_avg

        plt.plot(self.application_rates, "g-", alpha=0.3, label="Daily Rate")
        plt.plot(padded_avg, "g-", linewidth=2, label="7-day Avg")
        plt.title("Application Rate Trend (7-day MA)")
        plt.xlabel("Date")
        plt.ylabel("Rate (0-1)")
        plt.grid(True, alpha=0.3)
        plt.xticks(tick_indices, tick_labels, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save or display
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()


def display_text_dashboard(metrics: CorrectionMetrics):
    """Display a text-based dashboard of metrics."""
    stats = metrics.get_summary_stats()

    print("\n" + "=" * 60)
    print(" " * 15 + "CORRECTION SYSTEM DASHBOARD")
    print("=" * 60 + "\n")

    print("SUMMARY STATISTICS:")
    print(f"- Total interactions: {stats['total_interactions']:,}")
    print(f"- Total corrections applied: {stats['total_corrections']:,}")
    print(f"- Correction application rate: {stats['avg_application_rate']:.2%}")
    print(f"- Average correction-response similarity: {stats['avg_similarity']:.2%}")

    print("\nPERFORMANCE INDICATORS:")

    # Performance rating based on application rate
    application_rate = stats["avg_application_rate"]
    if application_rate >= 0.85:
        rating = "EXCELLENT"
    elif application_rate >= 0.70:
        rating = "GOOD"
    elif application_rate >= 0.50:
        rating = "FAIR"
    else:
        rating = "NEEDS IMPROVEMENT"

    print(f"- Overall correction effectiveness: {rating}")

    # Trend indicator
    trend = stats["application_trend"]
    if trend >= 0.05:
        trend_msg = "STRONG IMPROVEMENT"
    elif trend >= 0.02:
        trend_msg = "MODERATE IMPROVEMENT"
    elif trend > -0.02:
        trend_msg = "STABLE"
    elif trend > -0.05:
        trend_msg = "MODERATE DECLINE"
    else:
        trend_msg = "SIGNIFICANT DECLINE"

    print(f"- Recent trend: {trend_msg} ({trend:.1%} change)")

    print("\nRECOMMENDATIONS:")

    if application_rate < 0.50:
        print("- CRITICAL: Review correction detection and application mechanisms")
        print("- Increase priority boost for correction units")
        print("- Check LLM prompt templates to emphasize corrections")
    elif application_rate < 0.70:
        print("- Improve correction format consistency")
        print("- Refine similarity thresholds for correction retrieval")
    elif trend < -0.02:
        print("- Monitor declining trend in correction application")
        print("- Check for changes in query patterns or knowledge base growth")
    else:
        print("- System performing well, continue monitoring")

    print("\n" + "=" * 60)


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description="Correction System Monitoring Dashboard")
    parser.add_argument(
        "--metrics-file", default="correction_metrics.json", help="Path to metrics JSON file"
    )
    parser.add_argument("--output", help="Save visualization to file")
    parser.add_argument(
        "--text-only", action="store_true", help="Display text-only dashboard (no visualization)"
    )
    args = parser.parse_args()

    # Load and process metrics
    metrics = CorrectionMetrics(args.metrics_file)
    if not metrics.load_metrics():
        print("Failed to load metrics data. Exiting.")
        return 1

    # Display text dashboard
    display_text_dashboard(metrics)

    # Visualize metrics if not text-only mode
    if not args.text_only:
        metrics.visualize_metrics(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
