"""
Monitoring and telemetry system for the Long-Term Memory Agent.

This module provides tools for tracking performance, health metrics,
and usage patterns of the memory system.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceMetric:
    """Track a single performance metric over time."""

    def __init__(self, name: str, window_size: int = 100, alert_threshold: float | None = None):
        """
        Initialize a performance metric.

        Args:
            name: Name of the metric
            window_size: Number of values to retain for moving statistics
            alert_threshold: Optional threshold for alerting (if metric exceeds this value)
        """
        self.name = name
        self.values = deque(maxlen=window_size)
        self.alert_threshold = alert_threshold
        self.total = 0
        self.count = 0
        self.min_value = float("inf")
        self.max_value = float("-inf")
        self.alert_callbacks = []

    def add_value(self, value: float) -> None:
        """
        Add a new value to the metric.

        Args:
            value: Value to add
        """
        self.values.append(value)
        self.total += value
        self.count += 1

        # Update min/max
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

        # Check alert threshold
        if self.alert_threshold is not None and value > self.alert_threshold:
            for callback in self.alert_callbacks:
                callback(self.name, value, self.alert_threshold)

    def get_average(self) -> float:
        """
        Get the average value over all recorded values.

        Returns:
            Average value
        """
        return self.total / self.count if self.count > 0 else 0

    def get_moving_average(self) -> float:
        """
        Get the moving average over the recent window.

        Returns:
            Moving average
        """
        return sum(self.values) / len(self.values) if self.values else 0

    def add_alert_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """
        Add a callback to be notified when metric exceeds threshold.

        Args:
            callback: Function taking (metric_name, value, threshold)
        """
        self.alert_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics for this metric.

        Returns:
            Dictionary of statistics
        """
        return {
            "name": self.name,
            "count": self.count,
            "average": self.get_average(),
            "moving_average": self.get_moving_average(),
            "min": self.min_value if self.count > 0 else None,
            "max": self.max_value if self.count > 0 else None,
            "current": self.values[-1] if self.values else None,
            "alert_threshold": self.alert_threshold,
        }


class MemoryMonitor:
    """
    Central monitoring system for memory operations.

    Tracks performance metrics, system health, and provides logging
    and alerting capabilities.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the memory monitor.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.metrics = {}
        self.operation_timers = {}
        self.start_time = time.time()
        self.event_log = deque(maxlen=self.config.get("event_log_size", 1000))
        self.telemetry_thread = None
        self.telemetry_running = False
        self.telemetry_interval = self.config.get("telemetry_interval_seconds", 60)
        self.telemetry_handlers = []
        self.memory_snapshots = deque(maxlen=self.config.get("snapshot_count", 100))

        # Initialize default metrics
        self._initialize_default_metrics()

        # Start telemetry if enabled
        if self.config.get("enable_telemetry", False):
            self.start_telemetry()

    def _initialize_default_metrics(self) -> None:
        """Initialize default performance metrics."""
        # Retrieval metrics
        self.create_metric("retrieval_time", alert_threshold=1.0)  # Alert if retrieval takes > 1s
        self.create_metric("retrieval_count", alert_threshold=None)
        self.create_metric("retrieval_results", alert_threshold=None)

        # Storage metrics
        self.create_metric("storage_time", alert_threshold=1.0)  # Alert if storage takes > 1s
        self.create_metric("storage_count", alert_threshold=None)

        # Embedding metrics
        self.create_metric("embedding_time", alert_threshold=2.0)  # Alert if embedding takes > 2s
        self.create_metric("embedding_count", alert_threshold=None)

        # Processing metrics
        self.create_metric("contextualizing_time", alert_threshold=3.0)
        self.create_metric("pruning_time", alert_threshold=5.0)

        # System metrics
        self.create_metric(
            "memory_usage_mb", alert_threshold=self.config.get("memory_alert_threshold_mb", 1000)
        )
        self.create_metric(
            "cpu_usage_percent", alert_threshold=self.config.get("cpu_alert_threshold_percent", 90)
        )

    def create_metric(self, name: str, alert_threshold: float | None = None) -> None:
        """
        Create a new metric to track.

        Args:
            name: Name of the metric
            alert_threshold: Value that should trigger alerts if exceeded
        """
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(
                name, self.config.get("metric_window_size", 100), alert_threshold
            )

            # Add default alert callback
            if alert_threshold is not None:
                self.metrics[name].add_alert_callback(self._default_alert_handler)

    def _default_alert_handler(self, metric_name: str, value: float, threshold: float) -> None:
        """
        Default handler for metric alerts.

        Args:
            metric_name: Name of the metric
            value: Current value
            threshold: Alert threshold
        """
        logger.warning(f"Metric alert: {metric_name} = {value:.4f} (threshold: {threshold:.4f})")

        # Log the event
        self.log_event(
            "alert",
            {
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "timestamp": time.time(),
            },
        )

    def record_metric(self, name: str, value: float) -> None:
        """
        Record a value for a metric.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.create_metric(name)

        self.metrics[name].add_value(value)

    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.

        Args:
            operation: Name of the operation
        """
        self.operation_timers[operation] = time.time()

    def stop_timer(self, operation: str) -> float:
        """
        Stop timing an operation and record the duration.

        Args:
            operation: Name of the operation

        Returns:
            Elapsed time in seconds
        """
        if operation not in self.operation_timers:
            logger.warning(f"Timer for operation '{operation}' was never started")
            return 0

        start_time = self.operation_timers.pop(operation)
        elapsed = time.time() - start_time

        # Record as a metric
        metric_name = f"{operation}_time"
        self.record_metric(metric_name, elapsed)

        return elapsed

    def log_event(self, event_type: str, details: dict[str, Any]) -> None:
        """
        Log an event for later analysis.

        Args:
            event_type: Type of event
            details: Event details
        """
        event = {"type": event_type, "timestamp": time.time(), "details": details}

        self.event_log.append(event)

        # Log important events
        if event_type in ["error", "alert", "warning"]:
            logger.warning(f"Event: {event_type} - {details}")

    def get_uptime(self) -> float:
        """
        Get monitor uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary of metric summaries
        """
        return {name: metric.get_stats() for name, metric in self.metrics.items()}

    def get_recent_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get recent events, optionally filtered by type.

        Args:
            event_type: Optional type to filter by
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        if event_type:
            events = [e for e in self.event_log if e["type"] == event_type]
        else:
            events = list(self.event_log)

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        return events[:limit]

    def take_system_snapshot(self) -> dict[str, Any]:
        """
        Take a snapshot of system metrics.

        Returns:
            Dictionary of system metrics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        snapshot = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
        }

        # Record metrics
        self.record_metric("memory_usage_mb", snapshot["memory_usage_mb"])
        self.record_metric("cpu_usage_percent", snapshot["cpu_percent"])

        # Add to snapshots
        self.memory_snapshots.append(snapshot)

        return snapshot

    def _telemetry_worker(self) -> None:
        """Background worker for collecting telemetry."""
        while self.telemetry_running:
            try:
                # Take system snapshot
                snapshot = self.take_system_snapshot()

                # Get metrics summary
                metrics = self.get_metrics_summary()

                # Combine into telemetry data
                telemetry_data = {
                    "timestamp": time.time(),
                    "uptime": self.get_uptime(),
                    "snapshot": snapshot,
                    "metrics": metrics,
                }

                # Call handlers
                for handler in self.telemetry_handlers:
                    try:
                        handler(telemetry_data)
                    except Exception as e:
                        logger.error(f"Error in telemetry handler: {e}")

            except Exception as e:
                logger.error(f"Error in telemetry worker: {e}")

            # Sleep until next interval
            time.sleep(self.telemetry_interval)

    def start_telemetry(self) -> None:
        """Start background telemetry collection."""
        if self.telemetry_thread is not None and self.telemetry_thread.is_alive():
            logger.warning("Telemetry already running")
            return

        self.telemetry_running = True
        self.telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self.telemetry_thread.start()
        logger.info("Started telemetry collection")

    def stop_telemetry(self) -> None:
        """Stop background telemetry collection."""
        self.telemetry_running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1.0)
            self.telemetry_thread = None
            logger.info("Stopped telemetry collection")

    def add_telemetry_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """
        Add a handler for telemetry data.

        Args:
            handler: Function to call with telemetry data
        """
        self.telemetry_handlers.append(handler)

    def save_metrics_to_file(self, filepath: str) -> None:
        """
        Save current metrics to a JSON file.

        Args:
            filepath: Path to save JSON file
        """
        try:
            data = {
                "timestamp": time.time(),
                "uptime": self.get_uptime(),
                "metrics": self.get_metrics_summary(),
                "snapshot": (
                    self.take_system_snapshot()
                    if not self.memory_snapshots
                    else self.memory_snapshots[-1]
                ),
                "events": list(self.event_log),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved metrics to {filepath}")

        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")


class FileBasedTelemetryHandler:
    """Handler that saves telemetry data to rolling log files."""

    def __init__(self, log_dir: str, max_files: int = 10, max_file_size_mb: int = 10):
        """
        Initialize the file-based telemetry handler.

        Args:
            log_dir: Directory to store log files
            max_files: Maximum number of log files to keep
            max_file_size_mb: Maximum size of each log file in MB
        """
        self.log_dir = log_dir
        self.max_files = max_files
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.current_file = None
        self.current_size = 0

        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Initialize first log file
        self._rotate_if_needed()

    def _get_log_filename(self) -> str:
        """
        Generate a log filename based on current time.

        Returns:
            Log filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"telemetry_{timestamp}.jsonl")

    def _rotate_if_needed(self) -> None:
        """Rotate log file if it's too large or doesn't exist."""
        # Check if current file exists and is too large
        if (
            self.current_file is not None
            and os.path.exists(self.current_file)
            and os.path.getsize(self.current_file) < self.max_file_size_bytes
        ):
            return

        # Create new log file
        self.current_file = self._get_log_filename()
        self.current_size = 0

        # Cleanup old files if needed
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Delete oldest log files if there are too many."""
        log_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("telemetry_") and filename.endswith(".jsonl"):
                filepath = os.path.join(self.log_dir, filename)
                log_files.append((filepath, os.path.getmtime(filepath)))

        # Sort by modification time (oldest first)
        log_files.sort(key=lambda x: x[1])

        # Delete oldest files beyond the limit
        files_to_delete = log_files[: -self.max_files] if len(log_files) > self.max_files else []
        for filepath, _ in files_to_delete:
            try:
                os.remove(filepath)
                logger.info(f"Deleted old telemetry file: {filepath}")
            except Exception as e:
                logger.error(f"Error deleting old telemetry file {filepath}: {e}")

    def __call__(self, telemetry_data: dict[str, Any]) -> None:
        """
        Handle telemetry data by writing to log file.

        Args:
            telemetry_data: Telemetry data to write
        """
        self._rotate_if_needed()

        try:
            # Serialize data as a JSON line
            json_line = json.dumps(telemetry_data) + "\n"
            line_bytes = len(json_line.encode("utf-8"))

            # Write to file
            with open(self.current_file, "a") as f:
                f.write(json_line)

            # Update size
            self.current_size += line_bytes

        except Exception as e:
            logger.error(f"Error writing telemetry data to file: {e}")


# Create singleton monitor instance
_global_monitor = None


def get_monitor(config: dict[str, Any] | None = None) -> MemoryMonitor:
    """
    Get or create the global monitor instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Global MemoryMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(config)
    return _global_monitor
