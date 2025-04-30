"""
Background maintenance system for the Long-Term Memory Agent.

This module provides automated background maintenance tasks for optimizing
memory performance, cleaning up stale data, and consolidating related memories.
"""

import json
import logging
import os
import queue
import random
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .monitoring import get_monitor
from .resilience import ResilientOperation, get_resilience_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MaintenanceTaskPriority(Enum):
    """Priority levels for maintenance tasks."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class MaintenanceTask:
    """Base class for maintenance tasks."""

    def __init__(
        self,
        name: str,
        priority: MaintenanceTaskPriority = MaintenanceTaskPriority.MEDIUM,
        interval_seconds: int | None = None,
    ):
        """
        Initialize a maintenance task.

        Args:
            name: Task name
            priority: Task priority
            interval_seconds: Optional interval for recurring tasks
        """
        self.name = name
        self.priority = priority
        self.interval_seconds = interval_seconds
        self.last_run = None
        self.next_run = None
        self.running = False
        self.stats = {
            "runs": 0,
            "failures": 0,
            "success_rate": 0,
            "total_runtime": 0,
            "avg_runtime": 0,
            "last_runtime": None,
            "last_error": None,
        }

        # If interval is set, schedule next run
        if interval_seconds is not None:
            self.schedule_next_run()

    def schedule_next_run(self, delay: int | None = None) -> None:
        """
        Schedule the next run of this task.

        Args:
            delay: Optional specific delay in seconds
        """
        if delay is not None:
            self.next_run = time.time() + delay
        elif self.interval_seconds is not None:
            # Add a small random jitter (±10%) to prevent thundering herd
            jitter = random.uniform(0.9, 1.1)
            self.next_run = time.time() + (self.interval_seconds * jitter)
        else:
            self.next_run = None

    def is_due(self) -> bool:
        """
        Check if the task is due to run.

        Returns:
            True if task is due
        """
        if self.next_run is None:
            return False
        return time.time() >= self.next_run

    def execute(self, context: dict[str, Any]) -> Any:
        """
        Execute the task.

        Args:
            context: Execution context

        Returns:
            Task result
        """
        try:
            self.running = True
            start_time = time.time()

            # Call implementation-specific run method
            result = self.run(context)

            # Update stats
            elapsed = time.time() - start_time
            self.stats["runs"] += 1
            self.stats["total_runtime"] += elapsed
            self.stats["avg_runtime"] = self.stats["total_runtime"] / self.stats["runs"]
            self.stats["last_runtime"] = elapsed
            self.stats["success_rate"] = (
                (self.stats["runs"] - self.stats["failures"]) / self.stats["runs"] * 100
            )

            # Update timestamps
            self.last_run = time.time()
            self.schedule_next_run()

            return result

        except Exception as e:
            # Update failure stats
            self.stats["failures"] += 1
            self.stats["last_error"] = str(e)
            self.stats["success_rate"] = (
                (self.stats["runs"] - self.stats["failures"]) / self.stats["runs"] * 100
            )

            # Reschedule soon for critical tasks, or normally for others
            if self.priority == MaintenanceTaskPriority.CRITICAL:
                self.schedule_next_run(delay=300)  # Try again in 5 minutes
            else:
                self.schedule_next_run()

            # Re-raise for caller to handle
            raise

        finally:
            self.running = False

    def run(self, context: dict[str, Any]) -> Any:
        """
        Implementation-specific task logic.

        Args:
            context: Execution context

        Returns:
            Task result
        """
        raise NotImplementedError("Subclasses must implement run()")


class MemoryConsolidationTask(MaintenanceTask):
    """Task to consolidate related memories."""

    def __init__(
        self,
        interval_seconds: int = 3600,  # Default: every hour
        similarity_threshold: float = 0.8,
        batch_size: int = 100,
    ):
        """
        Initialize memory consolidation task.

        Args:
            interval_seconds: Run interval in seconds
            similarity_threshold: Threshold for considering memories similar
            batch_size: Number of memories to process in one batch
        """
        super().__init__(
            name="memory_consolidation",
            priority=MaintenanceTaskPriority.MEDIUM,
            interval_seconds=interval_seconds,
        )
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run memory consolidation.

        Args:
            context: Execution context with memory_manager

        Returns:
            Consolidation results
        """
        memory_manager = context.get("memory_manager")
        if not memory_manager:
            raise ValueError("No memory_manager provided in context")

        logger.info("Starting memory consolidation task")

        # Get all knowledge units
        all_units = memory_manager.get_all_knowledge_units()

        # Skip if too few units
        if len(all_units) < 2:
            logger.info("Too few knowledge units to consolidate")
            return {"consolidated": 0, "total": len(all_units)}

        # Process in batches
        consolidated = 0
        total_processed = 0

        for i in range(0, len(all_units), self.batch_size):
            batch = all_units[i : i + self.batch_size]

            # Find similar knowledge units
            similar_groups = memory_manager.find_similar_knowledge_units(
                batch, threshold=self.similarity_threshold
            )

            # Consolidate each group
            for group in similar_groups:
                if len(group) < 2:
                    continue

                # Perform consolidation
                memory_manager.consolidate_knowledge_units(group)
                consolidated += 1

            total_processed += len(batch)
            logger.info(
                f"Processed {total_processed}/{len(all_units)} knowledge units, consolidated {consolidated} groups"
            )

        return {"consolidated": consolidated, "total": len(all_units)}


class MemoryPruningTask(MaintenanceTask):
    """Task to prune old or redundant memories."""

    def __init__(
        self,
        interval_seconds: int = 86400,  # Default: daily
        age_threshold_days: int = 90,
        relevance_threshold: float = 0.2,
        max_units_to_prune: int = 100,
    ):
        """
        Initialize memory pruning task.

        Args:
            interval_seconds: Run interval in seconds
            age_threshold_days: Age threshold for considering memories old
            relevance_threshold: Threshold for considering memories irrelevant
            max_units_to_prune: Maximum number of units to prune in one run
        """
        super().__init__(
            name="memory_pruning",
            priority=MaintenanceTaskPriority.LOW,
            interval_seconds=interval_seconds,
        )
        self.age_threshold_days = age_threshold_days
        self.relevance_threshold = relevance_threshold
        self.max_units_to_prune = max_units_to_prune

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run memory pruning.

        Args:
            context: Execution context with memory_manager

        Returns:
            Pruning results
        """
        memory_manager = context.get("memory_manager")
        if not memory_manager:
            raise ValueError("No memory_manager provided in context")

        logger.info("Starting memory pruning task")

        # Calculate age threshold
        age_threshold = datetime.now() - timedelta(days=self.age_threshold_days)

        # Get all knowledge units
        all_units = memory_manager.get_all_knowledge_units()

        # Identify candidates for pruning
        pruning_candidates = []

        for unit in all_units:
            # Old and not recently accessed
            if unit.created_at < age_threshold and (
                not unit.last_accessed_at or unit.last_accessed_at < age_threshold
            ):
                pruning_candidates.append((unit, "age"))
                continue

            # Low relevance score
            if hasattr(unit, "relevance_score") and unit.relevance_score < self.relevance_threshold:
                pruning_candidates.append((unit, "relevance"))
                continue

            # Redundant (high similarity to other units but not adding value)
            if hasattr(unit, "redundancy_score") and unit.redundancy_score > 0.8:
                pruning_candidates.append((unit, "redundancy"))
                continue

        # Sort by priority (redundancy, then relevance, then age)
        priority_map = {"redundancy": 3, "relevance": 2, "age": 1}
        pruning_candidates.sort(key=lambda x: priority_map.get(x[1], 0), reverse=True)

        # Limit number of units to prune
        pruning_candidates = pruning_candidates[: self.max_units_to_prune]

        # Perform pruning
        pruned_count = 0
        pruned_by_reason = {"age": 0, "relevance": 0, "redundancy": 0}

        for unit, reason in pruning_candidates:
            # Archive unit before pruning
            memory_manager.archive_knowledge_unit(unit.id)

            # Prune unit
            memory_manager.remove_knowledge_unit(unit.id)

            pruned_count += 1
            pruned_by_reason[reason] += 1

        logger.info(f"Pruned {pruned_count} knowledge units: {pruned_by_reason}")

        return {"pruned": pruned_count, "total": len(all_units), "by_reason": pruned_by_reason}


class DatabaseOptimizationTask(MaintenanceTask):
    """Task to optimize the vector database."""

    def __init__(self, interval_seconds: int = 604800):  # Default: weekly
        """
        Initialize database optimization task.

        Args:
            interval_seconds: Run interval in seconds
        """
        super().__init__(
            name="database_optimization",
            priority=MaintenanceTaskPriority.LOW,
            interval_seconds=interval_seconds,
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run database optimization.

        Args:
            context: Execution context with memory_store

        Returns:
            Optimization results
        """
        memory_store = context.get("memory_store")
        if not memory_store:
            raise ValueError("No memory_store provided in context")

        logger.info("Starting database optimization task")

        # Check if optimization is supported
        if not hasattr(memory_store, "optimize") or not callable(memory_store.optimize):
            logger.warning("Memory store does not support optimization")
            return {"status": "unsupported"}

        # Perform optimization
        result = memory_store.optimize()

        logger.info(f"Database optimization completed: {result}")

        return {"status": "completed", "details": result}


class MemoryBackupTask(MaintenanceTask):
    """Task to back up the memory system."""

    def __init__(
        self,
        interval_seconds: int = 86400,  # Default: daily
        backup_dir: str = "backups",
        max_backups: int = 10,
    ):
        """
        Initialize memory backup task.

        Args:
            interval_seconds: Run interval in seconds
            backup_dir: Directory to store backups
            max_backups: Maximum number of backups to keep
        """
        super().__init__(
            name="memory_backup",
            priority=MaintenanceTaskPriority.HIGH,
            interval_seconds=interval_seconds,
        )
        self.backup_dir = backup_dir
        self.max_backups = max_backups

        # Create backup directory if needed
        os.makedirs(backup_dir, exist_ok=True)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run memory backup.

        Args:
            context: Execution context with memory_manager

        Returns:
            Backup results
        """
        memory_manager = context.get("memory_manager")
        if not memory_manager:
            raise ValueError("No memory_manager provided in context")

        logger.info("Starting memory backup task")

        # Create backup timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"memory_backup_{timestamp}.json")

        # Get all knowledge units
        all_units = memory_manager.get_all_knowledge_units()

        # Create backup data
        backup_data = {
            "timestamp": timestamp,
            "count": len(all_units),
            "knowledge_units": [unit.to_dict() for unit in all_units],
        }

        # Write backup
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)

        # Clean up old backups
        self._cleanup_old_backups()

        logger.info(f"Memory backup completed: {backup_file}")

        return {"status": "completed", "file": backup_file, "count": len(all_units)}

    def _cleanup_old_backups(self) -> None:
        """Clean up old backups beyond the maximum count."""
        backups = []

        for file in os.listdir(self.backup_dir):
            if file.startswith("memory_backup_") and file.endswith(".json"):
                backup_path = os.path.join(self.backup_dir, file)
                backups.append((backup_path, os.path.getmtime(backup_path)))

        # Sort by modification time (oldest first)
        backups.sort(key=lambda x: x[1])

        # Remove oldest beyond the limit
        for file_path, _ in backups[: -self.max_backups]:
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up old backup: {file_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up old backup {file_path}: {e}")


class MemoryIntegrityCheckTask(MaintenanceTask):
    """Task to verify the integrity of the memory system."""

    def __init__(self, interval_seconds: int = 43200):  # Default: twice daily
        """
        Initialize memory integrity check task.

        Args:
            interval_seconds: Run interval in seconds
        """
        super().__init__(
            name="memory_integrity_check",
            priority=MaintenanceTaskPriority.HIGH,
            interval_seconds=interval_seconds,
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run memory integrity check.

        Args:
            context: Execution context with memory_manager and memory_store

        Returns:
            Integrity check results
        """
        memory_manager = context.get("memory_manager")
        memory_store = context.get("memory_store")

        if not memory_manager or not memory_store:
            raise ValueError("Missing memory_manager or memory_store in context")

        logger.info("Starting memory integrity check task")

        # Check 1: Consistency between manager and store
        manager_units = memory_manager.get_all_knowledge_units()
        manager_ids = {unit.id for unit in manager_units}

        store_ids = set(memory_store.get_all_ids())

        # Check for inconsistencies
        missing_in_store = manager_ids - store_ids
        missing_in_manager = store_ids - manager_ids

        # Check 2: Verify embeddings exist for all units
        missing_embeddings = []

        for unit in manager_units:
            if not memory_store.has_embedding(unit.id):
                missing_embeddings.append(unit.id)

        # Check 3: Verify relationships integrity
        relationship_errors = []

        if hasattr(memory_manager, "knowledge_graph"):
            for source, target, _ in memory_manager.knowledge_graph.get_all_relationships():
                if source not in manager_ids or target not in manager_ids:
                    relationship_errors.append((source, target))

        # Prepare results
        issues_found = (
            len(missing_in_store)
            + len(missing_in_manager)
            + len(missing_embeddings)
            + len(relationship_errors)
        )

        results = {
            "status": "issues_found" if issues_found > 0 else "ok",
            "issues_count": issues_found,
            "missing_in_store": list(missing_in_store),
            "missing_in_manager": list(missing_in_manager),
            "missing_embeddings": missing_embeddings,
            "relationship_errors": relationship_errors,
        }

        # Log issues
        if issues_found > 0:
            logger.warning(f"Memory integrity check found {issues_found} issues")

            # Attempt repairs if configured
            if context.get("auto_repair", False):
                self._repair_integrity_issues(memory_manager, memory_store, results)
                results["repairs_attempted"] = True
        else:
            logger.info("Memory integrity check completed: no issues found")

        return results

    def _repair_integrity_issues(self, memory_manager, memory_store, issues):
        """
        Attempt to repair integrity issues.

        Args:
            memory_manager: Memory manager instance
            memory_store: Memory store instance
            issues: Integrity issues dict
        """
        # Repair 1: Add missing units to store
        for unit_id in issues["missing_in_store"]:
            unit = memory_manager.get_knowledge_unit(unit_id)
            if unit:
                try:
                    memory_store.add(unit_id, unit.text, None)
                    logger.info(f"Repaired: Added missing unit {unit_id} to store")
                except Exception as e:
                    logger.error(f"Repair failed for unit {unit_id}: {e}")

        # Repair 2: Remove orphaned IDs from store
        for unit_id in issues["missing_in_manager"]:
            try:
                memory_store.delete(unit_id)
                logger.info(f"Repaired: Removed orphaned unit {unit_id} from store")
            except Exception as e:
                logger.error(f"Repair failed for orphaned unit {unit_id}: {e}")

        # Repair 3: Regenerate missing embeddings
        for unit_id in issues["missing_embeddings"]:
            unit = memory_manager.get_knowledge_unit(unit_id)
            if unit:
                try:
                    embedding = memory_manager.generate_embedding(unit.text)
                    memory_store.update_embedding(unit_id, embedding)
                    logger.info(f"Repaired: Regenerated embedding for unit {unit_id}")
                except Exception as e:
                    logger.error(f"Repair failed for embedding {unit_id}: {e}")

        # Repair 4: Fix relationship errors
        if hasattr(memory_manager, "knowledge_graph"):
            for source, target in issues["relationship_errors"]:
                try:
                    memory_manager.knowledge_graph.remove_relationship(source, target)
                    logger.info(f"Repaired: Removed invalid relationship {source} -> {target}")
                except Exception as e:
                    logger.error(f"Repair failed for relationship {source} -> {target}: {e}")


class MaintenanceTaskScheduler:
    """
    Scheduler for background maintenance tasks.

    Executes tasks based on priority and schedule.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the maintenance task scheduler.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.tasks = {}
        self.context = {}
        self.task_queue = queue.PriorityQueue()
        self.worker_thread = None
        self.running = False
        self.monitor = get_monitor()
        self.resilience_manager = get_resilience_manager()

        # Add default tasks if enabled
        if self.config.get("enable_default_tasks", True):
            self._add_default_tasks()

    def _add_default_tasks(self) -> None:
        """Add default maintenance tasks."""
        # Memory consolidation (hourly)
        self.add_task(
            MemoryConsolidationTask(
                interval_seconds=self.config.get("consolidation_interval", 3600)
            )
        )

        # Memory pruning (daily)
        self.add_task(
            MemoryPruningTask(
                interval_seconds=self.config.get("pruning_interval", 86400),
                age_threshold_days=self.config.get("pruning_age_days", 90),
            )
        )

        # Database optimization (weekly)
        self.add_task(
            DatabaseOptimizationTask(
                interval_seconds=self.config.get("optimization_interval", 604800)
            )
        )

        # Memory backup (daily)
        self.add_task(
            MemoryBackupTask(
                interval_seconds=self.config.get("backup_interval", 86400),
                backup_dir=self.config.get("backup_dir", "backups"),
            )
        )

        # Integrity check (twice daily)
        self.add_task(
            MemoryIntegrityCheckTask(
                interval_seconds=self.config.get("integrity_check_interval", 43200)
            )
        )

    def add_task(self, task: MaintenanceTask) -> None:
        """
        Add a maintenance task to the scheduler.

        Args:
            task: Task to add
        """
        self.tasks[task.name] = task
        logger.info(f"Added maintenance task: {task.name} (priority: {task.priority.name})")

    def remove_task(self, task_name: str) -> None:
        """
        Remove a task from the scheduler.

        Args:
            task_name: Name of task to remove
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.info(f"Removed maintenance task: {task_name}")

    def set_context(self, **kwargs) -> None:
        """
        Set context values for task execution.

        Args:
            **kwargs: Context values
        """
        self.context.update(kwargs)

    def _worker_loop(self) -> None:
        """Background worker loop for executing tasks."""
        while self.running:
            try:
                # Check for due tasks and add to queue
                self._check_due_tasks()

                # Process one task from queue if available
                try:
                    # Get task with timeout to allow for graceful shutdown
                    priority, task_name = self.task_queue.get(timeout=1.0)

                    # Skip if task no longer exists
                    if task_name not in self.tasks:
                        self.task_queue.task_done()
                        continue

                    task = self.tasks[task_name]

                    # Execute task with resilience
                    self._execute_task_with_resilience(task)

                    # Mark task as done
                    self.task_queue.task_done()

                except queue.Empty:
                    # No tasks in queue, continue
                    pass

            except Exception as e:
                logger.error(f"Error in maintenance worker loop: {e}")

            # Sleep briefly to prevent CPU burning
            time.sleep(0.1)

    def _check_due_tasks(self) -> None:
        """Check for due tasks and add them to the queue."""
        for task_name, task in self.tasks.items():
            # Skip if already in queue or currently running
            if task.running:
                continue

            # Check if task is due
            if task.is_due():
                # Add to queue with priority
                priority = -task.priority.value  # Negative for highest first
                self.task_queue.put((priority, task_name))
                logger.debug(f"Scheduled due task: {task_name}")

    def _execute_task_with_resilience(self, task: MaintenanceTask) -> None:
        """
        Execute a task with resilience.

        Args:
            task: Task to execute
        """
        logger.info(f"Executing maintenance task: {task.name}")
        self.monitor.log_event("maintenance_task_start", {"task": task.name})

        try:
            # Create resilient operation
            with ResilientOperation(
                operation_name=f"maintenance_{task.name}",
                component_name="maintenance_scheduler",
                resilience_manager=self.resilience_manager,
                context={"task": task, "retry_operation": lambda: task.execute(self.context)},
            ) as op:
                # Execute task
                result = task.execute(self.context)

                # Record completion in monitoring
                self.monitor.log_event(
                    "maintenance_task_complete",
                    {"task": task.name, "result": result, "stats": task.stats},
                )

                logger.info(f"Completed maintenance task: {task.name}")

        except Exception as e:
            logger.error(f"Failed to execute maintenance task {task.name}: {e}")

            # Record failure in monitoring
            self.monitor.log_event(
                "maintenance_task_failure",
                {"task": task.name, "error": str(e), "stats": task.stats},
            )

    def start(self) -> None:
        """Start the maintenance scheduler."""
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Maintenance scheduler already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("Started maintenance scheduler")

        # Record in monitoring system
        self.monitor.log_event("maintenance_scheduler_start", {"tasks": list(self.tasks.keys())})

    def stop(self) -> None:
        """Stop the maintenance scheduler."""
        self.running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            self.worker_thread = None

        logger.info("Stopped maintenance scheduler")

        # Record in monitoring system
        self.monitor.log_event("maintenance_scheduler_stop", {})

    def get_task_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all tasks.

        Returns:
            Dictionary of task statistics
        """
        return {name: task.stats for name, task in self.tasks.items()}

    def run_task_now(self, task_name: str) -> Any:
        """
        Run a specific task immediately.

        Args:
            task_name: Name of task to run

        Returns:
            Task result
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task not found: {task_name}")

        task = self.tasks[task_name]

        logger.info(f"Manually running maintenance task: {task_name}")

        # Execute with resilience
        with ResilientOperation(
            operation_name=f"maintenance_{task.name}",
            component_name="maintenance_scheduler",
            resilience_manager=self.resilience_manager,
            context={"task": task, "retry_operation": lambda: task.execute(self.context)},
        ):
            return task.execute(self.context)


# Create singleton maintenance scheduler instance
_global_maintenance_scheduler = None


def get_maintenance_scheduler(config: dict[str, Any] | None = None) -> MaintenanceTaskScheduler:
    """
    Get or create the global maintenance scheduler instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Global MaintenanceTaskScheduler instance
    """
    global _global_maintenance_scheduler
    if _global_maintenance_scheduler is None:
        _global_maintenance_scheduler = MaintenanceTaskScheduler(config)
    return _global_maintenance_scheduler
