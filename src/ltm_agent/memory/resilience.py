"""
Resilience and failure recovery system for the Long-Term Memory Agent.

This module provides mechanisms for ensuring data integrity, recovering from
failures, and ensuring continuous operation of the memory system.
"""

import logging
import os
import pickle
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .monitoring import get_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""

    # Data failures
    DATA_CORRUPTION = "data_corruption"
    DATA_INCONSISTENCY = "data_inconsistency"
    DATA_LOSS = "data_loss"

    # System failures
    SYSTEM_CRASH = "system_crash"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_EXHAUSTION = "disk_exhaustion"

    # Service failures
    DATABASE_UNAVAILABLE = "database_unavailable"
    EMBEDDING_SERVICE_ERROR = "embedding_service_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"

    # Other failures
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Strategies for recovering from failures."""

    # Continue with reduced functionality
    GRACEFUL_DEGRADATION = "graceful_degradation"

    # Use backup data
    USE_BACKUP = "use_backup"

    # Reinitialize component
    REINITIALIZE = "reinitialize"

    # Retry operation
    RETRY = "retry"

    # Abort operation
    ABORT = "abort"


class ResilienceManager:
    """
    Central manager for resilience and recovery operations.

    Provides:
    1. Error detection and diagnostics
    2. Recovery strategy selection and execution
    3. Checkpointing and backup management
    4. Circuit breaking for failing dependencies
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the resilience manager.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.monitor = get_monitor()
        self.failures = []
        self.recovery_handlers = {}
        self.circuit_breakers = {}
        self.checkpoint_lock = threading.Lock()
        self.backup_dir = self.config.get("backup_dir", "backups")
        self.last_backup_time = None
        self.backup_interval = self.config.get("backup_interval_hours", 24)
        self.max_backups = self.config.get("max_backups", 5)
        self.initialized = False

        # Create backup directory if needed
        os.makedirs(self.backup_dir, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

        self.initialized = True

    def _register_default_handlers(self) -> None:
        """Register default recovery handlers."""
        # Data corruption/loss handlers
        self.register_recovery_handler(
            FailureType.DATA_CORRUPTION, RecoveryStrategy.USE_BACKUP, self._recover_from_backup
        )

        self.register_recovery_handler(
            FailureType.DATA_LOSS, RecoveryStrategy.USE_BACKUP, self._recover_from_backup
        )

        # Service failure handlers
        self.register_recovery_handler(
            FailureType.DATABASE_UNAVAILABLE, RecoveryStrategy.RETRY, self._retry_with_backoff
        )

        self.register_recovery_handler(
            FailureType.EMBEDDING_SERVICE_ERROR,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            self._use_fallback_embedding,
        )

        self.register_recovery_handler(
            FailureType.RATE_LIMIT_EXCEEDED, RecoveryStrategy.RETRY, self._retry_with_backoff
        )

    def register_recovery_handler(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """
        Register a handler for a specific failure type and strategy.

        Args:
            failure_type: Type of failure
            strategy: Recovery strategy
            handler: Function to handle recovery
        """
        if failure_type not in self.recovery_handlers:
            self.recovery_handlers[failure_type] = {}

        self.recovery_handlers[failure_type][strategy] = handler

        logger.debug(f"Registered recovery handler for {failure_type.value} using {strategy.value}")

    def record_failure(
        self,
        failure_type: FailureType,
        details: dict[str, Any],
        exception: Exception | None = None,
    ) -> dict[str, Any]:
        """
        Record a failure for analysis.

        Args:
            failure_type: Type of failure
            details: Failure details
            exception: Optional exception that caused the failure

        Returns:
            Failure record
        """
        failure = {
            "id": str(uuid.uuid4()),
            "type": failure_type.value,
            "timestamp": time.time(),
            "details": details,
            "exception": str(exception) if exception else None,
            "traceback": self._get_traceback(exception) if exception else None,
            "recovered": False,
            "recovery_attempts": 0,
        }

        self.failures.append(failure)

        # Log the failure
        logger.error(f"Failure detected: {failure_type.value}")
        if exception:
            logger.error(f"Exception: {exception}")

        # Record in monitoring system
        self.monitor.log_event(
            "failure",
            {
                "failure_id": failure["id"],
                "type": failure_type.value,
                "details": details,
                "exception": str(exception) if exception else None,
            },
        )

        return failure

    def _get_traceback(self, exception: Exception) -> str | None:
        """
        Get formatted traceback from an exception.

        Args:
            exception: Exception to get traceback from

        Returns:
            Formatted traceback or None
        """
        if exception is None:
            return None

        import traceback

        return "".join(
            traceback.format_exception(type(exception), exception, exception.__traceback__)
        )

    def recover_from_failure(
        self,
        failure: dict[str, Any],
        strategy: RecoveryStrategy | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, Any]:
        """
        Attempt to recover from a failure.

        Args:
            failure: Failure record
            strategy: Optional specific strategy to use
            context: Optional context information for recovery

        Returns:
            Tuple of (success, result)
        """
        failure_type = FailureType(failure["type"])
        context = context or {}

        # Update failure record
        failure["recovery_attempts"] += 1

        # Choose strategy if not specified
        if strategy is None:
            strategy = self._select_recovery_strategy(failure_type, failure)

        logger.info(f"Attempting recovery from {failure_type.value} using {strategy.value}")

        # Get appropriate handler
        handler = self._get_recovery_handler(failure_type, strategy)
        if not handler:
            logger.warning(f"No recovery handler for {failure_type.value} using {strategy.value}")
            return False, None

        # Attempt recovery
        try:
            start_time = time.time()
            result = handler({"failure": failure, "context": context})
            elapsed = time.time() - start_time

            # Update failure record
            failure["recovered"] = True
            failure["recovery_strategy"] = strategy.value
            failure["recovery_time"] = elapsed

            # Log success
            logger.info(f"Successfully recovered from {failure_type.value} using {strategy.value}")

            # Record in monitoring system
            self.monitor.log_event(
                "recovery_success",
                {
                    "failure_id": failure["id"],
                    "type": failure_type.value,
                    "strategy": strategy.value,
                    "elapsed_time": elapsed,
                },
            )

            return True, result

        except Exception as e:
            # Log failure
            logger.error(f"Recovery from {failure_type.value} failed: {e}")

            # Record in monitoring system
            self.monitor.log_event(
                "recovery_failure",
                {
                    "failure_id": failure["id"],
                    "type": failure_type.value,
                    "strategy": strategy.value,
                    "exception": str(e),
                },
            )

            return False, None

    def _select_recovery_strategy(
        self, failure_type: FailureType, failure: dict[str, Any]
    ) -> RecoveryStrategy:
        """
        Select an appropriate recovery strategy for a failure.

        Args:
            failure_type: Type of failure
            failure: Failure record

        Returns:
            Selected recovery strategy
        """
        # Check if we have handlers for this failure type
        if failure_type not in self.recovery_handlers:
            logger.warning(f"No recovery handlers for {failure_type.value}")
            return RecoveryStrategy.ABORT

        # Prioritize strategies
        strategies = list(self.recovery_handlers[failure_type].keys())

        # Attempt count affects strategy
        attempts = failure.get("recovery_attempts", 0)

        # First attempt: prefer graceful degradation or retry
        if attempts == 0:
            if RecoveryStrategy.GRACEFUL_DEGRADATION in strategies:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
            elif RecoveryStrategy.RETRY in strategies:
                return RecoveryStrategy.RETRY

        # Second attempt: try backup if available
        elif attempts == 1:
            if RecoveryStrategy.USE_BACKUP in strategies:
                return RecoveryStrategy.USE_BACKUP

        # Last resort: reinitialize or abort
        if RecoveryStrategy.REINITIALIZE in strategies:
            return RecoveryStrategy.REINITIALIZE

        return RecoveryStrategy.ABORT

    def _get_recovery_handler(
        self, failure_type: FailureType, strategy: RecoveryStrategy
    ) -> Callable | None:
        """
        Get recovery handler for a failure type and strategy.

        Args:
            failure_type: Type of failure
            strategy: Recovery strategy

        Returns:
            Handler function or None
        """
        if (
            failure_type in self.recovery_handlers
            and strategy in self.recovery_handlers[failure_type]
        ):
            return self.recovery_handlers[failure_type][strategy]

        return None

    def create_checkpoint(self, data: Any, component_name: str) -> str:
        """
        Create a checkpoint/backup of component data.

        Args:
            data: Data to checkpoint
            component_name: Name of the component

        Returns:
            Path to checkpoint file
        """
        with self.checkpoint_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(
                self.backup_dir, f"{component_name}_{timestamp}.checkpoint"
            )

            # Create a temporary file first
            temp_file = checkpoint_file + ".tmp"

            try:
                with open(temp_file, "wb") as f:
                    pickle.dump(data, f)

                # Rename for atomic operation
                os.rename(temp_file, checkpoint_file)

                self.last_backup_time = datetime.now()

                # Clean up old backups
                self._cleanup_old_backups(component_name)

                logger.info(f"Created checkpoint for {component_name}: {checkpoint_file}")

                return checkpoint_file

            except Exception as e:
                logger.error(f"Error creating checkpoint for {component_name}: {e}")

                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                raise

    def restore_from_checkpoint(
        self, component_name: str, checkpoint_file: str | None = None
    ) -> Any:
        """
        Restore data from a checkpoint.

        Args:
            component_name: Name of the component
            checkpoint_file: Optional specific checkpoint file to restore from

        Returns:
            Restored data
        """
        with self.checkpoint_lock:
            # Find latest checkpoint if not specified
            if checkpoint_file is None:
                checkpoint_file = self._find_latest_checkpoint(component_name)

            if not checkpoint_file or not os.path.exists(checkpoint_file):
                logger.error(f"No checkpoint found for {component_name}")
                return None

            try:
                with open(checkpoint_file, "rb") as f:
                    data = pickle.load(f)

                logger.info(f"Restored {component_name} from checkpoint: {checkpoint_file}")

                return data

            except Exception as e:
                logger.error(f"Error restoring {component_name} from checkpoint: {e}")

                # Try an older checkpoint
                older_checkpoint = self._find_next_older_checkpoint(component_name, checkpoint_file)
                if older_checkpoint:
                    logger.info(f"Attempting to restore from older checkpoint: {older_checkpoint}")
                    return self.restore_from_checkpoint(component_name, older_checkpoint)

                return None

    def _find_latest_checkpoint(self, component_name: str) -> str | None:
        """
        Find the latest checkpoint for a component.

        Args:
            component_name: Name of the component

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = []

        for file in os.listdir(self.backup_dir):
            if file.startswith(f"{component_name}_") and file.endswith(".checkpoint"):
                checkpoints.append(os.path.join(self.backup_dir, file))

        if not checkpoints:
            return None

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return checkpoints[0]

    def _find_next_older_checkpoint(
        self, component_name: str, current_checkpoint: str
    ) -> str | None:
        """
        Find the next older checkpoint after the current one.

        Args:
            component_name: Name of the component
            current_checkpoint: Current checkpoint file

        Returns:
            Path to next older checkpoint or None
        """
        checkpoints = []

        for file in os.listdir(self.backup_dir):
            filepath = os.path.join(self.backup_dir, file)
            if (
                file.startswith(f"{component_name}_")
                and file.endswith(".checkpoint")
                and filepath != current_checkpoint
            ):
                checkpoints.append(filepath)

        if not checkpoints:
            return None

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return checkpoints[0] if checkpoints else None

    def _cleanup_old_backups(self, component_name: str) -> None:
        """
        Clean up old backups beyond the maximum count.

        Args:
            component_name: Name of the component
        """
        checkpoints = []

        for file in os.listdir(self.backup_dir):
            if file.startswith(f"{component_name}_") and file.endswith(".checkpoint"):
                checkpoints.append(os.path.join(self.backup_dir, file))

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x))

        # Remove oldest beyond the limit
        files_to_delete = (
            checkpoints[: -self.max_backups] if len(checkpoints) > self.max_backups else []
        )

        for file in files_to_delete:
            try:
                os.remove(file)
                logger.debug(f"Cleaned up old checkpoint: {file}")
            except Exception as e:
                logger.warning(f"Error cleaning up old checkpoint {file}: {e}")

    def should_create_backup(self, component_name: str) -> bool:
        """
        Check if a backup should be created for a component.

        Args:
            component_name: Name of the component

        Returns:
            True if backup should be created
        """
        # Skip if too recent
        if self.last_backup_time:
            elapsed = datetime.now() - self.last_backup_time
            if elapsed < timedelta(hours=self.backup_interval):
                return False

        return True

    def _retry_with_backoff(self, context: dict[str, Any]) -> Any:
        """
        Retry an operation with exponential backoff.

        Args:
            context: Context information

        Returns:
            Result of operation
        """
        failure = context["failure"]
        retry_context = context.get("context", {})

        max_retries = retry_context.get("max_retries", 3)
        base_delay = retry_context.get("base_delay", 1.0)
        max_delay = retry_context.get("max_delay", 60.0)
        operation = retry_context.get("operation")

        if not operation or not callable(operation):
            raise ValueError("No operation function provided for retry")

        attempts = failure.get("recovery_attempts", 0)

        # Calculate backoff delay
        delay = min(base_delay * (2**attempts), max_delay)

        if attempts >= max_retries:
            logger.warning(f"Maximum retry attempts ({max_retries}) exceeded")
            raise Exception("Maximum retry attempts exceeded")

        logger.info(
            f"Retrying operation in {delay:.2f} seconds (attempt {attempts + 1}/{max_retries})"
        )

        # Sleep with backoff
        time.sleep(delay)

        # Retry operation
        return operation()

    def _recover_from_backup(self, context: dict[str, Any]) -> Any:
        """
        Recover data from backup.

        Args:
            context: Context information

        Returns:
            Restored data
        """
        component_name = context.get("context", {}).get("component_name")

        if not component_name:
            raise ValueError("No component_name provided for backup recovery")

        return self.restore_from_checkpoint(component_name)

    def _use_fallback_embedding(self, context: dict[str, Any]) -> Any:
        """
        Use fallback embedding provider.

        Args:
            context: Context information

        Returns:
            Fallback embedding result
        """
        retry_context = context.get("context", {})
        text = retry_context.get("text")

        if not text:
            raise ValueError("No text provided for fallback embedding")

        # Use dummy embedding (random but deterministic based on text hash)
        import hashlib

        import numpy as np

        # Use text hash as seed for reproducibility
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash % 2**32)

        # Generate random embedding (default 384 dimensions)
        dimensions = retry_context.get("dimensions", 384)
        embedding = np.random.normal(0, 1, dimensions)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def check_circuit_breaker(self, service_name: str) -> bool:
        """
        Check if a circuit breaker is tripped.

        Args:
            service_name: Name of the service to check

        Returns:
            True if service is available (circuit closed)
        """
        if service_name not in self.circuit_breakers:
            return True

        breaker = self.circuit_breakers[service_name]

        # Check if the cooling period has passed
        if breaker["status"] == "open" and time.time() > breaker["reset_time"]:
            breaker["status"] = "half-open"
            logger.info(f"Circuit breaker for {service_name} is now half-open")

        return breaker["status"] in ["closed", "half-open"]

    def update_circuit_breaker(self, service_name: str, success: bool) -> None:
        """
        Update circuit breaker status based on operation success.

        Args:
            service_name: Name of the service
            success: Whether the operation succeeded
        """
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "status": "closed",
                "failures": 0,
                "threshold": self.config.get("circuit_breaker_threshold", 5),
                "cooling_period": self.config.get("circuit_breaker_cooling_period", 60),
                "reset_time": 0,
            }

        breaker = self.circuit_breakers[service_name]

        if success:
            # Successful operation
            if breaker["status"] == "half-open":
                breaker["status"] = "closed"
                breaker["failures"] = 0
                logger.info(f"Circuit breaker for {service_name} is now closed")

        else:
            # Failed operation
            if breaker["status"] == "closed":
                breaker["failures"] += 1

                if breaker["failures"] >= breaker["threshold"]:
                    breaker["status"] = "open"
                    breaker["reset_time"] = time.time() + breaker["cooling_period"]
                    logger.warning(
                        f"Circuit breaker for {service_name} is now open until {datetime.fromtimestamp(breaker['reset_time'])}"
                    )

            elif breaker["status"] == "half-open":
                breaker["status"] = "open"
                breaker["reset_time"] = time.time() + breaker["cooling_period"]
                logger.warning(
                    f"Circuit breaker for {service_name} is now open until {datetime.fromtimestamp(breaker['reset_time'])}"
                )


class ResilientOperation:
    """
    Context manager for executing operations with resilience.

    Handles failures and recovery automatically.
    """

    def __init__(
        self,
        operation_name: str,
        component_name: str | None = None,
        resilience_manager: ResilienceManager | None = None,
        failure_mapping: dict[type[Exception], FailureType] | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize the resilient operation.

        Args:
            operation_name: Name of the operation
            component_name: Optional name of the component
            resilience_manager: Optional resilience manager to use
            failure_mapping: Optional mapping of exception types to failure types
            context: Optional context information for recovery
        """
        self.operation_name = operation_name
        self.component_name = component_name
        self.resilience_manager = resilience_manager or ResilienceManager()
        self.monitor = get_monitor()
        self.failure_mapping = failure_mapping or {}
        self.context = context or {}
        self.failure = None
        self.start_time = None

        # Add default exception mappings
        if not self.failure_mapping:
            self.failure_mapping = {
                ConnectionError: FailureType.DATABASE_UNAVAILABLE,
                TimeoutError: FailureType.TIMEOUT,
                ValueError: FailureType.DATA_INCONSISTENCY,
                Exception: FailureType.UNKNOWN,
            }

    def __enter__(self):
        """Enter the context manager."""
        self.start_time = time.time()
        self.monitor.start_timer(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, handling any exceptions.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            True if exception was handled
        """
        elapsed = time.time() - self.start_time
        self.monitor.stop_timer(self.operation_name)

        # No exception occurred
        if exc_type is None:
            return False

        # Find matching failure type
        failure_type = FailureType.UNKNOWN
        for exc_class, f_type in self.failure_mapping.items():
            if isinstance(exc_val, exc_class):
                failure_type = f_type
                break

        # Record failure
        self.failure = self.resilience_manager.record_failure(
            failure_type,
            {
                "operation": self.operation_name,
                "component": self.component_name,
                "elapsed": elapsed,
            },
            exc_val,
        )

        # Attempt recovery
        if self.component_name:
            self.context["component_name"] = self.component_name

        success, result = self.resilience_manager.recover_from_failure(
            self.failure, context=self.context
        )

        # If recovery succeeded, store result and suppress exception
        if success:
            self.result = result
            return True

        # Let exception propagate
        return False


# Create singleton resilience manager instance
_global_resilience_manager = None


def get_resilience_manager(config: dict[str, Any] | None = None) -> ResilienceManager:
    """
    Get or create the global resilience manager instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Global ResilienceManager instance
    """
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager(config)
    return _global_resilience_manager
