"""
Batch processor for efficient vector operations in the Long-Term Memory Agent.

This module provides batch processing capabilities for vector operations
to improve performance when working with large knowledge bases.
"""

import logging
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Generic, TypeVar

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar("T")
R = TypeVar("R")


class BatchTask(Generic[T, R]):
    """Generic batch task representation."""

    def __init__(self, item: T, callback: Callable[[R], None] | None = None):
        """
        Initialize a batch task.

        Args:
            item: The item to process
            callback: Optional callback to execute after processing
        """
        self.item = item
        self.callback = callback
        self.result = None
        self.processed = False
        self.error = None


class BatchProcessor(Generic[T, R]):
    """
    Generic batch processor for efficient processing of similar operations.

    This class provides:
    1. Batched processing of similar operations
    2. Automatic flushing based on batch size or time
    3. Asynchronous processing option
    4. Result callbacks
    """

    def __init__(
        self,
        batch_processor: Callable[[list[T]], list[R]],
        max_batch_size: int = 100,
        max_wait_time: float = 1.0,
        async_processing: bool = False,
    ):
        """
        Initialize the batch processor.

        Args:
            batch_processor: Function to process a batch of items
            max_batch_size: Maximum items in a batch before processing
            max_wait_time: Maximum time (seconds) to wait before processing
            async_processing: Whether to process batches asynchronously
        """
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.async_processing = async_processing

        self.queue = []
        self.last_flush_time = time.time()

        # For async processing
        self._stop_requested = False
        self._task_queue = Queue() if async_processing else None
        self._worker_thread = None

        if async_processing:
            self._start_worker()

    def add(self, item: T, callback: Callable[[R], None] | None = None) -> None:
        """
        Add an item to the batch.

        Args:
            item: Item to process
            callback: Optional callback to execute with result
        """
        task = BatchTask(item, callback)

        if self.async_processing:
            self._task_queue.put(task)
        else:
            self.queue.append(task)

            # Check if we should flush
            current_time = time.time()
            time_elapsed = current_time - self.last_flush_time

            if len(self.queue) >= self.max_batch_size or time_elapsed >= self.max_wait_time:
                self.flush()

    def flush(self) -> list[BatchTask[T, R]]:
        """
        Process all queued items.

        Returns:
            List of processed batch tasks
        """
        if not self.queue:
            return []

        # Get items from queue
        current_batch = self.queue
        self.queue = []
        self.last_flush_time = time.time()

        # Extract items for processing
        items = [task.item for task in current_batch]

        try:
            # Process the batch
            results = self.batch_processor(items)

            # Match results with tasks
            for i, result in enumerate(results):
                if i < len(current_batch):
                    current_batch[i].result = result
                    current_batch[i].processed = True

                    # Execute callback if provided
                    if current_batch[i].callback:
                        try:
                            current_batch[i].callback(result)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

            # Mark all tasks as errored
            for task in current_batch:
                task.error = str(e)
                task.processed = True

        return current_batch

    def _start_worker(self) -> None:
        """Start the asynchronous worker thread."""
        if self._worker_thread is not None:
            return

        self._stop_requested = False
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread loop for async processing."""
        while not self._stop_requested:
            # Collect batch
            batch = []

            try:
                # Get at least one item (blocking)
                task = self._task_queue.get(timeout=self.max_wait_time)
                batch.append(task)
                self._task_queue.task_done()

                # Try to get more items without blocking
                while len(batch) < self.max_batch_size:
                    try:
                        task = self._task_queue.get(block=False)
                        batch.append(task)
                        self._task_queue.task_done()
                    except Empty:
                        break

            except Empty:
                # Timeout occurred with no items
                continue

            if batch:
                # Extract items for processing
                items = [task.item for task in batch]

                try:
                    # Process the batch
                    results = self.batch_processor(items)

                    # Match results with tasks
                    for i, result in enumerate(results):
                        if i < len(batch):
                            batch[i].result = result
                            batch[i].processed = True

                            # Execute callback if provided
                            if batch[i].callback:
                                try:
                                    batch[i].callback(result)
                                except Exception as e:
                                    logger.error(f"Error in callback: {e}")

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

                    # Mark all tasks as errored
                    for task in batch:
                        task.error = str(e)
                        task.processed = True

    def stop(self) -> None:
        """Stop the batch processor and wait for pending items."""
        if not self.async_processing:
            # Just flush any remaining items
            self.flush()
            return

        # Signal worker to stop
        self._stop_requested = True

        # Wait for queue to empty
        if self._task_queue:
            self._task_queue.join()

        # Wait for worker thread to terminate
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def __del__(self) -> None:
        """Clean up resources when the object is deleted."""
        try:
            self.stop()
        except:
            pass


class VectorBatchProcessor:
    """
    Batch processor specifically for vector operations.

    This class provides batched operations for:
    1. Embedding generation
    2. Vector similarity search
    3. Vector additions and updates
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_function: Callable[[str], list[float]],
        max_batch_size: int = 100,
        max_wait_time: float = 1.0,
        async_processing: bool = False,
    ):
        """
        Initialize the vector batch processor.

        Args:
            vector_store: Vector store for operations
            embedding_function: Function to generate embeddings from text
            max_batch_size: Maximum items in a batch
            max_wait_time: Maximum time to wait before processing
            async_processing: Whether to process batches asynchronously
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function

        # Create batch processors for different operations
        self.embedding_batch = BatchProcessor(
            self._process_embedding_batch,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            async_processing=async_processing,
        )

        self.add_batch = BatchProcessor(
            self._process_add_batch,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            async_processing=async_processing,
        )

        self.search_batch = BatchProcessor(
            self._process_search_batch,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            async_processing=async_processing,
        )

    def generate_embedding(
        self, text: str, callback: Callable[[list[float]], None] | None = None
    ) -> None:
        """
        Generate embedding for text in batch.

        Args:
            text: Text to generate embedding for
            callback: Optional callback for result
        """
        self.embedding_batch.add(text, callback)

    def add_knowledge_unit(
        self, unit: KnowledgeUnit, callback: Callable[[None], None] | None = None
    ) -> None:
        """
        Add knowledge unit to vector store in batch.

        Args:
            unit: Knowledge unit to add
            callback: Optional callback for result
        """
        self.add_batch.add(unit, callback)

    def search_similar(
        self,
        query: tuple[list[float], int, float],
        callback: Callable[[list[tuple[KnowledgeUnit, float]]], None] | None = None,
    ) -> None:
        """
        Search for similar knowledge units in batch.

        Args:
            query: Tuple of (embedding, limit, threshold)
            callback: Optional callback for result
        """
        self.search_batch.add(query, callback)

    def _process_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Process a batch of text embedding requests.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        # For simplicity, use the embedding function on each text
        # In a real implementation, this might call a batch API
        embeddings = [self.embedding_function(text) for text in texts]
        return embeddings

    def _process_add_batch(self, units: list[KnowledgeUnit]) -> list[None]:
        """
        Process a batch of knowledge unit additions.

        Args:
            units: List of knowledge units to add

        Returns:
            List of None values (same length as input)
        """
        # Add units in a batch if the vector store supports it
        if hasattr(self.vector_store, "add_batch"):
            self.vector_store.add_batch(units)
        else:
            # Fall back to individual additions
            for unit in units:
                self.vector_store.add(unit)

        return [None] * len(units)

    def _process_search_batch(
        self, queries: list[tuple[list[float], int, float]]
    ) -> list[list[tuple[KnowledgeUnit, float]]]:
        """
        Process a batch of similarity search queries.

        Args:
            queries: List of (embedding, limit, threshold) tuples

        Returns:
            List of search result lists
        """
        # Execute each search individually since they may have different parameters
        results = []

        for embedding, limit, threshold in queries:
            search_result = self.vector_store.search(embedding, limit, threshold)
            results.append(search_result)

        return results

    def flush_all(self) -> None:
        """Flush all batch processors."""
        self.embedding_batch.flush()
        self.add_batch.flush()
        self.search_batch.flush()

    def stop(self) -> None:
        """Stop all batch processors."""
        self.embedding_batch.stop()
        self.add_batch.stop()
        self.search_batch.stop()


class BatchedVectorStore(VectorStore):
    """
    Vector store wrapper that adds batch processing capabilities.

    This class wraps any vector store implementation and adds:
    1. Batched additions and updates
    2. Async processing option
    3. Improved performance for bulk operations
    """

    def __init__(
        self,
        base_store: VectorStore,
        embedding_function: Callable[[str], list[float]],
        max_batch_size: int = 100,
        max_wait_time: float = 1.0,
        async_processing: bool = False,
    ):
        """
        Initialize the batched vector store.

        Args:
            base_store: Underlying vector store
            embedding_function: Function to generate embeddings
            max_batch_size: Maximum items in a batch
            max_wait_time: Maximum time to wait before processing
            async_processing: Whether to process batches asynchronously
        """
        self.base_store = base_store
        self.embedding_function = embedding_function
        self.async_processing = async_processing

        # Create batch processor
        self.batch_processor = VectorBatchProcessor(
            base_store,
            embedding_function,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            async_processing=async_processing,
        )

        # Cache for pending operations
        self.pending_adds = {}
        self.pending_updates = {}
        self.pending_deletes = set()

    def add(self, unit: KnowledgeUnit) -> None:
        """
        Add a knowledge unit to the vector store.

        Args:
            unit: Knowledge unit to add
        """
        # Check if embedding exists, generate if needed
        if unit.embedding is None:
            # Generate embedding
            if unit.original_chunk:
                unit.embedding = self.embedding_function(unit.original_chunk)
            else:
                unit.embedding = self.embedding_function(unit.processed_chunk or "")

        # Add to pending cache
        self.pending_adds[unit.unique_id] = unit

        # Add to batch processor
        self.batch_processor.add_knowledge_unit(
            unit, lambda _: self._on_add_complete(unit.unique_id)
        )

    def _on_add_complete(self, unit_id: str) -> None:
        """
        Callback when add operation completes.

        Args:
            unit_id: ID of the added knowledge unit
        """
        if unit_id in self.pending_adds:
            del self.pending_adds[unit_id]

    def update(self, unit: KnowledgeUnit) -> None:
        """
        Update a knowledge unit in the store.

        Args:
            unit: Knowledge unit to update
        """
        # Check if embedding exists, generate if needed
        if unit.embedding is None:
            # Generate embedding
            if unit.original_chunk:
                unit.embedding = self.embedding_function(unit.original_chunk)
            else:
                unit.embedding = self.embedding_function(unit.processed_chunk or "")

        # Add to pending cache
        self.pending_updates[unit.unique_id] = unit

        # Update in base store
        self.base_store.update(unit)

        # Remove from pending cache
        if unit.unique_id in self.pending_updates:
            del self.pending_updates[unit.unique_id]

    def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the store.

        Args:
            unique_id: ID of the knowledge unit to delete

        Returns:
            True if deleted, False otherwise
        """
        # Add to pending deletes
        self.pending_deletes.add(unique_id)

        # Delete from base store
        result = self.base_store.delete(unique_id)

        # Remove from pending deletes
        if unique_id in self.pending_deletes:
            self.pending_deletes.remove(unique_id)

        # Also remove from pending adds/updates if present
        if unique_id in self.pending_adds:
            del self.pending_adds[unique_id]
        if unique_id in self.pending_updates:
            del self.pending_updates[unique_id]

        return result

    def get(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get a knowledge unit by ID.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        # Check pending adds first
        if unique_id in self.pending_adds:
            return self.pending_adds[unique_id]

        # Check pending updates
        if unique_id in self.pending_updates:
            return self.pending_updates[unique_id]

        # Check if pending delete
        if unique_id in self.pending_deletes:
            return None

        # Fall back to base store
        return self.base_store.get(unique_id)

    def search(
        self, embedding: list[float], limit: int = 5, threshold: float = 0.0
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for similar knowledge units.

        Args:
            embedding: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (knowledge_unit, similarity_score) tuples
        """
        # Flush batch processors to ensure all changes are applied
        self.batch_processor.flush_all()

        # Delegate to base store
        return self.base_store.search(embedding, limit, threshold)

    def list_all(self) -> list[KnowledgeUnit]:
        """
        List all knowledge units in the store.

        Returns:
            List of all knowledge units
        """
        # Flush batch processors to ensure all changes are applied
        self.batch_processor.flush_all()

        # Delegate to base store
        return self.base_store.list_all()

    def clear(self) -> None:
        """Clear all knowledge units from the store."""
        # Clear pending operations
        self.pending_adds.clear()
        self.pending_updates.clear()
        self.pending_deletes.clear()

        # Clear base store
        self.base_store.clear()

    def count(self) -> int:
        """
        Count the number of knowledge units in the store.

        Returns:
            Number of knowledge units
        """
        # Flush batch processors to ensure all changes are applied
        self.batch_processor.flush_all()

        # Delegate to base store
        return self.base_store.count()

    def flush(self) -> None:
        """Flush all pending operations."""
        self.batch_processor.flush_all()

    def close(self) -> None:
        """Close the vector store and persist any pending changes."""
        # Flush any pending operations
        self.flush()

        # Stop batch processors
        self.batch_processor.stop()

        # Close base store
        self.base_store.close()
