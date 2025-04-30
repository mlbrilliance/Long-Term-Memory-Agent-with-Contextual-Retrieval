"""
Batch operations for memory management.

This module provides utilities for efficient batch operations on knowledge units,
improving performance for bulk operations.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ltm_agent.core.exceptions import MemoryError
from ltm_agent.memory.manager import MemoryManager

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")


class BatchProcessor:
    """
    Batch processor for efficient memory operations.

    This class provides utilities for processing operations in batches,
    improving performance for bulk operations.
    """

    def __init__(self, memory_manager: MemoryManager, batch_size: int = 50):
        """
        Initialize the batch processor.

        Args:
            memory_manager: Memory manager to use for operations
            batch_size: Size of batches for operations
        """
        self.memory_manager = memory_manager
        self.batch_size = batch_size
        logger.info(f"Initialized BatchProcessor with batch_size={batch_size}")

    async def add_knowledge_batch(
        self,
        contents: list[str],
        source: str,
        contexts: list[str] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        tags_list: list[list[str]] | None = None,
    ) -> list[str]:
        """
        Add multiple knowledge units in batches.

        Args:
            contents: List of content strings
            source: Source of the knowledge
            contexts: Optional list of context strings (same length as contents)
            metadata_list: Optional list of metadata dictionaries
            tags_list: Optional list of tag lists

        Returns:
            List of unique IDs for the added knowledge units

        Raises:
            MemoryError: If batch operation fails
            ValueError: If input lists have different lengths
        """
        # Validate input lists have the same length
        n_items = len(contents)

        if contexts is not None and len(contexts) != n_items:
            raise ValueError(
                f"Contents and contexts must have the same length: {n_items} != {len(contexts)}"
            )

        if metadata_list is not None and len(metadata_list) != n_items:
            raise ValueError(
                f"Contents and metadata_list must have the same length: {n_items} != {len(metadata_list)}"
            )

        if tags_list is not None and len(tags_list) != n_items:
            raise ValueError(
                f"Contents and tags_list must have the same length: {n_items} != {len(tags_list)}"
            )

        # Default empty lists for optional parameters
        contexts = contexts or [""] * n_items
        metadata_list = metadata_list or [{}] * n_items
        tags_list = tags_list or [None] * n_items

        # Process in batches
        all_ids = []

        try:
            for i in range(0, n_items, self.batch_size):
                batch_end = min(i + self.batch_size, n_items)
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}: items {i} to {batch_end}"
                )

                # Create batch of tasks
                tasks = []
                for j in range(i, batch_end):
                    task = self.memory_manager.add_knowledge(
                        content=contents[j],
                        source=source,
                        context=contexts[j],
                        metadata=metadata_list[j],
                    )
                    tasks.append(task)

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle any exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error adding item {i + j}: {str(result)}")
                        all_ids.append(None)  # Use None to indicate failure
                    else:
                        # Add tags if provided
                        if tags_list[i + j]:
                            ku = await self.memory_manager.retrieve_knowledge(result)
                            if ku:
                                for tag in tags_list[i + j]:
                                    ku.add_tag(tag)
                                await self.memory_manager.update_knowledge(
                                    ku.unique_id, None, None, None
                                )

                        all_ids.append(result)

            logger.info(
                f"Successfully added {len([id for id in all_ids if id is not None])}/{n_items} knowledge units"
            )
            return all_ids

        except Exception as e:
            error_msg = f"Error in batch add operation: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def update_knowledge_batch(
        self,
        unique_ids: list[str],
        contents: list[str] | None = None,
        contexts: list[str] | None = None,
        sources: list[str] | None = None,
        metadata_updates: list[dict[str, Any]] | None = None,
        tags_to_add: list[list[str]] | None = None,
        tags_to_remove: list[list[str]] | None = None,
    ) -> list[bool]:
        """
        Update multiple knowledge units in batches.

        Args:
            unique_ids: List of knowledge unit IDs to update
            contents: Optional list of new content strings
            contexts: Optional list of new context strings
            sources: Optional list of new source strings
            metadata_updates: Optional list of metadata dictionaries to update
            tags_to_add: Optional list of tag lists to add
            tags_to_remove: Optional list of tag lists to remove

        Returns:
            List of success flags for each update operation

        Raises:
            MemoryError: If batch operation fails
            ValueError: If input lists have different lengths
        """
        # Validate input lists have the same length if provided
        n_items = len(unique_ids)

        for name, lst in [
            ("contents", contents),
            ("contexts", contexts),
            ("sources", sources),
            ("metadata_updates", metadata_updates),
            ("tags_to_add", tags_to_add),
            ("tags_to_remove", tags_to_remove),
        ]:
            if lst is not None and len(lst) != n_items:
                raise ValueError(
                    f"{name} must have the same length as unique_ids: {n_items} != {len(lst)}"
                )

        # Default None for optional parameters
        contents = contents or [None] * n_items
        contexts = contexts or [None] * n_items
        sources = sources or [None] * n_items
        metadata_updates = metadata_updates or [None] * n_items
        tags_to_add = tags_to_add or [None] * n_items
        tags_to_remove = tags_to_remove or [None] * n_items

        # Process in batches
        all_results = []

        try:
            for i in range(0, n_items, self.batch_size):
                batch_end = min(i + self.batch_size, n_items)
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}: items {i} to {batch_end}"
                )

                # Process each item in the batch
                batch_results = []
                for j in range(i, batch_end):
                    idx = j - i  # Index within current batch

                    try:
                        # First retrieve the knowledge unit
                        ku = await self.memory_manager.retrieve_knowledge(unique_ids[j])

                        # Apply metadata updates if provided
                        if metadata_updates[j]:
                            for key, value in metadata_updates[j].items():
                                ku.update_metadata(key, value)

                        # Apply tag updates if provided
                        if tags_to_add[j]:
                            for tag in tags_to_add[j]:
                                ku.add_tag(tag)

                        if tags_to_remove[j]:
                            for tag in tags_to_remove[j]:
                                ku.remove_tag(tag)

                        # Update the knowledge unit
                        result = await self.memory_manager.update_knowledge(
                            unique_id=unique_ids[j],
                            content=contents[j],
                            context=contexts[j],
                            source=sources[j],
                        )

                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error updating item {j}: {str(e)}")
                        batch_results.append(False)

                all_results.extend(batch_results)

            success_count = len([r for r in all_results if r])
            logger.info(f"Successfully updated {success_count}/{n_items} knowledge units")
            return all_results

        except Exception as e:
            error_msg = f"Error in batch update operation: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def delete_knowledge_batch(self, unique_ids: list[str]) -> list[bool]:
        """
        Delete multiple knowledge units in batches.

        Args:
            unique_ids: List of knowledge unit IDs to delete

        Returns:
            List of success flags for each delete operation

        Raises:
            MemoryError: If batch operation fails
        """
        n_items = len(unique_ids)
        all_results = []

        try:
            for i in range(0, n_items, self.batch_size):
                batch_end = min(i + self.batch_size, n_items)
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}: items {i} to {batch_end}"
                )

                # Create batch of tasks
                tasks = []
                for j in range(i, batch_end):
                    task = self.memory_manager.delete_knowledge(unique_ids[j])
                    tasks.append(task)

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle any exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in delete operation: {str(result)}")
                        all_results.append(False)
                    else:
                        all_results.append(result)

            success_count = len([r for r in all_results if r])
            logger.info(f"Successfully deleted {success_count}/{n_items} knowledge units")
            return all_results

        except Exception as e:
            error_msg = f"Error in batch delete operation: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def bulk_import_from_texts(
        self,
        texts: list[str],
        source: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_process: bool = True,
        add_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        Import and chunk a collection of texts into knowledge units.

        Args:
            texts: List of text documents to import
            source: Source of the knowledge
            chunk_size: Size of chunks to split texts into
            chunk_overlap: Overlap between chunks
            batch_process: Whether to process in batches
            add_metadata: Optional metadata to add to all knowledge units
            tags: Optional tags to add to all knowledge units

        Returns:
            List of unique IDs for the added knowledge units

        Raises:
            MemoryError: If import operation fails
        """
        # Split texts into chunks
        all_chunks = []
        all_contexts = []

        for text in texts:
            # Simple chunking by splitting text into segments
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                # Create context from surrounding chunks
                context = ""
                if i > 0:
                    # Add previous chunk as context
                    context += f"Previous text: {chunks[i - 1][:200]}...\n"
                if i < len(chunks) - 1:
                    # Add next chunk as context
                    context += f"Next text: {chunks[i + 1][:200]}...\n"

                all_chunks.append(chunk)
                all_contexts.append(context)

        # Prepare metadata and tags
        n_chunks = len(all_chunks)
        metadata_list = [add_metadata.copy() if add_metadata else {} for _ in range(n_chunks)]
        tags_list = [tags.copy() if tags else [] for _ in range(n_chunks)]

        # Add chunks as knowledge units
        if batch_process:
            return await self.add_knowledge_batch(
                contents=all_chunks,
                source=source,
                contexts=all_contexts,
                metadata_list=metadata_list,
                tags_list=tags_list,
            )
        else:
            # Process sequentially if batching is disabled
            ids = []
            for i in range(n_chunks):
                try:
                    unit_id = await self.memory_manager.add_knowledge(
                        content=all_chunks[i],
                        source=source,
                        context=all_contexts[i],
                        metadata=metadata_list[i],
                    )

                    # Add tags if provided
                    if tags_list[i]:
                        ku = await self.memory_manager.retrieve_knowledge(unit_id)
                        if ku:
                            for tag in tags_list[i]:
                                ku.add_tag(tag)
                            await self.memory_manager.update_knowledge(
                                ku.unique_id, None, None, None
                            )

                    ids.append(unit_id)
                except Exception as e:
                    logger.error(f"Error adding chunk {i}: {str(e)}")
                    ids.append(None)

            return ids

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # If text is shorter than chunk_size, return it as is
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Get chunk of size chunk_size or remainder if shorter
            end = min(start + chunk_size, len(text))

            # If not at the beginning and not at the end of text,
            # try to find a good split point (e.g., end of sentence or paragraph)
            if start > 0 and end < len(text):
                # Look for paragraph break first
                paragraph_end = text.rfind("\n\n", start, end)
                if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                    end = paragraph_end + 2  # Include the newlines
                else:
                    # Look for sentence end
                    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                        sentence_end = text.rfind(sep, start, end)
                        if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                            end = sentence_end + len(sep)
                            break

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position for next chunk, accounting for overlap
            start = end - chunk_overlap if end < len(text) else len(text)

        return chunks


async def process_in_batches(
    items: list[T],
    processor: Callable[[list[T]], asyncio.Future],
    batch_size: int = 50,
    description: str = "items",
) -> list[Any]:
    """
    Process a list of items in batches using the provided processor function.

    Args:
        items: List of items to process
        processor: Async function that processes a batch of items
        batch_size: Size of each batch
        description: Description of the items for logging

    Returns:
        Combined results from all batches

    Raises:
        MemoryError: If batch processing fails
    """
    n_items = len(items)
    all_results = []

    try:
        for i in range(0, n_items, batch_size):
            batch_end = min(i + batch_size, n_items)
            logger.info(f"Processing batch of {description}: {i} to {batch_end}")

            batch = items[i:batch_end]
            batch_results = await processor(batch)
            all_results.extend(batch_results)

        return all_results
    except Exception as e:
        error_msg = f"Error processing batches of {description}: {str(e)}"
        logger.error(error_msg)
        raise MemoryError(error_msg) from e
