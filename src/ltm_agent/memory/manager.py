"""
Memory manager for the Long-Term Memory Agent.

This module provides a high-level interface for managing the agent's memory,
including storing, retrieving, and organizing knowledge units.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ltm_agent.core.config import Settings, get_settings
from ltm_agent.core.exceptions import (
    EmbeddingError,
    KnowledgeUnitDeleteError,
    KnowledgeUnitNotFoundError,
    KnowledgeUnitUpdateError,
    MemoryError,
    MemoryInitializationError,
)
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.contextualizer import Contextualizer, SimpleContextualizer
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.in_memory_store import InMemoryVectorStore
from ltm_agent.memory.interfaces import HybridRetriever, VectorStore

# Configure logging
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory manager for the Long-Term Memory Agent.

    This class provides a high-level interface for interacting with the agent's memory
    system, abstracting away the details of the underlying storage implementation.
    """

    def __init__(
        self,
        memory_store: VectorStore | None = None,
        contextualizer: Contextualizer | None = None,
        bm25_store: BaseBM25Store | None = None,
        config: Settings | None = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize the memory manager.

        Args:
            memory_store: Optional memory store to use. If not provided, an
                InMemoryVectorStore will be used.
            contextualizer: Optional contextualizer to use for enhancing knowledge.
                If not provided, a SimpleContextualizer will be used.
            bm25_store: Optional BM25 store for lexical search. If not provided,
                it will be created on demand when needed.
            config: Optional configuration for the memory manager.
            max_retries: Maximum number of retry attempts for operations.
            retry_delay: Delay between retry attempts in seconds.
        """
        self.config = config or get_settings()
        self.memory_store = memory_store or InMemoryVectorStore()
        self.contextualizer = contextualizer or SimpleContextualizer()
        self.bm25_store = bm25_store
        self.initialized = False
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized MemoryManager with {self.memory_store.__class__.__name__}")

    async def _retry_operation(self, operation, *args, **kwargs):
        """
        Retry an operation with exponential backoff.

        Args:
            operation: The async operation to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            MemoryError: If all retry attempts fail
        """
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    # Calculate exponential backoff delay
                    delay = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(
                        f"Operation failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Operation failed after {self.max_retries} attempts: {str(e)}")

        # If we get here, all retries failed
        error_msg = f"Operation failed after {self.max_retries} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        raise MemoryError(error_msg)

    async def initialize(self) -> None:
        """
        Initialize the memory manager and underlying storage.

        Raises:
            MemoryInitializationError: If initialization fails
        """
        try:
            await self.memory_store.initialize()
            self.initialized = True
            logger.info("MemoryManager initialized")
        except Exception as e:
            error_msg = f"Failed to initialize memory manager: {str(e)}"
            logger.error(error_msg)
            raise MemoryInitializationError(error_msg) from e

    async def _ensure_initialized(self) -> None:
        """
        Ensure the memory manager is initialized.

        Raises:
            MemoryInitializationError: If initialization fails
        """
        if not self.initialized:
            await self.initialize()

    async def process_potential_knowledge(
        self,
        content: str,
        source: str = "action",
        context: str = "",
        metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        update_if_similar: bool = True,
    ) -> str:
        """
        Process new potential knowledge text, deciding whether to add or update.

        This is the main orchestration method that handles the decision logic
        and delegates to the appropriate add/update methods.

        Args:
            content: The content of the knowledge unit
            source: The source of the knowledge (action, feedback, or corpus)
            context: Optional contextual information
            metadata: Optional metadata for the knowledge unit
            similarity_threshold: Threshold for considering content similar (0.0 to 1.0)
            update_if_similar: Whether to update existing similar knowledge (True) or
                               add as new regardless of similarity (False)

        Returns:
            str: The unique ID of the added or updated knowledge unit

        Raises:
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # Enhance the context and metadata using the contextualizer
            enhanced = await self.contextualizer.generate_context(
                content=content, existing_context=context, metadata=metadata
            )

            enhanced_context = enhanced["context"]
            enhanced_metadata = enhanced["metadata"]

            logger.info(f"Processing potential knowledge: '{content[:50]}...' (source: {source})")

            # Determine whether to add as new or update existing
            if not update_if_similar:
                # Force add as new if update_if_similar is False
                return await self._add_new_knowledge(
                    content=content,
                    source=source,
                    context=enhanced_context,
                    metadata=enhanced_metadata,
                )

            # Use the find_similar_knowledge method to check for similar content
            similar_unit = await self._find_similar_knowledge(
                content=content, similarity_threshold=similarity_threshold
            )

            if similar_unit:
                # Update existing knowledge
                logger.info(
                    f"Found similar knowledge unit {similar_unit.unique_id}, updating instead of adding new"
                )
                return await self._update_existing_knowledge(
                    unique_id=similar_unit.unique_id,
                    content=content,
                    source=source,
                    context=enhanced_context,
                    metadata=enhanced_metadata,
                )
            else:
                # Add as new knowledge
                logger.info("No similar knowledge found, adding as new")
                return await self._add_new_knowledge(
                    content=content,
                    source=source,
                    context=enhanced_context,
                    metadata=enhanced_metadata,
                )

        except Exception as e:
            error_msg = f"Error processing potential knowledge: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def _add_new_knowledge(
        self,
        content: str,
        source: str,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a new knowledge unit to memory.

        This internal helper method handles the actual addition of new knowledge.

        Args:
            content: The content of the knowledge unit
            source: The source of the knowledge (action, feedback, or corpus)
            context: Contextual information
            metadata: Metadata for the knowledge unit

        Returns:
            str: The unique ID of the stored knowledge unit
        """
        # Create a new knowledge unit
        knowledge_unit = KnowledgeUnit(
            original_chunk=content,
            contextual_text=context,
            knowledge_source=source,
            metadata=metadata or {},
        )

        # Add the knowledge unit to the memory store with retry
        unique_id = await self._retry_operation(self.memory_store.add, knowledge_unit)

        # If we have a BM25 store, update it with the new knowledge unit
        if self.bm25_store is not None:
            try:
                await self.bm25_store.update_unit(unique_id, knowledge_unit)
                logger.debug(f"Updated BM25 index with new knowledge unit {unique_id}")
            except Exception as e:
                logger.warning(f"Failed to update BM25 index with new unit: {e}")

        logger.info(f"Added new knowledge unit with ID: {unique_id}")
        return unique_id

    async def _update_existing_knowledge(
        self,
        unique_id: str,
        content: str,
        source: str | None = None,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Update an existing knowledge unit.

        Args:
            unique_id: The ID of the knowledge unit to update
            content: The updated content
            source: The source of the knowledge (action, feedback, or corpus)
            context: Optional contextual information
            metadata: Optional updated metadata

        Returns:
            str: The unique ID of the updated knowledge unit

        Raises:
            KnowledgeUnitNotFoundError: If the knowledge unit is not found
            KnowledgeUnitUpdateError: If the update fails
        """
        await self._ensure_initialized()

        try:
            existing_ku = await self.retrieve_knowledge(unique_id)
            if not existing_ku:
                raise KnowledgeUnitNotFoundError(f"Knowledge unit {unique_id} not found")

            # Merge the content using our advanced merging algorithm
            original_content = existing_ku.original_chunk
            updated_content = self._merge_content(original_content, content)

            existing_ku.original_chunk = updated_content

            if context is not None:
                existing_ku.contextual_text = context
            if source is not None:
                existing_ku.knowledge_source = source

            # Update metadata
            if metadata:
                if existing_ku.metadata is None:
                    existing_ku.metadata = {}

                # Merge metadata instead of replacing
                for key, value in metadata.items():
                    existing_ku.metadata[key] = value

                # Add update tracking
                if "updates" not in existing_ku.metadata:
                    existing_ku.metadata["updates"] = []

                existing_ku.metadata["updates"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": source or existing_ku.knowledge_source,
                        "original_length": len(original_content),
                        "new_length": len(updated_content),
                    }
                )

            # Update timestamps
            existing_ku.last_accessed = datetime.now()
            existing_ku.updated_at = datetime.now()

            # Update the knowledge unit with retry
            try:
                # Generate new embedding for updated content
                existing_ku.embedding_vector = await self._retry_operation(
                    self.memory_store.generate_embedding, updated_content
                )
            except Exception as e:
                logger.warning(f"Failed to update embedding for knowledge unit {unique_id}: {e}")

            # Update the knowledge unit
            result = await self._retry_operation(self.memory_store.update, existing_ku)

            if not result:
                error_msg = f"Failed to update knowledge unit {unique_id}"
                logger.error(error_msg)
                raise KnowledgeUnitUpdateError(error_msg)

            logger.info(f"Updated knowledge unit: {unique_id}")
            return unique_id
        except KnowledgeUnitNotFoundError:
            # Re-raise not found errors
            raise
        except Exception as e:
            error_msg = f"Error updating knowledge unit {unique_id}: {str(e)}"
            logger.error(error_msg)
            raise KnowledgeUnitUpdateError(error_msg) from e

    def _merge_content(self, original_content: str, new_content: str) -> str:
        """
        Advanced content merging using NLP techniques to combine information from original and new content.

        Args:
            original_content: The original content string
            new_content: The new content string to merge

        Returns:
            str: The merged content
        """
        import re
        from collections import Counter

        # Extract sentences for better granularity
        def extract_sentences(text: str) -> list:
            # Improved sentence splitting that handles abbreviations and other edge cases
            text = text.replace("!", ". ").replace("?", ". ")
            # Replace multiple periods (e.g., ...) with a single one for clean splitting
            text = re.sub(r"\.{2,}", ".", text)
            # Split by period followed by space or end of string
            sentences = re.split(r"\.(?:\s|$)", text)
            # Remove empty strings and strip whitespace
            return [s.strip() for s in sentences if s.strip()]

        orig_sentences = extract_sentences(original_content)
        new_sentences = extract_sentences(new_content)

        # Extract key topics from both contents
        def extract_topics(sentences: list) -> Counter:
            # Simple tokenization with stopword removal
            stopwords = {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "about",
                "of",
                "as",
                "its",
                "it",
                "this",
                "that",
                "these",
                "those",
            }

            topics = []
            for sentence in sentences:
                words = re.findall(r"\b\w+\b", sentence.lower())
                topics.extend([w for w in words if w not in stopwords and len(w) > 2])

            return Counter(topics)

        orig_topics = extract_topics(orig_sentences)
        new_topics = extract_topics(new_sentences)

        # Find unique information in the new content
        unique_sentences = []
        updated_sentences = []

        for new_sentence in new_sentences:
            # Check if this sentence contains unique information
            new_sentence_topics = extract_topics([new_sentence])

            # Measure importance of this sentence based on topic uniqueness
            unique_topic_count = sum(1 for topic in new_sentence_topics if topic not in orig_topics)
            total_topic_count = sum(new_sentence_topics.values())

            # If the sentence has unique topics, consider it new information
            if unique_topic_count > 0 and (unique_topic_count / max(1, total_topic_count)) > 0.3:
                unique_sentences.append(new_sentence)
                continue

            # Check sentence similarity with existing content
            is_unique = True
            for orig_sentence in orig_sentences:
                similarity = self._compute_text_similarity(orig_sentence, new_sentence)
                if similarity > 0.6:  # If 60% similar to an existing sentence
                    is_unique = False

                    # Check if the new sentence is more detailed (longer)
                    if len(new_sentence) > len(orig_sentence) * 1.2:
                        # Replace the original with the more detailed version
                        if orig_sentence not in updated_sentences:
                            updated_sentences.append((orig_sentence, new_sentence))
                    break

            if is_unique:
                unique_sentences.append(new_sentence)

        # Create the merged content
        if not updated_sentences and not unique_sentences:
            # No new information, keep original if it's more detailed
            return original_content if len(original_content) >= len(new_content) else new_content

        # Start with the original content
        result = original_content

        # Replace sentences that have more detailed versions
        for orig_sentence, better_sentence in updated_sentences:
            result = result.replace(orig_sentence, better_sentence)

        # Add unique sentences
        if unique_sentences:
            if not result.endswith("."):
                result += "."
            result += " " + ". ".join(unique_sentences)
            if not result.endswith("."):
                result += "."

        return result

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute a simple similarity score between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple Jaccard similarity of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def add_knowledge(
        self,
        content: str,
        source: str,
        context: str = "",
        metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        update_if_similar: bool = True,
    ) -> str:
        """
        Add a new piece of knowledge to the memory or update if similar content exists.

        Args:
            content: The content of the knowledge unit
            source: The source of the knowledge (action, feedback, or corpus)
            context: Optional contextual information
            metadata: Optional metadata for the knowledge unit
            similarity_threshold: Threshold for considering content similar (0.0 to 1.0)
            update_if_similar: Whether to update existing similar knowledge (True) or
                               add as new regardless of similarity (False)

        Returns:
            str: The unique ID of the added or updated knowledge unit

        Raises:
            MemoryError: If the operation fails after retries
        """
        # Delegate to process_potential_knowledge for unified handling
        return await self.process_potential_knowledge(
            content=content,
            source=source,
            context=context,
            metadata=metadata,
            similarity_threshold=similarity_threshold,
            update_if_similar=update_if_similar,
        )

    async def retrieve_knowledge(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Retrieve a specific knowledge unit by its ID.

        Args:
            unique_id: The unique ID of the knowledge unit

        Returns:
            Optional[KnowledgeUnit]: The retrieved knowledge unit or None if not found

        Raises:
            KnowledgeUnitNotFoundError: If the knowledge unit cannot be found
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            result = await self._retry_operation(self.memory_store.get, unique_id)

            if result is None:
                error_msg = f"Knowledge unit not found: {unique_id}"
                logger.warning(error_msg)
                raise KnowledgeUnitNotFoundError(error_msg)

            return result
        except KnowledgeUnitNotFoundError:
            # Re-raise not found errors
            raise
        except Exception as e:
            error_msg = f"Failed to retrieve knowledge unit {unique_id}: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def update_knowledge(
        self,
        unique_id: str,
        content: str | None = None,
        context: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update an existing knowledge unit.

        Args:
            unique_id: The unique ID of the knowledge unit to update
            content: Optional new content for the knowledge unit
            context: Optional new context for the knowledge unit
            source: Optional new source for the knowledge unit
            metadata: Optional new metadata for the knowledge unit

        Returns:
            bool: True if the update was successful

        Raises:
            KnowledgeUnitNotFoundError: If the knowledge unit cannot be found
            KnowledgeUnitUpdateError: If the update fails
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # First retrieve the existing knowledge unit
            existing_ku = await self.retrieve_knowledge(unique_id)

            # Update the fields if provided
            if content is not None:
                await self._update_existing_knowledge(unique_id, content, source, context, metadata)
                return True
            else:
                return False
        except KnowledgeUnitNotFoundError:
            # Re-raise not found errors
            raise
        except Exception as e:
            error_msg = f"Error updating knowledge unit {unique_id}: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def delete_knowledge(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from memory.

        Args:
            unique_id: The unique ID of the knowledge unit to delete

        Returns:
            bool: True if the deletion was successful

        Raises:
            KnowledgeUnitDeleteError: If the deletion fails
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # Delete the knowledge unit with retry
            result = await self._retry_operation(self.memory_store.delete, unique_id)

            if not result:
                error_msg = f"Failed to delete knowledge unit {unique_id}"
                logger.error(error_msg)
                raise KnowledgeUnitDeleteError(error_msg)

            logger.info(f"Deleted knowledge unit: {unique_id}")
            return result
        except Exception as e:
            error_msg = f"Error deleting knowledge unit {unique_id}: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        source_filter: str | None = None,
        min_relevance_score: float = 0.0,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for knowledge units matching the query.

        Args:
            query: The search query
            limit: Maximum number of results to return
            source_filter: Optional filter for knowledge source
            min_relevance_score: Minimum relevance score threshold

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of knowledge units with relevance scores

        Raises:
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # Prepare filter criteria
            filter_criteria = {}
            if source_filter:
                filter_criteria["knowledge_source"] = source_filter

            # Perform the search with retry
            results = await self._retry_operation(
                self.memory_store.search,
                query,
                limit=limit,
                filter_criteria=filter_criteria if filter_criteria else None,
            )

            # Filter by minimum relevance score if specified
            if min_relevance_score > 0:
                results = [(ku, score) for ku, score in results if score >= min_relevance_score]

            return results
        except Exception as e:
            error_msg = f"Error searching knowledge: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def get_related_knowledge(
        self,
        content: str,
        limit: int = 10,
        source_filter: str | None = None,
        min_similarity_score: float = 0.0,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Find knowledge related to the given content.

        Args:
            content: The content to find related knowledge for
            limit: Maximum number of results to return
            source_filter: Optional filter for knowledge source
            min_similarity_score: Minimum similarity score threshold

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of related knowledge units with similarity scores

        Raises:
            EmbeddingError: If embedding generation fails
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # For vector stores, we can use the embedding-based search
            if isinstance(self.memory_store, VectorStore):
                try:
                    # Generate embedding for the content with retry
                    embedding = await self._retry_operation(
                        self.memory_store.generate_embedding, content
                    )
                except Exception as e:
                    error_msg = f"Failed to generate embedding: {str(e)}"
                    logger.error(error_msg)
                    raise EmbeddingError(error_msg) from e

                # Prepare filter criteria
                filter_criteria = {}
                if source_filter:
                    filter_criteria["knowledge_source"] = source_filter

                # Perform similarity search with retry
                results = await self._retry_operation(
                    self.memory_store.similarity_search,
                    embedding,
                    limit=limit,
                    filter_criteria=filter_criteria if filter_criteria else None,
                )

                # Filter by minimum similarity score if specified
                if min_similarity_score > 0:
                    results = [
                        (ku, score) for ku, score in results if score >= min_similarity_score
                    ]

                return results

            # Fall back to text-based search for non-vector stores
            return await self.search_knowledge(content, limit, source_filter, min_similarity_score)
        except EmbeddingError:
            # Re-raise embedding errors
            raise
        except Exception as e:
            error_msg = f"Error getting related knowledge: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def list_knowledge(
        self,
        limit: int = 100,
        offset: int = 0,
        source_filter: str | None = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> list[KnowledgeUnit]:
        """
        List knowledge units in the memory.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            source_filter: Optional filter for knowledge source
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            List[KnowledgeUnit]: List of knowledge units

        Raises:
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # Prepare filter criteria
            filter_criteria = {}
            if source_filter:
                filter_criteria["knowledge_source"] = source_filter

            # Perform the list operation with retry
            return await self._retry_operation(
                self.memory_store.list,
                limit=limit,
                offset=offset,
                filter_criteria=filter_criteria if filter_criteria else None,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        except Exception as e:
            error_msg = f"Error listing knowledge: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def count_knowledge(self, source_filter: str | None = None) -> int:
        """
        Count the number of knowledge units in memory.

        Args:
            source_filter: Optional filter for knowledge source

        Returns:
            int: Number of knowledge units

        Raises:
            MemoryError: If the operation fails after retries
        """
        await self._ensure_initialized()

        try:
            # Prepare filter criteria
            filter_criteria = {}
            if source_filter:
                filter_criteria["knowledge_source"] = source_filter

            # Perform the count operation with retry
            return await self._retry_operation(
                self.memory_store.count,
                filter_criteria=filter_criteria if filter_criteria else None,
            )
        except Exception as e:
            error_msg = f"Error counting knowledge: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def create_hybrid_retriever(
        self,
        bm25_index_path: str | None = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        min_score_threshold: float = 0.0,
        length_penalty_factor: float = 0.1,
    ) -> HybridRetriever:
        """
        Create a hybrid retriever that combines vector and BM25 search.

        Args:
            bm25_index_path: Path to store the BM25 index (defaults to a temp file)
            vector_weight: Weight for vector similarity scores (0.0 to 1.0)
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            min_score_threshold: Minimum score to include in results
            length_penalty_factor: Factor for length penalization in reranking

        Returns:
            HybridRetriever: Configured hybrid retriever

        Raises:
            MemoryError: If creation fails
        """
        await self._ensure_initialized()

        try:
            # Create a temporary file for BM25 index if not provided
            if not bm25_index_path:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
                    bm25_index_path = temp_file.name
                logger.info(f"Using temporary BM25 index file: {bm25_index_path}")

            # Create a BM25 store
            bm25_store = RankBM25Store(index_path=bm25_index_path)

            # Initialize the BM25 store with current knowledge units
            try:
                # List all knowledge units
                units_dict = {}
                async for unit in self.memory_store.list():
                    units_dict[unit.unique_id] = unit

                # Update the BM25 index
                await bm25_store.update_index(units_dict)
                logger.info(f"Initialized BM25 store with {len(units_dict)} knowledge units")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25 store with existing units: {e}")

            # Create and configure the hybrid retriever
            hybrid_retriever = LangchainHybridRetriever(
                vector_store=self.memory_store,
                bm25_store=bm25_store,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                min_score_threshold=min_score_threshold,
                length_penalty_factor=length_penalty_factor,
            )

            logger.info(
                f"Created hybrid retriever with weights: vector={vector_weight}, BM25={bm25_weight}"
            )
            return hybrid_retriever

        except Exception as e:
            error_msg = f"Error creating hybrid retriever: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e

    async def _find_similar_knowledge(
        self, content: str, similarity_threshold: float = 0.85
    ) -> KnowledgeUnit | None:
        """
        Find knowledge units with content similar to the provided content.

        Args:
            content: The content to find similar knowledge units for
            similarity_threshold: Threshold for considering content similar (0.0 to 1.0)

        Returns:
            Optional[KnowledgeUnit]: The most similar knowledge unit, or None if none found
        """
        await self._ensure_initialized()

        try:
            # Use get_related_knowledge to find semantically similar knowledge units
            results = await self.get_related_knowledge(
                content,
                limit=10,  # Increased from 5 to 10 to improve recall
                min_similarity_score=0.0,  # Get all results so we can filter ourselves
            )

            logger.debug(f"Found {len(results)} potentially similar knowledge units")

            if not results:
                return None

            # First check for topic similarity - words that indicate the same subject
            # Extract key topics from the input content
            import re
            from collections import Counter

            # Simple tokenization - extract words and remove common stopwords
            stopwords = {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "about",
                "of",
                "as",
            }

            # Extract words from content
            content_words = re.findall(r"\b\w+\b", content.lower())
            content_topics = [w for w in content_words if w not in stopwords and len(w) > 2]
            content_topics_counter = Counter(content_topics)

            best_match = None
            best_score = 0.0
            best_topic_match = 0

            # Check if any result exceeds the similarity threshold
            for knowledge_unit, similarity_score in results:
                # Get topics from knowledge unit
                unit_words = re.findall(r"\b\w+\b", knowledge_unit.original_chunk.lower())
                unit_topics = [w for w in unit_words if w not in stopwords and len(w) > 2]
                unit_topics_counter = Counter(unit_topics)

                # Count topic matches
                topic_matches = sum((content_topics_counter & unit_topics_counter).values())

                logger.debug(
                    f"Similarity for {knowledge_unit.unique_id}: score={similarity_score:.4f}, "
                    f"topic_matches={topic_matches}"
                )

                # Prioritize results with both good similarity and topic matches
                combined_score = (similarity_score * 0.6) + (
                    min(1.0, topic_matches / max(3, len(content_topics))) * 0.4
                )

                # If this is a better match than what we've seen so far
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = knowledge_unit
                    best_topic_match = topic_matches

            # If the best match exceeds our adjusted threshold, return it
            adjusted_threshold = (
                similarity_threshold * 0.7
            )  # Lower the threshold to 70% of original
            if best_score >= adjusted_threshold:
                logger.info(
                    f"Found similar knowledge unit {best_match.unique_id} "
                    f"with similarity score {best_score:.4f} and {best_topic_match} topic matches"
                )
                return best_match

            return None

        except Exception as e:
            logger.warning(f"Error finding similar knowledge: {str(e)}")
            return None
