"""
Base implementations for memory systems.

This module provides base classes that implement common functionality for
memory systems, providing a foundation for specific implementations to build upon.
"""

import builtins
import logging
from abc import abstractmethod
from typing import Any, Generic, TypeVar

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.interfaces import MemoryStore, VectorStore

# Type variable for the storage backend
T = TypeVar("T")

# Configure logging
logger = logging.getLogger(__name__)


class BaseMemoryStore(MemoryStore, Generic[T]):
    """
    Base implementation of the MemoryStore interface.

    This class provides common functionality and a consistent API for memory storage
    implementations. It delegates specific storage operations to the underlying
    storage backend.
    """

    def __init__(self, storage_backend: T):
        """
        Initialize the BaseMemoryStore with a storage backend.

        Args:
            storage_backend: The backend storage system to use
        """
        self.storage = storage_backend
        self.initialized = False
        logger.debug(
            f"Initialized {self.__class__.__name__} with {storage_backend.__class__.__name__}"
        )

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the memory store.

        This method should be called before any other method to ensure the
        storage backend is properly set up. Implementations should set
        self.initialized to True when complete.
        """
        pass

    async def _ensure_initialized(self) -> None:
        """
        Ensure the memory store is initialized before operations.

        Raises:
            RuntimeError: If the memory store has not been initialized
        """
        if not self.initialized:
            await self.initialize()

    async def add(self, knowledge_unit: KnowledgeUnit) -> str:
        """Add a knowledge unit to the memory store."""
        await self._ensure_initialized()
        logger.debug(f"Adding knowledge unit: {knowledge_unit.unique_id}")
        return await self._add_implementation(knowledge_unit)

    @abstractmethod
    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """
        Implementation-specific method for adding a knowledge unit.

        Args:
            knowledge_unit: The KnowledgeUnit to add

        Returns:
            str: The unique ID of the added knowledge unit
        """
        pass

    async def get(self, unique_id: str) -> KnowledgeUnit | None:
        """Retrieve a specific knowledge unit by its unique ID."""
        await self._ensure_initialized()
        logger.debug(f"Getting knowledge unit: {unique_id}")
        return await self._get_implementation(unique_id)

    @abstractmethod
    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Implementation-specific method for retrieving a knowledge unit.

        Args:
            unique_id: The unique ID of the knowledge unit to retrieve

        Returns:
            Optional[KnowledgeUnit]: The retrieved knowledge unit or None if not found
        """
        pass

    async def delete(self, unique_id: str) -> bool:
        """Delete a knowledge unit from the memory store."""
        await self._ensure_initialized()
        logger.debug(f"Deleting knowledge unit: {unique_id}")
        return await self._delete_implementation(unique_id)

    @abstractmethod
    async def _delete_implementation(self, unique_id: str) -> bool:
        """
        Implementation-specific method for deleting a knowledge unit.

        Args:
            unique_id: The unique ID of the knowledge unit to delete

        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        pass

    async def update(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Update an existing knowledge unit in the memory store."""
        await self._ensure_initialized()
        logger.debug(f"Updating knowledge unit: {knowledge_unit.unique_id}")
        return await self._update_implementation(knowledge_unit)

    @abstractmethod
    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """
        Implementation-specific method for updating a knowledge unit.

        Args:
            knowledge_unit: The knowledge unit to update

        Returns:
            bool: True if the update was successful, False otherwise
        """
        pass

    async def search(
        self, query: str, limit: int = 10, filter_criteria: dict[str, Any] | None = None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units related to the query."""
        await self._ensure_initialized()
        logger.debug(f"Searching for: '{query}' with limit={limit}")
        return await self._search_implementation(query, limit, filter_criteria)

    @abstractmethod
    async def _search_implementation(
        self, query: str, limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Implementation-specific method for searching knowledge units.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter the results

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of knowledge units with relevance scores
        """
        pass

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_criteria: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> list[KnowledgeUnit]:
        """List knowledge units in the memory store."""
        await self._ensure_initialized()
        logger.debug(f"Listing knowledge units with limit={limit}, offset={offset}")
        return await self._list_implementation(limit, offset, filter_criteria, sort_by, sort_order)

    @abstractmethod
    async def _list_implementation(
        self,
        limit: int,
        offset: int,
        filter_criteria: dict[str, Any] | None,
        sort_by: str | None,
        sort_order: str,
    ) -> builtins.list[KnowledgeUnit]:
        """
        Implementation-specific method for listing knowledge units.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            filter_criteria: Optional criteria to filter the results
            sort_by: Optional field to sort by
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            List[KnowledgeUnit]: List of knowledge units
        """
        pass

    async def count(self, filter_criteria: dict[str, Any] | None = None) -> int:
        """Count the number of knowledge units matching the criteria."""
        await self._ensure_initialized()
        logger.debug(f"Counting knowledge units with filter: {filter_criteria}")
        return await self._count_implementation(filter_criteria)

    @abstractmethod
    async def _count_implementation(self, filter_criteria: dict[str, Any] | None) -> int:
        """
        Implementation-specific method for counting knowledge units.

        Args:
            filter_criteria: Optional criteria to filter the count

        Returns:
            int: Number of matching knowledge units
        """
        pass


class BaseVectorStore(BaseMemoryStore[T], VectorStore):
    """
    Base implementation of the VectorStore interface.

    This class extends BaseMemoryStore to provide additional functionality
    for vector-based storage systems.
    """

    async def similarity_search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units based on embedding similarity."""
        await self._ensure_initialized()
        logger.debug(f"Performing similarity search with embedding dimension {len(embedding)}")
        return await self._similarity_search_implementation(embedding, limit, filter_criteria)

    @abstractmethod
    async def _similarity_search_implementation(
        self, embedding: list[float], limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Implementation-specific method for similarity search.

        Args:
            embedding: The query embedding vector
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter the results

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of knowledge units with similarity scores
        """
        pass

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for the given text."""
        await self._ensure_initialized()
        logger.debug(f"Generating embedding for text: '{text[:50]}...' if len(text) > 50 else text")
        return await self._generate_embedding_implementation(text)

    @abstractmethod
    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """
        Implementation-specific method for generating embeddings.

        Args:
            text: The text to generate an embedding for

        Returns:
            List[float]: The embedding vector
        """
        pass
