"""
Interfaces for memory operations in the LTM Agent.

This module defines the abstract base classes and interfaces that all memory
implementations must adhere to. It establishes the contract for storing,
retrieving, and managing knowledge units in the agent's memory.
"""

from abc import ABC, abstractmethod
from typing import Any

from ltm_agent.core.models import KnowledgeUnit


class MemoryStore(ABC):
    """
    Abstract base class for memory storage implementations.

    This interface defines the methods that any memory storage system must implement,
    providing a consistent API for storing and retrieving knowledge units regardless
    of the underlying storage mechanism.
    """

    @abstractmethod
    async def add(self, knowledge_unit: KnowledgeUnit) -> str:
        """
        Add a knowledge unit to the memory store.

        Args:
            knowledge_unit: The KnowledgeUnit to store

        Returns:
            str: The unique ID of the stored knowledge unit
        """
        pass

    @abstractmethod
    async def get(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Retrieve a specific knowledge unit by its unique ID.

        Args:
            unique_id: The unique identifier of the knowledge unit

        Returns:
            Optional[KnowledgeUnit]: The retrieved knowledge unit or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the memory store.

        Args:
            unique_id: The unique identifier of the knowledge unit to delete

        Returns:
            bool: True if the knowledge unit was successfully deleted, False otherwise
        """
        pass

    @abstractmethod
    async def update(self, knowledge_unit: KnowledgeUnit) -> bool:
        """
        Update an existing knowledge unit in the memory store.

        Args:
            knowledge_unit: The updated KnowledgeUnit (must have the same unique_id)

        Returns:
            bool: True if the knowledge unit was successfully updated, False otherwise
        """
        pass

    @abstractmethod
    async def search(
        self, query: str, limit: int = 10, filter_criteria: dict[str, Any] | None = None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for knowledge units related to the query.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter the search results

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of knowledge units with their relevance scores
        """
        pass

    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_criteria: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> list[KnowledgeUnit]:
        """
        List knowledge units in the memory store.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            filter_criteria: Optional criteria to filter the results
            sort_by: Optional field to sort by
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            List[KnowledgeUnit]: List of matching knowledge units
        """
        pass

    @abstractmethod
    async def count(self, filter_criteria: dict[str, Any] | None = None) -> int:
        """
        Count the number of knowledge units matching the criteria.

        Args:
            filter_criteria: Optional criteria to filter the count

        Returns:
            int: Number of matching knowledge units
        """
        pass


class VectorStore(MemoryStore):
    """
    Extension of the MemoryStore interface for vector-based storage implementations.

    This interface adds vector-specific methods for storing and retrieving embeddings.
    """

    @abstractmethod
    async def similarity_search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for knowledge units based on embedding similarity.

        Args:
            embedding: The query embedding vector
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter the search results

        Returns:
            List[Tuple[KnowledgeUnit, float]]: List of knowledge units with their similarity scores
        """
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: The text to generate an embedding for

        Returns:
            List[float]: The embedding vector
        """
        pass


class HybridRetriever(ABC):
    """
    Interface for hybrid retrieval implementations.

    This interface defines methods for combining results from multiple
    retrieval strategies, such as vector similarity and BM25 lexical search.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Perform a hybrid search using both vector and BM25 retrieval methods.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter search results
            vector_weight: Weight to apply to vector search results (0.0 to 1.0)
            bm25_weight: Weight to apply to BM25 search results (0.0 to 1.0)

        Returns:
            List[Tuple[KnowledgeUnit, float]]: Combined and re-ranked results
                                              with their relevance scores
        """
        pass

    @abstractmethod
    async def set_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """
        Set the weights used for combining vector and BM25 search results.

        Args:
            vector_weight: Weight to apply to vector search results (0.0 to 1.0)
            bm25_weight: Weight to apply to BM25 search results (0.0 to 1.0)
        """
        pass

    @abstractmethod
    async def rerank_results(
        self,
        query: str,
        combined_results: list[tuple[KnowledgeUnit, float]],
        limit: int = 10,
        use_reranker: bool = False,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Rerank combined results using optional advanced reranking strategies.

        Args:
            query: The original search query
            combined_results: List of knowledge units with their initial scores
            limit: Maximum number of results to return
            use_reranker: Whether to use an advanced reranker (if available)

        Returns:
            List[Tuple[KnowledgeUnit, float]]: Reranked results with updated scores
        """
        pass


class BaseContextualizer(ABC):
    """
    Abstract base class for contextualizers.

    A BaseContextualizer generates or enhances contextual information for knowledge units.
    This helps in organizing, categorizing, and retrieving knowledge more effectively.
    """

    @abstractmethod
    def enhance_context(
        self, content: str, existing_context: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate or enhance contextual information for knowledge content.

        Args:
            content: The main content of the knowledge unit
            existing_context: Optional existing context to enhance
            metadata: Optional existing metadata to enhance

        Returns:
            Dict with two keys:
            - 'contextual_text': Enhanced context for the knowledge unit
            - 'metadata': Enhanced metadata for the knowledge unit
        """
        pass
