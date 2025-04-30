"""
Vector store implementation using ChromaDB.

This module provides a concrete implementation of the VectorStore interface
using ChromaDB as the underlying storage mechanism for vector embeddings.
"""

import logging
from datetime import datetime
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from ltm_agent.core.config import Settings, get_settings
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store implementation using ChromaDB.

    This class provides a concrete implementation of the VectorStore interface
    using ChromaDB as the backend for storing and retrieving vector embeddings.
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize the ChromaDB vector store.

        Args:
            config: Optional Settings object for configuration
        """
        self.config = config or get_settings()
        self.collection_name = "knowledge_units"

        # Initialize ChromaDB client with persistent storage (new API)
        self.client = chromadb.PersistentClient(path=self.config.vector_db_path)

        # Initialize embedding model based on configuration
        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)

        # We'll initialize the collection in the initialize method
        self.collection = None
        self.initialized = False

        logger.info(f"Initialized ChromaVectorStore with path: {self.config.vector_db_path}")
        super().__init__(self.client)

    async def initialize(self) -> None:
        """Initialize the ChromaDB collection."""
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"description": "Long-term memory knowledge units"}
        )

        # Create index if needed (in a real implementation, this would be more sophisticated)
        # ChromaDB automatically maintains the index as documents are added

        self.initialized = True
        logger.info(f"Initialized ChromaDB collection: {self.collection_name}")

    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """Add a knowledge unit to ChromaDB."""
        # Generate embedding if not provided
        if knowledge_unit.embedding_vector is None:
            embedding = await self.generate_embedding(knowledge_unit.original_chunk)
            knowledge_unit.embedding_vector = embedding

        # Serialize metadata
        metadata = knowledge_unit.model_dump(exclude={"embedding_vector"})
        # Convert datetime to string
        metadata["timestamp"] = metadata["timestamp"].isoformat()

        # Add to ChromaDB
        self.collection.add(
            ids=[knowledge_unit.unique_id],
            embeddings=[knowledge_unit.embedding_vector],
            metadatas=[metadata],
            documents=[knowledge_unit.original_chunk],
        )

        return knowledge_unit.unique_id

    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """Retrieve a knowledge unit from ChromaDB by ID."""
        result = self.collection.get(
            ids=[unique_id], include=["embeddings", "metadatas", "documents"]
        )

        if not result["ids"]:
            return None

        # Extract the data
        metadata = result["metadatas"][0]
        document = result["documents"][0]
        embedding = result["embeddings"][0]

        # Convert timestamp back to datetime
        metadata["timestamp"] = datetime.fromisoformat(metadata["timestamp"])

        # Construct and return the knowledge unit
        metadata["original_chunk"] = document
        metadata["embedding_vector"] = embedding
        return KnowledgeUnit(**metadata)

    async def _delete_implementation(self, unique_id: str) -> bool:
        """Delete a knowledge unit from ChromaDB."""
        try:
            self.collection.delete(ids=[unique_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting knowledge unit {unique_id}: {str(e)}")
            return False

    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Update a knowledge unit in ChromaDB."""
        # Check if the knowledge unit exists
        existing = await self._get_implementation(knowledge_unit.unique_id)
        if not existing:
            return False

        # Delete the existing entry
        await self._delete_implementation(knowledge_unit.unique_id)

        # Add the updated entry
        await self._add_implementation(knowledge_unit)

        return True

    async def _search_implementation(
        self, query: str, limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units in ChromaDB based on text query."""
        # Generate embedding for the query
        query_embedding = await self.generate_embedding(query)

        # Search using the embedding
        return await self._similarity_search_implementation(query_embedding, limit, filter_criteria)

    async def _similarity_search_implementation(
        self, embedding: list[float], limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units in ChromaDB based on embedding similarity."""
        # Convert filter criteria to ChromaDB format if provided
        where = {}
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key in ["knowledge_source", "contextual_text"]:
                    where[key] = value

        # Execute the query
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where if where else None,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        # Convert results to KnowledgeUnit objects
        knowledge_units = []

        if not results["ids"][0]:
            return []

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            embedding_vector = results["embeddings"][0][i]
            distance = results["distances"][0][i]

            # Convert distance to similarity score (1 - normalized distance)
            # ChromaDB uses cosine distance, which is in [0,2]
            # Convert to a similarity score in [0,1] where 1 is most similar
            similarity = 1 - (distance / 2)

            # Convert timestamp back to datetime
            metadata["timestamp"] = datetime.fromisoformat(metadata["timestamp"])

            # Construct the knowledge unit
            metadata["original_chunk"] = document
            metadata["embedding_vector"] = embedding_vector
            knowledge_units.append((KnowledgeUnit(**metadata), similarity))

        return knowledge_units

    async def _list_implementation(
        self,
        limit: int,
        offset: int,
        filter_criteria: dict[str, Any] | None,
        sort_by: str | None,
        sort_order: str,
    ) -> list[KnowledgeUnit]:
        """List knowledge units from ChromaDB."""
        # Convert filter criteria to ChromaDB format if provided
        where = {}
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key in ["knowledge_source", "contextual_text"]:
                    where[key] = value

        # ChromaDB doesn't support offset pagination directly
        # We'll need to get more results and then apply offset in memory
        # This is not efficient for large datasets but works for demo purposes
        results = self.collection.get(
            where=where if where else None, include=["embeddings", "metadatas", "documents"]
        )

        # Convert results to KnowledgeUnit objects
        knowledge_units = []

        if not results["ids"]:
            return []

        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            embedding_vector = results["embeddings"][i]

            # Convert timestamp back to datetime
            metadata["timestamp"] = datetime.fromisoformat(metadata["timestamp"])

            # Construct the knowledge unit
            metadata["original_chunk"] = document
            metadata["embedding_vector"] = embedding_vector
            knowledge_units.append(KnowledgeUnit(**metadata))

        # Apply sorting if requested
        if sort_by and knowledge_units and hasattr(knowledge_units[0], sort_by):
            reverse = sort_order.lower() == "desc"
            knowledge_units.sort(key=lambda ku: getattr(ku, sort_by), reverse=reverse)

        # Apply pagination
        return knowledge_units[offset : offset + limit]

    async def _count_implementation(self, filter_criteria: dict[str, Any] | None) -> int:
        """Count knowledge units in ChromaDB."""
        # Convert filter criteria to ChromaDB format if provided
        where = {}
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key in ["knowledge_source", "contextual_text"]:
                    where[key] = value

        # Get count
        results = self.collection.get(
            where=where if where else None,
            include=[],  # Only need IDs for counting
        )

        return len(results["ids"])

    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """Generate an embedding using the sentence transformer model."""
        # Generate embedding
        embedding = self.embedding_model.encode(text)

        # Convert to a list of floats
        return embedding.tolist()
