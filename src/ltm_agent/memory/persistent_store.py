"""
Persistent vector store implementations for long-term memory.

This module provides adapters for persistent vector databases like Chroma and Pinecone,
enabling efficient scaling to large knowledge bases.
"""

import json
import logging
import time
from datetime import datetime

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
CHROMA_AVAILABLE = False
PINECONE_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    import pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    logger.warning("Pinecone not available. Install with: pip install pinecone-client")


class ChromaVectorStore(VectorStore):
    """
    Vector store implementation using ChromaDB for persistence.

    This provides efficient vector storage and retrieval with local persistence.
    """

    def __init__(
        self,
        collection_name: str = "ltm_agent_memory",
        persist_directory: str | None = "./chroma_db",
        embedding_dimensions: int = 1536,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory where ChromaDB data will be stored
            embedding_dimensions: Dimensions of embedding vectors
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimensions = embedding_dimensions

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        # Cache for knowledge units
        self.knowledge_cache = {}

        # Background tasks for batching
        self._batch_queue = []
        self._batch_size = 100
        self._last_batch_time = time.time()
        self._batch_timeout = 5  # seconds

        logger.info(f"ChromaVectorStore initialized with collection: {collection_name}")

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            return self.client.get_collection(self.collection_name)
        except ValueError:
            return self.client.create_collection(
                name=self.collection_name, metadata={"created_at": datetime.now().isoformat()}
            )

    def add(self, unit: KnowledgeUnit) -> None:
        """
        Add a knowledge unit to the vector store.

        Args:
            unit: Knowledge unit to add
        """
        # Prepare data for ChromaDB
        embeddings = (
            unit.embedding.tolist() if hasattr(unit.embedding, "tolist") else unit.embedding
        )
        content = unit.original_chunk
        metadata = {
            "source": unit.source or "",
            "created_at": unit.created_at or datetime.now().isoformat(),
            "knowledge_unit": json.dumps(unit.to_dict()),
        }

        # Add custom metadata if present
        if unit.metadata:
            for key, value in unit.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    # Convert complex types to JSON string
                    metadata[f"{key}_json"] = json.dumps(value)

        # Add to queue for batch processing
        self._batch_queue.append(
            {
                "id": unit.unique_id,
                "embeddings": embeddings,
                "documents": content,
                "metadata": metadata,
            }
        )

        # Cache the knowledge unit
        self.knowledge_cache[unit.unique_id] = unit

        # Process batch if needed
        self._process_batch_if_needed()

    def _process_batch_if_needed(self, force: bool = False) -> None:
        """
        Process batch if it's full or timed out.

        Args:
            force: Whether to force processing regardless of batch size
        """
        current_time = time.time()
        batch_timeout_reached = (current_time - self._last_batch_time) > self._batch_timeout

        if force or len(self._batch_queue) >= self._batch_size or batch_timeout_reached:
            if not self._batch_queue:
                return

            try:
                # Extract batched data
                ids = [item["id"] for item in self._batch_queue]
                documents = [item["documents"] for item in self._batch_queue]
                embeddings = [item["embeddings"] for item in self._batch_queue]
                metadatas = [item["metadata"] for item in self._batch_queue]

                # Add to ChromaDB
                self.collection.add(
                    ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
                )

                logger.debug(f"Added batch of {len(ids)} items to ChromaDB")

            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")

            # Reset batch
            self._batch_queue = []
            self._last_batch_time = current_time

    def update(self, unit: KnowledgeUnit) -> None:
        """
        Update a knowledge unit in the store.

        Args:
            unit: Knowledge unit to update
        """
        # Process any pending additions first
        self._process_batch_if_needed(force=True)

        # Prepare updated data
        embeddings = (
            unit.embedding.tolist() if hasattr(unit.embedding, "tolist") else unit.embedding
        )
        content = unit.original_chunk
        metadata = {
            "source": unit.source or "",
            "created_at": unit.created_at or datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "knowledge_unit": json.dumps(unit.to_dict()),
        }

        # Add custom metadata if present
        if unit.metadata:
            for key, value in unit.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    # Convert complex types to JSON string
                    metadata[f"{key}_json"] = json.dumps(value)

        try:
            # Update in ChromaDB
            self.collection.update(
                ids=[unit.unique_id],
                documents=[content],
                embeddings=[embeddings],
                metadatas=[metadata],
            )

            # Update the cache
            self.knowledge_cache[unit.unique_id] = unit

        except Exception as e:
            logger.error(f"Error updating ChromaDB: {e}")

    def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the store.

        Args:
            unique_id: ID of the knowledge unit to delete

        Returns:
            True if deleted, False otherwise
        """
        # Process any pending additions first
        self._process_batch_if_needed(force=True)

        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[unique_id])

            # Remove from cache
            if unique_id in self.knowledge_cache:
                del self.knowledge_cache[unique_id]

            return True

        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return False

    def get(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get a knowledge unit by ID.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        # Check cache first
        if unique_id in self.knowledge_cache:
            return self.knowledge_cache[unique_id]

        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Get from ChromaDB
            result = self.collection.get(
                ids=[unique_id], include=["documents", "embeddings", "metadatas"]
            )

            if not result or not result["ids"]:
                return None

            # Extract knowledge unit from metadata
            metadata = result["metadatas"][0]
            knowledge_unit_dict = json.loads(metadata.get("knowledge_unit", "{}"))

            # Create and cache knowledge unit
            unit = KnowledgeUnit.from_dict(knowledge_unit_dict)
            self.knowledge_cache[unique_id] = unit

            return unit

        except Exception as e:
            logger.error(f"Error getting from ChromaDB: {e}")
            return None

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
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Convert threshold to distance (Chroma uses distance, not similarity)
            # For cosine similarity, this conversion works
            distance_threshold = 1.0 - threshold if threshold > 0 else None

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["ids"] or not results["ids"][0]:
                return []

            # Extract results
            units_with_scores = []

            for i, unit_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else 0

                # Skip if below threshold
                similarity = 1.0 - distance
                if similarity < threshold:
                    continue

                # Get or create knowledge unit
                if unit_id in self.knowledge_cache:
                    unit = self.knowledge_cache[unit_id]
                else:
                    knowledge_unit_dict = json.loads(metadata.get("knowledge_unit", "{}"))
                    unit = KnowledgeUnit.from_dict(knowledge_unit_dict)
                    self.knowledge_cache[unit_id] = unit

                units_with_scores.append((unit, similarity))

            return units_with_scores

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    def list_all(self) -> list[KnowledgeUnit]:
        """
        List all knowledge units in the store.

        Returns:
            List of all knowledge units
        """
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Get all from ChromaDB
            results = self.collection.get(include=["metadatas"])

            if not results or not results["ids"]:
                return []

            # Extract knowledge units
            units = []

            for i, unit_id in enumerate(results["ids"]):
                # Skip if already in cache
                if unit_id in self.knowledge_cache:
                    units.append(self.knowledge_cache[unit_id])
                    continue

                # Extract from metadata
                metadata = results["metadatas"][i]
                knowledge_unit_dict = json.loads(metadata.get("knowledge_unit", "{}"))

                unit = KnowledgeUnit.from_dict(knowledge_unit_dict)
                self.knowledge_cache[unit_id] = unit
                units.append(unit)

            return units

        except Exception as e:
            logger.error(f"Error listing all from ChromaDB: {e}")
            return []

    def clear(self) -> None:
        """Clear all knowledge units from the store."""
        try:
            # Reset the collection
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()

            # Clear cache and batch queue
            self.knowledge_cache = {}
            self._batch_queue = []
            self._last_batch_time = time.time()

            logger.info(f"Cleared ChromaDB collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")

    def count(self) -> int:
        """
        Count the number of knowledge units in the store.

        Returns:
            Number of knowledge units
        """
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Get count from ChromaDB
            return self.collection.count()

        except Exception as e:
            logger.error(f"Error counting ChromaDB: {e}")
            return 0

    def close(self) -> None:
        """Close the vector store and persist any pending changes."""
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        # Explicitly persist (though ChromaDB should auto-persist)
        if hasattr(self.client, "persist"):
            try:
                self.client.persist()
                logger.info("ChromaDB state persisted")
            except Exception as e:
                logger.error(f"Error persisting ChromaDB: {e}")


class PineconeVectorStore(VectorStore):
    """
    Vector store implementation using Pinecone for cloud persistence.

    This provides efficient vector storage and retrieval with cloud-based scaling.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str = "ltm-agent-memory",
        namespace: str = "default",
        embedding_dimensions: int = 1536,
        create_index_if_missing: bool = True,
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            embedding_dimensions: Dimensions of embedding vectors
            create_index_if_missing: Whether to create the index if it doesn't exist
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone is not installed. Install with: pip install pinecone-client"
            )

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_dimensions = embedding_dimensions

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Create index if needed
        if create_index_if_missing and index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=embedding_dimensions, metric="cosine")
            logger.info(f"Created Pinecone index: {index_name}")

        # Connect to index
        self.index = pinecone.Index(index_name)

        # Cache for knowledge units
        self.knowledge_cache = {}

        # Background tasks for batching
        self._batch_queue = []
        self._batch_size = 100
        self._last_batch_time = time.time()
        self._batch_timeout = 5  # seconds

        logger.info(f"PineconeVectorStore initialized with index: {index_name}")

    def add(self, unit: KnowledgeUnit) -> None:
        """
        Add a knowledge unit to the vector store.

        Args:
            unit: Knowledge unit to add
        """
        # Prepare data for Pinecone
        embeddings = (
            unit.embedding.tolist() if hasattr(unit.embedding, "tolist") else unit.embedding
        )
        metadata = {
            "text": unit.original_chunk,
            "source": unit.source or "",
            "created_at": unit.created_at or datetime.now().isoformat(),
            "knowledge_unit": json.dumps(unit.to_dict()),
        }

        # Add custom metadata if present
        if unit.metadata:
            for key, value in unit.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    # Convert complex types to JSON string
                    metadata[f"{key}_json"] = json.dumps(value)

        # Add to queue for batch processing
        self._batch_queue.append({"id": unit.unique_id, "vector": embeddings, "metadata": metadata})

        # Cache the knowledge unit
        self.knowledge_cache[unit.unique_id] = unit

        # Process batch if needed
        self._process_batch_if_needed()

    def _process_batch_if_needed(self, force: bool = False) -> None:
        """
        Process batch if it's full or timed out.

        Args:
            force: Whether to force processing regardless of batch size
        """
        current_time = time.time()
        batch_timeout_reached = (current_time - self._last_batch_time) > self._batch_timeout

        if force or len(self._batch_queue) >= self._batch_size or batch_timeout_reached:
            if not self._batch_queue:
                return

            try:
                # Convert to Pinecone format
                vectors = []

                for item in self._batch_queue:
                    vectors.append(
                        {"id": item["id"], "values": item["vector"], "metadata": item["metadata"]}
                    )

                # Upsert to Pinecone
                self.index.upsert(vectors=vectors, namespace=self.namespace)

                logger.debug(f"Added batch of {len(vectors)} items to Pinecone")

            except Exception as e:
                logger.error(f"Error adding batch to Pinecone: {e}")

            # Reset batch
            self._batch_queue = []
            self._last_batch_time = current_time

    def update(self, unit: KnowledgeUnit) -> None:
        """
        Update a knowledge unit in the store.

        Args:
            unit: Knowledge unit to update
        """
        # For Pinecone, update is the same as add (upsert)
        self.add(unit)

    def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the store.

        Args:
            unique_id: ID of the knowledge unit to delete

        Returns:
            True if deleted, False otherwise
        """
        # Process any pending additions first
        self._process_batch_if_needed(force=True)

        try:
            # Delete from Pinecone
            self.index.delete(ids=[unique_id], namespace=self.namespace)

            # Remove from cache
            if unique_id in self.knowledge_cache:
                del self.knowledge_cache[unique_id]

            return True

        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return False

    def get(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Get a knowledge unit by ID.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        # Check cache first
        if unique_id in self.knowledge_cache:
            return self.knowledge_cache[unique_id]

        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Fetch from Pinecone
            result = self.index.fetch(ids=[unique_id], namespace=self.namespace)

            if not result or unique_id not in result["vectors"]:
                return None

            # Extract metadata
            vector_data = result["vectors"][unique_id]
            metadata = vector_data["metadata"]

            # Create knowledge unit from metadata
            knowledge_unit_dict = json.loads(metadata.get("knowledge_unit", "{}"))
            unit = KnowledgeUnit.from_dict(knowledge_unit_dict)

            # Cache the unit
            self.knowledge_cache[unique_id] = unit

            return unit

        except Exception as e:
            logger.error(f"Error getting from Pinecone: {e}")
            return None

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
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Query Pinecone
            results = self.index.query(
                vector=embedding, top_k=limit, namespace=self.namespace, include_metadata=True
            )

            if not results or not results["matches"]:
                return []

            # Extract results
            units_with_scores = []

            for match in results["matches"]:
                # Skip if below threshold
                if match["score"] < threshold:
                    continue

                unit_id = match["id"]
                metadata = match["metadata"]
                similarity = match["score"]

                # Get or create knowledge unit
                if unit_id in self.knowledge_cache:
                    unit = self.knowledge_cache[unit_id]
                else:
                    knowledge_unit_dict = json.loads(metadata.get("knowledge_unit", "{}"))
                    unit = KnowledgeUnit.from_dict(knowledge_unit_dict)
                    self.knowledge_cache[unit_id] = unit

                units_with_scores.append((unit, similarity))

            return units_with_scores

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []

    def list_all(self) -> list[KnowledgeUnit]:
        """
        List all knowledge units in the store.

        Note: This is inefficient for Pinecone as it doesn't have a native "list all" operation.
        It will only return units that have been cached locally.

        Returns:
            List of cached knowledge units
        """
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        logger.warning(
            "Pinecone doesn't support efficient listing of all vectors. "
            "Only returning cached units."
        )

        return list(self.knowledge_cache.values())

    def clear(self) -> None:
        """Clear all knowledge units from the store."""
        try:
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=self.namespace)

            # Clear cache and batch queue
            self.knowledge_cache = {}
            self._batch_queue = []
            self._last_batch_time = time.time()

            logger.info(f"Cleared Pinecone namespace: {self.namespace}")

        except Exception as e:
            logger.error(f"Error clearing Pinecone: {e}")

    def count(self) -> int:
        """
        Count the number of knowledge units in the store.

        Returns:
            Number of knowledge units (approximate for Pinecone)
        """
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        try:
            # Get stats from Pinecone
            stats = self.index.describe_index_stats()

            # Get count for this namespace
            namespaces = stats.get("namespaces", {})
            namespace_stats = namespaces.get(self.namespace, {})

            return namespace_stats.get("vector_count", 0)

        except Exception as e:
            logger.error(f"Error counting Pinecone: {e}")
            return 0

    def close(self) -> None:
        """Close the vector store and persist any pending changes."""
        # Process any pending additions
        self._process_batch_if_needed(force=True)

        # No explicit close needed for Pinecone
        logger.debug("Closed Pinecone vector store")
