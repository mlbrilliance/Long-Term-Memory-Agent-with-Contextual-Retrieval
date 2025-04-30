"""
Vector store implementation using Langchain and FAISS.

This module provides a concrete implementation of the VectorStore interface
using Langchain and FAISS as the underlying storage mechanism for vector embeddings.
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Any

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from ltm_agent.core.config import Settings, get_settings
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)


class LangchainVectorStore(BaseVectorStore):
    """
    Vector store implementation using Langchain and FAISS.

    This class provides a concrete implementation of the VectorStore interface
    using Langchain's FAISS integration for storing and retrieving vector embeddings.
    """

    def __init__(self, config: Settings | None = None, persist_directory: str | None = None):
        """
        Initialize the Langchain vector store.

        Args:
            config: Optional Settings object for configuration
            persist_directory: Directory for persisting the vector store
        """
        self.config = config or get_settings()
        self.persist_directory = persist_directory or self.config.vector_db_path

        # Initialize embedding model
        model_name = self.config.embedding_model_name

        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=os.path.join(os.path.dirname(self.persist_directory), "model_cache"),
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2: {e}")
            # Fall back to a more reliable model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # We'll initialize the FAISS index in the initialize method
        self._faiss = None
        self.initialized = False

        # In-memory storage for metadata (id -> knowledge unit mapping)
        self.metadata_store = {}

        logger.info(f"Initialized LangchainVectorStore with model: {model_name}")
        super().__init__()

    async def initialize(self) -> None:
        """Initialize the Langchain FAISS vector store."""
        try:
            # Try to load existing index if available
            if self.persist_directory and os.path.exists(
                os.path.join(self.persist_directory, "index.faiss")
            ):
                self._faiss = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
                # Load metadata from disk
                metadata_path = os.path.join(self.persist_directory, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        self.metadata_store = json.load(f)

                logger.info(f"Loaded existing FAISS index from {self.persist_directory}")
            else:
                # Create a new empty vector store
                self._faiss = FAISS.from_documents(
                    [Document(page_content="placeholder", metadata={"placeholder": True})],
                    self.embedding_model,
                )

                # Remove the placeholder document
                if self._faiss._index is not None and len(self._faiss.index_to_docstore_id) > 0:
                    placeholder_id = list(self._faiss.index_to_docstore_id.values())[0]
                    self._faiss.delete([placeholder_id])

                # Create persist directory if needed
                if self.persist_directory and not os.path.exists(self.persist_directory):
                    os.makedirs(self.persist_directory, exist_ok=True)

                logger.info("Created new FAISS index")

                # Save the initial state
                if self.persist_directory:
                    self._faiss.save_local(self.persist_directory)

            self.initialized = True

        except Exception as e:
            logger.exception(f"Error initializing FAISS: {e}")
            # Create a minimal in-memory store as fallback
            self._faiss = FAISS.from_documents(
                [Document(page_content="placeholder", metadata={"placeholder": True})],
                self.embedding_model,
            )
            self.initialized = True

    async def _save_metadata(self):
        """Save metadata to disk."""
        if not self.persist_directory:
            return

        # Serialize metadata
        serialized_metadata = {}
        for unit_id, unit_data in self.metadata_store.items():
            if isinstance(unit_data, dict):
                serialized_metadata[unit_id] = unit_data
            else:
                # Convert KnowledgeUnit to dict if needed
                serialized_metadata[unit_id] = unit_data.model_dump()

                # Convert timestamp to string
                if "timestamp" in serialized_metadata[unit_id]:
                    timestamp = serialized_metadata[unit_id]["timestamp"]
                    if isinstance(timestamp, datetime):
                        serialized_metadata[unit_id]["timestamp"] = timestamp.isoformat()

        # Save to file
        metadata_path = os.path.join(self.persist_directory, "metadata.json")
        temp_path = f"{metadata_path}.tmp"

        try:
            with open(temp_path, "w") as f:
                json.dump(serialized_metadata, f)

            # Atomic replace
            shutil.move(temp_path, metadata_path)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """Add a knowledge unit to the vector store."""
        if not self.initialized:
            await self.initialize()

        try:
            # Ensure the unit has a unique ID
            if not knowledge_unit.unique_id:
                knowledge_unit.unique_id = str(uuid.uuid4())

            # Add content to FAISS
            texts = [knowledge_unit.content]
            metadatas = [
                {
                    "id": knowledge_unit.unique_id,
                    "source": knowledge_unit.knowledge_source,
                    "timestamp": knowledge_unit.timestamp.isoformat(),
                }
            ]

            # Add to FAISS
            ids = self._faiss.add_texts(texts, metadatas)

            # Store full KnowledgeUnit in metadata store
            self.metadata_store[knowledge_unit.unique_id] = knowledge_unit

            # Save changes
            if self.persist_directory:
                self._faiss.save_local(self.persist_directory)
                await self._save_metadata()

            return knowledge_unit.unique_id

        except Exception as e:
            logger.exception(f"Error adding knowledge unit: {e}")
            return ""

    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """Retrieve a knowledge unit by ID."""
        if not self.initialized:
            await self.initialize()

        # Check metadata store first
        if unique_id in self.metadata_store:
            return self.metadata_store[unique_id]

        return None

    async def _delete_implementation(self, unique_id: str) -> bool:
        """Delete a knowledge unit from the vector store."""
        if not self.initialized:
            await self.initialize()

        try:
            # Get the langchain document ID if it exists
            doc_ids = []
            for langchain_id, docstore_id in self._faiss.index_to_docstore_id.items():
                doc = self._faiss.docstore.search(docstore_id)
                if doc.metadata.get("id") == unique_id:
                    doc_ids.append(docstore_id)

            if doc_ids:
                # Delete from FAISS
                self._faiss.delete(doc_ids)

                # Delete from metadata store
                if unique_id in self.metadata_store:
                    del self.metadata_store[unique_id]

                # Save changes
                if self.persist_directory:
                    self._faiss.save_local(self.persist_directory)
                    await self._save_metadata()

                return True

            return False

        except Exception as e:
            logger.exception(f"Error deleting knowledge unit: {e}")
            return False

    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """Update a knowledge unit in the vector store."""
        if not self.initialized:
            await self.initialize()

        try:
            # Delete existing unit
            deleted = await self._delete_implementation(knowledge_unit.unique_id)

            # Add updated unit
            await self._add_implementation(knowledge_unit)

            return True

        except Exception as e:
            logger.exception(f"Error updating knowledge unit: {e}")
            return False

    async def _search_implementation(
        self, query: str, limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units based on text query."""
        if not self.initialized:
            await self.initialize()

        try:
            # Convert filter criteria to Langchain format
            filter_dict = {}
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "knowledge_source":
                        filter_dict["source"] = value

            # Perform similarity search with FAISS
            results = self._faiss.similarity_search_with_score(
                query, k=limit, filter=filter_dict if filter_dict else None
            )

            # Process results
            knowledge_units = []
            for doc, score in results:
                unit_id = doc.metadata.get("id")
                if unit_id and unit_id in self.metadata_store:
                    # Get knowledge unit from metadata store
                    unit = self.metadata_store[unit_id]

                    # Convert score to similarity score (FAISS returns distance)
                    # Lower distance means higher similarity
                    similarity = 1.0 - min(1.0, score / 10.0)  # Normalize to [0, 1]

                    knowledge_units.append((unit, similarity))

            return knowledge_units

        except Exception as e:
            logger.exception(f"Error searching knowledge units: {e}")
            return []

    async def _similarity_search_implementation(
        self, embedding: list[float], limit: int, filter_criteria: dict[str, Any] | None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Search for knowledge units based on embedding similarity."""
        if not self.initialized:
            await self.initialize()

        try:
            # Convert filter criteria to Langchain format
            filter_dict = {}
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "knowledge_source":
                        filter_dict["source"] = value

            # Perform similarity search with FAISS
            results = self._faiss.similarity_search_by_vector_with_score(
                embedding, k=limit, filter=filter_dict if filter_dict else None
            )

            # Process results
            knowledge_units = []
            for doc, score in results:
                unit_id = doc.metadata.get("id")
                if unit_id and unit_id in self.metadata_store:
                    # Get knowledge unit from metadata store
                    unit = self.metadata_store[unit_id]

                    # Convert score to similarity score (FAISS returns distance)
                    similarity = 1.0 - min(1.0, score / 10.0)  # Normalize to [0, 1]

                    knowledge_units.append((unit, similarity))

            return knowledge_units

        except Exception as e:
            logger.exception(f"Error in similarity search: {e}")
            return []

    async def _list_implementation(
        self,
        limit: int,
        offset: int,
        filter_criteria: dict[str, Any] | None,
        sort_by: str | None,
        sort_order: str,
    ) -> list[KnowledgeUnit]:
        """List knowledge units from the vector store."""
        if not self.initialized:
            await self.initialize()

        try:
            # Get all units from metadata store
            all_units = list(self.metadata_store.values())

            # Apply filters
            if filter_criteria:
                filtered_units = []
                for unit in all_units:
                    matches = True
                    for key, value in filter_criteria.items():
                        if not hasattr(unit, key) or getattr(unit, key) != value:
                            matches = False
                            break
                    if matches:
                        filtered_units.append(unit)
                all_units = filtered_units

            # Apply sorting
            if sort_by and hasattr(all_units[0], sort_by) if all_units else False:
                reverse = sort_order.lower() == "desc"
                all_units.sort(key=lambda unit: getattr(unit, sort_by), reverse=reverse)

            # Apply pagination
            return all_units[offset : offset + limit]

        except Exception as e:
            logger.exception(f"Error listing knowledge units: {e}")
            return []

    async def _count_implementation(self, filter_criteria: dict[str, Any] | None) -> int:
        """Count knowledge units in the vector store."""
        if not self.initialized:
            await self.initialize()

        try:
            # Get all units from metadata store
            all_units = list(self.metadata_store.values())

            # Apply filters
            if filter_criteria:
                filtered_units = []
                for unit in all_units:
                    matches = True
                    for key, value in filter_criteria.items():
                        if not hasattr(unit, key) or getattr(unit, key) != value:
                            matches = False
                            break
                    if matches:
                        filtered_units.append(unit)
                all_units = filtered_units

            return len(all_units)

        except Exception as e:
            logger.exception(f"Error counting knowledge units: {e}")
            return 0

    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """Generate an embedding using the Langchain embedding model."""
        if not self.initialized:
            await self.initialize()

        try:
            # Generate embedding
            embedding = self.embedding_model.embed_query(text)

            return embedding

        except Exception as e:
            logger.exception(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 384  # Common embedding dimension
