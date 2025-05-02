"""
SQLite implementation of the memory store interfaces.

This module provides a SQLite-based implementation of the memory store
interfaces for persistent storage of knowledge units.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from ltm_agent.core.exceptions import (
    KnowledgeUnitDeleteError,
    KnowledgeUnitUpdateError,
    MemoryStoreError,
)
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)

import hashlib
import random

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite-based implementation of the VectorStore interface.

    This class provides a persistent storage implementation using SQLite.
    """

    def __init__(
        self,
        database_path: str | Path = "memory.db",
        embedding_dim: int = 384,
        create_tables: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the SQLite vector store.

        Args:
            database_path: Path to the SQLite database file
            embedding_dim: Dimension of embedding vectors
            create_tables: Whether to create tables on initialization
            embedding_model_name: Name of the SentenceTransformer model to use
        """
        self.database_path = Path(database_path)
        self.embedding_dim = embedding_dim
        self.create_tables_on_init = create_tables
        self.initialized = False
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None

        # Try to initialize the embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                # Verify dimension matches model
                actual_dim = self.embedding_model.get_sentence_embedding_dimension()
                if actual_dim != embedding_dim:
                    logger.warning(
                        f"Provided embedding_dim {embedding_dim} does not match model {embedding_model_name} dimension {actual_dim}. Using {actual_dim}."
                    )
                    self.embedding_dim = actual_dim
                logger.info(f"Initialized embedding model: {embedding_model_name}")
            except Exception as e:
                logger.error(
                    f"Failed to load embedding model {embedding_model_name}. Error: {e}. Using fallback random embeddings."
                )
                self.embedding_model = None
        else:
            logger.warning("SentenceTransformers not available. Using fallback random embeddings.")

        super().__init__(storage_backend=self.database_path)
        logger.info(f"Initialized SQLiteVectorStore with database at {self.database_path}")

    async def initialize(self) -> None:
        """Initialize the SQLite store and create tables if configured to do so."""
        if self.create_tables_on_init:
            await self._create_tables()
        self.initialized = True
        logger.info(f"SQLiteVectorStore initialized with database at {self.database_path}")

    async def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self.initialized:
            await self.initialize()

    async def _create_tables(self) -> None:
        """Create the database tables if they don't exist."""
        try:
            async with aiosqlite.connect(self.database_path) as conn:
                # Begin transaction for schema creation
                await conn.execute("BEGIN TRANSACTION")

                try:
                    # Enable foreign keys support
                    await conn.execute("PRAGMA foreign_keys = ON")

                    # Create knowledge units table
                    await conn.execute(
                        """
                    CREATE TABLE IF NOT EXISTS knowledge_units (
                        unique_id TEXT PRIMARY KEY,
                        original_chunk TEXT NOT NULL,
                        contextual_text TEXT,
                        knowledge_source TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        tags TEXT
                    )
                    """
                    )

                    # Create embeddings table
                    await conn.execute(
                        """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        unique_id TEXT PRIMARY KEY,
                        embedding BLOB NOT NULL,
                        FOREIGN KEY (unique_id) REFERENCES knowledge_units (unique_id) ON DELETE CASCADE
                    )
                    """
                    )

                    # Create indexes for faster queries
                    await conn.execute(
                        """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_units (knowledge_source)
                    """
                    )

                    await conn.execute(
                        """
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge_units (timestamp)
                    """
                    )

                    await conn.execute(
                        """
                    CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_units (tags)
                    """
                    )

                    # Commit the transaction
                    await conn.commit()
                    logger.info(f"Created database tables in {self.database_path}")
                except Exception as e:
                    # Rollback on error
                    await conn.execute("ROLLBACK")
                    error_msg = f"Error creating database tables: {str(e)}"
                    logger.error(error_msg)
                    raise MemoryStoreError(error_msg) from e
        except Exception as e:
            error_msg = f"Error connecting to SQLite database: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    def _serialize_knowledge_unit(self, knowledge_unit: KnowledgeUnit) -> dict[str, Any]:
        """
        Serialize a knowledge unit for storage in SQLite.

        Args:
            knowledge_unit: The knowledge unit to serialize

        Returns:
            Dictionary of serialized values
        """
        return {
            "unique_id": knowledge_unit.unique_id,
            "original_chunk": knowledge_unit.original_chunk,
            "contextual_text": knowledge_unit.contextual_text,
            "knowledge_source": knowledge_unit.knowledge_source,
            "timestamp": knowledge_unit.timestamp.isoformat(),
            "metadata": json.dumps(knowledge_unit.metadata),
            "tags": ",".join(knowledge_unit.tags) if knowledge_unit.tags else "",
        }

    def _deserialize_knowledge_unit(self, row: dict[str, Any]) -> KnowledgeUnit:
        """
        Deserialize a knowledge unit from SQLite storage.

        Args:
            row: The database row to deserialize

        Returns:
            Deserialized knowledge unit
        """
        # Convert SQLite row to dict if it's not already one
        row_dict = {}
        if hasattr(row, "keys"):
            # This is a sqlite3.Row object
            try:
                # Safely extract keys that exist in the row
                for key in row:
                    try:
                        row_dict[key] = row[key]
                    except (IndexError, KeyError):
                        # Skip keys that can't be accessed
                        continue
            except Exception as e:
                logger.warning(f"Error extracting keys from row: {e}")
        else:
            # Already a dict
            row_dict = row

        # Handle potential missing fields in the row (important for JOIN results)
        unique_id = row_dict.get("unique_id", "")
        original_chunk = row_dict.get("original_chunk", "")
        contextual_text = row_dict.get("contextual_text", "")
        knowledge_source = row_dict.get(
            "knowledge_source", "corpus"
        )  # Default to corpus if missing

        # Handle timestamp - parse from ISO format or use current time if missing
        timestamp_str = row_dict.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Handle metadata - parse JSON or use empty dict if missing/invalid
        metadata_str = row_dict.get("metadata")
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = {}

        # Handle tags - split by comma or use empty list if missing/empty
        tags_str = row_dict.get("tags")
        if tags_str and isinstance(tags_str, str):
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        else:
            tags = []

        return KnowledgeUnit(
            unique_id=unique_id,
            original_chunk=original_chunk,
            contextual_text=contextual_text,
            knowledge_source=knowledge_source,
            timestamp=timestamp,
            metadata=metadata,
            tags=tags,
        )

    def _serialize_embedding(self, embedding: list[float] | None) -> bytes | None:
        """
        Serialize an embedding vector to bytes for storage.

        Args:
            embedding: The embedding vector to serialize

        Returns:
            Serialized embedding as bytes or None if the input is None
        """
        if embedding is None:
            return None

        try:
            # Convert list of floats to bytes using pickle
            import pickle

            return pickle.dumps(embedding)
        except Exception as e:
            logger.warning(f"Error serializing embedding: {str(e)}")
            return None

    def _deserialize_embedding(self, embedding_bytes: bytes | None) -> list[float] | None:
        """
        Deserialize an embedding vector from bytes.

        Args:
            embedding_bytes: The serialized embedding as bytes

        Returns:
            Deserialized embedding vector or None if the input is None or invalid
        """
        if embedding_bytes is None:
            return None

        try:
            # Convert bytes back to list of floats using pickle
            import pickle

            embedding = pickle.loads(embedding_bytes)

            # Convert any NumPy arrays to Python list to avoid truth value ambiguity
            if hasattr(embedding, "__array__"):  # Check if it's a NumPy array
                embedding = embedding.tolist()

            return embedding
        except Exception as e:
            logger.warning(f"Error deserializing embedding: {str(e)}")
            return None

    def _normalize_vector(self, vector: list[float] | None) -> np.ndarray | None:
        """
        Normalize a vector to unit length.

        Args:
            vector: The vector to normalize

        Returns:
            np.ndarray: The normalized vector, or None if normalization fails
        """
        if vector is None:
            return None

        try:
            np_vector = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(np_vector)
            if norm > 0:
                return np_vector / norm
            else:
                # Handle zero vector case
                logger.warning("Attempted to normalize a zero vector.")
                return None  # Or return zero vector if appropriate
        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            return None

    def _build_where_clause(self, filter_criteria: dict[str, Any] | None) -> tuple[str, list]:
        """Builds the WHERE clause and parameters for filtering.

        Args:
            filter_criteria: Dictionary of criteria to filter by

        Returns:
            Tuple of (where_clause, params)
        """
        if not filter_criteria:
            return "", []

        conditions = []
        params = []
        for key, value in filter_criteria.items():
            if key == "knowledge_source":
                conditions.append("ku.knowledge_source = ?")
                params.append(value)
            elif key == "tags":
                if isinstance(value, list):
                    tag_conditions = []
                    for tag in value:
                        tag_conditions.append("ku.tags LIKE ?")
                        params.append(f"%{tag}%")
                    conditions.append(f"({' OR '.join(tag_conditions)})")
                else:
                    conditions.append("ku.tags LIKE ?")
                    params.append(f"%{value}%")
            # Add other filterable fields here if needed
            else:
                logger.warning(f"Unsupported filter key: {key}")

        if not conditions:
            return "", []

        return f"WHERE {' AND '.join(conditions)}", params

    async def _add_implementation(self, knowledge_unit: KnowledgeUnit) -> str:
        """
        Add a knowledge unit to the SQLite store.

        Args:
            knowledge_unit: The knowledge unit to store

        Returns:
            The unique ID of the stored knowledge unit

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()
        unique_id = knowledge_unit.unique_id
        logger.debug(f"Attempting to add unit {unique_id} to SQLite.")

        # Generate embedding if not present (ensure this doesn't raise unhandled exceptions)
        if knowledge_unit.embedding_vector is None:
            try:
                text_to_embed = knowledge_unit.original_chunk
                if knowledge_unit.contextual_text:
                    text_to_embed = f"{text_to_embed}\n{knowledge_unit.contextual_text}"
                knowledge_unit.embedding_vector = await self.generate_embedding(text_to_embed)
                logger.debug(f"Generated embedding for unit {unique_id}")
            except Exception as e:
                logger.error(
                    f"Failed to generate embedding for {unique_id}: {str(e)}. Unit will be added without embedding.",
                    exc_info=True,
                )
                knowledge_unit.embedding_vector = None  # Ensure it's None if generation failed

        serialized_unit = self._serialize_knowledge_unit(knowledge_unit)
        serialized_embedding = self._serialize_embedding(knowledge_unit.embedding_vector)

        try:
            async with aiosqlite.connect(self.database_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    # Insert into knowledge_units table
                    await conn.execute(
                        """
                        INSERT INTO knowledge_units (
                            unique_id, original_chunk, contextual_text,
                            knowledge_source, timestamp, metadata, tags
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            unique_id,
                            serialized_unit["original_chunk"],
                            serialized_unit["contextual_text"],
                            serialized_unit["knowledge_source"],
                            serialized_unit["timestamp"],
                            serialized_unit["metadata"],
                            serialized_unit["tags"],
                        ),
                    )
                    logger.debug(f"Inserted unit data for {unique_id}")

                    # Insert into embeddings table ONLY if embedding exists
                    if serialized_embedding is not None:
                        await conn.execute(
                            """
                            INSERT INTO embeddings (unique_id, embedding)
                            VALUES (?, ?)
                            """,
                            (unique_id, serialized_embedding),
                        )
                        logger.debug(f"Inserted embedding for {unique_id}")
                    else:
                        logger.warning(f"No embedding to insert for unit {unique_id}")

                    await conn.commit()
                    logger.info(f"Successfully added unit {unique_id} to SQLite.")
                    return unique_id
                except Exception as e:
                    await conn.execute("ROLLBACK")
                    error_msg = f"Error during transaction for adding unit {unique_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise MemoryStoreError(error_msg) from e
        except Exception as e:
            error_msg = f"Error connecting/adding to SQLite for unit {unique_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryStoreError(error_msg) from e

    async def _get_implementation(self, unique_id: str) -> KnowledgeUnit | None:
        """
        Retrieve a knowledge unit from the SQLite store.

        Args:
            unique_id: The unique ID of the knowledge unit

        Returns:
            The retrieved knowledge unit or None if not found

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()
        logger.debug(f"Attempting to get unit {unique_id} from SQLite.")
        try:
            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row
                query = """
                SELECT ku.*, e.embedding
                FROM knowledge_units ku
                LEFT JOIN embeddings e ON ku.unique_id = e.unique_id
                WHERE ku.unique_id = ?
                """
                async with conn.execute(query, (unique_id,)) as cursor:
                    row = await cursor.fetchone()

                    if row is None:
                        logger.warning(f"Unit {unique_id} not found in SQLite.")
                        return None

                    try:
                        # Deserialize the knowledge unit
                        knowledge_unit = self._deserialize_knowledge_unit(row)

                        # Add embedding if present
                        if "embedding" in row and row["embedding"] is not None:
                            knowledge_unit.embedding_vector = self._deserialize_embedding(
                                row["embedding"]
                            )
                        else:
                            knowledge_unit.embedding_vector = (
                                None  # Explicitly set to None if not found
                            )

                        logger.debug(f"Successfully retrieved and deserialized unit {unique_id}.")
                        return knowledge_unit
                    except Exception as deser_e:
                        logger.error(
                            f"Failed to deserialize unit {unique_id} from DB row. Error: {deser_e}",
                            exc_info=True,
                        )
                        return None  # Treat deserialization failure as not found

        except Exception as e:
            # Log the error but return None to indicate not found
            error_msg = f"Error retrieving unit {unique_id} from SQLite: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None  # Return None to indicate not found or error during retrieval

    async def _update_implementation(self, knowledge_unit: KnowledgeUnit) -> bool:
        """
        Update a knowledge unit in the SQLite store.

        Args:
            knowledge_unit: The updated knowledge unit

        Returns:
            True if the update was successful, False if the unit wasn't found

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()

        # Serialize the knowledge unit
        serialized_unit = self._serialize_knowledge_unit(knowledge_unit)

        try:
            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Begin transaction
                await conn.execute("BEGIN TRANSACTION")

                try:
                    # Check if the knowledge unit exists
                    async with conn.execute(
                        "SELECT COUNT(*) as count FROM knowledge_units WHERE unique_id = ?",
                        (serialized_unit["unique_id"],),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row["count"] == 0:
                            # Knowledge unit not found
                            await conn.execute("ROLLBACK")
                            return False

                    # Update the knowledge unit
                    await conn.execute(
                        """
                        UPDATE knowledge_units
                        SET original_chunk = ?,
                            contextual_text = ?,
                            knowledge_source = ?,
                            timestamp = ?,
                            metadata = ?,
                            tags = ?
                        WHERE unique_id = ?
                        """,
                        (
                            serialized_unit["original_chunk"],
                            serialized_unit["contextual_text"],
                            serialized_unit["knowledge_source"],
                            serialized_unit["timestamp"],
                            serialized_unit["metadata"],
                            serialized_unit["tags"],
                            serialized_unit["unique_id"],
                        ),
                    )

                    # Handle embedding update if present
                    if knowledge_unit.embedding_vector is not None:
                        # Check if embedding exists
                        async with conn.execute(
                            "SELECT COUNT(*) as count FROM embeddings WHERE unique_id = ?",
                            (serialized_unit["unique_id"],),
                        ) as cursor:
                            row = await cursor.fetchone()

                            if row["count"] > 0:
                                # Update existing embedding
                                await conn.execute(
                                    """
                                    UPDATE embeddings
                                    SET embedding = ?
                                    WHERE unique_id = ?
                                    """,
                                    (
                                        self._serialize_embedding(knowledge_unit.embedding_vector),
                                        serialized_unit["unique_id"],
                                    ),
                                )
                            else:
                                # Insert new embedding
                                await conn.execute(
                                    """
                                    INSERT INTO embeddings (unique_id, embedding)
                                    VALUES (?, ?)
                                    """,
                                    (
                                        serialized_unit["unique_id"],
                                        self._serialize_embedding(knowledge_unit.embedding_vector),
                                    ),
                                )

                    # Commit the transaction
                    await conn.commit()
                    logger.info(
                        f"Updated knowledge unit in SQLite store: {serialized_unit['unique_id']}"
                    )

                    return True
                except Exception as e:
                    # Rollback on error
                    await conn.execute("ROLLBACK")
                    error_msg = f"Error updating knowledge unit in SQLite store: {str(e)}"
                    logger.error(error_msg)
                    raise KnowledgeUnitUpdateError(error_msg) from e
        except Exception as e:
            error_msg = f"Error connecting to SQLite database: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _delete_implementation(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the SQLite store.

        Args:
            unique_id: The unique ID of the knowledge unit to delete

        Returns:
            True if the deletion was successful, False if the unit wasn't found

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Begin transaction
                await conn.execute("BEGIN TRANSACTION")

                try:
                    # Check if the knowledge unit exists
                    async with conn.execute(
                        "SELECT COUNT(*) as count FROM knowledge_units WHERE unique_id = ?",
                        (unique_id,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row["count"] == 0:
                            # Knowledge unit not found
                            await conn.execute("ROLLBACK")
                            return False

                    # Delete from knowledge_units (will cascade to embeddings)
                    await conn.execute(
                        "DELETE FROM knowledge_units WHERE unique_id = ?", (unique_id,)
                    )

                    # Commit the transaction
                    await conn.commit()
                    logger.info(f"Deleted knowledge unit from SQLite store: {unique_id}")

                    return True
                except Exception as e:
                    # Rollback on error
                    await conn.execute("ROLLBACK")
                    error_msg = f"Error deleting knowledge unit from SQLite store: {str(e)}"
                    logger.error(error_msg)
                    raise KnowledgeUnitDeleteError(error_msg) from e
        except Exception as e:
            error_msg = f"Error connecting to SQLite database: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _search_implementation(
        self, query: str, limit: int = 10, filter_criteria: dict[str, Any] | None = None
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for knowledge units matching the query in the SQLite store.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter results

        Returns:
            List of tuples of (knowledge unit, relevance score)

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()
        logger.debug(f"Attempting search in SQLite (limit={limit}).")
        try:
            # Fallback to simple keyword search if query is empty
            if not query or query.strip() == "":
                return await self._list_implementation(limit=limit, filter_criteria=filter_criteria)

            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Try searching with simple LIKE query instead of FTS for now
                # This avoids complex FTS syntax issues
                simple_query = """
                SELECT * FROM knowledge_units
                WHERE original_chunk LIKE ? OR contextual_text LIKE ?
                LIMIT ?
                """

                search_term = f"%{query}%"
                params = [search_term, search_term, limit]

                logger.info(
                    f"Executing simplified search query: {simple_query} with params {params}"
                )

                results = []
                try:
                    async with conn.execute(simple_query, params) as cursor:
                        async for row in cursor:
                            # Deserialize the knowledge unit
                            knowledge_unit = self._deserialize_knowledge_unit(row)

                            # Get embedding if present
                            embedding_query = """
                            SELECT embedding FROM embeddings WHERE unique_id = ?
                            """
                            async with conn.execute(
                                embedding_query, (knowledge_unit.unique_id,)
                            ) as embedding_cursor:
                                embedding_row = await embedding_cursor.fetchone()

                                if embedding_row and embedding_row["embedding"] is not None:
                                    knowledge_unit.embedding_vector = self._deserialize_embedding(
                                        embedding_row["embedding"]
                                    )

                            # Simple relevance score based on position in results
                            score = 1.0 - (len(results) / (limit + 1))

                            results.append((knowledge_unit, score))

                    return results

                except Exception as inner_e:
                    # Log the specific error with the query
                    logger.error(f"Error executing search query: {str(inner_e)}")
                    logger.error(f"Query was: {simple_query} with params {params}")
                    raise

        except Exception as e:
            error_msg = f"Error searching knowledge units in SQLite store: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _list_implementation(
        self,
        limit: int | None = None,
        offset: int = 0,
        filter_criteria: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> list[KnowledgeUnit]:
        """
        List knowledge units, optionally filtered by criteria.

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            filter_criteria: Dictionary of field names and values to filter by
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            List of knowledge units matching the criteria

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()

        try:
            query_parts = ["SELECT * FROM knowledge_units"]
            params = []

            # Add WHERE clauses for filtering
            if filter_criteria:
                where_clause, params = self._build_where_clause(filter_criteria)
                query_parts.append(where_clause)

            # Add ORDER BY for consistent ordering
            if sort_by:
                query_parts.append(f"ORDER BY knowledge_units.{sort_by} {sort_order.upper()}")
            else:
                query_parts.append("ORDER BY knowledge_units.timestamp DESC")

            # Add LIMIT and OFFSET
            if limit is not None:
                query_parts.append("LIMIT ?")
                params.append(limit)

            query_parts.append("OFFSET ?")
            params.append(offset)

            # Combine into final query
            query = " ".join(query_parts)

            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Execute the query
                async with conn.execute(query, params) as cursor:
                    results = []

                    async for row in cursor:
                        # Deserialize the knowledge unit
                        knowledge_unit = self._deserialize_knowledge_unit(row)

                        # Fetch the embedding if available
                        embedding_query = """
                        SELECT embedding FROM embeddings WHERE unique_id = ?
                        """
                        async with conn.execute(
                            embedding_query, (knowledge_unit.unique_id,)
                        ) as embedding_cursor:
                            embedding_row = await embedding_cursor.fetchone()

                            if embedding_row and embedding_row["embedding"] is not None:
                                knowledge_unit.embedding_vector = self._deserialize_embedding(
                                    embedding_row["embedding"]
                                )

                        results.append(knowledge_unit)

                    return results
        except Exception as e:
            error_msg = f"Error listing knowledge units from SQLite store: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _count_implementation(self, filter_criteria: dict[str, Any] | None = None) -> int:
        """
        Count knowledge units, optionally filtered by criteria.

        Args:
            filter_criteria: Dictionary of field names and values to filter by

        Returns:
            Number of knowledge units matching the criteria

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()

        try:
            query_parts = ["SELECT COUNT(*) as count FROM knowledge_units"]
            params = []

            # Add WHERE clauses for filtering
            if filter_criteria:
                where_clause, params = self._build_where_clause(filter_criteria)
                query_parts.append(where_clause)

            # Combine into final query
            query = " ".join(query_parts)

            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Execute the query
                async with conn.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    return row["count"]
        except Exception as e:
            error_msg = f"Error counting knowledge units in SQLite store: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _similarity_search_implementation(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Search for knowledge units by embedding similarity.

        Args:
            embedding: The query embedding
            limit: The maximum number of results to return
            filter_criteria: Additional criteria to filter the results

        Returns:
            A list of (knowledge_unit, similarity) tuples

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()
        logger.debug(f"Attempting similarity search in SQLite (limit={limit}).")
        try:
            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # First, retrieve all embedding vectors with filter to calculate similarity
                where_clause, params = self._build_where_clause(filter_criteria)
                embeddings_query = f"""
                SELECT ku.unique_id, e.embedding
                FROM knowledge_units ku
                JOIN embeddings e ON ku.unique_id = e.unique_id
                {where_clause}
                """
                logger.debug(
                    f"Executing query to fetch embeddings: {embeddings_query} with params {params}"
                )

                all_embeddings = {}
                async with conn.execute(embeddings_query, params) as cursor:
                    async for row in cursor:
                        try:
                            unit_id = row["unique_id"]
                            stored_embedding = self._deserialize_embedding(row["embedding"])
                            if stored_embedding is not None:
                                all_embeddings[unit_id] = stored_embedding
                            else:
                                logger.warning(
                                    f"Skipping unit {unit_id} due to missing/invalid embedding."
                                )
                        except Exception as deser_e:
                            logger.error(
                                f"Failed to deserialize embedding for unit {row['unique_id']}: {deser_e}",
                                exc_info=True,
                            )
                            continue

                logger.debug(
                    f"Retrieved {len(all_embeddings)} embeddings for similarity comparison."
                )
                if not all_embeddings:
                    return []

                # Normalize query vector
                query_vector = self._normalize_vector(embedding)
                if query_vector is None:
                    logger.error("Query vector normalization failed.")
                    return []

                # Calculate similarities and sort
                results_with_scores = []
                for unit_id, stored_embedding in all_embeddings.items():
                    try:
                        stored_vector = self._normalize_vector(stored_embedding)
                        if stored_vector is not None:
                            # Check if dimensions match before attempting dot product
                            if len(query_vector) != len(stored_vector):
                                logger.warning(
                                    f"Dimension mismatch: query_vector ({len(query_vector)}) != "
                                    f"stored_vector ({len(stored_vector)}) for unit {unit_id}. Skipping."
                                )
                                continue

                            # Calculate cosine similarity
                            similarity = float(np.dot(query_vector, stored_vector))
                            # Ensure similarity is in [0, 1] range
                            similarity = max(0.0, min(1.0, similarity))
                            results_with_scores.append((unit_id, similarity))
                        else:
                            logger.warning(f"Could not normalize stored vector for unit {unit_id}")
                    except Exception as e:
                        logger.error(f"Error processing embedding for unit {unit_id}: {e}")
                        continue

                # Sort by similarity (descending)
                results_with_scores.sort(key=lambda x: x[1], reverse=True)
                logger.debug(f"Calculated similarities for {len(results_with_scores)} units.")

                # Only fetch the full knowledge units for the top results
                top_results = []
                for unit_id, score in results_with_scores[:limit]:
                    # Fetch the full knowledge unit
                    unit = await self._get_implementation(unit_id)
                    if unit:
                        # For debugging, log the content of high scoring units
                        if score > 0.5:
                            logger.debug(
                                f"High similarity unit (score={score:.4f}): {unit.original_chunk[:50]}..."
                            )
                        top_results.append((unit, score))
                    else:
                        logger.warning(
                            f"Could not retrieve full unit data for {unit_id} during similarity search."
                        )

                logger.debug(
                    f"Similarity search yielded {len(top_results)} results after limiting to top {limit}."
                )
                return top_results

        except Exception as e:
            error_msg = f"Error performing similarity search in SQLite store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryStoreError(error_msg) from e

    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """
        Generate an embedding for the text.

        Args:
            text: The text to generate an embedding for

        Returns:
            list[float]: The embedding

        Raises:
            EmbeddingError: If the embedding generation fails
        """
        await self._ensure_initialized()

        # Use real embeddings if model is available
        if self.embedding_model is not None:
            try:
                # Generate embedding using SentenceTransformer
                embedding = self.embedding_model.encode(text)
                # Convert to list of floats
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding with SentenceTransformer: {e}")
                # Fall back to random embedding if model fails

        # Fall back to deterministic random embedding
        logger.debug(f"Using fallback random embedding for text: {text[:50]}...")
        hash_object = hashlib.md5(text.encode("utf-8"))
        text_hash = int(hash_object.hexdigest(), 16)
        random.seed(text_hash)
        embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

        # Normalize the embedding
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding = (embedding_array / norm).tolist()

        # Reset random seed
        random.seed(None)

        return embedding
