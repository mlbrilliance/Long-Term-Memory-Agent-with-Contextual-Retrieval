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

from ltm_agent.core.exceptions import (
    KnowledgeUnitDeleteError,
    KnowledgeUnitNotFoundError,
    KnowledgeUnitUpdateError,
    MemoryStoreError,
)
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite-based implementation of the VectorStore interface.

    This class provides a persistent storage implementation using SQLite.
    """

    def __init__(
        self,
        database_path: str | Path = "memory.db",
        embedding_dim: int = 768,
        create_tables: bool = True,
    ):
        """
        Initialize the SQLite vector store.

        Args:
            database_path: Path to the SQLite database file
            embedding_dim: Dimension of embedding vectors
            create_tables: Whether to create tables on initialization
        """
        self.database_path = Path(database_path)
        self.embedding_dim = embedding_dim
        self.create_tables_on_init = create_tables
        self.initialized = False
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
        if hasattr(row, "keys"):
            # This is a sqlite3.Row object
            row_dict = {key: row[key] for key in row}
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

            return pickle.loads(embedding_bytes)
        except Exception as e:
            logger.warning(f"Error deserializing embedding: {str(e)}")
            return None

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

        # Generate embedding if not present
        if knowledge_unit.embedding_vector is None:
            try:
                # Generate embedding from the content
                text_to_embed = knowledge_unit.original_chunk
                if knowledge_unit.contextual_text:
                    text_to_embed = f"{text_to_embed}\n{knowledge_unit.contextual_text}"

                knowledge_unit.embedding_vector = await self.generate_embedding(text_to_embed)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}. Creating a random one.")
                # Generate a random embedding as fallback
                import random

                knowledge_unit.embedding_vector = [
                    random.uniform(-1, 1) for _ in range(self.embedding_dim)
                ]

        # Serialize the knowledge unit
        serialized_unit = self._serialize_knowledge_unit(knowledge_unit)

        try:
            async with aiosqlite.connect(self.database_path) as conn:
                # Begin a transaction
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
                            serialized_unit["unique_id"],
                            serialized_unit["original_chunk"],
                            serialized_unit["contextual_text"],
                            serialized_unit["knowledge_source"],
                            serialized_unit["timestamp"],
                            serialized_unit["metadata"],
                            serialized_unit["tags"],
                        ),
                    )

                    # Handle embedding if present
                    if knowledge_unit.embedding_vector is not None:
                        # Serialize the embedding
                        serialized_embedding = self._serialize_embedding(
                            knowledge_unit.embedding_vector
                        )

                        # Insert into embeddings table
                        await conn.execute(
                            """
                            INSERT INTO embeddings (unique_id, embedding)
                            VALUES (?, ?)
                            """,
                            (serialized_unit["unique_id"], serialized_embedding),
                        )

                    # Commit the transaction
                    await conn.commit()

                    return serialized_unit["unique_id"]
                except Exception as e:
                    # Rollback on error
                    await conn.execute("ROLLBACK")
                    error_msg = f"Error adding knowledge unit to SQLite store: {str(e)}"
                    logger.error(error_msg)
                    raise MemoryStoreError(error_msg) from e
        except Exception as e:
            error_msg = f"Error connecting to SQLite database: {str(e)}"
            logger.error(error_msg)
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

        try:
            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Query for the knowledge unit
                query = """
                SELECT knowledge_units.*, embeddings.embedding
                FROM knowledge_units
                LEFT JOIN embeddings ON knowledge_units.unique_id = embeddings.unique_id
                WHERE knowledge_units.unique_id = ?
                """

                async with conn.execute(query, (unique_id,)) as cursor:
                    row = await cursor.fetchone()

                    if row is None:
                        return None

                    # Deserialize the knowledge unit
                    knowledge_unit = self._deserialize_knowledge_unit(row)

                    # Add embedding if present in the result
                    if "embedding" in row and row["embedding"] is not None:
                        knowledge_unit.embedding_vector = self._deserialize_embedding(
                            row["embedding"]
                        )

                    return knowledge_unit
        except Exception as e:
            error_msg = f"Error retrieving knowledge unit from SQLite store: {str(e)}"
            logger.error(error_msg)
            raise KnowledgeUnitNotFoundError(error_msg) from e

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

                # Begin a transaction
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
                            embedding_query = "SELECT embedding FROM embeddings WHERE unique_id = ?"
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
                where_clauses = []

                for key, value in filter_criteria.items():
                    if key == "knowledge_source":
                        where_clauses.append("knowledge_units.knowledge_source = ?")
                        params.append(value)
                    elif key == "tags":
                        # Handle tags as a special case
                        if isinstance(value, list) and value:
                            # For each tag, check if it's in the comma-separated list
                            tag_clauses = []
                            for tag in value:
                                tag_clauses.append("knowledge_units.tags LIKE ?")
                                params.append(f"%{tag}%")

                            # Combine tag clauses with OR
                            where_clauses.append(f"({' OR '.join(tag_clauses)})")
                        elif isinstance(value, str) and value:
                            where_clauses.append("knowledge_units.tags LIKE ?")
                            params.append(f"%{value}%")
                    # Add more filters as needed

                if where_clauses:
                    query_parts.append("WHERE " + " AND ".join(where_clauses))

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
                where_clauses = []

                for key, value in filter_criteria.items():
                    if key == "knowledge_source":
                        where_clauses.append("knowledge_units.knowledge_source = ?")
                        params.append(value)
                    elif key == "tags":
                        # Handle tags as a special case
                        if isinstance(value, list) and value:
                            # For each tag, check if it's in the comma-separated list
                            tag_clauses = []
                            for tag in value:
                                tag_clauses.append("knowledge_units.tags LIKE ?")
                                params.append(f"%{tag}%")

                            # Combine tag clauses with OR
                            where_clauses.append(f"({' OR '.join(tag_clauses)})")
                        elif isinstance(value, str) and value:
                            where_clauses.append("knowledge_units.tags LIKE ?")
                            params.append(f"%{value}%")
                    # Add more filters as needed

                if where_clauses:
                    query_parts.append("WHERE " + " AND ".join(where_clauses))

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
        Search for knowledge units by embedding similarity in the SQLite store.

        Args:
            embedding: The query embedding
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter results

        Returns:
            List of tuples of (knowledge unit, similarity score)

        Raises:
            MemoryStoreError: If the operation fails
        """
        await self._ensure_initialized()

        try:
            # Convert embedding to JSON for comparison
            query_embedding_json = json.dumps(embedding)

            async with aiosqlite.connect(self.database_path) as conn:
                conn.row_factory = aiosqlite.Row

                # Build the where clause for filter criteria
                where_clause = ""
                params = []

                if filter_criteria:
                    conditions = []

                    for key, value in filter_criteria.items():
                        if key == "knowledge_source":
                            conditions.append("knowledge_units.knowledge_source = ?")
                            params.append(value)
                        elif key == "tags":
                            if isinstance(value, list):
                                # Match any tag in the list
                                tag_conditions = []
                                for tag in value:
                                    tag_conditions.append("knowledge_units.tags LIKE ?")
                                    params.append(f"%{tag}%")
                                conditions.append(f"({' OR '.join(tag_conditions)})")
                            else:
                                conditions.append("knowledge_units.tags LIKE ?")
                                params.append(f"%{value}%")

                    if conditions:
                        where_clause = f"WHERE {' AND '.join(conditions)}"

                # Use a user-defined function for cosine similarity
                # This is a simplified approach for demonstration
                # In production, consider using vectors extension or other optimizations

                # Load all embeddings and perform similarity comparison in Python
                # This isn't scalable for large datasets but works for demo purposes

                # Build the query to get all embeddings with filter
                query = f"""
                SELECT knowledge_units.*, embeddings.embedding
                FROM knowledge_units
                JOIN embeddings ON knowledge_units.unique_id = embeddings.unique_id
                {where_clause}
                """

                results = []
                async with conn.execute(query, params) as cursor:
                    async for row in cursor:
                        # Deserialize the knowledge unit
                        knowledge_unit = self._deserialize_knowledge_unit(row)

                        # Deserialize the embedding
                        stored_embedding = self._deserialize_embedding(row["embedding"])
                        knowledge_unit.embedding_vector = stored_embedding

                        # Calculate cosine similarity
                        # For simplicity, we're using dot product as a proxy for cosine similarity
                        # This is valid if vectors are normalized
                        dot_product = sum(
                            q * s for q, s in zip(embedding, stored_embedding, strict=False)
                        )

                        # Normalize to 0-1 range (assuming normalized vectors)
                        similarity = (dot_product + 1) / 2

                        results.append((knowledge_unit, similarity))

                # Sort by similarity (descending) and limit results
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]
        except Exception as e:
            error_msg = f"Error performing similarity search in SQLite store: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e

    async def _generate_embedding_implementation(self, text: str) -> list[float]:
        """
        Generate an embedding for a text string.

        This implementation creates a random embedding with a hash-based seed to ensure
        the same text always gets the same embedding.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector

        Raises:
            MemoryStoreError: If the operation fails
        """
        import hashlib
        import random

        try:
            # Use a hash of the text as a random seed for reproducibility
            hash_object = hashlib.md5(text.encode("utf-8"))
            text_hash = int(hash_object.hexdigest(), 16)
            random.seed(text_hash)

            # Generate a random embedding vector
            embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

            # Normalize the vector to unit length
            magnitude = (sum(x**2 for x in embedding)) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]

            # Reset the random seed
            random.seed(None)

            return embedding
        except Exception as e:
            error_msg = f"Error generating embedding: {str(e)}"
            logger.error(error_msg)
            raise MemoryStoreError(error_msg) from e
