"""
Multi-modal knowledge storage for Long-Term Memory Agent.

This module provides support for storing and retrieving multiple modalities
of knowledge, including text, images, audio, and structured data.
"""

import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, BinaryIO

from PIL import Image

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)


class MultiModalKnowledgeUnit(KnowledgeUnit):
    """
    Extended knowledge unit supporting multiple modalities.

    This class extends the base KnowledgeUnit with support for non-text data.
    """

    def __init__(
        self,
        unique_id: str | None = None,
        original_chunk: str | None = None,
        processed_chunk: str | None = None,
        embedding: list[float] | None = None,
        created_at: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        modality: str = "text",
        media_data: bytes | None = None,
        media_format: str | None = None,
        media_path: str | None = None,
    ):
        """
        Initialize a multi-modal knowledge unit.

        Args:
            unique_id: Unique identifier for the knowledge unit
            original_chunk: Original text content
            processed_chunk: Processed text content
            embedding: Vector embedding of the knowledge
            created_at: Creation timestamp
            source: Source of the knowledge
            metadata: Additional metadata
            modality: Type of data ('text', 'image', 'audio', 'video', 'structured')
            media_data: Binary data for non-text modalities
            media_format: Format of the media (e.g., 'jpg', 'png', 'mp3')
            media_path: Path to stored media file
        """
        super().__init__(
            unique_id=unique_id,
            original_chunk=original_chunk,
            processed_chunk=processed_chunk,
            embedding=embedding,
            created_at=created_at,
            source=source,
            metadata=metadata,
        )

        self.modality = modality
        self.media_data = media_data
        self.media_format = media_format
        self.media_path = media_path

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the knowledge unit to a dictionary.

        Returns:
            Dictionary representation of the knowledge unit
        """
        # Get base dictionary from parent
        base_dict = super().to_dict()

        # Add multi-modal fields
        base_dict["modality"] = self.modality
        base_dict["media_format"] = self.media_format
        base_dict["media_path"] = self.media_path

        # Include media data as base64 if available and not too large
        if self.media_data and len(self.media_data) < 1024 * 1024:  # Less than 1MB
            base_dict["media_data_b64"] = base64.b64encode(self.media_data).decode("utf-8")

        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiModalKnowledgeUnit":
        """
        Create a knowledge unit from a dictionary.

        Args:
            data: Dictionary representation of the knowledge unit

        Returns:
            MultiModalKnowledgeUnit instance
        """
        # Extract multi-modal fields
        modality = data.pop("modality", "text")
        media_format = data.pop("media_format", None)
        media_path = data.pop("media_path", None)

        # Extract and decode media data if present
        media_data = None
        if "media_data_b64" in data:
            try:
                media_data = base64.b64decode(data.pop("media_data_b64"))
            except Exception as e:
                logger.warning(f"Failed to decode media data: {e}")

        # Create unit with base fields
        unit = super().from_dict(data)

        # Convert to multi-modal unit
        return MultiModalKnowledgeUnit(
            unique_id=unit.unique_id,
            original_chunk=unit.original_chunk,
            processed_chunk=unit.processed_chunk,
            embedding=unit.embedding,
            created_at=unit.created_at,
            source=unit.source,
            metadata=unit.metadata,
            modality=modality,
            media_data=media_data,
            media_format=media_format,
            media_path=media_path,
        )

    def get_text_representation(self) -> str:
        """
        Get a text representation of the knowledge unit.

        Returns:
            Text representation of the unit
        """
        if self.modality == "text":
            return self.original_chunk or ""

        # For non-text modalities, use the processed text description
        if self.processed_chunk:
            return self.processed_chunk

        # Fall back to metadata
        if self.metadata and "description" in self.metadata:
            return self.metadata["description"]

        # Last resort
        return f"{self.modality} content ({self.media_format})"

    def get_media_data(self) -> bytes | None:
        """
        Get the media data, loading from file if necessary.

        Returns:
            Binary media data or None if not available
        """
        if self.media_data:
            return self.media_data

        # Try to load from file path
        if self.media_path and os.path.exists(self.media_path):
            try:
                with open(self.media_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load media data from {self.media_path}: {e}")

        return None

    def save_media_to_file(self, directory: str) -> str | None:
        """
        Save media data to a file.

        Args:
            directory: Directory to save the file in

        Returns:
            Path to the saved file or None if failed
        """
        if not self.media_data:
            logger.warning("No media data to save")
            return None

        # Create directory if needed
        os.makedirs(directory, exist_ok=True)

        # Generate filename
        ext = self.media_format or "bin"
        filename = f"{self.unique_id}.{ext}"
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(self.media_data)

            # Update media path
            self.media_path = filepath
            return filepath

        except Exception as e:
            logger.error(f"Failed to save media data to {filepath}: {e}")
            return None


class MultiModalVectorStore(VectorStore):
    """
    Vector store supporting multiple modalities of knowledge.

    This class extends the base VectorStore to support storing and retrieving
    text, images, audio, and other modalities of knowledge.
    """

    def __init__(
        self,
        base_store: VectorStore,
        media_directory: str = "./media",
        text_encoder=None,
        image_encoder=None,
        audio_encoder=None,
    ):
        """
        Initialize the multi-modal vector store.

        Args:
            base_store: Underlying vector store for embedding storage
            media_directory: Directory to store media files
            text_encoder: Function to encode text to embeddings
            image_encoder: Function to encode images to embeddings
            audio_encoder: Function to encode audio to embeddings
        """
        self.base_store = base_store
        self.media_directory = media_directory
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder

        # Create media directory if it doesn't exist
        os.makedirs(media_directory, exist_ok=True)

    def add(self, unit: KnowledgeUnit | MultiModalKnowledgeUnit) -> None:
        """
        Add a knowledge unit to the store.

        Args:
            unit: Knowledge unit to add (standard or multi-modal)
        """
        # Convert standard unit to multi-modal if needed
        if not isinstance(unit, MultiModalKnowledgeUnit):
            unit = MultiModalKnowledgeUnit(
                unique_id=unit.unique_id,
                original_chunk=unit.original_chunk,
                processed_chunk=unit.processed_chunk,
                embedding=unit.embedding,
                created_at=unit.created_at,
                source=unit.source,
                metadata=unit.metadata,
                modality="text",
            )

        # Save media data to file if present
        if unit.media_data and not unit.media_path:
            unit.save_media_to_file(self.media_directory)

        # Add to base store
        self.base_store.add(unit)

    def update(self, unit: KnowledgeUnit | MultiModalKnowledgeUnit) -> None:
        """
        Update a knowledge unit in the store.

        Args:
            unit: Knowledge unit to update (standard or multi-modal)
        """
        # Convert standard unit to multi-modal if needed
        if not isinstance(unit, MultiModalKnowledgeUnit):
            # Get existing unit to preserve modality info
            existing = self.get(unit.unique_id)

            if existing and isinstance(existing, MultiModalKnowledgeUnit):
                unit = MultiModalKnowledgeUnit(
                    unique_id=unit.unique_id,
                    original_chunk=unit.original_chunk,
                    processed_chunk=unit.processed_chunk,
                    embedding=unit.embedding,
                    created_at=unit.created_at,
                    source=unit.source,
                    metadata=unit.metadata,
                    modality=existing.modality,
                    media_data=existing.media_data,
                    media_format=existing.media_format,
                    media_path=existing.media_path,
                )
            else:
                unit = MultiModalKnowledgeUnit(
                    unique_id=unit.unique_id,
                    original_chunk=unit.original_chunk,
                    processed_chunk=unit.processed_chunk,
                    embedding=unit.embedding,
                    created_at=unit.created_at,
                    source=unit.source,
                    metadata=unit.metadata,
                    modality="text",
                )

        # Save media data to file if present
        if unit.media_data and not unit.media_path:
            unit.save_media_to_file(self.media_directory)

        # Update in base store
        self.base_store.update(unit)

    def delete(self, unique_id: str) -> bool:
        """
        Delete a knowledge unit from the store.

        Args:
            unique_id: ID of the knowledge unit to delete

        Returns:
            True if deleted, False otherwise
        """
        # Get the unit first to check for media files
        unit = self.get(unique_id)

        if isinstance(unit, MultiModalKnowledgeUnit) and unit.media_path:
            # Delete media file if it exists
            try:
                if os.path.exists(unit.media_path):
                    os.remove(unit.media_path)
            except Exception as e:
                logger.warning(f"Failed to delete media file {unit.media_path}: {e}")

        # Delete from base store
        return self.base_store.delete(unique_id)

    def get(self, unique_id: str) -> KnowledgeUnit | MultiModalKnowledgeUnit | None:
        """
        Get a knowledge unit by ID.

        Args:
            unique_id: ID of the knowledge unit

        Returns:
            Knowledge unit if found, None otherwise
        """
        return self.base_store.get(unique_id)

    def search(
        self, embedding: list[float], limit: int = 5, threshold: float = 0.0
    ) -> list[tuple[KnowledgeUnit | MultiModalKnowledgeUnit, float]]:
        """
        Search for similar knowledge units.

        Args:
            embedding: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (knowledge_unit, similarity_score) tuples
        """
        return self.base_store.search(embedding, limit, threshold)

    def list_all(self) -> list[KnowledgeUnit | MultiModalKnowledgeUnit]:
        """
        List all knowledge units in the store.

        Returns:
            List of all knowledge units
        """
        return self.base_store.list_all()

    def clear(self) -> None:
        """Clear all knowledge units from the store."""
        # Get all units to find media files
        units = self.list_all()

        # Delete media files
        for unit in units:
            if isinstance(unit, MultiModalKnowledgeUnit) and unit.media_path:
                try:
                    if os.path.exists(unit.media_path):
                        os.remove(unit.media_path)
                except Exception as e:
                    logger.warning(f"Failed to delete media file {unit.media_path}: {e}")

        # Clear base store
        self.base_store.clear()

    def count(self) -> int:
        """
        Count the number of knowledge units in the store.

        Returns:
            Number of knowledge units
        """
        return self.base_store.count()

    def add_image(
        self,
        image_data: bytes | BinaryIO | Image.Image,
        description: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        image_format: str = "png",
    ) -> MultiModalKnowledgeUnit:
        """
        Add an image to the vector store.

        Args:
            image_data: Image data as bytes, file object, or PIL Image
            description: Text description of the image
            source: Optional source of the image
            metadata: Optional metadata about the image
            image_format: Format of the image ('png', 'jpg', etc.)

        Returns:
            MultiModalKnowledgeUnit containing the image
        """
        # Convert image to bytes if needed
        binary_data = None

        if isinstance(image_data, Image.Image):
            # Convert PIL Image to bytes
            img_io = io.BytesIO()
            image_data.save(img_io, format=image_format)
            binary_data = img_io.getvalue()

        elif hasattr(image_data, "read"):
            # Read from file-like object
            binary_data = image_data.read()

        else:
            # Assume it's already bytes
            binary_data = image_data

        # Generate embedding for the image
        embedding = None
        if self.image_encoder:
            try:
                # Convert bytes to PIL Image for encoding
                img = Image.open(io.BytesIO(binary_data))
                embedding = self.image_encoder(img)
            except Exception as e:
                logger.warning(f"Failed to encode image: {e}")

        # If no image embedding, use text description
        if embedding is None and self.text_encoder and description:
            embedding = self.text_encoder(description)

        # Create multi-modal knowledge unit
        unit = MultiModalKnowledgeUnit(
            unique_id=str(uuid.uuid4()),
            original_chunk=None,
            processed_chunk=description,
            embedding=embedding,
            created_at=datetime.now().isoformat(),
            source=source,
            metadata=metadata or {},
            modality="image",
            media_data=binary_data,
            media_format=image_format,
        )

        # Add to store
        self.add(unit)

        return unit

    def add_audio(
        self,
        audio_data: bytes | BinaryIO,
        description: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        audio_format: str = "mp3",
    ) -> MultiModalKnowledgeUnit:
        """
        Add audio to the vector store.

        Args:
            audio_data: Audio data as bytes or file object
            description: Text description of the audio
            source: Optional source of the audio
            metadata: Optional metadata about the audio
            audio_format: Format of the audio ('mp3', 'wav', etc.)

        Returns:
            MultiModalKnowledgeUnit containing the audio
        """
        # Convert to bytes if needed
        binary_data = None

        if hasattr(audio_data, "read"):
            # Read from file-like object
            binary_data = audio_data.read()
        else:
            # Assume it's already bytes
            binary_data = audio_data

        # Generate embedding for the audio
        embedding = None
        if self.audio_encoder:
            try:
                embedding = self.audio_encoder(binary_data)
            except Exception as e:
                logger.warning(f"Failed to encode audio: {e}")

        # If no audio embedding, use text description
        if embedding is None and self.text_encoder and description:
            embedding = self.text_encoder(description)

        # Create multi-modal knowledge unit
        unit = MultiModalKnowledgeUnit(
            unique_id=str(uuid.uuid4()),
            original_chunk=None,
            processed_chunk=description,
            embedding=embedding,
            created_at=datetime.now().isoformat(),
            source=source,
            metadata=metadata or {},
            modality="audio",
            media_data=binary_data,
            media_format=audio_format,
        )

        # Add to store
        self.add(unit)

        return unit

    def add_structured_data(
        self,
        data: dict[str, Any],
        description: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiModalKnowledgeUnit:
        """
        Add structured data to the vector store.

        Args:
            data: Structured data as a dictionary
            description: Text description of the data
            source: Optional source of the data
            metadata: Optional metadata about the data

        Returns:
            MultiModalKnowledgeUnit containing the structured data
        """
        # Convert data to JSON
        json_data = json.dumps(data).encode("utf-8")

        # Generate embedding from text description
        embedding = None
        if self.text_encoder:
            # Use both description and serialized data for better embedding
            combined_text = f"{description}\n{json.dumps(data, indent=2)}"
            embedding = self.text_encoder(combined_text)

        # Create multi-modal knowledge unit
        unit = MultiModalKnowledgeUnit(
            unique_id=str(uuid.uuid4()),
            original_chunk=json.dumps(data, indent=2),
            processed_chunk=description,
            embedding=embedding,
            created_at=datetime.now().isoformat(),
            source=source,
            metadata=metadata or {},
            modality="structured",
            media_data=json_data,
            media_format="json",
        )

        # Add to store
        self.add(unit)

        return unit

    def search_by_modality(
        self, embedding: list[float], modality: str, limit: int = 5, threshold: float = 0.0
    ) -> list[tuple[MultiModalKnowledgeUnit, float]]:
        """
        Search for similar knowledge units of a specific modality.

        Args:
            embedding: Query embedding
            modality: Modality to filter by ('text', 'image', 'audio', 'video', 'structured')
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (knowledge_unit, similarity_score) tuples
        """
        # Search across all units
        all_results = self.search(embedding, limit=limit * 2, threshold=threshold)

        # Filter by modality
        filtered_results = []
        for unit, score in all_results:
            if isinstance(unit, MultiModalKnowledgeUnit) and unit.modality == modality:
                filtered_results.append((unit, score))

        # Sort by score and limit
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:limit]

    def get_media_stats(self) -> dict[str, int]:
        """
        Get statistics about media types in the store.

        Returns:
            Dictionary mapping modality to count
        """
        stats = {"text": 0, "image": 0, "audio": 0, "video": 0, "structured": 0, "other": 0}

        # Count units by modality
        units = self.list_all()

        for unit in units:
            if isinstance(unit, MultiModalKnowledgeUnit):
                if unit.modality in stats:
                    stats[unit.modality] += 1
                else:
                    stats["other"] += 1
            else:
                stats["text"] += 1

        return stats
