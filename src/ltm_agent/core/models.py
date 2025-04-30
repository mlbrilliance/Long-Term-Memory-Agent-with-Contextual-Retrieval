"""
Core data structures for the Long-Term Memory Agent.

This module defines the fundamental data models used throughout the application,
including the KnowledgeUnit which represents a single piece of knowledge in the
agent's memory.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class KnowledgeUnit(BaseModel):
    """
    Represents a single unit of knowledge in the agent's memory.

    Each knowledge unit contains:
    - A unique identifier
    - The original text chunk
    - Optional contextual text providing additional context
    - Optional embedding vector used for similarity search
    - Source of the knowledge (action, feedback, or corpus)
    - Timestamp of when the knowledge was created
    - Metadata for additional information and filtering
    """

    unique_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_chunk: str
    contextual_text: str = ""
    embedding_vector: list[float] | None = None
    knowledge_source: Literal["action", "feedback", "corpus"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, tags: list[str] | str | None) -> list[str]:
        """
        Normalize tags to a list of lowercase strings.

        Args:
            tags: Tags to normalize, can be a list, comma-separated string, or None

        Returns:
            List of normalized tags
        """
        if tags is None:
            return []

        if isinstance(tags, str):
            # Split comma-separated string and strip whitespace
            tags = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        else:
            # Normalize list tags to lowercase
            tags = [tag.strip().lower() for tag in tags if tag.strip()]

        # Remove duplicates while preserving order
        seen = set()
        return [tag for tag in tags if not (tag in seen or seen.add(tag))]

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the knowledge unit.

        Args:
            tag: Tag to add
        """
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag from the knowledge unit.

        Args:
            tag: Tag to remove

        Returns:
            True if the tag was removed, False if it wasn't present
        """
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update a metadata field.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata field.

        Args:
            key: Metadata key
            default: Default value if key doesn't exist

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def has_tag(self, tag: str) -> bool:
        """
        Check if the knowledge unit has a specific tag.

        Args:
            tag: Tag to check for

        Returns:
            True if the tag is present, False otherwise
        """
        return tag.strip().lower() in self.tags

    class Config:
        """Configuration for the KnowledgeUnit model."""

        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
