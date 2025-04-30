#!/usr/bin/env python
"""
Memory Unit

Core component of the enhanced memory system that represents individual knowledge units.
"""

import re
import uuid
from datetime import datetime
from typing import Any


class MemoryUnit:
    """Enhanced memory unit for knowledge storage."""

    def __init__(
        self,
        content: str,
        unit_type: str = "fact",
        metadata: dict[str, Any] | None = None,
        unit_id: str | None = None,
    ):
        """
        Initialize a memory unit.

        Args:
            content: The content of the memory unit
            unit_type: The type of the memory unit (fact, correction, consolidation, etc.)
            metadata: Additional metadata for the memory unit
            unit_id: Optional ID for the memory unit (if None, a UUID will be generated)
        """
        self.id = unit_id if unit_id else str(uuid.uuid4())
        self.content = content
        self.type = unit_type
        self.metadata = metadata or {}
        self.creation_time = datetime.now().isoformat()
        self.relations = []  # Store connections to other units
        self.concepts = self._extract_concepts()
        self.entities = self._extract_entities()

    def _extract_concepts(self) -> set[str]:
        """Extract key concepts from content."""
        # This would use NLP in a production system, but for simplicity:
        words = re.findall(r"\b\w+\b", self.content.lower())
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "of",
            "that",
            "this",
            "these",
            "those",
            "it",
            "they",
        }

        concepts = set()
        for word in words:
            if word not in stop_words and len(word) > 3:
                concepts.add(word)

        # Add specific phrases
        phrases = [
            "machine learning",
            "artificial intelligence",
            "neural network",
            "deep learning",
            "natural language",
            "knowledge graph",
        ]

        for phrase in phrases:
            if phrase in self.content.lower():
                concepts.add(phrase)

        return concepts

    def _extract_entities(self) -> dict[str, list[str]]:
        """Extract named entities from content."""
        # In a production system, this would use NLP for entity recognition
        entities = {"people": [], "places": [], "organizations": [], "technologies": [], "time": []}

        # Simple name detection
        # Look for capitalized words that aren't at the start of a sentence
        potential_names = re.findall(r"(?<!^)(?<!\. )[A-Z][a-z]+", self.content)
        for name in potential_names:
            entities["people"].append(name)

        # Detect time references
        time_words = ["year", "month", "day", "century", "decade"]
        numbers = re.findall(r"\b\d{4}\b", self.content)  # Years

        for number in numbers:
            if int(number) > 1500 and int(number) <= datetime.now().year:
                entities["time"].append(number)

        # Technology terms
        tech_terms = ["Python", "JavaScript", "AI", "ML", "neural network", "algorithm", "model"]
        for term in tech_terms:
            if term.lower() in self.content.lower():
                entities["technologies"].append(term)

        return entities

    def add_relation(self, target_id: str, relation_type: str) -> None:
        """
        Add a relationship to another memory unit.

        Args:
            target_id: The ID of the target memory unit
            relation_type: The type of the relationship
        """
        # Check if relation already exists
        for relation in self.relations:
            if relation["target_id"] == target_id and relation["type"] == relation_type:
                return

        self.relations.append({"target_id": target_id, "type": relation_type})

    def to_dict(self) -> dict[str, Any]:
        """Convert the memory unit to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "concepts": list(self.concepts),
            "entities": self.entities,
            "relations": self.relations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryUnit":
        """Create a memory unit from a dictionary."""
        unit = cls(
            content=data["content"],
            unit_type=data["type"],
            metadata=data["metadata"],
            unit_id=data["id"],
        )
        unit.creation_time = data["creation_time"]
        unit.relations = data["relations"]
        return unit
