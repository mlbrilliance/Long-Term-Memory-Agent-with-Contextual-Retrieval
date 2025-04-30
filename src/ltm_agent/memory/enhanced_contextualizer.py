"""
Enhanced Contextualizer for Long-Term Memory Agent.

This module provides an improved contextualizer that better identifies
relationships between knowledge pieces and enhances multi-hop reasoning.
"""

import logging
import re
from collections import Counter
from typing import Any

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.contextualizer import SimpleContextualizer

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedContextualizer(SimpleContextualizer):
    """
    Enhanced contextualizer with improved relationship detection.

    This contextualizer extends the SimpleContextualizer with additional capabilities:
    1. Better entity and relationship extraction
    2. Contradiction detection
    3. Enhanced relevance scoring for multi-hop reasoning
    """

    def __init__(self):
        """Initialize the enhanced contextualizer."""
        super().__init__()

        # Common stopwords to filter out when analyzing content
        self.stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "about",
            "of",
            "as",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "have",
            "has",
            "had",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "cannot",
        }

        # Entity recognition patterns
        self.entity_patterns = [
            # Named entities (capitalized phrases)
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # Technical terms (camelCase, PascalCase, snake_case)
            r"\b([a-z]+(?:[A-Z][a-z]*)+)\b",  # camelCase
            r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b",  # PascalCase
            r"\b([a-z]+(?:_[a-z]+)+)\b",  # snake_case
            # Domains and topics (lowercase sequences)
            r"\b((?:python|javascript|ruby|java|c\+\+|php|go|rust|machine\slearning|deep\slearning|neural\snetworks|artificial\sintelligence|ai|ml|nlp|computer\svision|cv|data\sscience)[a-z]*)\b",
        ]

        # Patterns for identifying potential contradictions
        self.contradiction_patterns = [
            (r"not\s+([a-z]+)", r"\1"),  # "not X" vs "X"
            (r"no\s+([a-z]+)", r"[a-z]*\s*\1"),  # "no X" vs "any X"
            (r"never\s+([a-z]+)", r"[a-z]*\s*\1"),  # "never X" vs "X"
            (r"isn\'t\s+([a-z]+)", r"is\s+\1"),  # "isn't X" vs "is X"
            (r"aren\'t\s+([a-z]+)", r"are\s+\1"),  # "aren't X" vs "are X"
            (r"doesn\'t\s+([a-z]+)", r"does\s+\1"),  # "doesn't X" vs "does X"
            (r"don\'t\s+([a-z]+)", r"do\s+\1"),  # "don't X" vs "do X"
            (r"cannot\s+([a-z]+)", r"can\s+\1"),  # "cannot X" vs "can X"
            (r"can\'t\s+([a-z]+)", r"can\s+\1"),  # "can't X" vs "can X"
            (r"should\s+not\s+([a-z]+)", r"should\s+\1"),  # "should not X" vs "should X"
            (r"shouldn\'t\s+([a-z]+)", r"should\s+\1"),  # "shouldn't X" vs "should X"
        ]

        # Relation extraction patterns (subject-verb-object patterns)
        self.relation_patterns = [
            # X is Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+is\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "is_a",
            ),
            # X has Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+has\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "has",
            ),
            # X contains Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+contains\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "contains",
            ),
            # X uses Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+uses?\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "uses",
            ),
            # X provides Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+provides?\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "provides",
            ),
            # X works with Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+works?\s+with\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "works_with",
            ),
            # X depends on Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+depends?\s+on\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "depends_on",
            ),
            # X was developed by Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+was\s+developed\s+by\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "developed_by",
            ),
            # X is part of Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+is\s+(?:a)?\s*part\s+of\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "part_of",
            ),
            # X is related to Y
            (
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+is\s+related\s+to\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "related_to",
            ),
        ]

    def extract_metadata(self, content: str) -> dict[str, Any]:
        """
        Extract metadata from content, including entities and relationships.

        Args:
            content: Text content to analyze

        Returns:
            Dict with extracted metadata
        """
        # Start with basic metadata from parent class
        metadata = super().extract_metadata(content)

        # Extract entities
        entities = self._extract_entities(content)
        if entities:
            metadata["entities"] = list(entities)

        # Extract potential relationships
        relations = self._extract_relations(content)
        if relations:
            metadata["relations"] = relations

        # Extract topics (main subjects of the content)
        topics = self._extract_topics(content)
        if topics:
            metadata["topics"] = topics

        return metadata

    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract entities and important terms from text.

        Args:
            text: Text to extract entities from

        Returns:
            Set of entity strings
        """
        entities = set()

        # Apply each pattern to extract entities
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Ignore single character matches and stopwords
                if len(match) > 1 and match.lower() not in self.stopwords:
                    entities.add(match)

        return entities

    def _extract_relations(self, text: str) -> list[dict[str, str]]:
        """
        Extract subject-verb-object relationships from text.

        Args:
            text: Text to extract relationships from

        Returns:
            List of relationship dictionaries
        """
        relations = []

        # Apply relation patterns
        for pattern, relation_type in self.relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                subject, obj = match

                # Clean up the extracted terms
                subject = subject.strip().lower()
                obj = obj.strip().lower()

                # Skip if either term is too short or a stopword
                if (
                    len(subject) <= 2
                    or len(obj) <= 2
                    or subject in self.stopwords
                    or obj in self.stopwords
                ):
                    continue

                relations.append({"subject": subject, "predicate": relation_type, "object": obj})

        return relations

    def _extract_topics(self, text: str) -> list[str]:
        """
        Extract main topics from text using frequency and position.

        Args:
            text: Text to extract topics from

        Returns:
            List of main topics
        """
        # Extract words and count frequency
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        word_counts = Counter([w for w in words if w not in self.stopwords])

        # Consider words in the first and last sentences more important
        sentences = text.split(".")
        first_sentence = sentences[0] if sentences else ""
        last_sentence = sentences[-1] if len(sentences) > 1 else ""

        first_last_words = re.findall(
            r"\b[a-zA-Z]{3,}\b", (first_sentence + " " + last_sentence).lower()
        )

        # Boost words that appear in first/last sentences
        for word in first_last_words:
            if word not in self.stopwords:
                word_counts[word] += 1

        # Get top topics based on frequency
        topics = [word for word, count in word_counts.most_common(5) if count > 1]

        return topics

    def detect_contradiction(
        self, new_content: str, existing_content: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Detect if new content contradicts existing content.

        Args:
            new_content: New knowledge content
            existing_content: Existing knowledge content

        Returns:
            Tuple of (contradiction_found, contradiction_details)
        """
        # Simple negation pattern contradiction detection
        for neg_pattern, pos_pattern in self.contradiction_patterns:
            # Find negations in the new content
            neg_matches = re.findall(neg_pattern, new_content.lower())

            for neg_match in neg_matches:
                # Check if the opposite exists in the existing content
                pos_compiled = re.compile(pos_pattern, re.IGNORECASE)

                if pos_compiled.search(existing_content.lower()):
                    return True, {
                        "type": "negation_contradiction",
                        "new_pattern": neg_pattern.replace(r"\1", neg_match),
                        "existing_pattern": pos_pattern.replace(r"\1", neg_match),
                    }

        # Contradicting facts detection
        new_relations = self._extract_relations(new_content)
        existing_relations = self._extract_relations(existing_content)

        for new_rel in new_relations:
            for existing_rel in existing_relations:
                # Check for same subject-predicate but different object
                if (
                    new_rel["subject"] == existing_rel["subject"]
                    and new_rel["predicate"] == existing_rel["predicate"]
                    and new_rel["object"] != existing_rel["object"]
                ):
                    return True, {
                        "type": "fact_contradiction",
                        "subject": new_rel["subject"],
                        "predicate": new_rel["predicate"],
                        "new_object": new_rel["object"],
                        "existing_object": existing_rel["object"],
                    }

        return False, None

    def calculate_relevance(
        self, query: str, content: str, metadata: dict[str, Any] | None = None
    ) -> float:
        """
        Calculate relevance between query and content with improved logic.

        Args:
            query: Search query
            content: Content to compare against
            metadata: Optional metadata about the content

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Get basic relevance from parent implementation
        basic_relevance = super().calculate_relevance(query, content)

        # Extract entities from query and content
        query_entities = self._extract_entities(query)
        content_entities = self._extract_entities(content)

        # Calculate entity overlap
        common_entities = query_entities.intersection(content_entities)
        entity_score = len(common_entities) / max(
            1, min(len(query_entities), len(content_entities))
        )

        # Extract topics from query and content
        query_topics = self._extract_topics(query)
        content_topics = metadata.get("topics", []) if metadata else self._extract_topics(content)

        # Calculate topic overlap
        common_topics = set(query_topics).intersection(set(content_topics))
        topic_score = len(common_topics) / max(1, min(len(query_topics), len(content_topics)))

        # Combine scores (with weights)
        combined_score = (
            basic_relevance * 0.4  # Base text similarity
            + entity_score * 0.3  # Entity matching
            + topic_score * 0.3  # Topic matching
        )

        return min(1.0, combined_score)  # Cap at 1.0

    def enrich_knowledge(
        self, unit: KnowledgeUnit, related_units: list[KnowledgeUnit] | None = None
    ) -> KnowledgeUnit:
        """
        Enrich a knowledge unit with improved processing and context.

        Args:
            unit: Knowledge unit to enrich
            related_units: Optional list of related knowledge units

        Returns:
            Enriched knowledge unit
        """
        # First apply the basic enrichment from parent class
        enriched_unit = super().enrich_knowledge(unit, related_units)

        # Extract additional metadata
        metadata = self.extract_metadata(unit.original_chunk)

        # Combine with existing metadata
        if enriched_unit.metadata:
            for key, value in metadata.items():
                if key not in enriched_unit.metadata:
                    enriched_unit.metadata[key] = value
                elif isinstance(value, list) and isinstance(enriched_unit.metadata.get(key), list):
                    # Merge lists without duplicates
                    combined = list(set(enriched_unit.metadata[key] + value))
                    enriched_unit.metadata[key] = combined
        else:
            enriched_unit.metadata = metadata

        # If we have related units, enhance with relationship information
        if related_units:
            self._enhance_with_relationships(enriched_unit, related_units)

        return enriched_unit

    def _enhance_with_relationships(
        self, unit: KnowledgeUnit, related_units: list[KnowledgeUnit]
    ) -> None:
        """
        Enhance a knowledge unit with explicit relationship information.

        Args:
            unit: Knowledge unit to enhance
            related_units: Related knowledge units
        """
        if not unit.metadata:
            unit.metadata = {}

        if "related_to" not in unit.metadata:
            unit.metadata["related_to"] = []

        # Track relationship types
        relationship_types = {}

        for related in related_units:
            if related.unique_id == unit.unique_id:
                continue

            # Only add if not already present
            if related.unique_id not in unit.metadata["related_to"]:
                unit.metadata["related_to"].append(related.unique_id)

            # Try to determine relationship type
            unit_entities = self._extract_entities(unit.original_chunk)
            related_entities = self._extract_entities(related.original_chunk)

            # Find common entities
            common = unit_entities.intersection(related_entities)

            if common:
                relationship_types[related.unique_id] = {
                    "type": "shared_entities",
                    "entities": list(common),
                }

        # Store relationship types if any found
        if relationship_types:
            if "relationship_types" not in unit.metadata:
                unit.metadata["relationship_types"] = {}

            unit.metadata["relationship_types"].update(relationship_types)
