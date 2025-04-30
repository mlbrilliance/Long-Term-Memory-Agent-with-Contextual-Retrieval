"""
Context analysis and metadata enrichment for knowledge retrieval.

This module provides utilities for analyzing and enriching context items,
extracting key entities, topics, and relationships to improve retrieval quality.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analyzes and enriches context items for improved retrieval and understanding.

    This class provides functionality for:
    - Entity extraction and topic detection
    - Content type classification (text, code, list, etc.)
    - Relevance score adjustment based on content quality
    - Temporal analysis and recency scoring
    - Relationship detection between context items
    - Metadata enrichment for better context integration
    """

    def __init__(
        self,
        enable_entity_extraction: bool = True,
        enable_content_classification: bool = True,
        enable_relationship_detection: bool = False,  # More compute-intensive
        custom_entity_patterns: dict[str, str] | None = None,
        quality_scoring_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the context analyzer.

        Args:
            enable_entity_extraction: Whether to extract entities from content
            enable_content_classification: Whether to classify content types
            enable_relationship_detection: Whether to detect relationships between items
            custom_entity_patterns: Optional dict of entity type -> regex pattern
            quality_scoring_weights: Optional weights for quality scoring
        """
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_content_classification = enable_content_classification
        self.enable_relationship_detection = enable_relationship_detection

        # Default entity patterns for extraction
        self.entity_patterns = custom_entity_patterns or {
            "name": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Capitalized phrases
            "date": r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b|\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://\S+|www\.\S+",
            "quoted": r'"([^"]*)"',
            "number": r"\b\d+(?:\.\d+)?\b",
        }

        # Default weights for quality scoring components
        self.quality_weights = quality_scoring_weights or {
            "length": 0.15,  # Longer content (up to a point)
            "entities": 0.25,  # More named entities
            "structure": 0.20,  # Better structure (lists, formatting)
            "recency": 0.30,  # More recent content
            "diversity": 0.10,  # Content type diversity
        }

        logger.info(
            f"Initialized ContextAnalyzer with entity extraction: {enable_entity_extraction}"
        )

    def analyze_context_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze and enrich a single context item.

        Args:
            item: The context item to analyze

        Returns:
            Enriched context item with additional metadata
        """
        if not item:
            return {}

        # Create a copy to avoid modifying the original
        enriched = item.copy()

        # Ensure metadata exists
        if "metadata" not in enriched:
            enriched["metadata"] = {}

        # Extract content for analysis
        content = enriched.get("content", "")
        context = enriched.get("context", "")
        combined_text = f"{context} {content}".strip()

        # Skip empty content
        if not combined_text:
            return enriched

        # Add basic content stats
        content_length = len(content)
        enriched["metadata"]["content_length"] = content_length

        # Categorize by length
        if content_length < 100:
            length_category = "short"
        elif content_length < 500:
            length_category = "medium"
        else:
            length_category = "long"
        enriched["metadata"]["length_category"] = length_category

        # Extract entities if enabled
        if self.enable_entity_extraction:
            entities_by_type = self._extract_entities(combined_text)
            if entities_by_type:
                enriched["metadata"]["entities"] = entities_by_type
                # Add flat list of all entities
                all_entities = []
                for entity_list in entities_by_type.values():
                    all_entities.extend(entity_list)
                enriched["metadata"]["all_entities"] = list(set(all_entities))

        # Classify content type if enabled
        if self.enable_content_classification:
            content_type = self._classify_content(content)
            enriched["metadata"]["content_type"] = content_type

            # Add specific attributes based on content type
            if content_type == "code":
                enriched["metadata"]["language"] = self._detect_code_language(content)
            elif content_type == "list":
                enriched["metadata"]["list_items"] = self._count_list_items(content)
            elif content_type == "table":
                rows, cols = self._analyze_table_dimensions(content)
                enriched["metadata"]["table_rows"] = rows
                enriched["metadata"]["table_cols"] = cols

        # Extract topics if not already present
        if "topics" not in enriched["metadata"]:
            topics = self._extract_topics(combined_text)
            if topics:
                enriched["metadata"]["topics"] = topics

        # Calculate adjusted quality score
        quality_score = self._calculate_quality_score(enriched)
        enriched["metadata"]["quality_score"] = quality_score

        # Adjust relevance score if present, incorporating quality
        if "relevance" in enriched:
            # Blend original relevance with quality (70% original, 30% quality)
            original_relevance = enriched["relevance"]
            adjusted_relevance = (0.7 * original_relevance) + (0.3 * quality_score)
            enriched["adjusted_relevance"] = min(1.0, max(0.0, adjusted_relevance))

        logger.debug(
            f"Analyzed context item: added {len(enriched['metadata']) - len(item.get('metadata', {}))} metadata fields"
        )
        return enriched

    def analyze_context_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze and enrich a list of context items, including cross-item relationships.

        Args:
            items: List of context items to analyze

        Returns:
            List of enriched context items
        """
        if not items:
            return []

        # First, analyze each item individually
        enriched_items = [self.analyze_context_item(item) for item in items]

        # Skip relationship detection if disabled or fewer than 2 items
        if not self.enable_relationship_detection or len(items) < 2:
            return enriched_items

        # Detect relationships between items
        relationships = self._detect_relationships(enriched_items)

        # Add relationship information to each item
        for i, item in enumerate(enriched_items):
            if i in relationships:
                item["metadata"]["related_items"] = relationships[i]

        return enriched_items

    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> list[str]:
        """
        Extract key phrases from text that could be used for retrieval.

        Args:
            text: Text to analyze
            max_phrases: Maximum number of phrases to extract

        Returns:
            List of key phrases
        """
        if not text:
            return []

        # Simple approach using regex for noun phrases
        # A more sophisticated approach would use NLP with part-of-speech tagging

        # Look for noun phrases: adjective(s) + noun(s)
        noun_phrases = re.findall(r"\b([A-Za-z]+\s+){1,3}[A-Za-z]+\b", text)

        # Look for capitalized phrases (potential named entities)
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)

        # Look for quoted phrases
        quoted = re.findall(r'"([^"]*)"', text)

        # Combine and rank
        candidates = []

        # Add noun phrases, filtering for minimum length
        candidates.extend([np.strip() for np in noun_phrases if len(np.strip()) > 5])

        # Add capitalized phrases
        candidates.extend([cp for cp in capitalized if len(cp) > 4])

        # Add quoted phrases of reasonable length
        candidates.extend([q for q in quoted if 4 <= len(q) <= 30])

        # Sort by length, longer phrases often more specific
        candidates.sort(key=len, reverse=True)

        # De-duplicate and limit
        seen = set()
        result = []
        for phrase in candidates:
            normalized = phrase.lower()
            if normalized not in seen and len(result) < max_phrases:
                seen.add(normalized)
                result.append(phrase)

        return result[:max_phrases]

    def calculate_semantic_similarity(self, item1: dict[str, Any], item2: dict[str, Any]) -> float:
        """
        Calculate a simple lexical similarity score between two context items.

        Args:
            item1: First context item
            item2: Second context item

        Returns:
            Similarity score (0-1)
        """
        # Extract text from both items
        text1 = f"{item1.get('context', '')} {item1.get('content', '')}".strip().lower()
        text2 = f"{item2.get('context', '')} {item2.get('content', '')}".strip().lower()

        if not text1 or not text2:
            return 0.0

        # Split into words
        words1 = set(re.findall(r"\b\w+\b", text1))
        words2 = set(re.findall(r"\b\w+\b", text2))

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        return intersection / union

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extract entities from text using regex patterns.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of entity types to lists of entities
        """
        entities = {}

        # Apply each pattern
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)

            # Handle multi-group matches (e.g., from date pattern)
            extracted = []
            for match in matches:
                if isinstance(match, tuple):
                    # Take the first non-empty group
                    for group in match:
                        if group:
                            extracted.append(group)
                            break
                else:
                    extracted.append(match)

            # Only add non-empty entity lists
            if extracted:
                # Remove duplicates while preserving order
                unique_entities = []
                seen = set()
                for entity in extracted:
                    if entity not in seen:
                        seen.add(entity)
                        unique_entities.append(entity)

                entities[entity_type] = unique_entities

        return entities

    def _classify_content(self, text: str) -> str:
        """
        Classify the type of content in the text.

        Args:
            text: Text to classify

        Returns:
            Content type classification
        """
        if not text:
            return "empty"

        # Check for code
        code_patterns = [
            r"```[\s\S]*```",  # Markdown code blocks
            r"def\s+\w+\(.*\):",  # Python function definitions
            r"class\s+\w+[(:)]",  # Class definitions
            r"function\s+\w+\(.*\)",  # JavaScript functions
            r"import\s+[\w.]+",  # Import statements
            r"<\/?[a-z][\w-]*>",  # HTML tags
            r"[a-zA-Z$_][\w$_]*\s*=\s*function\s*\(",  # JS function assignments
            r"const|let|var\s+\w+",  # JS variable declarations
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return "code"

        # Check for tables
        table_patterns = [
            r"\|\s*-+\s*\|",  # Markdown tables
            r"<table[\s>][\s\S]*<\/table>",  # HTML tables
            r"^\s*\+[-+]+\+\s*$",  # ASCII tables
        ]
        for pattern in table_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "table"

        # Check for lists
        list_patterns = [
            r"^\s*[\*\-\+•]\s+\w+",  # Bullet lists
            r"^\s*\d+\.\s+\w+",  # Numbered lists
        ]
        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "list"

        # Check for JSON
        if text.strip().startswith("{") and text.strip().endswith("}"):
            try:
                json.loads(text)
                return "json"
            except:
                pass

        # Check for Q&A format
        if re.search(r"(?:^|\n)Q:[\s\S]*?(?:^|\n)A:", text):
            return "qa_format"

        # Default to plain text
        return "text"

    def _detect_code_language(self, code: str) -> str:
        """
        Attempt to detect the programming language of code.

        Args:
            code: Code text to analyze

        Returns:
            Detected language or "unknown"
        """
        # Simple detection based on keywords and syntax patterns
        if re.search(
            r'def\s+\w+\(|import\s+\w+|class\s+\w+\(|if\s+__name__\s*==\s*[\'"]__main__[\'"]:', code
        ):
            return "python"
        elif re.search(
            r"function\s+\w+\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|=>\s*{", code
        ):
            return "javascript"
        elif re.search(r"<html|<div|<body|<script|<style", code):
            return "html"
        elif re.search(r"#include\s*<\w+\.h>|int\s+main\(", code):
            return "c/c++"
        elif re.search(r"public\s+class|public\s+static\s+void\s+main", code):
            return "java"

        return "unknown"

    def _count_list_items(self, text: str) -> int:
        """
        Count the number of list items in the text.

        Args:
            text: Text to analyze

        Returns:
            Number of list items
        """
        # Count bullet list items
        bullet_items = len(re.findall(r"^\s*[\*\-\+•]\s+\w+", text, re.MULTILINE))

        # Count numbered list items
        numbered_items = len(re.findall(r"^\s*\d+\.\s+\w+", text, re.MULTILINE))

        return bullet_items + numbered_items

    def _analyze_table_dimensions(self, text: str) -> tuple[int, int]:
        """
        Analyze the dimensions of a table.

        Args:
            text: Text containing a table

        Returns:
            Tuple of (rows, columns)
        """
        # Check for Markdown table
        if "|" in text:
            lines = text.split("\n")
            table_lines = [line for line in lines if line.strip().startswith("|")]

            if table_lines:
                rows = len(table_lines)
                if rows > 1:  # Accounting for header row
                    rows -= 1

                # Estimate columns by counting pipe separators in a row
                columns = len(re.findall(r"\|", table_lines[0])) - 1
                return rows, columns

        # Default dimensions if analysis fails
        return 0, 0

    def _extract_topics(self, text: str) -> list[str]:
        """
        Extract potential topics from text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted topics
        """
        if not text:
            return []

        # Use a simple keyword frequency approach
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "of",
            "to",
            "for",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "his",
            "her",
            "he",
            "she",
            "i",
            "you",
            "we",
        }

        # Extract words, filtering out punctuation and stop words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [word for word in words if word not in stop_words]

        # Count word frequency
        counter = Counter(filtered)

        # Get most common words as topics (up to 5)
        top_words = [word for word, _ in counter.most_common(5)]

        # Add any capitalized phrases as potential topics
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        for phrase in capitalized:
            if len(phrase) > 3 and len(top_words) < 7:  # Limit to 7 total topics
                top_words.append(phrase)

        return list(set(top_words))  # Remove duplicates

    def _calculate_quality_score(self, item: dict[str, Any]) -> float:
        """
        Calculate a quality score for the context item.

        Args:
            item: The context item to score

        Returns:
            Quality score (0-1)
        """
        metadata = item.get("metadata", {})

        # Score for content length (0-1)
        content_length = metadata.get("content_length", 0)
        # Prefer medium-length content (not too short, not too long)
        if content_length < 50:
            length_score = 0.3 * (content_length / 50)
        elif content_length < 1000:
            length_score = 0.3 + 0.7 * (content_length / 1000)
        else:
            # Penalize extremely long content slightly
            length_score = 1.0 - (0.2 * min(1.0, (content_length - 1000) / 4000))

        # Score for entities (0-1)
        entity_count = len(metadata.get("all_entities", []))
        entity_score = min(1.0, entity_count / 10)  # 10+ entities = 1.0

        # Score for structure (0-1)
        content_type = metadata.get("content_type", "text")
        if content_type in ["list", "table"]:
            structure_score = 0.8  # Well-structured content
        elif content_type == "code":
            structure_score = 0.7  # Code is structured but not always the best for all contexts
        elif content_type == "qa_format":
            structure_score = 0.9  # Q&A is excellent for context
        else:
            structure_score = 0.5  # Plain text

        # Score for recency (0-1)
        recency_score = 0.5  # Default middle score
        timestamp = item.get("timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = timestamp

                # Calculate age in days
                now = datetime.now(timezone.utc)
                age_days = (now - dt).total_seconds() / (24 * 3600)

                # Exponential decay: 1.0 for fresh, 0.5 after 30 days, approaching 0
                recency_score = max(0.2, min(1.0, 2 ** (-age_days / 30)))
            except (ValueError, TypeError):
                pass

        # Score for diversity (0-1)
        topics = metadata.get("topics", [])
        diversity_score = min(1.0, len(topics) / 5)  # 5+ topics = 1.0

        # Calculate weighted average
        weighted_score = (
            self.quality_weights["length"] * length_score
            + self.quality_weights["entities"] * entity_score
            + self.quality_weights["structure"] * structure_score
            + self.quality_weights["recency"] * recency_score
            + self.quality_weights["diversity"] * diversity_score
        )

        return min(1.0, max(0.0, weighted_score))

    def _detect_relationships(self, items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        """
        Detect relationships between context items.

        Args:
            items: List of context items

        Returns:
            Dictionary mapping item indices to lists of related items
        """
        relationships = {}

        # Skip if too few items
        if len(items) < 2:
            return relationships

        # Compare each pair of items
        for i in range(len(items)):
            item1 = items[i]
            related = []

            for j in range(len(items)):
                if i == j:
                    continue

                item2 = items[j]

                # Calculate similarity
                similarity = self.calculate_semantic_similarity(item1, item2)

                # Check for shared entities
                shared_entities = set()
                entities1 = set(item1.get("metadata", {}).get("all_entities", []))
                entities2 = set(item2.get("metadata", {}).get("all_entities", []))

                if entities1 and entities2:
                    shared_entities = entities1.intersection(entities2)

                # Check for shared topics
                shared_topics = set()
                topics1 = set(item1.get("metadata", {}).get("topics", []))
                topics2 = set(item2.get("metadata", {}).get("topics", []))

                if topics1 and topics2:
                    shared_topics = topics1.intersection(topics2)

                # Determine if items are related
                is_related = (
                    similarity > 0.3  # Significant text overlap
                    or len(shared_entities) >= 2  # Multiple shared entities
                    or len(shared_topics) >= 2  # Multiple shared topics
                )

                if is_related:
                    related.append(
                        {
                            "index": j,
                            "similarity": similarity,
                            "shared_entities": list(shared_entities),
                            "shared_topics": list(shared_topics),
                        }
                    )

            # Sort related items by similarity (highest first)
            if related:
                related.sort(key=lambda x: x["similarity"], reverse=True)
                relationships[i] = related

        return relationships
