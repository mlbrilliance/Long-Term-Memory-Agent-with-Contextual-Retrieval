"""
Utility functions for testing the LTM Agent.

This module provides helper functions for generating test data
and configuring test environments.
"""

import datetime
import random
import string
import uuid
from typing import Any

from ltm_agent.core.config import Settings
from ltm_agent.core.models import KnowledgeUnit


def get_test_settings() -> Settings:
    """
    Create a Settings instance with test values.

    This function provides a consistent configuration for tests
    without relying on environment variables.

    Returns:
        Settings: A Settings instance with test values.
    """
    # Use a mock API key for testing
    return Settings(
        anthropic_api_key="test-api-key",
        perplexity_api_key="test-perplexity-key",
        logging_level="INFO",
        chunk_size=100,
        chunk_overlap=10,
        similarity_threshold=0.7,
        cache_size=1000,
        retry_attempts=3,
        retry_delay=0.1,
        max_batch_size=100,
    )


def create_test_knowledge_unit(
    content: str = "Test content",
    source: str = "test",
    context: str = "",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    unique_id: str | None = None,
) -> KnowledgeUnit:
    """
    Create a KnowledgeUnit instance for testing.

    Args:
        content: The original text content
        source: The source of the knowledge unit
        context: Contextual information for the content
        metadata: Additional metadata
        tags: Tags for categorization
        embedding: Vector embedding (random if not provided)
        unique_id: Unique identifier (random UUID if not provided)

    Returns:
        KnowledgeUnit: A test knowledge unit
    """
    if unique_id is None:
        unique_id = str(uuid.uuid4())

    if embedding is None:
        # Generate a random embedding vector for testing
        embedding = [random.uniform(-1.0, 1.0) for _ in range(128)]

    # Source must be one of the valid values
    if source == "test":
        source = "corpus"  # Default to a valid source

    # Ensure timezone awareness for timestamp
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    # Create the knowledge unit
    return KnowledgeUnit(
        unique_id=unique_id,
        original_chunk=content,
        contextual_text=context,
        embedding_vector=embedding,
        knowledge_source=source,
        timestamp=timestamp,  # Use datetime object directly
        metadata=metadata or {},
        tags=tags or [],
    )


def create_sample_knowledge_units(count: int = 5) -> list[KnowledgeUnit]:
    """
    Create a list of sample knowledge units for testing.

    Args:
        count: Number of knowledge units to create

    Returns:
        List[KnowledgeUnit]: A list of test knowledge units
    """
    # Sample content templates
    content_templates = [
        "This is sample content about {topic}.",
        "{topic} is an important concept in computer science.",
        "Understanding {topic} is essential for AI development.",
        "The implementation of {topic} requires careful consideration.",
        "Recent advances in {topic} have revolutionized the field.",
    ]

    # Sample topics
    topics = [
        "machine learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "deep learning",
        "artificial intelligence",
        "data science",
        "knowledge representation",
        "human-computer interaction",
    ]

    # Sample sources
    sources = ["corpus", "action", "feedback"]

    # Sample tags
    tag_sets = [
        ["ai", "ml"],
        ["nlp", "language"],
        ["vision", "perception"],
        ["reinforcement", "learning"],
        ["neural", "networks"],
        ["deep", "learning"],
        ["ai", "intelligence"],
        ["data", "science"],
        ["knowledge", "representation"],
        ["hci", "interaction"],
    ]

    # Generate the requested number of knowledge units
    result = []
    for i in range(count):
        topic = random.choice(topics)
        template = random.choice(content_templates)
        content = template.format(topic=topic)

        context = f"Additional context about {topic}."
        source = random.choice(sources)
        tags = tag_sets[i % len(tag_sets)]

        # Create a metadata dictionary
        metadata = {
            "importance": random.randint(1, 5),
            "domain": topic.split()[0],
            "complexity": random.choice(["low", "medium", "high"]),
        }

        # Create a unique ID with a deterministic component for testing
        unique_id = f"test-{i}-{uuid.uuid4()}"

        # Create the knowledge unit and add it to the result
        ku = create_test_knowledge_unit(
            content=content,
            source=source,
            context=context,
            metadata=metadata,
            tags=tags,
            unique_id=unique_id,
        )
        result.append(ku)

    return result


def generate_random_text(length: int = 100) -> str:
    """
    Generate random text of specified length for testing.

    Args:
        length: Length of the text to generate

    Returns:
        str: Random text
    """
    words = []
    chars_left = length

    # Common words for more realistic text
    common_words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
    ]

    while chars_left > 0:
        if chars_left < 5:  # For very small remaining length
            word = "".join(random.choices(string.ascii_lowercase, k=chars_left))
        else:
            word = random.choice(common_words)
            # Add some punctuation occasionally
            if random.random() < 0.1:
                word += random.choice([",", ".", ":", ";", "!"])

        if len(word) <= chars_left:
            words.append(word)
            chars_left -= len(word) + 1  # +1 for space
        else:
            break

    return " ".join(words)


async def assert_knowledge_unit_equal(
    ku1: KnowledgeUnit,
    ku2: KnowledgeUnit,
    check_embedding: bool = True,
    check_timestamp: bool = True,
    check_metadata: bool = True,
    check_tags: bool = True,
) -> None:
    """
    Assert that two KnowledgeUnit objects are equal.

    This utility function makes it easier to test equality between two
    KnowledgeUnit objects with customizable strictness.

    Args:
        ku1: First KnowledgeUnit object
        ku2: Second KnowledgeUnit object
        check_embedding: Whether to check if embeddings are equal
        check_timestamp: Whether to check if timestamps are equal
        check_metadata: Whether to check if metadata is equal
        check_tags: Whether to check if tags are equal

    Raises:
        AssertionError: If the two KnowledgeUnit objects are not equal
    """
    # Check required fields
    assert ku1.unique_id == ku2.unique_id, f"Unique IDs differ: {ku1.unique_id} != {ku2.unique_id}"
    assert (
        ku1.original_chunk == ku2.original_chunk
    ), f"Original chunks differ: {ku1.original_chunk} != {ku2.original_chunk}"
    assert (
        ku1.contextual_text == ku2.contextual_text
    ), f"Contextual texts differ: {ku1.contextual_text} != {ku2.contextual_text}"
    assert (
        ku1.knowledge_source == ku2.knowledge_source
    ), f"Knowledge sources differ: {ku1.knowledge_source} != {ku2.knowledge_source}"

    # Check embedding if required
    if check_embedding:
        # Handle None values properly
        if ku1.embedding_vector is None and ku2.embedding_vector is None:
            # Both are None, which is valid
            pass
        elif ku1.embedding_vector is None or ku2.embedding_vector is None:
            # One is None but the other isn't
            assert (
                False
            ), f"One embedding is None but the other isn't: {ku1.embedding_vector} != {ku2.embedding_vector}"
        else:
            # Both have values, compare them
            assert len(ku1.embedding_vector) == len(
                ku2.embedding_vector
            ), f"Embedding lengths differ: {len(ku1.embedding_vector)} != {len(ku2.embedding_vector)}"
            for i, (v1, v2) in enumerate(
                zip(ku1.embedding_vector, ku2.embedding_vector, strict=False)
            ):
                assert (
                    abs(v1 - v2) < 1e-6
                ), f"Embedding values at index {i} differ significantly: {v1} != {v2}"

    # Check timestamp if required
    if check_timestamp:
        # Timestamps are timezone-aware datetime objects
        # Compare them with a small tolerance to account for microsecond differences
        time_diff = abs((ku1.timestamp - ku2.timestamp).total_seconds())
        assert time_diff < 0.001, f"Timestamps differ: {ku1.timestamp} != {ku2.timestamp}"

    # Check metadata if required
    if check_metadata:
        assert ku1.metadata == ku2.metadata, f"Metadata differs: {ku1.metadata} != {ku2.metadata}"

    # Check tags if required
    if check_tags:
        assert set(ku1.tags) == set(ku2.tags), f"Tags differ: {ku1.tags} != {ku2.tags}"
