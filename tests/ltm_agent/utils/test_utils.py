"""
Test utilities for the LTM Agent tests.

This module provides common utilities and fixtures for testing
the LTM Agent components.
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings
from ltm_agent.core.models import KnowledgeUnit


# Test data generators
def create_test_knowledge_unit(
    content: str = "Test content",
    source: str = "action",
    context: str = "",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    """
    Create a test knowledge unit with the specified parameters.

    Args:
        content: Content for the knowledge unit
        source: Source of the knowledge
        context: Optional context
        metadata: Optional metadata
        tags: Optional tags

    Returns:
        KnowledgeUnit: A test knowledge unit
    """
    return KnowledgeUnit(
        original_chunk=content,
        contextual_text=context,
        knowledge_source=source,
        metadata=metadata or {},
        tags=tags or [],
    )


def create_sample_knowledge_units(count: int = 10) -> list[KnowledgeUnit]:
    """
    Create a list of sample knowledge units for testing.

    Args:
        count: Number of knowledge units to create

    Returns:
        List[KnowledgeUnit]: List of sample knowledge units
    """
    sources = ["action", "feedback", "corpus"]
    units = []

    for i in range(count):
        source = sources[i % len(sources)]
        content = f"Sample content {i + 1} for testing with source {source}"
        context = f"Context for sample {i + 1}" if i % 2 == 0 else ""
        metadata = {"test_id": i, "importance": i % 3 + 1} if i % 3 == 0 else {}
        tags = [f"tag{i}", "test"] if i % 2 == 0 else []

        unit = create_test_knowledge_unit(
            content=content, source=source, context=context, metadata=metadata, tags=tags
        )

        units.append(unit)

    return units


def get_test_settings() -> Settings:
    """
    Get test settings with mock API keys.

    Returns:
        Settings: Test settings
    """
    # Set required environment variables for testing
    os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"
    os.environ["PINECONE_API_KEY"] = "test_pinecone_key"
    os.environ["PINECONE_ENVIRONMENT"] = "test_env"
    os.environ["PINECONE_INDEX"] = "test_index"

    return Settings.from_env()


# Assert helpers
async def assert_knowledge_unit_equal(ku1: KnowledgeUnit, ku2: KnowledgeUnit) -> None:
    """
    Assert that two knowledge units are equal.

    Args:
        ku1: First knowledge unit
        ku2: Second knowledge unit
    """
    assert ku1.unique_id == ku2.unique_id
    assert ku1.original_chunk == ku2.original_chunk
    assert ku1.contextual_text == ku2.contextual_text
    assert ku1.knowledge_source == ku2.knowledge_source
    # Timestamps might have microsecond differences, so compare only up to seconds
    assert ku1.timestamp.replace(microsecond=0) == ku2.timestamp.replace(microsecond=0)
    assert ku1.metadata == ku2.metadata
    assert sorted(ku1.tags) == sorted(ku2.tags)

    if ku1.embedding_vector is not None and ku2.embedding_vector is not None:
        assert len(ku1.embedding_vector) == len(ku2.embedding_vector)
        # Compare embeddings approximately
        for v1, v2 in zip(ku1.embedding_vector, ku2.embedding_vector, strict=False):
            assert abs(v1 - v2) < 1e-5
    else:
        assert ku1.embedding_vector == ku2.embedding_vector


# Test fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Provide a temporary file path for database testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name

    yield db_path

    # Clean up the file after the test
    try:
        os.unlink(db_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary database file: {str(e)}")
