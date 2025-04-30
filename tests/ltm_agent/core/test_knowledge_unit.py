"""
Tests for the KnowledgeUnit model.

This test suite verifies the functionality of the KnowledgeUnit model,
ensuring that it properly initializes with required fields, generates
default values for optional fields, and validates inputs correctly.
"""

import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit


class TestKnowledgeUnit:
    """Test suite for the KnowledgeUnit model."""

    def test_initialization_with_required_fields(self):
        """Test that KnowledgeUnit can be instantiated with just the required fields."""
        ku = KnowledgeUnit(original_chunk="This is a test chunk", knowledge_source="action")
        assert ku.original_chunk == "This is a test chunk"
        assert ku.knowledge_source == "action"
        assert ku.contextual_text == ""  # Default value
        assert ku.embedding_vector is None  # Default value

    def test_default_uuid_generation(self):
        """Test that unique_id is automatically generated as a UUID string."""
        ku = KnowledgeUnit(original_chunk="Test chunk", knowledge_source="feedback")
        # Check that unique_id is a valid UUID string
        assert isinstance(ku.unique_id, str)
        # UUID format validation (simple regex check)
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, ku.unique_id) is not None

    def test_default_timestamp_generation(self):
        """Test that timestamp is automatically set to current UTC time."""
        before = datetime.now(timezone.utc)
        ku = KnowledgeUnit(original_chunk="Test chunk", knowledge_source="corpus")
        after = datetime.now(timezone.utc)

        # Check that timestamp is a datetime object
        assert isinstance(ku.timestamp, datetime)
        # Check that timestamp is in UTC timezone
        assert ku.timestamp.tzinfo == timezone.utc
        # Check that timestamp is between before and after
        assert before <= ku.timestamp <= after

    def test_optional_fields(self):
        """Test that optional fields can be set during initialization."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        ku = KnowledgeUnit(
            original_chunk="Test chunk",
            knowledge_source="action",
            contextual_text="Additional context",
            embedding_vector=embedding,
        )

        assert ku.contextual_text == "Additional context"
        assert ku.embedding_vector == embedding

    def test_knowledge_source_validation(self):
        """Test that knowledge_source must be one of the allowed values."""
        # Valid values
        for source in ["action", "feedback", "corpus"]:
            ku = KnowledgeUnit(original_chunk="Test chunk", knowledge_source=source)
            assert ku.knowledge_source == source

        # Invalid value
        with pytest.raises(ValueError):
            KnowledgeUnit(original_chunk="Test chunk", knowledge_source="invalid_source")

    def test_custom_unique_id(self):
        """Test that a custom unique_id can be provided."""
        custom_id = str(uuid.uuid4())
        ku = KnowledgeUnit(
            original_chunk="Test chunk", knowledge_source="action", unique_id=custom_id
        )
        assert ku.unique_id == custom_id

    def test_custom_timestamp(self):
        """Test that a custom timestamp can be provided."""
        custom_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ku = KnowledgeUnit(
            original_chunk="Test chunk", knowledge_source="action", timestamp=custom_time
        )
        assert ku.timestamp == custom_time
