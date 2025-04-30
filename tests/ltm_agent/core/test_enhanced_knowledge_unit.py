"""
Tests for the enhanced KnowledgeUnit model with metadata and tags.

This test suite verifies the functionality of the enhanced KnowledgeUnit model,
ensuring that metadata and tags are properly handled.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit


class TestEnhancedKnowledgeUnit:
    """Test suite for the enhanced KnowledgeUnit model."""

    def test_metadata_initialization(self):
        """Test that metadata can be initialized during creation."""
        metadata = {"source_url": "https://example.com", "priority": 5}
        ku = KnowledgeUnit(
            original_chunk="Test content with metadata",
            knowledge_source="corpus",
            metadata=metadata,
        )

        assert ku.metadata == metadata
        assert ku.get_metadata("source_url") == "https://example.com"
        assert ku.get_metadata("priority") == 5

    def test_default_metadata(self):
        """Test that metadata defaults to an empty dict if not provided."""
        ku = KnowledgeUnit(
            original_chunk="Test content without metadata", knowledge_source="action"
        )

        assert ku.metadata == {}
        assert ku.get_metadata("non_existent") is None
        assert ku.get_metadata("non_existent", "default") == "default"

    def test_update_metadata(self):
        """Test updating metadata fields."""
        ku = KnowledgeUnit(
            original_chunk="Test content for metadata updates", knowledge_source="feedback"
        )

        # Update metadata
        ku.update_metadata("category", "test")
        ku.update_metadata("importance", 3)

        assert ku.metadata == {"category": "test", "importance": 3}

        # Update existing field
        ku.update_metadata("importance", 5)
        assert ku.get_metadata("importance") == 5

    def test_tags_initialization(self):
        """Test that tags can be initialized during creation."""
        tags = ["important", "test", "documentation"]
        ku = KnowledgeUnit(
            original_chunk="Test content with tags", knowledge_source="corpus", tags=tags
        )

        assert sorted(ku.tags) == sorted(tags)
        assert ku.has_tag("important")
        assert ku.has_tag("test")
        assert not ku.has_tag("non_existent")

    def test_default_tags(self):
        """Test that tags default to an empty list if not provided."""
        ku = KnowledgeUnit(original_chunk="Test content without tags", knowledge_source="action")

        assert ku.tags == []
        assert not ku.has_tag("any_tag")

    def test_add_tag(self):
        """Test adding tags to a knowledge unit."""
        ku = KnowledgeUnit(
            original_chunk="Test content for tag operations", knowledge_source="feedback"
        )

        # Add tags
        ku.add_tag("important")
        ku.add_tag("test")

        assert sorted(ku.tags) == ["important", "test"]
        assert ku.has_tag("important")

        # Adding duplicate tag should have no effect
        ku.add_tag("important")
        assert len(ku.tags) == 2

    def test_remove_tag(self):
        """Test removing tags from a knowledge unit."""
        ku = KnowledgeUnit(
            original_chunk="Test content for tag removal",
            knowledge_source="corpus",
            tags=["tag1", "tag2", "tag3"],
        )

        # Remove a tag
        result = ku.remove_tag("tag2")
        assert result is True
        assert sorted(ku.tags) == ["tag1", "tag3"]

        # Try to remove a non-existent tag
        result = ku.remove_tag("non_existent")
        assert result is False
        assert sorted(ku.tags) == ["tag1", "tag3"]

    def test_tag_normalization(self):
        """Test that tags are normalized to lowercase and duplicates are removed."""
        # Test with a list of tags
        ku1 = KnowledgeUnit(
            original_chunk="Test content for tag normalization",
            knowledge_source="corpus",
            tags=["Tag1", "TAG2", "tag1", "  tag3  "],
        )

        assert sorted(ku1.tags) == ["tag1", "tag2", "tag3"]

        # Test with a comma-separated string
        ku2 = KnowledgeUnit(
            original_chunk="Test content for tag normalization",
            knowledge_source="corpus",
            tags="Tag1, TAG2, tag1,  tag3  ",
        )

        assert sorted(ku2.tags) == ["tag1", "tag2", "tag3"]

    def test_add_tag_normalization(self):
        """Test that tags are normalized when added."""
        ku = KnowledgeUnit(
            original_chunk="Test content for add tag normalization", knowledge_source="action"
        )

        ku.add_tag("TAG1")
        ku.add_tag("  Tag2  ")
        ku.add_tag("tag1")  # Should be ignored as duplicate

        assert sorted(ku.tags) == ["tag1", "tag2"]

    def test_has_tag_normalization(self):
        """Test that case is ignored when checking for tags."""
        ku = KnowledgeUnit(
            original_chunk="Test content for has_tag normalization",
            knowledge_source="feedback",
            tags=["tag1", "tag2"],
        )

        assert ku.has_tag("TAG1")
        assert ku.has_tag("  tag2  ")
        assert not ku.has_tag("tag3")

    def test_remove_tag_normalization(self):
        """Test that case is ignored when removing tags."""
        ku = KnowledgeUnit(
            original_chunk="Test content for remove_tag normalization",
            knowledge_source="corpus",
            tags=["tag1", "tag2"],
        )

        result = ku.remove_tag("TAG1")
        assert result is True
        assert ku.tags == ["tag2"]

        result = ku.remove_tag("  TAG2  ")
        assert result is True
        assert ku.tags == []
