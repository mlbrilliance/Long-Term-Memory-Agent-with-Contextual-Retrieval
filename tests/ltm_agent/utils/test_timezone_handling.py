"""
Tests for timezone handling in the LTM Agent.

This module contains tests specifically focused on verifying
proper timezone handling throughout the system to prevent
regression of timezone-related issues.
"""

import datetime
import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.retrieval.context_retriever import ContextRetriever
from ltm_agent.utils.test_utils import create_test_knowledge_unit


class TestTimezoneHandling:
    """Test suite for timezone handling across the codebase."""

    def test_knowledge_unit_default_timestamp(self):
        """Test that KnowledgeUnit creates timezone-aware timestamps by default."""
        # Create a knowledge unit with default timestamp
        ku = KnowledgeUnit(
            unique_id="test-id",
            original_chunk="Test content",
            contextual_text="",
            embedding_vector=[0.1] * 128,
            knowledge_source="corpus",
        )

        # Verify timestamp is timezone-aware
        assert ku.timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
        assert ku.timestamp.tzinfo.tzname(ku.timestamp) == "UTC", "Timestamp should be in UTC"

    def test_test_utils_timestamp(self):
        """Test that create_test_knowledge_unit creates timezone-aware timestamps."""
        # Create a knowledge unit using the test utility
        ku = create_test_knowledge_unit(content="Test content", source="corpus")

        # Verify timestamp is timezone-aware
        assert (
            ku.timestamp.tzinfo is not None
        ), "Test utility should create timezone-aware timestamps"
        assert (
            ku.timestamp.tzinfo.tzname(ku.timestamp) == "UTC"
        ), "Test utility timestamps should be in UTC"

    def test_timestamp_serialization_consistency(self):
        """Test that timestamp serialization and deserialization maintains timezone info."""
        # Create a knowledge unit with a timezone-aware timestamp
        original_timestamp = datetime.datetime.now(datetime.timezone.utc)
        ku = KnowledgeUnit(
            unique_id="test-id",
            original_chunk="Test content",
            contextual_text="",
            embedding_vector=[0.1] * 128,
            knowledge_source="corpus",
            timestamp=original_timestamp,
        )

        # Serialize timestamp to ISO format
        iso_timestamp = ku.timestamp.isoformat()

        # Deserialize back to datetime
        deserialized_timestamp = datetime.datetime.fromisoformat(iso_timestamp)

        # Verify timezone information is preserved
        assert (
            deserialized_timestamp.tzinfo is not None
        ), "Deserialized timestamp should be timezone-aware"

        # Verify the deserialized timestamp matches the original
        assert deserialized_timestamp == original_timestamp, "Timestamp should round-trip correctly"

    def test_timestamp_comparison_consistency(self):
        """Test that timestamp comparisons work correctly across timezone-aware datetimes."""
        # Create timestamps with explicit timezone
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        one_hour_ago_utc = now_utc - datetime.timedelta(hours=1)

        # Verify comparison operators work as expected
        assert (
            now_utc > one_hour_ago_utc
        ), "Later timestamp should be greater than earlier timestamp"
        assert one_hour_ago_utc < now_utc, "Earlier timestamp should be less than later timestamp"

        # Test with different timezone offset but same actual time
        offset = datetime.timezone(datetime.timedelta(hours=-5))  # EST timezone
        now_est = now_utc.astimezone(offset)

        # Verify same actual time compares as equal regardless of timezone
        assert now_utc == now_est, "Same time in different timezones should be equal"

    @pytest.mark.asyncio
    async def test_context_retriever_temporal_timezone_handling(self):
        """Test that ContextRetriever correctly handles timezones in temporal retrieval."""

        # Create a mock memory manager
        class MockMemoryManager:
            """Mock memory manager with timezone-aware timestamps."""

            def __init__(self):
                # Create knowledge units with different timestamps
                now = datetime.datetime.now(datetime.timezone.utc)
                self.units = [
                    KnowledgeUnit(
                        unique_id=f"test-{i}",
                        original_chunk=f"Test content from {i} days ago",
                        contextual_text="",
                        embedding_vector=[0.1] * 128,
                        knowledge_source="corpus",
                        timestamp=now - datetime.timedelta(days=i),
                    )
                    for i in range(5)
                ]

            async def list_knowledge(self, **kwargs):
                """Return knowledge units sorted by timestamp."""
                sort_by = kwargs.get("sort_by", "timestamp")
                sort_order = kwargs.get("sort_order", "desc")

                if sort_by == "timestamp":
                    reverse = sort_order == "desc"
                    return sorted(self.units, key=lambda u: u.timestamp, reverse=reverse)
                return self.units

            # Mock methods not used in this test
            async def search_knowledge(self, *args, **kwargs):
                return []

            async def get_related_knowledge(self, *args, **kwargs):
                return []

        # Create context retriever with mock manager
        memory_manager = MockMemoryManager()
        retriever = ContextRetriever(memory_manager=memory_manager)

        # Test temporal retrieval
        results = await retriever.retrieve_context(
            query="",
            strategy="temporal",  # Not used in temporal retrieval
        )

        # Verify results are returned and contain proper timezone information
        assert len(results) > 0, "Temporal retrieval should return results"

        # Verify timestamps are properly formatted
        for result in results:
            assert "timestamp" in result, "Results should include timestamps"
            # Timestamps in results are ISO strings, verify they can be parsed
            timestamp = datetime.datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
            assert timestamp.tzinfo is not None, "Parsed timestamp should be timezone-aware"

        # Verify sorting by comparing adjacent timestamps (most recent first)
        if len(results) > 1:
            for i in range(len(results) - 1):
                current = datetime.datetime.fromisoformat(
                    results[i]["timestamp"].replace("Z", "+00:00")
                )
                next_one = datetime.datetime.fromisoformat(
                    results[i + 1]["timestamp"].replace("Z", "+00:00")
                )
                assert (
                    current >= next_one
                ), "Results should be sorted by timestamp (most recent first)"
