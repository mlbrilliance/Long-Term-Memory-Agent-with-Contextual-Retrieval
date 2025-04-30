"""
Focused test for the temporal retrieval functionality.

This module contains a simplified test that focuses specifically on
temporal retrieval to verify it's working correctly with timezone-aware datetimes.
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


class MockMemoryManager:
    """Simple mock memory manager for temporal retrieval testing."""

    def __init__(self):
        """Initialize with time-ordered test data."""
        now = datetime.datetime.now(datetime.timezone.utc)

        # Create test knowledge units with different timestamps
        self.units = []
        for i in range(5):
            # Use valid sources
            source = ["corpus", "action", "feedback"][i % 3]

            # Create a timestamp with proper timezone info
            timestamp = now - datetime.timedelta(days=i)

            # Create and add the knowledge unit
            unit = KnowledgeUnit(
                unique_id=f"test-{i}",
                original_chunk=f"Test content from {i} days ago",
                contextual_text="",
                embedding_vector=[0.1] * 128,  # Simple embedding for testing
                knowledge_source=source,
                timestamp=timestamp,
                metadata={},
                tags=[],
            )
            self.units.append(unit)

    async def list_knowledge(
        self, limit=100, offset=0, source_filter=None, sort_by="timestamp", sort_order="desc"
    ):
        """Return knowledge units sorted by timestamp."""
        # Filter by source if needed
        filtered = self.units
        if source_filter:
            filtered = [u for u in filtered if u.knowledge_source == source_filter]

        # Sort by timestamp
        if sort_by == "timestamp":
            reverse = sort_order == "desc"
            filtered = sorted(filtered, key=lambda u: u.timestamp, reverse=reverse)

        # Apply offset and limit
        return filtered[offset : offset + limit]

    # Mock other required methods that won't be used in this test
    async def search_knowledge(self, *args, **kwargs):
        return []

    async def get_related_knowledge(self, *args, **kwargs):
        return []


class TestTemporalRetrieval:
    """Test suite focused on temporal retrieval."""

    @pytest.fixture
    async def retriever(self):
        """Provide a ContextRetriever with our mock manager."""
        manager = MockMemoryManager()
        retriever = ContextRetriever(memory_manager=manager, max_context_items=10)
        return retriever

    @pytest.mark.asyncio
    async def test_temporal_retrieval_works(self, retriever):
        """Test that temporal retrieval returns time-ordered results."""
        # Get temporal results
        results = await retriever.retrieve_context(
            query="",
            strategy="temporal",  # Not used in temporal retrieval
        )

        # Verify we got results
        assert len(results) > 0, "No results returned from temporal retrieval"
        print(f"Retrieved {len(results)} results from temporal retrieval")

        # Verify each result has the expected structure
        for item in results:
            assert "content" in item, "Missing content field"
            assert "timestamp" in item, "Missing timestamp field"
            assert "source" in item, "Missing source field"
            assert "relevance" in item, "Missing relevance field"

        # Print the first couple of results for debugging
        for i, item in enumerate(results[:2]):
            print(f"Result {i}:")
            print(f"  Content: {item.get('content')}")
            print(f"  Timestamp: {item.get('timestamp')}")

        # Verify sorting by timestamp (most recent first)
        if len(results) > 1:
            # Parse ISO timestamps for comparison
            first_timestamp = datetime.datetime.fromisoformat(
                results[0]["timestamp"].replace("Z", "+00:00")
            )
            second_timestamp = datetime.datetime.fromisoformat(
                results[1]["timestamp"].replace("Z", "+00:00")
            )

            # Check that the first timestamp is more recent than the second
            assert first_timestamp >= second_timestamp, "Results not in descending timestamp order"
