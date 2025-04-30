"""
Debug script for temporal retrieval issues.

This script bypasses the pytest framework to directly test the temporal retrieval
functionality of the ContextRetriever.
"""

import asyncio
import datetime
import logging
import random
import sys
import traceback
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("debug_temporal")

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import necessary components
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.retrieval.context_retriever import ContextRetriever


class MockMemoryManager:
    """Mock memory manager for testing context retrieval."""

    def __init__(self, sample_units=None):
        # Create test knowledge units with different timestamps
        now = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(f"Current time: {now}")

        # Create knowledge units with different timestamps
        self.sample_units = []
        for i in range(5):
            # Use valid knowledge source values: 'action', 'feedback', or 'corpus'
            sources = ["action", "feedback", "corpus"]
            source = sources[i % len(sources)]

            # Create timestamp with timezone info
            timestamp = now - datetime.timedelta(days=i)

            # Cannot directly pass timestamp, so create a custom KnowledgeUnit
            unit = KnowledgeUnit(
                unique_id=f"test-{i}-{uuid.uuid4()}",
                original_chunk=f"Test content from {i} days ago",
                contextual_text="",
                embedding_vector=[random.uniform(-1.0, 1.0) for _ in range(128)],
                knowledge_source=source,
                timestamp=timestamp,  # Pass datetime object directly
                metadata={},
                tags=[],
            )
            self.sample_units.append(unit)
            logger.debug(
                f"Created unit with timestamp: {unit.timestamp}, type: {type(unit.timestamp)}"
            )

        # Print the created units for debugging
        print("=== Sample Units Created ===")
        for i, unit in enumerate(self.sample_units):
            print(f"Unit {i}:")
            print(f"  Content: {unit.original_chunk}")
            print(f"  Source: {unit.knowledge_source}")
            print(f"  Timestamp: {unit.timestamp}")
            print(f"  Timestamp type: {type(unit.timestamp)}")

    async def list_knowledge(
        self, limit=100, offset=0, source_filter=None, sort_by="timestamp", sort_order="desc"
    ):
        """Mock list_knowledge method."""
        logger.debug(f"list_knowledge called with sort_by={sort_by}, sort_order={sort_order}")

        # Filter by source if specified
        filtered_units = self.sample_units
        if source_filter:
            filtered_units = [ku for ku in filtered_units if ku.knowledge_source == source_filter]

        # Sort by timestamp
        if sort_by == "timestamp":
            reverse = sort_order == "desc"

            # Make sure all timestamps are datetime objects
            def get_timestamp(ku):
                if isinstance(ku.timestamp, str):
                    try:
                        return datetime.datetime.fromisoformat(ku.timestamp)
                    except ValueError:
                        logger.warning(f"Failed to parse timestamp: {ku.timestamp}")
                        return datetime.datetime.min
                return ku.timestamp

            filtered_units = sorted(filtered_units, key=get_timestamp, reverse=reverse)

            # Debug output
            logger.debug(f"Sorted units by timestamp ({sort_order}):")
            for i, unit in enumerate(filtered_units):
                logger.debug(f"  {i}: {unit.timestamp}")

        # Apply offset and limit
        return filtered_units[offset : offset + limit]

    # Mock other required methods that might be called
    async def search_knowledge(self, *args, **kwargs):
        logger.debug("search_knowledge called (mock returning empty list)")
        return []

    async def get_related_knowledge(self, *args, **kwargs):
        logger.debug("get_related_knowledge called (mock returning empty list)")
        return []


async def main():
    """Main function to test temporal retrieval."""
    try:
        print("\n=== Creating test components ===")

        # Create mock memory manager
        memory_manager = MockMemoryManager()

        # Create context retriever
        context_retriever = ContextRetriever(memory_manager=memory_manager, max_context_items=10)
        print("Context retriever created")

        # Get the internal _retrieve_temporal method
        retrieve_temporal = context_retriever._retrieve_temporal

        # Print the method definition
        print(f"\nRetrieve temporal method: {retrieve_temporal}")

        # Test temporal retrieval directly first
        print("\n=== Testing _retrieve_temporal directly ===")
        try:
            direct_results = await retrieve_temporal()
            print(f"Direct retrieval results: {len(direct_results)}")
            for i, item in enumerate(direct_results):
                print(f"Result {i}:")
                print(f"  Content: {item.get('content', 'N/A')}")
                print(f"  Timestamp: {item.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"ERROR in direct retrieval: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

        # Test via the public API
        print("\n=== Testing retrieve_context with temporal strategy ===")
        try:
            results = await context_retriever.retrieve_context(
                query="",
                strategy="temporal",  # Query doesn't matter for temporal
            )

            # Print results
            print(f"Retrieved {len(results)} results")
            for i, item in enumerate(results):
                print(f"Result {i}:")
                print(f"  Content: {item.get('content', 'N/A')}")
                print(f"  Source: {item.get('source', 'N/A')}")
                print(f"  Timestamp: {item.get('timestamp', 'N/A')}")

            print("Test completed successfully")
        except Exception as e:
            print(f"ERROR in retrieve_context: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"ERROR in main: {type(e).__name__}: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())
