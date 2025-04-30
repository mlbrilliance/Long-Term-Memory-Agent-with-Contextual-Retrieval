"""
Tests for batch operations and processors.

This module tests the batch processing functionality to ensure
efficient bulk operations on knowledge units.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.batch import BatchProcessor, process_in_batches


class MockMemoryManager:
    """Mock memory manager for testing batch operations."""

    def __init__(self):
        self.store = {}
        self.add_calls = 0
        self.update_calls = 0
        self.delete_calls = 0
        self.retrieve_calls = 0

    async def add_knowledge(
        self,
        content: str,
        source: str,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Mock add_knowledge method."""
        self.add_calls += 1
        ku = KnowledgeUnit(
            original_chunk=content,
            contextual_text=context,
            knowledge_source=source,
            metadata=metadata or {},
        )
        self.store[ku.unique_id] = ku
        return ku.unique_id

    async def update_knowledge(
        self,
        unique_id: str,
        content: str | None = None,
        context: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Mock update_knowledge method."""
        self.update_calls += 1
        if unique_id not in self.store:
            return False

        ku = self.store[unique_id]
        if content is not None:
            ku.original_chunk = content
        if context is not None:
            ku.contextual_text = context
        if source is not None:
            ku.knowledge_source = source

        return True

    async def delete_knowledge(self, unique_id: str) -> bool:
        """Mock delete_knowledge method."""
        self.delete_calls += 1
        if unique_id in self.store:
            del self.store[unique_id]
            return True
        return False

    async def retrieve_knowledge(self, unique_id: str) -> KnowledgeUnit | None:
        """Mock retrieve_knowledge method."""
        self.retrieve_calls += 1
        return self.store.get(unique_id)


class TestBatchProcessor:
    """Test suite for the BatchProcessor."""

    @pytest.fixture
    def memory_manager(self):
        """Fixture providing a mock memory manager."""
        return MockMemoryManager()

    @pytest.fixture
    def batch_processor(self, memory_manager):
        """Fixture providing a BatchProcessor with mock manager."""
        return BatchProcessor(memory_manager, batch_size=2)

    @pytest.mark.asyncio
    async def test_add_knowledge_batch(self, batch_processor, memory_manager):
        """Test adding knowledge in batches."""
        # Prepare test data
        contents = ["Content 1", "Content 2", "Content 3"]
        contexts = ["Context 1", "Context 2", "Context 3"]
        metadata_list = [{"id": 1}, {"id": 2}, {"id": 3}]
        tags_list = [["tag1"], ["tag2"], ["tag3"]]

        # Add in batch
        result_ids = await batch_processor.add_knowledge_batch(
            contents=contents,
            source="test",
            contexts=contexts,
            metadata_list=metadata_list,
            tags_list=tags_list,
        )

        # Verify results
        assert len(result_ids) == 3
        assert all(id is not None for id in result_ids)

        # Should have made 3 add calls and 3 retrieve calls for tag updates
        assert memory_manager.add_calls == 3
        assert memory_manager.retrieve_calls == 3

        # Verify contents in store
        for i, content in enumerate(contents):
            ku = memory_manager.store[result_ids[i]]
            assert ku.original_chunk == content
            assert ku.contextual_text == contexts[i]
            assert ku.metadata == metadata_list[i]

    @pytest.mark.asyncio
    async def test_update_knowledge_batch(self, batch_processor, memory_manager):
        """Test updating knowledge in batches."""
        # Add test units first
        ids = []
        for i in range(3):
            id = await memory_manager.add_knowledge(content=f"Original {i}", source="test")
            ids.append(id)

        # Reset call counters
        memory_manager.add_calls = 0
        memory_manager.update_calls = 0
        memory_manager.retrieve_calls = 0

        # Prepare update data
        contents = ["Updated 0", "Updated 1", "Updated 2"]
        contexts = ["New context 0", "New context 1", "New context 2"]

        # Update in batch
        results = await batch_processor.update_knowledge_batch(
            unique_ids=ids, contents=contents, contexts=contexts
        )

        # Verify results
        assert len(results) == 3
        assert all(results)

        # Should have made 3 retrieve calls and 3 update calls
        assert memory_manager.retrieve_calls == 3
        assert memory_manager.update_calls == 3

        # Verify updates in store
        for i, id in enumerate(ids):
            ku = memory_manager.store[id]
            assert ku.original_chunk == contents[i]
            assert ku.contextual_text == contexts[i]

    @pytest.mark.asyncio
    async def test_delete_knowledge_batch(self, batch_processor, memory_manager):
        """Test deleting knowledge in batches."""
        # Add test units first
        ids = []
        for i in range(3):
            id = await memory_manager.add_knowledge(content=f"Item {i}", source="test")
            ids.append(id)

        # Reset call counters
        memory_manager.add_calls = 0
        memory_manager.delete_calls = 0

        # Delete in batch
        results = await batch_processor.delete_knowledge_batch(ids)

        # Verify results
        assert len(results) == 3
        assert all(results)

        # Should have made 3 delete calls
        assert memory_manager.delete_calls == 3

        # Verify all items are gone
        assert len(memory_manager.store) == 0

    @pytest.mark.asyncio
    async def test_bulk_import_from_texts(self, batch_processor, memory_manager):
        """Test importing and chunking texts."""
        # Prepare test texts
        texts = [
            "This is a long text that should be chunked into multiple parts. " * 10,
            "Another text with different content for testing the chunking. " * 10,
        ]

        # Import with small chunk size to force splitting
        chunk_size = 100
        chunk_overlap = 20

        # Bulk import
        result_ids = await batch_processor.bulk_import_from_texts(
            texts=texts,
            source="import",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_metadata={"import_source": "test"},
            tags=["imported", "test"],
        )

        # Verify results
        assert len(result_ids) > 2  # Should have created multiple chunks
        assert all(id is not None for id in result_ids)

        # Verify chunks in store
        for id in result_ids:
            ku = memory_manager.store[id]
            # Each chunk should be less than or equal to chunk_size
            assert len(ku.original_chunk) <= chunk_size
            # Each chunk should have the metadata and source
            assert ku.metadata == {"import_source": "test"}
            assert ku.knowledge_source == "import"


class TestProcessInBatches:
    """Test suite for the process_in_batches utility function."""

    @pytest.mark.asyncio
    async def test_process_in_batches(self):
        """Test processing items in batches."""
        # Items to process
        items = list(range(10))

        # Mock batch processor function
        async def mock_processor(batch):
            return [item * 2 for item in batch]

        # Process in batches
        results = await process_in_batches(
            items=items, processor=mock_processor, batch_size=3, description="numbers"
        )

        # Verify results
        assert len(results) == 10
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    @pytest.mark.asyncio
    async def test_process_in_batches_error_handling(self):
        """Test error handling in batch processing."""
        # Items to process
        items = list(range(5))

        # Mock processor that fails on batch with item 3
        async def failing_processor(batch):
            if 3 in batch:
                raise ValueError("Error processing batch with item 3")
            return [item * 2 for item in batch]

        # Process with error
        with pytest.raises(Exception):
            await process_in_batches(
                items=items,
                processor=failing_processor,
                batch_size=2,
                description="numbers with error",
            )
