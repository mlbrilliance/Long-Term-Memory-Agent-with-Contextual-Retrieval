"""
End-to-end integration tests for the LTM Agent.

This module tests the entire system, integrating all components
to verify they work together correctly as a cohesive whole.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.llm.anthropic_client import AnthropicClient
from ltm_agent.memory.batch import BatchProcessor
from ltm_agent.memory.manager import MemoryManager
from ltm_agent.memory.sqlite_store import SQLiteVectorStore
from ltm_agent.retrieval.context_retriever import ContextRetriever, PromptBuilder
from ltm_agent.utils.test_utils import (
    create_test_knowledge_unit,
    get_test_settings,
)


class TestFullSystem:
    """Integration tests for the full LTM Agent system."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary file path for database testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        yield db_path

        # Clean up the file after the test
        try:
            os.unlink(db_path)
        except Exception as e:
            print(f"Failed to clean up temporary database file: {str(e)}")

    @pytest.fixture
    def settings(self):
        """Fixture providing test settings."""
        return get_test_settings()

    @pytest.fixture
    async def vector_store(self, temp_db_path):
        """Fixture providing an initialized SQLiteVectorStore."""
        store = SQLiteVectorStore(database_path=temp_db_path, embedding_dim=128)
        await store.initialize()
        return store

    @pytest.fixture
    async def memory_manager(self, vector_store, settings):
        """Fixture providing a MemoryManager with SQLite storage."""
        manager = MemoryManager(memory_store=vector_store, config=settings)
        await manager.initialize()
        return manager

    @pytest.fixture
    async def populated_memory_manager(self, memory_manager):
        """Fixture providing a MemoryManager with sample data."""
        # Add sample knowledge units
        batch_processor = BatchProcessor(memory_manager)

        contents = [
            "Python is a high-level programming language known for its readability and versatility.",
            "TensorFlow is a machine learning framework developed by Google.",
            "PyTorch is another popular deep learning framework, often preferred for research.",
            "Natural Language Processing (NLP) involves the interaction between computers and human language.",
            "The transformer architecture revolutionized NLP with its attention mechanism.",
            "Large Language Models (LLMs) are neural networks trained on vast amounts of text data.",
            "Fine-tuning is the process of adapting a pre-trained model to a specific task.",
            "Claude is an AI assistant developed by Anthropic, designed to be helpful, harmless, and honest.",
            "Contextual retrieval enhances LLM responses by providing relevant information from a knowledge base.",
            "Vector embeddings convert text into numerical representations that capture semantic meaning.",
        ]

        contexts = [
            "Python was created by Guido van Rossum in the late 1980s.",
            "TensorFlow offers both high and low-level APIs for building machine learning models.",
            "PyTorch is developed by Facebook's AI Research lab and is known for its dynamic computation graph.",
            "NLP applications include machine translation, sentiment analysis, and question answering.",
            "Attention mechanisms allow models to focus on different parts of the input when generating output.",
            "Examples of LLMs include GPT-4, Claude, and LLaMA.",
            "Fine-tuning typically requires less data than training from scratch.",
            "Claude is designed to be more conversational and provide nuanced responses.",
            "Retrieval-augmented generation combines LLMs with external knowledge retrieval.",
            "Cosine similarity is often used to measure the similarity between vector embeddings.",
        ]

        # Assign appropriate sources and tags
        sources = ["corpus"] * 5 + ["action"] * 3 + ["feedback"] * 2

        tags_list = [
            ["python", "programming"],
            ["tensorflow", "ml", "framework"],
            ["pytorch", "ml", "framework"],
            ["nlp", "ai"],
            ["transformer", "nlp", "architecture"],
            ["llm", "ai", "models"],
            ["fine-tuning", "ml", "training"],
            ["claude", "anthropic", "assistant"],
            ["contextual-retrieval", "knowledge-base"],
            ["embeddings", "vector", "representation"],
        ]

        # Create metadata
        metadata_list = [
            {"importance": (i % 3) + 1, "domain": "ai" if i > 2 else "programming"}
            for i in range(10)
        ]

        # Add in batch
        await batch_processor.add_knowledge_batch(
            contents=contents,
            source="corpus",
            contexts=contexts,
            metadata_list=metadata_list,
            tags_list=tags_list,
        )

        return memory_manager

    @pytest.fixture
    def context_retriever(self, populated_memory_manager):
        """Fixture providing a ContextRetriever with sample data."""
        return ContextRetriever(
            memory_manager=populated_memory_manager,
            max_context_items=5,
            relevance_threshold=0.5,
            max_token_limit=2000,
        )

    @pytest.fixture
    def mock_anthropic_client(self):
        """Fixture providing a mocked AnthropicClient."""
        # Create a real client with a test API key
        client = AnthropicClient(api_key="test-api-key")

        # Mock the generate_response method
        async def mock_generate_response(*args, **kwargs):
            return {
                "content": [{"text": "This is a test response from the mocked Claude API."}],
                "model": "claude-2",
                "stop_reason": "end_turn",
            }

        # Replace the real method with our mock
        client.generate_response = AsyncMock(side_effect=mock_generate_response)

        # Mock the generate_with_context method
        async def mock_generate_with_context(*args, **kwargs):
            context_items = kwargs.get("context_items", [])
            # Add the context items to the response for verification
            return {
                "content": "This is a test response that incorporates relevant context.",
                "model": "claude-2",
                "stop_reason": "end_turn",
                "context_items_used": len(context_items),
            }

        # Replace the real method with our mock
        client.generate_with_context = AsyncMock(side_effect=mock_generate_with_context)

        return client

    @pytest.mark.asyncio
    async def test_memory_store_integration(self, vector_store):
        """Test basic operations on the SQLite vector store."""
        # Create and add a knowledge unit
        ku = create_test_knowledge_unit(
            content="Integration test content",
            source="corpus",
            context="Integration test context",
            metadata={"test": True},
            tags=["integration", "test"],
        )

        unique_id = await vector_store.add(ku)

        # Retrieve it
        retrieved_ku = await vector_store.get(unique_id)

        # Verify it matches
        assert retrieved_ku is not None
        assert retrieved_ku.unique_id == ku.unique_id
        assert retrieved_ku.original_chunk == ku.original_chunk
        assert retrieved_ku.contextual_text == ku.contextual_text
        assert retrieved_ku.metadata == ku.metadata
        assert sorted(retrieved_ku.tags) == sorted(ku.tags)

        # Verify embedding was generated
        assert retrieved_ku.embedding_vector is not None
        assert len(retrieved_ku.embedding_vector) == vector_store.embedding_dim

    @pytest.mark.asyncio
    async def test_memory_manager_integration(self, memory_manager):
        """Test the MemoryManager integration with storage."""
        # Add knowledge through the manager
        unique_id = await memory_manager.add_knowledge(
            content="Memory manager integration test",
            source="action",
            context="Testing the memory manager",
            metadata={"test_type": "integration"},
        )

        # Retrieve it
        ku = await memory_manager.retrieve_knowledge(unique_id)

        # Verify it matches
        assert ku is not None
        assert ku.original_chunk == "Memory manager integration test"
        assert ku.contextual_text == "Testing the memory manager"
        assert ku.knowledge_source == "action"
        assert ku.metadata == {"test_type": "integration"}

        # Update it
        result = await memory_manager.update_knowledge(
            unique_id=unique_id, content="Updated content", context="Updated context"
        )
        assert result is True

        # Verify update
        updated_ku = await memory_manager.retrieve_knowledge(unique_id)
        assert updated_ku.original_chunk == "Updated content"
        assert updated_ku.contextual_text == "Updated context"

        # Delete it
        result = await memory_manager.delete_knowledge(unique_id)
        assert result is True

        # Verify deletion
        deleted_ku = await memory_manager.retrieve_knowledge(unique_id)
        assert deleted_ku is None

    @pytest.mark.asyncio
    async def test_search_and_retrieval(self, populated_memory_manager):
        """Test search and retrieval operations."""
        # Test keyword search
        search_results = await populated_memory_manager.search_knowledge(
            query="Python programming", limit=3
        )

        # Should find Python-related content
        assert len(search_results) > 0
        python_found = False
        for ku, score in search_results:
            if "Python" in ku.original_chunk:
                python_found = True
                break
        assert python_found

        # Test filtered search
        filtered_results = await populated_memory_manager.search_knowledge(
            query="language model", limit=5, source_filter="action"
        )

        # Should only include action source
        for ku, _ in filtered_results:
            assert ku.knowledge_source == "action"

    @pytest.mark.asyncio
    async def test_batch_processing(self, memory_manager):
        """Test batch processing operations."""
        # Create batch processor
        batch_processor = BatchProcessor(memory_manager, batch_size=3)

        # Test batch add
        contents = ["Batch item 1", "Batch item 2", "Batch item 3", "Batch item 4"]
        contexts = ["Context 1", "Context 2", "Context 3", "Context 4"]

        ids = await batch_processor.add_knowledge_batch(
            contents=contents, source="corpus", contexts=contexts, tags_list=[["batch", "test"]] * 4
        )

        # Verify all were added
        assert len(ids) == 4
        assert all(id is not None for id in ids)

        # Test batch update
        new_contents = ["Updated 1", "Updated 2", "Updated 3", "Updated 4"]

        results = await batch_processor.update_knowledge_batch(
            unique_ids=ids, contents=new_contents
        )

        # Verify all were updated
        assert len(results) == 4
        assert all(results)

        # Verify updates
        for i, id in enumerate(ids):
            ku = await memory_manager.retrieve_knowledge(id)
            assert ku.original_chunk == new_contents[i]
            assert "batch" in ku.tags

        # Test batch delete
        delete_results = await batch_processor.delete_knowledge_batch(ids)

        # Verify all were deleted
        assert len(delete_results) == 4
        assert all(delete_results)

    @pytest.mark.asyncio
    async def test_context_retrieval(self, context_retriever, populated_memory_manager):
        """Test context retrieval for an LLM query."""
        # Retrieve context for a query
        context_items = await context_retriever.retrieve_context(
            query="How do large language models work with vector embeddings?", strategy="hybrid"
        )

        # Verify results include relevant items
        assert len(context_items) > 0

        # Should include content about LLMs and embeddings
        llm_found = False
        embedding_found = False

        for item in context_items:
            if "LLM" in item["content"] or "Large Language Models" in item["content"]:
                llm_found = True
            if "embedding" in item["content"].lower():
                embedding_found = True

        assert llm_found
        assert embedding_found

        # Check structure of context items
        for item in context_items:
            assert "content" in item
            assert "source" in item
            assert "relevance" in item
            assert "id" in item

    @pytest.mark.asyncio
    async def test_prompt_building(self, context_retriever, populated_memory_manager):
        """Test building prompts with retrieved context."""
        # Get context items
        context_items = await context_retriever.retrieve_context(
            query="How do transformers work in NLP?", strategy="semantic"
        )

        # Create a prompt builder
        prompt_builder = PromptBuilder(
            system_prompt_template="You are a helpful AI assistant. Answer based on this context: "
        )

        # Build a prompt
        prompt_data = prompt_builder.build_prompt(
            query="Explain transformer architecture in simple terms", context_items=context_items
        )

        # Verify prompt structure
        assert "messages" in prompt_data
        assert len(prompt_data["messages"]) == 2  # System and user

        system_message = prompt_data["messages"][0]["content"]
        user_message = prompt_data["messages"][1]["content"]

        # System message should contain the retrieved context
        assert "transformer" in system_message.lower()
        assert "attention" in system_message.lower()

        # User message should be the query
        assert user_message == "Explain transformer architecture in simple terms"

    @pytest.mark.asyncio
    async def test_end_to_end_flow(
        self, context_retriever, populated_memory_manager, mock_anthropic_client
    ):
        """Test the complete end-to-end flow from query to response."""
        # Create a client using our mock
        client = mock_anthropic_client

        # 1. Retrieve context for a query
        context_items = await context_retriever.retrieve_context(
            query="How do large language models use contextual retrieval?", strategy="hybrid"
        )

        # 2. Generate a response using the context
        response = await client.generate_with_context(
            user_query="How do large language models use contextual retrieval?",
            context_items=context_items,
            system_prompt="You are a helpful AI assistant specialized in explaining AI concepts.",
        )

        # Verify the response
        assert "content" in response
        assert isinstance(response["content"], str)
        assert "model" in response

        # 3. Add the interaction to memory
        interaction_id = await populated_memory_manager.add_knowledge(
            content=f"Q: How do large language models use contextual retrieval?\nA: {response['content']}",
            source="feedback",
            metadata={"interaction_type": "question_answer"},
        )

        # Verify the interaction was added
        interaction = await populated_memory_manager.retrieve_knowledge(interaction_id)
        assert interaction is not None
        assert "contextual retrieval" in interaction.original_chunk
        assert interaction.knowledge_source == "feedback"
