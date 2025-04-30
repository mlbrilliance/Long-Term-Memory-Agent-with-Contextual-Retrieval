"""
Tests for the context retrieval system.

This module tests the context retriever functionality to ensure proper
selection and formatting of relevant context for LLM queries.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.retrieval.context_retriever import ContextRetriever, PromptBuilder
from ltm_agent.utils.test_utils import create_sample_knowledge_units


class MockMemoryManager:
    """Mock memory manager for testing context retrieval."""

    def __init__(self, sample_units=None):
        self.sample_units = sample_units or create_sample_knowledge_units(10)

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        source_filter: str | None = None,
        min_relevance_score: float = 0.0,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Mock search_knowledge method."""
        # Filter by source if specified
        filtered_units = self.sample_units
        if source_filter:
            filtered_units = [ku for ku in filtered_units if ku.knowledge_source == source_filter]

        # Mock relevance scores based on simple word matching
        results = []
        for ku in filtered_units:
            # Simple relevance calculation - count word overlap
            query_words = set(query.lower().split())
            content_words = set(ku.original_chunk.lower().split())
            overlap = len(query_words.intersection(content_words))

            # Add a boost for exact matching of important terms
            term_boost = 0.0
            if "python" in query.lower() and "python" in ku.original_chunk.lower():
                term_boost += 0.3
            if (
                "machine learning" in query.lower()
                and "machine learning" in ku.original_chunk.lower()
            ):
                term_boost += 0.3

            # Final score calculation - ensure it's capped at 1.0
            score = min(1.0, 0.5 + (0.1 * overlap) + term_boost)

            if score >= min_relevance_score:
                results.append((ku, score))

        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_related_knowledge(
        self,
        content: str,
        limit: int = 10,
        source_filter: str | None = None,
        min_similarity_score: float = 0.0,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Mock get_related_knowledge method."""
        # For testing, just use the same implementation as search_knowledge
        return await self.search_knowledge(content, limit, source_filter, min_similarity_score)

    async def list_knowledge(
        self,
        limit: int = 100,
        offset: int = 0,
        source_filter: str | None = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> list[KnowledgeUnit]:
        """Mock list_knowledge method."""
        # Filter by source if specified
        filtered_units = self.sample_units
        if source_filter:
            filtered_units = [ku for ku in filtered_units if ku.knowledge_source == source_filter]

        # Sort by timestamp or other field
        if sort_by == "timestamp":
            reverse = sort_order == "desc"
            filtered_units = sorted(filtered_units, key=lambda ku: ku.timestamp, reverse=reverse)

        # Apply offset and limit
        return filtered_units[offset : offset + limit]


class TestContextRetriever:
    """Test suite for the ContextRetriever."""

    @pytest.fixture
    def memory_manager(self):
        """Fixture providing a mock memory manager with sample data."""
        # Create sample units with specific content for testing
        now = datetime.now(tz=timezone.utc)

        units = [
            # Create knowledge units with proper datetime objects
            KnowledgeUnit(
                unique_id="test-0",
                original_chunk="Python is a programming language",
                contextual_text="",
                embedding_vector=[0.1] * 128,
                knowledge_source="corpus",
                timestamp=now,
                metadata={},
                tags=["python", "programming"],
            ),
            KnowledgeUnit(
                unique_id="test-1",
                original_chunk="Machine learning is a subset of AI",
                contextual_text="",
                embedding_vector=[0.1] * 128,
                knowledge_source="corpus",
                timestamp=now - timedelta(days=1),
                metadata={},
                tags=["ml", "ai"],
            ),
            KnowledgeUnit(
                unique_id="test-2",
                original_chunk="Python is commonly used for data science",
                contextual_text="",
                embedding_vector=[0.1] * 128,
                knowledge_source="action",
                timestamp=now - timedelta(days=2),
                metadata={},
                tags=["python", "data_science"],
            ),
            KnowledgeUnit(
                unique_id="test-3",
                original_chunk="Neural networks are used in deep learning",
                contextual_text="",
                embedding_vector=[0.1] * 128,
                knowledge_source="feedback",
                timestamp=now - timedelta(days=3),
                metadata={},
                tags=["ml", "deep_learning"],
            ),
            KnowledgeUnit(
                unique_id="test-4",
                original_chunk="Python packages for data science include pandas and numpy",
                contextual_text="These packages provide data structures and mathematical functions",
                embedding_vector=[0.1] * 128,
                knowledge_source="action",
                timestamp=now - timedelta(days=4),
                metadata={},
                tags=["python", "data_science", "packages"],
            ),
        ]
        return MockMemoryManager(units)

    @pytest.fixture
    def context_retriever(self, memory_manager):
        """Fixture providing a ContextRetriever with mock manager."""
        return ContextRetriever(
            memory_manager=memory_manager,
            max_context_items=3,
            relevance_threshold=0.5,
            max_token_limit=1000,
        )

    @pytest.mark.asyncio
    async def test_semantic_retrieval(self, context_retriever):
        """Test semantic retrieval strategy."""
        # Retrieve context for a query
        results = await context_retriever.retrieve_context(
            query="Tell me about Python for data science", strategy="semantic"
        )

        # Verify results
        assert len(results) > 0
        # Python data science content should be included
        python_ds_found = False
        for item in results:
            if "Python packages for data science" in item["content"]:
                python_ds_found = True
                break
        assert python_ds_found

        # Verify structure of results
        for item in results:
            assert "content" in item
            assert "source" in item
            assert "relevance" in item
            assert 0 <= item["relevance"] <= 1

    @pytest.mark.asyncio
    async def test_keyword_retrieval(self, context_retriever):
        """Test keyword retrieval strategy."""
        # Retrieve context for a query
        results = await context_retriever.retrieve_context(
            query="machine learning and AI", strategy="keyword"
        )

        # Verify results
        assert len(results) > 0
        # ML/AI content should be included
        ml_found = False
        for item in results:
            if "Machine learning" in item["content"]:
                ml_found = True
                break
        assert ml_found

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, context_retriever):
        """Test hybrid retrieval strategy combining semantic and keyword."""
        # Create a simpler mock response directly
        semantic_results = [
            {
                "content": "Python is a programming language",
                "context": "",
                "source": "corpus",
                "relevance": 0.9,
                "id": "1",
                "metadata": {},
                "tags": ["python"],
            }
        ]

        keyword_results = [
            {
                "content": "Machine learning is a subset of AI",
                "context": "",
                "source": "corpus",
                "relevance": 0.8,
                "id": "2",
                "metadata": {},
                "tags": ["ml"],
            }
        ]

        # Mock the internal retrieval methods
        with (
            patch.object(context_retriever, "_retrieve_semantic", return_value=semantic_results),
            patch.object(context_retriever, "_retrieve_keyword", return_value=keyword_results),
        ):
            # Retrieve context for a query
            results = await context_retriever.retrieve_context(
                query="Python and machine learning", strategy="hybrid"
            )

            # Debug: Print results to understand what's returned
            print("Hybrid retrieval results:")
            for item in results:
                print(f"- {item['content']}")

            # Verify results include both Python and ML content
            assert len(results) > 0

            # Check for both types of content
            python_found = False
            ml_found = False
            for item in results:
                if "python" in item["content"].lower():
                    python_found = True
                if "machine learning" in item["content"].lower():
                    ml_found = True

            assert python_found, "No Python-related content found in hybrid results"
            assert ml_found, "No machine learning-related content found in hybrid results"

    @pytest.mark.asyncio
    async def test_temporal_retrieval(self, context_retriever):
        """Test temporal retrieval strategy."""
        print("\n===== Starting temporal retrieval test =====")

        results = await context_retriever.retrieve_context(
            query="",
            strategy="temporal",  # Query doesn't matter for temporal
        )

        # Print results for debugging
        print(f"Retrieved {len(results)} results")
        for i, item in enumerate(results[:3]):  # Show first 3 results
            print(f"Result {i}:")
            print(f"  Content: {item.get('content', 'N/A')}")
            print(f"  Source: {item.get('source', 'N/A')}")
            print(f"  Timestamp: {item.get('timestamp', 'N/A')}")

        # Basic validation
        assert len(results) > 0, "No results returned from temporal retrieval"

        # Simple check that required fields exist
        assert "content" in results[0], "Missing content field"
        assert "timestamp" in results[0], "Missing timestamp field"
        assert "source" in results[0], "Missing source field"

    @pytest.mark.asyncio
    async def test_source_filtering(self, context_retriever):
        """Test filtering by knowledge source."""
        # Retrieve only corpus sources
        results = await context_retriever.retrieve_context(
            query="Python programming", filters={"knowledge_source": "corpus"}, strategy="semantic"
        )

        # Verify all results are from corpus
        assert len(results) > 0
        for item in results:
            assert item["source"] == "corpus"

    @pytest.mark.asyncio
    async def test_relevance_threshold(self, context_retriever):
        """Test that relevance threshold is applied."""
        # Set a high threshold
        context_retriever.relevance_threshold = 0.9

        # Retrieve with high threshold
        results = await context_retriever.retrieve_context(
            query="completely unrelated query about economics", strategy="semantic"
        )

        # Should return few or no results
        assert len(results) <= 1  # Might still get 1 with mock relevance

        # Reset threshold to normal
        context_retriever.relevance_threshold = 0.5

    @pytest.mark.asyncio
    async def test_max_context_items(self, context_retriever):
        """Test that max_context_items limit is applied."""
        # Set a small limit
        context_retriever.max_context_items = 2

        # Retrieve with small limit
        results = await context_retriever.retrieve_context(
            query="Python or machine learning", strategy="hybrid"
        )

        # Should respect the limit
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_conversation_context_integration(self, context_retriever):
        """Test integration of conversation context with retrieved knowledge."""
        # Create sample conversation
        conversation = [
            {"role": "user", "content": "Tell me about neural networks"},
            {
                "role": "assistant",
                "content": "Neural networks are a type of machine learning model...",
            },
        ]

        # Retrieve with conversation context
        results = await context_retriever.retrieve_context(
            query="How are they used in deep learning?",
            recent_conversation=conversation,
            strategy="semantic",
        )

        # Should include both conversation turns and knowledge
        assert len(results) > 2  # At least the conversation turns plus some knowledge

        # Verify conversation items are included
        conv_found = False
        for item in results:
            if item["source"].startswith("conversation_"):
                conv_found = True
                break
        assert conv_found


class TestPromptBuilder:
    """Test suite for the PromptBuilder."""

    @pytest.fixture
    def prompt_builder(self):
        """Fixture providing a PromptBuilder instance."""
        return PromptBuilder(
            system_prompt_template="You are a helpful assistant. Answer based on this context: ",
            max_prompt_tokens=1000,
        )

    def test_build_prompt_basic(self, prompt_builder):
        """Test building a basic prompt with context."""
        # Create sample context items
        context_items = [
            {
                "content": "Python is a programming language.",
                "context": "",
                "source": "corpus",
                "relevance": 0.9,
                "timestamp": "2025-01-01T12:00:00+00:00",
                "id": "1",
                "metadata": {},
                "tags": [],
            },
            {
                "content": "Python is often used for data science.",
                "context": "Popular libraries include pandas and numpy.",
                "source": "action",
                "relevance": 0.8,
                "timestamp": "2025-01-02T12:00:00+00:00",
                "id": "2",
                "metadata": {"author": "user"},
                "tags": ["python", "data_science"],
            },
        ]

        # Build prompt
        prompt_data = prompt_builder.build_prompt(
            query="Tell me about Python", context_items=context_items
        )

        # Verify structure
        assert "messages" in prompt_data
        assert len(prompt_data["messages"]) == 2  # System and user messages
        assert prompt_data["messages"][0]["role"] == "system"
        assert prompt_data["messages"][1]["role"] == "user"
        assert prompt_data["messages"][1]["content"] == "Tell me about Python"

        # Check system prompt contains context
        system_content = prompt_data["messages"][0]["content"]
        assert "Python is a programming language" in system_content
        assert "Python is often used for data science" in system_content

        # Check context count
        assert prompt_data["context_items_used"] == 2

    def test_build_prompt_with_metadata(self, prompt_builder):
        """Test building a prompt with metadata included."""
        # Create sample context item with metadata
        context_items = [
            {
                "content": "Python packages for data science.",
                "context": "",
                "source": "corpus",
                "relevance": 0.9,
                "timestamp": "2025-01-01T12:00:00+00:00",
                "id": "1",
                "metadata": {"importance": "high", "author": "expert"},
                "tags": ["python", "data_science"],
            }
        ]

        # Build prompt with metadata
        prompt_data = prompt_builder.build_prompt(
            query="Python packages", context_items=context_items, include_metadata=True
        )

        # Verify metadata is included
        system_content = prompt_data["messages"][0]["content"]
        assert "importance: high" in system_content
        assert "author: expert" in system_content

    def test_build_prompt_with_override(self, prompt_builder):
        """Test building a prompt with system prompt override."""
        # Create sample context items
        context_items = [
            {
                "content": "Python syntax example: print('Hello')",
                "context": "",
                "source": "corpus",
                "relevance": 0.9,
                "timestamp": "2025-01-01T12:00:00+00:00",
                "id": "1",
                "metadata": {},
                "tags": [],
            }
        ]

        # Build prompt with override
        custom_system = "You are a Python tutor. Explain code examples clearly."
        prompt_data = prompt_builder.build_prompt(
            query="Explain this code",
            context_items=context_items,
            system_prompt_override=custom_system,
        )

        # Verify custom system prompt is used
        system_content = prompt_data["messages"][0]["content"]
        assert custom_system in system_content
