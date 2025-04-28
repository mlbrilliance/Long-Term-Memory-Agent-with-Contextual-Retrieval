"""
Tests for the Hybrid Retriever implementation.

This module contains tests for the hybrid retriever, which combines
results from both vector similarity and BM25 lexical search.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
import asyncio
from typing import Dict, List, Tuple, Any, Optional

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.interfaces import VectorStore
from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.utils.test_utils import create_test_knowledge_unit


# Mark all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio


class TestHybridRetriever:
    """Tests for the Hybrid Retriever implementation."""
    
    @pytest.fixture
    def vector_store_mock(self):
        """Fixture providing a mocked VectorStore."""
        mock = MagicMock(spec=VectorStore)
        
        # Set up mock methods
        mock.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.similarity_search = AsyncMock()
        mock.get = AsyncMock()
        
        return mock
    
    @pytest.fixture
    def bm25_store_mock(self):
        """Fixture providing a mocked BM25Store."""
        mock = MagicMock(spec=BaseBM25Store)
        
        # Set up mock methods
        mock.search = AsyncMock()
        
        return mock
    
    @pytest.fixture
    def sample_knowledge_units(self):
        """Fixture providing sample knowledge units."""
        units = {}
        
        # Create units with different content for testing
        units['id1'] = create_test_knowledge_unit(
            content="Python is a high-level programming language",
            context="Python is known for its readability and versatility"
        )
        units['id2'] = create_test_knowledge_unit(
            content="Machine learning algorithms learn from data",
            context="ML systems improve through experience with data"
        )
        units['id3'] = create_test_knowledge_unit(
            content="Neural networks are inspired by the human brain",
            context="Deep learning models use multiple layers of neurons"
        )
        
        return units
    
    async def test_init_validates_weights(self, vector_store_mock, bm25_store_mock):
        """Test that the constructor validates weights correctly."""
        # Valid weights should work
        retriever = LangchainHybridRetriever(vector_store_mock, bm25_store_mock, 0.5, 0.5)
        assert retriever.vector_weight == 0.5
        assert retriever.bm25_weight == 0.5
        
        # Invalid vector weight should raise ValueError
        with pytest.raises(ValueError):
            LangchainHybridRetriever(vector_store_mock, bm25_store_mock, -0.1, 0.5)
        
        with pytest.raises(ValueError):
            LangchainHybridRetriever(vector_store_mock, bm25_store_mock, 1.1, 0.5)
        
        # Invalid BM25 weight should raise ValueError
        with pytest.raises(ValueError):
            LangchainHybridRetriever(vector_store_mock, bm25_store_mock, 0.5, -0.1)
        
        with pytest.raises(ValueError):
            LangchainHybridRetriever(vector_store_mock, bm25_store_mock, 0.5, 1.1)
    
    async def test_set_weights(self, vector_store_mock, bm25_store_mock):
        """Test that set_weights updates weights correctly."""
        retriever = LangchainHybridRetriever(vector_store_mock, bm25_store_mock, 0.5, 0.5)
        
        # Update weights
        await retriever.set_weights(0.7, 0.3)
        assert retriever.vector_weight == 0.7
        assert retriever.bm25_weight == 0.3
        
        # Invalid weights should raise ValueError
        with pytest.raises(ValueError):
            await retriever.set_weights(1.1, 0.3)
    
    async def test_search_with_vector_only(self, vector_store_mock, bm25_store_mock, sample_knowledge_units):
        """Test search with vector search only (BM25 weight = 0)."""
        # Set up vector store mock
        units = list(sample_knowledge_units.values())
        vector_results = [(units[0], 0.9), (units[1], 0.7), (units[2], 0.5)]
        vector_store_mock.similarity_search.return_value = vector_results
        
        # Create retriever with vector weight = 1, BM25 weight = 0
        retriever = LangchainHybridRetriever(
            vector_store_mock, bm25_store_mock, 
            vector_weight=1.0, bm25_weight=0.0
        )
        
        # Perform search
        results = await retriever.search("python", limit=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0][0] == units[0]  # Highest scoring unit
        assert results[1][0] == units[1]  # Second highest scoring unit
        
        # Verify vector search was called
        vector_store_mock.generate_embedding.assert_called_once()
        vector_store_mock.similarity_search.assert_called_once()
        
        # Verify BM25 search was not called
        bm25_store_mock.search.assert_not_called()
    
    async def test_search_with_bm25_only(self, vector_store_mock, bm25_store_mock, sample_knowledge_units):
        """Test search with BM25 search only (vector weight = 0)."""
        # Set up mock data
        units = list(sample_knowledge_units.values())
        unit_map = {unit.unique_id: unit for unit in units}
        
        # Set up BM25 search mock to return IDs and scores
        bm25_id_results = [(units[0].unique_id, 0.8), (units[1].unique_id, 0.6)]
        bm25_store_mock.search.return_value = bm25_id_results
        
        # Set up vector store get method to return units by ID
        async def mock_get(unit_id):
            return unit_map.get(unit_id)
        
        vector_store_mock.get.side_effect = mock_get
        
        # Create retriever with vector weight = 0, BM25 weight = 1
        retriever = LangchainHybridRetriever(
            vector_store_mock, bm25_store_mock, 
            vector_weight=0.0, bm25_weight=1.0
        )
        
        # Perform search
        results = await retriever.search("python", limit=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0][0].unique_id == units[0].unique_id
        assert results[1][0].unique_id == units[1].unique_id
        
        # Verify vector search was not called
        vector_store_mock.generate_embedding.assert_not_called()
        vector_store_mock.similarity_search.assert_not_called()
        
        # Verify BM25 search was called
        bm25_store_mock.search.assert_called_once()
        
        # Verify vector store get was called for each BM25 result
        assert vector_store_mock.get.call_count == 2
    
    async def test_search_with_hybrid_weights(self, vector_store_mock, bm25_store_mock, sample_knowledge_units):
        """Test search with both vector and BM25 search with custom weights."""
        # Set up mock data
        units = list(sample_knowledge_units.values())
        unit_map = {unit.unique_id: unit for unit in units}
        
        # Set up vector store mock
        vector_results = [(units[0], 0.9), (units[2], 0.7)]
        vector_store_mock.similarity_search.return_value = vector_results
        
        # Set up BM25 store mock
        bm25_id_results = [(units[0].unique_id, 0.8), (units[1].unique_id, 0.9)]
        bm25_store_mock.search.return_value = bm25_id_results
        
        # Set up vector store get method
        async def mock_get(unit_id):
            return unit_map.get(unit_id)
        
        vector_store_mock.get.side_effect = mock_get
        
        # Create retriever with custom weights
        retriever = LangchainHybridRetriever(
            vector_store_mock, bm25_store_mock, 
            vector_weight=0.4, bm25_weight=0.6
        )
        
        # Perform search
        results = await retriever.search("python", limit=3)
        
        # Verify results
        assert len(results) == 3
        
        # Expected order based on combined scores
        # Results should include all 3 units with appropriate scores
        result_ids = {result[0].unique_id for result in results}
        assert len(result_ids) == 3
        
        # Verify correct methods were called
        vector_store_mock.generate_embedding.assert_called_once()
        vector_store_mock.similarity_search.assert_called_once()
        bm25_store_mock.search.assert_called_once()
        assert vector_store_mock.get.call_count == 2
    
    async def test_rerank_results(self, vector_store_mock, bm25_store_mock, sample_knowledge_units):
        """Test that rerank_results applies length penalization correctly."""
        # Create retriever
        retriever = LangchainHybridRetriever(
            vector_store_mock, bm25_store_mock, 
            vector_weight=0.5, bm25_weight=0.5
        )
        
        # Create results with different content lengths
        units = list(sample_knowledge_units.values())
        
        # Override one unit to have very long content for testing length penalty
        long_unit = create_test_knowledge_unit(
            content="A" * 2000,  # Very long content
            context="Long content unit"
        )
        
        combined_results = [
            (units[0], 0.9),     # Normal length
            (long_unit, 0.9),    # Very long, should be penalized
            (units[1], 0.8),     # Normal length
        ]
        
        # Rerank results
        reranked = await retriever.rerank_results("query", combined_results)
        
        # The first unit should still be first (highest score)
        assert reranked[0][0] == units[0]
        
        # The long unit should be penalized and end up below the normal length unit
        # with a slightly lower original score
        assert reranked[1][0] == units[1]
        assert reranked[2][0] == long_unit
