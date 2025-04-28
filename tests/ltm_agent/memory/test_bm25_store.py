"""
Tests for the BM25 store implementation.

This module contains tests for the BM25 store, which provides lexical search
capabilities using the BM25 algorithm.
"""

import os
import pickle
import pytest
import sys
from typing import Dict, List, Tuple, Any
import numpy as np

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.utils.test_utils import create_test_knowledge_unit


# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio


class TestBM25Store:
    """Tests for the BM25 store implementation."""
    
    @pytest.fixture
    def bm25_store(self):
        """Fixture providing a RankBM25Store instance."""
        store = RankBM25Store(index_path="test_bm25_store.pkl")
        return store
    
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
    
    async def test_update_and_search(self, bm25_store, sample_knowledge_units):
        """Test update_index and search functionality together."""
        # Update the index with sample units
        await bm25_store.update_index(sample_knowledge_units)
        
        # Verify that the corpus is created correctly
        assert len(bm25_store.tokenized_corpus) == len(sample_knowledge_units)
        assert len(bm25_store.id_map) == len(sample_knowledge_units)
        
        # Perform a search
        results = await bm25_store.search("python", k=2)
        
        # Verify the search results
        assert len(results) <= 2  # Should return at most k results
        
        # Results should be tuples of (id, score)
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], str)  # ID should be a string
            assert isinstance(result[1], (float, np.float64))  # Score should be a float
    
    async def test_contains_and_remove(self, bm25_store, sample_knowledge_units):
        """Test contains and remove functionality."""
        # Update the index with sample units
        await bm25_store.update_index(sample_knowledge_units)
        
        # Check contains for existing IDs
        assert await bm25_store.contains('id1')
        assert await bm25_store.contains('id2')
        assert await bm25_store.contains('id3')
        assert not await bm25_store.contains('non_existent_id')
        
        # Remove some units
        await bm25_store.remove(['id1', 'id3'])
        
        # Verify units were removed
        assert not await bm25_store.contains('id1')
        assert await bm25_store.contains('id2')
        assert not await bm25_store.contains('id3')
    
    async def test_save_and_load(self, bm25_store, sample_knowledge_units, tmp_path):
        """Test save_index and load_index functionality."""
        # Update the index with sample units
        await bm25_store.update_index(sample_knowledge_units)
        
        # Create a temporary file path for the test
        test_file = os.path.join(tmp_path, "test_save_load.pkl")
        
        # Save the index
        await bm25_store.save_index(test_file)
        
        # Create a new store instance
        new_store = RankBM25Store(index_path=test_file)
        
        # Load the index
        await new_store.load_index()
        
        # Verify the loaded index contains the right IDs
        assert await new_store.contains('id1')
        assert await new_store.contains('id2')
        assert await new_store.contains('id3')
        
        # Verify search works with the loaded index
        results = await new_store.search("python", k=1)
        assert len(results) == 1
