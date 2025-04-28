"""
Base store interfaces for the LTM Agent.

This module defines the abstract base classes for different types of stores
used in the LTM Agent, including vector stores and BM25 stores.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ltm_agent.core.models import KnowledgeUnit


class BaseBM25Store(ABC):
    """
    Abstract base class for BM25-based storage implementations.
    
    This interface defines the methods that any BM25 storage system must implement,
    providing a consistent API for lexical search regardless of the underlying
    implementation.
    """
    
    @abstractmethod
    async def update_index(self, units: Dict[str, KnowledgeUnit]) -> None:
        """
        Update the BM25 index with the provided knowledge units.
        
        This method should process the units and update the internal BM25 index.
        The index should use both the contextual_text and original_chunk text
        from each knowledge unit.
        
        Args:
            units: Dictionary mapping unique IDs to KnowledgeUnit objects
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """
        Perform a BM25 search against the index.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (unit_id, score) tuples, where 
                unit_id is the unique ID of a knowledge unit and score is the 
                BM25 relevance score (higher is better)
        """
        pass
    
    @abstractmethod
    async def save_index(self, path: Optional[str] = None) -> None:
        """
        Save the BM25 index to disk.
        
        Args:
            path: Optional path to save the index to. If None, use the default path.
        """
        pass
    
    @abstractmethod
    async def load_index(self, path: Optional[str] = None) -> None:
        """
        Load the BM25 index from disk.
        
        Args:
            path: Optional path to load the index from. If None, use the default path.
            
        Returns:
            bool: True if the index was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def contains(self, unit_id: str) -> bool:
        """
        Check if a knowledge unit is indexed.
        
        Args:
            unit_id: The unique ID of the knowledge unit
            
        Returns:
            bool: True if the unit is indexed, False otherwise
        """
        pass
    
    @abstractmethod
    async def remove(self, unit_ids: List[str]) -> None:
        """
        Remove knowledge units from the index.
        
        Args:
            unit_ids: List of unique IDs of knowledge units to remove
        """
        pass
