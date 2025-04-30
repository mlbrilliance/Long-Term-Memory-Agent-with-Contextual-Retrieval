"""
BM25 store implementation for the LTM Agent.

This module provides a BM25-based storage implementation for lexical search
of knowledge units. It uses the rank_bm25 library to implement the BM25 algorithm.
"""

import logging
import os
import pickle

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "The rank_bm25 package is required for BM25 search. Install it with: pip install rank-bm25"
    )

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base_store import BaseBM25Store

# Configure logger
logger = logging.getLogger(__name__)


class RankBM25Store(BaseBM25Store):
    """
    BM25 store implementation using the rank_bm25 library.

    This class provides lexical search capabilities using the BM25 algorithm,
    which is a bag-of-words retrieval function that ranks documents based on
    the appearance of query terms in each document.
    """

    def __init__(self, index_path: str | None = None):
        """
        Initialize a new RankBM25Store.

        Args:
            index_path: Optional path to store/load the BM25 index.
                If None, a default path will be used.
        """
        self.index_path = index_path or "bm25_index.pkl"
        self.bm25 = None
        self.tokenized_corpus = []
        self.id_map = {}  # Maps unit_id to index in corpus

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: The text to tokenize

        Returns:
            List[str]: List of tokens
        """
        # Simple tokenization: lowercase and split on whitespace
        # In a production system, you might want to use a more sophisticated
        # tokenizer that handles stemming, stopwords, etc.
        return text.lower().split()

    def _combine_texts(self, unit: KnowledgeUnit) -> str:
        """
        Combine contextual_text and original_chunk for indexing.

        Args:
            unit: The knowledge unit

        Returns:
            str: Combined text
        """
        contextual = unit.contextual_text or ""
        original = unit.original_chunk or ""
        return f"{contextual} {original}".strip()

    async def update_index(self, units: dict[str, KnowledgeUnit]) -> None:
        """
        Update the BM25 index with the provided knowledge units.

        This method will rebuild the entire index from the provided units.
        Existing units with the same IDs will be replaced.

        Args:
            units: Dictionary mapping unique IDs to KnowledgeUnit objects
        """
        logger.info(f"Updating BM25 index with {len(units)} units")

        # Create a new corpus and id_map
        new_corpus = []
        new_id_map = {}

        # Process each knowledge unit
        for unit_id, unit in units.items():
            combined_text = self._combine_texts(unit)
            tokenized_text = self._tokenize(combined_text)

            # Add to new corpus and id_map
            new_id_map[unit_id] = len(new_corpus)
            new_corpus.append(tokenized_text)

        # If we have existing data, merge it
        if self.bm25 is not None:
            # For each existing unit not in the new units
            for unit_id, idx in self.id_map.items():
                if unit_id not in units:
                    # Add it to the new corpus and id_map
                    new_id_map[unit_id] = len(new_corpus)
                    new_corpus.append(self.tokenized_corpus[idx])

        # Update instance variables
        self.tokenized_corpus = new_corpus
        self.id_map = new_id_map

        # Create a new BM25 instance
        if new_corpus:
            logger.info(f"Creating BM25Okapi with {len(new_corpus)} documents")
            self.bm25 = BM25Okapi(new_corpus)
        else:
            logger.warning("No documents to index, BM25 instance set to None")
            self.bm25 = None

    async def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Perform a BM25 search against the index.

        Args:
            query: The search query
            k: Maximum number of results to return

        Returns:
            List[Tuple[str, float]]: List of (unit_id, score) tuples, ordered by descending score
        """
        logger.info(f"Searching for: '{query}' with k={k}")

        if not self.bm25 or not self.id_map:
            logger.warning("BM25 index is empty, returning empty results")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning("Query tokenized to empty list, returning empty results")
            return []

        # Get scores from BM25
        try:
            scores = self.bm25.get_scores(tokenized_query)
            logger.info(f"Retrieved {len(scores)} scores from BM25")
        except Exception as e:
            logger.error(f"Error getting scores from BM25: {str(e)}")
            return []

        # Create list of (unit_id, score) tuples
        id_scores = []
        for unit_id, idx in self.id_map.items():
            id_scores.append((unit_id, scores[idx]))

        # Sort by score (descending) and take top k
        id_scores.sort(key=lambda x: x[1], reverse=True)
        return id_scores[:k]

    async def save_index(self, path: str | None = None) -> None:
        """
        Save the BM25 index to disk.

        Args:
            path: Optional path to save the index to. If None, use the default path.
        """
        save_path = path or self.index_path
        logger.info(f"Saving BM25 index to {save_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Data to serialize
        data = {
            "tokenized_corpus": self.tokenized_corpus,
            "id_map": self.id_map,
            # We don't serialize the BM25 object itself since it can be recreated
        }

        # Save to disk
        try:
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            logger.info("BM25 index saved successfully")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")
            raise

    async def load_index(self, path: str | None = None) -> None:
        """
        Load the BM25 index from disk.

        Args:
            path: Optional path to load the index from. If None, use the default path.
        """
        load_path = path or self.index_path
        logger.info(f"Loading BM25 index from {load_path}")

        if not os.path.exists(load_path):
            logger.warning(f"BM25 index file not found at {load_path}")
            return

        try:
            # Load from disk
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            # Update instance variables
            self.tokenized_corpus = data["tokenized_corpus"]
            self.id_map = data["id_map"]

            # Recreate BM25 object
            if self.tokenized_corpus:
                logger.info(f"Creating BM25Okapi with {len(self.tokenized_corpus)} documents")
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            else:
                logger.warning("No documents to index, BM25 instance set to None")
                self.bm25 = None

            logger.info(f"BM25 index loaded with {len(self.id_map)} units")
        except Exception as e:
            logger.error(f"Error loading BM25 index: {str(e)}")
            raise

    async def contains(self, unit_id: str) -> bool:
        """
        Check if a knowledge unit is indexed.

        Args:
            unit_id: The unique ID of the knowledge unit

        Returns:
            bool: True if the unit is indexed, False otherwise
        """
        result = unit_id in self.id_map
        logger.debug(f"Contains check for '{unit_id}': {result}")
        return result

    async def remove(self, unit_ids: list[str]) -> None:
        """
        Remove knowledge units from the index.

        This rebuilds the index without the specified units.

        Args:
            unit_ids: List of unique IDs of knowledge units to remove
        """
        logger.info(f"Removing {len(unit_ids)} units from BM25 index")

        if not unit_ids or not self.id_map:
            logger.info("No units to remove or empty index")
            return

        # Create sets for faster lookups
        remove_ids = set(unit_ids)
        current_ids = set(self.id_map.keys())

        # Determine which IDs to keep
        keep_ids = current_ids - remove_ids

        # Create a new corpus and id_map with only the units to keep
        new_corpus = []
        new_id_map = {}

        for unit_id in keep_ids:
            idx = self.id_map[unit_id]
            new_id_map[unit_id] = len(new_corpus)
            new_corpus.append(self.tokenized_corpus[idx])

        # Update instance variables
        self.tokenized_corpus = new_corpus
        self.id_map = new_id_map

        # Recreate BM25 object
        if new_corpus:
            logger.info(f"Creating BM25Okapi with {len(new_corpus)} documents")
            self.bm25 = BM25Okapi(new_corpus)
        else:
            logger.warning("No documents to index, BM25 instance set to None")
            self.bm25 = None

        logger.info(
            f"Removed {len(remove_ids.intersection(current_ids))} units from BM25 index. Remaining: {len(self.id_map)}"
        )
