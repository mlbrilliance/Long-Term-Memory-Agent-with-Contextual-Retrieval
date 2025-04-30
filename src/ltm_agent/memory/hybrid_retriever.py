"""
Hybrid Retriever for the LTM Agent.

This module provides a hybrid retrieval implementation that combines results from
both vector similarity and BM25 lexical search to improve retrieval quality.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.base_store import BaseBM25Store
from ltm_agent.memory.interfaces import HybridRetriever, VectorStore

# Configure logger
logger = logging.getLogger(__name__)


class LangchainHybridRetriever(HybridRetriever):
    """
    Hybrid retrieval implementation that combines vector and BM25 search results.

    This class provides methods for combining and reranking results from both
    vector similarity and BM25 lexical search approaches, offering improved
    retrieval performance over either method alone.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BaseBM25Store,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        min_score_threshold: float = 0.0,
    ):
        """
        Initialize a new LangchainHybridRetriever.

        Args:
            vector_store: The vector store for semantic search
            bm25_store: The BM25 store for lexical search
            vector_weight: Weight to apply to vector search results (0.0 to 1.0)
            bm25_weight: Weight to apply to BM25 search results (0.0 to 1.0)
            min_score_threshold: Minimum score threshold for results (0.0 to 1.0)
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.min_score_threshold = min_score_threshold

        # Validate weights
        self._validate_weights(vector_weight, bm25_weight)

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """
        Validate that the weights are in the correct range.

        Args:
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results

        Raises:
            ValueError: If weights are invalid
        """
        if not (0.0 <= vector_weight <= 1.0):
            raise ValueError(f"Vector weight must be between 0.0 and 1.0, got {vector_weight}")

        if not (0.0 <= bm25_weight <= 1.0):
            raise ValueError(f"BM25 weight must be between 0.0 and 1.0, got {bm25_weight}")

    async def set_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """
        Set the weights used for combining vector and BM25 search results.

        Args:
            vector_weight: Weight to apply to vector search results (0.0 to 1.0)
            bm25_weight: Weight to apply to BM25 search results (0.0 to 1.0)

        Raises:
            ValueError: If weights are invalid
        """
        self._validate_weights(vector_weight, bm25_weight)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        logger.info(f"Weights updated: vector={vector_weight}, bm25={bm25_weight}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Perform a hybrid search using both vector and BM25 retrieval methods.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_criteria: Optional criteria to filter search results
            vector_weight: Optional override for vector search weight
            bm25_weight: Optional override for BM25 search weight

        Returns:
            List[Tuple[KnowledgeUnit, float]]: Combined and re-ranked results
                                              with their relevance scores
        """
        # Use provided weights if given, otherwise use instance weights
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        b_weight = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Validate weights if they were provided
        if vector_weight is not None or bm25_weight is not None:
            self._validate_weights(v_weight, b_weight)

        logger.info(f"Performing hybrid search for query: '{query}' with limit={limit}")
        logger.info(f"Using weights: vector={v_weight}, bm25={b_weight}")

        # Skip retrieval methods with zero weight
        vector_results = []
        bm25_results = []

        # Run both searches in parallel
        search_tasks = []

        if v_weight > 0:
            # Generate embedding for vector search
            embedding = await self.vector_store.generate_embedding(query)
            search_tasks.append(
                self.vector_store.similarity_search(
                    embedding,
                    limit=limit * 2,  # Get more results initially for better merging
                    filter_criteria=filter_criteria,
                )
            )

        if b_weight > 0:
            search_tasks.append(
                self.bm25_store.search(
                    query,
                    k=limit * 2,  # Get more results initially for better merging
                    filter_criteria=filter_criteria,
                )
            )

        # Execute searches concurrently
        results = await asyncio.gather(*search_tasks)

        # Process results based on which searches were performed
        result_idx = 0
        if v_weight > 0:
            vector_results = results[result_idx]
            result_idx += 1
        if b_weight > 0:
            # BM25 returns (id, score) tuples, so we need to convert to KnowledgeUnit
            bm25_id_results = results[result_idx]

            # Fetch the actual KnowledgeUnit objects for the BM25 results
            bm25_results = []
            for unit_id, score in bm25_id_results:
                unit = await self.vector_store.get(unit_id)
                if unit:
                    bm25_results.append((unit, score))

        # Combine and rerank results
        combined_results = await self._combine_results(
            vector_results, bm25_results, v_weight, b_weight
        )

        # Rerank with optional advanced methods
        final_results = await self.rerank_results(query, combined_results, limit)

        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results

    async def _combine_results(
        self,
        vector_results: list[tuple[KnowledgeUnit, float]],
        bm25_results: list[tuple[KnowledgeUnit, float]],
        vector_weight: float,
        bm25_weight: float,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Combine results from vector and BM25 searches with the specified weights.

        Args:
            vector_results: Results from vector similarity search
            bm25_results: Results from BM25 lexical search
            vector_weight: Weight to apply to vector results
            bm25_weight: Weight to apply to BM25 results

        Returns:
            List[Tuple[KnowledgeUnit, float]]: Combined results with normalized scores
        """
        # Normalize scores for each method if they have results
        if vector_results:
            vector_max_score = max(score for _, score in vector_results)
            if vector_max_score > 0:
                vector_results = [
                    (unit, score / vector_max_score) for unit, score in vector_results
                ]

        if bm25_results:
            bm25_max_score = max(score for _, score in bm25_results)
            if bm25_max_score > 0:
                bm25_results = [(unit, score / bm25_max_score) for unit, score in bm25_results]

        # Combine all results
        combined_scores = defaultdict(float)
        unit_map = {}

        # Add vector results with vector weight
        for unit, score in vector_results:
            unit_id = unit.unique_id
            combined_scores[unit_id] += score * vector_weight
            unit_map[unit_id] = unit

        # Add BM25 results with BM25 weight
        for unit, score in bm25_results:
            unit_id = unit.unique_id
            combined_scores[unit_id] += score * bm25_weight
            unit_map[unit_id] = unit

        # Create combined results list
        combined_results = [
            (unit_map[unit_id], score)
            for unit_id, score in combined_scores.items()
            if score >= self.min_score_threshold
        ]

        # Sort by descending score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results

    async def rerank_results(
        self,
        query: str,
        combined_results: list[tuple[KnowledgeUnit, float]],
        limit: int = 10,
        use_reranker: bool = False,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Rerank combined results using optional advanced reranking strategies.

        Args:
            query: The original search query
            combined_results: List of knowledge units with their initial scores
            limit: Maximum number of results to return
            use_reranker: Whether to use an advanced reranker (if available)

        Returns:
            List[Tuple[KnowledgeUnit, float]]: Reranked results with updated scores
        """
        # If advanced reranker is requested, apply it here
        if use_reranker:
            # This is where an advanced reranker could be applied
            # For example, a cross-encoder reranker or a custom scoring function
            logger.info("Advanced reranking requested but not implemented")
            # For now, just pass through the existing results

        # Apply basic length penalization (slightly favor shorter, more focused content)
        reranked_results = []
        for unit, score in combined_results:
            content_length = len(unit.original_chunk)
            # Apply a mild length penalty for very long content
            length_factor = 1.0 if content_length < 1000 else (1000 / content_length) ** 0.25
            reranked_score = score * length_factor
            reranked_results.append((unit, reranked_score))

        # Sort by descending score and limit results
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:limit]


# Utility functions for evaluating retrieval performance


async def evaluate_retrieval(
    queries: list[str],
    ground_truth: dict[str, list[str]],
    retriever: HybridRetriever,
    limit: int = 10,
) -> dict[str, float]:
    """
    Evaluate retrieval performance on a set of queries with known relevant documents.

    Args:
        queries: List of test queries
        ground_truth: Mapping of query to list of relevant document IDs
        retriever: The retriever to evaluate
        limit: Maximum number of results to retrieve per query

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0  # Mean Reciprocal Rank

    for query in queries:
        if query not in ground_truth:
            continue

        relevant_ids = set(ground_truth[query])
        results = await retriever.search(query, limit=limit)
        retrieved_ids = {unit.unique_id for unit, _ in results}

        # Calculate precision, recall, F1
        if retrieved_ids:
            precision = len(relevant_ids.intersection(retrieved_ids)) / len(retrieved_ids)
            recall = len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0

        # Calculate MRR
        mrr = 0.0
        for i, (unit, _) in enumerate(results):
            if unit.unique_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

    # Calculate averages
    num_queries = len(queries)
    metrics = {
        "precision": total_precision / num_queries,
        "recall": total_recall / num_queries,
        "f1": total_f1 / num_queries,
        "mrr": total_mrr / num_queries,
    }

    return metrics
