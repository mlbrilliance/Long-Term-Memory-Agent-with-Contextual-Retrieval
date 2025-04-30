"""
Memory pruning system for Long-Term Memory Agent.

This module provides tools for pruning redundant or outdated knowledge
from the agent's memory to improve efficiency and relevance.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.knowledge_graph import KnowledgeGraph
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)


class MemoryPruner:
    """
    Memory pruning system for maintaining efficient knowledge bases.

    This class provides methods to identify and remove:
    1. Redundant knowledge (duplicates or near-duplicates)
    2. Outdated knowledge (superseded by newer information)
    3. Low-relevance knowledge (rarely accessed or low importance)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the memory pruner.

        Args:
            vector_store: Vector store containing knowledge units
            knowledge_graph: Optional knowledge graph for relationship-aware pruning
            config: Configuration parameters for pruning
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.config = config or {}

        # Default configuration
        self.similarity_threshold = self.config.get("similarity_threshold", 0.95)
        self.age_threshold_days = self.config.get("age_threshold_days", 90)
        self.access_count_threshold = self.config.get("access_count_threshold", 2)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.3)
        self.max_knowledge_units = self.config.get("max_knowledge_units", 10000)
        self.preserve_fraction = self.config.get("preserve_fraction", 0.8)
        self.batch_size = self.config.get("batch_size", 100)
        self.pruning_strategy = self.config.get("pruning_strategy", "combined")

        # Statistic tracking
        self.stats = {
            "last_pruning": None,
            "total_pruned": 0,
            "redundant_pruned": 0,
            "outdated_pruned": 0,
            "low_relevance_pruned": 0,
            "preserved_connections": 0,
        }

    def find_redundant_units(
        self, units: list[KnowledgeUnit], keep_newer: bool = True
    ) -> list[tuple[KnowledgeUnit, KnowledgeUnit, float]]:
        """
        Find redundant (near-duplicate) knowledge units.

        Args:
            units: List of knowledge units to check
            keep_newer: Whether to keep newer units when finding duplicates

        Returns:
            List of tuples (unit_to_keep, unit_to_remove, similarity_score)
        """
        redundant_pairs = []

        # Group units by content length for more efficient comparison
        grouped_by_length = defaultdict(list)
        for unit in units:
            length = len(unit.original_chunk)
            # Group by length ranges (within 20% of each other)
            length_key = length // 50  # Group by ~50 character ranges
            grouped_by_length[length_key].append(unit)

        # Process each length group
        for length_key, group in grouped_by_length.items():
            # Skip tiny groups
            if len(group) <= 1:
                continue

            # Compare each pair in the group
            for i in range(len(group)):
                unit1 = group[i]
                embedding1 = unit1.embedding

                for j in range(i + 1, len(group)):
                    unit2 = group[j]
                    embedding2 = unit2.embedding

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(embedding1, embedding2)

                    # Check if they're similar enough to be redundant
                    if similarity >= self.similarity_threshold:
                        # Determine which unit to keep
                        if keep_newer:
                            # Compare timestamps
                            time1 = (
                                datetime.fromisoformat(unit1.created_at)
                                if unit1.created_at
                                else datetime.min
                            )
                            time2 = (
                                datetime.fromisoformat(unit2.created_at)
                                if unit2.created_at
                                else datetime.min
                            )

                            if time1 >= time2:
                                redundant_pairs.append((unit1, unit2, similarity))
                            else:
                                redundant_pairs.append((unit2, unit1, similarity))
                        else:
                            # In this case, just choose the first unit arbitrarily
                            redundant_pairs.append((unit1, unit2, similarity))

        return redundant_pairs

    def find_outdated_units(
        self, units: list[KnowledgeUnit], reference_date: datetime | None = None
    ) -> list[KnowledgeUnit]:
        """
        Find outdated knowledge units based on age and access patterns.

        Args:
            units: List of knowledge units to check
            reference_date: Optional reference date (defaults to now)

        Returns:
            List of outdated knowledge units
        """
        if reference_date is None:
            reference_date = datetime.now()

        outdated_units = []
        threshold_date = reference_date - timedelta(days=self.age_threshold_days)

        for unit in units:
            # Skip units with explicit "preserve" flag in metadata
            if unit.metadata and unit.metadata.get("preserve", False):
                continue

            # Get creation date
            created_at = datetime.fromisoformat(unit.created_at) if unit.created_at else None

            # Get last accessed date and access count from metadata
            last_accessed = None
            access_count = 0

            if unit.metadata:
                if "last_accessed" in unit.metadata:
                    try:
                        last_accessed = datetime.fromisoformat(unit.metadata["last_accessed"])
                    except (ValueError, TypeError):
                        pass

                access_count = unit.metadata.get("access_count", 0)

            # Consider outdated if:
            # 1. Created before threshold date AND
            # 2. Either never accessed OR last accessed before threshold date
            # 3. Low access count
            if (
                created_at
                and created_at < threshold_date
                and (not last_accessed or last_accessed < threshold_date)
                and access_count <= self.access_count_threshold
            ):
                outdated_units.append(unit)

        return outdated_units

    def find_low_relevance_units(
        self,
        units: list[KnowledgeUnit],
        relevance_function: Callable[[KnowledgeUnit], float] | None = None,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """
        Find knowledge units with low relevance scores.

        Args:
            units: List of knowledge units to check
            relevance_function: Optional function to calculate relevance

        Returns:
            List of (unit, relevance_score) tuples
        """

        # Default relevance function based on metadata
        def default_relevance(unit: KnowledgeUnit) -> float:
            # Base relevance starts at 0.5
            relevance = 0.5

            if not unit.metadata:
                return relevance

            # Adjust based on metadata factors

            # Factor 1: Access count (more access = higher relevance)
            access_count = unit.metadata.get("access_count", 0)
            relevance += min(0.3, 0.02 * access_count)  # Up to +0.3 for frequent access

            # Factor 2: Connections (more connections = higher relevance)
            num_connections = 0
            if "related_to" in unit.metadata and isinstance(unit.metadata["related_to"], list):
                num_connections = len(unit.metadata["related_to"])
            relevance += min(0.2, 0.02 * num_connections)  # Up to +0.2 for well-connected units

            # Factor 3: Importance flag (if explicitly marked important)
            if unit.metadata.get("importance", 0) > 0:
                relevance += 0.2  # +0.2 for important units

            # Factor 4: Recency (newer = higher relevance)
            if "last_accessed" in unit.metadata:
                try:
                    last_accessed = datetime.fromisoformat(unit.metadata["last_accessed"])
                    days_since_access = (datetime.now() - last_accessed).days

                    if days_since_access < 7:
                        relevance += 0.1  # +0.1 for very recent access
                    elif days_since_access < 30:
                        relevance += 0.05  # +0.05 for moderately recent access
                except (ValueError, TypeError):
                    pass

            return min(1.0, relevance)  # Cap at 1.0

        # Use provided function or default
        relevance_func = relevance_function or default_relevance

        # Calculate relevance for each unit
        unit_relevance = [(unit, relevance_func(unit)) for unit in units]

        # Sort by relevance (lowest first)
        unit_relevance.sort(key=lambda x: x[1])

        # Return units below threshold
        return [(unit, score) for unit, score in unit_relevance if score < self.relevance_threshold]

    def should_preserve_unit(
        self, unit: KnowledgeUnit, redundant_units: list[KnowledgeUnit] | None = None
    ) -> bool:
        """
        Determine if a unit should be preserved despite pruning criteria.

        Args:
            unit: Knowledge unit to check
            redundant_units: Optional list of already identified redundant units

        Returns:
            True if the unit should be preserved
        """
        # Always preserve units with explicit "preserve" flag
        if unit.metadata and unit.metadata.get("preserve", False):
            return True

        # If we have a knowledge graph, check if this unit is important for connections
        if self.knowledge_graph:
            # Check if it's a key connector node (has many connections)
            connected = self.knowledge_graph.get_connected_units(unit.unique_id, max_distance=1)

            # Preserve if it has many direct connections (hub node)
            if len(connected) >= 3:
                return True

            # Check if it's on a unique path between other nodes
            if redundant_units and self.knowledge_graph:
                # Only check a sample of units for performance
                sample_units = self._sample_units(redundant_units, 10)

                for unit1 in sample_units:
                    for unit2 in sample_units:
                        if (
                            unit1.unique_id == unit2.unique_id
                            or unit1.unique_id == unit.unique_id
                            or unit2.unique_id == unit.unique_id
                        ):
                            continue

                        # Check if this unit is on a path between unit1 and unit2
                        paths = self.knowledge_graph.find_paths(unit1.unique_id, unit2.unique_id, 3)

                        for path in paths:
                            # Check if this unit is in the path
                            path_ids = [node_id for node_id, _ in path]
                            if unit.unique_id in path_ids:
                                return True

        return False

    def prune_memory(self) -> dict[str, Any]:
        """
        Prune the memory by removing redundant, outdated, and low-relevance units.

        Returns:
            Statistics about the pruning operation
        """
        start_time = time.time()
        logger.info("Starting memory pruning operation")

        # Reset stats for this pruning operation
        pruning_stats = {
            "redundant_pruned": 0,
            "outdated_pruned": 0,
            "low_relevance_pruned": 0,
            "preserved_connections": 0,
            "total_size_before": 0,
            "total_size_after": 0,
            "duration_seconds": 0,
        }

        # Get all knowledge units
        all_units = self.vector_store.list_all()
        pruning_stats["total_size_before"] = len(all_units)

        logger.info(f"Found {len(all_units)} knowledge units to analyze")

        # Determine how many units to keep based on max size
        target_size = min(self.max_knowledge_units, int(len(all_units) * self.preserve_fraction))
        num_to_prune = max(0, len(all_units) - target_size)

        # Skip if no pruning needed
        if num_to_prune <= 0:
            logger.info("No pruning needed, current size is within limits")
            pruning_stats["total_size_after"] = len(all_units)
            pruning_stats["duration_seconds"] = time.time() - start_time
            return pruning_stats

        logger.info(f"Target: remove approximately {num_to_prune} knowledge units")

        # Units to be pruned
        to_prune = set()

        # Find redundant units
        redundant_pairs = self.find_redundant_units(all_units)
        redundant_units = set()

        for keep, remove, score in redundant_pairs:
            if remove.unique_id not in to_prune and remove.unique_id not in redundant_units:
                # Check if we should preserve this unit despite redundancy
                if self.should_preserve_unit(remove, all_units):
                    pruning_stats["preserved_connections"] += 1
                    continue

                redundant_units.add(remove.unique_id)
                to_prune.add(remove.unique_id)
                pruning_stats["redundant_pruned"] += 1

        logger.info(f"Identified {len(redundant_units)} redundant units for pruning")

        # If still need more pruning, check for outdated units
        if len(to_prune) < num_to_prune and "outdated" in self.pruning_strategy:
            # Find outdated units
            remaining_units = [u for u in all_units if u.unique_id not in to_prune]
            outdated_units = self.find_outdated_units(remaining_units)

            for unit in outdated_units:
                if len(to_prune) >= num_to_prune:
                    break

                # Check if we should preserve this unit despite being outdated
                if self.should_preserve_unit(unit, remaining_units):
                    pruning_stats["preserved_connections"] += 1
                    continue

                to_prune.add(unit.unique_id)
                pruning_stats["outdated_pruned"] += 1

            logger.info(f"Identified {pruning_stats['outdated_pruned']} outdated units for pruning")

        # If still need more pruning, check for low relevance units
        if len(to_prune) < num_to_prune and "relevance" in self.pruning_strategy:
            # Find low relevance units
            remaining_units = [u for u in all_units if u.unique_id not in to_prune]
            low_relevance_units = self.find_low_relevance_units(remaining_units)

            for unit, score in low_relevance_units:
                if len(to_prune) >= num_to_prune:
                    break

                # Check if we should preserve this unit despite low relevance
                if self.should_preserve_unit(unit, remaining_units):
                    pruning_stats["preserved_connections"] += 1
                    continue

                to_prune.add(unit.unique_id)
                pruning_stats["low_relevance_pruned"] += 1

            logger.info(
                f"Identified {pruning_stats['low_relevance_pruned']} low-relevance units for pruning"
            )

        # Actually perform the pruning
        pruned_count = 0

        # Process in batches
        to_prune_list = list(to_prune)
        for i in range(0, len(to_prune_list), self.batch_size):
            batch = to_prune_list[i : i + self.batch_size]

            for unit_id in batch:
                if self.vector_store.delete(unit_id):
                    pruned_count += 1

                    # Also remove from knowledge graph if present
                    if self.knowledge_graph and self.knowledge_graph.graph.has_node(unit_id):
                        self.knowledge_graph.graph.remove_node(unit_id)

            logger.debug(f"Pruned batch of {len(batch)} units")

        # Update final stats
        pruning_stats["total_pruned"] = pruned_count
        pruning_stats["total_size_after"] = self.vector_store.count()
        pruning_stats["duration_seconds"] = time.time() - start_time

        # Update persistent stats
        self.stats["last_pruning"] = datetime.now().isoformat()
        self.stats["total_pruned"] += pruned_count
        self.stats["redundant_pruned"] += pruning_stats["redundant_pruned"]
        self.stats["outdated_pruned"] += pruning_stats["outdated_pruned"]
        self.stats["low_relevance_pruned"] += pruning_stats["low_relevance_pruned"]
        self.stats["preserved_connections"] += pruning_stats["preserved_connections"]

        logger.info(
            f"Memory pruning complete. Removed {pruned_count} units in "
            f"{pruning_stats['duration_seconds']:.2f} seconds"
        )

        return pruning_stats

    def auto_prune_if_needed(self, force: bool = False) -> dict[str, Any] | None:
        """
        Automatically prune if size threshold is exceeded or force is requested.

        Args:
            force: Whether to force pruning regardless of size

        Returns:
            Pruning statistics if pruning was performed, None otherwise
        """
        # Check current size
        current_size = self.vector_store.count()

        # Prune if size exceeds threshold or forced
        if force or current_size > self.max_knowledge_units:
            logger.info(
                f"Auto-pruning triggered (current size: {current_size}, "
                f"max: {self.max_knowledge_units})"
            )
            return self.prune_memory()

        return None

    def get_pruning_stats(self) -> dict[str, Any]:
        """
        Get statistics about pruning operations.

        Returns:
            Dictionary of pruning statistics
        """
        return self.stats

    def _cosine_similarity(self, embedding1, embedding2) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (0-1)
        """
        # Convert to numpy arrays if they're not already
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)

        # Handle zero division
        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def _sample_units(self, units: list[KnowledgeUnit], max_samples: int) -> list[KnowledgeUnit]:
        """
        Sample a subset of units for efficient processing.

        Args:
            units: List of knowledge units
            max_samples: Maximum number of samples to take

        Returns:
            Sampled list of knowledge units
        """
        if len(units) <= max_samples:
            return units

        # Simple random sampling
        import random

        return random.sample(units, max_samples)
