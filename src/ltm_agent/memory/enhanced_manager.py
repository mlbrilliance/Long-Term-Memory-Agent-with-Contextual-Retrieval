"""
Enhanced Memory Manager for Long-Term Memory Agent.

This module provides an enhanced memory manager that integrates
graph-based memory representation for improved knowledge connections,
multi-step reasoning, and contradiction detection.
"""

import logging
from datetime import datetime
from typing import Any

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from ltm_agent.memory.knowledge_graph import KnowledgeGraph
from ltm_agent.memory.manager import MemoryManager
from ltm_agent.memory.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedMemoryManager(MemoryManager):
    """
    Enhanced Memory Manager with graph-based connections and improved reasoning.

    This manager extends the base MemoryManager with:
    1. Graph-based knowledge representation
    2. Improved connection tracking
    3. Contradiction detection and resolution
    4. Knowledge consolidation
    5. Multi-hop reasoning capabilities
    """

    def __init__(
        self,
        memory_store: VectorStore,
        contextualizer: EnhancedContextualizer | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the enhanced memory manager.

        Args:
            memory_store: Vector store for memory storage
            contextualizer: Optional enhanced contextualizer
            config: Optional configuration parameters
        """
        # Use EnhancedContextualizer by default
        if contextualizer is None:
            contextualizer = EnhancedContextualizer()

        # Initialize the base MemoryManager
        super().__init__(memory_store, contextualizer, config)

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        # Contradiction handling config
        self.config = config or {}
        self.contradiction_resolution = self.config.get("contradiction_resolution", "newest")

        # Consolidation settings
        self.auto_consolidate = self.config.get("auto_consolidate", True)
        self.consolidation_threshold = self.config.get("consolidation_threshold", 10)
        self.last_consolidation_count = 0

    def add_knowledge(
        self, content: str, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> KnowledgeUnit:
        """
        Add new knowledge to memory with enhanced processing.

        Args:
            content: Text content to add
            source: Optional source of the knowledge
            metadata: Optional metadata about the knowledge

        Returns:
            Added knowledge unit
        """
        # Create the knowledge unit using the base implementation
        unit = super().add_knowledge(content, source, metadata)

        # Add to knowledge graph
        self.knowledge_graph.add_knowledge_unit(unit)

        # Check if consolidation is needed
        if self.auto_consolidate:
            self.last_consolidation_count += 1
            if self.last_consolidation_count >= self.consolidation_threshold:
                self.consolidate_knowledge()
                self.last_consolidation_count = 0

        return unit

    def update_knowledge(
        self, unique_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> KnowledgeUnit | None:
        """
        Update existing knowledge with enhanced processing.

        Args:
            unique_id: ID of knowledge unit to update
            content: New content for the knowledge unit
            metadata: Optional updated metadata

        Returns:
            Updated knowledge unit or None if not found
        """
        # First, get the existing unit
        existing_unit = self.get_knowledge(unique_id)
        if not existing_unit:
            logger.warning(f"Knowledge unit with ID {unique_id} not found")
            return None

        # Detect possible contradictions with enhanced contextualizer
        if isinstance(self.contextualizer, EnhancedContextualizer):
            is_contradiction, details = self.contextualizer.detect_contradiction(
                content, existing_unit.original_chunk
            )

            if is_contradiction:
                logger.info(f"Detected contradiction in update for {unique_id}: {details}")

                # Handle contradictions based on configuration
                if self.contradiction_resolution == "reject":
                    logger.warning(f"Rejecting contradictory update for {unique_id}")
                    return existing_unit
                elif self.contradiction_resolution == "flag":
                    # Add contradiction to metadata and proceed with update
                    if metadata is None:
                        metadata = {}

                    if "contradictions" not in metadata:
                        metadata["contradictions"] = []

                    metadata["contradictions"].append(
                        {
                            "detected_at": datetime.now().isoformat(),
                            "previous_content": existing_unit.original_chunk,
                            "details": details,
                        }
                    )

        # Use base implementation for the actual update
        updated_unit = super().update_knowledge(unique_id, content, metadata)

        # Update knowledge graph if update was successful
        if updated_unit:
            self.knowledge_graph.add_knowledge_unit(updated_unit)

        return updated_unit

    def retrieve_relevant(
        self, query: str, limit: int = 5, threshold: float = 0.0
    ) -> list[KnowledgeUnit]:
        """
        Retrieve knowledge relevant to the query with enhanced logic.

        Args:
            query: Query string to find relevant knowledge
            limit: Maximum number of units to return
            threshold: Minimum relevance score (0.0-1.0)

        Returns:
            List of relevant knowledge units
        """
        # First, do the basic vector retrieval
        vector_results = super().retrieve_relevant(query, limit * 2, threshold)

        # If no results from vector search, return empty list
        if not vector_results:
            return []

        # Use the top result to find graph connections
        top_unit = vector_results[0]

        # Get connected units from the knowledge graph
        connected_ids = []
        if len(vector_results) > 0:
            connected = self.knowledge_graph.get_connected_units(
                top_unit.unique_id, max_distance=2, min_weight=0.3
            )
            connected_ids = [unit_id for unit_id, _, _ in connected]

        # Combine vector results with graph connections
        result_ids = {unit.unique_id: unit for unit in vector_results}

        # Add any connected units not already in results
        for unit_id in connected_ids:
            if unit_id not in result_ids:
                unit = self.get_knowledge(unit_id)
                if unit:
                    # Calculate relevance for this unit
                    relevance = self.contextualizer.calculate_relevance(
                        query, unit.original_chunk, unit.metadata
                    )

                    # Only include if above threshold
                    if relevance >= threshold:
                        result_ids[unit_id] = unit

        # Sort all results by relevance to the query
        all_results = list(result_ids.values())
        all_results.sort(
            key=lambda unit: self.contextualizer.calculate_relevance(
                query, unit.original_chunk, unit.metadata
            ),
            reverse=True,
        )

        # Return top results up to the limit
        return all_results[:limit]

    def find_connections(
        self, unit_id: str, max_connections: int = 5, max_distance: int = 2
    ) -> list[dict[str, Any]]:
        """
        Find connections between a knowledge unit and other units.

        Args:
            unit_id: ID of knowledge unit to find connections for
            max_connections: Maximum number of connections to return
            max_distance: Maximum graph distance to consider

        Returns:
            List of connection info dictionaries
        """
        # Check if the unit exists
        unit = self.get_knowledge(unit_id)
        if not unit:
            logger.warning(f"Knowledge unit with ID {unit_id} not found")
            return []

        # Get connected units from the knowledge graph
        connected = self.knowledge_graph.get_connected_units(
            unit_id, max_distance=max_distance, min_weight=0.3
        )

        # Format the results
        results = []
        for connected_id, relevance, distance in connected[:max_connections]:
            connected_unit = self.get_knowledge(connected_id)
            if connected_unit:
                results.append(
                    {
                        "unit_id": connected_id,
                        "content": connected_unit.original_chunk,
                        "relevance_score": relevance,
                        "distance": distance,
                        "metadata": connected_unit.metadata,
                    }
                )

        return results

    def find_multi_hop_path(
        self, start_id: str, end_id: str, max_hops: int = 3
    ) -> list[list[dict[str, Any]]]:
        """
        Find paths connecting two knowledge units for multi-hop reasoning.

        Args:
            start_id: ID of start knowledge unit
            end_id: ID of end knowledge unit
            max_hops: Maximum number of hops in the path

        Returns:
            List of paths, where each path is a list of unit info dictionaries
        """
        # Check if both units exist
        start_unit = self.get_knowledge(start_id)
        end_unit = self.get_knowledge(end_id)

        if not (start_unit and end_unit):
            logger.warning(f"One or both knowledge units not found: {start_id}, {end_id}")
            return []

        # Find paths in the knowledge graph
        paths = self.knowledge_graph.find_paths(start_id, end_id, max_hops)

        # Format the results
        formatted_paths = []

        for path in paths:
            formatted_path = []

            for node_id, relation in path:
                unit = self.get_knowledge(node_id)
                if unit:
                    formatted_path.append(
                        {
                            "unit_id": node_id,
                            "content": unit.original_chunk,
                            "relation": relation,
                            "metadata": unit.metadata,
                        }
                    )

            formatted_paths.append(formatted_path)

        # Sort paths by length (shorter paths first)
        formatted_paths.sort(key=len)

        return formatted_paths

    def find_common_connections(
        self, unit_ids: list[str], max_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Find knowledge units that connect multiple input units.

        Args:
            unit_ids: List of knowledge unit IDs
            max_results: Maximum number of results to return

        Returns:
            List of common connection info dictionaries
        """
        # Validate input units
        valid_ids = []
        for unit_id in unit_ids:
            if self.get_knowledge(unit_id):
                valid_ids.append(unit_id)

        if len(valid_ids) < 2:
            logger.warning("Need at least 2 valid knowledge units for common connections")
            return []

        # Find common connections in the knowledge graph
        common = self.knowledge_graph.get_common_connections(valid_ids, max_results)

        # Format the results
        results = []
        for unit_id, relevance in common:
            unit = self.get_knowledge(unit_id)
            if unit:
                results.append(
                    {
                        "unit_id": unit_id,
                        "content": unit.original_chunk,
                        "relevance_score": relevance,
                        "metadata": unit.metadata,
                    }
                )

        return results

    def detect_contradictions(self, unit_id: str) -> list[dict[str, Any]]:
        """
        Detect contradictions between a knowledge unit and others.

        Args:
            unit_id: ID of knowledge unit to check for contradictions

        Returns:
            List of contradiction info dictionaries
        """
        # Check if the unit exists
        unit = self.get_knowledge(unit_id)
        if not unit:
            logger.warning(f"Knowledge unit with ID {unit_id} not found")
            return []

        # Ensure we have the enhanced contextualizer
        if not isinstance(self.contextualizer, EnhancedContextualizer):
            logger.warning("Enhanced contextualizer required for contradiction detection")
            return []

        # Get potential related units to check
        related_units = []

        # First, check units explicitly marked as related in metadata
        if unit.metadata and "related_to" in unit.metadata:
            for related_id in unit.metadata["related_to"]:
                related_unit = self.get_knowledge(related_id)
                if related_unit:
                    related_units.append(related_unit)

        # Also check units that share entities
        if unit.metadata and "entities" in unit.metadata:
            for entity in unit.metadata["entities"]:
                # Find units with this entity using vector search
                entity_units = super().retrieve_relevant(entity, limit=5)
                related_units.extend(entity_units)

        # Deduplicate related units
        related_units = list({unit.unique_id: unit for unit in related_units}.values())

        # Check for contradictions
        contradictions = []

        for related in related_units:
            is_contradiction, details = self.contextualizer.detect_contradiction(
                unit.original_chunk, related.original_chunk
            )

            if is_contradiction and details:
                contradictions.append(
                    {
                        "unit_id": related.unique_id,
                        "content": related.original_chunk,
                        "details": details,
                    }
                )

        return contradictions

    def consolidate_knowledge(self) -> dict[str, Any]:
        """
        Consolidate related knowledge units to improve memory organization.

        Returns:
            Statistics about the consolidation process
        """
        logger.info("Starting knowledge consolidation")

        # Stats to track consolidation results
        stats = {
            "groups_found": 0,
            "units_consolidated": 0,
            "metadata_enriched": 0,
            "contradictions_found": 0,
        }

        # Get all knowledge units
        all_units = self.list_all_knowledge()

        # Use graph to find clusters of highly connected units
        consolidated_ids = set()

        # Process each unit that hasn't been consolidated yet
        for unit in all_units:
            if unit.unique_id in consolidated_ids:
                continue

            # Get strongly connected units
            connected = self.knowledge_graph.get_connected_units(
                unit.unique_id,
                max_distance=1,
                min_weight=0.7,  # High threshold for strong connections
            )

            # Only consider groups with at least 2 units (the unit itself + others)
            if len(connected) < 1:  # Only has itself
                continue

            # We found a group to consolidate
            stats["groups_found"] += 1
            group_ids = [unit.unique_id] + [unit_id for unit_id, _, _ in connected]

            # Mark as processed
            consolidated_ids.update(group_ids)

            # Get the units in the group
            group_units = [unit]
            for unit_id, _, _ in connected:
                other_unit = self.get_knowledge(unit_id)
                if other_unit:
                    group_units.append(other_unit)

            # Perform consolidation on this group
            self._consolidate_group(group_units, stats)

        logger.info(f"Knowledge consolidation complete: {stats}")
        return stats

    def _consolidate_group(self, units: list[KnowledgeUnit], stats: dict[str, int]) -> None:
        """
        Consolidate a group of related knowledge units.

        Args:
            units: List of related knowledge units
            stats: Statistics dictionary to update
        """
        if not units or len(units) < 2:
            return

        # First, check for contradictions within the group
        contradictions = []

        if isinstance(self.contextualizer, EnhancedContextualizer):
            # Compare each pair of units
            for i, unit1 in enumerate(units):
                for unit2 in units[i + 1 :]:
                    is_contradiction, details = self.contextualizer.detect_contradiction(
                        unit1.original_chunk, unit2.original_chunk
                    )

                    if is_contradiction and details:
                        contradictions.append((unit1, unit2, details))

        # Update contradiction stats
        stats["contradictions_found"] += len(contradictions)

        # Handle contradictions if found
        if contradictions:
            for unit1, unit2, details in contradictions:
                # Flag contradictions in metadata
                for unit in [unit1, unit2]:
                    if "contradictions" not in unit.metadata:
                        unit.metadata["contradictions"] = []

                    unit.metadata["contradictions"].append(
                        {
                            "detected_at": datetime.now().isoformat(),
                            "contradicts": unit2.unique_id if unit == unit1 else unit1.unique_id,
                            "details": details,
                        }
                    )

                    # Update the unit with new metadata
                    self.update_knowledge(unit.unique_id, unit.original_chunk, unit.metadata)

        # Enrich metadata for all units in the group
        all_entities = set()
        all_topics = set()
        all_relations = []

        # Collect metadata from all units
        for unit in units:
            metadata = unit.metadata or {}

            # Collect entities
            if "entities" in metadata:
                all_entities.update(metadata["entities"])

            # Collect topics
            if "topics" in metadata:
                all_topics.update(metadata["topics"])

            # Collect relations
            if "relations" in metadata:
                all_relations.extend(metadata["relations"])

        # Update metadata for each unit
        for unit in units:
            if not unit.metadata:
                unit.metadata = {}

            # Ensure 'related_to' field exists
            if "related_to" not in unit.metadata:
                unit.metadata["related_to"] = []

            # Update related_to with other units in group
            for other in units:
                if (
                    other.unique_id != unit.unique_id
                    and other.unique_id not in unit.metadata["related_to"]
                ):
                    unit.metadata["related_to"].append(other.unique_id)

            # Enrich with collected metadata
            unit.metadata["entities"] = list(all_entities)
            unit.metadata["topics"] = list(all_topics)
            unit.metadata["consolidated_at"] = datetime.now().isoformat()
            unit.metadata["group_size"] = len(units)

            # Only include relations if we have them
            if all_relations:
                unit.metadata["relations"] = all_relations

            # Update the unit with enriched metadata
            self.update_knowledge(unit.unique_id, unit.original_chunk, unit.metadata)

            stats["metadata_enriched"] += 1

        stats["units_consolidated"] += len(units)

    def get_knowledge_graph_stats(self) -> dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict with graph statistics
        """
        return self.knowledge_graph.get_stats()

    def export_knowledge_graph(self) -> dict[str, Any]:
        """
        Export the knowledge graph for visualization.

        Returns:
            Dict with serializable graph data
        """
        return self.knowledge_graph.to_dict()
