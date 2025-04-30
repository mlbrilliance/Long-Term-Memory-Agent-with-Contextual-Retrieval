"""
Knowledge Graph for Long-Term Memory Agent.

This module provides a graph-based structure for representing relationships
between knowledge units, enhancing the agent's ability to connect related information.
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import networkx as nx

from ltm_agent.core.models import KnowledgeUnit

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Graph-based structure for tracking relationships between knowledge units.

    This class maintains a directed graph where:
    - Nodes represent knowledge units
    - Edges represent relationships between knowledge units (with types and weights)
    - Node attributes store metadata about knowledge units
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        # Main directed graph for relationships
        self.graph = nx.DiGraph()

        # Track entity mentions across knowledge units
        self.entity_to_units = defaultdict(set)

        # Common stopwords to filter out
        self.stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "about",
            "of",
            "as",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "have",
            "has",
            "had",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "cannot",
        }

        # Entity recognition patterns
        self.entity_patterns = [
            # Named entities (capitalized phrases)
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # Technical terms (camelCase, PascalCase, snake_case)
            r"\b([a-z]+(?:[A-Z][a-z]*)+)\b",  # camelCase
            r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b",  # PascalCase
            r"\b([a-z]+(?:_[a-z]+)+)\b",  # snake_case
            # Domains and topics (lowercase sequences)
            r"\b((?:python|javascript|ruby|java|c\+\+|php|go|rust|machine\slearning|deep\slearning|neural\snetworks|artificial\sintelligence|ai|ml|nlp|computer\svision|cv|data\sscience)[a-z]*)\b",
        ]

    def add_knowledge_unit(self, unit: KnowledgeUnit) -> None:
        """
        Add a knowledge unit to the graph.

        Args:
            unit: KnowledgeUnit to add
        """
        # Check if node already exists
        if self.graph.has_node(unit.unique_id):
            logger.debug(f"Knowledge unit {unit.unique_id} already in graph, updating")
            self._update_node(unit)
            return

        # Add the node with its attributes
        self.graph.add_node(
            unit.unique_id,
            content=unit.original_chunk,
            processed_content=unit.processed_chunk,
            creation_time=datetime.now().isoformat() if not unit.created_at else unit.created_at,
            source=unit.source,
            metadata=unit.metadata or {},
        )

        # Extract entities from the content
        entities = self._extract_entities(unit.original_chunk)

        # Update entity tracking
        for entity in entities:
            self.entity_to_units[entity].add(unit.unique_id)

        # Look for existing connections
        self._find_and_add_connections(unit)

        # Add explicit connections from metadata if they exist
        if unit.metadata and "related_to" in unit.metadata:
            related_ids = unit.metadata["related_to"]
            if isinstance(related_ids, list):
                for related_id in related_ids:
                    if self.graph.has_node(related_id):
                        # Add a bidirectional connection with metadata relation type
                        self._add_connection(unit.unique_id, related_id, "metadata_explicit", 1.0)

    def _update_node(self, unit: KnowledgeUnit) -> None:
        """
        Update a knowledge unit node in the graph.

        Args:
            unit: Updated KnowledgeUnit
        """
        # Update node attributes
        self.graph.nodes[unit.unique_id].update(
            {
                "content": unit.original_chunk,
                "processed_content": unit.processed_chunk,
                "updated_time": datetime.now().isoformat(),
                "source": unit.source,
                "metadata": unit.metadata or {},
            }
        )

        # Remove old entity associations
        for entity, unit_ids in list(self.entity_to_units.items()):
            if unit.unique_id in unit_ids:
                unit_ids.remove(unit.unique_id)
                if not unit_ids:
                    del self.entity_to_units[entity]

        # Extract and update entities
        entities = self._extract_entities(unit.original_chunk)
        for entity in entities:
            self.entity_to_units[entity].add(unit.unique_id)

        # Reevaluate connections with existing nodes
        self._find_and_add_connections(unit)

    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract entities and important terms from text.

        Args:
            text: Text to extract entities from

        Returns:
            Set of entity strings
        """
        entities = set()

        # Apply each pattern to extract entities
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Ignore single character matches and stopwords
                if len(match) > 1 and match.lower() not in self.stopwords:
                    entities.add(match)

        # Extract topic keywords (non-stopword terms that appear multiple times)
        words = re.findall(r"\b\w+\b", text.lower())
        word_counts = Counter(words)

        for word, count in word_counts.items():
            if (
                count >= 2  # Appears at least twice
                and word not in self.stopwords  # Not a stopword
                and len(word) > 3  # At least 4 characters (more meaningful terms)
            ):
                entities.add(word)

        return entities

    def _find_and_add_connections(self, unit: KnowledgeUnit) -> None:
        """
        Find and add connections between a knowledge unit and existing units.

        Args:
            unit: Knowledge unit to find connections for
        """
        # Extract entities from the unit
        entities = self._extract_entities(unit.original_chunk)

        # Track which units are connected through entities
        connected_units = Counter()

        # Find units that share entities
        for entity in entities:
            for other_id in self.entity_to_units.get(entity, set()):
                if other_id != unit.unique_id:
                    connected_units[other_id] += 1

        # Add connections for units that share significant entities
        for other_id, count in connected_units.items():
            if count >= 2:  # At least 2 shared entities
                # Connection strength based on number of shared entities
                strength = min(1.0, count / 10)  # Cap at 1.0, scale from count

                # Add bidirectional connection
                self._add_connection(unit.unique_id, other_id, "shared_entities", strength)

    def _add_connection(
        self, from_id: str, to_id: str, relation_type: str, weight: float = 1.0
    ) -> None:
        """
        Add a connection (edge) between two knowledge units.

        Args:
            from_id: ID of the source knowledge unit
            to_id: ID of the target knowledge unit
            relation_type: Type of relationship
            weight: Weight/strength of the connection (0.0 to 1.0)
        """
        # Check if both nodes exist
        if not (self.graph.has_node(from_id) and self.graph.has_node(to_id)):
            logger.warning(f"Cannot add connection, nodes don't exist: {from_id} -> {to_id}")
            return

        # Add or update the edge with its attributes
        if self.graph.has_edge(from_id, to_id):
            # Get existing edge data
            edge_data = self.graph[from_id][to_id]

            # If same relation type, use the maximum weight
            if edge_data.get("relation_type") == relation_type:
                edge_data["weight"] = max(edge_data.get("weight", 0.0), weight)
            else:
                # Different relation type, add it to the relation types list
                relation_types = edge_data.get("relation_types", [])
                if relation_type not in relation_types:
                    relation_types.append(relation_type)

                edge_data["relation_types"] = relation_types
                # Use new weight if it's higher
                edge_data["weight"] = max(edge_data.get("weight", 0.0), weight)
        else:
            # Add new edge
            self.graph.add_edge(
                from_id,
                to_id,
                relation_type=relation_type,
                relation_types=[relation_type],
                weight=weight,
                created_at=datetime.now().isoformat(),
            )

        # Add reverse connection if it doesn't create a conflict
        # This ensures we can traverse the graph in both directions
        if not self.graph.has_edge(to_id, from_id):
            self.graph.add_edge(
                to_id,
                from_id,
                relation_type=f"inverse_{relation_type}",
                relation_types=[f"inverse_{relation_type}"],
                weight=weight * 0.8,  # Slightly lower weight for inverse relation
                created_at=datetime.now().isoformat(),
            )

    def get_connected_units(
        self, unit_id: str, max_distance: int = 2, min_weight: float = 0.3
    ) -> list[tuple[str, float, int]]:
        """
        Get knowledge units connected to the given unit.

        Args:
            unit_id: ID of the knowledge unit to find connections for
            max_distance: Maximum path length to consider
            min_weight: Minimum connection weight to include

        Returns:
            List of tuples (unit_id, relevance_score, distance)
        """
        if not self.graph.has_node(unit_id):
            logger.warning(f"Knowledge unit {unit_id} not found in graph")
            return []

        connected = []
        visited = {unit_id}

        # BFS to find connected units up to max_distance
        frontier = [(unit_id, 0, 1.0)]  # (node_id, distance, accumulated_weight)

        while frontier:
            current_id, distance, acc_weight = frontier.pop(0)

            if distance > 0:  # Don't include the starting node
                connected.append((current_id, acc_weight, distance))

            if distance < max_distance:
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited:
                        # Get edge weight
                        edge_weight = self.graph[current_id][neighbor].get("weight", 0.0)

                        # Calculate new accumulated weight
                        new_weight = acc_weight * edge_weight

                        # Only explore if the connection is strong enough
                        if new_weight >= min_weight:
                            visited.add(neighbor)
                            frontier.append((neighbor, distance + 1, new_weight))

        # Sort by relevance (highest first) and then by distance (closest first)
        connected.sort(key=lambda x: (-x[1], x[2]))

        return connected

    def find_paths(
        self, from_id: str, to_id: str, max_length: int = 3
    ) -> list[list[tuple[str, str]]]:
        """
        Find paths between two knowledge units.

        Args:
            from_id: ID of the source knowledge unit
            to_id: ID of the target knowledge unit
            max_length: Maximum path length

        Returns:
            List of paths, where each path is a list of (node_id, relation_type) tuples
        """
        if not (self.graph.has_node(from_id) and self.graph.has_node(to_id)):
            return []

        # Find all simple paths up to max_length
        try:
            simple_paths = list(
                nx.all_simple_paths(self.graph, source=from_id, target=to_id, cutoff=max_length)
            )
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []

        # Format paths with node IDs and relation types
        formatted_paths = []

        for path in simple_paths:
            formatted_path = []

            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]

                # Get the relation type (primary one)
                relation = self.graph[current][next_node].get("relation_type", "related")

                # Add node with its relation to the next node
                formatted_path.append((current, relation))

            # Add the target node (with no outgoing relation)
            formatted_path.append((to_id, None))
            formatted_paths.append(formatted_path)

        return formatted_paths

    def get_common_connections(
        self, unit_ids: list[str], max_results: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find knowledge units that are connected to multiple input units.

        Args:
            unit_ids: List of knowledge unit IDs
            max_results: Maximum number of results to return

        Returns:
            List of tuples (unit_id, relevance_score)
        """
        if not unit_ids or len(unit_ids) < 2:
            return []

        # Get connected units for each input unit
        all_connected = {}

        for unit_id in unit_ids:
            connected = self.get_connected_units(unit_id)
            connected_dict = {unit: (score, dist) for unit, score, dist in connected}
            all_connected[unit_id] = connected_dict

        # Find units connected to multiple input units
        connection_counts = Counter()
        cumulative_scores = defaultdict(float)

        for connections in all_connected.values():
            for unit, (score, _) in connections.items():
                if unit not in unit_ids:  # Don't include the input units
                    connection_counts[unit] += 1
                    cumulative_scores[unit] += score

        # Calculate final scores - based on both connection count and strength
        final_scores = []
        for unit, count in connection_counts.items():
            if count > 1:  # Connected to at least 2 input units
                # Score = average connection strength * (count / total_inputs)
                score = (cumulative_scores[unit] / count) * (count / len(unit_ids))
                final_scores.append((unit, score))

        # Sort by score (highest first) and limit results
        final_scores.sort(key=lambda x: -x[1])
        return final_scores[:max_results]

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict with graph statistics
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_entities": len(self.entity_to_units),
            "avg_connections_per_node": (
                self.graph.number_of_edges() / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0
                else 0
            ),
            "strongly_connected_components": nx.number_strongly_connected_components(self.graph),
            "weakly_connected_components": nx.number_weakly_connected_components(self.graph),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the knowledge graph to a dictionary representation.

        Returns:
            Dict with serializable graph data
        """
        nodes = []
        for node_id in self.graph.nodes:
            nodes.append({"id": node_id, **self.graph.nodes[node_id]})

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})

        return {"nodes": nodes, "edges": edges, "stats": self.get_stats()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """
        Create a knowledge graph from a dictionary representation.

        Args:
            data: Dict with graph data

        Returns:
            KnowledgeGraph instance
        """
        graph = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.graph.add_node(node_id, **node_data)

            # Rebuild entity mapping
            if "content" in node_data:
                entities = graph._extract_entities(node_data["content"])
                for entity in entities:
                    graph.entity_to_units[entity].add(node_id)

        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.graph.add_edge(source, target, **edge_data)

        return graph
