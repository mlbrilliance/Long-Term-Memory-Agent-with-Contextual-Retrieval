#!/usr/bin/env python
"""
Knowledge Graph

Manages relationships between memory units for enhanced contextual retrieval.
"""

import re
from typing import Any

from .memory_unit import MemoryUnit


class KnowledgeGraph:
    """Knowledge graph for tracking relationships between memory units."""

    def __init__(self):
        """Initialize the knowledge graph."""
        self.units = {}  # id -> MemoryUnit
        self.concept_index = {}  # concept -> list of unit ids
        self.entity_index = {}  # entity type -> entity -> list of unit ids
        self.relation_types = {
            # Family relations
            "parent_of": {"inverse": "child_of", "transitive": False},
            "child_of": {"inverse": "parent_of", "transitive": False},
            "sibling_of": {"inverse": "sibling_of", "transitive": False},
            "spouse_of": {"inverse": "spouse_of", "transitive": False},
            "aunt_of": {"inverse": "niece_nephew_of", "transitive": False},
            "uncle_of": {"inverse": "niece_nephew_of", "transitive": False},
            "niece_nephew_of": {"inverse": "aunt_uncle_of", "transitive": False},
            # Hierarchical relations
            "subset_of": {"inverse": "superset_of", "transitive": True},
            "superset_of": {"inverse": "subset_of", "transitive": True},
            "instance_of": {"inverse": "has_instance", "transitive": False},
            "part_of": {"inverse": "contains", "transitive": True},
            "contains": {"inverse": "part_of", "transitive": True},
            # General relations
            "related_to": {"inverse": "related_to", "transitive": False},
            "corrects": {"inverse": "corrected_by", "transitive": False},
            "contradicts": {"inverse": "contradicted_by", "transitive": False},
            "follows": {"inverse": "precedes", "transitive": True},
            "precedes": {"inverse": "follows", "transitive": True},
        }

        # Inference rules for derived relationships
        self.inference_rules = {
            # Family relationship inferences
            (
                "sibling_of",
                "parent_of",
            ): "aunt_uncle_of",  # If X is sibling of Y and Y is parent of Z, then X is aunt/uncle of Z
            (
                "parent_of",
                "parent_of",
            ): "grandparent_of",  # If X is parent of Y and Y is parent of Z, then X is grandparent of Z
            # Hierarchical relationship inferences (transitive)
            (
                "subset_of",
                "subset_of",
            ): "subset_of",  # If A is subset of B and B is subset of C, then A is subset of C
            (
                "part_of",
                "part_of",
            ): "part_of",  # If A is part of B and B is part of C, then A is part of C
        }

    def add_unit(self, unit: MemoryUnit) -> str:
        """
        Add a memory unit to the graph.

        Args:
            unit: The memory unit to add

        Returns:
            The ID of the added unit
        """
        # Add to units
        self.units[unit.id] = unit

        # Update concept index
        for concept in unit.concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(unit.id)

        # Update entity index
        for entity_type, entities in unit.entities.items():
            if entity_type not in self.entity_index:
                self.entity_index[entity_type] = {}

            for entity in entities:
                if entity not in self.entity_index[entity_type]:
                    self.entity_index[entity_type][entity] = []
                self.entity_index[entity_type][entity].append(unit.id)

        # Analyze for relationships with existing units
        self._analyze_relationships(unit)

        return unit.id

    def _analyze_relationships(self, unit: MemoryUnit) -> None:
        """
        Analyze the memory unit for relationships with existing units.

        Args:
            unit: The memory unit to analyze
        """
        # Check for family relationships
        self._analyze_family_relationships(unit)

        # Check for hierarchical relationships
        self._analyze_hierarchical_relationships(unit)

        # Check for corrections
        if unit.type == "correction" and "corrects_id" in unit.metadata:
            corrected_id = unit.metadata["corrects_id"]
            if corrected_id in self.units:
                unit.add_relation(corrected_id, "corrects")
                self.units[corrected_id].add_relation(unit.id, "corrected_by")

        # Check for temporal relationships
        self._analyze_temporal_relationships(unit)

        # Check for general concept relationships
        self._analyze_concept_relationships(unit)

    def _analyze_family_relationships(self, unit: MemoryUnit) -> None:
        """
        Analyze the memory unit for family relationships.

        Args:
            unit: The memory unit to analyze
        """
        content_lower = unit.content.lower()

        # Family relationship patterns
        patterns = {
            "parent_of": [
                r"(\w+) is (\w+)'s (father|mother|parent)",
                r"(\w+) is the (father|mother|parent) of (\w+)",
            ],
            "child_of": [
                r"(\w+) is (\w+)'s (son|daughter|child)",
                r"(\w+) is the (son|daughter|child) of (\w+)",
            ],
            "sibling_of": [
                r"(\w+) is (\w+)'s (brother|sister|sibling)",
                r"(\w+) is the (brother|sister|sibling) of (\w+)",
            ],
        }

        for relation_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, content_lower)

                for match in matches:
                    # Handle based on pattern type
                    if len(match) == 3:  # First pattern type
                        if "is" in pattern and "'s" in pattern:
                            # Pattern: "X is Y's father/mother/etc."
                            entity1 = match[0].capitalize()  # X
                            entity2 = match[1].capitalize()  # Y

                            # Find units containing these entities
                            self._create_entity_relationship(unit, entity1, entity2, relation_type)

    def _analyze_hierarchical_relationships(self, unit: MemoryUnit) -> None:
        """
        Analyze the memory unit for hierarchical relationships.

        Args:
            unit: The memory unit to analyze
        """
        content_lower = unit.content.lower()

        # Hierarchical relationship patterns
        patterns = {
            "subset_of": [
                r"(\w+) is a subset of (\w+)",
                r"(\w+) is a type of (\w+)",
                r"(\w+) is a kind of (\w+)",
            ],
            "part_of": [
                r"(\w+) is part of (\w+)",
                r"(\w+) is contained in (\w+)",
                r"(\w+) belongs to (\w+)",
            ],
        }

        for relation_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, content_lower)

                for match in matches:
                    entity1 = match[0]
                    entity2 = match[1]

                    # Find units containing these concepts
                    for concept1_unit_id in self.concept_index.get(entity1, []):
                        for concept2_unit_id in self.concept_index.get(entity2, []):
                            # Add relationship between units
                            if concept1_unit_id != concept2_unit_id:
                                self.units[concept1_unit_id].add_relation(
                                    concept2_unit_id, relation_type
                                )
                                self.units[concept2_unit_id].add_relation(
                                    concept1_unit_id, self.relation_types[relation_type]["inverse"]
                                )

    def _analyze_temporal_relationships(self, unit: MemoryUnit) -> None:
        """
        Analyze the memory unit for temporal relationships.

        Args:
            unit: The memory unit to analyze
        """
        # Look for time entities
        for time in unit.entities.get("time", []):
            # Find other units with the same time reference
            for other_unit_id in self.entity_index.get("time", {}).get(time, []):
                if other_unit_id != unit.id:
                    # Add temporal relationship
                    unit.add_relation(other_unit_id, "related_to")
                    self.units[other_unit_id].add_relation(unit.id, "related_to")

    def _analyze_concept_relationships(self, unit: MemoryUnit) -> None:
        """
        Analyze the memory unit for concept relationships.

        Args:
            unit: The memory unit to analyze
        """
        # Find units with overlapping concepts
        for concept in unit.concepts:
            # Skip very common concepts
            if concept in ["know", "like", "use", "have", "make"]:
                continue

            for other_unit_id in self.concept_index.get(concept, []):
                if other_unit_id != unit.id:
                    # If they share multiple concepts, they're more likely related
                    shared_concepts = unit.concepts.intersection(self.units[other_unit_id].concepts)

                    if len(shared_concepts) >= 2:
                        unit.add_relation(other_unit_id, "related_to")
                        self.units[other_unit_id].add_relation(unit.id, "related_to")

    def _create_entity_relationship(
        self, unit: MemoryUnit, entity1: str, entity2: str, relation_type: str
    ) -> None:
        """
        Create a relationship between entities mentioned in the unit.

        Args:
            unit: The memory unit
            entity1: The first entity
            entity2: The second entity
            relation_type: The type of relationship
        """
        # Find units containing these entities
        entity1_units = []
        entity2_units = []

        for other_unit_id, other_unit in self.units.items():
            # Check if the unit mentions entity1
            if entity1 in other_unit.content:
                entity1_units.append(other_unit_id)

            # Check if the unit mentions entity2
            if entity2 in other_unit.content:
                entity2_units.append(other_unit_id)

        # Also include the current unit if it mentions either entity
        if entity1 in unit.content:
            entity1_units.append(unit.id)
        if entity2 in unit.content:
            entity2_units.append(unit.id)

        # Create relationships between units
        for e1_unit_id in entity1_units:
            for e2_unit_id in entity2_units:
                if e1_unit_id != e2_unit_id:
                    self.units[e1_unit_id].add_relation(e2_unit_id, relation_type)
                    self.units[e2_unit_id].add_relation(
                        e1_unit_id, self.relation_types[relation_type]["inverse"]
                    )

    def get_unit(self, unit_id: str) -> MemoryUnit | None:
        """
        Get a memory unit by ID.

        Args:
            unit_id: The ID of the memory unit

        Returns:
            The memory unit, or None if not found
        """
        return self.units.get(unit_id)

    def get_related_units(self, unit_id: str, relation_type: str | None = None) -> list[MemoryUnit]:
        """
        Get units related to a given unit.

        Args:
            unit_id: The ID of the memory unit
            relation_type: The type of relationship to filter by (optional)

        Returns:
            A list of related memory units
        """
        if unit_id not in self.units:
            return []

        unit = self.units[unit_id]
        related_units = []

        for relation in unit.relations:
            if relation_type is None or relation["type"] == relation_type:
                related_unit = self.units.get(relation["target_id"])
                if related_unit:
                    related_units.append(related_unit)

        return related_units

    def find_units_by_concept(self, concept: str) -> list[MemoryUnit]:
        """
        Find memory units by concept.

        Args:
            concept: The concept to search for

        Returns:
            A list of memory units containing the concept
        """
        unit_ids = self.concept_index.get(concept, [])
        return [self.units[unit_id] for unit_id in unit_ids]

    def find_units_by_entity(self, entity: str, entity_type: str | None = None) -> list[MemoryUnit]:
        """
        Find memory units by entity.

        Args:
            entity: The entity to search for
            entity_type: The type of entity to filter by (optional)

        Returns:
            A list of memory units containing the entity
        """
        if entity_type:
            unit_ids = self.entity_index.get(entity_type, {}).get(entity, [])
        else:
            # Search across all entity types
            unit_ids = []
            for type_index in self.entity_index.values():
                unit_ids.extend(type_index.get(entity, []))

        return [self.units[unit_id] for unit_id in unit_ids]

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3,
        relation_types: list[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Find a path between two memory units.

        Args:
            start_id: The ID of the start unit
            end_id: The ID of the end unit
            max_depth: The maximum depth to search
            relation_types: The types of relationships to consider (optional)

        Returns:
            A list of path steps, or None if no path is found
        """
        if start_id not in self.units or end_id not in self.units:
            return None

        visited = set()

        def dfs(current_id, path, depth):
            if depth > max_depth:
                return None

            if current_id == end_id:
                return path

            if current_id in visited:
                return None

            visited.add(current_id)

            for relation in self.units[current_id].relations:
                target_id = relation["target_id"]
                relation_type = relation["type"]

                # Skip if not in specified relation types
                if relation_types and relation_type not in relation_types:
                    continue

                new_path = path + [
                    {"from_id": current_id, "to_id": target_id, "relation": relation_type}
                ]

                result = dfs(target_id, new_path, depth + 1)
                if result:
                    return result

            return None

        return dfs(start_id, [], 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert the knowledge graph to a dictionary."""
        return {
            "units": {unit_id: unit.to_dict() for unit_id, unit in self.units.items()},
            "relation_types": self.relation_types,
            "inference_rules": self.inference_rules,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Create a knowledge graph from a dictionary."""
        graph = cls()
        graph.relation_types = data["relation_types"]
        graph.inference_rules = data.get("inference_rules", {})

        # Recreate units
        for unit_data in data["units"].values():
            unit = MemoryUnit.from_dict(unit_data)
            graph.units[unit.id] = unit

        # Rebuild indexes
        for unit_id, unit in graph.units.items():
            # Update concept index
            for concept in unit.concepts:
                if concept not in graph.concept_index:
                    graph.concept_index[concept] = []
                graph.concept_index[concept].append(unit_id)

            # Update entity index
            for entity_type, entities in unit.entities.items():
                if entity_type not in graph.entity_index:
                    graph.entity_index[entity_type] = {}

                for entity in entities:
                    if entity not in graph.entity_index[entity_type]:
                        graph.entity_index[entity_type][entity] = []
                    graph.entity_index[entity_type][entity].append(unit_id)

        return graph
