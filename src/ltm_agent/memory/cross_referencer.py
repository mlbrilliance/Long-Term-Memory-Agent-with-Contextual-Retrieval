"""
Cross-referencer module for establishing relationships between knowledge units.
"""

import logging
from datetime import datetime, timezone

from ltm_agent.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class KnowledgeCrossReferencer:
    """
    Create and maintain cross-references between related knowledge units
    to improve multi-step learning and knowledge connectivity.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the cross-referencer with a memory manager.

        Args:
            memory_manager: The memory manager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

    async def create_references(
        self, knowledge_id: str, similarity_threshold: float = 0.4
    ) -> list[str]:
        """
        Create bidirectional references between a knowledge unit and its related units.

        Args:
            knowledge_id: The ID of the knowledge unit to cross-reference
            similarity_threshold: Threshold for considering units related (0.0 to 1.0)

        Returns:
            List[str]: List of IDs of related knowledge units that were cross-referenced
        """
        # Get the target knowledge unit
        knowledge_unit = await self.memory_manager.retrieve_knowledge(knowledge_id)
        if not knowledge_unit:
            self.logger.warning(
                f"Knowledge unit {knowledge_id} not found, cannot create references"
            )
            return []

        # Find related knowledge units
        related_units = await self.memory_manager.get_related_knowledge(
            content=knowledge_unit.original_chunk,
            limit=10,
            min_similarity_score=similarity_threshold,
        )

        # Filter out the unit itself
        related_units = [
            (unit, score) for unit, score in related_units if unit.unique_id != knowledge_id
        ]

        if not related_units:
            self.logger.info(f"No related units found for knowledge unit {knowledge_id}")
            return []

        # Update the target unit with references to related units
        knowledge_metadata = knowledge_unit.metadata or {}
        if "related_units" not in knowledge_metadata:
            knowledge_metadata["related_units"] = []

        # Track which units we've created references to
        referenced_units = []

        for related_unit, score in related_units:
            # Add reference to the related unit
            if related_unit.unique_id not in knowledge_metadata["related_units"]:
                knowledge_metadata["related_units"].append(related_unit.unique_id)

                # Also add relationship metadata
                if "relationships" not in knowledge_metadata:
                    knowledge_metadata["relationships"] = {}

                knowledge_metadata["relationships"][related_unit.unique_id] = {
                    "similarity": score,
                    "established_at": datetime.now(timezone.utc).isoformat(),
                }

                referenced_units.append(related_unit.unique_id)

                # Now update the related unit with a reference back to the target unit
                related_metadata = related_unit.metadata or {}
                if "related_units" not in related_metadata:
                    related_metadata["related_units"] = []

                if knowledge_id not in related_metadata["related_units"]:
                    related_metadata["related_units"].append(knowledge_id)

                    # Add relationship metadata
                    if "relationships" not in related_metadata:
                        related_metadata["relationships"] = {}

                    related_metadata["relationships"][knowledge_id] = {
                        "similarity": score,
                        "established_at": datetime.now(timezone.utc).isoformat(),
                    }

                    # Update the related unit
                    await self.memory_manager.update_knowledge(
                        unique_id=related_unit.unique_id, metadata=related_metadata
                    )

        # Update the target unit with all the new references
        if referenced_units:
            await self.memory_manager.update_knowledge(
                unique_id=knowledge_id, metadata=knowledge_metadata
            )

        self.logger.info(
            f"Created cross-references for knowledge unit {knowledge_id} with {len(referenced_units)} related units"
        )
        return referenced_units

    async def create_domain_links(
        self, domain_keywords: dict[str, list[str]], min_matching_keywords: int = 2
    ) -> dict[str, list[str]]:
        """
        Create links between knowledge units within the same domain based on keywords.

        Args:
            domain_keywords: Dictionary mapping domain names to lists of keywords
            min_matching_keywords: Minimum number of keywords that must match to link to a domain

        Returns:
            Dict[str, List[str]]: Dictionary mapping domains to lists of knowledge unit IDs
        """
        # Get all knowledge units
        all_units = await self.memory_manager.list_knowledge()

        # Map for tracking which units belong to which domains
        domain_units: dict[str, list[str]] = {domain: [] for domain in domain_keywords}

        # Process each knowledge unit
        for unit in all_units:
            # Check which domains this unit might belong to
            unit_text = unit.original_chunk.lower()

            for domain, keywords in domain_keywords.items():
                # Count how many keywords match
                matches = sum(1 for keyword in keywords if keyword.lower() in unit_text)

                if matches >= min_matching_keywords:
                    # This unit belongs to this domain
                    domain_units[domain].append(unit.unique_id)

                    # Update the unit's metadata to indicate domain membership
                    unit_metadata = unit.metadata or {}
                    if "domains" not in unit_metadata:
                        unit_metadata["domains"] = []

                    if domain not in unit_metadata["domains"]:
                        unit_metadata["domains"].append(domain)

                        # Update the unit
                        await self.memory_manager.update_knowledge(
                            unique_id=unit.unique_id, metadata=unit_metadata
                        )

        # Now that we've identified domains, create cross-references within each domain
        for domain, unit_ids in domain_units.items():
            if len(unit_ids) > 1:  # Only proceed if we have multiple units in this domain
                self.logger.info(
                    f"Creating cross-references within domain '{domain}' for {len(unit_ids)} units"
                )

                # Create cross-references between all units in this domain
                for unit_id in unit_ids:
                    await self.create_references(unit_id, similarity_threshold=0.3)

        return domain_units

    async def enhance_multi_step_learning(
        self, query_id: str, response_id: str, follow_up_id: str = None
    ) -> bool:
        """
        Create explicit linkages for multi-step learning scenarios.

        Args:
            query_id: ID of the knowledge unit for the initial query
            response_id: ID of the knowledge unit for the response
            follow_up_id: Optional ID of a follow-up query/response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the knowledge units
            query_unit = await self.memory_manager.retrieve_knowledge(query_id)
            response_unit = await self.memory_manager.retrieve_knowledge(response_id)

            if not query_unit or not response_unit:
                self.logger.warning("Could not retrieve query or response units")
                return False

            # Create a sequence relationship
            query_metadata = query_unit.metadata or {}
            if "sequences" not in query_metadata:
                query_metadata["sequences"] = []

            # Add or update sequence info
            sequence_info = {
                "type": "query_response",
                "next_id": response_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Check if this sequence already exists
            exists = False
            for seq in query_metadata.get("sequences", []):
                if seq.get("next_id") == response_id:
                    exists = True
                    break

            if not exists:
                query_metadata["sequences"].append(sequence_info)

                # Update the query unit
                await self.memory_manager.update_knowledge(
                    unique_id=query_id, metadata=query_metadata
                )

            # Update the response unit to point back to the query
            response_metadata = response_unit.metadata or {}
            if "sequences" not in response_metadata:
                response_metadata["sequences"] = []

            # Add or update sequence info
            back_sequence_info = {
                "type": "response_to_query",
                "previous_id": query_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Check if this sequence already exists
            exists = False
            for seq in response_metadata.get("sequences", []):
                if seq.get("previous_id") == query_id:
                    exists = True
                    break

            if not exists:
                response_metadata["sequences"].append(back_sequence_info)

                # Update the response unit
                await self.memory_manager.update_knowledge(
                    unique_id=response_id, metadata=response_metadata
                )

            # If we have a follow-up, link it as well
            if follow_up_id:
                follow_up_unit = await self.memory_manager.retrieve_knowledge(follow_up_id)
                if follow_up_unit:
                    # Link response to follow-up
                    if "sequences" not in response_metadata:
                        response_metadata["sequences"] = []

                    follow_up_sequence = {
                        "type": "response_follow_up",
                        "next_id": follow_up_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                    # Check if this sequence already exists
                    exists = False
                    for seq in response_metadata.get("sequences", []):
                        if seq.get("next_id") == follow_up_id:
                            exists = True
                            break

                    if not exists:
                        response_metadata["sequences"].append(follow_up_sequence)

                        # Update the response unit again
                        await self.memory_manager.update_knowledge(
                            unique_id=response_id, metadata=response_metadata
                        )

                    # Link follow-up back to response
                    follow_up_metadata = follow_up_unit.metadata or {}
                    if "sequences" not in follow_up_metadata:
                        follow_up_metadata["sequences"] = []

                    back_follow_up_sequence = {
                        "type": "follow_up_to_response",
                        "previous_id": response_id,
                        "query_id": query_id,  # Also link to the original query
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                    # Check if this sequence already exists
                    exists = False
                    for seq in follow_up_metadata.get("sequences", []):
                        if seq.get("previous_id") == response_id:
                            exists = True
                            break

                    if not exists:
                        follow_up_metadata["sequences"].append(back_follow_up_sequence)

                        # Update the follow-up unit
                        await self.memory_manager.update_knowledge(
                            unique_id=follow_up_id, metadata=follow_up_metadata
                        )

            self.logger.info(
                f"Enhanced multi-step learning between units {query_id}, {response_id}, and {follow_up_id or 'None'}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error enhancing multi-step learning: {str(e)}")
            return False
