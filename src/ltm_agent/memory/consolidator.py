"""
Memory consolidation module for periodically reviewing and refining knowledge connections.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.cross_referencer import KnowledgeCrossReferencer
from ltm_agent.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Periodically reviews and refines connections between knowledge units
    to improve recall, update accuracy, and task success.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        cross_referencer: KnowledgeCrossReferencer | None = None,
        consolidation_interval: int = 24,
    ):
        """
        Initialize the memory consolidator.

        Args:
            memory_manager: The memory manager instance
            cross_referencer: Optional cross-referencer instance (will create if None)
            consolidation_interval: Hours between consolidation runs
        """
        self.memory_manager = memory_manager
        self.cross_referencer = cross_referencer or KnowledgeCrossReferencer(memory_manager)
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = None
        self.is_consolidating = False
        self.logger = logging.getLogger(__name__)

    async def start_background_consolidation(self):
        """
        Start background consolidation task.
        """
        asyncio.create_task(self._consolidation_loop())

    async def _consolidation_loop(self):
        """
        Background loop for periodic consolidation.
        """
        while True:
            try:
                # Check if it's time to consolidate
                current_time = datetime.now(timezone.utc)
                if self.last_consolidation is None or (
                    current_time - self.last_consolidation
                ) > timedelta(hours=self.consolidation_interval):
                    # Run consolidation
                    await self.consolidate_memory()
                    self.last_consolidation = current_time

                # Sleep before checking again
                await asyncio.sleep(60 * 60)  # Check every hour

            except Exception as e:
                self.logger.error(f"Error in consolidation loop: {str(e)}")
                await asyncio.sleep(60 * 15)  # Wait 15 minutes if there's an error

    async def consolidate_memory(self):
        """
        Perform a full memory consolidation.
        """
        if self.is_consolidating:
            self.logger.info("Consolidation already in progress, skipping")
            return

        self.is_consolidating = True
        try:
            self.logger.info("Starting memory consolidation")

            # Get all knowledge units
            all_units = await self.memory_manager.list_knowledge()

            if not all_units:
                self.logger.info("No knowledge units found, skipping consolidation")
                return

            self.logger.info(f"Found {len(all_units)} knowledge units for consolidation")

            # 1. Identify and merge duplicate or highly similar knowledge
            await self._merge_duplicates(all_units)

            # 2. Create cross-references for recently added or updated knowledge
            recent_units = [
                unit
                for unit in all_units
                if (datetime.now(timezone.utc) - unit.timestamp < timedelta(days=7))
            ]

            for unit in recent_units:
                await self.cross_referencer.create_references(unit.unique_id)

            # 3. Group knowledge into domains for better organization
            domains = {
                "geography": [
                    "city",
                    "country",
                    "capital",
                    "place",
                    "location",
                    "europe",
                    "asia",
                    "america",
                ],
                "technology": [
                    "computer",
                    "software",
                    "programming",
                    "algorithm",
                    "machine learning",
                    "AI",
                ],
                "science": ["physics", "chemistry", "biology", "research", "experiment", "theory"],
                "history": ["ancient", "century", "historical", "war", "civilization", "period"],
            }

            await self.cross_referencer.create_domain_links(domains)

            # 4. Strengthen multi-step learning connections
            await self._strengthen_multi_step_connections(all_units)

            # 5. Refresh embeddings for a sample of older knowledge to prevent drift
            await self._refresh_embeddings(all_units)

            self.logger.info("Memory consolidation completed successfully")

        except Exception as e:
            self.logger.error(f"Error during memory consolidation: {str(e)}")
        finally:
            self.is_consolidating = False

    async def _merge_duplicates(self, all_units: list[KnowledgeUnit]):
        """
        Identify and merge duplicate or highly similar knowledge units.

        Args:
            all_units: List of all knowledge units
        """
        # Group by high similarity for potential merging
        # This requires O(n²) comparisons, so we'll limit to newer units
        # for performance in large knowledge bases
        recent_units = [
            unit
            for unit in all_units
            if datetime.now(timezone.utc) - unit.timestamp < timedelta(days=14)
        ]

        merged_count = 0

        for i, unit1 in enumerate(recent_units):
            for unit2 in recent_units[i + 1 :]:
                # Skip if either unit has already been merged in this session
                if not await self.memory_manager.retrieve_knowledge(
                    unit1.unique_id
                ) or not await self.memory_manager.retrieve_knowledge(unit2.unique_id):
                    continue

                # Check similarity between units
                similarity = await self._compute_unit_similarity(unit1, unit2)

                if similarity > 0.85:  # Very high similarity threshold for merging
                    self.logger.info(
                        f"Found highly similar units: {unit1.unique_id} and {unit2.unique_id}"
                    )

                    # Merge the units
                    # We keep the older unit and merge the newer unit into it
                    if unit1.timestamp < unit2.timestamp:
                        base_unit, merge_unit = unit1, unit2
                    else:
                        base_unit, merge_unit = unit2, unit1

                    # Use the memory manager's merge function
                    merged_content = self.memory_manager._merge_content(
                        base_unit.original_chunk, merge_unit.original_chunk
                    )

                    # Merge metadata
                    merged_metadata = base_unit.metadata or {}
                    merge_source_metadata = merge_unit.metadata or {}

                    for key, value in merge_source_metadata.items():
                        if key not in merged_metadata:
                            merged_metadata[key] = value
                        elif key == "related_units" and isinstance(value, list):
                            # For related units, combine the lists
                            merged_metadata[key] = list(set(merged_metadata[key] + value))
                        elif key == "relationships" and isinstance(value, dict):
                            # For relationships, merge the dictionaries
                            if "relationships" not in merged_metadata:
                                merged_metadata["relationships"] = {}
                            merged_metadata["relationships"].update(value)

                    # Record the merge in metadata
                    if "merges" not in merged_metadata:
                        merged_metadata["merges"] = []

                    merged_metadata["merges"].append(
                        {
                            "merged_id": merge_unit.unique_id,
                            "merged_at": datetime.now(timezone.utc).isoformat(),
                            "similarity": similarity,
                        }
                    )

                    # Update the base unit with merged content and metadata
                    await self.memory_manager.update_knowledge(
                        unique_id=base_unit.unique_id,
                        content=merged_content,
                        metadata=merged_metadata,
                    )

                    # Update any references to the merged unit
                    for unit in all_units:
                        if (
                            unit.unique_id != base_unit.unique_id
                            and unit.unique_id != merge_unit.unique_id
                        ):
                            unit_metadata = unit.metadata or {}

                            # Check related_units references
                            if (
                                "related_units" in unit_metadata
                                and merge_unit.unique_id in unit_metadata["related_units"]
                            ):
                                # Replace reference to merged unit with reference to base unit
                                unit_metadata["related_units"] = [
                                    base_unit.unique_id if id == merge_unit.unique_id else id
                                    for id in unit_metadata["related_units"]
                                ]

                                # Remove duplicates
                                unit_metadata["related_units"] = list(
                                    set(unit_metadata["related_units"])
                                )

                                # Update the unit
                                await self.memory_manager.update_knowledge(
                                    unique_id=unit.unique_id, metadata=unit_metadata
                                )

                    # Delete the merged unit
                    await self.memory_manager.delete_knowledge(merge_unit.unique_id)

                    merged_count += 1

        self.logger.info(f"Merged {merged_count} duplicate knowledge units")

    async def _compute_unit_similarity(self, unit1: KnowledgeUnit, unit2: KnowledgeUnit) -> float:
        """
        Compute similarity between two knowledge units using both content and embeddings.

        Args:
            unit1: First knowledge unit
            unit2: Second knowledge unit

        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        # Start with embedding similarity if available
        embedding_similarity = 0.0
        if unit1.embedding_vector and unit2.embedding_vector:
            try:
                import numpy as np

                # Convert to numpy arrays
                vec1 = np.array(unit1.embedding_vector)
                vec2 = np.array(unit2.embedding_vector)

                # Normalize vectors
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)

                # Compute cosine similarity
                embedding_similarity = float(np.dot(vec1, vec2))
            except Exception as e:
                self.logger.warning(f"Error computing embedding similarity: {str(e)}")

        # Compute text similarity using Jaccard similarity on word sets
        text1 = f"{unit1.contextual_text} {unit1.original_chunk}".lower()
        text2 = f"{unit2.contextual_text} {unit2.original_chunk}".lower()

        # Remove punctuation and split into words
        import re

        words1 = set(re.sub(r"[^\w\s]", " ", text1).split())
        words2 = set(re.sub(r"[^\w\s]", " ", text2).split())

        # Filter out common stop words to reduce noise
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "for",
            "with",
            "in",
            "on",
            "at",
            "by",
            "this",
            "that",
        }

        significant_words1 = words1 - stop_words
        significant_words2 = words2 - stop_words

        # Compute Jaccard similarity
        text_similarity = 0.0
        if significant_words1 and significant_words2:
            intersection = significant_words1.intersection(significant_words2)
            union = significant_words1.union(significant_words2)

            # Weighted by word length (gives more importance to longer, more specific terms)
            intersection_weight = sum(len(word) for word in intersection) / max(
                1, sum(len(word) for word in union)
            )

            # Basic Jaccard
            basic_jaccard = len(intersection) / max(1, len(union))

            # Combined text similarity score
            text_similarity = 0.7 * basic_jaccard + 0.3 * intersection_weight

        # Look for metadata connections (e.g., if units already reference each other)
        metadata_similarity = 0.0

        unit1_metadata = unit1.metadata or {}
        unit2_metadata = unit2.metadata or {}

        # Check if units already reference each other
        if (
            "related_units" in unit1_metadata
            and unit2.unique_id in unit1_metadata.get("related_units", [])
        ) or (
            "related_units" in unit2_metadata
            and unit1.unique_id in unit2_metadata.get("related_units", [])
        ):
            metadata_similarity = 0.3

        # Check if units are part of the same sequence
        if "sequences" in unit1_metadata and any(
            unit2.unique_id in [seq.get("next_id", ""), seq.get("previous_id", "")]
            for seq in unit1_metadata.get("sequences", [])
        ):
            metadata_similarity = max(metadata_similarity, 0.4)

        if "sequences" in unit2_metadata and any(
            unit1.unique_id in [seq.get("next_id", ""), seq.get("previous_id", "")]
            for seq in unit2_metadata.get("sequences", [])
        ):
            metadata_similarity = max(metadata_similarity, 0.4)

        # Check for temporal proximity (units created close in time might be related)
        time_diff_hours = abs((unit1.timestamp - unit2.timestamp).total_seconds()) / 3600
        temporal_similarity = max(
            0.0, 0.2 - 0.01 * min(20, time_diff_hours)
        )  # Up to 0.2 for units created within 20 hours

        # Source similarity - units from same source might be related
        source_similarity = 0.0
        if (
            unit1_metadata.get("source") == unit2_metadata.get("source")
            and unit1_metadata.get("source") is not None
        ):
            source_similarity = 0.15

        # Special handling for corrections
        if unit1_metadata.get("type") == "correction" or unit2_metadata.get("type") == "correction":
            # If both are corrections, they might be duplicates
            if (
                unit1_metadata.get("type") == "correction"
                and unit2_metadata.get("type") == "correction"
            ):
                # If they correct the same response, high similarity
                if unit1_metadata.get("corrects_response") == unit2_metadata.get(
                    "corrects_response"
                ):
                    metadata_similarity = max(metadata_similarity, 0.8)

        # Calculate final similarity with weighted components
        # Embeddings are most reliable, then text content, then metadata
        if embedding_similarity > 0:
            final_similarity = (
                0.5 * embedding_similarity
                + 0.3 * text_similarity
                + 0.15 * metadata_similarity
                + 0.05 * max(temporal_similarity, source_similarity)
            )
        else:
            # If no embeddings, rely more on text and metadata
            final_similarity = (
                0.6 * text_similarity
                + 0.25 * metadata_similarity
                + 0.15 * max(temporal_similarity, source_similarity)
            )

        return min(1.0, max(0.0, final_similarity))

    async def _strengthen_multi_step_connections(self, all_units: list[KnowledgeUnit]):
        """
        Strengthen connections for multi-step learning scenarios.

        Args:
            all_units: List of all knowledge units
        """
        # Find units that are part of a query-response-feedback pattern
        for unit in all_units:
            unit_metadata = unit.metadata or {}

            # If this is a query
            if unit_metadata.get("type") == "query" or "query" in unit.original_chunk.lower():
                # Look for responses to this query
                response_candidates = []

                for other_unit in all_units:
                    if other_unit.unique_id == unit.unique_id:
                        continue

                    other_metadata = other_unit.metadata or {}

                    # If this other unit references the query
                    if "related_units" in other_metadata and unit.unique_id in other_metadata.get(
                        "related_units", []
                    ):
                        # Or if it appears to be a response
                        response_candidates.append(other_unit)

                # If we found response candidates
                if response_candidates:
                    # Sort by creation time to find the most likely response
                    # (typically the first response after the query)
                    response_candidates.sort(key=lambda u: u.timestamp)

                    for response_unit in response_candidates:
                        # Look for follow-ups to this response
                        follow_up_candidates = []

                        for other_unit in all_units:
                            if (
                                other_unit.unique_id == unit.unique_id
                                or other_unit.unique_id == response_unit.unique_id
                            ):
                                continue

                            other_metadata = other_unit.metadata or {}

                            # If this other unit references the response
                            if (
                                "related_units" in other_metadata
                                and response_unit.unique_id
                                in other_metadata.get("related_units", [])
                            ):
                                # And was created after the response
                                if other_unit.timestamp > response_unit.timestamp:
                                    follow_up_candidates.append(other_unit)

                        # If we found follow-up candidates
                        if follow_up_candidates:
                            # Sort by creation time to find the most likely follow-up
                            follow_up_candidates.sort(key=lambda u: u.timestamp)
                            follow_up_unit = follow_up_candidates[0]

                            # Create explicit connections between query, response, and follow-up
                            await self.cross_referencer.enhance_multi_step_learning(
                                query_id=unit.unique_id,
                                response_id=response_unit.unique_id,
                                follow_up_id=follow_up_unit.unique_id,
                            )
                        else:
                            # Just create connections between query and response
                            await self.cross_referencer.enhance_multi_step_learning(
                                query_id=unit.unique_id, response_id=response_unit.unique_id
                            )

    async def _refresh_embeddings(self, all_units: list[KnowledgeUnit]):
        """
        Refresh embeddings for a sample of older knowledge to prevent embedding drift.

        Args:
            all_units: List of all knowledge units
        """
        # Sort units by last updated time (oldest first)
        all_units.sort(key=lambda u: u.timestamp)

        # Take the oldest 10% of units, up to 50 units max
        num_to_refresh = min(50, max(1, len(all_units) // 10))
        units_to_refresh = all_units[:num_to_refresh]

        self.logger.info(f"Refreshing embeddings for {len(units_to_refresh)} older knowledge units")

        refresh_count = 0

        for unit in units_to_refresh:
            try:
                # Generate a new embedding for the unit
                new_embedding = await self.memory_manager.memory_store.generate_embedding(
                    unit.original_chunk
                )

                # Update the unit with the new embedding
                unit.embedding_vector = new_embedding
                unit.timestamp = datetime.now(timezone.utc)

                # Update the unit in the memory store
                await self.memory_manager.memory_store.update(unit)

                refresh_count += 1

            except Exception as e:
                self.logger.warning(
                    f"Error refreshing embedding for unit {unit.unique_id}: {str(e)}"
                )

        self.logger.info(f"Successfully refreshed embeddings for {refresh_count} knowledge units")
