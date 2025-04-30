"""
Long-Term Memory Agent.

This module provides the LongTermMemoryAgent class, which integrates
the memory components with the core LLM to implement the main action cycle.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ltm_agent.agent.learning import format_action_result, format_human_feedback
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.consolidator import MemoryConsolidator
from ltm_agent.memory.cross_referencer import KnowledgeCrossReferencer
from ltm_agent.memory.interfaces import BaseContextualizer
from ltm_agent.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class LongTermMemoryAgent:
    """
    Agent with long-term memory capabilities for accumulating and leveraging knowledge.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        contextualizer: BaseContextualizer,
        llm: Any,
        knowledge_limit: int = 5,
        enable_consolidation: bool = True,
        consolidation_interval: int = 24,
    ):
        """
        Initialize the Long-Term Memory Agent.

        Args:
            memory_manager: The memory manager for knowledge retrieval and storage
            contextualizer: The contextualizer for adding context to knowledge
            llm: The LLM to use for generating responses
            knowledge_limit: Maximum number of knowledge units to retrieve per query
            enable_consolidation: Whether to enable periodic memory consolidation
            consolidation_interval: Hours between consolidation runs
        """
        self.memory_manager = memory_manager
        self.contextualizer = contextualizer
        self.llm = llm
        self.knowledge_limit = knowledge_limit
        self.last_query = None
        self.last_response = None

        # Initialize cross-referencer and consolidator
        self.cross_referencer = KnowledgeCrossReferencer(memory_manager)
        self.consolidator = MemoryConsolidator(
            memory_manager=memory_manager,
            cross_referencer=self.cross_referencer,
            consolidation_interval=consolidation_interval,
        )

        # Start background consolidation if enabled
        if enable_consolidation:
            self._start_consolidation()

        self.logger = logging.getLogger(__name__)
        self.logger.info("LongTermMemoryAgent initialized")

    def _start_consolidation(self):
        """Start the background memory consolidation process."""
        try:
            self.logger.info(
                f"Memory consolidation scheduled to run every {self.consolidator.consolidation_interval} hours"
            )
            # We'll defer the actual start of consolidation to the first invocation
            # This avoids issues with event loop handling during initialization
            self._consolidation_started = False
        except Exception as e:
            self.logger.warning(f"Failed to set up memory consolidation: {e}")

    async def _ensure_consolidation_started(self):
        """Ensure that consolidation has been started."""
        if not hasattr(self, "_consolidation_started") or not self._consolidation_started:
            try:
                # Start the background consolidation
                asyncio.create_task(self.consolidator.start_background_consolidation())
                self._consolidation_started = True
                self.logger.info("Background memory consolidation started")
            except Exception as e:
                self.logger.warning(f"Failed to start memory consolidation: {e}")

    async def async_invoke(self, query: str) -> str:
        """
        Process a query and return a response based on relevant knowledge (async version).

        Args:
            query: The user's query

        Returns:
            str: The generated response
        """
        self.logger.info(f"Processing query: {query}")

        # Ensure consolidation is started if enabled
        if hasattr(self, "consolidator") and not hasattr(self, "_consolidation_started"):
            await self._ensure_consolidation_started()

        # Get related knowledge from memory
        knowledge_units = await self.memory_manager.get_related_knowledge(
            content=query, limit=self.knowledge_limit
        )

        self.logger.info(f"Retrieved {len(knowledge_units)} relevant knowledge units")

        # Format the knowledge units into context
        context = self._format_context(knowledge_units)

        # Prepare the prompt for the LLM
        prompt = self._prepare_prompt(query, context)

        # Generate a response using the LLM
        response = self.llm.invoke(prompt)

        # Process the response for learning
        await self._async_process_learning(response)

        # Store query and response for feedback handling
        self.last_query = query
        self.last_response = response

        # Create cross-references for query and response
        try:
            # Process query as potential knowledge
            query_id = await self.memory_manager.process_potential_knowledge(
                content=query,
                source="action",
                context="User query",
                metadata={"type": "query", "timestamp": datetime.now().isoformat()},
            )

            # Process response as potential knowledge
            response_id = await self.memory_manager.process_potential_knowledge(
                content=response,
                source="action",
                context=f"Response to: {query}",
                metadata={
                    "type": "response",
                    "query_id": query_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Create cross-references between query and response
            await self.cross_referencer.enhance_multi_step_learning(
                query_id=query_id, response_id=response_id
            )
        except Exception as e:
            self.logger.warning(f"Error creating cross-references: {e}")

        return response

    def invoke(self, query: str) -> str:
        """
        Process a query with the agent (sync wrapper).

        Args:
            query: The user's query

        Returns:
            str: The agent's response
        """
        # For backwards compatibility, we'll use a sync wrapper
        # that can be called from non-async contexts
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context already, we need to
                # run our async code in a new thread to avoid loop conflicts
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.async_invoke(query))
                    return future.result()
            else:
                # No loop running, so we can just run our async code
                return loop.run_until_complete(self.async_invoke(query))
        except RuntimeError:
            # No event loop, so create one
            return asyncio.run(self.async_invoke(query))

    def _format_context(self, knowledge_units: list[tuple[KnowledgeUnit, float]]) -> str:
        """
        Format the knowledge units into a context string for the LLM.

        Args:
            knowledge_units: The list of knowledge unit tuples to format

        Returns:
            str: The formatted context
        """
        if not knowledge_units:
            return ""

        # Sort knowledge units by relevance score (highest first)
        sorted_units = sorted(knowledge_units, key=lambda x: x[1], reverse=True)

        context_parts = ["RELEVANT KNOWLEDGE:"]

        # Group knowledge units by topic to improve context coherence
        topic_groups = {}

        for unit, score in sorted_units:
            # Extract main topic (simple approach: first entity or subject)
            first_sentence = unit.original_chunk.split(".")[0].strip()
            main_topic = first_sentence.split()[0] if first_sentence else "Other"

            if main_topic not in topic_groups:
                topic_groups[main_topic] = []

            # Store the unit with its score in the appropriate topic group
            topic_groups[main_topic].append((unit, score))

        # Format each topic group
        index = 1
        for topic, units in topic_groups.items():
            for unit, score in units:
                # Format with relevance score and apply emphasis based on relevance
                if score > 0.7:  # High relevance
                    context_parts.append(
                        f"{index}. [HIGH RELEVANCE] {unit.original_chunk} (relevance: {score:.2f})"
                    )
                elif score > 0.4:  # Medium relevance
                    context_parts.append(f"{index}. {unit.original_chunk} (relevance: {score:.2f})")
                else:  # Low relevance
                    context_parts.append(f"{index}. {unit.original_chunk} (relevance: {score:.2f})")
                index += 1

        return "\n".join(context_parts)

    def _prepare_prompt(self, query: str, context: str) -> str:
        """
        Prepare the prompt for the LLM.

        Args:
            query: The user's query
            context: The formatted context from knowledge units

        Returns:
            str: The prepared prompt
        """
        if context:
            return f"""You are an AI assistant with access to the following knowledge:

{context}

Important instructions:
1. Focus primarily on knowledge marked as [HIGH RELEVANCE]
2. Only use information that is directly relevant to the query
3. If multiple pieces of information seem relevant, synthesize them into a coherent response
4. If no information seems relevant, acknowledge this and provide a general response
5. Always prioritize the most recent and specific information available

Based on this knowledge, please respond to the query:
{query}"""
        else:
            return f"""You are an AI assistant. Please respond to the following query:
{query}

Since I don't have specific information about this in my knowledge base, I'll provide a general response."""

    async def _async_process_learning(self, response: str) -> None:
        """
        Process the LLM response for learning (async version).

        Args:
            response: The LLM's response
        """
        self.logger.info("Processing response for learning")

        # Format the response properly
        formatted_response = format_action_result(response)

        # Add to memory via the memory manager
        await self.memory_manager.process_potential_knowledge(
            content=response, source="action", metadata={"type": "response"}
        )

        self.logger.info("Response processed for learning")

    def _process_learning(self, response: str) -> None:
        """
        Process the LLM response for learning (sync wrapper).

        Args:
            response: The LLM's response
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context already, we need to
                # run our async code in a new thread to avoid loop conflicts
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_process_learning(response))
                    return future.result()
            else:
                # No loop running, so we can just run our async code
                return loop.run_until_complete(self._async_process_learning(response))
        except RuntimeError:
            # No event loop, so create one
            return asyncio.run(self._async_process_learning(response))

    async def async_process_feedback(self, feedback: str) -> None:
        """
        Process human feedback asynchronously with enhanced connectivity between related knowledge.

        Args:
            feedback: The feedback text
        """
        try:
            self.logger.info(f"Processing feedback: {feedback}")

            # Ensure consolidation is started if enabled
            await self._ensure_consolidation_started()

            # Format the feedback for storage
            formatted_feedback = format_human_feedback(feedback)

            # Store the feedback as new knowledge
            feedback_id = await self.memory_manager.add_knowledge(
                content=formatted_feedback, source="human_feedback"
            )

            # Store explicit metadata linking this feedback to any relevant knowledge
            if feedback_id:
                # Find knowledge related to this feedback
                related_knowledge = await self.memory_manager.get_related_knowledge(
                    content=formatted_feedback, limit=5
                )

                if related_knowledge:
                    # Mark this feedback unit with connections to related knowledge
                    feedback_metadata = {
                        "related_units": [unit.unique_id for unit, _ in related_knowledge],
                        "feedback_type": "enhancement",
                        "processed_at": str(datetime.now()),
                    }

                    # Analyze if this is a correction by comparing with recent responses
                    if hasattr(self, "last_response") and self.last_response:
                        # Check for correction patterns in the feedback
                        correction_indicators = [
                            "incorrect",
                            "wrong",
                            "not right",
                            "error",
                            "mistake",
                            "actually",
                            "in fact",
                            "should be",
                            "correct",
                            "instead",
                            "rather",
                        ]

                        is_correction = any(
                            indicator in feedback.lower() for indicator in correction_indicators
                        )

                        # If it seems like a correction, update the metadata
                        if is_correction:
                            feedback_metadata["feedback_type"] = "correction"
                            feedback_metadata["corrects_response"] = self.last_response

                            # Set higher importance for corrections
                            feedback_metadata["importance"] = "high"

                    # Update the feedback unit with metadata
                    await self.memory_manager.update_knowledge(
                        unique_id=feedback_id, metadata=feedback_metadata
                    )

                    # Create bidirectional references between the feedback and related knowledge
                    for knowledge_unit, similarity in related_knowledge:
                        # Skip if the similarity is too low
                        if similarity < 0.3:
                            continue

                        # Check if this knowledge already has feedback metadata
                        knowledge_metadata = knowledge_unit.metadata or {}

                        # Initialize feedback-related metadata if needed
                        if "feedback_references" not in knowledge_metadata:
                            knowledge_metadata["feedback_references"] = []

                        # Add this feedback reference
                        feedback_reference = {
                            "feedback_id": feedback_id,
                            "similarity": similarity,
                            "feedback_type": feedback_metadata.get("feedback_type", "enhancement"),
                            "processed_at": str(datetime.now()),
                        }

                        # Only add if not already present
                        if not any(
                            ref.get("feedback_id") == feedback_id
                            for ref in knowledge_metadata.get("feedback_references", [])
                        ):
                            knowledge_metadata["feedback_references"].append(feedback_reference)

                        # For corrections, adjust the "corrected" flag
                        if feedback_metadata.get("feedback_type") == "correction":
                            knowledge_metadata["corrected"] = True
                            knowledge_metadata["correction_id"] = feedback_id

                        # Also add to related_units if needed
                        if "related_units" not in knowledge_metadata:
                            knowledge_metadata["related_units"] = []

                        if feedback_id not in knowledge_metadata["related_units"]:
                            knowledge_metadata["related_units"].append(feedback_id)

                        # Update the knowledge unit
                        await self.memory_manager.update_knowledge(
                            unique_id=knowledge_unit.unique_id, metadata=knowledge_metadata
                        )

                        # Also check if there are other units related to this knowledge
                        # that might benefit from this feedback
                        if knowledge_metadata.get("related_units"):
                            for related_id in knowledge_metadata.get("related_units", []):
                                # Skip the feedback unit itself
                                if related_id == feedback_id:
                                    continue

                                # Get the related unit
                                related_unit = await self.memory_manager.retrieve_knowledge(
                                    related_id
                                )
                                if related_unit:
                                    # Check if this unit should be connected to the feedback
                                    if self._is_content_similar(
                                        related_unit.original_chunk, formatted_feedback, 0.3
                                    ):
                                        # Update metadata for the related unit
                                        related_metadata = related_unit.metadata or {}

                                        if "related_units" not in related_metadata:
                                            related_metadata["related_units"] = []

                                        if feedback_id not in related_metadata["related_units"]:
                                            related_metadata["related_units"].append(feedback_id)

                                        # Update the feedback unit with these references
                                        await self.memory_manager.update_knowledge(
                                            unique_id=feedback_id, metadata=feedback_metadata
                                        )

            # Special handling for feedback that corrects previous responses
            if (
                hasattr(self, "last_query")
                and hasattr(self, "last_response")
                and self.last_query
                and self.last_response
            ):
                # Determine if this is likely a correction to the last response
                is_correction = False
                correction_indicators = [
                    "incorrect",
                    "wrong",
                    "not right",
                    "error",
                    "mistake",
                    "actually",
                    "in fact",
                    "should be",
                    "correct",
                    "instead",
                    "rather",
                ]

                if any(indicator in feedback.lower() for indicator in correction_indicators):
                    is_correction = True

                # If this seems like a correction or the feedback is very similar to the query context
                if is_correction or self._is_content_similar(self.last_query, feedback, 0.4):
                    # Create a more detailed composite knowledge unit that explicitly connects query, response, and feedback
                    composite_content = (
                        f"Query: '{self.last_query}' → Correct response: '{feedback}'"
                    )

                    composite_context = f"""
                    Original query: {self.last_query}
                    Incorrect response (to be avoided): {self.last_response}
                    Corrected information: {feedback}

                    This is a correction to a previous response. When asked about "{self.last_query}"
                    or similar questions, use this corrected information instead of the incorrect response.
                    """

                    # Store this composite knowledge with high importance
                    correction_id = await self.memory_manager.process_potential_knowledge(
                        content=composite_content,
                        source="feedback",
                        context=composite_context.strip(),
                        metadata={
                            "type": "correction",
                            "subtype": "composite_correction",
                            "processed_at": str(datetime.now()),
                            "feedback_id": feedback_id,
                            "importance": "very_high",  # Increased importance
                            "related_query": self.last_query,
                            "incorrect_response": self.last_response,
                        },
                        update_if_similar=False,  # Always add as new to preserve the correction pattern
                    )

                    # If the correction was successfully stored, also create a negatively weighted example
                    # to help the system learn what NOT to do
                    if correction_id:
                        negative_example = f"""
                        INCORRECT EXAMPLE - DO NOT USE:
                        Query: {self.last_query}
                        Incorrect response: {self.last_response}

                        This is marked as an incorrect response pattern to avoid repeating.
                        """

                        await self.memory_manager.process_potential_knowledge(
                            content=negative_example.strip(),
                            source="system",
                            context="This is a negative example created from user correction feedback.",
                            metadata={
                                "type": "negative_example",
                                "processed_at": str(datetime.now()),
                                "related_correction_id": correction_id,
                                "importance": "high",
                                "negative_weight": 0.9,  # High negative weight
                            },
                            update_if_similar=False,
                        )

            self.logger.info("Human feedback processed for learning with enhanced connectivity")
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            raise

    def process_feedback(self, feedback: str) -> None:
        """
        Process human feedback for learning (sync wrapper).

        Args:
            feedback: The human feedback
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context already, we need to
                # run our async code in a new thread to avoid loop conflicts
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.async_process_feedback(feedback))
                    return future.result()
            else:
                # No loop running, so we can just run our async code
                return loop.run_until_complete(self.async_process_feedback(feedback))
        except RuntimeError:
            # No event loop, so create one
            return asyncio.run(self.async_process_feedback(feedback))

    def _is_content_similar(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """
        Check if two pieces of text are similar based on common words.

        Args:
            text1: First text string
            text2: Second text string
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            bool: True if texts are similar, False otherwise
        """
        # Simple word-based similarity check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold
