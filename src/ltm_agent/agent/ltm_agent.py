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
        # Initialize logger first to avoid reference errors
        self.logger = logging.getLogger(__name__)

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
            try:
                return asyncio.run(self.async_invoke(query))
            except ValueError as e:
                # Handle the array truth value error and other ValueError cases directly
                self.logger.error(f"Error during processing: {e}", exc_info=True)
                return f"I encountered an error while processing your query: {str(e)}"
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}", exc_info=True)
                return "I encountered an unexpected error. Please try a different query."
        except ValueError as e:
            # Handle ValueError without trying to get event loop again
            self.logger.error(f"ValueError during processing: {e}", exc_info=True)
            return f"I encountered an error while accessing my memory: {str(e)}"
        except Exception as e:
            # General error handling to prevent cascading failures
            self.logger.error(f"Error in invoke: {e}", exc_info=True)
            return "Sorry, I encountered an error while processing your request."

    def _format_context(self, knowledge_units: list[tuple[KnowledgeUnit, float]]) -> str:
        """
        Format the knowledge units into a context string for the LLM.

        Args:
            knowledge_units: The list of knowledge unit tuples to format

        Returns:
            str: The formatted context
        """
        if not knowledge_units:
            return "NO RELEVANT KNOWLEDGE FOUND."

        # Sort knowledge units by relevance score (highest first)
        sorted_units = sorted(knowledge_units, key=lambda x: x[1], reverse=True)

        context_parts = ["RELEVANT KNOWLEDGE FROM MY MEMORY:"]

        # Add each knowledge unit with its similarity score
        for i, (unit, score) in enumerate(sorted_units, start=1):
            unit_text = unit.original_chunk.strip()
            # Skip empty knowledge units
            if not unit_text:
                continue

            # Format with relevance score and index
            importance = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.3 else "LOW"
            source = unit.knowledge_source.upper() if unit.knowledge_source else "UNKNOWN"

            context_parts.append(
                f"[KNOWLEDGE {i}] (RELEVANCE: {importance}, SCORE: {score:.2f}, SOURCE: {source})\n{unit_text}"
            )

        # Format the context parts with clear separation
        return "\n\n".join(context_parts)

    def _prepare_prompt(self, query: str, context: str) -> str:
        """
        Prepare the prompt for the LLM.

        Args:
            query: The user's query
            context: The context from memory

        Returns:
            str: The formatted prompt
        """
        base_prompt = f"""
You are a helpful assistant with accurate long-term memory. Your memory contains knowledge that users have shared with you.

{context}

User query: {query}

Instructions:
1. Answer the user's query using ONLY the knowledge provided above
2. If the knowledge directly answers the query, use it to give a complete and accurate response
3. If the knowledge contains information about LangChain, Python, Windsurf, or the user's name, use this information in your answer
4. If the relevant knowledge is missing or insufficient, state this clearly
5. Do not invent or make up information that isn't in the knowledge

Remember: When users provide feedback (statements beginning with "feedback:"), this information becomes part of your memory for future reference.

Your response:
"""
        return base_prompt

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
            formatted_feedback_content = format_human_feedback(feedback)

            # Prepare metadata for the feedback unit
            feedback_metadata = {
                "type": "user_feedback",
                "original_input": feedback,
                "timestamp": datetime.now().isoformat(),
                "related_query": self.last_query if hasattr(self, "last_query") else None,
            }

            # Store the feedback as new knowledge with metadata
            feedback_id = await self.memory_manager.add_knowledge(
                content=formatted_feedback_content, source="feedback", metadata=feedback_metadata
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
