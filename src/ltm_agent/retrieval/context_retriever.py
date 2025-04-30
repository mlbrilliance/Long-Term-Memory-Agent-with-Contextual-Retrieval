"""
Context retrieval logic for selecting relevant knowledge for LLM queries.

This module provides strategies for retrieving and organizing context
from the agent's memory to enhance LLM responses with relevant knowledge.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ltm_agent.core.exceptions import MemoryError
from ltm_agent.memory.interfaces import HybridRetriever
from ltm_agent.memory.manager import MemoryManager
from ltm_agent.retrieval.context_analyzer import ContextAnalyzer
from ltm_agent.retrieval.enhanced_prompts import EnhancedPromptBuilder

# Configure logging
logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Context retriever for selecting relevant knowledge for LLM queries.

    This class coordinates the retrieval of contextually relevant knowledge
    from the agent's memory based on the user's query.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        max_context_items: int = 10,
        relevance_threshold: float = 0.6,
        max_token_limit: int = 4000,
        hybrid_retriever: HybridRetriever | None = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_enhanced_prompts: bool = True,
        enable_context_analysis: bool = True,
        context_organization_strategy: str = "relevance_first",
    ):
        """
        Initialize the context retriever.

        Args:
            memory_manager: The memory manager to use for retrieving knowledge
            max_context_items: Maximum number of knowledge units to include in context
            relevance_threshold: Minimum relevance score for including knowledge
            max_token_limit: Maximum number of tokens in the retrieved context
            hybrid_retriever: Optional HybridRetriever instance to use
            vector_weight: Weight for vector similarity scores in hybrid retrieval
            bm25_weight: Weight for BM25 scores in hybrid retrieval
            use_enhanced_prompts: Whether to use the EnhancedPromptBuilder
            enable_context_analysis: Whether to use the ContextAnalyzer
            context_organization_strategy: Strategy for organizing context items
        """
        self.memory_manager = memory_manager
        self.max_context_items = max_context_items
        self.relevance_threshold = relevance_threshold
        self.max_token_limit = max_token_limit
        self.hybrid_retriever = hybrid_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_enhanced_prompts = use_enhanced_prompts
        self.enable_context_analysis = enable_context_analysis

        # Initialize prompt builders
        if use_enhanced_prompts:
            self.prompt_builder = EnhancedPromptBuilder(
                max_prompt_tokens=max_token_limit,
                context_organization_strategy=context_organization_strategy,
            )
        else:
            self.prompt_builder = PromptBuilder(max_prompt_tokens=max_token_limit)

        # Initialize context analyzer if enabled
        if enable_context_analysis:
            self.context_analyzer = ContextAnalyzer()
        else:
            self.context_analyzer = None

        logger.info(
            f"Initialized ContextRetriever with max_items={max_context_items}, "
            f"threshold={relevance_threshold}, enhanced_prompts={use_enhanced_prompts}"
        )

        if hybrid_retriever:
            logger.info(f"Using provided hybrid retriever: {hybrid_retriever.__class__.__name__}")

    async def retrieve_context(
        self,
        query: str,
        recent_conversation: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
        strategy: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The user query to retrieve context for
            recent_conversation: Optional list of recent conversation turns
            filters: Optional filters to apply to knowledge retrieval
            strategy: Retrieval strategy (semantic, keyword, hybrid, temporal, or advanced_hybrid)

        Returns:
            List of context items with source information

        Raises:
            MemoryError: If retrieval fails
        """
        logger.info(f"Retrieving context for query: {query[:50]}... using {strategy} strategy")

        try:
            # Determine source filter from filters
            source_filter = None
            if filters and "knowledge_source" in filters:
                source_filter = filters["knowledge_source"]

            if strategy == "semantic":
                context = await self._retrieve_semantic(query, source_filter)
            elif strategy == "keyword":
                context = await self._retrieve_keyword(query, source_filter)
            elif strategy == "temporal":
                context = await self._retrieve_temporal(source_filter)
            elif strategy == "hybrid":
                context = await self._retrieve_hybrid(query, source_filter)
            elif strategy == "advanced_hybrid":
                context = await self._retrieve_with_hybrid_retriever(query, source_filter)
            else:
                logger.warning(f"Unknown retrieval strategy: {strategy}, using hybrid as default")
                context = await self._retrieve_hybrid(query, source_filter)

            # Add conversation context if provided
            if recent_conversation:
                conversation_context = self._extract_conversation_context(recent_conversation)
                context = self._combine_and_deduplicate(context, conversation_context)

            # Apply context analysis if enabled
            if self.enable_context_analysis and self.context_analyzer:
                context = self.context_analyzer.analyze_context_items(context)

                # Sort by adjusted relevance if available, otherwise by original relevance
                context.sort(
                    key=lambda x: x.get("adjusted_relevance", x.get("relevance", 0)), reverse=True
                )

            # Limit context to stay within token budget
            filtered_context = self._limit_context_tokens(context)

            logger.info(f"Retrieved {len(filtered_context)} context items")
            return filtered_context

        except Exception as e:
            logger.exception(f"Error retrieving context: {str(e)}")
            raise MemoryError(f"Failed to retrieve context: {str(e)}")

    def set_hybrid_retriever(self, hybrid_retriever: HybridRetriever):
        """
        Set or update the hybrid retriever.

        Args:
            hybrid_retriever: The hybrid retriever to use
        """
        self.hybrid_retriever = hybrid_retriever
        logger.info(f"Updated hybrid retriever to: {hybrid_retriever.__class__.__name__}")

    def update_retrieval_weights(self, vector_weight: float, bm25_weight: float):
        """
        Update the weights used for hybrid retrieval.

        Args:
            vector_weight: Weight for vector similarity scores
            bm25_weight: Weight for BM25 scores
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        logger.info(f"Updated retrieval weights: vector={vector_weight}, bm25={bm25_weight}")

    async def _retrieve_with_hybrid_retriever(
        self, query: str, source_filter: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve context using the advanced hybrid retriever.

        Args:
            query: The query to retrieve context for
            source_filter: Optional filter for knowledge source
            limit: Maximum number of results to return

        Returns:
            List of context items
        """
        if not self.hybrid_retriever:
            logger.warning("No hybrid retriever set, falling back to basic hybrid retrieval")
            return await self._retrieve_hybrid(query, source_filter, limit)

        if limit is None:
            limit = self.max_context_items

        try:
            # Get bm25 store from memory manager
            bm25_store = self.memory_manager.get_bm25_store()
            if not bm25_store:
                logger.warning("No BM25 store available, falling back to semantic retrieval")
                return await self._retrieve_semantic(query, source_filter, limit)

            # Apply filters
            filter_dict = {}
            if source_filter:
                filter_dict["knowledge_source"] = source_filter

            # Perform hybrid search
            results = await self.hybrid_retriever.search(
                query,
                k=limit,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                filters=filter_dict if filter_dict else None,
            )

            # Convert to context items
            context_items = []
            for unit, score in results:
                item = {
                    "content": unit.content,
                    "context": unit.context,
                    "source": unit.knowledge_source,
                    "relevance": score,
                    "id": unit.unique_id,
                    "timestamp": unit.timestamp.isoformat() if unit.timestamp else None,
                    "metadata": unit.metadata,
                }
                context_items.append(item)

            return context_items

        except Exception as e:
            logger.exception(f"Error in hybrid retriever: {str(e)}")
            # Fall back to semantic if hybrid fails
            return await self._retrieve_semantic(query, source_filter, limit)

    async def _retrieve_semantic(
        self, query: str, source_filter: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve context using semantic similarity.

        Args:
            query: The query to retrieve context for
            source_filter: Optional filter for knowledge source
            limit: Maximum number of results to return

        Returns:
            List of context items
        """
        if limit is None:
            limit = self.max_context_items

        try:
            # Prepare filter dict if source filter provided
            filter_dict = {}
            if source_filter:
                filter_dict["knowledge_source"] = source_filter

            # Retrieve semantically similar knowledge
            similar_units = await self.memory_manager.search_vector_store(
                query, k=limit, filters=filter_dict if filter_dict else None, with_scores=True
            )

            # Convert to context items
            context_items = []
            for unit, score in similar_units:
                item = {
                    "content": unit.content,
                    "context": unit.context,
                    "source": unit.knowledge_source,
                    "relevance": score,
                    "id": unit.unique_id,
                    "timestamp": unit.timestamp.isoformat() if unit.timestamp else None,
                    "metadata": unit.metadata,
                }
                context_items.append(item)

            return context_items

        except Exception as e:
            logger.exception(f"Error in semantic retrieval: {str(e)}")
            return []

    async def _retrieve_keyword(
        self, query: str, source_filter: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve context using keyword matching.

        Args:
            query: The query to retrieve context for
            source_filter: Optional filter for knowledge source
            limit: Maximum number of results to return

        Returns:
            List of context items
        """
        if limit is None:
            limit = self.max_context_items

        try:
            # Prepare filter dict if source filter provided
            filter_dict = {}
            if source_filter:
                filter_dict["knowledge_source"] = source_filter

            # Retrieve keyword matches
            bm25_store = self.memory_manager.get_bm25_store()
            if not bm25_store:
                logger.warning("No BM25 store available for keyword retrieval")
                return []

            matching_units = await bm25_store.search(
                query, k=limit, filters=filter_dict if filter_dict else None
            )

            # Convert to context items
            context_items = []
            for unit, score in matching_units:
                item = {
                    "content": unit.content,
                    "context": unit.context,
                    "source": unit.knowledge_source,
                    "relevance": score,
                    "id": unit.unique_id,
                    "timestamp": unit.timestamp.isoformat() if unit.timestamp else None,
                    "metadata": unit.metadata,
                }
                context_items.append(item)

            return context_items

        except Exception as e:
            logger.exception(f"Error in keyword retrieval: {str(e)}")
            return []

    async def _retrieve_temporal(
        self, source_filter: str | None = None, days: int = 7, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent context items.

        Args:
            source_filter: Optional filter for knowledge source
            days: Number of days to look back
            limit: Maximum number of results to return

        Returns:
            List of context items
        """
        if limit is None:
            limit = self.max_context_items

        try:
            # Calculate date range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)

            # Prepare filter dict
            filter_dict = {"timestamp": {"$gte": start_time, "$lte": end_time}}
            if source_filter:
                filter_dict["knowledge_source"] = source_filter

            # Retrieve recent knowledge
            recent_units = await self.memory_manager.get_recent_knowledge(
                limit=limit, filters=filter_dict if filter_dict else None
            )

            # Convert to context items
            context_items = []
            for unit in recent_units:
                # Assign a relevance score based on recency
                # More recent items get higher scores
                age_seconds = (end_time - unit.timestamp).total_seconds()
                max_age_seconds = days * 24 * 3600
                recency_score = 1.0 - min(1.0, age_seconds / max_age_seconds)

                item = {
                    "content": unit.content,
                    "context": unit.context,
                    "source": unit.knowledge_source,
                    "relevance": recency_score,
                    "id": unit.unique_id,
                    "timestamp": unit.timestamp.isoformat() if unit.timestamp else None,
                    "metadata": unit.metadata,
                }
                context_items.append(item)

            # Sort by timestamp (newest first)
            context_items.sort(key=lambda x: x["timestamp"] or "", reverse=True)

            return context_items

        except Exception as e:
            logger.exception(f"Error in temporal retrieval: {str(e)}")
            return []

    async def _retrieve_hybrid(
        self, query: str, source_filter: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve context using both semantic and keyword retrieval.

        Args:
            query: The query to retrieve context for
            source_filter: Optional filter for knowledge source
            limit: Maximum number of results to return

        Returns:
            List of context items combining semantic and keyword results
        """
        if limit is None:
            limit = self.max_context_items

        # Allocate half of the limit to each method
        semantic_limit = limit // 2
        keyword_limit = limit // 2

        try:
            # Retrieve using both methods in parallel
            semantic_task = asyncio.create_task(
                self._retrieve_semantic(query, source_filter, semantic_limit)
            )
            keyword_task = asyncio.create_task(
                self._retrieve_keyword(query, source_filter, keyword_limit)
            )

            # Wait for both to complete
            semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)

            # Combine results
            combined = self._combine_and_deduplicate(semantic_results, keyword_results)

            return combined[:limit]

        except Exception as e:
            logger.exception(f"Error in hybrid retrieval: {str(e)}")
            # Try semantic retrieval as fallback
            return await self._retrieve_semantic(query, source_filter, limit)

    def _combine_and_deduplicate(
        self, contexts1: list[dict[str, Any]], contexts2: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Combine and deduplicate context items.

        Args:
            contexts1: First list of context items
            contexts2: Second list of context items

        Returns:
            Combined and deduplicated list of context items
        """
        # Combine lists
        all_contexts = contexts1 + contexts2

        # Deduplicate by id
        seen_ids = set()
        unique_contexts = []

        for item in all_contexts:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_contexts.append(item)
            elif not item_id:
                # Items without id are always included
                unique_contexts.append(item)

        # Sort by relevance
        unique_contexts.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return unique_contexts

    def _extract_conversation_context(
        self, conversation: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Extract context from recent conversation turns.

        Args:
            conversation: List of recent conversation turns

        Returns:
            List of context items extracted from conversation
        """
        context_items = []

        # Skip if empty
        if not conversation:
            return context_items

        # Process each turn
        for i, turn in enumerate(conversation):
            role = turn.get("role", "")
            content = turn.get("content", "")

            if not content:
                continue

            # Determine source based on role
            source = f"conversation_{role}"

            # Extract timestamp if available, or use current time with decreasing precision
            # for proper ordering of conversation turns
            now = datetime.now(timezone.utc)
            time_offset = timedelta(seconds=len(conversation) - i)
            timestamp = (now - time_offset).isoformat()

            # Create context item
            item = {
                "content": content,
                "context": f"Previous conversation turn ({role})",
                "source": source,
                "relevance": 0.95 - (0.05 * i),  # Decrease relevance for older turns
                "id": f"conv-{i}",
                "timestamp": timestamp,
                "metadata": {"turn_index": i, "role": role},
            }

            context_items.append(item)

        return context_items

    def _limit_context_tokens(self, context_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Limit context to stay within token budget.

        Args:
            context_items: List of context items

        Returns:
            Filtered list of context items within token budget
        """

        # Helper function to estimate tokens
        def estimate_tokens(text: str) -> int:
            # Simple approximation: ~4 chars per token
            return len(text) // 4

        # Filter by relevance threshold
        filtered_items = [
            item for item in context_items if item.get("relevance", 0) >= self.relevance_threshold
        ]

        # Sort by relevance
        filtered_items.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        # Limit by max items
        filtered_items = filtered_items[: self.max_context_items]

        # Calculate token usage
        total_tokens = 0
        final_items = []

        for item in filtered_items:
            item_text = f"{item['content']} {item.get('context', '')}"
            item_tokens = estimate_tokens(item_text)

            if total_tokens + item_tokens <= self.max_token_limit:
                final_items.append(item)
                total_tokens += item_tokens
            else:
                break

        logger.info(f"Limited context to {len(final_items)} items with ~{total_tokens} tokens")
        return final_items

    def build_prompt(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        task_type: str = "question_answering",
        additional_instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a prompt for LLM using the retrieved context.

        Args:
            query: The user's query
            context_items: Retrieved context items
            task_type: Type of task (question_answering, summarization, etc.)
            additional_instructions: Optional additional instructions for the LLM

        Returns:
            Formatted prompt in a messages structure
        """
        return self.prompt_builder.build_prompt(
            query,
            context_items,
            task_type=task_type,
            additional_instructions=additional_instructions,
        )

    def build_prompt_with_history(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        conversation_history: list[dict[str, str]],
        task_type: str = "conversational",
        additional_instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a prompt with conversation history and context.

        Args:
            query: The user's query
            context_items: Retrieved context items
            conversation_history: Previous conversation turns
            task_type: Type of task (usually conversational)
            additional_instructions: Optional additional instructions for the LLM

        Returns:
            Formatted prompt in a messages structure with history
        """
        # Use enhanced prompt builder if available
        if self.use_enhanced_prompts:
            return self.prompt_builder.build_prompt_with_history(
                query,
                context_items,
                conversation_history,
                task_type=task_type,
                additional_instructions=additional_instructions,
            )

        # Otherwise build a basic prompt with history
        prompt_result = self.prompt_builder.build_prompt(query, context_items)

        # Insert conversation history before the user's final query
        messages = prompt_result["messages"]

        # Extract system message
        system_message = messages[0]

        # Build new messages list with history
        new_messages = [system_message]

        # Add conversation history
        for turn in conversation_history:
            new_messages.append({"role": turn["role"], "content": turn["content"]})

        # Add final user query
        new_messages.append({"role": "user", "content": query})

        # Update the result
        prompt_result["messages"] = new_messages
        prompt_result["history_turns"] = len(conversation_history)

        return prompt_result


class PromptBuilder:
    """
    Builds prompts for LLM interaction incorporating retrieved context.

    This class transforms retrieved context into formatted prompts suitable
    for LLM input, handling context organization and instruction formatting.
    """

    def __init__(
        self,
        system_prompt_template: str | None = None,
        context_format_template: str | None = None,
        max_prompt_tokens: int = 8000,
    ):
        """
        Initialize the prompt builder.

        Args:
            system_prompt_template: Optional custom system prompt template
            context_format_template: Optional custom context format template
            max_prompt_tokens: Maximum number of tokens in the complete prompt
        """
        self.max_prompt_tokens = max_prompt_tokens

        # Default system prompt template if none provided
        self.system_prompt_template = system_prompt_template or (
            "You are a helpful assistant with access to the following relevant information. "
            "Use this information to answer the user's questions accurately. "
            "If the information provided doesn't contain the answer, just say you don't know "
            "rather than making up information. Answer in a friendly, helpful, and concise manner."
        )

        # Default context format template if none provided
        self.context_format_template = context_format_template or (
            "Context Item {index}:\nSource: {source}\nContent: {content}\n{context_section}"
        )

        logger.info("Initialized PromptBuilder")

    def build_prompt(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        include_metadata: bool = False,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a complete prompt with context for LLM.

        Args:
            query: The user's query
            context_items: Retrieved context items
            include_metadata: Whether to include metadata in context
            system_prompt_override: Optional override for the system prompt

        Returns:
            Complete prompt for LLM in message format
        """
        # Sort context by relevance
        sorted_context = sorted(context_items, key=lambda x: x.get("relevance", 0), reverse=True)

        # Format each context item
        formatted_contexts = []
        for i, item in enumerate(sorted_context):
            context_text = ""
            if item.get("context"):
                context_text = f"Additional Context: {item['context']}\n"

            # Add metadata if requested
            metadata_text = ""
            if include_metadata and item.get("metadata"):
                metadata_str = ", ".join(f"{k}: {v}" for k, v in item["metadata"].items())
                metadata_text = f"Metadata: {metadata_str}\n"

            # Format the context item
            formatted_context = self.context_format_template.format(
                index=i + 1,
                source=item["source"],
                content=item["content"],
                context_section=context_text + metadata_text,
            )

            formatted_contexts.append(formatted_context)

        # Combine contexts
        combined_context = "\n\n".join(formatted_contexts)

        # Create system prompt with context
        system_prompt = system_prompt_override or self.system_prompt_template
        if combined_context:
            system_prompt = f"{system_prompt}\n\nRelevant Information:\n{combined_context}"

        # Build message format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        return {"messages": messages, "context_items_used": len(context_items)}
