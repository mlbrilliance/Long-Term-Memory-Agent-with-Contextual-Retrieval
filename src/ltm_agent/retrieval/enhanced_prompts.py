"""
Enhanced prompt building for LLM interactions.

This module provides advanced prompt building capabilities that transform
retrieved context into formatted prompts suitable for LLM input, with
special handling for different knowledge types and metadata enrichment.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedPromptBuilder:
    """
    Advanced prompt builder for LLM interaction with enhanced context handling.

    This class provides sophisticated prompt building capabilities, including:
    - Custom formatting for different knowledge types
    - Metadata enrichment and tagging
    - Context organization by relevance, recency, and source type
    - Support for different prompt templates per task type
    """

    # Template types for different use cases
    TEMPLATE_TYPES = {
        "qa": "question_answering",
        "summarization": "summarization",
        "reasoning": "reasoning",
        "chat": "conversational",
        "task": "task_execution",
    }

    def __init__(
        self,
        system_prompt_template: str | None = None,
        context_format_templates: dict[str, str] | None = None,
        max_prompt_tokens: int = 8000,
        enable_metadata_enrichment: bool = True,
        context_organization_strategy: str = "relevance_first",
    ):
        """
        Initialize the enhanced prompt builder.

        Args:
            system_prompt_template: Optional custom system prompt template
            context_format_templates: Optional dict of format templates for different sources
            max_prompt_tokens: Maximum number of tokens in the complete prompt
            enable_metadata_enrichment: Whether to enrich context with metadata
            context_organization_strategy: Strategy for organizing context
                                          ("relevance_first", "recency_first", or "hybrid")
        """
        self.max_prompt_tokens = max_prompt_tokens
        self.enable_metadata_enrichment = enable_metadata_enrichment
        self.context_organization_strategy = context_organization_strategy

        # Default system prompt template if none provided
        self.system_prompt_template = system_prompt_template or (
            "You are a helpful assistant with access to the following relevant information. "
            "Use this information to answer the user's questions accurately. "
            "If the information provided doesn't contain the answer, just say you don't know "
            "rather than making up information. Answer in a friendly, helpful, and concise manner."
        )

        # Default context format templates for different sources
        self.context_format_templates = context_format_templates or {
            "default": (
                "Context Item {index}:\nSource: {source}\nContent: {content}\n{context_section}"
            ),
            "corpus": (
                "Knowledge Item {index} [{relevance_score}]:\n"
                "Topic: {context}\n"
                "Content: {content}\n"
                "{metadata_section}"
            ),
            "action": (
                "Previous Action {index} [{timestamp}]:\n"
                "Action: {context}\n"
                "Result: {content}\n"
                "{metadata_section}"
            ),
            "feedback": (
                "User Feedback {index}:\nTopic: {context}\nFeedback: {content}\n{metadata_section}"
            ),
            "conversation": (
                "Conversation History {index}:\n"
                "Role: {role}\n"
                "Message: {content}\n"
                "{metadata_section}"
            ),
        }

        # Template presets for different task types
        self.task_templates = {
            "question_answering": (
                "You are a helpful assistant answering a question. Use the following "
                "information to provide an accurate, thorough answer. If the information "
                "doesn't contain everything needed, acknowledge the limitations of your answer. "
                "If the information contains different perspectives, acknowledge them. "
                "Respond in a concise but complete manner."
            ),
            "summarization": (
                "You are a helpful assistant tasked with summarizing information. "
                "Use the following information to create a comprehensive summary. "
                "Focus on the main points, exclude minor details, and organize the "
                "information in a logical flow. Be objective and thorough."
            ),
            "reasoning": (
                "You are a helpful assistant tasked with solving a problem that requires "
                "reasoning. Use the following information to analyze the situation and "
                "provide step-by-step reasoning toward the answer. Consider multiple "
                "perspectives if applicable, and clearly articulate your chain of thought."
            ),
            "conversational": (
                "You are a helpful conversational assistant. Use the provided conversation "
                "history and relevant information to respond naturally to the user. Maintain "
                "a friendly, helpful tone. Remember details from earlier in the conversation. "
                "If you don't know something, simply say so rather than guessing."
            ),
            "task_execution": (
                "You are a helpful assistant tasked with executing a specific task. "
                "Use the following information to understand the requirements and "
                "context of the task. Provide a clear, actionable response that "
                "directly addresses the task at hand. Include any necessary steps, "
                "resources, or considerations."
            ),
        }

        logger.info(
            f"Initialized EnhancedPromptBuilder with strategy: {context_organization_strategy}"
        )

    def set_template_for_task(self, task_type: str, template: str) -> None:
        """
        Set a custom template for a specific task type.

        Args:
            task_type: The type of task (one of TEMPLATE_TYPES keys or values)
            template: The template to use for this task type
        """
        if task_type in self.TEMPLATE_TYPES:
            task_type = self.TEMPLATE_TYPES[task_type]

        self.task_templates[task_type] = template
        logger.info(f"Set custom template for task type: {task_type}")

    def build_prompt(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        task_type: str = "question_answering",
        include_metadata: bool = True,
        system_prompt_override: str | None = None,
        additional_instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a complete prompt with enhanced context for LLM.

        Args:
            query: The user's query or instruction
            context_items: Retrieved context items
            task_type: Type of task (qa, summarization, reasoning, etc.)
            include_metadata: Whether to include metadata in context
            system_prompt_override: Optional override for the system prompt
            additional_instructions: Optional additional instructions to add

        Returns:
            Complete prompt for LLM in message format
        """
        # Enrich context items with metadata if enabled
        if self.enable_metadata_enrichment:
            enriched_items = [self._enrich_context_item(item) for item in context_items]
        else:
            enriched_items = context_items

        # Organize context based on strategy
        organized_context = self._organize_context(enriched_items)

        # Format each context item based on its source type
        formatted_contexts = []
        for i, item in enumerate(organized_context):
            # Get the appropriate format template based on source
            source_type = self._get_source_type(item)
            format_template = self.context_format_templates.get(
                source_type, self.context_format_templates["default"]
            )

            # Format this specific context item
            formatted_item = self._format_context_item(
                item, format_template, i + 1, include_metadata
            )
            formatted_contexts.append(formatted_item)

        # Combine formatted contexts
        combined_context = "\n\n".join(formatted_contexts)

        # Select system prompt template based on task type
        template_key = self.TEMPLATE_TYPES.get(task_type, task_type)
        task_template = self.task_templates.get(
            template_key, self.task_templates["question_answering"]
        )

        # Create final system prompt
        system_prompt = system_prompt_override or task_template
        if combined_context:
            system_prompt = f"{system_prompt}\n\nRelevant Information:\n{combined_context}"

        # Add additional instructions if provided
        if additional_instructions:
            system_prompt = (
                f"{system_prompt}\n\nAdditional Instructions:\n{additional_instructions}"
            )

        # Build message format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Log information about the constructed prompt
        logger.info(
            f"Built prompt for task type '{task_type}' with {len(organized_context)} context items"
        )

        return {
            "messages": messages,
            "context_items_used": len(organized_context),
            "task_type": task_type,
            "total_context_tokens": self._estimate_tokens(combined_context),
        }

    def build_prompt_with_history(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        conversation_history: list[dict[str, str]],
        task_type: str = "conversational",
        include_metadata: bool = True,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a prompt that includes conversation history.

        Args:
            query: The user's latest query
            context_items: Retrieved context items
            conversation_history: List of previous conversation messages
            task_type: Type of task (defaults to "conversational")
            include_metadata: Whether to include metadata in context
            system_prompt_override: Optional override for the system prompt

        Returns:
            Complete prompt for LLM in message format
        """
        # Enrich and organize context
        if self.enable_metadata_enrichment:
            enriched_items = [self._enrich_context_item(item) for item in context_items]
        else:
            enriched_items = context_items

        organized_context = self._organize_context(enriched_items)

        # Format context items
        formatted_contexts = []
        for i, item in enumerate(organized_context):
            source_type = self._get_source_type(item)
            format_template = self.context_format_templates.get(
                source_type, self.context_format_templates["default"]
            )
            formatted_item = self._format_context_item(
                item, format_template, i + 1, include_metadata
            )
            formatted_contexts.append(formatted_item)

        # Combine contexts
        combined_context = "\n\n".join(formatted_contexts)

        # Select appropriate system prompt
        template_key = self.TEMPLATE_TYPES.get(task_type, task_type)
        task_template = self.task_templates.get(template_key, self.task_templates["conversational"])
        system_prompt = system_prompt_override or task_template

        # Add context to system prompt
        if combined_context:
            system_prompt = f"{system_prompt}\n\nRelevant Information:\n{combined_context}"

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:  # Skip empty messages
                messages.append({"role": role, "content": content})

        # Add the latest user query
        messages.append({"role": "user", "content": query})

        logger.info(
            f"Built prompt with conversation history ({len(conversation_history)} turns) and {len(organized_context)} context items"
        )

        return {
            "messages": messages,
            "context_items_used": len(organized_context),
            "history_turns": len(conversation_history),
            "task_type": task_type,
        }

    def _get_source_type(self, item: dict[str, Any]) -> str:
        """
        Determine the source type of a context item.

        Args:
            item: The context item

        Returns:
            Source type string
        """
        source = item.get("source", "").lower()

        # Check for conversation sources
        if source.startswith("conversation_"):
            return "conversation"

        # Check for known source types
        if source in ["corpus", "action", "feedback"]:
            return source

        # Default
        return "default"

    def _format_context_item(
        self, item: dict[str, Any], template: str, index: int, include_metadata: bool
    ) -> str:
        """
        Format a single context item using the specified template.

        Args:
            item: The context item to format
            template: The format template to use
            index: The index of this item in the context list
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        # Extract basic fields with defaults
        content = item.get("content", "")
        context = item.get("context", "")
        source = item.get("source", "unknown")
        relevance = item.get("relevance", 0.0)
        timestamp = item.get("timestamp", "")
        role = "unknown"

        # Extract role from source for conversation items
        if source.startswith("conversation_") and len(source) > 13:
            role = source[13:]  # Extract the part after "conversation_"

        # Format timestamp for display if present
        display_timestamp = ""
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = timestamp
                display_timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                display_timestamp = str(timestamp)

        # Prepare metadata section
        metadata_section = ""
        if include_metadata and item.get("metadata"):
            metadata_items = []
            for k, v in item["metadata"].items():
                if k != "role":  # Skip role since we display it separately for conversations
                    metadata_items.append(f"{k}: {v}")

            if metadata_items:
                metadata_str = ", ".join(metadata_items)
                metadata_section = f"Metadata: {metadata_str}\n"

        # Format with template
        try:
            formatted_item = template.format(
                index=index,
                content=content,
                context=context,
                source=source,
                relevance_score=f"{relevance:.2f}" if relevance else "N/A",
                timestamp=display_timestamp,
                role=role,
                metadata_section=metadata_section,
                context_section=f"Context: {context}\n" if context else "",
            )
            return formatted_item
        except KeyError as e:
            # Fallback to basic formatting if template has missing keys
            logger.warning(f"Error formatting context item with template: {e}")
            return (
                f"Context Item {index}:\nSource: {source}\nContent: {content}\n{metadata_section}"
            )

    def _organize_context(self, context_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Organize context items based on the configured strategy.

        Args:
            context_items: The context items to organize

        Returns:
            Organized list of context items
        """
        if not context_items:
            return []

        if self.context_organization_strategy == "relevance_first":
            # Sort primarily by relevance
            return sorted(context_items, key=lambda x: x.get("relevance", 0), reverse=True)
        elif self.context_organization_strategy == "recency_first":
            # Sort primarily by timestamp if available
            def get_timestamp(item):
                ts = item.get("timestamp")
                if not ts:
                    return 0

                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        return 0
                elif isinstance(ts, datetime):
                    return ts.timestamp()
                return 0

            return sorted(context_items, key=get_timestamp, reverse=True)
        elif self.context_organization_strategy == "hybrid":
            # Balance relevance and recency
            def hybrid_score(item):
                # Get relevance (0-1 scale)
                relevance = item.get("relevance", 0)

                # Get timestamp and convert to recency score (0-1 scale)
                ts = item.get("timestamp")
                recency = 0

                if ts:
                    try:
                        if isinstance(ts, str):
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        else:
                            dt = ts

                        # Calculate recency: 1.0 for now, decaying with age
                        now = datetime.now(timezone.utc)
                        age_hours = (now - dt).total_seconds() / 3600

                        # Exponential decay: 1.0 for fresh, 0.5 after 24h, approaching 0
                        recency = max(0, min(1, 2 ** (-age_hours / 24)))
                    except (ValueError, TypeError):
                        pass

                # Combine scores (65% relevance, 35% recency)
                return (0.65 * relevance) + (0.35 * recency)

            return sorted(context_items, key=hybrid_score, reverse=True)
        else:
            # Default to relevance sorting
            return sorted(context_items, key=lambda x: x.get("relevance", 0), reverse=True)

    def _enrich_context_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a context item with additional metadata and analysis.

        Args:
            item: The context item to enrich

        Returns:
            Enriched context item
        """
        # Create a copy of the item to avoid modifying the original
        enriched = item.copy()

        # Ensure metadata exists
        if "metadata" not in enriched:
            enriched["metadata"] = {}

        # Add length information
        content = enriched.get("content", "")
        content_length = len(content)
        enriched["metadata"]["length"] = content_length

        # Categorize by length
        if content_length < 100:
            length_category = "short"
        elif content_length < 500:
            length_category = "medium"
        else:
            length_category = "long"
        enriched["metadata"]["length_category"] = length_category

        # Extract entities if not already present
        if "entities" not in enriched["metadata"]:
            # Simple regex-based entity extraction
            # This is a basic implementation - in a real system, you might use NLP
            enriched["metadata"]["entities"] = self._extract_simple_entities(content)

        # Add content type analysis
        if "content_type" not in enriched["metadata"]:
            enriched["metadata"]["content_type"] = self._analyze_content_type(content)

        return enriched

    def _extract_simple_entities(self, text: str) -> list[str]:
        """
        Extract simple entities from text using regex patterns.

        Args:
            text: The text to analyze

        Returns:
            List of detected entities
        """
        entities = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', text)
        if quoted:
            entities.extend(quoted)

        # Extract potential named entities (capitalized phrases)
        # This is very basic - a real system would use NER
        capitalized = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", text)
        if capitalized:
            entities.extend([e for e in capitalized if len(e) > 1])

        return list(set(entities))

    def _analyze_content_type(self, text: str) -> str:
        """
        Analyze the type of content in the text.

        Args:
            text: The text to analyze

        Returns:
            Content type classification
        """
        # Check for code blocks
        if re.search(r"```|def\s+\w+\(|class\s+\w+[\(:]|import\s+\w+", text):
            return "code"

        # Check for lists
        if re.search(r"^\s*[\*\-\+•]\s+|^\s*\d+\.\s+", text, re.MULTILINE):
            return "list"

        # Check for json/structured data
        if re.search(r"^\s*[\{\[].*[\}\]]\s*$", text, re.MULTILINE):
            if re.search(r"[\{\[][\s\S]*[\}\]]", text):
                try:
                    json.loads(text)
                    return "json"
                except:
                    pass

        # Check for Q&A format
        if re.search(r"^\s*Q:|^\s*Question:|^\s*A:|^\s*Answer:", text, re.MULTILINE):
            return "qa_format"

        # Default to text
        return "text"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 chars per token on average
        return len(text) // 4
