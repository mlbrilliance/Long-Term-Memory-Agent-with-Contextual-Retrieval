"""
Contextualizer interfaces and implementations for the LTM Agent.

This module provides interfaces and implementations for generating contextual
information for knowledge units. A Contextualizer helps enhance knowledge
with relevant metadata and additional context.
"""

from abc import ABC, abstractmethod
from typing import Any

from ltm_agent.memory.interfaces import BaseContextualizer


class Contextualizer(ABC):
    """
    Abstract base class for contextualizers.

    A Contextualizer generates or enhances contextual information for knowledge units.
    This helps in organizing, categorizing, and retrieving knowledge more effectively.
    """

    @abstractmethod
    async def generate_context(
        self, content: str, existing_context: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate or enhance contextual information for knowledge content.

        Args:
            content: The main content of the knowledge unit
            existing_context: Optional existing context to enhance
            metadata: Optional existing metadata to enhance

        Returns:
            Dict with two keys:
            - 'context': Enhanced context for the knowledge unit
            - 'metadata': Enhanced metadata for the knowledge unit
        """
        pass


class SimpleContextualizer(BaseContextualizer):
    """
    A simple implementation of the BaseContextualizer interface.

    This contextualizer preserves existing context and metadata without modification.
    It can be extended to provide more sophisticated context generation.
    """

    def enhance_context(
        self, content: str, existing_context: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Passes through existing context and metadata without modifications.

        Args:
            content: The main content of the knowledge unit
            existing_context: Optional existing context to preserve
            metadata: Optional existing metadata to preserve

        Returns:
            Dict containing contextual_text and metadata
        """
        if not existing_context:
            # If no existing context, use the content as the context
            contextual_text = content
        else:
            contextual_text = existing_context

        return {"contextual_text": contextual_text, "metadata": metadata or {}}

    async def generate_context(
        self, content: str, existing_context: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Compatibility method to implement the original Contextualizer interface.

        This delegates to enhance_context but returns the response in the format expected
        by the original Contextualizer interface.

        Args:
            content: The main content of the knowledge unit
            existing_context: Optional existing context to enhance
            metadata: Optional existing metadata to enhance

        Returns:
            Dict with two keys:
            - 'context': Enhanced context for the knowledge unit
            - 'metadata': Enhanced metadata for the knowledge unit
        """
        enhanced = self.enhance_context(content, existing_context, metadata)

        # Convert to original Contextualizer format
        return {"context": enhanced["contextual_text"], "metadata": enhanced["metadata"]}
