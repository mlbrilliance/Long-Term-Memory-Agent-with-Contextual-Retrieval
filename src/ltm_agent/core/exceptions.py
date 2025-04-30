"""
Exception classes for the LTM Agent.

This module defines custom exceptions used throughout the application to
provide more specific error handling and messaging.
"""


class LTMAgentError(Exception):
    """Base exception class for all LTM Agent errors."""

    pass


class MemoryError(LTMAgentError):
    """Base exception class for memory-related errors."""

    pass


class MemoryStoreError(MemoryError):
    """Exception raised for errors in the memory store."""

    pass


class MemoryInitializationError(MemoryError):
    """Exception raised when memory initialization fails."""

    pass


class KnowledgeUnitError(MemoryError):
    """Base exception class for knowledge unit errors."""

    pass


class KnowledgeUnitNotFoundError(KnowledgeUnitError):
    """Exception raised when a knowledge unit cannot be found."""

    pass


class KnowledgeUnitUpdateError(KnowledgeUnitError):
    """Exception raised when updating a knowledge unit fails."""

    pass


class KnowledgeUnitDeleteError(KnowledgeUnitError):
    """Exception raised when deleting a knowledge unit fails."""

    pass


class EmbeddingError(MemoryError):
    """Exception raised for errors in embedding generation."""

    pass
