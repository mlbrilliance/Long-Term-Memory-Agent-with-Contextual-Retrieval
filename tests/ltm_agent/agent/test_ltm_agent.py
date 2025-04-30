"""
Tests for LongTermMemoryAgent.

This module contains tests for the LongTermMemoryAgent class, which integrates
the memory components with the core LLM for the main action cycle.
"""

from datetime import datetime
from unittest.mock import MagicMock, call

import pytest

from ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.interfaces import BaseContextualizer
from ltm_agent.memory.manager import MemoryManager


class TestLongTermMemoryAgent:
    """Tests for the LongTermMemoryAgent."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = MagicMock(spec=MemoryManager)
        # Setup common return values
        manager.get_related_knowledge.return_value = []
        return manager

    @pytest.fixture
    def mock_contextualizer(self):
        """Create a mock contextualizer."""
        contextualizer = MagicMock(spec=BaseContextualizer)
        contextualizer.enhance_context.return_value = "Enhanced context"
        return contextualizer

    @pytest.fixture
    def mock_llm(self):
        """Create a mock language model."""
        llm = MagicMock()
        llm.invoke.return_value = "Mock LLM response"
        return llm

    def test_init(self, mock_memory_manager, mock_contextualizer, mock_llm):
        """Test initializing the LongTermMemoryAgent with dependencies."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        assert agent.memory_manager == mock_memory_manager
        assert agent.contextualizer == mock_contextualizer
        assert agent.llm == mock_llm

    def test_invoke_with_no_relevant_knowledge(
        self, mock_memory_manager, mock_contextualizer, mock_llm
    ):
        """Test invoking the agent with no relevant knowledge from memory."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        # Mock that no relevant knowledge is found
        mock_memory_manager.get_related_knowledge.return_value = []

        query = "What is the capital of France?"
        response = agent.invoke(query)

        # Verify that the memory manager was called to retrieve knowledge
        mock_memory_manager.get_related_knowledge.assert_called_once_with(
            content=query,
            limit=10,  # Assuming default limit is 10
        )

        # Verify that the LLM was called with the query
        mock_llm.invoke.assert_called_once()

        # Check that we got a response from the LLM
        assert response is not None
        assert response == "Mock LLM response"

    def test_invoke_with_relevant_knowledge(
        self, mock_memory_manager, mock_contextualizer, mock_llm
    ):
        """Test invoking the agent with relevant knowledge from memory."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        # Mock relevant knowledge units
        knowledge_units = [
            KnowledgeUnit(
                unique_id="1",
                original_chunk="Paris is the capital of France",
                contextual_text="Paris is the capital of France and one of the most populous cities in Europe.",
                embedding_vector=[],
                knowledge_source="corpus",
                timestamp=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                metadata={"source": "Wikipedia"},
            )
        ]

        # Mock that relevant knowledge is found
        mock_memory_manager.get_related_knowledge.return_value = knowledge_units

        query = "What is the capital of France?"
        response = agent.invoke(query)

        # Verify that the memory manager was called to retrieve knowledge
        mock_memory_manager.get_related_knowledge.assert_called_once_with(
            content=query,
            limit=10,  # Assuming default limit is 10
        )

        # Verify that the LLM was called with the query and context
        mock_llm.invoke.assert_called_once()

        # Check that the call to LLM included the context from the knowledge unit
        llm_args = mock_llm.invoke.call_args[0][0]
        assert "Paris is the capital of France" in llm_args

        # Check that we got a response from the LLM
        assert response is not None
        assert response == "Mock LLM response"

    def test_learn_from_interaction(self, mock_memory_manager, mock_contextualizer, mock_llm):
        """Test that the agent learns from interactions."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        # Set up the mock LLM to return a response that contains information
        llm_response = "Paris is the capital of France. It's known as the City of Light."
        mock_llm.invoke.return_value = llm_response

        query = "What is the capital of France?"
        response = agent.invoke(query)

        # Verify that the agent tried to process the response as potential knowledge
        mock_memory_manager.process_potential_knowledge.assert_called_once()

        # The expected call with keyword arguments
        expected_call = call(
            content=llm_response, source="LLM Response", metadata={"type": "response"}
        )

        # Check if the actual call matches the expected call
        assert mock_memory_manager.process_potential_knowledge.call_args == expected_call

    def test_format_context(self, mock_memory_manager, mock_contextualizer, mock_llm):
        """Test that the agent formats context correctly for the LLM."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        # Mock relevant knowledge units
        knowledge_units = [
            KnowledgeUnit(
                unique_id="1",
                original_chunk="Paris is the capital of France",
                contextual_text="Paris is the capital of France and one of the most populous cities in Europe.",
                embedding_vector=[],
                knowledge_source="corpus",
                timestamp=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
                metadata={"source": "Wikipedia"},
            ),
            KnowledgeUnit(
                unique_id="2",
                original_chunk="France is in Western Europe",
                contextual_text="France is a country located in Western Europe with a rich history and culture.",
                embedding_vector=[],
                knowledge_source="corpus",
                timestamp=datetime.fromisoformat("2023-02-01T00:00:00+00:00"),
                metadata={"source": "Textbook"},
            ),
        ]

        # Test the internal _format_context method
        context = agent._format_context(knowledge_units)

        # Context should include information from both knowledge units
        assert "Paris is the capital of France" in context
        assert "France is in Western Europe" in context

        # Context should be formatted in a structured way
        assert "RELEVANT KNOWLEDGE" in context

    def test_process_learning(self, mock_memory_manager, mock_contextualizer, mock_llm):
        """Test the learning processing method."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        response = "Paris is the capital of France. It's known as the City of Light."

        # Call the method directly
        agent._process_learning(response)

        # Verify that the memory manager processed the response
        mock_memory_manager.process_potential_knowledge.assert_called_once_with(
            content=response, source="LLM Response", metadata={"type": "response"}
        )

    def test_process_human_feedback(self, mock_memory_manager, mock_contextualizer, mock_llm):
        """Test processing human feedback."""
        agent = LongTermMemoryAgent(
            memory_manager=mock_memory_manager, contextualizer=mock_contextualizer, llm=mock_llm
        )

        feedback = "The Eiffel Tower is in Paris, France."

        # Call the method directly
        agent.process_feedback(feedback)

        # Verify that the memory manager processed the feedback
        mock_memory_manager.process_potential_knowledge.assert_called_once_with(
            content=feedback, source="Human Feedback", metadata={"type": "feedback"}
        )
