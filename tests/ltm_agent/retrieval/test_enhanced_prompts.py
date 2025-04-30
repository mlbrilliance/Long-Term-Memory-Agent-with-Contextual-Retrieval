"""
Tests for enhanced prompt building functionality.

This module provides tests for the EnhancedPromptBuilder class, which handles
advanced context formatting and prompt construction for LLM queries.
"""

from datetime import datetime, timezone
from typing import Any

from ltm_agent.retrieval.enhanced_prompts import EnhancedPromptBuilder


def create_test_context_item(
    content: str,
    context: str = "",
    source: str = "corpus",
    relevance: float = 0.8,
    unique_id: str = "test-123",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a test context item for testing."""
    return {
        "content": content,
        "context": context,
        "source": source,
        "relevance": relevance,
        "id": unique_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


class TestEnhancedPromptBuilder:
    """Test suite for EnhancedPromptBuilder."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        builder = EnhancedPromptBuilder()
        assert builder.max_prompt_tokens == 8000
        assert builder.enable_metadata_enrichment is True
        assert builder.context_organization_strategy == "relevance_first"
        assert "default" in builder.context_format_templates
        assert "corpus" in builder.context_format_templates
        assert "action" in builder.context_format_templates
        assert "feedback" in builder.context_format_templates
        assert "conversation" in builder.context_format_templates

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_templates = {
            "default": "Custom default template {content}",
            "corpus": "Custom corpus template {content}",
        }
        builder = EnhancedPromptBuilder(
            system_prompt_template="Custom system prompt",
            context_format_templates=custom_templates,
            max_prompt_tokens=4000,
            enable_metadata_enrichment=False,
            context_organization_strategy="recency_first",
        )

        assert builder.max_prompt_tokens == 4000
        assert builder.enable_metadata_enrichment is False
        assert builder.context_organization_strategy == "recency_first"
        assert builder.system_prompt_template == "Custom system prompt"
        assert builder.context_format_templates["default"] == custom_templates["default"]
        assert builder.context_format_templates["corpus"] == custom_templates["corpus"]

        # Should still have the default templates for sources not specified
        assert "action" in builder.context_format_templates
        assert "feedback" in builder.context_format_templates
        assert "conversation" in builder.context_format_templates

    def test_set_template_for_task(self):
        """Test setting a custom template for a task type."""
        builder = EnhancedPromptBuilder()

        # Set a custom template for an existing task type
        builder.set_template_for_task("qa", "Custom QA template")
        assert builder.task_templates["question_answering"] == "Custom QA template"

        # Set a custom template for a new task type
        builder.set_template_for_task("custom_task", "Custom task template")
        assert builder.task_templates["custom_task"] == "Custom task template"

    def test_build_prompt_basic(self):
        """Test basic prompt building without context items."""
        builder = EnhancedPromptBuilder()
        query = "What is machine learning?"

        result = builder.build_prompt(query, [])

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == query
        assert result["context_items_used"] == 0
        assert result["task_type"] == "question_answering"

    def test_build_prompt_with_context(self):
        """Test prompt building with context items."""
        builder = EnhancedPromptBuilder()
        query = "Explain reinforcement learning"

        context_items = [
            create_test_context_item(
                content="Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment to maximize rewards.",
                context="Reinforcement Learning Basics",
                source="corpus",
                relevance=0.95,
                unique_id="rl-1",
                metadata={"domain": "machine_learning", "difficulty": "intermediate"},
            ),
            create_test_context_item(
                content="Deep Q-Networks (DQN) combine deep learning with Q-learning for reinforcement learning with high-dimensional state spaces.",
                context="Deep Reinforcement Learning",
                source="corpus",
                relevance=0.85,
                unique_id="rl-2",
                metadata={"domain": "machine_learning", "subtopic": "deep_rl"},
            ),
        ]

        result = builder.build_prompt(query, context_items)

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "Reinforcement learning is a type" in result["messages"][0]["content"]
        assert "Deep Q-Networks" in result["messages"][0]["content"]
        assert result["messages"][1]["content"] == query
        assert result["context_items_used"] == 2
        assert result["task_type"] == "question_answering"

        # Check that context items are sorted by relevance (highest first)
        system_content = result["messages"][0]["content"]
        rl_basics_pos = system_content.find("Reinforcement Learning Basics")
        deep_rl_pos = system_content.find("Deep Reinforcement Learning")
        assert rl_basics_pos < deep_rl_pos

    def test_build_prompt_with_different_task_types(self):
        """Test prompt building with different task types."""
        builder = EnhancedPromptBuilder()
        query = "Summarize the key concepts of neural networks"

        context_item = create_test_context_item(
            content="Neural networks are computational models inspired by the human brain, consisting of layers of interconnected nodes that process information.",
            context="Neural Networks Overview",
        )

        # Test with summarization task type
        result_summary = builder.build_prompt(query, [context_item], task_type="summarization")

        # Test with reasoning task type
        result_reasoning = builder.build_prompt(query, [context_item], task_type="reasoning")

        # Verify different system prompts were used
        assert result_summary["task_type"] == "summarization"
        assert result_reasoning["task_type"] == "reasoning"
        assert (
            result_summary["messages"][0]["content"] != result_reasoning["messages"][0]["content"]
        )

    def test_metadata_enrichment(self):
        """Test that context items are properly enriched with metadata."""
        builder = EnhancedPromptBuilder(enable_metadata_enrichment=True)

        context_item = create_test_context_item(
            content="Python is a high-level, interpreted programming language.",
            context="Python Overview",
            metadata={"domain": "programming"},
        )

        result = builder.build_prompt("What is Python?", [context_item])

        # Check that the metadata was included and enriched
        system_content = result["messages"][0]["content"]
        assert "domain: programming" in system_content
        assert "length_category" in system_content  # Added by enrichment

    def test_organization_strategies(self):
        """Test different context organization strategies."""
        # Create items with different relevance scores and timestamps
        old_time = datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat()
        new_time = datetime(2023, 6, 1, tzinfo=timezone.utc).isoformat()

        items = [
            # High relevance, old timestamp
            {"content": "Item 1", "relevance": 0.9, "timestamp": old_time, "id": "1"},
            # Low relevance, new timestamp
            {"content": "Item 2", "relevance": 0.5, "timestamp": new_time, "id": "2"},
        ]

        # Test relevance_first strategy
        builder_relevance = EnhancedPromptBuilder(context_organization_strategy="relevance_first")
        result_relevance = builder_relevance.build_prompt("Query", items)
        system_content_relevance = result_relevance["messages"][0]["content"]
        assert system_content_relevance.find("Item 1") < system_content_relevance.find("Item 2")

        # Test recency_first strategy
        builder_recency = EnhancedPromptBuilder(context_organization_strategy="recency_first")
        result_recency = builder_recency.build_prompt("Query", items)
        system_content_recency = result_recency["messages"][0]["content"]
        assert system_content_recency.find("Item 2") < system_content_recency.find("Item 1")

    def test_build_prompt_with_history(self):
        """Test building a prompt with conversation history."""
        builder = EnhancedPromptBuilder()
        query = "Can you explain more about that?"

        context_item = create_test_context_item(
            content="Gradient descent is an optimization algorithm used to minimize a function by iteratively moving toward the steepest descent.",
            context="Optimization Algorithms",
        )

        conversation_history = [
            {"role": "user", "content": "What is gradient descent?"},
            {
                "role": "assistant",
                "content": "Gradient descent is an optimization algorithm used in machine learning.",
            },
        ]

        result = builder.build_prompt_with_history(query, [context_item], conversation_history)

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 4  # system + 2 history + 1 query
        assert result["messages"][0]["role"] == "system"
        assert "Gradient descent" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "What is gradient descent?"
        assert result["messages"][2]["role"] == "assistant"
        assert result["messages"][3]["role"] == "user"
        assert result["messages"][3]["content"] == query
        assert result["context_items_used"] == 1
        assert result["history_turns"] == 2
        assert result["task_type"] == "conversational"

    def test_different_source_types(self):
        """Test prompt building with different source types."""
        builder = EnhancedPromptBuilder()

        context_items = [
            create_test_context_item(
                content="Neural networks are used in deep learning.",
                context="Deep Learning",
                source="corpus",
            ),
            create_test_context_item(
                content="Used the model to predict house prices.",
                context="Price Prediction",
                source="action",
            ),
            create_test_context_item(
                content="The explanation was clear and helpful.",
                context="User Satisfaction",
                source="feedback",
            ),
            {
                "content": "What is a neural network?",
                "source": "conversation_user",
                "relevance": 0.8,
                "id": "conv-1",
                "metadata": {"turn_index": 0},
            },
        ]

        result = builder.build_prompt("Tell me more about neural networks", context_items)

        # Verify each source type uses its own format template
        system_content = result["messages"][0]["content"]
        assert "Knowledge Item" in system_content  # corpus template
        assert "Previous Action" in system_content  # action template
        assert "User Feedback" in system_content  # feedback template
        assert "Conversation History" in system_content  # conversation template

    def test_build_prompt_with_additional_instructions(self):
        """Test building a prompt with additional instructions."""
        builder = EnhancedPromptBuilder()
        query = "Compare supervised and unsupervised learning"

        context_item = create_test_context_item(
            content="Supervised learning uses labeled data, while unsupervised learning works with unlabeled data.",
            context="Machine Learning Methods",
        )

        additional_instructions = "Format your response as a table with clear pros and cons."

        result = builder.build_prompt(
            query, [context_item], additional_instructions=additional_instructions
        )

        assert "Additional Instructions" in result["messages"][0]["content"]
        assert additional_instructions in result["messages"][0]["content"]
