"""
Tests for context analysis functionality.

This module provides tests for the ContextAnalyzer class, which handles
context item analysis, metadata enrichment, and content classification.
"""

from datetime import datetime, timezone
from typing import Any

from ltm_agent.retrieval.context_analyzer import ContextAnalyzer


def create_test_context_item(
    content: str,
    context: str = "",
    source: str = "corpus",
    relevance: float = 0.8,
    unique_id: str = "test-123",
    timestamp: str = None,
    metadata: dict[str, Any] = None,
) -> dict[str, Any]:
    """Create a test context item for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "content": content,
        "context": context,
        "source": source,
        "relevance": relevance,
        "id": unique_id,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }


class TestContextAnalyzer:
    """Tests for the ContextAnalyzer class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        analyzer = ContextAnalyzer()
        assert analyzer.enable_entity_extraction is True
        assert analyzer.enable_content_classification is True
        assert analyzer.enable_relationship_detection is False
        assert isinstance(analyzer.entity_patterns, dict)
        assert isinstance(analyzer.quality_weights, dict)

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_patterns = {"test_entity": r"\btest\b"}
        custom_weights = {"length": 0.5, "entities": 0.5}

        analyzer = ContextAnalyzer(
            enable_entity_extraction=False,
            enable_content_classification=False,
            enable_relationship_detection=True,
            custom_entity_patterns=custom_patterns,
            quality_scoring_weights=custom_weights,
        )

        assert analyzer.enable_entity_extraction is False
        assert analyzer.enable_content_classification is False
        assert analyzer.enable_relationship_detection is True
        assert analyzer.entity_patterns == custom_patterns
        assert analyzer.quality_weights == custom_weights

    def test_analyze_context_item_basic(self):
        """Test basic context item analysis."""
        analyzer = ContextAnalyzer()

        item = create_test_context_item(
            content="This is a simple test content.", context="Test Context"
        )

        enriched = analyzer.analyze_context_item(item)

        # Check basic metadata
        assert "metadata" in enriched
        assert "content_length" in enriched["metadata"]
        assert "length_category" in enriched["metadata"]
        assert enriched["metadata"]["length_category"] == "short"

        # Check content type
        assert "content_type" in enriched["metadata"]
        assert enriched["metadata"]["content_type"] == "text"

        # Check quality score
        assert "quality_score" in enriched["metadata"]
        assert 0 <= enriched["metadata"]["quality_score"] <= 1

    def test_analyze_context_item_with_entities(self):
        """Test entity extraction in context analysis."""
        analyzer = ContextAnalyzer()

        item = create_test_context_item(
            content="John Smith contacted us on April 12, 2023 at john.smith@example.com.",
            context="Customer Contact",
        )

        enriched = analyzer.analyze_context_item(item)

        # Check entities
        assert "entities" in enriched["metadata"]
        entities = enriched["metadata"]["entities"]

        # Should extract name, date, and email entities
        assert "name" in entities
        assert "John Smith" in entities["name"]

        assert "date" in entities
        assert "April 12, 2023" in entities["date"]

        assert "email" in entities
        assert "john.smith@example.com" in entities["email"]

        # Check all_entities list exists
        assert "all_entities" in enriched["metadata"]
        assert len(enriched["metadata"]["all_entities"]) >= 3

    def test_analyze_different_content_types(self):
        """Test content type classification."""
        analyzer = ContextAnalyzer()

        # Test code classification
        code_item = create_test_context_item(
            content="""
            def hello_world():
                print("Hello, world!")

            class Example:
                def __init__(self):
                    self.value = 42
            """,
            context="Python Code Example",
        )

        # Test list classification
        list_item = create_test_context_item(
            content="""
            Shopping list:
            - Apples
            - Bananas
            - Milk
            - Bread
            """,
            context="Shopping List",
        )

        # Test table classification
        table_item = create_test_context_item(
            content="""
            | Name | Age | City |
            |------|-----|------|
            | John | 30  | New York |
            | Alice | 25 | London |
            | Bob | 35 | Paris |
            """,
            context="User Table",
        )

        # Analyze all items
        code_result = analyzer.analyze_context_item(code_item)
        list_result = analyzer.analyze_context_item(list_item)
        table_result = analyzer.analyze_context_item(table_item)

        # Check content types
        assert code_result["metadata"]["content_type"] == "code"
        assert code_result["metadata"]["language"] == "python"

        assert list_result["metadata"]["content_type"] == "list"
        assert list_result["metadata"]["list_items"] == 4

        assert table_result["metadata"]["content_type"] == "table"
        assert table_result["metadata"]["table_rows"] == 3
        assert table_result["metadata"]["table_cols"] == 3

    def test_analyze_context_items_with_relationships(self):
        """Test relationship detection between context items."""
        analyzer = ContextAnalyzer(enable_relationship_detection=True)

        # Create similar items that should be related
        item1 = create_test_context_item(
            content="Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
            context="AI Fundamentals",
            unique_id="item1",
        )

        item2 = create_test_context_item(
            content="Artificial intelligence encompasses machine learning, deep learning, and neural networks.",
            context="AI Overview",
            unique_id="item2",
        )

        # Create unrelated item
        item3 = create_test_context_item(
            content="The weather forecast predicts rain tomorrow morning with clearing skies in the afternoon.",
            context="Weather Update",
            unique_id="item3",
        )

        # Analyze as a group
        items = [item1, item2, item3]
        enriched_items = analyzer.analyze_context_items(items)

        # Check relationships
        assert "related_items" in enriched_items[0]["metadata"]
        assert "related_items" in enriched_items[1]["metadata"]

        # Item 1 and 2 should be related to each other
        assert enriched_items[0]["metadata"]["related_items"][0]["index"] == 1
        assert enriched_items[1]["metadata"]["related_items"][0]["index"] == 0

        # Verify similarity scores and shared attributes
        assert "similarity" in enriched_items[0]["metadata"]["related_items"][0]
        assert "shared_topics" in enriched_items[0]["metadata"]["related_items"][0]

    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        analyzer = ContextAnalyzer()

        text = """
        Quantum computing leverages quantum mechanics to perform complex calculations.
        IBM and Google are leading companies in this field.
        "Quantum supremacy" was a milestone achieved in recent years.
        """

        phrases = analyzer.extract_key_phrases(text)

        assert isinstance(phrases, list)
        assert len(phrases) > 0
        assert "Quantum computing" in phrases or "quantum mechanics" in phrases
        assert "IBM" in phrases or "Google" in phrases
        assert "Quantum supremacy" in phrases

    def test_calculate_semantic_similarity(self):
        """Test semantic similarity calculation between context items."""
        analyzer = ContextAnalyzer()

        item1 = create_test_context_item(
            content="Neural networks are used in deep learning applications.",
            context="Deep Learning",
        )

        item2 = create_test_context_item(
            content="Deep learning applications often use neural networks for pattern recognition.",
            context="AI Applications",
        )

        item3 = create_test_context_item(
            content="The book has a red cover and was published in 2020.",
            context="Book Description",
        )

        # Calculate similarities
        sim1_2 = analyzer.calculate_semantic_similarity(item1, item2)
        sim1_3 = analyzer.calculate_semantic_similarity(item1, item3)

        # Similar items should have higher similarity
        assert sim1_2 > 0.3
        assert sim1_3 < 0.2
        assert sim1_2 > sim1_3

    def test_quality_scoring(self):
        """Test quality scoring calculation."""
        analyzer = ContextAnalyzer()

        # Test item with good quality indicators
        good_item = create_test_context_item(
            content="""
            # Machine Learning Overview

            Machine learning is a subset of artificial intelligence focusing on algorithms that learn from data.

            Key techniques include:
            - Supervised learning
            - Unsupervised learning
            - Reinforcement learning

            Common applications:
            1. Image recognition
            2. Natural language processing
            3. Recommendation systems

            Google, Microsoft, and Facebook all use machine learning extensively.
            """,
            context="ML Introduction",
            metadata={"topics": ["machine learning", "AI", "algorithms", "data science"]},
        )

        # Test item with lower quality indicators
        basic_item = create_test_context_item(
            content="Machine learning is used for many applications.", context="ML Note"
        )

        # Calculate quality scores
        enriched_good = analyzer.analyze_context_item(good_item)
        enriched_basic = analyzer.analyze_context_item(basic_item)

        good_score = enriched_good["metadata"]["quality_score"]
        basic_score = enriched_basic["metadata"]["quality_score"]

        # Good quality item should have higher score
        assert good_score > 0.6
        assert basic_score < 0.6
        assert good_score > basic_score

    def test_empty_and_edge_cases(self):
        """Test handling of empty or edge case inputs."""
        analyzer = ContextAnalyzer()

        # Empty content
        empty_item = create_test_context_item(content="")
        empty_result = analyzer.analyze_context_item(empty_item)
        assert empty_result["metadata"]["content_length"] == 0

        # Empty item dictionary
        null_result = analyzer.analyze_context_item({})
        assert null_result == {}

        # Empty list for analyze_context_items
        null_list_result = analyzer.analyze_context_items([])
        assert null_list_result == []
