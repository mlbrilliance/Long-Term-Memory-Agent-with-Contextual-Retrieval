"""
Simple Enhanced Retrieval Example

This script demonstrates the enhanced prompt building and context analysis
capabilities without relying on external dependencies.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ltm_agent.retrieval.context_analyzer import ContextAnalyzer
from ltm_agent.retrieval.enhanced_prompts import EnhancedPromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("simple_enhanced_retrieval")


def create_test_context_items() -> list[dict[str, Any]]:
    """Create test context items for the example."""

    return [
        {
            "content": "Machine learning is a field of inquiry devoted to understanding and building methods "
            + "that 'learn', that is, methods that leverage data to improve performance on some task.",
            "context": "Introduction to Machine Learning",
            "source": "corpus",
            "relevance": 0.92,
            "id": "k1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "importance": "high"},
        },
        {
            "content": "Deep learning is a type of machine learning based on artificial neural networks in "
            + "which multiple layers of processing are used to extract progressively higher level features from data.",
            "context": "Deep Learning Overview",
            "source": "corpus",
            "relevance": 0.85,
            "id": "k2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "subtopic": "deep_learning"},
        },
        {
            "content": "Reinforcement learning is an area of machine learning concerned with how intelligent "
            + "agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
            "context": "Reinforcement Learning",
            "source": "corpus",
            "relevance": 0.78,
            "id": "k3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "subtopic": "reinforcement_learning"},
        },
        {
            "content": """
            def train_model(data, labels, epochs=10):
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(data.shape[1],)),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                history = model.fit(data, labels, epochs=epochs, validation_split=0.2)
                return model, history
            """,
            "context": "Neural Network Training Function",
            "source": "corpus",
            "relevance": 0.75,
            "id": "k5",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "domain": "machine_learning",
                "content_type": "code",
                "language": "python",
            },
        },
        {
            "content": "The bias-variance tradeoff is a central problem in supervised learning. Ideally, one "
            + "wants to choose a model that both accurately captures the regularities in its training "
            + "data, but also generalizes well to unseen data.",
            "context": "Bias-Variance Tradeoff",
            "source": "corpus",
            "relevance": 0.89,
            "id": "k7",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "concept": "bias_variance"},
        },
        {
            "content": """
            # Common Activation Functions

            1. ReLU (Rectified Linear Unit)
            2. Sigmoid
            3. Tanh (Hyperbolic Tangent)
            4. Leaky ReLU
            5. ELU (Exponential Linear Unit)

            Each has different properties and use cases in neural networks.
            """,
            "context": "Neural Network Activation Functions",
            "source": "corpus",
            "relevance": 0.72,
            "id": "k8",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "content_type": "list"},
        },
        {
            "content": """
            Q: What is cross-validation?
            A: Cross-validation is a resampling procedure used to evaluate machine learning models
               on a limited data sample. The procedure has a single parameter called k that refers
               to the number of groups that a given data sample is to be split into.

            Q: Why is cross-validation important?
            A: It helps to assess how the results of a statistical analysis will generalize to an
               independent dataset, reducing problems like overfitting and selection bias.
            """,
            "context": "Cross-validation in Machine Learning",
            "source": "corpus",
            "relevance": 0.83,
            "id": "k10",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"domain": "machine_learning", "content_type": "qa_format"},
        },
    ]


def demonstrate_context_analysis():
    """Demonstrate the context analysis functionality."""

    logger.info("=== Testing Context Analysis ===")

    # Create context items
    context_items = create_test_context_items()

    # Create context analyzer
    context_analyzer = ContextAnalyzer(
        enable_entity_extraction=True,
        enable_content_classification=True,
        enable_relationship_detection=True,
    )

    # Analyze context items
    enriched_items = context_analyzer.analyze_context_items(context_items)

    # Show analysis results
    logger.info(f"Analyzed {len(enriched_items)} context items")

    for i, item in enumerate(enriched_items):
        original_item = context_items[i]
        original_keys = set(original_item.get("metadata", {}).keys())
        enriched_keys = set(item.get("metadata", {}).keys())
        new_keys = enriched_keys - original_keys

        logger.info(f"\nItem {i + 1} - ID: {item['id']}")
        logger.info(f"  Content: {item['content'][:50]}...")
        logger.info(f"  Added {len(new_keys)} new metadata fields: {', '.join(new_keys)}")

        if "content_type" in item["metadata"]:
            logger.info(f"  Classified as: {item['metadata']['content_type']}")

        if "entities" in item["metadata"]:
            entity_count = sum(len(entities) for entities in item["metadata"]["entities"].values())
            if entity_count > 0:
                logger.info(f"  Extracted {entity_count} entities:")
                for entity_type, entities in item["metadata"]["entities"].items():
                    if entities:
                        logger.info(f"    - {entity_type}: {', '.join(entities[:3])}")

        if "quality_score" in item["metadata"]:
            logger.info(f"  Quality score: {item['metadata']['quality_score']:.4f}")

        if "adjusted_relevance" in item:
            original = original_item.get("relevance", 0)
            adjusted = item["adjusted_relevance"]
            logger.info(f"  Relevance adjusted: {original:.4f} → {adjusted:.4f}")

        if "topics" in item["metadata"]:
            logger.info(f"  Detected topics: {', '.join(item['metadata']['topics'])}")

    # Test key phrase extraction
    logger.info("\n=== Testing Key Phrase Extraction ===")

    sample_text = """
    Machine learning has revolutionized many industries including healthcare, finance, and transportation.
    Deep neural networks, a subset of machine learning models, have shown remarkable performance on tasks
    like image recognition and natural language processing. Companies like Google, Amazon, and Microsoft
    have made significant investments in AI research and development.
    """

    key_phrases = context_analyzer.extract_key_phrases(sample_text)
    logger.info(f"Extracted key phrases: {key_phrases}")

    # Test relationship detection
    if any("related_items" in item.get("metadata", {}) for item in enriched_items):
        logger.info("\n=== Testing Relationship Detection ===")

        for i, item in enumerate(enriched_items):
            if "related_items" in item.get("metadata", {}):
                related = item["metadata"]["related_items"]

                logger.info(f"Item {i + 1} has {len(related)} related items:")
                for rel in related[:2]:  # Show top 2 related items
                    rel_idx = rel["index"]
                    rel_sim = rel["similarity"]
                    rel_item_id = enriched_items[rel_idx]["id"]
                    logger.info(f"  - Related to {rel_item_id} with similarity {rel_sim:.4f}")

                    if "shared_topics" in rel and rel["shared_topics"]:
                        logger.info(f"    Shared topics: {', '.join(rel['shared_topics'][:3])}")


def demonstrate_enhanced_prompts():
    """Demonstrate the enhanced prompt building functionality."""

    logger.info("\n=== Testing Enhanced Prompt Building ===")

    # Create context items
    context_items = create_test_context_items()

    # Create enhanced prompt builder
    prompt_builder = EnhancedPromptBuilder(
        context_organization_strategy="hybrid", enable_metadata_enrichment=True
    )

    # Test different query types
    test_queries = [
        ("What is machine learning?", "question_answering"),
        ("Summarize the key concepts of neural networks", "summarization"),
        ("Explain the pros and cons of different activation functions", "reasoning"),
    ]

    for query, task_type in test_queries:
        logger.info(f"\nQuery: {query} (Task: {task_type})")

        # Build prompt
        prompt_result = prompt_builder.build_prompt(query, context_items, task_type=task_type)

        logger.info(f"Built prompt using {prompt_result['context_items_used']} context items")

        # Show the system message (first 100 chars)
        system_content = prompt_result["messages"][0]["content"]
        logger.info(f"System prompt (first 100 chars): {system_content[:100]}...")

        # Count tokens
        approx_tokens = len(system_content) // 4  # Rough approximation
        logger.info(f"Approximate token count: {approx_tokens}")

    # Test with conversation history
    logger.info("\n=== Testing Conversation History Integration ===")

    # Create conversation history
    conversation_history = [
        {"role": "user", "content": "Tell me about machine learning"},
        {
            "role": "assistant",
            "content": "Machine learning is a field that focuses on algorithms that learn from data.",
        },
        {"role": "user", "content": "What about deep learning?"},
        {
            "role": "assistant",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        },
    ]

    # Build conversational prompt
    conv_result = prompt_builder.build_prompt_with_history(
        "How does cross-validation help with model evaluation?", context_items, conversation_history
    )

    logger.info(
        f"Built conversational prompt with {conv_result['context_items_used']} context items and {conv_result['history_turns']} conversation turns"
    )

    # Show message structure
    logger.info(f"Message structure: {len(conv_result['messages'])} total messages")
    for i, msg in enumerate(conv_result["messages"]):
        logger.info(f"  Message {i + 1}: role={msg['role']}, content_length={len(msg['content'])}")


if __name__ == "__main__":
    try:
        demonstrate_context_analysis()
        demonstrate_enhanced_prompts()
        logger.info("\nAll demonstrations completed successfully!")
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}", exc_info=True)
