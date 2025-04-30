"""
Enhanced Retrieval Example

This script demonstrates the integration of the enhanced prompt building,
context analysis, and hybrid retrieval capabilities of the LTM Agent.
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.langchain_vector_store import LangchainVectorStore
from ltm_agent.memory.manager import MemoryManager
from ltm_agent.retrieval.context_analyzer import ContextAnalyzer
from ltm_agent.retrieval.context_retriever import ContextRetriever
from ltm_agent.retrieval.enhanced_prompts import EnhancedPromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("enhanced_retrieval_example")


def create_test_knowledge_unit(
    unique_id: str,
    content: str,
    context: str = "",
    metadata: dict[str, Any] = None,
    timestamp: datetime | None = None,
    knowledge_source: str = "testing",
) -> KnowledgeUnit:
    """Create a test knowledge unit for the example."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return KnowledgeUnit(
        unique_id=unique_id,
        content=content,
        context=context,
        knowledge_source=knowledge_source,
        metadata=metadata or {},
        timestamp=timestamp,
    )


async def setup_memory() -> tuple[MemoryManager, list[KnowledgeUnit]]:
    """Set up the memory manager with test data."""

    # Create test knowledge units
    test_units = [
        create_test_knowledge_unit(
            "k1",
            "Machine learning is a field of inquiry devoted to understanding and building methods "
            + "that 'learn', that is, methods that leverage data to improve performance on some task.",
            "Introduction to Machine Learning",
            metadata={"domain": "machine_learning", "importance": "high"},
        ),
        create_test_knowledge_unit(
            "k2",
            "Deep learning is a type of machine learning based on artificial neural networks in "
            + "which multiple layers of processing are used to extract progressively higher level features from data.",
            "Deep Learning Overview",
            metadata={"domain": "machine_learning", "subtopic": "deep_learning"},
        ),
        create_test_knowledge_unit(
            "k3",
            "Reinforcement learning is an area of machine learning concerned with how intelligent "
            + "agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
            "Reinforcement Learning",
            metadata={"domain": "machine_learning", "subtopic": "reinforcement_learning"},
        ),
        create_test_knowledge_unit(
            "k4",
            "Python is a high-level, general-purpose programming language. Its design philosophy "
            + "emphasizes code readability with the use of significant indentation.",
            "Python Programming Language",
            metadata={"domain": "programming", "language": "python"},
        ),
        create_test_knowledge_unit(
            "k5",
            """
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
            "Neural Network Training Function",
            metadata={"domain": "machine_learning", "content_type": "code", "language": "python"},
        ),
        create_test_knowledge_unit(
            "k6",
            "Support vector machines (SVMs) are supervised learning models with associated learning "
            + "algorithms that analyze data for classification and regression analysis.",
            "Support Vector Machines",
            metadata={"domain": "machine_learning", "algorithm": "svm"},
        ),
        create_test_knowledge_unit(
            "k7",
            "The bias-variance tradeoff is a central problem in supervised learning. Ideally, one "
            + "wants to choose a model that both accurately captures the regularities in its training "
            + "data, but also generalizes well to unseen data.",
            "Bias-Variance Tradeoff",
            metadata={"domain": "machine_learning", "concept": "bias_variance"},
        ),
        create_test_knowledge_unit(
            "k8",
            """
            # Common Activation Functions

            1. ReLU (Rectified Linear Unit)
            2. Sigmoid
            3. Tanh (Hyperbolic Tangent)
            4. Leaky ReLU
            5. ELU (Exponential Linear Unit)

            Each has different properties and use cases in neural networks.
            """,
            "Neural Network Activation Functions",
            metadata={"domain": "machine_learning", "content_type": "list"},
        ),
        create_test_knowledge_unit(
            "k9",
            "Feature engineering is the process of using domain knowledge to extract features from "
            + "raw data via data mining techniques. These features can be used to improve the performance "
            + "of machine learning algorithms.",
            "Feature Engineering",
            metadata={"domain": "data_science", "subtopic": "feature_engineering"},
        ),
        create_test_knowledge_unit(
            "k10",
            """
            Q: What is cross-validation?
            A: Cross-validation is a resampling procedure used to evaluate machine learning models
               on a limited data sample. The procedure has a single parameter called k that refers
               to the number of groups that a given data sample is to be split into.

            Q: Why is cross-validation important?
            A: It helps to assess how the results of a statistical analysis will generalize to an
               independent dataset, reducing problems like overfitting and selection bias.
            """,
            "Cross-validation in Machine Learning",
            metadata={"domain": "machine_learning", "content_type": "qa_format"},
        ),
    ]

    try:
        # Create temporary directory for vector store
        temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp_vectorstore")
        os.makedirs(temp_dir, exist_ok=True)

        # Create in-memory stores
        vector_store = LangchainVectorStore(persist_directory=temp_dir)
        bm25_store = RankBM25Store()

        # Initialize memory manager
        memory_manager = MemoryManager(vector_store=vector_store, bm25_store=bm25_store)

        # Initialize stores
        await vector_store.initialize()

        # Add knowledge units to memory
        for unit in test_units:
            await memory_manager.add_knowledge(unit)

        logger.info(f"Added {len(test_units)} knowledge units to memory")

        return memory_manager, test_units
    except Exception as e:
        logger.error(f"Error setting up memory: {str(e)}")
        logger.error(traceback.format_exc())
        raise


async def demonstrate_enhanced_retrieval():
    """Demonstrate the enhanced retrieval capabilities."""

    try:
        # Set up memory
        memory_manager, test_units = await setup_memory()

        # Create and configure the hybrid retriever
        hybrid_retriever = LangchainHybridRetriever(
            vector_store=memory_manager.vector_store, bm25_store=memory_manager.bm25_store
        )

        # Create context analyzer
        context_analyzer = ContextAnalyzer(
            enable_entity_extraction=True,
            enable_content_classification=True,
            enable_relationship_detection=True,
        )

        # Create enhanced prompt builder
        prompt_builder = EnhancedPromptBuilder(
            context_organization_strategy="hybrid", enable_metadata_enrichment=True
        )

        # Create context retriever with all components
        context_retriever = ContextRetriever(
            memory_manager=memory_manager,
            max_context_items=5,
            relevance_threshold=0.5,
            max_token_limit=4000,
            hybrid_retriever=hybrid_retriever,
            vector_weight=0.6,
            bm25_weight=0.4,
            use_enhanced_prompts=True,
            enable_context_analysis=True,
            context_organization_strategy="hybrid",
        )

        logger.info("=== Testing Different Retrieval Strategies ===")

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain the bias-variance tradeoff",
            "Show me some Python code for training a model",
        ]

        # Test different retrieval strategies
        strategies = ["semantic", "keyword", "hybrid", "advanced_hybrid"]

        for query in test_queries:
            logger.info(f"\nQuery: {query}")

            for strategy in strategies:
                try:
                    logger.info(f"Strategy: {strategy}")
                    context = await context_retriever.retrieve_context(query, strategy=strategy)
                    logger.info(f"Retrieved {len(context)} context items")

                    # Show top result
                    if context:
                        top_item = context[0]
                        logger.info(f"Top item: {top_item['content'][:100]}...")
                        logger.info(f"Relevance: {top_item.get('relevance', 0):.4f}")

                        # Show metadata if available
                        if "metadata" in top_item and top_item["metadata"]:
                            enriched_keys = [
                                k
                                for k in top_item["metadata"].keys()
                                if k not in test_units[0].metadata
                            ]
                            if enriched_keys:
                                logger.info(f"Enriched metadata: {', '.join(enriched_keys)}")
                except Exception as e:
                    logger.error(f"Error testing {strategy} strategy: {str(e)}")
                    logger.error(traceback.format_exc())

        logger.info("\n=== Testing Prompt Building ===")

        # Test prompt building
        query = "Explain how machine learning models are trained and evaluated"
        try:
            context = await context_retriever.retrieve_context(query, strategy="hybrid")

            # Build prompt in different formats
            qa_prompt = context_retriever.build_prompt(
                query, context, task_type="question_answering"
            )
            logger.info(f"Built QA prompt with {qa_prompt['context_items_used']} context items")

            summary_prompt = context_retriever.build_prompt(
                query, context, task_type="summarization"
            )
            logger.info(
                f"Built summary prompt with {summary_prompt['context_items_used']} context items"
            )

            # Test with conversation history
            conversation_history = [
                {"role": "user", "content": "Tell me about machine learning"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a field where algorithms learn from data.",
                },
            ]

            conv_prompt = context_retriever.build_prompt_with_history(
                "How are models evaluated?", context, conversation_history
            )
            logger.info(
                f"Built conversation prompt with {conv_prompt['context_items_used']} context items and {conv_prompt['history_turns']} history turns"
            )
        except Exception as e:
            logger.error(f"Error testing prompt building: {str(e)}")
            logger.error(traceback.format_exc())

        logger.info("\n=== Testing Context Analysis ===")

        # Test context analysis directly
        try:
            # Create context items from knowledge units
            context_items = [
                {
                    "content": unit.content,
                    "context": unit.context,
                    "source": unit.knowledge_source,
                    "relevance": 0.8,
                    "id": unit.unique_id,
                    "timestamp": unit.timestamp.isoformat(),
                    "metadata": unit.metadata.copy(),
                }
                for unit in test_units[:3]
            ]

            # Analyze context items
            enriched_items = context_analyzer.analyze_context_items(context_items)

            # Show enrichment results
            for i, item in enumerate(enriched_items):
                original_keys = set(context_items[i]["metadata"].keys())
                enriched_keys = set(item["metadata"].keys())
                new_keys = enriched_keys - original_keys

                logger.info(f"Item {i + 1} enrichment:")
                logger.info(f"  - Added {len(new_keys)} new metadata fields: {', '.join(new_keys)}")

                if "content_type" in item["metadata"]:
                    logger.info(f"  - Classified as: {item['metadata']['content_type']}")

                if "entities" in item["metadata"]:
                    entity_count = sum(
                        len(entities) for entities in item["metadata"]["entities"].values()
                    )
                    logger.info(f"  - Extracted {entity_count} entities")

                if "quality_score" in item["metadata"]:
                    logger.info(f"  - Quality score: {item['metadata']['quality_score']:.4f}")

                if "adjusted_relevance" in item:
                    original = context_items[i]["relevance"]
                    adjusted = item["adjusted_relevance"]
                    logger.info(f"  - Relevance adjusted: {original:.4f} → {adjusted:.4f}")
        except Exception as e:
            logger.error(f"Error testing context analysis: {str(e)}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_retrieval())
