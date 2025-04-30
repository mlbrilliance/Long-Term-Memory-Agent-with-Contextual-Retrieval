"""
Advanced hybrid retrieval usage example.

This script demonstrates how to use the hybrid retriever integrated with the LTM Agent
system to combine vector similarity and BM25 lexical search for improved retrieval.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltm_agent.memory.manager import MemoryManager
from ltm_agent.retrieval.context_retriever import ContextRetriever, PromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("advanced_hybrid_example")


async def populate_test_knowledge(memory_manager, num_items=20):
    """Populate the memory store with test knowledge units."""
    logger.info(f"Adding {num_items} test knowledge units to memory...")

    # Create test documents in various domains
    test_documents = [
        # Python programming knowledge
        (
            "Python Basics",
            "Python is a high-level, interpreted programming language known for its readability and simplicity. It supports multiple programming paradigms.",
            "corpus",
        ),
        (
            "Python Data Structures",
            "Python provides built-in data structures like lists, dictionaries, sets, and tuples. Lists are ordered collections, while dictionaries store key-value pairs.",
            "corpus",
        ),
        (
            "Python Functions",
            "Functions in Python are defined using the 'def' keyword. They can accept arguments, return values, and be used as first-class objects.",
            "corpus",
        ),
        # Machine Learning knowledge
        (
            "Machine Learning Basics",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.",
            "corpus",
        ),
        (
            "Supervised Learning",
            "Supervised learning algorithms learn patterns from labeled training data to make predictions on new data. Examples include linear regression and random forests.",
            "corpus",
        ),
        (
            "Neural Networks",
            "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.",
            "corpus",
        ),
        # User feedback
        (
            "Query Understanding",
            "When processing user queries, it's important to handle synonyms and different phrasings that might express the same intent.",
            "feedback",
        ),
        (
            "Context Preservation",
            "Maintaining context across conversation turns is essential for natural interaction. The system should remember previously mentioned entities.",
            "feedback",
        ),
        # Agent actions
        (
            "Database Query",
            "Executed SQL query to retrieve user information from the database: SELECT * FROM users WHERE user_id = 12345",
            "action",
        ),
        (
            "API Request",
            "Made API request to weather service to get current conditions for Seattle: GET /api/weather?location=seattle",
            "action",
        ),
        # Additional knowledge
        (
            "Web Development",
            "Web development involves creating websites and web applications using languages like HTML, CSS, and JavaScript.",
            "corpus",
        ),
        (
            "Data Analysis",
            "Data analysis is the process of inspecting, cleansing, transforming, and modeling data to discover useful information and support decision-making.",
            "corpus",
        ),
        (
            "Natural Language Processing",
            "NLP is a field of AI focused on the interaction between computers and human language. It involves tasks like text classification and sentiment analysis.",
            "corpus",
        ),
        (
            "Software Engineering",
            "Software engineering is the systematic application of engineering approaches to software development. It includes requirements analysis, design, testing, and maintenance.",
            "corpus",
        ),
        (
            "Database Systems",
            "Database systems store and organize data for easy retrieval and manipulation. Common types include relational, NoSQL, and graph databases.",
            "corpus",
        ),
        # More domain-specific knowledge
        (
            "Quantum Computing",
            "Quantum computing utilizes quantum-mechanical phenomena like superposition and entanglement to perform computations. Quantum bits or qubits are the basic units of information.",
            "corpus",
        ),
        (
            "Blockchain Technology",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records secured from tampering and revision.",
            "corpus",
        ),
        (
            "Cybersecurity",
            "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, damage, or unauthorized access.",
            "corpus",
        ),
        (
            "Cloud Computing",
            "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software.",
            "corpus",
        ),
        (
            "Artificial Intelligence Ethics",
            "AI ethics addresses moral questions related to the development and use of AI, including bias, privacy, transparency, and accountability.",
            "corpus",
        ),
    ]

    # Add each document to memory
    for i, (title, content, source) in enumerate(test_documents):
        try:
            await memory_manager.add_knowledge(
                content=content,
                source=source,
                context=title,
                metadata={
                    "domain": title.split()[0].lower(),
                    "importance": "high" if i < 10 else "medium",
                },
            )
            logger.info(f"  - Added knowledge: {title}")
        except Exception as e:
            logger.error(f"  - Failed to add knowledge {title}: {e}")

    logger.info(f"Added {len(test_documents)} knowledge units to memory")


async def compare_retrieval_strategies(context_retriever):
    """Compare different retrieval strategies for various queries."""
    test_queries = [
        "How do Python data structures work?",
        "Explain the basics of neural networks in machine learning",
        "What is the difference between supervised and unsupervised learning?",
        "How can I create a web application?",
        "What are best practices for database security?",
        "Explain natural language processing techniques",
    ]

    strategies = ["semantic", "keyword", "hybrid", "advanced_hybrid"]

    logger.info("\n=== Comparing Retrieval Strategies ===")
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")

        for strategy in strategies:
            try:
                start_time = asyncio.get_event_loop().time()
                results = await context_retriever.retrieve_context(query=query, strategy=strategy)
                elapsed = asyncio.get_event_loop().time() - start_time

                logger.info(
                    f"\n[{strategy.upper()} Strategy] - {len(results)} results in {elapsed:.3f}s"
                )

                # Show top 3 results with relevance scores
                for i, item in enumerate(results[:3]):
                    logger.info(
                        f"  {i + 1}. [{item.get('relevance', 0):.4f}] {item.get('context')}: {item.get('content')[:80]}..."
                    )

            except Exception as e:
                logger.error(f"  Error with {strategy} strategy: {e}")


async def test_hybrid_weights(context_retriever, memory_manager):
    """Test different weight configurations for the hybrid retriever."""
    logger.info("\n=== Testing Different Hybrid Weight Configurations ===")

    # Create a hybrid retriever if one doesn't exist
    if not context_retriever.hybrid_retriever:
        logger.info("Creating hybrid retriever...")
        hybrid_retriever = await memory_manager.create_hybrid_retriever(
            vector_weight=0.5, bm25_weight=0.5
        )
        await context_retriever.set_hybrid_retriever(hybrid_retriever)

    test_query = "How are machine learning algorithms used in natural language processing?"

    weight_configs = [
        {"name": "Vector Only", "vector": 1.0, "bm25": 0.0},
        {"name": "BM25 Only", "vector": 0.0, "bm25": 1.0},
        {"name": "Balanced", "vector": 0.5, "bm25": 0.5},
        {"name": "Vector Heavy", "vector": 0.7, "bm25": 0.3},
        {"name": "BM25 Heavy", "vector": 0.3, "bm25": 0.7},
    ]

    for config in weight_configs:
        logger.info(
            f"\nTesting {config['name']} configuration (V={config['vector']}, B={config['bm25']})"
        )

        # Update weights
        await context_retriever.update_retrieval_weights(
            vector_weight=config["vector"], bm25_weight=config["bm25"]
        )

        # Get results
        results = await context_retriever.retrieve_context(
            query=test_query, strategy="advanced_hybrid"
        )

        # Show top 3 results
        logger.info(f"Results for query: '{test_query}'")
        for i, item in enumerate(results[:3]):
            logger.info(
                f"  {i + 1}. [{item.get('relevance', 0):.4f}] {item.get('context')}: {item.get('content')[:80]}..."
            )


async def demonstrate_llm_prompting(context_retriever, memory_manager):
    """Demonstrate how retrieved context is used in LLM prompting."""
    logger.info("\n=== Demonstrating LLM Prompt Construction ===")

    # Create a prompt builder
    prompt_builder = PromptBuilder(
        system_prompt_template="You are an AI assistant with access to the following relevant information. Use this knowledge to answer the user's question accurately and thoroughly.",
        max_prompt_tokens=8000,
    )

    test_query = "How can machine learning be applied to natural language processing?"

    # Configure the hybrid retriever for optimal results
    if context_retriever.hybrid_retriever:
        await context_retriever.update_retrieval_weights(0.6, 0.4)

    # Retrieve context with advanced hybrid
    context_items = await context_retriever.retrieve_context(
        query=test_query,
        strategy="advanced_hybrid" if context_retriever.hybrid_retriever else "hybrid",
    )

    # Build the prompt
    prompt = prompt_builder.build_prompt(
        query=test_query, context_items=context_items, include_metadata=True
    )

    # Display the prompt
    logger.info(f"Generated prompt for query: '{test_query}'")
    logger.info(f"System prompt: {prompt['messages'][0]['content'][:500]}...")
    logger.info(f"User prompt: {prompt['messages'][1]['content']}")
    logger.info(f"Used {prompt['context_items_used']} context items in the prompt")


async def main():
    """Run the advanced hybrid retrieval example."""
    logger.info("=== Advanced Hybrid Retrieval Example ===\n")

    # Create a temporary database file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    logger.info(f"Using temporary database: {db_path}")

    try:
        # Initialize the memory manager
        logger.info("\n1. Initializing memory manager...")
        from ltm_agent.memory.sqlite_store import SQLiteVectorStore

        memory_store = SQLiteVectorStore(
            database_path=db_path, embedding_dim=384, create_tables=True
        )

        memory_manager = MemoryManager(memory_store=memory_store)
        await memory_manager.initialize()

        # Populate with test knowledge
        logger.info("\n2. Populating memory with test knowledge...")
        await populate_test_knowledge(memory_manager)

        # Create a context retriever
        logger.info("\n3. Creating context retriever...")
        context_retriever = ContextRetriever(
            memory_manager=memory_manager, max_context_items=10, relevance_threshold=0.3
        )

        # Create a hybrid retriever
        logger.info("\n4. Creating hybrid retriever...")
        hybrid_retriever = await memory_manager.create_hybrid_retriever(
            vector_weight=0.6, bm25_weight=0.4, min_score_threshold=0.0
        )

        # Set the hybrid retriever on the context retriever
        await context_retriever.set_hybrid_retriever(hybrid_retriever)

        # Compare retrieval strategies
        logger.info("\n5. Comparing retrieval strategies...")
        await compare_retrieval_strategies(context_retriever)

        # Test different hybrid weights
        logger.info("\n6. Testing different hybrid weights...")
        await test_hybrid_weights(context_retriever, memory_manager)

        # Demonstrate LLM prompting
        logger.info("\n7. Demonstrating LLM prompting...")
        await demonstrate_llm_prompting(context_retriever, memory_manager)

        logger.info("\n=== Example completed successfully ===")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
                logger.info(f"Removed temporary database: {db_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary database: {e}")


if __name__ == "__main__":
    asyncio.run(main())
