"""
Resilient Memory System Example

This script demonstrates the full suite of memory system enhancements
including adaptive embeddings, monitoring, resilience, and background maintenance.
"""

import argparse
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ltm_agent.agent.ltm_agent import LongTermMemoryAgent
from src.ltm_agent.core.models import KnowledgeUnit
from src.ltm_agent.memory.adaptive_embeddings import AdaptiveEmbeddingManager
from src.ltm_agent.memory.enhanced_contextualizer import EnhancedContextualizer
from src.ltm_agent.memory.enhanced_manager import EnhancedMemoryManager
from src.ltm_agent.memory.in_memory_store import InMemoryVectorStore
from src.ltm_agent.memory.knowledge_graph import KnowledgeGraph
from src.ltm_agent.memory.maintenance import get_maintenance_scheduler
from src.ltm_agent.memory.monitoring import FileBasedTelemetryHandler, get_monitor
from src.ltm_agent.memory.persistent_store import ChromaVectorStore
from src.ltm_agent.memory.resilience import FailureType, ResilientOperation, get_resilience_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories for the example."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("backups", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def initialize_resilient_memory_system(persistent=False, use_monitoring=True):
    """
    Initialize the resilient memory system with all components.

    Args:
        persistent: Whether to use persistent storage
        use_monitoring: Whether to enable monitoring

    Returns:
        Tuple of (memory_manager, embedding_manager, agent)
    """
    # Initialize monitoring if enabled
    if use_monitoring:
        monitor = get_monitor(
            {
                "enable_telemetry": True,
                "telemetry_interval_seconds": 30,
                "memory_alert_threshold_mb": 1000,
                "cpu_alert_threshold_percent": 90,
            }
        )

        # Add file-based telemetry handler
        monitor.add_telemetry_handler(
            FileBasedTelemetryHandler(log_dir="logs", max_files=10, max_file_size_mb=5)
        )

        logger.info("Initialized monitoring system")

    # Initialize resilience manager
    resilience_manager = get_resilience_manager(
        {
            "backup_dir": "backups",
            "backup_interval_hours": 1,
            "max_backups": 5,
            "circuit_breaker_threshold": 3,
            "circuit_breaker_cooling_period": 60,
        }
    )

    logger.info("Initialized resilience manager")

    # Initialize adaptive embedding manager
    embedding_manager = AdaptiveEmbeddingManager(
        {"load_domain_models": True, "use_openai": False, "cache_size": 1000}
    )

    logger.info("Initialized adaptive embedding manager")

    # Initialize vector store (persistent or in-memory)
    if persistent:
        vector_store = ChromaVectorStore(
            collection_name="resilient_memory_example", persist_directory="data/chroma"
        )
        logger.info("Using persistent ChromaVectorStore")
    else:
        vector_store = InMemoryVectorStore()
        logger.info("Using InMemoryVectorStore")

    # Initialize knowledge graph
    knowledge_graph = KnowledgeGraph()

    # Initialize contextualizer
    contextualizer = EnhancedContextualizer(
        # Use custom embedding function from our adaptive manager
        embedding_fn=lambda text: embedding_manager.embed(text)
    )

    # Initialize memory manager
    memory_manager = EnhancedMemoryManager(
        memory_store=vector_store, contextualizer=contextualizer, knowledge_graph=knowledge_graph
    )

    logger.info("Initialized memory manager with all components")

    # Initialize maintenance scheduler
    maintenance_scheduler = get_maintenance_scheduler(
        {
            "enable_default_tasks": True,
            "consolidation_interval": 600,  # 10 minutes for demo
            "pruning_interval": 1800,  # 30 minutes for demo
            "backup_interval": 900,  # 15 minutes for demo
            "backup_dir": "backups",
        }
    )

    # Set context for maintenance tasks
    maintenance_scheduler.set_context(memory_manager=memory_manager, memory_store=vector_store)

    # Start maintenance scheduler
    maintenance_scheduler.start()

    logger.info("Started background maintenance scheduler")

    # Initialize LTM agent that uses our memory manager
    agent = LongTermMemoryAgent(memory_manager=memory_manager)

    logger.info("Initialized LongTermMemoryAgent")

    return memory_manager, embedding_manager, agent


def generate_sample_knowledge(domains=None):
    """
    Generate sample knowledge units for testing.

    Args:
        domains: Optional list of domains to generate knowledge for

    Returns:
        List of knowledge units
    """
    if domains is None:
        domains = ["general", "code", "science", "medical", "legal", "financial"]

    knowledge_units = []

    # General knowledge
    general_texts = [
        "Paris is the capital city of France and is known for the Eiffel Tower.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
        "A marathon is 26.2 miles or 42.195 kilometers in length.",
        "Coffee is one of the most popular beverages worldwide and contains caffeine.",
        "The Great Wall of China is over 13,000 miles long and was built over multiple dynasties.",
    ]

    # Code knowledge
    code_texts = [
        "Python is a high-level, interpreted programming language known for its readability and versatility.",
        """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
        """,
        "React is a JavaScript library for building user interfaces, particularly single-page applications.",
        "A memory leak occurs when a program incorrectly manages memory allocations by failing to release memory that is no longer needed.",
        "Docker containers are lightweight, standalone executable packages that include everything needed to run an application.",
    ]

    # Science knowledge
    science_texts = [
        "The theory of relativity, developed by Albert Einstein, describes the relationship between space and time.",
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for development and functioning of all living organisms.",
        "The periodic table organizes chemical elements according to their atomic number, electron configuration, and chemical properties.",
        "Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy.",
        "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    ]

    # Medical knowledge
    medical_texts = [
        "Antibiotics are medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "The cardiovascular system consists of the heart, blood vessels, and blood, which are responsible for transporting oxygen, nutrients, and hormones throughout the body.",
        "Type 2 diabetes is a metabolic disorder characterized by high blood sugar levels resulting from insulin resistance and relative insulin deficiency.",
        "Alzheimer's disease is a progressive neurological disorder that causes brain cells to degenerate and die, leading to memory loss and cognitive decline.",
        "Vaccines work by stimulating the immune system to recognize and fight specific infectious agents without causing the disease itself.",
    ]

    # Legal knowledge
    legal_texts = [
        "Habeas corpus is a legal principle that allows individuals to challenge their detention or imprisonment before a court.",
        "Tort law deals with civil wrongs that cause someone to suffer loss or harm, resulting in legal liability for the person who committed the act.",
        "The doctrine of precedent, or stare decisis, is a legal principle where courts are bound by previous decisions in similar cases.",
        "A contract is a legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
        "Intellectual property rights protect creations of the mind such as inventions, literary and artistic works, designs, and symbols used in commerce.",
    ]

    # Financial knowledge
    financial_texts = [
        "Compound interest is the interest calculated on both the initial principal and the accumulated interest from previous periods.",
        "A diversified investment portfolio spreads investments across various asset classes to reduce risk exposure.",
        "The price-to-earnings (P/E) ratio is a valuation metric that compares a company's current share price to its per-share earnings.",
        "Inflation is the rate at which the general level of prices for goods and services rises, causing purchasing power to fall.",
        "A hedge fund is an investment fund that pools capital from accredited individuals or institutional investors and invests in a variety of assets.",
    ]

    # Map domains to text lists
    domain_texts = {
        "general": general_texts,
        "code": code_texts,
        "science": science_texts,
        "medical": medical_texts,
        "legal": legal_texts,
        "financial": financial_texts,
    }

    # Create knowledge units for selected domains
    for domain in domains:
        if domain in domain_texts:
            for text in domain_texts[domain]:
                knowledge_units.append(
                    KnowledgeUnit(
                        id=str(uuid.uuid4()),
                        text=text,
                        metadata={"domain": domain, "confidence": random.uniform(0.7, 1.0)},
                        created_at=datetime.now(),
                    )
                )

    return knowledge_units


def demonstrate_resilience(memory_manager, embedding_manager):
    """
    Demonstrate resilience features by simulating failures and recovery.

    Args:
        memory_manager: Memory manager instance
        embedding_manager: Embedding manager instance
    """
    resilience_manager = get_resilience_manager()
    monitor = get_monitor()

    logger.info("Demonstrating resilience features...")

    # 1. Create a checkpoint before simulating failures
    logger.info("Creating checkpoint of memory state...")
    data_to_checkpoint = {
        "knowledge_units": memory_manager.get_all_knowledge_units(),
        "metadata": {"timestamp": datetime.now().isoformat()},
    }
    checkpoint_file = resilience_manager.create_checkpoint(data_to_checkpoint, "memory_manager")
    logger.info(f"Created checkpoint: {checkpoint_file}")

    # 2. Simulate embedding service failure
    logger.info("Simulating embedding service failure...")

    try:
        # Trigger a controlled failure
        with ResilientOperation(
            operation_name="demo_embedding_operation",
            component_name="embedding_service",
            failure_mapping={ValueError: FailureType.EMBEDDING_SERVICE_ERROR},
        ):
            # Simulate failure
            raise ValueError("Simulated embedding service failure")

    except Exception as e:
        logger.info(f"Caught expected exception: {e}")

    # 3. Demonstrate resilient query operation
    logger.info("Demonstrating resilient query operation...")

    try:
        with ResilientOperation(
            operation_name="resilient_query",
            component_name="memory_manager",
            context={
                "text": "What is DNA?",
                "retry_operation": lambda: memory_manager.search("What is DNA?"),
            },
        ) as op:
            # This should use the resilient operation context
            results = memory_manager.search("What is DNA?")
            logger.info(f"Successfully retrieved {len(results)} results")

    except Exception as e:
        logger.error(f"Unexpected error in resilient query: {e}")

    # 4. Demonstrate circuit breaker pattern
    logger.info("Demonstrating circuit breaker pattern...")

    # Initialize circuit breaker for a service
    service_name = "test_external_service"

    # Simulate multiple failures to trip the circuit breaker
    for i in range(5):
        success = False
        resilience_manager.update_circuit_breaker(service_name, success)

    # Check if circuit breaker is tripped
    available = resilience_manager.check_circuit_breaker(service_name)
    logger.info(f"Service '{service_name}' available: {available}")

    # 5. Demonstrate restoration from checkpoint
    logger.info("Demonstrating restoration from checkpoint...")

    # Corrupt some data
    all_units = memory_manager.get_all_knowledge_units()
    if all_units:
        removed_unit = all_units[0]
        memory_manager.remove_knowledge_unit(removed_unit.id)
        logger.info(f"Removed knowledge unit: {removed_unit.id}")

    # Restore from checkpoint
    restored_data = resilience_manager.restore_from_checkpoint("memory_manager")
    if restored_data:
        # Re-add knowledge units
        for unit in restored_data["knowledge_units"]:
            if not memory_manager.get_knowledge_unit(unit.id):
                memory_manager.add_knowledge_unit(unit)

        logger.info(
            f"Restored {len(restored_data['knowledge_units'])} knowledge units from checkpoint"
        )

    logger.info("Resilience demonstration completed")


def demonstrate_adaptive_embeddings(embedding_manager):
    """
    Demonstrate adaptive embedding selection based on content.

    Args:
        embedding_manager: Adaptive embedding manager instance
    """
    logger.info("Demonstrating adaptive embeddings...")

    # Test different types of content
    test_texts = {
        "general": "The weather in London is often rainy and cloudy throughout the year.",
        "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """,
        "science": "The mitochondria is the powerhouse of the cell, responsible for cellular respiration and ATP production.",
        "medical": "Patients with hypertension often require medications such as ACE inhibitors or calcium channel blockers.",
        "legal": "The plaintiff argued that the defendant was in breach of contract according to Section 2.3 of the agreement.",
        "financial": "The company reported Q3 earnings of $2.5 million, representing a 15% increase year-over-year.",
    }

    # Process each text type
    for domain, text in test_texts.items():
        # Detect domain
        detected_domain, confidence = embedding_manager.domain_detector.detect_domain(text)

        # Generate embedding
        start_time = time.time()
        embedding = embedding_manager.embed(text)
        elapsed = time.time() - start_time

        logger.info(f"Text type: {domain}")
        logger.info(f"Detected domain: {detected_domain} (confidence: {confidence:.4f})")
        logger.info(f"Embedding dimensions: {len(embedding)}")
        logger.info(f"Embedding time: {elapsed:.4f} seconds")
        logger.info(f"Selected model: {embedding_manager._select_model(text).name}")
        logger.info("---")

    # Show statistics
    stats = embedding_manager.get_stats()
    logger.info(f"Total embedding calls: {stats['call_count']}")
    logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
    logger.info(f"Available models: {embedding_manager.get_available_models()}")


def run_interactive_demo(agent, memory_manager, embedding_manager):
    """
    Run an interactive demo with the LTM agent.

    Args:
        agent: LongTermMemoryAgent instance
        memory_manager: Memory manager instance
        embedding_manager: Embedding manager instance
    """
    logger.info("Starting interactive demo...")

    print("\n" + "=" * 50)
    print("Resilient Memory System Interactive Demo")
    print("=" * 50)
    print("Type 'exit' to quit, 'help' for commands")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("\nEnter query or command: ")

            if user_input.lower() == "exit":
                break

            elif user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  query <text>   - Query the agent")
                print("  add <text>     - Add new knowledge")
                print("  stats          - Show system statistics")
                print("  resilience     - Run resilience demo")
                print("  embeddings     - Run embeddings demo")
                print("  maintenance    - Run maintenance tasks")
                print("  exit           - Exit the demo")

            elif user_input.lower().startswith("query "):
                query = user_input[6:].strip()
                if query:
                    # Use agent to answer
                    response = agent.query(query)
                    print(f"\nQuery: {query}")
                    print(f"Response: {response}")

                    # Show related knowledge
                    relevant_units = memory_manager.search(query, top_k=3)
                    if relevant_units:
                        print("\nRelevant knowledge:")
                        for i, unit in enumerate(relevant_units, 1):
                            print(f"{i}. {unit.text[:100]}...")

            elif user_input.lower().startswith("add "):
                knowledge = user_input[4:].strip()
                if knowledge:
                    # Detect domain
                    domain, _ = embedding_manager.domain_detector.detect_domain(knowledge)

                    # Create knowledge unit
                    unit = KnowledgeUnit(
                        id=str(uuid.uuid4()),
                        text=knowledge,
                        metadata={"domain": domain, "source": "user", "confidence": 1.0},
                        created_at=datetime.now(),
                    )

                    # Add to memory
                    memory_manager.add_knowledge_unit(unit)
                    print(f"\nAdded new knowledge (domain: {domain})")

                    # Check for related knowledge
                    related = memory_manager.find_related_knowledge(unit.text, top_k=2)
                    if related:
                        print("\nRelated existing knowledge:")
                        for i, rel_unit in enumerate(related, 1):
                            print(f"{i}. {rel_unit.text[:100]}...")

            elif user_input.lower() == "stats":
                monitor = get_monitor()

                print("\nSystem Statistics:")
                print(f"Total knowledge units: {len(memory_manager.get_all_knowledge_units())}")

                # Memory manager stats
                if hasattr(memory_manager, "knowledge_graph"):
                    print(f"Knowledge graph nodes: {len(memory_manager.knowledge_graph.nodes)}")
                    print(
                        f"Knowledge graph relationships: {len(memory_manager.knowledge_graph.get_all_relationships())}"
                    )

                # Embedding stats
                embedding_stats = embedding_manager.get_stats()
                print(f"Embedding calls: {embedding_stats['call_count']}")
                print(f"Embedding cache hit rate: {embedding_stats['cache_hit_rate']:.2f}")

                # Maintenance stats
                maintenance_scheduler = get_maintenance_scheduler()
                task_stats = maintenance_scheduler.get_task_stats()
                print("\nMaintenance Tasks:")
                for task_name, stats in task_stats.items():
                    print(
                        f"  {task_name}: {stats['runs']} runs, success rate: {stats['success_rate']:.1f}%"
                    )

                # Monitoring metrics
                metrics = monitor.get_metrics_summary()
                print("\nPerformance Metrics:")
                for name, metric in metrics.items():
                    if "time" in name and metric["count"] > 0:
                        print(f"  {name}: {metric['average']:.4f}s avg")

            elif user_input.lower() == "resilience":
                demonstrate_resilience(memory_manager, embedding_manager)
                print("\nResilience demonstration completed")

            elif user_input.lower() == "embeddings":
                demonstrate_adaptive_embeddings(embedding_manager)
                print("\nAdaptive embeddings demonstration completed")

            elif user_input.lower() == "maintenance":
                maintenance_scheduler = get_maintenance_scheduler()

                print("\nRunning maintenance tasks...")

                # Run consolidation task
                result = maintenance_scheduler.run_task_now("memory_consolidation")
                print(f"Consolidation: {result}")

                # Run integrity check task
                result = maintenance_scheduler.run_task_now("memory_integrity_check")
                print(f"Integrity check: {result['status']}, {result['issues_count']} issues found")

                # Run backup task
                result = maintenance_scheduler.run_task_now("memory_backup")
                print(
                    f"Backup: {result['status']}, {result['count']} units backed up to {result['file']}"
                )

                print("\nMaintenance tasks completed")

            else:
                print("Unknown command. Type 'help' for available commands.")

        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Resilient Memory System Example")
    parser.add_argument("--persistent", action="store_true", help="Use persistent storage")
    parser.add_argument(
        "--samples", type=int, default=15, help="Number of sample knowledge units to generate"
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    try:
        # Setup directories
        setup_directories()

        # Initialize memory system
        memory_manager, embedding_manager, agent = initialize_resilient_memory_system(
            persistent=args.persistent, use_monitoring=True
        )

        # Generate and add sample knowledge
        if args.samples > 0:
            samples = generate_sample_knowledge()
            samples = samples[: min(len(samples), args.samples)]

            logger.info(f"Adding {len(samples)} sample knowledge units...")
            for unit in samples:
                memory_manager.add_knowledge_unit(unit)

            logger.info(f"Added {len(samples)} sample knowledge units")

        # Demonstrate resilience features
        demonstrate_resilience(memory_manager, embedding_manager)

        # Demonstrate adaptive embeddings
        demonstrate_adaptive_embeddings(embedding_manager)

        # Run interactive demo if requested
        if args.interactive:
            run_interactive_demo(agent, memory_manager, embedding_manager)
        else:
            # Run a few sample queries
            sample_queries = ["What is Python?", "Tell me about DNA", "How does a hedge fund work?"]

            logger.info("Running sample queries...")
            for query in sample_queries:
                with ResilientOperation(operation_name="sample_query", component_name="agent"):
                    response = agent.query(query)
                    logger.info(f"Query: {query}")
                    logger.info(f"Response: {response}")
                    logger.info("---")

        # Show final stats
        monitor = get_monitor()
        maintenance_scheduler = get_maintenance_scheduler()

        logger.info("Final system statistics:")
        logger.info(f"Total knowledge units: {len(memory_manager.get_all_knowledge_units())}")
        logger.info(f"Embedding calls: {embedding_manager.get_stats()['call_count']}")
        logger.info(
            f"Maintenance tasks executed: {sum(task['runs'] for task in maintenance_scheduler.get_task_stats().values())}"
        )

        # Stop maintenance scheduler
        maintenance_scheduler.stop()
        logger.info("Maintenance scheduler stopped")

        logger.info("Example completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
