"""
Evaluation script for comparing retrieval approaches.

This script evaluates the performance of different retrieval methods:
1. Vector-based retrieval
2. BM25-based retrieval
3. Hybrid retrieval with various weight configurations

It reports metrics such as precision, recall, and mean reciprocal rank (MRR)
to provide a comprehensive comparison of the different approaches.
"""

import asyncio
import csv
import json
import logging
import os
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltm_agent.core.models import KnowledgeUnit
from ltm_agent.memory.bm25_store import RankBM25Store
from ltm_agent.memory.hybrid_retriever import LangchainHybridRetriever
from ltm_agent.memory.sqlite_store import SQLiteVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("retrieval_evaluation")

# Sample corpus for testing - this could be loaded from a file in practice
SAMPLE_CORPUS = [
    # Python-related documents
    (
        "Python Basics",
        "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    ),
    (
        "Python Libraries",
        "Popular Python libraries include NumPy for numerical computing, Pandas for data analysis, Matplotlib for data visualization, and TensorFlow and PyTorch for machine learning.",
    ),
    (
        "Python Web Frameworks",
        "Django and Flask are popular Python web frameworks. Django is a high-level framework that follows the model-view-controller (MVC) architectural pattern, while Flask is a lightweight, extensible microframework.",
    ),
    # Machine Learning related documents
    (
        "Machine Learning Basics",
        "Machine learning is a field of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on algorithms that can learn from and make predictions on data.",
    ),
    (
        "Supervised Learning",
        "Supervised learning is a machine learning paradigm where models are trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs provided during training.",
    ),
    (
        "Unsupervised Learning",
        "Unsupervised learning algorithms find patterns or structure in unlabeled data. Common techniques include clustering, dimensionality reduction, and anomaly detection.",
    ),
    (
        "Neural Networks",
        "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) that process and transform input data to make predictions or classifications.",
    ),
    # Data Science related documents
    (
        "Data Science Workflow",
        "A typical data science workflow includes data collection, cleaning, exploration, feature engineering, modeling, evaluation, and deployment. The process is iterative and may require revisiting earlier steps.",
    ),
    (
        "Data Cleaning",
        "Data cleaning involves handling missing values, outliers, and inconsistencies in datasets. It's a crucial step in data preprocessing that ensures high-quality input for analysis.",
    ),
    (
        "Feature Engineering",
        "Feature engineering involves transforming raw data into features that better represent the underlying problem to predictive models, resulting in improved model accuracy.",
    ),
    # Web Development related documents
    (
        "HTML Basics",
        "HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser. It defines the structure and content of web pages.",
    ),
    (
        "CSS Styling",
        "CSS (Cascading Style Sheets) is a style sheet language used for describing the presentation of a document written in HTML. It controls layout, colors, fonts, and other visual aspects.",
    ),
    (
        "JavaScript Fundamentals",
        "JavaScript is a scripting language that enables interactive web pages. It runs on the client-side in the browser and can also run server-side with Node.js.",
    ),
    # Database related documents
    (
        "SQL Basics",
        "SQL (Structured Query Language) is a domain-specific language used for managing and manipulating relational databases. Common operations include SELECT, INSERT, UPDATE, and DELETE.",
    ),
    (
        "NoSQL Databases",
        "NoSQL databases provide a mechanism for storage and retrieval of data that is modeled in means other than the tabular relations used in relational databases. Types include document, key-value, wide-column, and graph databases.",
    ),
    (
        "Database Indexing",
        "Database indexing improves the speed of data retrieval operations on a database at the cost of additional storage space and slower writes. Common index types include B-tree, hash, and bitmap indexes.",
    ),
    # Artificial Intelligence related documents
    (
        "Natural Language Processing",
        "Natural Language Processing (NLP) is a field of AI focused on the interaction between computers and human language. It involves tasks such as text classification, sentiment analysis, and machine translation.",
    ),
    (
        "Computer Vision",
        "Computer Vision is a field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs. Applications include image recognition, object detection, and scene understanding.",
    ),
    (
        "Reinforcement Learning",
        "Reinforcement Learning is an area of machine learning where agents learn to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It's used in robotics, game playing, and autonomous systems.",
    ),
    # Software Engineering related documents
    (
        "Version Control",
        "Version control systems track changes to files over time, enabling multiple developers to collaborate on a project. Git is a widely used distributed version control system.",
    ),
    (
        "Agile Development",
        "Agile development is an iterative approach to software development that emphasizes flexibility, customer collaboration, and rapid delivery of working software. Popular frameworks include Scrum and Kanban.",
    ),
    (
        "Software Testing",
        "Software testing is the process of evaluating a software system to identify defects and ensure that it meets requirements. Types include unit testing, integration testing, system testing, and acceptance testing.",
    ),
    # Additional mixed topics
    (
        "Cloud Computing",
        "Cloud computing provides on-demand computing resources over the internet, including servers, storage, databases, networking, and software. Major providers include AWS, Azure, and Google Cloud.",
    ),
    (
        "Big Data",
        "Big data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations. Technologies for processing big data include Hadoop, Spark, and Flink.",
    ),
    (
        "Cybersecurity",
        "Cybersecurity involves protecting systems, networks, and programs from digital attacks. Common security measures include encryption, authentication, authorization, and regular security audits.",
    ),
    (
        "DevOps Practices",
        "DevOps combines software development (Dev) and IT operations (Ops) to shorten the development lifecycle and provide continuous delivery of high-quality software. Key practices include CI/CD, infrastructure as code, and monitoring.",
    ),
    (
        "Blockchain Technology",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked using cryptography. It's the underlying technology for cryptocurrencies like Bitcoin.",
    ),
    (
        "Mobile App Development",
        "Mobile app development involves creating software applications that run on mobile devices. Popular frameworks include React Native, Flutter, and native development using Swift (iOS) or Kotlin (Android).",
    ),
]

# Evaluation queries with relevant document IDs (ground truth)
EVALUATION_QUERIES = [
    {
        "query": "What are the basics of Python programming?",
        "relevant_ids": {0, 1, 2},  # Python-related documents
    },
    {
        "query": "How do neural networks work in machine learning?",
        "relevant_ids": {3, 4, 5, 6},  # Machine Learning related documents
    },
    {
        "query": "What is the process of cleaning and preparing data?",
        "relevant_ids": {7, 8, 9},  # Data Science related documents
    },
    {
        "query": "How do HTML, CSS, and JavaScript work together?",
        "relevant_ids": {10, 11, 12},  # Web Development related documents
    },
    {
        "query": "What are different types of databases and indexing?",
        "relevant_ids": {13, 14, 15},  # Database related documents
    },
    {
        "query": "Explain natural language processing and computer vision",
        "relevant_ids": {16, 17, 18},  # AI related documents
    },
    {
        "query": "What are best practices in software development and testing?",
        "relevant_ids": {19, 20, 21},  # Software Engineering related documents
    },
    {
        "query": "How is cloud computing used for big data?",
        "relevant_ids": {22, 23},  # Cloud and Big Data
    },
    {"query": "What are the fundamentals of cybersecurity?", "relevant_ids": {24}},  # Cybersecurity
    {
        "query": "Compare DevOps practices and agile development",
        "relevant_ids": {20, 25},  # DevOps and Agile
    },
]


def create_knowledge_unit(index: int, title: str, content: str) -> KnowledgeUnit:
    """Create a knowledge unit for testing."""
    return KnowledgeUnit(
        unique_id=f"doc-{index}",
        original_chunk=content,
        contextual_text=title,
        knowledge_source="corpus",
        metadata={"domain": title.split()[0].lower(), "created_at": datetime.now().isoformat()},
    )


def calculate_precision_recall(
    results: list[tuple[KnowledgeUnit, float]], relevant_ids: set[int], k: int = 5
) -> tuple[float, float]:
    """
    Calculate precision and recall at k.

    Args:
        results: List of (knowledge_unit, score) tuples
        relevant_ids: Set of relevant document indices
        k: Number of top results to consider

    Returns:
        Tuple of (precision, recall)
    """
    # Get top k results
    top_k_results = results[:k]

    # Extract IDs and convert to integers
    retrieved_ids = set()
    for unit, _ in top_k_results:
        # Extract the numeric part of the ID (e.g., "doc-5" -> 5)
        doc_id = int(unit.unique_id.split("-")[1])
        retrieved_ids.add(doc_id)

    # Calculate intersection
    relevant_retrieved = retrieved_ids.intersection(relevant_ids)

    # Calculate precision and recall
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
    recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0

    return precision, recall


def calculate_mrr(results: list[tuple[KnowledgeUnit, float]], relevant_ids: set[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        results: List of (knowledge_unit, score) tuples
        relevant_ids: Set of relevant document indices

    Returns:
        MRR score
    """
    for i, (unit, _) in enumerate(results):
        doc_id = int(unit.unique_id.split("-")[1])
        if doc_id in relevant_ids:
            # Found a relevant document at rank i+1
            return 1.0 / (i + 1)

    # No relevant documents found
    return 0.0


async def run_evaluation():
    """Run the comprehensive evaluation of retrieval methods."""
    logger.info("=== Retrieval Methods Evaluation ===")

    # Create temporary files for the evaluation
    with (
        tempfile.NamedTemporaryFile(suffix=".db", delete=False) as vector_db_file,
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as bm25_file,
    ):
        vector_db_path = vector_db_file.name
        bm25_path = bm25_file.name

    logger.info(f"Vector DB path: {vector_db_path}")
    logger.info(f"BM25 index path: {bm25_path}")

    try:
        # Step 1: Prepare corpus
        logger.info("\nStep 1: Preparing test corpus...")
        knowledge_units = {}
        for i, (title, content) in enumerate(SAMPLE_CORPUS):
            unit = create_knowledge_unit(i, title, content)
            knowledge_units[unit.unique_id] = unit
            logger.info(f"  - Created unit {i}: {title}")

        # Step 2: Initialize vector store
        logger.info("\nStep 2: Initializing vector store...")
        vector_store = SQLiteVectorStore(
            database_path=vector_db_path, embedding_dim=384, create_tables=True
        )
        await vector_store._ensure_initialized()

        # Step 3: Add documents to vector store
        logger.info("\nStep 3: Adding documents to vector store...")
        for unit_id, unit in knowledge_units.items():
            await vector_store.add(unit)
            logger.info(f"  - Added {unit.contextual_text} to vector store")

        # Step 4: Initialize BM25 store
        logger.info("\nStep 4: Initializing BM25 store...")
        bm25_store = RankBM25Store(index_path=bm25_path)

        # Step 5: Update BM25 index
        logger.info("\nStep 5: Updating BM25 index...")
        await bm25_store.update_index(knowledge_units)
        logger.info(f"  - Updated BM25 index with {len(knowledge_units)} documents")

        # Step 6: Initialize hybrid retriever
        logger.info("\nStep 6: Initializing hybrid retriever...")
        hybrid_retriever = LangchainHybridRetriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            vector_weight=0.5,
            bm25_weight=0.5,
            min_score_threshold=0.0,
        )

        # Step 7: Prepare for evaluation
        logger.info("\nStep 7: Preparing for evaluation...")
        # Weight configurations to test
        weight_configs = [
            {"name": "Vector Only", "vector": 1.0, "bm25": 0.0},
            {"name": "BM25 Only", "vector": 0.0, "bm25": 1.0},
            {"name": "Balanced", "vector": 0.5, "bm25": 0.5},
            {"name": "Vector Heavy", "vector": 0.7, "bm25": 0.3},
            {"name": "BM25 Heavy", "vector": 0.3, "bm25": 0.7},
        ]

        # Metrics to calculate
        k_values = [3, 5, 10]

        # Prepare results storage
        results = []

        # Step 8: Run evaluation
        logger.info("\nStep 8: Running evaluation...")

        for weight_config in weight_configs:
            logger.info(f"\nEvaluating {weight_config['name']} configuration...")

            # Set weights
            await hybrid_retriever.set_weights(
                vector_weight=weight_config["vector"], bm25_weight=weight_config["bm25"]
            )

            config_results = {
                "configuration": weight_config["name"],
                "vector_weight": weight_config["vector"],
                "bm25_weight": weight_config["bm25"],
                "queries": [],
            }

            # Process each query
            for query_info in EVALUATION_QUERIES:
                query = query_info["query"]
                relevant_ids = query_info["relevant_ids"]

                logger.info(f"  Query: '{query}'")

                # Perform search
                search_results = await hybrid_retriever.search(query, limit=10)

                # Calculate metrics
                query_metrics = {
                    "query": query,
                    "mrr": calculate_mrr(search_results, relevant_ids),
                    "precision_recall": {},
                }

                for k in k_values:
                    precision, recall = calculate_precision_recall(search_results, relevant_ids, k)
                    query_metrics["precision_recall"][k] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        ),
                    }

                config_results["queries"].append(query_metrics)

            # Calculate average metrics across queries
            avg_mrr = statistics.mean([q["mrr"] for q in config_results["queries"]])
            avg_metrics_by_k = {}

            for k in k_values:
                avg_precision = statistics.mean(
                    [q["precision_recall"][k]["precision"] for q in config_results["queries"]]
                )
                avg_recall = statistics.mean(
                    [q["precision_recall"][k]["recall"] for q in config_results["queries"]]
                )
                avg_f1 = statistics.mean(
                    [q["precision_recall"][k]["f1"] for q in config_results["queries"]]
                )

                avg_metrics_by_k[k] = {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                }

            config_results["average_mrr"] = avg_mrr
            config_results["average_metrics_by_k"] = avg_metrics_by_k

            results.append(config_results)

            # Log summary for this configuration
            logger.info(f"  Average MRR: {avg_mrr:.4f}")
            for k in k_values:
                metrics = avg_metrics_by_k[k]
                logger.info(
                    f"  Metrics @{k}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
                )

        # Step 9: Save results to files
        logger.info("\nStep 9: Saving evaluation results...")

        # Save detailed results as JSON
        results_file = "retrieval_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  - Detailed results saved to {results_file}")

        # Save summary to CSV for easy import into spreadsheets
        summary_file = "retrieval_evaluation_summary.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = ["Configuration", "Vector Weight", "BM25 Weight", "MRR"]
            for k in k_values:
                header.extend([f"P@{k}", f"R@{k}", f"F1@{k}"])
            writer.writerow(header)

            # Write data for each configuration
            for config_result in results:
                row = [
                    config_result["configuration"],
                    config_result["vector_weight"],
                    config_result["bm25_weight"],
                    config_result["average_mrr"],
                ]

                for k in k_values:
                    metrics = config_result["average_metrics_by_k"][k]
                    row.extend([metrics["precision"], metrics["recall"], metrics["f1"]])

                writer.writerow(row)

        logger.info(f"  - Summary results saved to {summary_file}")

        # Step 10: Print final comparison
        logger.info("\nStep 10: Final comparison of configurations")

        # Sort by average MRR (descending)
        sorted_results = sorted(results, key=lambda x: x["average_mrr"], reverse=True)

        logger.info("\nRanking by MRR:")
        for i, config_result in enumerate(sorted_results, 1):
            logger.info(
                f"{i}. {config_result['configuration']} (V={config_result['vector_weight']}, B={config_result['bm25_weight']}): MRR={config_result['average_mrr']:.4f}"
            )

        # Sort by F1@5 (descending)
        sorted_results = sorted(
            results, key=lambda x: x["average_metrics_by_k"][5]["f1"], reverse=True
        )

        logger.info("\nRanking by F1@5:")
        for i, config_result in enumerate(sorted_results, 1):
            metrics = config_result["average_metrics_by_k"][5]
            logger.info(
                f"{i}. {config_result['configuration']} (V={config_result['vector_weight']}, B={config_result['bm25_weight']}): F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
            )

        logger.info("\n=== Evaluation completed successfully ===")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        for path in [vector_db_path, bm25_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
