"""
Master test runner for the LTM Agent project.

This script runs all custom test scripts and provides a comprehensive summary.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_results.log"), logging.StreamHandler()],
)
logger = logging.getLogger("test_runner")

# Add the project root and src directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def log_separator(title: str = None):
    """Print a separator line with optional title."""
    if title:
        logger.info(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        logger.info(f"\n{'=' * 50}")


async def run_tests():
    """Run all test scripts and report results."""
    start_time = time.time()

    log_separator("LTM AGENT TEST SUITE")
    logger.info(f"Starting test run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Import and run test modules
    test_results = {}

    # Test core models
    log_separator("RUNNING CORE MODELS TESTS")
    try:
        logger.info("Importing core models...")
        from ltm_agent.core.models import KnowledgeUnit

        # Basic KnowledgeUnit tests
        ku = KnowledgeUnit(original_chunk="Test chunk", knowledge_source="action")
        assert ku.unique_id is not None, "Unique ID should be generated"
        assert ku.timestamp is not None, "Timestamp should be generated"

        logger.info("PASSED: Core models tests")
        test_results["Core Models"] = True
    except Exception as e:
        logger.error(f"Core models tests failed: {e}", exc_info=True)
        test_results["Core Models"] = False

    # Test InMemoryStore
    log_separator("RUNNING IN-MEMORY STORE TESTS")
    try:
        logger.info("Running in-memory store tests...")
        import test_in_memory_store

        in_memory_success = await test_in_memory_store.run_tests()
        test_results["In-Memory Store"] = in_memory_success
    except Exception as e:
        logger.error(f"In-memory store tests failed: {e}", exc_info=True)
        test_results["In-Memory Store"] = False

    # Test Memory Manager
    log_separator("RUNNING MEMORY MANAGER TESTS")
    try:
        logger.info("Running memory manager tests...")
        import test_memory_manager

        manager_success = await test_memory_manager.run_tests()
        test_results["Memory Manager"] = manager_success
    except Exception as e:
        logger.error(f"Memory manager tests failed: {e}", exc_info=True)
        test_results["Memory Manager"] = False

    # Test Configuration
    log_separator("RUNNING CONFIGURATION TESTS")
    try:
        logger.info("Testing configuration loading...")
        from ltm_agent.core.config import get_settings

        # Load settings from .env file
        settings = get_settings()

        # Check some key settings
        assert settings.anthropic_api_key is not None, "Anthropic API key should be set"
        assert settings.perplexity_api_key is not None, "Perplexity API key should be set"
        assert settings.vector_db_path is not None, "Vector DB path should be set"

        logger.info("PASSED: Configuration tests")
        logger.info(f"  Vector DB Path: {settings.vector_db_path}")
        logger.info(f"  Embedding Model: {settings.embedding_model_name}")
        logger.info(f"  Chunk Size: {settings.chunk_size}")
        logger.info(f"  Log Level: {settings.log_level}")

        test_results["Configuration"] = True
    except Exception as e:
        logger.error(f"Configuration tests failed: {e}", exc_info=True)
        test_results["Configuration"] = False

    # Print summary of results
    log_separator("TEST RESULTS SUMMARY")

    end_time = time.time()
    duration = end_time - start_time

    all_passed = all(test_results.values())
    status = "PASSED" if all_passed else "FAILED"

    logger.info(f"Overall Status: {status}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("\nComponent Results:")

    for component, passed in test_results.items():
        status_str = "PASSED" if passed else "FAILED"
        logger.info(f"  {component}: {status_str}")

    log_separator("END OF TEST RUN")
    return all_passed


if __name__ == "__main__":
    # Run all tests using asyncio
    success = asyncio.run(run_tests())

    # Use exit code based on test results
    sys.exit(0 if success else 1)
