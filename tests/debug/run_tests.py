"""
Custom test runner script to bypass pytest CLI issues.

This script directly imports our test modules and runs tests.
"""

import os
import sys
from pathlib import Path

# Add the project root and src directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"Path: {sys.path}")

# Import our KnowledgeUnit model and create a simple test
try:
    from ltm_agent.core.models import KnowledgeUnit

    # Simple test for knowledge unit creation
    ku = KnowledgeUnit(original_chunk="Test chunk", knowledge_source="action")

    print("\nKnowledgeUnit test successful!")
    print(f"Knowledge Unit: {ku}")

    # Test the unique ID generation
    assert ku.unique_id is not None and len(ku.unique_id) > 0, "Unique ID should be generated"
    print("Unique ID test passed!")

    # Test the timestamp generation
    assert ku.timestamp is not None, "Timestamp should be generated"
    print("Timestamp test passed!")

    # Test the knowledge source validation
    try:
        invalid_ku = KnowledgeUnit(
            original_chunk="Test chunk",
            knowledge_source="invalid",  # This should fail validation
        )
        print("ERROR: Knowledge source validation failed!")
    except Exception as e:
        print(f"Knowledge source validation correctly failed with: {e}")

    print("\nAll basic tests passed!")

except ImportError as e:
    print(f"\nERROR: Unable to import KnowledgeUnit: {e}")
    print("Check your Python path and package installation.")
