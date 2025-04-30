"""
Simple script to test our Pydantic ValidationError fix.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import the Settings class
from ltm_agent.core.config import Settings


def test_required_env_vars():
    """Test that ValidationError is raised when required environment variables are missing."""
    print("Testing required environment variables...")
    try:
        # Clear environment variables and attempt to create Settings
        os.environ.clear()
        Settings.from_env()
        print("ERROR: Should have raised ValidationError")
        return False
    except ValueError as e:
        print(f"SUCCESS: Caught ValueError: {str(e)}")
        return True
    except Exception as e:
        print(f"ERROR: Caught unexpected exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False


def test_valid_env_vars():
    """Test that Settings can be created with valid environment variables."""
    print("\nTesting valid environment variables...")
    try:
        # Set valid environment variables
        os.environ["ANTHROPIC_API_KEY"] = "test_key"
        os.environ["PINECONE_API_KEY"] = "test_key"
        os.environ["PINECONE_ENVIRONMENT"] = "test_env"
        os.environ["PINECONE_INDEX"] = "test_index"

        # Attempt to create Settings
        settings = Settings.from_env()
        print(f"SUCCESS: Created Settings: {settings}")
        return True
    except Exception as e:
        print(f"ERROR: Caught unexpected exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests
    test1 = test_required_env_vars()
    test2 = test_valid_env_vars()

    # Print overall results
    print("\nTest Results:")
    print(f"- Required env vars test: {'PASSED' if test1 else 'FAILED'}")
    print(f"- Valid env vars test: {'PASSED' if test2 else 'FAILED'}")

    # Set exit code based on test results
    sys.exit(0 if test1 and test2 else 1)
