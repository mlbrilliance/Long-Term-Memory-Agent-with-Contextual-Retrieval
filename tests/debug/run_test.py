"""
Run the required environment variables test manually to see what's going on.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from pydantic import ValidationError

from ltm_agent.core.config import Settings


def test_required_env_vars():
    """Test that Settings.from_env raises error when required env vars are missing."""
    print("\nTesting with empty environment...")
    try:
        with patch.dict(os.environ, {}, clear=True):
            Settings.from_env()
            print("ERROR: Should have raised ValidationError")
            return False
    except Exception as e:
        print(f"SUCCESS: Caught exception: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        # ValidationError is a subclass of ValueError in Pydantic v2
        if isinstance(e, (ValidationError, ValueError)):
            print("Test PASSED (success)")
            return True
        else:
            print(f"Test FAILED: Expected ValidationError or ValueError, got {type(e).__name__}")
            return False


if __name__ == "__main__":
    success = test_required_env_vars()
    sys.exit(0 if success else 1)
