"""Test script to debug Pydantic ValidationError behavior with pytest."""

import inspect
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Record output to file
with open("pytest_debug.log", "w") as f:
    # Print ValidationError information
    f.write(f"ValidationError loaded from: {inspect.getfile(ValidationError)}\n")
    f.write(f"ValidationError MRO: {ValidationError.__mro__}\n\n")

    # Add src to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))

    from ltm_agent.core.config import Settings

    # Set up environment for test
    f.write("=== Testing test_required_env_vars ===\n")
    try:
        with patch.dict(os.environ, {}, clear=True):
            try:
                Settings.from_env()
                f.write("ERROR: Should have failed with ValidationError\n")
            except Exception as e:
                f.write(f"CAUGHT EXCEPTION: {type(e).__name__}\n")
                f.write(f"Exception message: {str(e)}\n")
                f.write("Exception traceback:\n")
                f.write(traceback.format_exc())
                f.write("\n")
    except Exception as outer_e:
        f.write(f"OUTER EXCEPTION: {type(outer_e).__name__}\n")
        f.write(f"Message: {str(outer_e)}\n")
        f.write(traceback.format_exc())
        f.write("\n")

    # Now test with pytest
    f.write("\n=== Testing with pytest directly ===\n")
    try:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as excinfo:
                Settings.from_env()

            f.write(f"Pytest excinfo type: {type(excinfo.value).__name__}\n")
            f.write(f"Pytest excinfo message: {str(excinfo.value)}\n")
            f.write(f"Pytest excinfo dir: {dir(excinfo.value)}\n")
    except Exception as e:
        f.write(f"PYTEST EXCEPTION: {type(e).__name__}\n")
        f.write(f"Message: {str(e)}\n")
        f.write(traceback.format_exc())

print("Debug information written to pytest_debug.log")
