"""
Debug script to check environment configuration.
"""

import importlib
import os
import sys


def check_module(module_name):
    """Try to import a module and return if it's available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Check Python version and executable
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

# Check if key modules are available
modules_to_check = [
    "pytest",
    "pytest_asyncio",
    "pytest_cov",
    "ltm_agent",
    "ltm_agent.core.models",
    "ltm_agent.memory.interfaces",
]

print("\nModule availability:")
for module in modules_to_check:
    result = check_module(module)
    print(f"  {module}: {'Available' if result else 'Not available'}")

if __name__ == "__main__":
    print("\nEnvironment check complete")
