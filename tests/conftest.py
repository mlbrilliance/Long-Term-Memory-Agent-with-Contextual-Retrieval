"""
Pytest configuration file for the ltm_agent test suite.

This file helps with module discovery and path configuration for tests.
"""

import sys
from pathlib import Path

import pytest

# Add the project root directory to the Python path so that the ltm_agent module can be imported
# This is necessary because we're not installing the package in development mode
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the src directory to the Python path
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Configure pytest-asyncio to use the "auto" mode
pytest_plugins = ["pytest_asyncio"]


# Set default event loop policy for asyncio tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio

    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
