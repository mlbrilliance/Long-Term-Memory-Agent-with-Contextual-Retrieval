"""
Run script for the Long-Term Memory Agent.

This script runs the Long-Term Memory Agent using the proper import approach.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python's module search path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import the main function from the module
from ltm_agent.main import main

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
