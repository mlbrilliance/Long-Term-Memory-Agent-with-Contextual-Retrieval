"""
Simple script to test if the Settings class works correctly with missing environment variables.
"""

import os

from src.ltm_agent.core.config import Settings

# Temporarily unset required environment variables
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]
if "PERPLEXITY_API_KEY" in os.environ:
    del os.environ["PERPLEXITY_API_KEY"]

try:
    # This should raise a ValidationError about missing required fields
    settings = Settings.from_env()
    print("ERROR: Settings creation should have failed!")
except Exception as e:
    print(f"SUCCESS: Expected error was raised: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    # Print the first error (if it's a validation error with errors list)
    if hasattr(e, "errors") and isinstance(e.errors, list) and len(e.errors) > 0:
        print(f"First error: {e.errors[0]}")
