"""Minimal test for ValidationError handling."""

import os

from src.ltm_agent.core.config import Settings

# Clear environment variables
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]
if "PERPLEXITY_API_KEY" in os.environ:
    del os.environ["PERPLEXITY_API_KEY"]

try:
    settings = Settings.from_env()
    print("ERROR: Should have failed")
except Exception as e:
    print(f"OK: Got {type(e).__name__}")
    print(f"Message: {str(e)[:100]}")  # Only first 100 chars
