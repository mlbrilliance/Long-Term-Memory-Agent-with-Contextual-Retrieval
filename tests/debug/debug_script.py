"""Script to debug ValidationError issues - writes to a file."""

import os
import traceback

# Create a debugging log
with open("debug_output.txt", "w") as f:
    try:
        # Import only after setting up logging
        from src.ltm_agent.core.config import Settings

        # Clear environment variables
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        if "PERPLEXITY_API_KEY" in os.environ:
            del os.environ["PERPLEXITY_API_KEY"]

        f.write("About to call Settings.from_env()\n")
        settings = Settings.from_env()
        f.write("ERROR: Should have failed\n")
    except Exception as e:
        f.write(f"Exception type: {type(e).__name__}\n")
        f.write(f"Exception message: {str(e)}\n")
        f.write(f"Exception dir: {dir(e)}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())

print("Debug output written to debug_output.txt")
