"""
Minimal test for AnthropicClient to identify issues.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import directly - no pytest or async
try:
    from ltm_agent.llm.anthropic_client import AnthropicClient

    print("Successfully imported AnthropicClient")
except Exception as e:
    print(f"Error importing AnthropicClient: {type(e).__name__}: {str(e)}")
    sys.exit(1)

# Try to create an instance
try:
    client = AnthropicClient(api_key="test-api-key")
    print("Successfully created AnthropicClient instance")
    print(f"Model: {client.model}")
    print(f"API Key: {client.api_key}")
except Exception as e:
    print(f"Error creating AnthropicClient: {type(e).__name__}: {str(e)}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("Test completed successfully")
