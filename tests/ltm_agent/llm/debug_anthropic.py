"""
Debug script for Anthropic client.

This script provides a minimal test case for debugging the AnthropicClient.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from unittest.mock import AsyncMock, patch

from ltm_agent.llm.anthropic_client import AnthropicClient


async def test_init():
    """Test basic initialization."""
    print("\n=== Testing initialization ===")
    try:
        client = AnthropicClient(api_key="test-api-key")
        print("Successfully initialized client")
        print(f"API key: {client.api_key}")
        print(f"Model: {client.model}")
        return client
    except Exception as e:
        print(f"Error initializing client: {type(e).__name__}: {str(e)}")
        raise


async def test_generate_response(client):
    """Test response generation with mocked API."""
    print("\n=== Testing generate_response ===")

    # Create mock response
    mock_response = {
        "id": "msg_123456",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a test response."}],
        "model": "claude-2",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    # Create test messages
    messages = [{"role": "user", "content": "Test message"}]

    # Mock the API call
    with patch("httpx.AsyncClient.post") as mock_post:
        print("Created mock for httpx.AsyncClient.post")
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = AsyncMock()

        print("Calling generate_response...")
        try:
            result = await client.generate_response(messages)
            print(f"Response received: {result}")

            # Check key fields
            if "content" in result and isinstance(result["content"], list):
                print("Content field is correct")
            else:
                print(f"WARNING: Content field has unexpected format: {result.get('content')}")

            if "model" in result:
                print(f"Model field is correct: {result['model']}")
            else:
                print("WARNING: Model field missing")

            return result
        except Exception as e:
            print(f"Error in generate_response: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


async def main():
    """Run all tests."""
    try:
        client = await test_init()
        await test_generate_response(client)
        print("\n=== All tests completed successfully ===")
    except Exception as e:
        print(f"\n=== Tests failed: {type(e).__name__}: {str(e)} ===")


if __name__ == "__main__":
    asyncio.run(main())
