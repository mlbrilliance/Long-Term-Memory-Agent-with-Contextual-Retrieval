"""
Simplified tests for the Anthropic API integration.

This module tests the basic functionality of the AnthropicClient.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.llm.anthropic_client import AnthropicClient, AnthropicError


class TestAnthropicSimple:
    """Simple test suite for the AnthropicClient."""

    def test_init_with_api_key(self):
        """Test basic initialization with API key."""
        # Print debug information
        print("\n=== Starting test_init_with_api_key ===")

        try:
            client = AnthropicClient(api_key="test-api-key")
            print("Successfully created client with API key")
            print(f"Client model: {client.model}")
            print(f"Client max_tokens: {client.max_tokens}")
            assert client.api_key == "test-api-key"
            assert client.model == "claude-2"  # Default
            print("=== test_init_with_api_key PASSED ===")
        except Exception as e:
            print(f"ERROR initializing client: {str(e)}")
            raise

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        print("\n=== Starting test_init_without_api_key ===")

        try:
            with pytest.raises(AnthropicError) as excinfo:
                AnthropicClient(api_key=None)
            print(f"Got expected error: {str(excinfo.value)}")
            print("=== test_init_without_api_key PASSED ===")
        except Exception as e:
            print(f"ERROR during test: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_generate_response_mock(self):
        """Test response generation with mocked API call."""
        print("\n=== Starting test_generate_response_mock ===")

        try:
            # Create client
            client = AnthropicClient(api_key="test-api-key")
            print("Created client successfully")

            # Mock the httpx response
            mock_response = {
                "id": "msg_123456",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test response."}],
                "model": "claude-2",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

            # Create messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about Python."},
            ]

            print("Set up test data successfully")

            # Mock the API call
            with patch("httpx.AsyncClient.post") as mock_post:
                print("Created patch for httpx.AsyncClient.post")
                mock_post.return_value = AsyncMock()
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = mock_response
                mock_post.return_value.raise_for_status = AsyncMock()

                print("Set up mock return values")

                # Call the method
                print("Calling generate_response...")
                result = await client.generate_response(messages)
                print(f"Got result: {result}")

                # Verify the API was called with correct parameters
                assert mock_post.called
                print("API was called")

                # Verify result contains expected fields
                assert result["content"][0]["text"] == "This is a test response."
                assert result["model"] == "claude-2"
                print("Result contains expected fields")
                print("=== test_generate_response_mock PASSED ===")
        except Exception as e:
            print(f"ERROR during test: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
