"""
Subset of tests for the Anthropic API integration.

This module tests just the basic initialization functionality
of the AnthropicClient to get the tests running.
"""

import sys
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.llm.anthropic_client import AnthropicClient, AnthropicError


class TestAnthropicSubset:
    """Simple subset test suite for the AnthropicClient."""

    def test_initialization(self):
        """Test client initialization and validation."""
        print("\n=== Testing initialization ===")
        # Valid initialization
        client = AnthropicClient(api_key="test-api-key")
        assert client.api_key == "test-api-key"
        assert client.model == "claude-2"  # Default
        print("Initialization test passed")

        # Invalid initialization (no API key)
        with pytest.raises(AnthropicError):
            AnthropicClient(api_key=None)
        print("Error handling test passed")

    def test_custom_parameters(self):
        """Test client initialization with custom parameters."""
        print("\n=== Testing custom parameters ===")
        client = AnthropicClient(
            api_key="test-api-key",
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0.2,
            request_timeout=120.0,
        )

        assert client.model == "claude-3-opus-20240229"
        assert client.max_tokens == 2000
        assert client.temperature == 0.2
        assert client.request_timeout == 120.0
        print("Custom parameters test passed")

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation with mocked API call."""
        print("\n=== Testing generate_response ===")
        try:
            # Create client
            client = AnthropicClient(api_key="test-api-key")
            print("Client created successfully")

            # Mock response data
            mock_response_data = {
                "id": "msg_123456",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test response."}],
                "model": "claude-2",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

            # Create a mock response object
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            print("Mock response created")

            # Create a mock post method that returns our mock response
            mock_post = AsyncMock(return_value=mock_response)
            print("Mock post method created")

            # Create messages for the API
            messages = [{"role": "user", "content": "Tell me about Python."}]

            # Patch the httpx.AsyncClient.post method
            print("Patching httpx.AsyncClient.post...")
            with patch("httpx.AsyncClient.post", mock_post):
                print("Patch applied")

                # Call the generate_response method
                print("Calling generate_response...")
                result = await client.generate_response(messages)
                print(f"Result received: {result}")

                # Verify results
                assert mock_post.called
                print("Mock post method was called")
                assert result["content"][0]["text"] == "This is a test response."
                print("Response text verified")
                assert result["model"] == "claude-2"
                print("Model verified")

            print("generate_response test passed")
        except Exception as e:
            print(f"ERROR in test_generate_response: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test generating a response with context."""
        print("\n=== Testing generate_with_context ===")
        try:
            # Create client
            client = AnthropicClient(api_key="test-api-key")
            print("Client created successfully")

            # Mock generate_response method
            async def mock_generate_response(*args, **kwargs):
                print("Mock generate_response called with args:", args)
                print("Mock generate_response called with kwargs:", kwargs)
                return {
                    "content": [{"text": "This is a response with context."}],
                    "model": "claude-2",
                    "stop_reason": "end_turn",
                }

            # Patch the generate_response method
            print("Patching client.generate_response...")
            with patch.object(client, "generate_response", side_effect=mock_generate_response):
                print("Patch applied")

                # Create context items
                context_items = [
                    {"content": "Python is a programming language.", "relevance": 0.9, "id": "123"}
                ]

                # Call generate_with_context
                print("Calling generate_with_context...")
                result = await client.generate_with_context(
                    user_query="Tell me about Python", context_items=context_items
                )
                print(f"Result received: {result}")

                # Verify results
                assert "content" in result
                print("Content field verified")
                assert "model" in result
                print("Model field verified")
                assert "context_items_used" in result
                print("Context items used field verified")

            print("generate_with_context test passed")
        except Exception as e:
            print(f"ERROR in test_generate_with_context: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise
