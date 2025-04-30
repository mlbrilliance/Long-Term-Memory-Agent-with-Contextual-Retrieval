"""
Tests for the Anthropic API integration.

This module tests the functionality of the AnthropicClient.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.llm.anthropic_client import AnthropicClient, AnthropicError


class TestAnthropicClient:
    """Test suite for the AnthropicClient."""

    def test_initialization(self):
        """Test client initialization and validation."""
        # Valid initialization
        client = AnthropicClient(api_key="test-api-key")
        assert client.api_key == "test-api-key"
        assert client.model == "claude-2"  # Default

        # Invalid initialization (no API key)
        with pytest.raises(AnthropicError):
            AnthropicClient(api_key=None)

    def test_custom_parameters(self):
        """Test client initialization with custom parameters."""
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
