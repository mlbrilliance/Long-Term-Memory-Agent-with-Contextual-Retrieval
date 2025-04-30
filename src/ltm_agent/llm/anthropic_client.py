"""
Anthropic API integration for the Long-Term Memory Agent.

This module provides integration with Anthropic's Claude model, allowing the agent
to generate responses with contextual awareness.
"""

import json
import logging
from typing import Any

import httpx

from ltm_agent.core.config import Settings
from ltm_agent.core.exceptions import LTMAgentError

# Configure logging
logger = logging.getLogger(__name__)


class AnthropicError(LTMAgentError):
    """Exception raised for Anthropic API errors."""

    pass


class AnthropicClient:
    """
    Client for interacting with Anthropic's Claude API.

    This class handles communication with the Anthropic API, including authentication,
    request formatting, and response parsing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: Settings | None = None,
        model: str = "claude-2",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        request_timeout: float = 60.0,
        base_url: str = "https://api.anthropic.com/v1",
    ):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (overrides config if provided)
            config: Application settings
            model: Model to use for generation
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            request_timeout: Timeout for API requests in seconds
            base_url: Base URL for Anthropic API

        Raises:
            AnthropicError: If API key is not provided
        """
        self.config = config
        self.api_key = api_key or (config.anthropic_api_key if config else None)

        if not self.api_key:
            raise AnthropicError("Anthropic API key not provided")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.base_url = base_url

        # Default headers
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        logger.info(f"Initialized AnthropicClient with model={model}, temperature={temperature}")

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response from Claude.

        Args:
            messages: List of message objects with 'role' and 'content'
            max_tokens: Maximum tokens to generate (overrides instance value)
            temperature: Temperature for generation (overrides instance value)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Optional custom stop sequences

        Returns:
            Response from Claude API

        Raises:
            AnthropicError: If the API request fails
        """
        # Format the conversation for the messages API
        system_message = None
        formatted_messages = []

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                formatted_messages.append({"role": message["role"], "content": message["content"]})

        # Construct the request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        # Add system message if provided
        if system_message:
            payload["system"] = system_message

        # Add optional parameters if provided
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        # Log the request (excluding sensitive content)
        log_payload = payload.copy()
        if "messages" in log_payload:
            log_payload["messages"] = f"[{len(log_payload['messages'])} messages]"
        if "system" in log_payload:
            log_payload["system"] = "[system prompt]"
        logger.info(f"Sending request to Anthropic API: {json.dumps(log_payload)}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                )

                # Check for successful response
                response.raise_for_status()
                result = response.json()

                logger.info(f"Received response from Anthropic API: status={response.status_code}")

                return result
        except httpx.HTTPStatusError as e:
            error_msg = f"Anthropic API HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
            except Exception:
                error_msg += f" - {e.response.text}"

            logger.error(error_msg)
            raise AnthropicError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Anthropic API request error: {str(e)}"
            logger.error(error_msg)
            raise AnthropicError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error calling Anthropic API: {str(e)}"
            logger.error(error_msg)
            raise AnthropicError(error_msg) from e

    async def generate_with_context(
        self,
        user_query: str,
        context_items: list[dict[str, Any]],
        system_prompt: str | None = None,
        include_metadata: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response with context from the agent's memory.

        Args:
            user_query: User's query
            context_items: Relevant context items
            system_prompt: Optional custom system prompt
            include_metadata: Whether to include metadata in context
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated response with metadata

        Raises:
            AnthropicError: If generation fails
        """
        from ltm_agent.retrieval.context_retriever import PromptBuilder

        # Create a prompt builder
        prompt_builder = PromptBuilder(system_prompt_template=system_prompt)

        # Build the prompt with context
        prompt_data = prompt_builder.build_prompt(
            user_query, context_items, include_metadata=include_metadata
        )

        # Generate the response
        response = await self.generate_response(
            prompt_data["messages"], max_tokens=max_tokens, temperature=temperature
        )

        # Format the result
        result = {
            "content": response["content"][0]["text"],
            "model": response["model"],
            "stop_reason": response.get("stop_reason", None),
            "context_items_used": prompt_data["context_items_used"],
        }

        return result
