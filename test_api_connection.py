"""
Test script to verify API connection and model availability for Claude.

This helps diagnose the exact model name issues we're seeing.
"""

import os

import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# Test models to try
models_to_test = [
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]

# Test models
print("Testing available Claude models...")
print("=" * 50)

for model in models_to_test:
    print(f"Testing model: {model}")
    try:
        # Simple test message
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello, are you available?"}],
        )
        print(f"✓ SUCCESS: {model} is available")
        print(f"Response: {response.content[0].text[:50]}...\n")
    except Exception as e:
        print(f"✗ ERROR: {model} - {str(e)}\n")

print("=" * 50)
print("Test complete. Use a working model name in src/ltm_agent/main.py")
