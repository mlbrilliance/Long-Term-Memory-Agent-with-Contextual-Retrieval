"""
Simple test script to verify Anthropic API connection.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key and display first few characters
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    print(f"API Key found: {api_key[:10]}...[truncated]")
else:
    print("ERROR: No Anthropic API key found in environment variables")
    exit(1)

# Try to initialize the LLM from langchain
try:
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        api_key=api_key,
        model="claude-3-haiku-20240307",  # Use a valid model name
        temperature=0.7,
        max_tokens=1000,
    )

    print("✅ API Connection Test: LLM Initialized Successfully")

    # Basic test - ask a simple question
    print("\nPerforming test query...")
    response = llm.invoke("What is your name? Keep it very brief.")
    print(f"Response: {response.content}")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please make sure langchain_anthropic is installed.")

except Exception as e:
    print(f"❌ API Connection Error: {e}")
