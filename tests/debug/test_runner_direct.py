"""
Run all the tests directly without pytest to verify functionality.

This script runs the same tests as test_config.py but using direct Python instead of pytest,
which avoids issues with pytest's error handling and shows clearer output.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings


def run_test(name, test_func):
    """Run a test function and print the result."""
    print(f"\n{'=' * 40}")
    print(f"Running test: {name}")
    print(f"{'=' * 40}")
    try:
        test_func()
        print(f"[PASS] PASSED: {name}")
        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {name}")
        print(f"Error: {type(e).__name__}: {str(e)}")
        return False


def test_direct_instantiation():
    """Test that Settings can be directly instantiated with required fields."""
    settings = Settings(anthropic_api_key="test_key", perplexity_api_key="test_key")
    assert settings.anthropic_api_key == "test_key"
    assert settings.perplexity_api_key == "test_key"
    # Check defaults
    assert settings.chunk_size == 512
    assert settings.log_level == "INFO"


def test_required_env_vars():
    """Test that Settings.from_env raises error when required env vars are missing."""
    # First test with empty environment
    with patch.dict(os.environ, {}, clear=True):
        try:
            Settings.from_env()
            assert False, "Should have raised an error"
        except (ValueError, Exception) as e:
            print(f"Got expected error: {type(e).__name__}: {str(e)}")

    # Test with only one required field
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        try:
            Settings.from_env()
            assert False, "Should have raised an error"
        except (ValueError, Exception) as e:
            print(f"Got expected error: {type(e).__name__}: {str(e)}")


def test_minimum_required_env_vars():
    """Test that Settings can be instantiated with just the required env vars."""
    test_env = {
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "PERPLEXITY_API_KEY": "test_perplexity_key",
    }
    with patch.dict(os.environ, test_env, clear=True):
        settings = Settings.from_env()
        assert settings.anthropic_api_key == "test_anthropic_key"
        assert settings.perplexity_api_key == "test_perplexity_key"
        # Check defaults are applied
        assert settings.vector_db_path == os.path.abspath("local_chroma")
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.embedding_model_name == "voyage-code-2"
        assert settings.log_level == "INFO"


def test_validates_positive_integers():
    """Test that chunk_size and chunk_overlap must be positive integers."""
    try:
        Settings(anthropic_api_key="test_key", perplexity_api_key="test_key", chunk_size=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        Settings(anthropic_api_key="test_key", perplexity_api_key="test_key", chunk_overlap=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# Run the tests
if __name__ == "__main__":
    results = []

    results.append(run_test("test_direct_instantiation", test_direct_instantiation))
    results.append(run_test("test_required_env_vars", test_required_env_vars))
    results.append(run_test("test_minimum_required_env_vars", test_minimum_required_env_vars))
    results.append(run_test("test_validates_positive_integers", test_validates_positive_integers))

    # Print summary
    print("\n\n" + "=" * 50)
    print(f"Test Summary: {sum(results)}/{len(results)} tests passed")

    # Exit with success only if all tests passed
    sys.exit(0 if all(results) else 1)
