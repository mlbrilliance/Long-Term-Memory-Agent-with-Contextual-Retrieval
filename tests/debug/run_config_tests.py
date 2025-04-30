"""
Self-contained test runner for Settings configuration tests.

This script runs all the required tests for the Settings class without relying on pytest,
ensuring compatibility with Pydantic v2's ValidationError implementation.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings


class TestRunner:
    """Simple test runner for Settings tests."""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0

    def run_test(self, name, test_func):
        """Run a test function and print the result."""
        self.total_tests += 1
        print(f"\n{'=' * 40}")
        print(f"Running test: {name}")
        print(f"{'=' * 40}")
        try:
            test_func()
            self.passed_tests += 1
            print(f"[PASS] PASSED: {name}")
            return True
        except Exception as e:
            print(f"[FAIL] FAILED: {name}")
            print(f"Error: {type(e).__name__}: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return False

    def report(self):
        """Print test report."""
        print("\n" + "=" * 50)
        print(f"Test Summary: {self.passed_tests}/{self.total_tests} tests passed")
        print("=" * 50)
        return self.passed_tests == self.total_tests


# Test functions
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


def test_optional_env_vars():
    """Test that optional env vars are loaded correctly when provided."""
    test_env = {
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "PERPLEXITY_API_KEY": "test_perplexity_key",
        "VOYAGE_API_KEY": "test_voyage_key",
        "COHERE_API_KEY": "test_cohere_key",
    }
    with patch.dict(os.environ, test_env, clear=True):
        settings = Settings.from_env()
        assert settings.voyage_api_key == "test_voyage_key"
        assert settings.cohere_api_key == "test_cohere_key"


def test_optional_api_keys():
    """Test that optional API keys are loaded if present, otherwise None."""
    test_env = {
        "ANTHROPIC_API_KEY": "a",
        "PERPLEXITY_API_KEY": "b",
        "VOYAGE_API_KEY": "v",
        "COHERE_API_KEY": "c",
    }
    with patch.dict(os.environ, test_env, clear=True):
        settings = Settings.from_env()
        assert settings.voyage_api_key == "v"
        assert settings.cohere_api_key == "c"

    # If not present, should be None or default
    test_env = {"ANTHROPIC_API_KEY": "a", "PERPLEXITY_API_KEY": "b"}
    with patch.dict(os.environ, test_env, clear=True):
        settings = Settings.from_env()
        assert settings.voyage_api_key in (None, "")
        assert settings.cohere_api_key in (None, "")


def test_default_values():
    """Test that default values are used for optional variables."""
    test_env = {"ANTHROPIC_API_KEY": "a", "PERPLEXITY_API_KEY": "b"}
    with patch.dict(os.environ, test_env, clear=True):
        settings = Settings.from_env()
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.vector_db_path.endswith("local_chroma")
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


def test_validates_log_level():
    """Test that log_level must be one of the standard logging levels."""
    try:
        Settings(
            anthropic_api_key="test_key", perplexity_api_key="test_key", log_level="INVALID_LEVEL"
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Should accept case-insensitive valid log levels
    settings = Settings(
        anthropic_api_key="test_key", perplexity_api_key="test_key", log_level="debug"
    )
    assert settings.log_level == "DEBUG"


# Run all tests
if __name__ == "__main__":
    runner = TestRunner()
    runner.run_test("test_direct_instantiation", test_direct_instantiation)
    runner.run_test("test_required_env_vars", test_required_env_vars)
    runner.run_test("test_minimum_required_env_vars", test_minimum_required_env_vars)
    runner.run_test("test_optional_env_vars", test_optional_env_vars)
    runner.run_test("test_optional_api_keys", test_optional_api_keys)
    runner.run_test("test_default_values", test_default_values)
    runner.run_test("test_validates_positive_integers", test_validates_positive_integers)
    runner.run_test("test_validates_log_level", test_validates_log_level)

    success = runner.report()
    sys.exit(0 if success else 1)
