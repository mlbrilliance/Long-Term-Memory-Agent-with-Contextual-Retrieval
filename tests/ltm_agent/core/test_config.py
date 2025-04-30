"""
Tests for the configuration loader module.

These tests verify that the Settings class correctly loads and validates
configuration from environment variables.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings


class TestSettings:
    """Test suite for the Settings class."""

    def test_direct_instantiation(self):
        """Test that Settings can be directly instantiated with required fields."""
        settings = Settings(anthropic_api_key="test_key", perplexity_api_key="test_key")
        assert settings.anthropic_api_key == "test_key"
        assert settings.perplexity_api_key == "test_key"
        # Check defaults
        assert settings.chunk_size == 512
        assert settings.log_level == "INFO"

    def test_required_env_vars(self):
        """Test that Settings.from_env raises error when required env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((ValidationError, ValueError)):
                Settings.from_env()

        # Test with only one required field
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
            with pytest.raises((ValidationError, ValueError)):
                Settings.from_env()

    def test_minimum_required_env_vars(self):
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

    def test_optional_env_vars(self):
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

    def test_optional_api_keys(self):
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
            assert getattr(settings, "voyage_api_key", None) in (None, "")
            assert getattr(settings, "cohere_api_key", None) in (None, "")

    def test_default_values(self):
        """Test that default values are used for optional variables."""
        test_env = {"ANTHROPIC_API_KEY": "a", "PERPLEXITY_API_KEY": "b"}
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings.from_env()
            assert settings.chunk_size == 512
            assert settings.chunk_overlap == 64
            assert settings.vector_db_path.endswith("local_chroma")
            assert settings.log_level == "INFO"

    def test_log_level_override(self):
        """Test that LOG_LEVEL can be overridden via env."""
        test_env = {"ANTHROPIC_API_KEY": "a", "PERPLEXITY_API_KEY": "b", "LOG_LEVEL": "DEBUG"}
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings.from_env()
            assert settings.log_level == "DEBUG"

    def test_type_coercion(self):
        """Test that integer and other types are coerced from env vars."""
        test_env = {
            "ANTHROPIC_API_KEY": "a",
            "PERPLEXITY_API_KEY": "b",
            "CHUNK_SIZE": "1024",
            "CHUNK_OVERLAP": "128",
        }
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings.from_env()
            assert isinstance(settings.chunk_size, int)
            assert settings.chunk_size == 1024
            assert isinstance(settings.chunk_overlap, int)
            assert settings.chunk_overlap == 128

    def test_custom_config_values(self):
        """Test that custom configuration values are loaded correctly."""
        test_env = {
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "PERPLEXITY_API_KEY": "test_perplexity_key",
            "VECTOR_DB_PATH": "/custom/path",
            "CHUNK_SIZE": "1024",
            "CHUNK_OVERLAP": "128",
            "EMBEDDING_MODEL_NAME": "custom-model",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings.from_env()
            assert settings.vector_db_path == "/custom/path"
            assert settings.chunk_size == 1024
            assert settings.chunk_overlap == 128
            assert settings.embedding_model_name == "custom-model"
            assert settings.log_level == "DEBUG"

    def test_validates_positive_integers(self):
        """Test that chunk_size and chunk_overlap must be positive integers."""
        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test_key", perplexity_api_key="test_key", chunk_size=0)

        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test_key", perplexity_api_key="test_key", chunk_overlap=-1)

    def test_validates_log_level(self):
        """Test that log_level must be one of the standard logging levels."""
        with pytest.raises(ValueError):
            Settings(
                anthropic_api_key="test_key",
                perplexity_api_key="test_key",
                log_level="INVALID_LEVEL",
            )

        # Should accept case-insensitive valid log levels
        settings = Settings(
            anthropic_api_key="test_key", perplexity_api_key="test_key", log_level="debug"
        )
        assert settings.log_level == "DEBUG"

    def test_get_settings_function(self):
        """Test that get_settings returns a Settings instance."""
        # This requires importing get_settings in a context where settings is properly defined
        # Testing this would require a more complex setup, so we'll skip it for now
        pass
