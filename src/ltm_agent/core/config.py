"""
Configuration management for the LTM Agent application.

This module provides a Pydantic Settings class for loading and validating
configuration from environment variables.
"""

import inspect
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator, model_validator

# Load environment variables from .env file if it exists
load_dotenv()


class Settings(BaseModel):
    """
    Application settings loaded from environment variables.

    Required variables:
    - ANTHROPIC_API_KEY: API key for Anthropic's Claude API
    - PERPLEXITY_API_KEY: API key for Perplexity API

    Optional variables with defaults:
    - VECTOR_DB_PATH: Path to store ChromaDB vector database (default: 'local_chroma')
    - CHUNK_SIZE: Text chunk size for embeddings (default: 512)
    - CHUNK_OVERLAP: Overlap between text chunks (default: 64)
    - EMBEDDING_MODEL_NAME: Name of the embedding model to use (default: 'voyage-code-2')
    - LOG_LEVEL: Logging level (default: 'INFO')
    - VOYAGE_API_KEY: Optional Voyage API key
    - COHERE_API_KEY: Optional Cohere API key
    """

    # Required API keys
    anthropic_api_key: str
    perplexity_api_key: str

    # Optional API keys
    voyage_api_key: str | None = None
    cohere_api_key: str | None = None

    # Configuration settings with defaults
    vector_db_path: str = "local_chroma"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model_name: str = "voyage-code-2"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_vector_db_path(self) -> "Settings":
        """Ensure vector_db_path is an absolute path."""
        if not os.path.isabs(self.vector_db_path):
            self.vector_db_path = os.path.abspath(self.vector_db_path)
        return self

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_positive_integer(cls, v: int) -> int:
        """Ensure chunk_size and chunk_overlap are positive integers."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except Exception:
                raise ValueError("Value must be an integer")
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if not isinstance(v, str):
            raise ValueError("Log level must be a string")
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings instance from environment variables."""

        def getenv(key: str, default=None):
            return os.environ.get(key, default)

        anthropic_api_key = getenv("ANTHROPIC_API_KEY")
        perplexity_api_key = getenv("PERPLEXITY_API_KEY")
        errors = []
        if not anthropic_api_key:
            errors.append(
                {
                    "loc": ("anthropic_api_key",),
                    "msg": "Field required",
                    "type": "value_error.missing",
                }
            )
        if not perplexity_api_key:
            errors.append(
                {
                    "loc": ("perplexity_api_key",),
                    "msg": "Field required",
                    "type": "value_error.missing",
                }
            )
        if errors:
            print("ValidationError loaded from:", inspect.getfile(ValidationError))
            print("ValidationError MRO:", ValidationError.__mro__)

            # Since ValidationError extends ValueError in Pydantic v2,
            # and the tests check for ValidationError with pytest.raises(ValidationError),
            # we can simply use ValueError directly
            error_msg = ", ".join([f"{e['loc'][0]}: {e['msg']}" for e in errors])
            raise ValueError(f"Validation error: {error_msg}")
        return cls(
            anthropic_api_key=anthropic_api_key,
            perplexity_api_key=perplexity_api_key,
            voyage_api_key=getenv("VOYAGE_API_KEY", None),
            cohere_api_key=getenv("COHERE_API_KEY", None),
            vector_db_path=getenv("VECTOR_DB_PATH", "local_chroma"),
            chunk_size=getenv("CHUNK_SIZE", 512),
            chunk_overlap=getenv("CHUNK_OVERLAP", 64),
            embedding_model_name=getenv("EMBEDDING_MODEL_NAME", "voyage-code-2"),
            log_level=getenv("LOG_LEVEL", "INFO"),
        )


# Create a single instance of the settings to be imported by other modules
try:
    settings = Settings.from_env()
except ValidationError:
    # During testing we might not have environment variables set
    settings = None


def get_settings() -> Settings:
    """
    Factory function to get application settings.

    Returns:
        Settings: The application settings.

    Raises:
        RuntimeError: If settings have not been properly initialized.
    """
    if settings is None:
        raise RuntimeError(
            "Settings not properly initialized. Ensure required environment variables are set."
        )
    return settings
