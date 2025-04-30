# Configuration Loader Specification

This document defines the requirements and design for configuration handling in the Long-Term Memory Agent project.

## Requirements
- **Load configuration and secrets** from environment variables, using `.env` files for local development.
- **Leverage Pydantic's `BaseSettings`** for type-safe config access and validation.
- **No hardcoded secrets** in code or config files.
- **Support for required and optional variables** with sensible defaults where appropriate.
- **Fail fast** when required variables are missing.
- **Support type coercion** for numeric and boolean config values.
- **Document all configuration variables** in `.env.example`.

## Variables
- `ANTHROPIC_API_KEY` (required, str): Anthropic API key.
- `PERPLEXITY_API_KEY` (required, str): Perplexity API key.
- `VOYAGE_API_KEY` (optional, str): Voyage API key.
- `COHERE_API_KEY` (optional, str): Cohere API key.
- `VECTOR_DB_PATH` (required, str): Path for the vector database (default: `local_chroma`).
- `CHUNK_SIZE` (optional, int): Chunk size for document splitting (default: 512).
- `CHUNK_OVERLAP` (optional, int): Overlap between chunks (default: 64).
- `EMBEDDING_MODEL_NAME` (required, str): Name of the embedding model.
- `LOG_LEVEL` (optional, str): Logging level (default: `INFO`).

## Implementation Guidance
- Use `python-dotenv` to load `.env` files in development.
- Use Pydantic's `BaseSettings` for config class.
- Use `Field(..., validation_alias=AliasChoices('ENV_VAR_NAME'))` if needed for env var mapping.
- Add model validators if complex logic is needed.

## TDD Anchors
- Test loading config with all required variables present.
- Test failure when required variables are missing.
- Test default value usage for optional variables.
- Test type coercion (e.g., int for `CHUNK_SIZE`).

---

See `.env.example` for all supported variables.
