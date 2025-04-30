"""
Simple test to verify our configuration implementation.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ltm_agent.core.config import Settings


def test_config_instantiation():
    """Test that Settings class can be directly instantiated."""
    settings = Settings(anthropic_api_key="test_key", perplexity_api_key="test_key")
    assert settings.anthropic_api_key == "test_key"
    assert settings.perplexity_api_key == "test_key"
    # Check defaults
    assert settings.chunk_size == 512
    assert settings.log_level == "INFO"
