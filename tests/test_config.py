"""
Tests for the configuration module.
"""
import os
import pytest
from pydantic import ValidationError

from src.config import Settings, LogLevel, GroqModel, SearchProvider


def test_settings_defaults():
    """Test that default settings are loaded correctly."""
    # Set minimal required env vars
    os.environ["GROQ_API_KEY"] = "test_key"
    
    settings = Settings()
    
    # Check defaults
    assert settings.app_name == "Car Repair Assistant"
    assert settings.debug is False
    assert settings.log_level == LogLevel.INFO
    assert settings.model_name == GroqModel.LLAMA3_70B
    assert settings.search_provider == SearchProvider.SERPAPI
    assert settings.vector_db_path == "./chroma_db"
    assert settings.session_expiry_minutes == 60
    assert settings.max_history_length == 50
    assert settings.similarity_threshold == 0.65


def test_settings_override():
    """Test that settings can be overridden with environment variables."""
    os.environ["APP_NAME"] = "Custom App Name"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["GROQ_API_KEY"] = "custom_key"
    os.environ["MODEL_NAME"] = "llama3-8b-8192"
    os.environ["SEARCH_PROVIDER"] = "brave"
    os.environ["SEARCH_API_KEY"] = "search_key"
    os.environ["VECTOR_DB_PATH"] = "./custom_db"
    os.environ["SESSION_EXPIRY_MINUTES"] = "120"
    os.environ["MAX_HISTORY_LENGTH"] = "100"
    os.environ["SIMILARITY_THRESHOLD"] = "0.75"
    
    settings = Settings()
    
    # Check overridden values
    assert settings.app_name == "Custom App Name"
    assert settings.debug is True
    assert settings.log_level == LogLevel.DEBUG
    assert settings.groq_api_key.get_secret_value() == "custom_key"
    assert settings.model_name == GroqModel.LLAMA3_8B
    assert settings.search_provider == SearchProvider.BRAVE
    assert settings.search_api_key.get_secret_value() == "search_key"
    assert settings.vector_db_path == "./custom_db"
    assert settings.session_expiry_minutes == 120
    assert settings.max_history_length == 100
    assert settings.similarity_threshold == 0.75


def test_google_search_validation():
    """Test validation for Google search provider."""
    # Setup Google search without CSE ID
    os.environ["GROQ_API_KEY"] = "test_key"
    os.environ["SEARCH_PROVIDER"] = "google"
    os.environ["SEARCH_API_KEY"] = "search_key"
    os.environ.pop("GOOGLE_CSE_ID", None)
    
    # Should raise validation error
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    
    assert "google_cse_id must be provided when using Google search" in str(excinfo.value)
    
    # Now add CSE ID
    os.environ["GOOGLE_CSE_ID"] = "cse_id"
    settings = Settings()
    
    assert settings.google_cse_id == "cse_id"


def test_invalid_enum_values():
    """Test validation for invalid enum values."""
    os.environ["GROQ_API_KEY"] = "test_key"
    
    # Invalid log level
    os.environ["LOG_LEVEL"] = "INVALID"
    with pytest.raises(ValidationError):
        Settings()
    
    # Invalid model name
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["MODEL_NAME"] = "invalid-model"
    with pytest.raises(ValidationError):
        Settings()
    
    # Invalid search provider
    os.environ["MODEL_NAME"] = "llama3-70b-8192"
    os.environ["SEARCH_PROVIDER"] = "invalid-provider"
    with pytest.raises(ValidationError):
        Settings()


def test_optional_settings():
    """Test optional settings."""
    os.environ["GROQ_API_KEY"] = "test_key"
    os.environ.pop("SEARCH_API_KEY", None)
    os.environ.pop("GOOGLE_CSE_ID", None)
    
    settings = Settings()
    
    assert settings.search_api_key is None
    assert settings.google_cse_id is None