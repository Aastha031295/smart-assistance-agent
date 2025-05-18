"""
Configuration management.
"""

import os
from enum import Enum
from typing import Optional

from pydantic import BaseSettings, Field, validator, SecretStr


class LogLevel(str, Enum):
    """Log level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class GroqModel(str, Enum):
    """Available Groq LLM models."""
    LLAMA3_70B = "llama3-70b-8192"
    LLAMA3_8B = "llama3-8b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"


class SearchProvider(str, Enum):
    """Supported search API providers."""
    SERPER = "serper"
    SERPAPI = "serpapi"
    BRAVE = "brave"
    GOOGLE = "google"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with validation.
    
    Attributes:
        app_name: Name of the application
        debug: Debug mode flag
        log_level: Logging level
        groq_api_key: Groq API key for LLM access
        model_name: Groq model to use
        search_api_key: API key for search provider
        search_provider: Search provider to use
        vector_db_path: Path to the vector database
        google_cse_id: Google Custom Search Engine ID (only needed for Google Search)
        session_expiry_minutes: Minutes until session expires
        max_history_length: Maximum number of messages in conversation history
        similarity_threshold: Threshold for determining relevance in RAG
    """
    # App settings
    app_name: str = Field("Car Repair Assistant", env="APP_NAME")
    debug: bool = Field(False, env="DEBUG")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    
    # API keys
    groq_api_key: SecretStr = Field(..., env="GROQ_API_KEY")
    search_api_key: Optional[SecretStr] = Field(None, env="SEARCH_API_KEY")
    
    # Model settings
    model_name: GroqModel = Field(GroqModel.LLAMA3_70B, env="MODEL_NAME")
    
    # Search settings
    search_provider: SearchProvider = Field(SearchProvider.SERPAPI, env="SEARCH_PROVIDER")
    google_cse_id: Optional[str] = Field(None, env="GOOGLE_CSE_ID")
    
    # Database settings
    vector_db_path: str = Field("./chroma_db", env="VECTOR_DB_PATH")
    
    # Application behavior
    session_expiry_minutes: int = Field(60, env="SESSION_EXPIRY_MINUTES")
    max_history_length: int = Field(50, env="MAX_HISTORY_LENGTH")
    similarity_threshold: float = Field(0.65, env="SIMILARITY_THRESHOLD")
    
    @validator("google_cse_id")
    def validate_google_cse_id(cls, v, values):
        """Validate that google_cse_id is provided if using Google search."""
        if values.get("search_provider") == SearchProvider.GOOGLE and not v:
            raise ValueError("google_cse_id must be provided when using Google search")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()