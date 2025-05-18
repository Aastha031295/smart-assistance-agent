"""
Configuration management.
"""

from enum import Enum
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    app_name: str = Field(default="Car Repair Assistant")
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # API keys
    groq_api_key: SecretStr
    search_api_key: Optional[SecretStr] = None

    # Model settings
    model_name: GroqModel = Field(default=GroqModel.LLAMA3_70B)

    # Search settings
    search_provider: SearchProvider = Field(default=SearchProvider.SERPAPI)
    google_cse_id: Optional[str] = None

    # Database settings
    vector_db_path: str = Field(default="./chroma_db")

    # Application behavior
    session_expiry_minutes: int = Field(default=60)
    max_history_length: int = Field(default=50)
    similarity_threshold: float = Field(default=0.65)

    # Config replaced with model_config in Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",  # No prefix by default
        extra="ignore",
    )

    @field_validator("google_cse_id")
    @classmethod
    def validate_google_cse_id(cls, v, info):
        """Validate that google_cse_id is provided if using Google search."""
        # In Pydantic v2, we need to use info.data to access other values
        if info.data.get("search_provider") == SearchProvider.GOOGLE and not v:
            raise ValueError("google_cse_id must be provided when using Google search")
        return v


# Create global settings instance
settings = Settings()
