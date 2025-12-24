"""Configuration management for Recaller."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RECALLER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Notion
    notion_token: str = Field(description="Notion integration token")
    notion_page_id: str = Field(description="ID of the 'Recaller' parent page in Notion")

    # Ollama LLM
    ollama_host: str = Field(
        default="https://ollama.com",
        description="Ollama API host",
    )
    ollama_model: str = Field(
        default="gpt-oss:120b-cloud",
        description="Ollama model to use for generation",
    )
    ollama_api_key: str = Field(
        default="",
        description="Ollama API key (default: from OLLAMA_API_KEY env var)",
    )

    # Gemini (optional, fallback)
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key (optional if using Ollama)",
    )

    # Embeddings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )
    similarity_threshold: float = Field(
        default=0.78,
        ge=0.5,
        le=0.95,
        description="Cosine similarity threshold for merging notes",
    )

    # Database
    database_path: Path = Field(
        default=Path("data/recaller.db"),
        description="Path to SQLite database file",
    )

    # Anki
    ankiconnect_url: str = Field(
        default="http://localhost:8765",
        description="URL for AnkiConnect API",
    )

    # Flashcard generation
    cards_per_note_min: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum flashcards to generate per note",
    )
    cards_per_note_max: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum flashcards to generate per note",
    )

    @property
    def database_url(self) -> str:
        """SQLAlchemy database URL."""
        return f"sqlite:///{self.database_path}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
