"""Application settings, loaded from environment variables or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to an upper-case env variable."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (any OpenAI-compatible server: Ollama by default, vLLM works too)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.1:8b"
    llm_api_key: str = "ollama"
    llm_timeout_s: float = Field(default=120.0, gt=0)

    # Embeddings
    embedder: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Ingestion
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    extraction_max_retries: int = Field(default=2, ge=0, le=5)
    max_upload_bytes: int = Field(default=5_000_000, gt=0)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=50)
    graph_hops: int = Field(default=2, ge=1, le=3)
    graph_entity_limit: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    # Agent / API
    max_iterations: int = Field(default=2, ge=0, le=5)
    query_timeout_s: float = Field(default=180.0, gt=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the cached process-wide settings."""
    return Settings()
