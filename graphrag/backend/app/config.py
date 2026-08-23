"""
Centralized configuration for the GraphRAG backend.
All values are overridable via environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM / embeddings provider ---
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_PROVIDER: str = "local"  # "local" (sentence-transformers) or "openai"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # --- Chroma (vector store) ---
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "child_chunks"

    # --- Chunking ---
    PARENT_CHUNK_TOKENS: int = 1000
    CHILD_CHUNK_TOKENS: int = 200
    CHUNK_OVERLAP_TOKENS: int = 40

    # --- Retrieval ---
    VECTOR_TOP_K: int = 6
    GRAPH_HOPS: int = 2

    # --- API ---
    CORS_ORIGINS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
