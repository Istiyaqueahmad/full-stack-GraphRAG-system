"""
Application-scoped singletons wired up via FastAPI's dependency system.
Instantiated once at app startup and reused across requests.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingService
from app.core.graph_store import Neo4jGraphStore
from app.core.llm import LLMService
from app.core.vector_store import ChromaVectorStore
from app.services.rag_service import RAGService


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(get_settings())


@lru_cache
def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService(get_settings())


@lru_cache
def get_rag_service() -> RAGService:
    settings: Settings = get_settings()
    return RAGService(
        settings=settings,
        vector_store=get_vector_store(),
        graph_store=get_graph_store(),
        embeddings=get_embedding_service(),
        llm=get_llm_service(),
    )
