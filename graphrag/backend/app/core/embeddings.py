"""
Embedding backend abstraction. Two providers are supported:

- "local": sentence-transformers, runs on-box, no API key required.
- "openai": OpenAI embeddings API — higher quality, needs OPENAI_API_KEY.

Both are exposed through the same `EmbeddingService.embed(texts)` interface
so the rest of the app never needs to know which one is active.
"""
from __future__ import annotations

from app.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._backend = settings.EMBEDDING_PROVIDER
        self._model = None
        self._openai_client = None

        if self._backend == "openai":
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)

    @property
    def dimension(self) -> int:
        if self._backend == "openai":
            # text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
            return 1536 if "small" in self.settings.OPENAI_EMBEDDING_MODEL else 3072
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend == "openai":
            resp = self._openai_client.embeddings.create(
                model=self.settings.OPENAI_EMBEDDING_MODEL, input=texts
            )
            return [d.embedding for d in resp.data]

        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
