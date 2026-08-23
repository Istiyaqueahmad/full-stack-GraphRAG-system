"""
Vector store for child chunks, backed by Chroma (persistent, embedded —
no separate service required). Stores parent_id / doc_id as metadata so a
vector hit can always be expanded to its parent context.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.config import Settings
from app.models.schemas import ChildChunk


@dataclass
class VectorHit:
    child_id: str
    parent_id: str
    doc_id: str
    text: str
    score: float


class ChromaVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add_children(self, children: list[ChildChunk], embeddings: list[list[float]]) -> None:
        if not children:
            return
        self._collection.upsert(
            ids=[c.id for c in children],
            embeddings=embeddings,
            documents=[c.text for c in children],
            metadatas=[
                {"parent_id": c.parent_id, "doc_id": c.doc_id, "order": c.order}
                for c in children
            ],
        )

    def query(
        self, query_embedding: list[float], top_k: int, doc_id: str | None = None
    ) -> list[VectorHit]:
        where = {"doc_id": doc_id} if doc_id else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        hits: list[VectorHit] = []
        if not result["ids"] or not result["ids"][0]:
            return hits
        for i in range(len(result["ids"][0])):
            distance = result["distances"][0][i]
            similarity = 1 - distance  # cosine distance -> similarity
            meta = result["metadatas"][0][i]
            hits.append(
                VectorHit(
                    child_id=result["ids"][0][i],
                    parent_id=meta["parent_id"],
                    doc_id=meta["doc_id"],
                    text=result["documents"][0][i],
                    score=float(similarity),
                )
            )
        return hits

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
