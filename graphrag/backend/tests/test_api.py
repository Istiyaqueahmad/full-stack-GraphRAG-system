"""
API-level tests. All external systems (Neo4j, Chroma, OpenAI) are mocked out
via a fake RAGService injected through FastAPI's dependency_overrides, so
these tests run with no network access and no real credentials.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.graph_store import GraphTripleRow
from app.core.vector_store import VectorHit
from app.dependencies import get_rag_service
from app.main import app
from app.models.schemas import Citation, IngestResponse


class FakeRAGService:
    def __init__(self):
        self.vector_store = MagicMock()
        self.graph_store = MagicMock()

    def ingest_document(self, doc_id, text):
        return IngestResponse(
            doc_id=doc_id,
            num_parent_chunks=1,
            num_child_chunks=3,
            num_entities=2,
            num_relationships=1,
        )

    def retrieve(self, query, doc_id, top_k, graph_hops):
        hits = [
            VectorHit(
                child_id="doc1::p0::c0",
                parent_id="doc1::p0",
                doc_id="doc1",
                text="Acme Corp acquired Startup Inc in 2024.",
                score=0.92,
            )
        ]
        triples = [
            GraphTripleRow(
                source="Acme Corp",
                source_type="ORG",
                relationship="ACQUIRED",
                target="Startup Inc",
                target_type="ORG",
            )
        ]
        return hits, triples

    def build_context(self, vector_hits, triples):
        citations = [
            Citation(
                parent_chunk_id=h.parent_id,
                child_chunk_id=h.child_id,
                doc_id=h.doc_id,
                text=h.text,
                score=h.score,
            )
            for h in vector_hits
        ]
        context = "Acme Corp acquired Startup Inc in 2024."
        return context, citations

    def stream_answer(self, query, context):
        for tok in ["Acme ", "Corp ", "acquired ", "Startup Inc."]:
            yield tok

    @staticmethod
    def triples_to_schema(triples):
        from app.services.rag_service import RAGService

        return RAGService.triples_to_schema(triples)

    def get_subgraph_for_doc(self, doc_id):
        from app.models.schemas import SubgraphEdge, SubgraphNode, SubgraphResponse

        return SubgraphResponse(
            nodes=[
                SubgraphNode(id="Acme Corp", label="Acme Corp", type="ORG"),
                SubgraphNode(id="Startup Inc", label="Startup Inc", type="ORG"),
            ],
            edges=[SubgraphEdge(source="Acme Corp", target="Startup Inc", type="ACQUIRED")],
        )

    def get_subgraph_for_query(self, seed_entities, hops):
        return self.get_subgraph_for_doc("doc1")


@pytest.fixture
def client():
    fake = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_sync(client):
    resp = client.post(
        "/api/v1/ingest/sync", json={"doc_id": "doc1", "text": "Acme Corp acquired Startup Inc."}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc1"
    assert body["num_entities"] == 2


def test_ingest_async_queues_job(client):
    resp = client.post("/api/v1/ingest", json={"doc_id": "doc2", "text": "some text"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_chat_non_streaming(client):
    resp = client.post(
        "/api/v1/chat", json={"query": "Who acquired Startup Inc?", "stream": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Acme" in body["answer"]
    assert body["citations"][0]["doc_id"] == "doc1"
    assert body["graph_triples"][0]["relationship"] == "ACQUIRED"


def test_chat_streaming_sse(client):
    with client.stream(
        "POST", "/api/v1/chat", json={"query": "Who acquired Startup Inc?", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: citations" in body
    assert "event: graph" in body
    assert "event: token" in body
    assert "event: done" in body


def test_subgraph_by_doc(client):
    resp = client.get("/api/v1/graph/subgraph", params={"doc_id": "doc1"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["nodes"]) == 2
    assert body["edges"][0]["type"] == "ACQUIRED"
