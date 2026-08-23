"""
RAGService orchestrates ingestion (chunk -> embed -> vector store -> extract
-> graph store) and chat (agent retrieval -> context assembly -> streamed
generation). This is the seam the API layer talks to; it has no HTTP
concerns of its own.
"""
from __future__ import annotations

from collections.abc import Iterator

from app.agents.graph_rag_agent import GraphRAGAgent
from app.config import Settings
from app.core.chunking import HierarchicalChunker
from app.core.embeddings import EmbeddingService
from app.core.graph_store import GraphTripleRow, Neo4jGraphStore
from app.core.llm import LLMService
from app.core.vector_store import ChromaVectorStore, VectorHit
from app.models.schemas import (
    Citation,
    GraphTriple,
    IngestResponse,
    SubgraphEdge,
    SubgraphNode,
    SubgraphResponse,
)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        vector_store: ChromaVectorStore,
        graph_store: Neo4jGraphStore,
        embeddings: EmbeddingService,
        llm: LLMService,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embeddings = embeddings
        self.llm = llm
        self.chunker = HierarchicalChunker(
            parent_chunk_tokens=settings.PARENT_CHUNK_TOKENS,
            child_chunk_tokens=settings.CHILD_CHUNK_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        )
        self.agent = GraphRAGAgent(vector_store, graph_store, embeddings, llm)

    # ------------------------------- ingest -----------------------------------

    def ingest_document(self, doc_id: str, text: str) -> IngestResponse:
        parents, children = self.chunker.chunk(doc_id, text)

        # 1. persist chunk hierarchy in the graph (for provenance traversal)
        for parent in parents:
            self.graph_store.upsert_parent_chunk(parent)
        for child in children:
            self.graph_store.upsert_child_chunk(
                child.id, child.parent_id, child.doc_id, child.order
            )

        # 2. embed + store children in the vector store
        child_embeddings = self.embeddings.embed([c.text for c in children])
        self.vector_store.add_children(children, child_embeddings)

        # 3. extract entities/relationships per parent chunk, linked by provenance
        num_entities = 0
        num_relationships = 0
        for parent in parents:
            extraction = self.llm.extract(parent.text)
            for entity in extraction.entities:
                self.graph_store.upsert_entity(entity, parent.id)
                num_entities += 1
            for rel in extraction.relationships:
                self.graph_store.upsert_relationship(rel, parent.id)
                num_relationships += 1

        return IngestResponse(
            doc_id=doc_id,
            num_parent_chunks=len(parents),
            num_child_chunks=len(children),
            num_entities=num_entities,
            num_relationships=num_relationships,
        )

    # -------------------------------- chat -------------------------------------

    def retrieve(
        self,
        query: str,
        doc_id: str | None,
        top_k: int | None,
        graph_hops: int | None,
    ) -> tuple[list[VectorHit], list[GraphTripleRow]]:
        state = self.agent.run(
            query=query,
            doc_id=doc_id,
            top_k=top_k or self.settings.VECTOR_TOP_K,
            max_hops=graph_hops or self.settings.GRAPH_HOPS,
        )
        return state.get("vector_hits", []), state.get("triples", [])

    def build_context(
        self, vector_hits: list[VectorHit], triples: list[GraphTripleRow]
    ) -> tuple[str, list[Citation]]:
        # Expand child hits to unique parent chunk text for LLM context
        seen_parents: dict[str, str] = {}
        citations: list[Citation] = []
        for hit in vector_hits:
            parent_text = self._get_parent_text_cached(hit.parent_id, seen_parents)
            citations.append(
                Citation(
                    parent_chunk_id=hit.parent_id,
                    child_chunk_id=hit.child_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    score=round(hit.score, 4),
                )
            )

        context_parts = ["## Retrieved passages"]
        for pid, text in seen_parents.items():
            context_parts.append(f"[{pid}]\n{text}")

        if triples:
            context_parts.append("\n## Knowledge graph relationships")
            for t in triples:
                context_parts.append(f"({t.source})-[{t.relationship}]->({t.target})")

        return "\n\n".join(context_parts), citations

    def _get_parent_text_cached(self, parent_id: str, cache: dict[str, str]) -> str:
        if parent_id in cache:
            return cache[parent_id]
        text = self.graph_store.get_parent_text(parent_id) if hasattr(
            self.graph_store, "get_parent_text"
        ) else ""
        cache[parent_id] = text
        return text

    def stream_answer(self, query: str, context: str) -> Iterator[str]:
        yield from self.llm.stream_answer(query, context)

    @staticmethod
    def triples_to_schema(triples: list[GraphTripleRow]) -> list[GraphTriple]:
        return [
            GraphTriple(
                source=t.source,
                relationship=t.relationship,
                target=t.target,
                source_type=t.source_type,
                target_type=t.target_type,
            )
            for t in triples
        ]

    def get_subgraph_for_doc(self, doc_id: str) -> SubgraphResponse:
        triples = self.graph_store.full_subgraph_for_doc(doc_id)
        nodes: dict[str, SubgraphNode] = {}
        edges: list[SubgraphEdge] = []
        for t in triples:
            nodes.setdefault(t.source, SubgraphNode(id=t.source, label=t.source, type=t.source_type))
            nodes.setdefault(t.target, SubgraphNode(id=t.target, label=t.target, type=t.target_type))
            edges.append(SubgraphEdge(source=t.source, target=t.target, type=t.relationship))
        return SubgraphResponse(nodes=list(nodes.values()), edges=edges)

    def get_subgraph_for_query(self, seed_entities: list[str], hops: int) -> SubgraphResponse:
        triples = self.graph_store.n_hop_subgraph(seed_entities, hops)
        nodes: dict[str, SubgraphNode] = {}
        edges: list[SubgraphEdge] = []
        for t in triples:
            nodes.setdefault(t.source, SubgraphNode(id=t.source, label=t.source, type=t.source_type))
            nodes.setdefault(t.target, SubgraphNode(id=t.target, label=t.target, type=t.target_type))
            edges.append(SubgraphEdge(source=t.source, target=t.target, type=t.relationship))
        return SubgraphResponse(nodes=list(nodes.values()), edges=edges)
