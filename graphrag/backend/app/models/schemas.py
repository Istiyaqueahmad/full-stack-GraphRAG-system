"""
Pydantic data contracts shared between API, services, and agents.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ----------------------------- Chunking -----------------------------------

class ParentChunk(BaseModel):
    id: str
    doc_id: str
    order: int
    text: str


class ChildChunk(BaseModel):
    id: str
    parent_id: str
    doc_id: str
    order: int
    text: str


# --------------------------- Graph modeling --------------------------------

class Entity(BaseModel):
    name: str = Field(..., description="Canonical entity name, e.g. 'Acme Corp'")
    type: str = Field(..., description="Entity type/label, e.g. PERSON, ORG, PRODUCT")
    description: Optional[str] = Field(
        default=None, description="One-sentence description grounded in the source text"
    )


class Relationship(BaseModel):
    source: str = Field(..., description="Name of the source entity")
    target: str = Field(..., description="Name of the target entity")
    type: str = Field(..., description="Relationship type in SCREAMING_SNAKE_CASE, e.g. ACQUIRED")
    description: Optional[str] = Field(
        default=None, description="Short justification for the relationship, grounded in text"
    )


class ExtractionResult(BaseModel):
    """Structured output the LLM must produce for a given parent chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ------------------------------- Ingest -------------------------------------

class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Caller-provided unique id for the document")
    text: str = Field(..., description="Raw document text to ingest")
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    num_parent_chunks: int
    num_child_chunks: int
    num_entities: int
    num_relationships: int


# -------------------------------- Chat --------------------------------------

class ChatRequest(BaseModel):
    query: str
    doc_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single document, if provided"
    )
    top_k: Optional[int] = None
    graph_hops: Optional[int] = None
    stream: bool = True


class Citation(BaseModel):
    parent_chunk_id: str
    child_chunk_id: str
    doc_id: str
    text: str
    score: float


class GraphTriple(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: Optional[str] = None
    target_type: Optional[str] = None


class ChatResponse(BaseModel):
    """Used for the non-streaming variant / final SSE event payload."""

    answer: str
    citations: list[Citation]
    graph_triples: list[GraphTriple]


# ------------------------------ Graph API ------------------------------------

class SubgraphNode(BaseModel):
    id: str
    label: str
    type: str


class SubgraphEdge(BaseModel):
    source: str
    target: str
    type: str


class SubgraphResponse(BaseModel):
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


class SSEEventType(str, Enum):
    TOKEN = "token"
    CITATIONS = "citations"
    GRAPH = "graph"
    DONE = "done"
    ERROR = "error"
