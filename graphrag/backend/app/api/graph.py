from fastapi import APIRouter, Depends, Query

from app.dependencies import get_rag_service
from app.models.schemas import SubgraphResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["graph"])


@router.get("/graph/subgraph", response_model=SubgraphResponse)
def get_subgraph(
    doc_id: str | None = Query(default=None, description="Return the full subgraph for a document"),
    entity: list[str] | None = Query(
        default=None, description="Seed entity name(s) to expand N hops from"
    ),
    hops: int = Query(default=2, ge=1, le=4),
    rag: RAGService = Depends(get_rag_service),
):
    """
    Returns node/edge JSON for graph visualization. Provide either `doc_id`
    (full document subgraph) or one or more `entity` values (N-hop expansion
    from those seeds) — the same shape the chat endpoint's graph inspector
    context is built from.
    """
    if doc_id:
        return rag.get_subgraph_for_doc(doc_id)
    return rag.get_subgraph_for_query(entity or [], hops)
