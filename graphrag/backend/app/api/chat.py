import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse, SSEEventType
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _sse(event: SSEEventType, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event.value}\ndata: {payload}\n\n"


def _sse_generator(rag: RAGService, request: ChatRequest):
    vector_hits, triples = rag.retrieve(
        query=request.query,
        doc_id=request.doc_id,
        top_k=request.top_k,
        graph_hops=request.graph_hops,
    )
    context, citations = rag.build_context(vector_hits, triples)
    graph_triples = rag.triples_to_schema(triples)

    # Send retrieved context up front so the UI can render citation cards
    # and the graph inspector while tokens are still streaming in.
    yield _sse(SSEEventType.CITATIONS, {"citations": [c.model_dump() for c in citations]})
    yield _sse(SSEEventType.GRAPH, {"graph_triples": [t.model_dump() for t in graph_triples]})

    full_answer = []
    try:
        for token in rag.stream_answer(request.query, context):
            full_answer.append(token)
            yield _sse(SSEEventType.TOKEN, {"token": token})
    except Exception as exc:  # pragma: no cover - defensive
        yield _sse(SSEEventType.ERROR, {"detail": str(exc)})
        return

    yield _sse(SSEEventType.DONE, {"answer": "".join(full_answer)})


@router.post("/chat")
def chat(request: ChatRequest, rag: RAGService = Depends(get_rag_service)):
    """
    Streams the answer via Server-Sent Events when `stream=True` (default).
    Set `stream=False` to receive a single consolidated JSON response instead
    (useful for testing or non-streaming clients).
    """
    if not request.stream:
        vector_hits, triples = rag.retrieve(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            graph_hops=request.graph_hops,
        )
        context, citations = rag.build_context(vector_hits, triples)
        answer = "".join(rag.stream_answer(request.query, context))
        return ChatResponse(
            answer=answer,
            citations=citations,
            graph_triples=rag.triples_to_schema(triples),
        )

    return StreamingResponse(
        _sse_generator(rag, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
