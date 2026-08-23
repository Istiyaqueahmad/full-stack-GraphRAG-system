from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.dependencies import get_rag_service
from app.models.schemas import IngestRequest, IngestResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["ingest"])

# In-memory job status tracking for async ingestion (swap for Redis/DB in production).
_JOB_STATUS: dict[str, str] = {}


def _run_ingest_job(rag: RAGService, doc_id: str, text: str) -> None:
    _JOB_STATUS[doc_id] = "processing"
    try:
        rag.ingest_document(doc_id, text)
        _JOB_STATUS[doc_id] = "completed"
    except Exception as exc:  # pragma: no cover - defensive logging path
        _JOB_STATUS[doc_id] = f"failed: {exc}"


@router.post("/ingest", response_model=dict)
def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    rag: RAGService = Depends(get_rag_service),
):
    """
    Asynchronously ingest a document: hierarchical chunking, vector store
    population, and graph extraction all run in a background task so the
    caller gets an immediate 202-style response with a job id to poll.
    """
    if request.doc_id in _JOB_STATUS and _JOB_STATUS[request.doc_id] == "processing":
        raise HTTPException(status_code=409, detail="Document is already being ingested")

    _JOB_STATUS[request.doc_id] = "queued"
    background_tasks.add_task(_run_ingest_job, rag, request.doc_id, request.text)
    return {"doc_id": request.doc_id, "status": "queued"}


@router.post("/ingest/sync", response_model=IngestResponse)
def ingest_document_sync(
    request: IngestRequest, rag: RAGService = Depends(get_rag_service)
):
    """Synchronous variant — useful for tests/small documents/demos."""
    return rag.ingest_document(request.doc_id, request.text)


@router.get("/ingest/{doc_id}/status")
def ingest_status(doc_id: str):
    status = _JOB_STATUS.get(doc_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    return {"doc_id": doc_id, "status": status}


@router.delete("/ingest/{doc_id}")
def delete_document(doc_id: str, rag: RAGService = Depends(get_rag_service)):
    rag.vector_store.delete_document(doc_id)
    rag.graph_store.delete_document(doc_id)
    _JOB_STATUS.pop(doc_id, None)
    return {"doc_id": doc_id, "status": "deleted"}
