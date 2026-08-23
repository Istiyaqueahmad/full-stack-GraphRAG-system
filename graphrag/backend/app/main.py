from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, graph, ingest
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="GraphRAG API",
    description=(
        "Full-stack GraphRAG service: hierarchical parent-child chunking, "
        "LLM-based entity/relationship extraction into Neo4j, hybrid "
        "vector + graph retrieval via a LangGraph multi-hop agent, and "
        "SSE-streamed chat responses."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(graph.router)


@app.get("/health")
def health():
    return {"status": "ok"}
