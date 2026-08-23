"""
Thin LLM wrapper around the OpenAI-compatible chat completions API.

Two responsibilities:
1. Structured extraction of entities/relationships (`extract`) using JSON
   schema-constrained output.
2. Streaming answer generation (`stream_answer`) for the chat endpoint.

Any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, local vLLM/Ollama
gateways, etc.) can be used by pointing `base_url` at it via env vars.
"""
from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.config import Settings
from app.models.schemas import ExtractionResult

_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extraction engine for a knowledge graph.
Given a passage of text, extract:
- entities: distinct named entities (people, organizations, products, places, events, concepts)
- relationships: explicit, typed relationships between those entities, e.g. (Company)-[ACQUIRED]->(Startup)

Rules:
- Only extract relationships that are explicitly stated or strongly implied in the text.
- Relationship `type` must be a short SCREAMING_SNAKE_CASE verb phrase (e.g. ACQUIRED, FOUNDED_BY, PARTNERED_WITH).
- `source` and `target` must exactly match an entity `name` you extracted.
- Do not invent facts not supported by the text.
- Return strictly valid JSON matching the provided schema, nothing else.
"""

_ANSWER_SYSTEM_PROMPT = """\
You are a GraphRAG assistant. Answer the user's question using ONLY the
provided context (retrieved parent chunks and knowledge graph relationship
triples). If the context is insufficient, say so plainly rather than
guessing. Cite which context you used implicitly by staying grounded in it;
the calling application will render explicit citation cards separately.
"""


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def extract(self, text: str) -> ExtractionResult:
        response = self._client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "schema": ExtractionResult.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=0,
        )
        content = response.choices[0].message.content
        return ExtractionResult.model_validate_json(content)

    def stream_answer(self, query: str, context: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def extract_candidate_mentions(self, query: str) -> list[str]:
        """Cheap heuristic seed-entity extraction from the user query (capitalized spans),
        used to anchor graph traversal before falling back to vector-only context."""
        import re

        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9&\.]*(?:\s+[A-Z][a-zA-Z0-9&\.]*)*\b", query)
        # de-dup while preserving order, drop very short/common tokens
        seen = set()
        result = []
        for c in candidates:
            c = c.strip()
            if len(c) > 2 and c.lower() not in seen:
                seen.add(c.lower())
                result.append(c)
        return result
