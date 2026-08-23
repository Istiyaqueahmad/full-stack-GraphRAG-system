"""
Hierarchical (small-to-big) chunking.

Strategy
--------
1. Split the document into *parent* chunks of ~PARENT_CHUNK_TOKENS tokens.
   These are the blocks passed to the LLM as generation context and as the
   unit of entity/relationship extraction.
2. Split each parent chunk further into *child* chunks of ~CHILD_CHUNK_TOKENS
   tokens. Children are what gets embedded and searched against — small
   enough for precise similarity matching, but always traceable back to
   their parent for context expansion.

Token counting uses tiktoken when available and falls back to a simple
whitespace-based approximation so the module has no hard runtime dependency
on a specific encoding being downloadable.
"""
from __future__ import annotations

from app.models.schemas import ChildChunk, ParentChunk

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _tokenize(text: str) -> list[str]:
        return [str(t) for t in _ENC.encode(text)]

    def _detokenize(tokens: list[str]) -> str:
        return _ENC.decode([int(t) for t in tokens])

except Exception:  # pragma: no cover - fallback path
    def _tokenize(text: str) -> list[str]:
        return text.split()

    def _detokenize(tokens: list[str]) -> str:
        return " ".join(tokens)


def _split_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    step = max(chunk_tokens - overlap_tokens, 1)
    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_tokens]
        chunks.append(_detokenize(window))
        if i + chunk_tokens >= len(tokens):
            break
        i += step
    return chunks


class HierarchicalChunker:
    def __init__(
        self,
        parent_chunk_tokens: int = 1000,
        child_chunk_tokens: int = 200,
        overlap_tokens: int = 40,
    ):
        self.parent_chunk_tokens = parent_chunk_tokens
        self.child_chunk_tokens = child_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc_id: str, text: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parent_texts = _split_by_tokens(
            text, self.parent_chunk_tokens, overlap_tokens=self.overlap_tokens
        )

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for p_order, p_text in enumerate(parent_texts):
            parent_id = f"{doc_id}::p{p_order}"
            parents.append(
                ParentChunk(id=parent_id, doc_id=doc_id, order=p_order, text=p_text)
            )

            child_texts = _split_by_tokens(
                p_text, self.child_chunk_tokens, overlap_tokens=self.overlap_tokens // 2
            )
            for c_order, c_text in enumerate(child_texts):
                child_id = f"{parent_id}::c{c_order}"
                children.append(
                    ChildChunk(
                        id=child_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        order=c_order,
                        text=c_text,
                    )
                )

        return parents, children
