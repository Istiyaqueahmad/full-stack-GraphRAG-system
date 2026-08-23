from app.core.chunking import HierarchicalChunker


def test_parent_child_hierarchy_links_correctly():
    chunker = HierarchicalChunker(
        parent_chunk_tokens=50, child_chunk_tokens=10, overlap_tokens=2
    )
    text = " ".join(f"word{i}" for i in range(300))
    parents, children = chunker.chunk("doc1", text)

    assert len(parents) > 1
    assert len(children) > len(parents)

    parent_ids = {p.id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.id.startswith(child.parent_id)
        assert child.doc_id == "doc1"


def test_single_short_document_produces_one_parent_and_child():
    chunker = HierarchicalChunker(
        parent_chunk_tokens=1000, child_chunk_tokens=200, overlap_tokens=40
    )
    parents, children = chunker.chunk("doc2", "just a few words here")

    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].parent_id == parents[0].id


def test_empty_document_produces_no_chunks():
    chunker = HierarchicalChunker()
    parents, children = chunker.chunk("doc3", "")
    assert parents == []
    assert children == []


def test_relationship_type_sanitization():
    from app.core.graph_store import _sanitize_rel_type

    assert _sanitize_rel_type("acquired") == "ACQUIRED"
    assert _sanitize_rel_type("founded-by") == "FOUNDED_BY"
    assert _sanitize_rel_type("weird; DROP TABLE") == "WEIRD_DROP_TABLE"
    assert _sanitize_rel_type("") == "RELATED_TO"
