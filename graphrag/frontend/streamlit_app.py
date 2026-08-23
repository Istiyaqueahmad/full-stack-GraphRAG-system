"""
GraphRAG chat frontend.

- Left: document ingestion panel (paste text, ingest, check status).
- Main: chat interface with token-by-token streaming, expandable citation
  cards showing retrieved parent chunks, and an expandable "Graph triples"
  panel showing the relationship paths used to ground the answer.
- Sidebar tab: interactive graph inspector rendering the sub-graph for the
  active document via streamlit-agraph (falls back to a simple table if the
  optional dependency isn't installed).
"""
import json
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="GraphRAG Chat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations, graph_triples}
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None


# ------------------------------- sidebar: ingest -------------------------------

with st.sidebar:
    st.header("📄 Ingest a document")
    doc_id = st.text_input("Document ID", value="doc1")
    doc_text = st.text_area("Paste document text", height=200)
    col_a, col_b = st.columns(2)
    with col_a:
        ingest_clicked = st.button("Ingest (sync)", use_container_width=True)
    with col_b:
        check_clicked = st.button("Check status", use_container_width=True)

    if ingest_clicked and doc_text.strip():
        with st.spinner("Chunking, embedding, and extracting graph relationships..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/v1/ingest/sync",
                    json={"doc_id": doc_id, "text": doc_text},
                    timeout=300,
                )
                resp.raise_for_status()
                st.session_state.active_doc_id = doc_id
                st.success(f"Ingested: {resp.json()}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    if check_clicked:
        try:
            resp = requests.get(f"{BACKEND_URL}/api/v1/ingest/{doc_id}/status", timeout=10)
            st.info(resp.json())
        except Exception as e:
            st.error(f"Status check failed: {e}")

    st.divider()
    st.header("🔎 Retrieval scope")
    restrict_to_doc = st.checkbox("Restrict chat to this doc_id", value=False)
    top_k = st.slider("Vector top_k", 1, 15, 6)
    graph_hops = st.slider("Max graph hops", 1, 4, 2)

    st.divider()
    st.header("🕸️ Graph Inspector")
    if st.button("Load subgraph for this document", use_container_width=True):
        try:
            resp = requests.get(
                f"{BACKEND_URL}/api/v1/graph/subgraph",
                params={"doc_id": doc_id},
                timeout=30,
            )
            resp.raise_for_status()
            st.session_state["subgraph"] = resp.json()
        except Exception as e:
            st.error(f"Could not load subgraph: {e}")

    subgraph = st.session_state.get("subgraph")
    if subgraph:
        try:
            from streamlit_agraph import Config, Edge, Node, agraph

            nodes = [Node(id=n["id"], label=n["label"], size=20) for n in subgraph["nodes"]]
            edges = [Edge(source=e["source"], target=e["target"], label=e["type"]) for e in subgraph["edges"]]
            config = Config(width=350, height=400, directed=True, physics=True, hierarchical=False)
            agraph(nodes=nodes, edges=edges, config=config)
        except ImportError:
            st.warning("Install `streamlit-agraph` for interactive graph rendering. Showing raw data:")
            st.json(subgraph)


# ------------------------------- main: chat -------------------------------

st.title("💬 GraphRAG Chat")
st.caption("Hybrid vector + knowledge-graph retrieval, powered by a LangGraph multi-hop agent.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 Citations ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['parent_chunk_id']}** · score `{c['score']:.3f}`")
                    st.text(c["text"])
                    st.divider()
        if msg.get("graph_triples"):
            with st.expander(f"🕸️ Graph relationships used ({len(msg['graph_triples'])})"):
                for t in msg["graph_triples"]:
                    st.markdown(f"`({t['source']}) -[{t['relationship']}]-> ({t['target']})`")

query = st.chat_input("Ask a question about your ingested documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_text = ""
        citations: list[dict] = []
        graph_triples: list[dict] = []

        payload = {
            "query": query,
            "doc_id": doc_id if restrict_to_doc else None,
            "top_k": top_k,
            "graph_hops": graph_hops,
            "stream": True,
        }

        try:
            with requests.post(
                f"{BACKEND_URL}/api/v1/chat", json=payload, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                event_type = None
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None or raw_line == "":
                        continue
                    if raw_line.startswith("event:"):
                        event_type = raw_line.split(":", 1)[1].strip()
                    elif raw_line.startswith("data:"):
                        data = json.loads(raw_line.split(":", 1)[1].strip())
                        if event_type == "token":
                            answer_text += data["token"]
                            placeholder.markdown(answer_text + "▌")
                        elif event_type == "citations":
                            citations = data["citations"]
                        elif event_type == "graph":
                            graph_triples = data["graph_triples"]
                        elif event_type == "error":
                            st.error(data["detail"])
                placeholder.markdown(answer_text)
        except Exception as e:
            st.error(f"Chat request failed: {e}")

        if citations:
            with st.expander(f"📚 Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(f"**{c['parent_chunk_id']}** · score `{c['score']:.3f}`")
                    st.text(c["text"])
                    st.divider()
        if graph_triples:
            with st.expander(f"🕸️ Graph relationships used ({len(graph_triples)})"):
                for t in graph_triples:
                    st.markdown(f"`({t['source']}) -[{t['relationship']}]-> ({t['target']})`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "graph_triples": graph_triples,
        }
    )
