"""
LangGraph agent orchestrating multi-hop GraphRAG retrieval.

Graph of nodes:
    retrieve_vector  -> expand_to_parents -> retrieve_graph_seed
        -> should_expand_hops? --(yes)--> retrieve_graph_more_hops --loop-->
                              --(no)---> build_context -> END

The "should_expand_hops" edge is a conditional edge: if the first-hop
subgraph is too sparse (few triples) and hop budget remains, the agent
widens the traversal before handing off context to the LLM. This is the
"multi-hop" behavior called for in the assignment, implemented as an
explicit, inspectable state machine rather than a single fixed-depth query.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.core.embeddings import EmbeddingService
from app.core.graph_store import GraphTripleRow, Neo4jGraphStore
from app.core.llm import LLMService
from app.core.vector_store import ChromaVectorStore, VectorHit


class AgentState(TypedDict, total=False):
    query: str
    doc_id: str | None
    top_k: int
    max_hops: int
    current_hop: int
    vector_hits: list[VectorHit]
    parent_ids: list[str]
    seed_entities: list[str]
    triples: list[GraphTripleRow]
    done: bool


class GraphRAGAgent:
    """Wraps a compiled LangGraph state machine for retrieval; generation/streaming
    of the final answer is handled by the caller (chat API) once context is assembled,
    so tokens can be streamed to the client as they're produced."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        graph_store: Neo4jGraphStore,
        embeddings: EmbeddingService,
        llm: LLMService,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embeddings = embeddings
        self.llm = llm
        self._graph = self._build_graph()

    # ------------------------------ nodes -----------------------------------

    def _retrieve_vector(self, state: AgentState) -> AgentState:
        query_embedding = self.embeddings.embed_query(state["query"])
        hits = self.vector_store.query(
            query_embedding, top_k=state["top_k"], doc_id=state.get("doc_id")
        )
        parent_ids = list(dict.fromkeys(h.parent_id for h in hits))  # de-dup, preserve order
        return {"vector_hits": hits, "parent_ids": parent_ids}

    def _seed_graph_entities(self, state: AgentState) -> AgentState:
        candidates = self.llm.extract_candidate_mentions(state["query"])
        seeds = self.graph_store.find_seed_entities(candidates)
        return {"seed_entities": seeds, "current_hop": 1}

    def _retrieve_graph(self, state: AgentState) -> AgentState:
        if not state.get("seed_entities"):
            return {"triples": [], "done": True}
        triples = self.graph_store.n_hop_subgraph(
            state["seed_entities"], hops=state["current_hop"]
        )
        return {"triples": triples}

    def _should_expand(self, state: AgentState) -> str:
        """Conditional edge: widen the hop radius if the subgraph came back sparse
        and we still have hop budget, otherwise proceed to context assembly."""
        sparse = len(state.get("triples", [])) < 3
        can_expand = state["current_hop"] < state["max_hops"]
        if sparse and can_expand and state.get("seed_entities"):
            return "expand"
        return "finish"

    def _expand_hops(self, state: AgentState) -> AgentState:
        return {"current_hop": state["current_hop"] + 1}

    def _finish(self, state: AgentState) -> AgentState:
        return {"done": True}

    # ------------------------------ wiring -----------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_vector", self._retrieve_vector)
        graph.add_node("seed_graph_entities", self._seed_graph_entities)
        graph.add_node("retrieve_graph", self._retrieve_graph)
        graph.add_node("expand_hops", self._expand_hops)
        graph.add_node("finish", self._finish)

        graph.set_entry_point("retrieve_vector")
        graph.add_edge("retrieve_vector", "seed_graph_entities")
        graph.add_edge("seed_graph_entities", "retrieve_graph")
        graph.add_conditional_edges(
            "retrieve_graph", self._should_expand, {"expand": "expand_hops", "finish": "finish"}
        )
        graph.add_edge("expand_hops", "retrieve_graph")
        graph.add_edge("finish", END)

        return graph.compile()

    # ------------------------------ public API --------------------------------

    def run(
        self, query: str, doc_id: str | None, top_k: int, max_hops: int
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "query": query,
            "doc_id": doc_id,
            "top_k": top_k,
            "max_hops": max_hops,
            "current_hop": 0,
            "vector_hits": [],
            "parent_ids": [],
            "seed_entities": [],
            "triples": [],
            "done": False,
        }
        final_state = self._graph.invoke(initial_state)
        return final_state
