"""
Neo4j-backed knowledge graph store.

Schema
------
(:Entity {name, type, description})
(:ParentChunk {id, doc_id, order})
(:ChildChunk  {id, doc_id, order})

(:ChildChunk)-[:CHILD_OF]->(:ParentChunk)
(:Entity)-[:MENTIONED_IN]->(:ParentChunk)         -- provenance
(:Entity)-[:<RELATIONSHIP_TYPE> {description}]->(:Entity)   -- typed, explicit edges

Relationship *types* are dynamic (e.g. ACQUIRED, FOUNDED_BY), so they are
interpolated into Cypher as a validated, sanitized label rather than bound
as a parameter (Neo4j does not support parameterized relationship types).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from neo4j import GraphDatabase

from app.config import Settings
from app.models.schemas import Entity, ParentChunk, Relationship

_SAFE_REL_TYPE = re.compile(r"[^A-Z0-9_]")


def _sanitize_rel_type(rel_type: str) -> str:
    """Coerce arbitrary LLM output into a safe Cypher relationship type token."""
    cleaned = rel_type.strip().upper().replace(" ", "_").replace("-", "_")
    cleaned = _SAFE_REL_TYPE.sub("", cleaned)
    return cleaned or "RELATED_TO"


@dataclass
class GraphTripleRow:
    source: str
    source_type: str
    relationship: str
    target: str
    target_type: str


class Neo4jGraphStore:
    def __init__(self, settings: Settings):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        self._ensure_constraints()

    def close(self) -> None:
        self._driver.close()

    def _ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT parent_id IF NOT EXISTS "
                "FOR (p:ParentChunk) REQUIRE p.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT child_id IF NOT EXISTS "
                "FOR (c:ChildChunk) REQUIRE c.id IS UNIQUE"
            )

    # ------------------------------ writes ---------------------------------

    def upsert_parent_chunk(self, parent: ParentChunk) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (p:ParentChunk {id: $id})
                SET p.doc_id = $doc_id, p.order = $order, p.text = $text
                """,
                id=parent.id,
                doc_id=parent.doc_id,
                order=parent.order,
                text=parent.text,
            )

    def get_parent_text(self, parent_id: str) -> str:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (p:ParentChunk {id: $id}) RETURN p.text AS text", id=parent_id
            )
            record = result.single()
            return record["text"] if record and record["text"] else ""

    def upsert_child_chunk(self, child_id: str, parent_id: str, doc_id: str, order: int) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:ChildChunk {id: $child_id})
                SET c.doc_id = $doc_id, c.order = $order
                WITH c
                MATCH (p:ParentChunk {id: $parent_id})
                MERGE (c)-[:CHILD_OF]->(p)
                """,
                child_id=child_id,
                parent_id=parent_id,
                doc_id=doc_id,
                order=order,
            )

    def upsert_entity(self, entity: Entity, parent_chunk_id: str) -> None:
        """Create/merge an entity node and link its provenance to the parent chunk it was extracted from."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $description
                ON MATCH SET e.description = coalesce(e.description, $description)
                WITH e
                MATCH (p:ParentChunk {id: $parent_chunk_id})
                MERGE (e)-[:MENTIONED_IN]->(p)
                """,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                parent_chunk_id=parent_chunk_id,
            )

    def upsert_relationship(self, rel: Relationship, parent_chunk_id: str) -> None:
        rel_type = _sanitize_rel_type(rel.type)
        query = f"""
            MERGE (s:Entity {{name: $source}})
            MERGE (t:Entity {{name: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.description = $description, r.source_chunk = $parent_chunk_id
        """
        with self._driver.session() as session:
            session.run(
                query,
                source=rel.source,
                target=rel.target,
                description=rel.description,
                parent_chunk_id=parent_chunk_id,
            )

    def delete_document(self, doc_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:ParentChunk {doc_id: $doc_id})
                OPTIONAL MATCH (c:ChildChunk {doc_id: $doc_id})
                DETACH DELETE p, c
                """,
                doc_id=doc_id,
            )

    # ------------------------------- reads ----------------------------------

    def find_seed_entities(self, mention_candidates: list[str]) -> list[str]:
        """Fuzzy-ish match candidate strings (from query text) against known entity names."""
        if not mention_candidates:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $candidates AS candidate
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(candidate)
                   OR toLower(candidate) CONTAINS toLower(e.name)
                RETURN DISTINCT e.name AS name
                LIMIT 10
                """,
                candidates=mention_candidates,
            )
            return [r["name"] for r in result]

    def n_hop_subgraph(self, seed_entities: list[str], hops: int) -> list[GraphTripleRow]:
        """Expand outward from seed entities up to `hops` relationship hops."""
        if not seed_entities:
            return []
        hops = max(1, min(hops, 4))
        query = f"""
            MATCH (s:Entity)
            WHERE s.name IN $seeds
            MATCH path = (s)-[r*1..{hops}]-(t:Entity)
            UNWIND relationships(path) AS rel
            WITH startNode(rel) AS src, endNode(rel) AS tgt, type(rel) AS rel_type
            WHERE rel_type <> 'MENTIONED_IN' AND rel_type <> 'CHILD_OF'
            RETURN DISTINCT
                src.name AS source, src.type AS source_type,
                rel_type AS relationship,
                tgt.name AS target, tgt.type AS target_type
            LIMIT 200
        """
        with self._driver.session() as session:
            result = session.run(query, seeds=seed_entities)
            return [
                GraphTripleRow(
                    source=r["source"],
                    source_type=r["source_type"] or "UNKNOWN",
                    relationship=r["relationship"],
                    target=r["target"],
                    target_type=r["target_type"] or "UNKNOWN",
                )
                for r in result
            ]

    def full_subgraph_for_doc(self, doc_id: str) -> list[GraphTripleRow]:
        query = """
            MATCH (e1:Entity)-[:MENTIONED_IN]->(p:ParentChunk {doc_id: $doc_id})
            WITH collect(DISTINCT e1.name) AS names
            MATCH (s:Entity)-[rel]->(t:Entity)
            WHERE s.name IN names AND t.name IN names
              AND type(rel) <> 'MENTIONED_IN' AND type(rel) <> 'CHILD_OF'
            RETURN DISTINCT
                s.name AS source, s.type AS source_type,
                type(rel) AS relationship,
                t.name AS target, t.type AS target_type
            LIMIT 500
        """
        with self._driver.session() as session:
            result = session.run(query, doc_id=doc_id)
            return [
                GraphTripleRow(
                    source=r["source"],
                    source_type=r["source_type"] or "UNKNOWN",
                    relationship=r["relationship"],
                    target=r["target"],
                    target_type=r["target_type"] or "UNKNOWN",
                )
                for r in result
            ]
