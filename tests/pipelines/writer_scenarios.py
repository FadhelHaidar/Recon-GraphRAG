"""Shared contract scenarios for backend graph writers."""

from __future__ import annotations

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)


class FakeGraphStore:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        return []


def make_graph_document() -> GraphDocument:
    return GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="hash"),
        chunks=[ChunkRecord(id="chunk:1", document_id="doc:1", text="hello", index=0)],
        entities=[
            EntityRecord(id="person:1", type="Person", properties={"name": "Alice"}),
            EntityRecord(id="person:2", type="Person", properties={"name": "Bob"}),
            EntityRecord(id="movie:1", type="Movie", properties={"name": "Inception"}),
        ],
        relationships=[
            RelationshipRecord(
                id="rel:1",
                source_id="person:1",
                target_id="movie:1",
                type="DIRECTED",
            ),
            RelationshipRecord(
                id="rel:2",
                source_id="person:2",
                target_id="movie:1",
                type="ACTED_IN",
            ),
        ],
        evidence_links=[EvidenceLink(chunk_id="chunk:1", entity_id="person:1")],
    )


def assert_writer_stats_and_query_shape(writer_cls) -> None:
    store = FakeGraphStore()
    stats = writer_cls(store).write_graph_document(make_graph_document())

    assert stats == {
        "documents": 1,
        "chunks": 1,
        "entities": 3,
        "relationships": 2,
        "evidence_links": 1,
    }
    query_text = "\n".join(store.queries)
    for fragment in (
        "MERGE (d:Document",
        "MERGE (c:Chunk",
        "MERGE (e:__Entity__:",
        "MERGE (c)-[r:FROM_CHUNK]",
        "MERGE (source)-[r:",
    ):
        assert fragment in query_text


def assert_writer_groups_entities_by_type(writer_cls) -> None:
    store = FakeGraphStore()
    writer_cls(store).write_graph_document(make_graph_document())

    entity_queries = [query for query in store.queries if "MERGE (e:__Entity__" in query]
    assert len(entity_queries) == 2


def assert_writer_groups_relationships_by_type(writer_cls) -> None:
    store = FakeGraphStore()
    writer_cls(store).write_graph_document(make_graph_document())

    relationship_queries = [
        query for query in store.queries if "MERGE (source)-[r:" in query
    ]
    assert len(relationship_queries) == 2
