"""Opt-in Neo4j integration tests for community detection."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from examples.config import get_neo4j_store


RUN_FLAG = "RUN_NEO4J_COMMUNITY_INTEGRATION_TESTS"
GRAPH_NAME = "community-weight-integration"
REQUIRED_NEO4J_ENV = [
    "NEO4J_URL",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]


def _neo4j_env_or_skip() -> None:
    load_dotenv()

    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to run Neo4j community detection tests.")

    missing = [name for name in REQUIRED_NEO4J_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required Neo4j env vars: {', '.join(missing)}")


def _preflight_or_fail(store) -> None:
    checks = [
        ("Neo4j connectivity", "RETURN 1 AS ok"),
        ("APOC", "RETURN apoc.version() AS version"),
        ("GDS", "RETURN gds.version() AS version"),
    ]
    for label, query in checks:
        try:
            store.execute_query(query)
        except Exception as exc:
            pytest.fail(f"{label} preflight failed: {exc}")


def _cleanup_graph(store) -> None:
    store.execute_query(
        """
        MATCH (n {graph_name: $graph_name})
        DETACH DELETE n
        """,
        {"graph_name": GRAPH_NAME},
    )
    store.execute_query(
        "CALL gds.graph.drop($graph_name, false)",
        {"graph_name": GRAPH_NAME},
    )


def _seed_weighted_graph(store) -> None:
    store.execute_query(
        """
        UNWIND $nodes AS row
        MERGE (e:__Entity__:TestEntity {id: row.id})
        SET e.graph_name = $graph_name,
            e.name = row.name,
            e.description = ''
        """,
        {
            "graph_name": GRAPH_NAME,
            "nodes": [
                {"id": "community-weight:a", "name": "A"},
                {"id": "community-weight:b", "name": "B"},
                {"id": "community-weight:c", "name": "C"},
                {"id": "community-weight:d", "name": "D"},
            ],
        },
    )
    store.execute_query(
        """
        UNWIND $relationships AS row
        MATCH (source:__Entity__ {id: row.source_id, graph_name: $graph_name})
        MATCH (target:__Entity__ {id: row.target_id, graph_name: $graph_name})
        MERGE (source)-[r:RELATED_TO]->(target)
        SET r.graph_name = $graph_name,
            r.weight = row.weight
        """,
        {
            "graph_name": GRAPH_NAME,
            "relationships": [
                {
                    "source_id": "community-weight:a",
                    "target_id": "community-weight:b",
                    "weight": 10.0,
                },
                {
                    "source_id": "community-weight:c",
                    "target_id": "community-weight:d",
                    "weight": 10.0,
                },
                {
                    "source_id": "community-weight:b",
                    "target_id": "community-weight:c",
                    "weight": 0.1,
                },
            ],
        },
    )


@pytest.fixture
def neo4j_store():
    _neo4j_env_or_skip()
    store = get_neo4j_store()
    _preflight_or_fail(store)
    _cleanup_graph(store)
    try:
        yield store
    finally:
        _cleanup_graph(store)
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()


@pytest.mark.integration
def test_neo4j_leiden_uses_configured_relationship_weight_property(neo4j_store):
    _seed_weighted_graph(neo4j_store)

    stats = neo4j_store.detect_communities(
        graph_name=GRAPH_NAME,
        relationship_types=["RELATED_TO"],
        relationship_weight_property="weight",
        max_levels=2,
        random_seed=42,
    )

    assert stats

    memberships = neo4j_store.execute_query(
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->(
            c:Community {graph_name: $graph_name}
        )
        RETURN count(*) AS count
        """,
        {"graph_name": GRAPH_NAME},
    )
    assert memberships[0]["count"] >= 4
