"""Opt-in movie workflow smoke test for Neo4j."""

from __future__ import annotations

import pytest

from examples.config import get_embedder, get_llm, get_neo4j_store
from recon_graphrag import IndexManager
from tests.integration.movie_smoke_support import (
    cleanup_graph,
    close_resources,
    run_movie_smoke,
)
from tests.integration.support import (
    require_integration_env,
    require_selected_provider_env,
)


RUN_FLAG = "RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS"
GRAPH_NAME = "neo4j-movie-smoke"
REQUIRED_ENV = [
    "NEO4J_URL",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]


def preflight_neo4j(store) -> None:
    for label, query in (
        ("Neo4j connectivity", "RETURN 1 AS ok"),
        ("APOC", "RETURN apoc.version() AS version"),
        ("GDS", "RETURN gds.version() AS version"),
    ):
        try:
            store.execute_query(query)
        except Exception as exc:
            pytest.fail(f"{label} preflight failed: {exc}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_movie_example_with_selected_providers():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Neo4j movie example smoke test",
        fail_on_missing=True,
    )
    llm_provider, embedder_provider = require_selected_provider_env(
        "Neo4j movie example smoke test"
    )
    store = get_neo4j_store()
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)

    try:
        preflight_neo4j(store)
        cleanup_graph(store, GRAPH_NAME)
        await run_movie_smoke(
            store=store,
            index_manager_cls=IndexManager,
            llm=llm,
            embedder=embedder,
            graph_name=GRAPH_NAME,
        )
    finally:
        try:
            cleanup_graph(store, GRAPH_NAME)
        finally:
            await close_resources(store, llm)
