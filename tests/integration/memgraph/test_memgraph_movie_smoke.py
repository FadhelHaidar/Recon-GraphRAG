"""Opt-in movie workflow smoke test for Memgraph."""

from __future__ import annotations

import pytest

from examples.config import get_embedder, get_llm, get_memgraph_store
from recon_graphrag.graphdb.memgraph.index_manager import (
    IndexManager as MemgraphIndexManager,
)
from tests.integration.movie_smoke_support import (
    cleanup_graph,
    close_resources,
    run_movie_smoke,
)
from tests.integration.support import (
    require_integration_env,
    require_selected_provider_env,
)


RUN_FLAG = "RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS"
GRAPH_NAME = "memgraph-movie-smoke"
REQUIRED_ENV = [
    "MEMGRAPH_URL",
]


def preflight_memgraph(store) -> None:
    try:
        store.execute_query("RETURN 1 AS ok")
    except Exception as exc:
        pytest.fail(f"Memgraph connectivity preflight failed: {exc}")

    try:
        procedures = store.execute_query(
            "CALL mg.procedures() YIELD name RETURN name"
        )
    except Exception as exc:
        pytest.fail(f"Memgraph MAGE preflight failed: {exc}")
    if not any(
        row.get("name") == "leiden_community_detection.get_subgraph"
        for row in procedures
    ):
        pytest.fail("Memgraph MAGE Leiden procedure is not available.")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memgraph_movie_example_with_selected_providers():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph movie example smoke test",
        fail_on_missing=True,
    )
    llm_provider, embedder_provider = require_selected_provider_env(
        "Memgraph movie example smoke test"
    )
    store = get_memgraph_store()
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)

    try:
        preflight_memgraph(store)
        cleanup_graph(store, GRAPH_NAME)
        await run_movie_smoke(
            store=store,
            index_manager_cls=MemgraphIndexManager,
            llm=llm,
            embedder=embedder,
            graph_name=GRAPH_NAME,
        )
    finally:
        try:
            cleanup_graph(store, GRAPH_NAME)
        finally:
            await close_resources(store, llm)
