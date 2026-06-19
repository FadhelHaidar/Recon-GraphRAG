"""Shared entity-resolution normalization contract tests."""

import pytest

from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _normalize_name as normalize_memgraph_name,
)
from recon_graphrag.graphdb.neo4j.entity_resolution import (
    _normalize_name as normalize_neo4j_name,
)


@pytest.mark.parametrize(
    "normalize_name",
    [normalize_neo4j_name, normalize_memgraph_name],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("OpenAI", "openai"),
        ("Open AI", "openai"),
        ("U.S.A.", "usa"),
        ("Microsoft Corp.", "microsoft"),
        ("Microsoft Corporation", "microsoft"),
        ("Acme Inc.", "acme"),
        ("Acme Ltd", "acme"),
    ],
)
def test_normalize_name_contract(normalize_name, value, expected):
    assert normalize_name(value) == expected
