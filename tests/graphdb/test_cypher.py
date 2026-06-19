"""Shared Cypher escaping contract tests."""

import pytest

from recon_graphrag.graphdb.memgraph.cypher import (
    escape_cypher_identifier as escape_memgraph_identifier,
)
from recon_graphrag.graphdb.neo4j.cypher import (
    escape_cypher_identifier as escape_neo4j_identifier,
)


@pytest.mark.parametrize(
    "escape_identifier",
    [escape_neo4j_identifier, escape_memgraph_identifier],
)
def test_escape_cypher_identifier(escape_identifier):
    assert escape_identifier("Movie") == "`Movie`"
    assert escape_identifier("M`ovie") == "`M``ovie`"
