"""Shared Cypher escaping contract tests."""

from recon_graphrag.graphdb.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)


def test_escape_cypher_identifier():
    assert escape_cypher_identifier("Movie") == "`Movie`"
    assert escape_cypher_identifier("M`ovie") == "`M``ovie`"


def test_cypher_string_literal_escapes_quotes_and_backslashes():
    assert cypher_string_literal("Bob's \\ Movie") == "'Bob\\'s \\\\ Movie'"
