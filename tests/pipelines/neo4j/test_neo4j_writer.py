"""Contract tests for Neo4jGraphWriter."""

from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter
from tests.pipelines.writer_scenarios import (
    assert_writer_groups_entities_by_type,
    assert_writer_groups_relationships_by_type,
    assert_writer_stats_and_query_shape,
)


def test_neo4j_writer_returns_stats_and_issues_all_writes():
    assert_writer_stats_and_query_shape(Neo4jGraphWriter)


def test_neo4j_writer_groups_entities_by_type():
    assert_writer_groups_entities_by_type(Neo4jGraphWriter)


def test_neo4j_writer_groups_relationships_by_type():
    assert_writer_groups_relationships_by_type(Neo4jGraphWriter)
