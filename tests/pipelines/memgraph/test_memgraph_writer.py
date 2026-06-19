"""Contract tests for MemgraphGraphWriter."""

from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter
from tests.pipelines.writer_scenarios import (
    assert_writer_groups_entities_by_type,
    assert_writer_groups_relationships_by_type,
    assert_writer_stats_and_query_shape,
)


def test_memgraph_writer_returns_stats_and_issues_all_writes():
    assert_writer_stats_and_query_shape(MemgraphGraphWriter)


def test_memgraph_writer_groups_entities_by_type():
    assert_writer_groups_entities_by_type(MemgraphGraphWriter)


def test_memgraph_writer_groups_relationships_by_type():
    assert_writer_groups_relationships_by_type(MemgraphGraphWriter)
