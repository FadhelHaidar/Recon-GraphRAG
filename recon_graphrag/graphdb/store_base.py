"""Shared graph store helpers.

Concrete backends keep their own packages and dialect-specific operations.
This base only contains backend-neutral Cypher-shaped helpers whose return
shape is part of the shared GraphStore contract.
"""

from __future__ import annotations

from recon_graphrag.models.artifacts import (
    CommunityReport,
    report_to_json,
    report_to_text,
)


class BaseGraphStore:
    """Mixin for graph-store methods that are identical across backends."""

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        raise NotImplementedError

    def get_entity_count(self) -> int:
        return self._count("MATCH (e:__Entity__) RETURN count(e) AS cnt")

    def get_chunk_count(self) -> int:
        return self._count("MATCH (c:Chunk) RETURN count(c) AS cnt")

    def get_evidence_link_count(self) -> int:
        return self._count(
            "MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__) RETURN count(r) AS cnt"
        )

    def get_relationship_count(self) -> int:
        return self._count(
            "MATCH (:__Entity__)-[r]-(:__Entity__) RETURN count(r) AS cnt"
        )

    def validate_graph_build(self) -> dict:
        counts = {
            "entity_count": self.get_entity_count(),
            "chunk_count": self.get_chunk_count(),
            "evidence_link_count": self.get_evidence_link_count(),
            "entity_relationship_count": self.get_relationship_count(),
        }

        for key, query in self._extra_validation_count_queries().items():
            counts[key] = self._count(query)
        return counts

    def _extra_validation_count_queries(self) -> dict[str, str]:
        return {}

    def _count(self, query: str) -> int:
        result = self.execute_query(query)
        return result[0]["cnt"] if result else 0

    def get_community_stats(self, graph_name: str) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, count(DISTINCT e) AS entity_count
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
        WITH c, entity_count, count(DISTINCT child) AS child_community_count
        RETURN c.id AS community_id,
               c.level AS level,
               entity_count,
               child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.execute_query(query, {"graph_name": graph_name})

    def store_community_summary(
        self,
        community_id: str,
        level: int,
        summary: str,
        graph_name: str,
    ) -> None:
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.summary = $summary,
            c.embedding = NULL,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "summary": summary,
            },
        )

    def store_community_report(
        self,
        report: CommunityReport,
        graph_name: str,
    ) -> None:
        report_text = report_to_text(report)
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.report_json = $report_json,
            c.report_text = $report_text,
            c.title = $title,
            c.summary = $report_text,
            c.rating = $rating,
            c.rating_explanation = $rating_explanation,
            c.report_status = 'success',
            c.report_error = NULL,
            c.schema_version = $schema_version,
            c.prompt_version = $prompt_version,
            c.input_fingerprint = $input_fingerprint,
            c.embedding = NULL,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": report.community_id,
                "level": report.level,
                "report_json": report_to_json(report),
                "report_text": report_text,
                "title": report.title,
                "rating": report.rating,
                "rating_explanation": report.rating_explanation,
                "schema_version": report.version.schema_version,
                "prompt_version": report.version.prompt_version,
                "input_fingerprint": report.version.input_fingerprint,
            },
        )

    def mark_community_report_failed(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        error: str,
    ) -> None:
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.report_status = 'failed',
            c.report_error = $error,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "error": error,
            },
        )
