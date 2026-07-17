"""Reporting and persistence helpers for token usage observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from recon_graphrag.observability.usage import TokenUsageEvent, TokenTotals


def render_usage_table(summary: dict | None) -> str:
    """Render an ASCII per-stage token usage table from a run summary.

    If ``summary`` is None, returns an empty string.
    """
    if summary is None:
        return ""

    run_id = summary.get("run_id", "unknown")
    totals = summary.get("totals", {})
    stages = summary.get("stages", {})

    rows = []
    for stage in sorted(stages):
        stage_totals = stages[stage]
        rows.append(
            (
                stage,
                stage_totals.get("calls", 0),
                stage_totals.get("input_tokens", 0),
                stage_totals.get("output_tokens", 0),
                stage_totals.get("total_tokens", 0),
                stage_totals.get("estimated_calls", 0),
            )
        )
    rows.append(
        (
            "TOTAL",
            totals.get("calls", 0),
            totals.get("input_tokens", 0),
            totals.get("output_tokens", 0),
            totals.get("total_tokens", 0),
            totals.get("estimated_calls", 0),
        )
    )

    # Column widths
    stage_width = max(len(str(row[0])) for row in rows)
    stage_width = max(stage_width, len("stage"))
    lines = [
        f"token usage (run {run_id}):",
        "  "
        + " ".join(
            [
                "stage".ljust(stage_width),
                "calls".rjust(6),
                "input".rjust(8),
                "output".rjust(8),
                "total".rjust(9),
                "est".rjust(4),
            ]
        ),
    ]
    for stage, calls, input_t, output_t, total_t, est in rows:
        lines.append(
            "  "
            + " ".join(
                [
                    stage.ljust(stage_width),
                    f"{calls:,}".rjust(6),
                    f"{input_t:,}".rjust(8),
                    f"{output_t:,}".rjust(8),
                    f"{total_t:,}".rjust(9),
                    f"{est}".rjust(4),
                ]
            )
        )
    return "\n".join(lines)


class JsonlUsageSink:
    """Append-only JSONL sink for token usage events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = False

    def __call__(self, event: TokenUsageEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def load_usage_events(path: str | Path) -> list[TokenUsageEvent]:
    """Load usage events from a JSONL file written by ``JsonlUsageSink``."""
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            events.append(
                TokenUsageEvent(
                    ts=data["ts"],
                    run_id=data["run_id"],
                    stage=data["stage"],
                    model=data.get("model"),
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                    estimated=data["estimated"],
                    latency_ms=data["latency_ms"],
                )
            )
    return events


def summarize_events(events: Iterable[TokenUsageEvent]) -> dict:
    """Roll up a list of events into per-run/per-stage totals.

    The returned dict has the same shape as ``TokenUsageLedger.rollup()``:
    ``{"runs": int, "totals": {...}, "stages": {...}}``.
    """
    runs: set[str] = set()
    totals = TokenTotals()
    stages: dict[str, TokenTotals] = {}
    for event in events:
        runs.add(event.run_id)
        totals.add(event)
        stage_totals = stages.setdefault(event.stage, TokenTotals())
        stage_totals.add(event)
    return {
        "runs": len(runs),
        "totals": totals.to_dict(),
        "stages": {stage: stage_totals.to_dict() for stage, stage_totals in stages.items()},
    }


__all__ = [
    "render_usage_table",
    "JsonlUsageSink",
    "load_usage_events",
    "summarize_events",
]
