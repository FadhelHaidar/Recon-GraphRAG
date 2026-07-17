"""Tests for JSONL persistence and usage table rendering."""

from __future__ import annotations

from recon_graphrag.observability import (
    JsonlUsageSink,
    TokenUsageEvent,
    TokenUsageLedger,
    load_usage_events,
    render_usage_table,
    summarize_events,
)


def _event(run_id="r1", stage="s1", input_tokens=10, output_tokens=5):
    return TokenUsageEvent(
        ts=1784254600.12,
        run_id=run_id,
        stage=stage,
        model="fake-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated=False,
        latency_ms=42,
    )


def test_jsonl_sink_round_trip(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = TokenUsageLedger(sinks=[JsonlUsageSink(path)])
    events = [
        _event(run_id="r1", stage="construction.extract"),
        _event(run_id="r1", stage="community.report", input_tokens=100, output_tokens=50),
        _event(run_id="r2", stage="drift.primer"),
    ]
    for event in events:
        ledger.record(event)

    loaded = load_usage_events(path)
    assert loaded == events


def test_summarize_events_matches_ledger_shapes(tmp_path):
    events = [
        _event(run_id="r1", stage="a"),
        _event(run_id="r1", stage="a"),
        _event(run_id="r2", stage="b", input_tokens=1, output_tokens=2),
    ]
    rollup = summarize_events(events)
    assert rollup["runs"] == 2
    assert rollup["totals"]["calls"] == 3
    assert rollup["totals"]["total_tokens"] == 33
    assert rollup["stages"]["a"]["calls"] == 2
    assert rollup["stages"]["b"]["input_tokens"] == 1


def test_render_usage_table_smoke():
    summary = {
        "run_id": "9f3c",
        "totals": {"calls": 3, "input_tokens": 120, "output_tokens": 15,
                   "total_tokens": 135, "estimated_calls": 1},
        "stages": {
            "construction.extract": {"calls": 2, "input_tokens": 110,
                                     "output_tokens": 10, "total_tokens": 120,
                                     "estimated_calls": 0},
            "community.report": {"calls": 1, "input_tokens": 10,
                                 "output_tokens": 5, "total_tokens": 15,
                                 "estimated_calls": 1},
        },
    }
    table = render_usage_table(summary)
    assert "token usage (run 9f3c)" in table
    assert "construction.extract" in table
    assert "community.report" in table
    assert "TOTAL" in table
    assert "135" in table


def test_render_usage_table_none():
    assert render_usage_table(None) == ""
