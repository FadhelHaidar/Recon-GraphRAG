"""Tests for token usage tracking: ledger, scopes, wrapper, leak proofs."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from recon_graphrag.llm.base import BaseLLM, LLMResponse, LLMUsage
from recon_graphrag.observability import (
    TokenUsageEvent,
    TokenUsageLedger,
    UsageTrackingLLM,
    get_default_ledger,
    reset_default_ledger,
    run_scope,
    token_stage,
    track_usage,
    usage_delta,
    usage_snapshot,
)
from recon_graphrag.observability.usage import _current_run_id, _current_stage


class FakeLLM:
    model_name = "fake-model"

    def __init__(self, usage: LLMUsage | None | str = "default"):
        self.usage = LLMUsage(10, 5, 15) if usage == "default" else usage
        self.prompts: list[str] = []
        self.closed = False

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content="pong", usage=self.usage)

    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse:
        return self.invoke(prompt, **kwargs)

    def supports_structured_output(self) -> bool:
        return False

    async def aclose(self) -> None:
        self.closed = True


def _event(run_id="r1", stage="s1", input_tokens=10, output_tokens=5, estimated=False):
    return TokenUsageEvent(
        ts=0.0,
        run_id=run_id,
        stage=stage,
        model="m",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated=estimated,
        latency_ms=1,
    )


class TestLedger:
    def test_per_stage_and_run_totals_exact(self):
        ledger = TokenUsageLedger()
        ledger.record(_event(run_id="r1", stage="a", input_tokens=100, output_tokens=20))
        ledger.record(_event(run_id="r1", stage="a", input_tokens=50, output_tokens=10))
        ledger.record(_event(run_id="r1", stage="b", input_tokens=7, output_tokens=3, estimated=True))
        ledger.record(_event(run_id="r2", stage="a", input_tokens=1, output_tokens=1))

        summary = ledger.run_summary("r1")
        assert summary["run_id"] == "r1"
        assert summary["totals"] == {
            "calls": 3,
            "input_tokens": 157,
            "output_tokens": 33,
            "total_tokens": 190,
            "estimated_calls": 1,
        }
        assert summary["stages"]["a"]["calls"] == 2
        assert summary["stages"]["a"]["total_tokens"] == 180
        assert summary["stages"]["b"]["estimated_calls"] == 1
        # total = input + output
        assert (
            summary["totals"]["total_tokens"]
            == summary["totals"]["input_tokens"] + summary["totals"]["output_tokens"]
        )

    def test_unknown_run_returns_none(self):
        assert TokenUsageLedger().run_summary("nope") is None

    def test_rollup_across_runs(self):
        ledger = TokenUsageLedger()
        ledger.record(_event(run_id="r1", stage="a"))
        ledger.record(_event(run_id="r2", stage="a"))
        ledger.record(_event(run_id="r2", stage="b"))

        rollup = ledger.rollup()
        assert rollup["runs"] == 2
        assert rollup["totals"]["calls"] == 3
        assert rollup["stages"]["a"]["calls"] == 2
        assert rollup["stages"]["b"]["calls"] == 1

    def test_sink_failure_swallowed(self):
        received = []

        def bad_sink(event):
            raise RuntimeError("boom")

        ledger = TokenUsageLedger(sinks=[bad_sink, received.append])
        ledger.record(_event())
        assert len(received) == 1
        assert ledger.run_summary("r1")["totals"]["calls"] == 1

    def test_forget_removes_run(self):
        ledger = TokenUsageLedger()
        ledger.record(_event(run_id="r1"))
        assert ledger.run_summary("r1") is not None
        ledger.forget("r1")
        assert ledger.run_summary("r1") is None

    def test_forget_unknown_run_is_noop(self):
        TokenUsageLedger().forget("nope")  # must not raise


class TestScopes:
    def test_token_stage_resets(self):
        assert _current_stage.get() == "unknown"
        with token_stage("construction.extract"):
            assert _current_stage.get() == "construction.extract"
        assert _current_stage.get() == "unknown"

    def test_token_stage_resets_on_exception(self):
        with pytest.raises(ValueError):
            with token_stage("x"):
                raise ValueError()
        assert _current_stage.get() == "unknown"

    def test_run_scope_generates_and_resets(self):
        assert _current_run_id.get() is None
        with run_scope() as scope:
            assert scope.id
            assert _current_run_id.get() == scope.id
        assert _current_run_id.get() is None

    def test_run_scope_resets_on_exception(self):
        with pytest.raises(ValueError):
            with run_scope():
                raise ValueError()
        assert _current_run_id.get() is None

    def test_nested_run_scope_reuses_outer(self):
        with run_scope("outer") as outer:
            with run_scope() as inner:
                assert inner.id == outer.id == "outer"
                assert _current_run_id.get() == "outer"
            assert _current_run_id.get() == "outer"

    def test_nested_explicit_run_id_overrides_then_restores(self):
        with run_scope("outer"):
            with run_scope("inner") as inner:
                assert inner.id == "inner"
                assert _current_run_id.get() == "inner"
            assert _current_run_id.get() == "outer"


class TestDefaultLedgerAutoForget:
    """The library doesn't retain history: a run is forgotten from the
    *default* ledger the moment its outermost run_scope exits, since by then
    every caller has already read its own usage_snapshot()."""

    def setup_method(self):
        reset_default_ledger()

    def teardown_method(self):
        reset_default_ledger()

    def test_outermost_scope_exit_forgets_default_ledger(self):
        llm = UsageTrackingLLM(FakeLLM())  # default ledger
        with run_scope("r1"):
            llm.invoke("p")
            assert get_default_ledger().run_summary("r1") is not None
        assert get_default_ledger().run_summary("r1") is None

    def test_nested_scope_exit_does_not_forget(self):
        llm = UsageTrackingLLM(FakeLLM())
        with run_scope("outer"):
            with run_scope():  # reuses "outer"; not the creator, no forget on exit
                llm.invoke("p")
            assert get_default_ledger().run_summary("outer") is not None
        assert get_default_ledger().run_summary("outer") is None

    def test_explicit_ledger_untouched_by_scope_exit(self):
        own_ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=own_ledger)
        with run_scope("r1"):
            llm.invoke("p")
        assert own_ledger.run_summary("r1") is not None
        assert get_default_ledger().run_summary("r1") is None

    def test_multiple_reads_within_same_scope_both_work(self):
        """Regression: some callers read usage_snapshot twice for one run
        before the scope exits (e.g. a printed summary, then the returned
        result) -- forgetting must happen on exit, not on read."""
        llm = UsageTrackingLLM(FakeLLM())
        with run_scope("r1"):
            llm.invoke("p")
            first = get_default_ledger().run_summary("r1")
            second = get_default_ledger().run_summary("r1")
            assert first == second is not None


class TestWrapper:
    def test_records_per_stage_and_run(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=ledger)
        with run_scope("r1"):
            with token_stage("construction.extract"):
                llm.invoke("p")
            with token_stage("community.report"):
                asyncio.run(llm.ainvoke("p"))

        summary = ledger.run_summary("r1")
        assert summary["totals"]["calls"] == 2
        assert summary["stages"]["construction.extract"]["input_tokens"] == 10
        assert summary["stages"]["community.report"]["output_tokens"] == 5
        assert summary["totals"]["estimated_calls"] == 0

    def test_no_active_run_records_nothing(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=ledger)
        llm.invoke("p")
        assert ledger.rollup()["totals"]["calls"] == 0

    def test_unlabeled_call_lands_in_unknown(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("p")
        assert "unknown" in ledger.run_summary("r1")["stages"]

    def test_passthrough_and_delegation(self):
        inner = FakeLLM()
        llm = UsageTrackingLLM(inner)
        response = llm.invoke("p")
        assert response.content == "pong"
        assert response.usage is inner.usage
        assert llm.model_name == "fake-model"
        assert llm.supports_structured_output() is False
        asyncio.run(llm.aclose())
        assert inner.closed
        assert isinstance(llm, BaseLLM)

    def test_track_usage_idempotent(self):
        llm = track_usage(FakeLLM())
        assert isinstance(llm, UsageTrackingLLM)
        assert track_usage(llm) is llm

    def test_missing_usage_falls_back_to_estimate(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(usage=None), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("some prompt text")
        totals = ledger.run_summary("r1")["totals"]
        assert totals["estimated_calls"] == 1
        assert totals["input_tokens"] > 0

    def test_no_tiktoken_records_zero_estimated(self, monkeypatch):
        import recon_graphrag.observability.usage as usage_mod

        class Broken:
            def __init__(self):
                raise ImportError("tiktoken missing")

        monkeypatch.setattr(usage_mod, "TiktokenTokenCounter", Broken)
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(usage=None), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("p")
        totals = ledger.run_summary("r1")["totals"]
        assert totals == {
            "calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_calls": 1,
        }

    def test_partial_usage_estimates_only_the_missing_side(self):
        ledger = TokenUsageLedger()
        usage = LLMUsage(request_tokens=5, response_tokens=None)
        llm = UsageTrackingLLM(FakeLLM(usage=usage), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("some prompt text")
        totals = ledger.run_summary("r1")["totals"]
        assert totals["input_tokens"] == 5  # trusted, not re-estimated
        assert totals["output_tokens"] > 0  # estimated
        assert totals["estimated_calls"] == 1

    def test_total_tokens_only_falls_back_to_full_estimate(self):
        ledger = TokenUsageLedger()
        usage = LLMUsage(request_tokens=None, response_tokens=None, total_tokens=500)
        llm = UsageTrackingLLM(FakeLLM(usage=usage), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("some prompt text")
        totals = ledger.run_summary("r1")["totals"]
        assert totals["estimated_calls"] == 1
        assert totals["input_tokens"] > 0
        assert totals["output_tokens"] > 0

    def test_explicit_zero_usage_is_trusted_not_estimated(self):
        ledger = TokenUsageLedger()
        usage = LLMUsage(request_tokens=0, response_tokens=0)
        llm = UsageTrackingLLM(FakeLLM(usage=usage), ledger=ledger)
        with run_scope("r1"):
            llm.invoke("p")
        totals = ledger.run_summary("r1")["totals"]
        assert totals["input_tokens"] == 0
        assert totals["output_tokens"] == 0
        assert totals["estimated_calls"] == 0


class TestLeakProofs:
    async def test_asyncio_tasks_isolated(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=ledger)

        async def worker(i: int):
            with run_scope(f"run-{i}"), token_stage(f"stage-{i % 5}"):
                for _ in range(3):
                    await llm.ainvoke("p")
                    await asyncio.sleep(0)  # force interleaving
            assert _current_run_id.get() is None
            assert _current_stage.get() == "unknown"

        await asyncio.gather(*(worker(i) for i in range(50)))

        for i in range(50):
            summary = ledger.run_summary(f"run-{i}")
            assert summary["totals"]["calls"] == 3
            assert summary["totals"]["input_tokens"] == 30
            assert set(summary["stages"]) == {f"stage-{i % 5}"}

    def test_threads_isolated_and_reused_threads_start_clean(self):
        ledger = TokenUsageLedger()
        llm = UsageTrackingLLM(FakeLLM(), ledger=ledger)

        def worker(i: int):
            # Reused pool threads must start from defaults.
            assert _current_run_id.get() is None
            assert _current_stage.get() == "unknown"
            with run_scope(f"thread-{i}"), token_stage("sync"):
                for _ in range(2):
                    llm.invoke("p")
            assert _current_run_id.get() is None

        # 8 tasks over 4 workers forces thread reuse.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(8)))

        for i in range(8):
            summary = ledger.run_summary(f"thread-{i}")
            assert summary["totals"]["calls"] == 2
            assert summary["stages"]["sync"]["input_tokens"] == 20


class TestHelpers:
    def test_usage_snapshot_untracked_or_no_run(self):
        assert usage_snapshot(FakeLLM()) is None
        llm = UsageTrackingLLM(FakeLLM(), ledger=TokenUsageLedger())
        assert usage_snapshot(llm) is None  # no active run

    def test_usage_snapshot_active_run(self):
        llm = UsageTrackingLLM(FakeLLM(), ledger=TokenUsageLedger())
        with run_scope("r1"):
            llm.invoke("p")
            snapshot = usage_snapshot(llm)
        assert snapshot["run_id"] == "r1"
        assert snapshot["totals"]["calls"] == 1

    def test_usage_delta(self):
        before = {
            "totals": {"calls": 1, "input_tokens": 10, "output_tokens": 5,
                       "total_tokens": 15, "estimated_calls": 0},
            "stages": {"a": {"calls": 1, "input_tokens": 10, "output_tokens": 5,
                             "total_tokens": 15, "estimated_calls": 0}},
        }
        after = {
            "totals": {"calls": 3, "input_tokens": 30, "output_tokens": 15,
                       "total_tokens": 45, "estimated_calls": 1},
            "stages": {
                "a": {"calls": 2, "input_tokens": 20, "output_tokens": 10,
                      "total_tokens": 30, "estimated_calls": 0},
                "b": {"calls": 1, "input_tokens": 10, "output_tokens": 5,
                      "total_tokens": 15, "estimated_calls": 1},
            },
        }
        delta = usage_delta(before, after)
        assert delta["totals"]["calls"] == 2
        assert delta["stages"]["a"]["input_tokens"] == 10
        assert delta["stages"]["b"]["calls"] == 1

    def test_usage_delta_none_and_clamping(self):
        assert usage_delta(None, None)["totals"]["calls"] == 0
        clamped = usage_delta(
            {"totals": {"calls": 5}, "stages": {}},
            {"totals": {"calls": 2}, "stages": {}},
        )
        assert clamped["totals"]["calls"] == 0
