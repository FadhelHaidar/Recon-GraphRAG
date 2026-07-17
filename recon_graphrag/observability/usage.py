"""Token-usage instrumentation for LLM calls.

Tracks provider-reported input/output tokens across Recon-GraphRAG stages
using a transparent ``UsageTrackingLLM`` wrapper, context-scoped stage labels,
and a thread-safe in-memory ledger with optional JSONL persistence.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from recon_graphrag.llm.base import BaseLLM, LLMResponse
from recon_graphrag.utils.tokens import TiktokenTokenCounter

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageEvent:
    """One recorded LLM invocation."""

    ts: float
    run_id: str
    stage: str
    model: str | None
    input_tokens: int
    output_tokens: int
    estimated: bool
    latency_ms: int

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "run_id": self.run_id,
            "stage": self.stage,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated": self.estimated,
            "latency_ms": self.latency_ms,
        }


@dataclass
class TokenTotals:
    """Running totals for a run or stage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_calls: int = 0

    def add(self, event: TokenUsageEvent) -> None:
        self.calls += 1
        self.input_tokens += event.input_tokens
        self.output_tokens += event.output_tokens
        self.total_tokens += event.input_tokens + event.output_tokens
        if event.estimated:
            self.estimated_calls += 1

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_calls": self.estimated_calls,
        }


# Context-scoped observability state.
_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "observability_stage", default="unknown"
)
_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "observability_run_id", default=None
)


@dataclass
class _ScopeToken:
    """Holder for contextvar reset tokens."""

    stage: contextvars.Token[str] | None = None
    run_id: contextvars.Token[Optional[str]] | None = None


class token_stage:
    """Context manager that sets the current observability stage.

    Resets the stage variable in ``finally`` so callers never leak labels
    across async tasks or thread reuse.
    """

    def __init__(self, stage: str):
        self.stage = stage
        self._token: contextvars.Token[str] | None = None

    def __enter__(self) -> token_stage:
        self._token = _current_stage.set(self.stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _current_stage.reset(self._token)


class run_scope:
    """Context manager that establishes a run boundary for token accounting.

    Generates a new ``run_id`` when no run is active and ``run_id`` is None.
    If a run is already active and no explicit ``run_id`` is passed, the
    existing run is reused (outermost-wins nesting).
    """

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        self._active_run_id: str | None = None
        self._token: contextvars.Token[Optional[str]] | None = None
        self._reset_on_exit = False

    def __enter__(self) -> run_scope:
        active = _current_run_id.get()
        if active is not None and self.run_id is None:
            self._active_run_id = active
            return self
        self._active_run_id = self.run_id or str(uuid.uuid4())
        self._token = _current_run_id.set(self._active_run_id)
        self._reset_on_exit = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._reset_on_exit and self._token is not None:
            _current_run_id.reset(self._token)
            if self._active_run_id is not None:
                get_default_ledger().forget(self._active_run_id)

    @property
    def id(self) -> str | None:
        return self._active_run_id


class TokenUsageLedger:
    """Thread-safe in-memory ledger of per-(run_id, stage) token totals.

    Optional ``sinks`` receive a copy of every recorded event; sink errors
    are logged and swallowed so they never affect the LLM call path.
    """

    def __init__(
        self,
        sinks: Optional[list[Callable[[TokenUsageEvent], None]]] = None,
    ):
        self._lock = threading.Lock()
        # run_id -> {stage: TokenTotals}
        self._data: dict[str, dict[str, TokenTotals]] = {}
        self.sinks = sinks or []

    def record(self, event: TokenUsageEvent) -> None:
        with self._lock:
            stages = self._data.setdefault(event.run_id, {})
            totals = stages.setdefault(event.stage, TokenTotals())
            totals.add(event)
        for sink in self.sinks:
            try:
                sink(event)
            except Exception as exc:
                logger.exception("usage sink failed: %s", exc)

    def run_summary(self, run_id: str) -> dict | None:
        with self._lock:
            stages = self._data.get(run_id)
            if not stages:
                return None
            totals = TokenTotals()
            stage_dict: dict[str, dict] = {}
            for stage, stage_totals in stages.items():
                totals.calls += stage_totals.calls
                totals.input_tokens += stage_totals.input_tokens
                totals.output_tokens += stage_totals.output_tokens
                totals.total_tokens += stage_totals.total_tokens
                totals.estimated_calls += stage_totals.estimated_calls
                stage_dict[stage] = stage_totals.to_dict()
            return {
                "run_id": run_id,
                "totals": totals.to_dict(),
                "stages": stage_dict,
            }

    def rollup(self) -> dict:
        with self._lock:
            run_ids = list(self._data.keys())
        totals = TokenTotals()
        stages: dict[str, TokenTotals] = {}
        for run_id in run_ids:
            summary = self.run_summary(run_id)
            if summary is None:
                continue
            run_totals = summary["totals"]
            totals.calls += run_totals["calls"]
            totals.input_tokens += run_totals["input_tokens"]
            totals.output_tokens += run_totals["output_tokens"]
            totals.total_tokens += run_totals["total_tokens"]
            totals.estimated_calls += run_totals["estimated_calls"]
            for stage, stage_dict in summary["stages"].items():
                stage_totals = stages.setdefault(stage, TokenTotals())
                stage_totals.calls += stage_dict["calls"]
                stage_totals.input_tokens += stage_dict["input_tokens"]
                stage_totals.output_tokens += stage_dict["output_tokens"]
                stage_totals.total_tokens += stage_dict["total_tokens"]
                stage_totals.estimated_calls += stage_dict["estimated_calls"]
        return {
            "runs": len(run_ids),
            "totals": totals.to_dict(),
            "stages": {stage: totals.to_dict() for stage, totals in stages.items()},
        }

    def reset(self) -> None:
        with self._lock:
            self._data.clear()

    def forget(self, run_id: str) -> None:
        """Drop a run's recorded entries. No-op if ``run_id`` is unknown."""
        with self._lock:
            self._data.pop(run_id, None)


# Module-level default ledger used when callers do not supply one.
_default_ledger: TokenUsageLedger = TokenUsageLedger()


def get_default_ledger() -> TokenUsageLedger:
    return _default_ledger


def reset_default_ledger() -> None:
    global _default_ledger
    _default_ledger = TokenUsageLedger()


def _estimate_one(text: str) -> int:
    """Estimate tokens for a single side via tiktoken; 0 if unavailable."""
    try:
        return TiktokenTokenCounter().count(text)
    except Exception:
        logger.debug("tiktoken unavailable; recording zero estimated tokens")
        return 0


def _estimate_tokens(prompt: str, completion: str) -> tuple[int, int, bool]:
    """Estimate both sides when a provider reports no usable usage.

    Returns (input_tokens, output_tokens, estimated_flag).
    """
    return _estimate_one(prompt), _estimate_one(completion), True


class UsageTrackingLLM:
    """Transparent wrapper that records token usage per stage/run.

    Delegates all other attributes/methods to the inner LLM so it satisfies
    the ``BaseLLM`` protocol structurally.
    """

    def __init__(
        self,
        inner: BaseLLM,
        ledger: Optional[TokenUsageLedger] = None,
    ):
        self._inner = inner
        self._ledger = ledger or _default_ledger

    @property
    def inner(self) -> BaseLLM:
        return self._inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, prompt: str, response: LLMResponse, latency_ms: int) -> None:
        run_id = _current_run_id.get()
        if run_id is None:
            return
        stage = _current_stage.get()
        usage = response.usage
        if usage is None or (
            usage.request_tokens is None and usage.response_tokens is None
        ):
            # No usable usage at all (or only a combined total_tokens with
            # no way to split it) -> estimate both sides.
            input_tokens, output_tokens, estimated = _estimate_tokens(
                prompt, response.content
            )
        else:
            estimated = False
            if usage.request_tokens is None:
                input_tokens = _estimate_one(prompt)
                estimated = True
            else:
                input_tokens = usage.request_tokens
            if usage.response_tokens is None:
                output_tokens = _estimate_one(response.content)
                estimated = True
            else:
                output_tokens = usage.response_tokens
        event = TokenUsageEvent(
            ts=time.time(),
            run_id=run_id,
            stage=stage,
            model=getattr(self._inner, "model_name", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
            latency_ms=latency_ms,
        )
        self._ledger.record(event)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        start = time.monotonic()
        response = self._inner.invoke(prompt, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        self._record(prompt, response, latency_ms)
        return response

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        start = time.monotonic()
        response = await self._inner.ainvoke(prompt, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        self._record(prompt, response, latency_ms)
        return response


def track_usage(llm: BaseLLM, *, ledger: Optional[TokenUsageLedger] = None) -> BaseLLM:
    """Wrap ``llm`` with usage tracking if it is not already wrapped."""
    if isinstance(llm, UsageTrackingLLM):
        return llm
    return UsageTrackingLLM(llm, ledger=ledger)


def usage_snapshot(llm: BaseLLM | None) -> dict | None:
    """Return a run summary for the currently active run, if any."""
    if not isinstance(llm, UsageTrackingLLM):
        return None
    run_id = _current_run_id.get()
    if run_id is None:
        return None
    return llm._ledger.run_summary(run_id)


def usage_delta(before: dict | None, after: dict | None) -> dict:
    """Subtract ``before`` stage totals from ``after`` stage totals.

    Clamps each stage value at 0. The returned dict has the same shape as a
    run summary (``totals`` + ``stages``).
    """
    if after is None:
        after = {}
    if before is None:
        before = {}

    def _sub(a: dict, b: dict) -> dict:
        return {
            key: max(int(a.get(key, 0)) - int(b.get(key, 0)), 0)
            for key in ("calls", "input_tokens", "output_tokens", "total_tokens", "estimated_calls")
        }

    after_total = after.get("totals", {})
    before_total = before.get("totals", {})
    after_stages = after.get("stages", {})
    before_stages = before.get("stages", {})
    all_stages = set(after_stages) | set(before_stages)
    return {
        "totals": _sub(after_total, before_total),
        "stages": {
            stage: _sub(after_stages.get(stage, {}), before_stages.get(stage, {}))
            for stage in all_stages
        },
    }


__all__ = [
    "TokenUsageEvent",
    "TokenTotals",
    "TokenUsageLedger",
    "UsageTrackingLLM",
    "token_stage",
    "run_scope",
    "track_usage",
    "usage_snapshot",
    "usage_delta",
    "get_default_ledger",
    "reset_default_ledger",
]
