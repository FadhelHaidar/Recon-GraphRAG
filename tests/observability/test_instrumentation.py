"""Integration tests: tracked LLM records the right stages through real call sites."""

from __future__ import annotations

import asyncio

from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.llm.base import LLMResponse, LLMUsage
from recon_graphrag.observability import (
    TokenUsageLedger,
    run_scope,
    track_usage,
    usage_delta,
    usage_snapshot,
)
from recon_graphrag.retrieval.drift_types import DriftQueryState
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever


class SequenceLLM:
    """Returns canned responses in order, with fixed provider usage."""

    model_name = "seq-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._calls = 0

    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse:
        content = self._responses[self._calls]
        self._calls += 1
        return LLMResponse(content=content, usage=LLMUsage(7, 3, 10))

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(prompt, **kwargs))


def _schema() -> GraphSchema:
    return GraphSchema(
        node_types=[
            NodeType(label="Person", properties=[PropertyType(name="name", type="STRING")]),
            NodeType(label="Movie", properties=[PropertyType(name="title", type="STRING")]),
        ],
        relationship_types=[RelationshipType(label="ACTED_IN")],
        patterns=[("Person", "ACTED_IN", "Movie")],
    )


INITIAL = '''{
    "nodes": [
        {"id": "p1", "label": "Person", "properties": {"name": "Alice"}},
        {"id": "m1", "label": "Movie", "properties": {"title": "Inception"}}
    ],
    "relationships": [
        {"source_id": "p1", "target_id": "m1", "type": "ACTED_IN", "properties": {"weight": 1.0}}
    ]
}'''

CONTINUATION = '''{
    "nodes": [{"id": "p2", "label": "Person", "properties": {"name": "Bob"}}],
    "relationships": []
}'''


async def test_extractor_records_extract_and_gleaning_stages():
    ledger = TokenUsageLedger()
    llm = track_usage(SequenceLLM([INITIAL, "yes", CONTINUATION]), ledger=ledger)
    extractor = LLMGraphExtractor(llm)

    with run_scope("build-run"):
        await extractor.extract("Alice acted in Inception.", _schema(), max_gleanings=1)

    summary = ledger.run_summary("build-run")
    assert summary["stages"]["construction.extract"]["calls"] == 1
    assert summary["stages"]["construction.gleaning_assess"]["calls"] == 1
    assert summary["stages"]["construction.gleaning_continue"]["calls"] == 1
    assert summary["totals"]["input_tokens"] == 21  # 3 calls x 7 provider tokens
    assert summary["totals"]["estimated_calls"] == 0


async def test_drift_invoke_llm_records_phase_stages():
    ledger = TokenUsageLedger()
    llm = track_usage(SequenceLLM(["a1", "a2", "a3"]), ledger=ledger)
    retriever = DriftSearchRetriever(object(), llm, object())
    state = DriftQueryState(query="q")
    lock = asyncio.Lock()

    with run_scope("drift-run"):
        await retriever._invoke_llm("p", state, 10, lock, "primer")
        await retriever._invoke_llm("p", state, 10, lock, "action")
        await retriever._invoke_llm("p", state, 10, lock, "action")

    summary = ledger.run_summary("drift-run")
    assert summary["stages"]["drift.primer"]["calls"] == 1
    assert summary["stages"]["drift.action"]["calls"] == 2
    # Existing approximate trace kept alongside provider-reported numbers.
    assert state.phase_tokens["primer"] > 0
    assert state.total_llm_calls == 3


async def test_usage_delta_gives_per_phase_attribution():
    ledger = TokenUsageLedger()
    llm = track_usage(SequenceLLM([INITIAL, INITIAL]), ledger=ledger)
    extractor = LLMGraphExtractor(llm)

    with run_scope("delta-run"):
        await extractor.extract("text one", _schema())
        before = usage_snapshot(llm)
        await extractor.extract("text two", _schema())
        delta = usage_delta(before, usage_snapshot(llm))

    assert delta["totals"]["calls"] == 1
    assert delta["stages"]["construction.extract"]["input_tokens"] == 7
