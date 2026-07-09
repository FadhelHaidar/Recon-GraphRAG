"""Tests for LLM-powered schema analysis."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from recon_graphrag.extraction.schema_analyzer import analyze_schema
from recon_graphrag.utils.tokens import ApproximateTokenCounter


GOOD_JSON = """```json
{
  "node_types": [
    {"label": "Person", "description": "A person",
     "properties": [{"name": "born", "type": "date"}]},
    {"label": "Movie", "description": "A film"}
  ],
  "relationship_types": [
    {"label": "ACTED_IN", "description": "Person acted in movie"}
  ],
  "patterns": [["Person", "ACTED_IN", "Movie"]]
}
```"""

MESSY_JSON = """{
  "node_types": [
    {"label": "Person", "description": "A person"},
    {"label": "Person", "description": "duplicate"},
    {"label": "Movie", "properties": [{"name": "budget", "type": "MONEY"}]}
  ],
  "relationship_types": [{"label": "ACTED_IN"}],
  "patterns": [
    ["Person", "ACTED_IN", "Movie"],
    ["Person", "DIRECTED", "Movie"],
    ["Person", "ACTED_IN"]
  ]
}"""


def _make_llm(*contents: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=c) for c in contents]
    return llm


def test_fenced_valid_response_builds_schema():
    schema = analyze_schema(_make_llm(GOOD_JSON), "some sample text")

    assert schema.node_labels() == {"Person", "Movie"}
    assert schema.relationship_labels() == {"ACTED_IN"}
    assert schema.patterns == [("Person", "ACTED_IN", "Movie")]
    # Property type normalized to uppercase; defaults injected
    person = schema.get_node_type("Person")
    assert {"name", "description", "born"} <= person.property_names
    assert next(p for p in person.properties if p.name == "born").type == "DATE"


def test_messy_response_is_sanitized():
    schema = analyze_schema(_make_llm(MESSY_JSON), "text")

    # Duplicate label deduped, unknown property type falls back to STRING,
    # patterns with unknown rel labels or wrong arity dropped
    assert schema.node_labels() == {"Person", "Movie"}
    assert schema.patterns == [("Person", "ACTED_IN", "Movie")]
    movie = schema.get_node_type("Movie")
    assert next(p for p in movie.properties if p.name == "budget").type == "STRING"


def test_invalid_json_twice_raises_after_retry():
    llm = _make_llm("not json", "still not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        analyze_schema(llm, "text")

    assert llm.invoke.call_count == 2


MERGED_JSON = """{
  "node_types": [
    {"label": "Person", "description": "A person"},
    {"label": "Movie", "description": "A film"},
    {"label": "Studio", "description": "A production company"}
  ],
  "relationship_types": [{"label": "ACTED_IN"}, {"label": "PRODUCED"}],
  "patterns": [
    ["Person", "ACTED_IN", "Movie"],
    ["Studio", "PRODUCED", "Movie"]
  ]
}"""


def test_oversized_input_maps_batches_then_merges():
    # Three 50-char texts are ~13 tokens each (ratio 4); a 15-token budget
    # fits one text but not two -> 3 batches -> 3 analysis calls, then
    # 1 merge call whose response becomes the final schema.
    llm = _make_llm(GOOD_JSON, GOOD_JSON, GOOD_JSON, MERGED_JSON)

    schema = analyze_schema(
        llm,
        ["a" * 50, "b" * 50, "c" * 50],
        max_sample_tokens=15,
        token_counter=ApproximateTokenCounter(),
    )

    assert llm.invoke.call_count == 4
    merge_prompt = llm.invoke.call_args_list[3].args[0]
    assert "Proposal 3" in merge_prompt
    assert schema.node_labels() == {"Person", "Movie", "Studio"}
    assert ("Studio", "PRODUCED", "Movie") in schema.patterns
