"""LLM-powered schema proposal from sample text.

For users who do not yet know what graph schema fits their documents:
feed sample text (plus an optional domain hint) to an LLM and get back
a ``GraphSchema`` proposal to inspect, tweak, and pass to the pipeline.

When the input fits within ``max_sample_tokens`` it is analyzed in a
single LLM call. Larger inputs are packed into batches, each batch is
analyzed independently (concurrently in the async variant), and a final
merge call unifies the partial proposals into one schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable

from recon_graphrag.extraction.parser import GraphExtractionParser
from recon_graphrag.extraction.schema import GraphSchema, build_schema, schema_to_dict
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.utils.tokens import TiktokenTokenCounter, TokenCounter


logger = logging.getLogger(__name__)

_ALLOWED_PROPERTY_TYPES = {
    "STRING",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "DATE",
    "DATETIME",
    "LIST",
}

_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. "
    "Return only the JSON object, with no markdown or commentary."
)

_SCHEMA_RULES_AND_FORMAT = """Rules:
1. Node labels use PascalCase (e.g. "Person"). Relationship labels use
   UPPER_SNAKE_CASE (e.g. "WORKS_AT").
2. No duplicate labels.
3. Every pattern is [source_node_label, relationship_label, target_node_label]
   and must only use labels you defined.
4. Do not include "name" or "description" node properties; they are added
   automatically. Only propose extra properties clearly supported by the input.
5. Property types must be one of: STRING, INTEGER, FLOAT, BOOLEAN, DATE,
   DATETIME, LIST.
6. Prefer a small, general schema over many overly specific types.
7. Return valid JSON only. Do not include markdown.

JSON format:
{
  "node_types": [
    {
      "label": "Company",
      "description": "A business organization",
      "properties": [{"name": "founded", "type": "DATE"}]
    }
  ],
  "relationship_types": [
    {"label": "ACQUIRED", "description": "A company acquired another company"}
  ],
  "patterns": [
    ["Company", "ACQUIRED", "Company"]
  ]
}"""


def analyze_schema(
    llm: BaseLLM,
    texts: str | Iterable[str],
    *,
    hint: str = "",
    max_sample_tokens: int = 2000,
    max_batches: int = 10,
    token_counter: TokenCounter | None = None,
) -> GraphSchema:
    """Propose a GraphSchema from sample text using an LLM.

    Args:
        llm: Any BaseLLM instance.
        texts: A sample text or iterable of sample texts.
        hint: Optional domain description (e.g. "legal contracts").
        max_sample_tokens: Per-batch token budget for the sample text
            (prompt template overhead is extra). Input beyond one batch
            is analyzed batch-by-batch, then merged with one extra LLM
            call (map-reduce). The sync variant analyzes batches
            sequentially; use aanalyze_schema for concurrent analysis.
        max_batches: Cap on analysis calls; batches beyond it are dropped.
        token_counter: Counter used for budgeting. Defaults to tiktoken
            with the cl100k_base encoding, like the pipeline chunker.

    Returns:
        A validated GraphSchema proposal to inspect, tweak, and pass
        to GraphBuilderPipeline.
    """
    batches = _make_batches(texts, max_sample_tokens, max_batches, token_counter)
    partials = [
        _invoke_and_parse(llm, _build_analysis_prompt(batch, hint))
        for batch in batches
    ]
    if len(partials) == 1:
        return partials[0]
    return _invoke_and_parse(llm, _build_merge_prompt(partials, hint))


async def aanalyze_schema(
    llm: BaseLLM,
    texts: str | Iterable[str],
    *,
    hint: str = "",
    max_sample_tokens: int = 2000,
    max_batches: int = 10,
    token_counter: TokenCounter | None = None,
) -> GraphSchema:
    """Async variant of :func:`analyze_schema`. Batches analyze concurrently."""
    batches = _make_batches(texts, max_sample_tokens, max_batches, token_counter)
    partials = await asyncio.gather(
        *(
            _ainvoke_and_parse(llm, _build_analysis_prompt(batch, hint))
            for batch in batches
        )
    )
    if len(partials) == 1:
        return partials[0]
    return await _ainvoke_and_parse(llm, _build_merge_prompt(list(partials), hint))


def _make_batches(
    texts: str | Iterable[str],
    max_sample_tokens: int,
    max_batches: int,
    token_counter: TokenCounter | None = None,
) -> list[str]:
    """Greedily pack whole texts into batches of at most max_sample_tokens.

    A text that would overflow the current batch starts the next batch
    intact; only a single text exceeding the whole budget is truncated.
    """
    counter = token_counter or TiktokenTokenCounter()
    items = [texts] if isinstance(texts, str) else list(texts)

    batches: list[str] = []
    current = ""
    current_tokens = 0
    for text in items:
        text_tokens = counter.count(text)
        # Separator tokens are ignored; the budget is for sample text and
        # callers should leave headroom for the prompt template anyway.
        if current and current_tokens + text_tokens <= max_sample_tokens:
            current = f"{current}\n\n{text}"
            current_tokens += text_tokens
            continue
        if current:
            batches.append(current)
        # ponytail: a single oversized text is head-truncated, not split
        current = counter.truncate(text, max_sample_tokens)
        current_tokens = counter.count(current)
    if current:
        batches.append(current)

    if len(batches) > max_batches:
        logger.warning(
            "schema analysis: input packs into %s batches; "
            "analyzing only the first %s (raise max_batches to cover more)",
            len(batches),
            max_batches,
        )
        batches = batches[:max_batches]

    if len(batches) > 1:
        logger.info(
            "schema analysis: %s analysis calls + 1 merge call", len(batches)
        )
    return batches


def _build_analysis_prompt(sample: str, hint: str) -> str:
    return f"""
You are a knowledge-graph schema designer. Analyze the sample text and
propose a graph schema for extracting entities and relationships from
documents like it.

{_SCHEMA_RULES_AND_FORMAT}
{_hint_section(hint)}
Sample text:
{sample}
""".strip()


def _build_merge_prompt(schemas: list[GraphSchema], hint: str) -> str:
    proposals = "\n\n".join(
        f"Proposal {i + 1}:\n{json.dumps(schema_to_dict(schema), indent=2)}"
        for i, schema in enumerate(schemas)
    )

    return f"""
You are a knowledge-graph schema designer. The following schema proposals
were each analyzed from a different sample of the same corpus. Merge them
into a single schema.

Merging guidance:
- Unify labels that mean the same thing (e.g. "Actor" and "Person") into
  the more general label, and rewrite patterns to use the unified labels.
- Keep every distinct entity and relationship type found in any proposal.

{_SCHEMA_RULES_AND_FORMAT}
{_hint_section(hint)}
Schema proposals:
{proposals}
""".strip()


def _hint_section(hint: str) -> str:
    return f"\nDomain context from the user:\n{hint}\n" if hint else ""


def _invoke_and_parse(llm: BaseLLM, prompt: str) -> GraphSchema:
    response = llm.invoke(prompt)
    try:
        return _parse_schema(response.content)
    except json.JSONDecodeError:
        response = llm.invoke(prompt + _RETRY_SUFFIX)
        try:
            return _parse_schema(response.content)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Schema analysis returned invalid JSON: {response.content[:500]}"
            ) from err


async def _ainvoke_and_parse(llm: BaseLLM, prompt: str) -> GraphSchema:
    response = await llm.ainvoke(prompt)
    try:
        return _parse_schema(response.content)
    except json.JSONDecodeError:
        response = await llm.ainvoke(prompt + _RETRY_SUFFIX)
        try:
            return _parse_schema(response.content)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Schema analysis returned invalid JSON: {response.content[:500]}"
            ) from err


def _parse_schema(content: str) -> GraphSchema:
    payload = GraphExtractionParser()._extract_json(content)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("expected a JSON object", payload, 0)

    node_types = _dedupe_by_label(data.get("node_types") or [])
    relationship_types = _dedupe_by_label(data.get("relationship_types") or [])
    for item in node_types + relationship_types:
        _normalize_property_types(item)

    # Drop malformed patterns and ones referencing unknown labels instead of
    # failing schema validation — the user inspects the proposal anyway.
    node_labels = {nt["label"] for nt in node_types}
    rel_labels = {rt["label"] for rt in relationship_types}
    patterns = [
        (p[0], p[1], p[2])
        for p in (data.get("patterns") or [])
        if isinstance(p, (list, tuple))
        and len(p) == 3
        and p[0] in node_labels
        and p[1] in rel_labels
        and p[2] in node_labels
    ]

    return build_schema(node_types, relationship_types, patterns)


def _dedupe_by_label(items: list) -> list[dict]:
    seen: set[str] = set()
    result = []
    for item in items:
        if not isinstance(item, dict) or not item.get("label"):
            continue
        if item["label"] in seen:
            continue
        seen.add(item["label"])
        result.append(item)
    return result


def _normalize_property_types(item: dict) -> None:
    for prop in item.get("properties") or []:
        if isinstance(prop, dict):
            prop_type = str(prop.get("type", "STRING")).upper()
            prop["type"] = (
                prop_type if prop_type in _ALLOWED_PROPERTY_TYPES else "STRING"
            )
