# Graph Builder Backend Rewrite Roadmap

## Goal

Keep the high-level SDK stable while replacing `neo4j-graphrag` runtime dependencies with internal Recon-GraphRAG interfaces and implementations.

The current SDK shape should continue to work:

```python
builder = GraphBuilderPipeline(
    graph_store=graph_store,
    llm=llm,
    embedder=embedder,
    schema=MOVIE_SCHEMA,
)

await builder.build_from_text(text)
```

## Current Problem

The SDK currently relies on `neo4j-graphrag` for several internal responsibilities:

```text
SimpleKGPipeline
experimental schema classes
LLM interfaces and provider wrappers
embedding interfaces and provider wrappers
Neo4j index helpers
vector upsert helpers
entity resolution
HybridCypherRetriever
```

This makes chunking, prompting, parsing, IDs, graph writes, retrieval behavior, and provider abstractions harder to own and evolve.

## Target Architecture

Graph building should follow this shape:

```text
GraphBuilderPipeline
    -> TextChunker / PageWindowBuilder
    -> SchemaPromptBuilder
    -> LLMGraphExtractor
    -> GraphExtractionParser
    -> SchemaValidator
    -> GraphDocumentAssembler
    -> GraphWriter
    -> EntityResolver
    -> EntityEmbedder
```

Provider and retrieval dependencies should use internal contracts:

```text
BaseLLM / LLMResponse
BaseEmbedder
GraphStore
GraphWriter
internal local / DRIFT retrievers
```

## Output Graph Contract

Neo4j output must preserve the graph shape expected by retrieval, entity embedding, and community detection:

```text
(:Document)
(:Chunk)
(:Chunk)-[:PART_OF]->(:Document)
(:Chunk)-[:FROM_CHUNK]->(:__Entity__)
(:__Entity__:DomainLabel)
(:__Entity__)-[:DOMAIN_RELATIONSHIP]->(:__Entity__)
```

Important direction:

```text
(:Chunk)-[:FROM_CHUNK]->(:__Entity__)
```

Do not reverse this relationship.

## PR Roadmap

- [PR A: Internal Schema and Neutral Types](plans/graph_builder_pr_a_schema_and_types.md)
- [PR B: Extraction Backend](plans/graph_builder_pr_b_extraction_backend.md)
- [PR C: Replace SimpleKGPipeline](plans/graph_builder_pr_c_replace_simplekgpipeline.md)
- [PR D: Internal LLM and Embedding Interfaces](plans/graph_builder_pr_d_llm_and_embedding_interfaces.md)
- [PR E: Neo4j Store, Indexes, and Resolution](plans/graph_builder_pr_e_neo4j_store_indexes_and_resolution.md)
- [PR F: Retrieval Without HybridCypherRetriever](plans/graph_builder_pr_f_retrieval_without_hybridcypherretriever.md)

## Dependency Audit

Already covered by PR A-C:

```text
SimpleKGPipeline
neo4j_graphrag.experimental.components.schema
GraphWriter / Neo4jGraphWriter replacement path
```

Covered by PR D-F:

```text
neo4j_graphrag.llm
neo4j_graphrag.embeddings
neo4j_graphrag.indexes
SinglePropertyExactMatchResolver
HybridCypherRetriever
```

Packaging and docs should be updated after PR D-F:

```text
pyproject.toml
uv.lock
README.md
```

## Overall Acceptance Criteria

The rewrite is complete when all of these are true:

1. No runtime import of `SimpleKGPipeline` remains.
2. No runtime import of `neo4j_graphrag.experimental.components.schema` remains.
3. No runtime import of `neo4j_graphrag.llm` remains.
4. No runtime import of `neo4j_graphrag.embeddings` remains.
5. No runtime import of `neo4j_graphrag.indexes` remains.
6. No runtime import of `SinglePropertyExactMatchResolver` remains.
7. No runtime import of `HybridCypherRetriever` remains.
8. `neo4j-graphrag` is removed from `pyproject.toml`.
9. README no longer describes the SDK as built on `neo4j-graphrag`.
10. Existing schema definitions still work.
11. `build_from_text()` keeps the existing high-level SDK behavior.
12. `build_from_pages()` works with sliding windows.
13. Neo4j output contains `Document`, `Chunk`, `__Entity__`, chunk evidence links, and entity relationships.
14. Entity embeddings can be generated.
15. Community detection and summarization can run after graph build.
16. Local, global, and DRIFT search smoke tests pass.

## Final Dependency Test

Run:

```bash
rg "neo4j_graphrag" recon_graphrag
rg "neo4j-graphrag" pyproject.toml README.md
python -m compileall recon_graphrag
pytest
```

Expected final state:

```text
No runtime package code imports neo4j_graphrag.
neo4j-graphrag is removed from packaging.
Neo4j remains supported through the official neo4j driver.
```

## Smoke Test Plan

After implementation, run a small graph build:

```python
text = """
Christopher Nolan directed Inception in 2010.
Hans Zimmer composed the music for Inception.
Inception explores dreams and time manipulation.
"""

result = await builder.build_from_text(
    text,
    metadata={"source": "smoke-test"},
)
```

Verify Neo4j:

```cypher
MATCH (d:Document) RETURN count(d);
MATCH (c:Chunk) RETURN count(c);
MATCH (e:__Entity__) RETURN labels(e), e.name LIMIT 20;
MATCH (:Chunk)-[:FROM_CHUNK]->(:__Entity__) RETURN count(*);
MATCH (:__Entity__)-[r]-(:__Entity__) RETURN type(r), count(*) ORDER BY count(*) DESC;
```

Then run:

```python
await community_pipeline.build()
await graphrag.search("Who worked on Inception?", mode="local")
await graphrag.search("What are the major themes?", mode="global")
await graphrag.search("Explain Inception using detailed and broader context", mode="drift")
```
