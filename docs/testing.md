# Testing

This guide explains how to run the Recon-GraphRAG test suite. It covers mandatory tests, optional integration tests, and which commands to run for different kinds of changes.

The short version:

- Run mandatory tests while developing: `pytest -m "not integration"`
- Run provider integration tests when changing provider setup.
- Run focused Neo4j integration tests when changing real database behavior.
- Run the Neo4j movie smoke test when you need full end-to-end confidence.

## Testing model

The suite is split into two layers.

### Mandatory tests

Mandatory tests are fast, local, and deterministic. They do not require real provider credentials or a live Neo4j database.

They use fake graph stores, fake Neo4j drivers/sessions, fake LLMs, fake embedders, monkeypatched provider clients, and in-memory assertions.

They answer questions like:

- Does our code call the right method?
- Does it pass the right parameters?
- Does it produce the right result shape?
- Does it handle expected failure paths?
- Does each local algorithm behave correctly?

### Optional integration tests

Integration tests are marked with `@pytest.mark.integration`. They are collected by `pytest`, but skip unless explicit run flags and required environment variables are set.

They answer questions like:

- Does this work against real external services?
- Does this work against real Neo4j?
- Does the complete workflow still run end to end?

These tests may be slower, flaky if external services are unhealthy, and may incur provider cost.

## Dependency classification

| Test area | Real LLM | Real embeddings | Real graph DB | What it proves |
| --- | ---: | ---: | ---: | --- |
| `tests/extraction/` | No | No | No | JSON parsing, schema validation, chunking, and graph document assembly. |
| `tests/llm/` | No | No | No | LLM protocol and factory behavior using fake or monkeypatched clients. |
| `tests/embeddings/` | No | No | No | Embedder protocol and factory behavior using fake or monkeypatched clients. |
| `tests/graphdb/neo4j/test_entity_resolution.py` | Fake only | Fake only | No | Entity resolution strategy logic without real Neo4j or real models. |
| `tests/graphdb/neo4j/test_index_manager.py` | No | No | No | Index manager index creation and resolver wrapper behavior. |
| `tests/graphdb/neo4j/test_neo4j_store.py` | No | No | No | Neo4j store query construction using fake driver/session objects. |
| `tests/pipelines/neo4j/test_writer.py` | No | No | No | Neo4j graph writer query generation using a fake graph store. |
| `tests/pipelines/test_graph_builder.py` | Fake only | Fake only | No | Pipeline orchestration with fake LLM, fake embedder, fake graph store, and fake writer. |
| `tests/communities/` | No | Fake only | No | Community embedding and Neo4j community detection query behavior using fakes. |
| `tests/retrieval/` | Fake only | Fake only | No | Retrieval, ranking, and answer-generation flow using fake LLM/embedder/store. |
| `tests/integration/test_azure_openai_env.py` | Yes | Yes | No | Real Azure OpenAI LLM and embedding endpoint checks. |
| `tests/integration/test_openrouter_env.py` | Yes | Yes | No | Real OpenRouter LLM and embedding endpoint checks. |
| `tests/integration/neo4j/test_entity_resolution_integration.py` | Optional | Optional | Yes | Real Neo4j entity resolution; real LLM/embedder only with the AI flag. |
| `tests/integration/test_movie_example_smoke.py` | Yes | Yes | Yes | Neo4j end-to-end movie example: real provider + real Neo4j + APOC/GDS. |
| `tests/manual/` | Yes | Yes | No | Manual diagnostics; not part of normal test guidance. |

Meaning:

- **No**: the dependency is not used.
- **Fake only**: the dependency is represented by a fake or mock.
- **Optional**: some tests run without it; extra scenarios use it when a flag is enabled.
- **Yes**: the test requires the real dependency when enabled.

## What to run by change type

| Change type | Recommended command |
| --------- | --------- |
| Parser, schema, validator, chunking, assembler | `pytest tests/extraction` |
| LLM wrapper/factory code | `pytest tests/llm` |
| Embedding wrapper/factory code | `pytest tests/embeddings` |
| Entity deduplication local logic | `pytest tests/graphdb/neo4j/test_entity_resolution.py tests/graphdb/neo4j/test_index_manager.py tests/pipelines/test_graph_builder.py` |
| Entity deduplication with real Neo4j | `RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_entity_resolution_integration.py` |
| Hybrid dedup with real Neo4j + real LLM/embedder | Enable both entity-resolution integration flags and run `pytest tests/integration/neo4j/test_entity_resolution_integration.py` |
| Neo4j store, indexes, writer queries | `pytest tests/graphdb/neo4j tests/pipelines/neo4j` |
| Graph builder pipeline | `pytest tests/pipelines/test_graph_builder.py tests/extraction tests/graphdb/neo4j/test_entity_resolution.py` |
| Community detection/embedding helpers | `pytest tests/communities` |
| Retrieval or search behavior | `pytest tests/retrieval` |
| Provider credentials or provider request shape | Run `pytest tests/llm tests/embeddings`, then the relevant provider integration test |
| Real Neo4j end-to-end graph build/search | `RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/test_movie_example_smoke.py` |
| Before normal commit | `pytest -m "not integration"` |
| Before release or major workflow change | `pytest`, then enabled integration tests for configured providers/services |

## Standard commands

### Mandatory suite

```bash
pytest -m "not integration"
```

### Optional suite in skip-safe mode

```bash
pytest -m integration
```

Tests will skip unless their run flags and required env vars are configured.

### Full skip-safe suite

```bash
pytest
```

This runs mandatory tests plus optional tests that skip unless enabled.

### Full suite with enabled optional tests (Bash)

```bash
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 \
RUN_OPENROUTER_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest
```

### Full suite with enabled optional tests (PowerShell)

```powershell
$env:RUN_AZURE_OPENAI_INTEGRATION_TESTS="1"
$env:RUN_OPENROUTER_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS="1"
$env:RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS="1"
pytest
```

Only enable optional flags for services you have configured. Provider checks may incur API cost.

## Provider integration tests

### Azure OpenAI

```bash
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py
```

Required environment variables:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
```

### OpenRouter

```bash
RUN_OPENROUTER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py
```

Required environment variables:

```text
OPENROUTER_API_KEY
OPENROUTER_LLM_MODEL
OPENROUTER_EMBED_MODEL
```

### Neo4j entity resolution

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_entity_resolution_integration.py
```

Required environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

With real Azure OpenAI LLM and embedder:

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/neo4j/test_entity_resolution_integration.py
```

### Neo4j movie example smoke test

```bash
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/test_movie_example_smoke.py
```

Required environment variables:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

## Environment setup

Install development dependencies. We recommend using `uv`:

```bash
uv sync --group dev
```

Or with `pip`:

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill only the values needed for the optional tests you intend to run.

Run the test suite with `uv`:

```bash
uv run pytest -m "not integration"
```

or with `pytest` directly if you installed into an activated virtual environment.

## Notes

- `pytest` uses `pythonpath = ["."]` from `pyproject.toml`, so the local package imports from the repository checkout.
- Integration tests should stay behind explicit run flags.
- New external-service tests should use `@pytest.mark.integration` and skip unless explicitly enabled.
- New graph database backends should get backend-specific integration tests under `tests/integration/<backend>/`.
