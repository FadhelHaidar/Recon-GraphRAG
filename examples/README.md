# Example

This directory contains an end-to-end movie-domain GraphRAG workflow for Neo4j and Memgraph:

```bash
python extract.py
python ingest.py --backend all
python communities.py --backend all
python search.py --backend neo4j
python search.py --backend memgraph
python compare_backends.py
```
