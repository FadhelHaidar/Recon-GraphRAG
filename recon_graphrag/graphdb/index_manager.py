"""Backend-dispatching :func:`IndexManager` factory.

Neo4j and Memgraph need different Cypher for dropping indexes and creating
constraints, so each backend ships its own IndexManager. This factory detects
the concrete graph store and returns the matching one, so callers use a single
``IndexManager(store, ...)`` entry point regardless of backend.
"""

from __future__ import annotations


def IndexManager(graph_store, *args, **kwargs):
    """Return the backend IndexManager matching the graph store's type."""
    from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
    from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

    if isinstance(graph_store, Neo4jGraphStore):
        from recon_graphrag.graphdb.neo4j.index_manager import (
            IndexManager as _Backend,
        )
    elif isinstance(graph_store, MemgraphGraphStore):
        from recon_graphrag.graphdb.memgraph.index_manager import (
            IndexManager as _Backend,
        )
    else:
        raise TypeError(
            f"IndexManager does not support graph store type "
            f"{type(graph_store).__name__!r}; expected Neo4jGraphStore or "
            f"MemgraphGraphStore."
        )
    return _Backend(graph_store, *args, **kwargs)
