"""Shared movie-example retrieval query suite."""

from __future__ import annotations

MOVIE_QUERY_SUITE = [
    # --- local (2) ---
    {
        "query": "Which movies were directed by Christopher Nolan and feature Cillian Murphy?",
        "modes": ["local"],
        "test_objective": "Verify DIRECTED and ACTED_IN relationships between Person and Movie.",
    },
    {
        "query": "Which movies in this corpus won the Oscar for Best Picture?",
        "modes": ["local"],
        "test_objective": "Verify WON_AWARD relationship extraction for Movie to Award.",
    },
    # --- global (2) ---
    {
        "query": "What are the major communities or clusters in this movie graph?",
        "modes": ["global"],
        "test_objective": "Assess global search quality using community reports.",
    },
    {
        "query": "Summarize the Nolan-related community in this graph.",
        "modes": ["global"],
        "test_objective": "Test whether global retrieval can summarize the Nolan subgraph involving movies, actors, composer, cinematographer, studio, and themes.",
    },
    # --- drift (2) ---
    {
        "query": "How does Hans Zimmer connect Inception to Dune?",
        "modes": ["local", "drift"],
        "test_objective": "Test multi-hop traversal from Movie to Person to Movie using COMPOSED_MUSIC relationships.",
    },
    {
        "query": "How is Cillian Murphy connected to Michelle Yeoh?",
        "modes": ["local", "drift"],
        "test_objective": "Test cross-film pathfinding through Sunshine, Oppenheimer, Everything Everywhere All At Once, and actor bridges.",
    },
]
