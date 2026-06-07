"""Offline unit tests for the graph-centrality solution.

We construct synthetic ecosystem graphs by hand and assert the structural
properties that define originality:
  - a foundational "source" repo (depended upon, depends on little) scores HIGH
  - a derivative "sink" repo (depends on many, used by none) scores LOW
  - scores are bounded and reproducible
No network access: graphs are built directly via the builder's merge logic.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.graph.builder import GraphBuildResult
from src.model.centrality_scorer import CentralityScorer

CONFIG = {
    "scoring": {"output_decimals": 4, "clip_min": 0.01, "clip_max": 0.99},
    "model": {
        "use_lgbm_rank": False, "lgbm_blend_weight": 0.0, "random_state": 42,
        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.9, "cv_folds": 3,
    },
    "runtime": {"parallel_workers": 2, "log_level": "WARNING"},
}


def _toy_build() -> GraphBuildResult:
    """Three repos: foundational, middle, derivative.

    Edges are A->B = 'A depends on B'.
      derivative -> middle -> foundational
      derivative -> foundational
    So foundational is depended upon by everyone and depends on nothing (source).
    derivative depends on two and is used by none (sink).
    """
    g = nx.DiGraph()
    g.add_node("CARGO::foundational", is_repo=True, resolved=True)
    g.add_node("CARGO::middle", is_repo=True, resolved=True)
    g.add_node("CARGO::derivative", is_repo=True, resolved=True)
    g.add_edge("CARGO::middle", "CARGO::foundational")
    g.add_edge("CARGO::derivative", "CARGO::middle")
    g.add_edge("CARGO::derivative", "CARGO::foundational")

    repo_to_node = {
        "https://github.com/x/foundational": "CARGO::foundational",
        "https://github.com/x/middle": "CARGO::middle",
        "https://github.com/x/derivative": "CARGO::derivative",
    }
    dependent_counts = {
        "https://github.com/x/foundational": 500,
        "https://github.com/x/middle": 20,
        "https://github.com/x/derivative": 0,
    }
    return GraphBuildResult(
        graph=g, repo_to_node=repo_to_node, repo_dependent_counts=dependent_counts
    )


def test_foundational_scores_higher_than_derivative() -> None:
    build = _toy_build()
    result = CentralityScorer(CONFIG).score(build).set_index("repo")["originality"]
    assert result["https://github.com/x/foundational"] > result["https://github.com/x/derivative"]


def test_scores_are_bounded() -> None:
    build = _toy_build()
    result = CentralityScorer(CONFIG).score(build)
    assert result["originality"].between(0.01, 0.99).all()


def test_ordering_is_monotonic() -> None:
    build = _toy_build()
    r = CentralityScorer(CONFIG).score(build).set_index("repo")["originality"]
    assert (
        r["https://github.com/x/foundational"]
        >= r["https://github.com/x/middle"]
        >= r["https://github.com/x/derivative"]
    )


def test_scoring_is_reproducible() -> None:
    build = _toy_build()
    first = CentralityScorer(CONFIG).score(build)["originality"].to_numpy()
    second = CentralityScorer(CONFIG).score(build)["originality"].to_numpy()
    assert np.allclose(first, second)


def test_empty_graph_raises() -> None:
    empty = GraphBuildResult(graph=nx.DiGraph(), repo_to_node={}, repo_dependent_counts={})
    with pytest.raises(ValueError):
        CentralityScorer(CONFIG).score(empty)


def test_rank_normalize_endpoints() -> None:
    scorer = CentralityScorer(CONFIG)
    norm = scorer._rank_normalize(np.array([10.0, 20.0, 30.0]))
    assert norm.min() == 0.0
    assert norm.max() == 1.0
