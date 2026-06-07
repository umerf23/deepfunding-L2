"""Centrality-based originality scorer.

Given the ecosystem graph, we compute per-node structural measures and convert
each repo's source-vs-sink balance into an originality score in (0, 1).

Measures used:
  - PageRank (reversed graph): authority of being depended upon. A repo that
    foundational packages point back to scores high. We reverse because edges
    are A->B = "A depends on B", and we want importance to flow to the depended-upon.
  - HITS authority: complements PageRank as a depended-upon signal.
  - out_degree: how many things the repo directly relies on (penalizes reliance).
  - in_degree:  how many things rely on the repo (rewards being foundational).
  - dependent_count: external popularity from deps.dev (rewards being foundational).

originality grows with (in-signals) and shrinks with (out-signals). Everything is
rank-normalized across repos so the score is a stable relative ordering, which is
what the contest rewards.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ..graph.builder import GraphBuildResult
from ..utils.common import setup_logging

logger = setup_logging()


class CentralityScorer:
    """Scores originality from a node's position in the ecosystem graph."""

    def __init__(self, config: dict[str, Any]) -> None:
        sc = config["scoring"]
        self.clip_min: float = sc["clip_min"]
        self.clip_max: float = sc["clip_max"]
        self.decimals: int = sc["output_decimals"]
        # Weights for the source-vs-sink balance. Documented and auditable.
        self.weights = {
            "authority": 0.35,      # depended-upon (PageRank reversed + HITS auth)
            "in_degree": 0.20,      # local depended-upon
            "dependent_count": 0.15,  # external popularity
            "out_degree": -0.30,    # local reliance (penalty)
        }

    def score(self, build: GraphBuildResult) -> pd.DataFrame:
        graph = build.graph
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot score an empty graph.")

        authority = self._authority(graph)
        in_deg = dict(graph.in_degree())
        out_deg = dict(graph.out_degree())

        rows = []
        for repo_url, node_id in build.repo_to_node.items():
            rows.append({
                "repo": repo_url,
                "authority": authority.get(node_id, 0.0),
                "in_degree": float(in_deg.get(node_id, 0)),
                "out_degree": float(out_deg.get(node_id, 0)),
                "dependent_count": float(build.repo_dependent_counts.get(repo_url, 0)),
            })
        frame = pd.DataFrame(rows)

        # Rank-normalize every signal to [0, 1] across repos for comparability.
        for col in ("authority", "in_degree", "dependent_count", "out_degree"):
            frame[f"{col}_n"] = self._rank_normalize(frame[col].to_numpy())

        linear = (
            self.weights["authority"] * frame["authority_n"]
            + self.weights["in_degree"] * frame["in_degree_n"]
            + self.weights["dependent_count"] * frame["dependent_count_n"]
            + self.weights["out_degree"] * frame["out_degree_n"]
        ).to_numpy()

        # Map the weighted balance onto (0,1) via min-max then clip; preserves order.
        scores = self._minmax(linear)
        scores = np.clip(scores, self.clip_min, self.clip_max)
        scores = np.round(scores, self.decimals)

        result = pd.DataFrame({"repo": frame["repo"].to_numpy(), "originality": scores})
        logger.info(
            "Scored %d repos via centrality; range [%.4f, %.4f]",
            len(result), result["originality"].min(), result["originality"].max(),
        )
        return result

    @staticmethod
    def _authority(graph: nx.DiGraph) -> dict[str, float]:
        """Combine reversed PageRank and HITS authority into one depended-upon score."""
        reversed_graph = graph.reverse(copy=False)
        try:
            pagerank = nx.pagerank(reversed_graph, alpha=0.85, max_iter=200, tol=1e-08)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge; using degree fallback.")
            total = max(graph.number_of_nodes(), 1)
            pagerank = {n: graph.in_degree(n) / total for n in graph.nodes()}

        try:
            _, authority = nx.hits(graph, max_iter=200, tol=1e-08, normalized=True)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            logger.warning("HITS did not converge; using PageRank alone.")
            authority = {n: 0.0 for n in graph.nodes()}

        combined: dict[str, float] = {}
        for node in graph.nodes():
            combined[node] = 0.6 * pagerank.get(node, 0.0) + 0.4 * authority.get(node, 0.0)
        return combined

    @staticmethod
    def _rank_normalize(values: np.ndarray) -> np.ndarray:
        """Map values to [0,1] by rank, robust to outliers and skew."""
        if len(values) <= 1:
            return np.zeros_like(values, dtype=float)
        order = values.argsort().argsort().astype(float)
        return order / (len(values) - 1)

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return np.full_like(values, 0.5, dtype=float)
        return (values - lo) / (hi - lo)
