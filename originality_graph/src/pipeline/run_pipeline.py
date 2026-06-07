"""End-to-end pipeline for the graph-centrality originality solution.

Stages:
  1. load the 98 target repos
  2. build the shared ecosystem dependency graph from deps.dev
  3. compute centrality and convert to originality scores
  4. optionally blend a LightGBM ranking head fit on the sample labels
  5. export the graph (for inspection) and write the submission CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ..graph.builder import EcosystemGraphBuilder
from ..model.centrality_scorer import CentralityScorer
from ..utils.common import load_config, setup_logging

logger = setup_logging()


class GraphPipeline:
    """Coordinates graph build, centrality scoring, and submission output."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.builder = EcosystemGraphBuilder(config)
        self.scorer = CentralityScorer(config)

    def load_targets(self) -> list[str]:
        path = Path(self.config["paths"]["repos_to_predict"])
        frame = pd.read_csv(path)
        if "repo" not in frame.columns:
            raise ValueError(f"Expected 'repo' column in {path}, got {list(frame.columns)}")
        repos = [str(r).strip() for r in frame["repo"].dropna() if str(r).strip()]
        logger.info("Loaded %d target repositories.", len(repos))
        return repos

    def run(self) -> pd.DataFrame:
        repos = self.load_targets()
        build = self.builder.build(repos)
        self._export_graph(build.graph)

        result = self.scorer.score(build)

        if self.config["model"]["use_lgbm_rank"]:
            result = self._apply_lgbm(build, result)

        result = self._restore_order(result, repos)
        self.write_submission(result)
        return result

    def _apply_lgbm(self, build, centrality_scores: pd.DataFrame) -> pd.DataFrame:
        """Blend in a LightGBM head fit on the sample labels."""
        sample_path = Path("data/raw/sample_submission.csv")
        if not sample_path.is_file():
            logger.warning("No sample labels; skipping LightGBM blend.")
            return centrality_scores

        # Rebuild the rank-normalized feature frame the scorer used.
        feature_frame = self._centrality_feature_frame(build)
        sample = pd.read_csv(sample_path)
        sample["repo"] = sample["repo"].astype(str).str.strip()
        merged = feature_frame.merge(sample, on="repo", how="inner")
        if merged.empty or "originality" not in merged.columns:
            logger.warning("Sample labels did not align; skipping LightGBM blend.")
            return centrality_scores

        from ..model.lgbm_rank import LGBMRankHead

        head = LGBMRankHead(self.config)
        labels = merged["originality"].to_numpy()
        head.cross_validate(merged, labels)
        head.fit(merged, labels)
        preds = head.predict(feature_frame)

        blend = float(self.config["model"]["lgbm_blend_weight"])
        blended = (1.0 - blend) * centrality_scores["originality"].to_numpy() + blend * preds
        decimals = self.config["scoring"]["output_decimals"]
        cmin, cmax = self.config["scoring"]["clip_min"], self.config["scoring"]["clip_max"]
        blended = np.round(np.clip(blended, cmin, cmax), decimals)
        logger.info("Applied LightGBM blend at weight %.2f", blend)
        return pd.DataFrame({"repo": centrality_scores["repo"].to_numpy(), "originality": blended})

    def _centrality_feature_frame(self, build) -> pd.DataFrame:
        """Recompute the normalized centrality features for the LightGBM head."""
        graph = build.graph
        authority = self.scorer._authority(graph)
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
        for col in ("authority", "in_degree", "dependent_count", "out_degree"):
            frame[f"{col}_n"] = self.scorer._rank_normalize(frame[col].to_numpy())
        out_path = Path(self.config["paths"]["engineered_features"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return frame

    def _export_graph(self, graph: nx.DiGraph) -> None:
        path = Path(self.config["paths"]["graph_export"])
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nx.write_gexf(graph, path)
            logger.info("Exported ecosystem graph to %s", path)
        except Exception as exc:  # noqa: BLE001 - export is best-effort
            logger.warning("Graph export failed (non-fatal): %s", exc)

    @staticmethod
    def _restore_order(result: pd.DataFrame, repos: list[str]) -> pd.DataFrame:
        order = {url.strip(): i for i, url in enumerate(repos)}
        result = result.copy()
        result["_order"] = result["repo"].map(order)
        return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    def write_submission(self, scores: pd.DataFrame) -> Path:
        out_path = Path(self.config["paths"]["submission"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores[["repo", "originality"]].to_csv(out_path, index=False)
        logger.info("Wrote submission with %d rows to %s", len(scores), out_path)
        return out_path


def main() -> None:
    config = load_config()
    setup_logging(config["runtime"]["log_level"])
    pipeline = GraphPipeline(config)
    scores = pipeline.run()
    logger.info(
        "Done. Originality range [%.4f, %.4f], mean %.4f",
        scores["originality"].min(), scores["originality"].max(), scores["originality"].mean(),
    )


if __name__ == "__main__":
    main()
