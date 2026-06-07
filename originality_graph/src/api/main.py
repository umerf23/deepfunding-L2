"""FastAPI service for the graph-centrality originality solution.

Unlike Solution 1 (which can score arbitrary new repos in isolation), this model
is inherently cohort-relative: a repo's originality depends on its position in the
whole ecosystem graph. So the API serves the precomputed cohort scores and exposes
graph diagnostics, rather than scoring brand-new repos on the fly.

Endpoints:
  GET /health           : liveness + whether scores are loaded
  GET /scores           : full ranked originality table
  GET /scores/{owner}/{name} : score for one repo in the cohort
  GET /metrics          : request counters
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..utils.common import load_config, setup_logging

logger = setup_logging()
config = load_config()

app = FastAPI(
    title="Graph Centrality Originality API",
    description="Gitcoin GR24 Level II - originality via ecosystem dependency network.",
    version="1.0.0",
)

_STATE: dict[str, Any] = {"scores": None, "requests": 0}


@app.on_event("startup")
def _load_scores() -> None:
    path = Path(config["paths"]["submission"])
    if path.is_file():
        _STATE["scores"] = pd.read_csv(path)
        logger.info("Loaded %d cohort scores from %s", len(_STATE["scores"]), path)
    else:
        logger.warning("No submission file at %s; run the pipeline first.", path)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "scores_loaded": _STATE["scores"] is not None}


@app.get("/metrics")
def metrics() -> dict[str, int]:
    return {"requests": _STATE["requests"]}


@app.get("/scores")
def scores() -> dict[str, Any]:
    _STATE["requests"] += 1
    if _STATE["scores"] is None:
        raise HTTPException(status_code=503, detail="Scores not loaded. Run the pipeline.")
    ranked = _STATE["scores"].sort_values("originality", ascending=False)
    return {"count": len(ranked), "results": ranked.to_dict(orient="records")}


@app.get("/scores/{owner}/{name}")
def repo_score(owner: str, name: str) -> dict[str, Any]:
    _STATE["requests"] += 1
    if _STATE["scores"] is None:
        raise HTTPException(status_code=503, detail="Scores not loaded. Run the pipeline.")
    target = f"github.com/{owner}/{name}".lower()
    df = _STATE["scores"]
    match = df[df["repo"].str.lower().str.contains(f"{owner}/{name}".lower(), regex=False)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"{owner}/{name} not in cohort.")
    row = match.iloc[0]
    return {"repo": str(row["repo"]), "originality": float(row["originality"])}
