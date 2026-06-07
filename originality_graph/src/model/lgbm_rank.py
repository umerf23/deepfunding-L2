"""Optional LightGBM ranking head.

The contest is effectively a ranking problem: what matters is the relative order
of originality, not absolute calibration. When config.model.use_lgbm_rank is true,
this fits a LightGBM regressor on the rank-normalized centrality features against
the sample labels and blends its prediction with the centrality score.

It is OFF by default for the same reason as Solution 1: the sample labels look
synthetic. The path is fully implemented and cross-validated, not a stub.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False

from ..utils.common import setup_logging

logger = setup_logging()

_RANK_FEATURES = ("authority_n", "in_degree_n", "dependent_count_n", "out_degree_n")


class LGBMRankHead:
    """LightGBM blend over centrality features."""

    def __init__(self, config: dict[str, Any]) -> None:
        if not _LGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is required for the ranking head. "
                "Install it or keep model.use_lgbm_rank false."
            )
        m = config["model"]
        self.params = {
            "n_estimators": m["n_estimators"],
            "max_depth": m["max_depth"],
            "learning_rate": m["learning_rate"],
            "subsample": m["subsample"],
            "colsample_bytree": m["colsample_bytree"],
            "random_state": m["random_state"],
            "n_jobs": config["runtime"]["parallel_workers"],
            "verbose": -1,
        }
        self.cv_folds: int = m["cv_folds"]
        self.model: LGBMRegressor | None = None

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        return frame[list(_RANK_FEATURES)].astype(float).to_numpy()

    def cross_validate(self, frame: pd.DataFrame, labels: np.ndarray) -> float:
        X, y = self._matrix(frame), labels.astype(float)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.params["random_state"])
        rmses = []
        for tr, va in kf.split(X):
            model = LGBMRegressor(**self.params)
            model.fit(X[tr], y[tr])
            rmses.append(float(np.sqrt(mean_squared_error(y[va], model.predict(X[va])))))
        mean_rmse = float(np.mean(rmses))
        logger.info("LGBM rank-head CV RMSE: %.4f (+/- %.4f)", mean_rmse, float(np.std(rmses)))
        return mean_rmse

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> "LGBMRankHead":
        self.model = LGBMRegressor(**self.params)
        self.model.fit(self._matrix(frame), labels.astype(float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Rank head must be fit before prediction.")
        return np.clip(self.model.predict(self._matrix(frame)), 0.0, 1.0)
