"""Apollo's LogisticRegressionDetector, one probe per layer.

Repo (detectors.py) settles the paper's gaps:
  StandardScaler on train acts -> LogisticRegression(C=1/reg_coeff, fit_intercept=False,
  random_state=42), sklearn default solver (lbfgs) and max_iter (100). reg_coeff = 10 -> C = 0.1.
  Positive class = deceptive, so higher score = more deceptive.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_layer(X: np.ndarray, y: np.ndarray, reg_coeff: float = 10.0, max_iter: int = 100):
    scaler = StandardScaler().fit(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = LogisticRegression(C=1 / reg_coeff, fit_intercept=False, random_state=42,
                                max_iter=max_iter)
        lr.fit(scaler.transform(X), y)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return {
        "mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "w": torch.tensor(lr.coef_[0], dtype=torch.float32),
        "converged": converged,
    }


def score_tokens(probe: dict, acts: torch.Tensor) -> np.ndarray:
    """acts: [n_tok, d] -> per-token scores [n_tok]."""
    return (((acts - probe["mean"]) / probe["scale"]) @ probe["w"]).numpy()
