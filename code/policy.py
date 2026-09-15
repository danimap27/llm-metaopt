"""Classical baseline policies trained on the same telemetry.

These policies answer the reviewer question "would a classical model trained on
the telemetry do just as well?". They consume exactly the same telemetry window
as the LLM and predict one of the discrete interventions of
``labeler.default_candidates()``.

Implemented without external ML dependencies: multinomial logistic regression
trained by full-batch gradient descent with L2 regularization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .labeler import Intervention

FEATURE_NAMES = (
    "improvement",
    "energy_std",
    "energy_slope_per_step",
    "grad_norm_last",
    "grad_norm_mean",
    "theta_var",
    "a_k_last",
    "c_k_last",
)


def window_features(window: Dict[str, object]) -> np.ndarray:
    """Fixed-length feature vector extracted from a telemetry window."""
    energy = window["energy"]
    grad = window["grad_norm"]
    eta = window["eta"]
    theta = window.get("theta", {})
    values = [
        float(window["improvement"]),
        float(energy["std"]),  # type: ignore[index]
        float(energy["slope_per_step"]),  # type: ignore[index]
        float(grad["last"]),  # type: ignore[index]
        float(grad["mean"]),  # type: ignore[index]
        float(theta.get("var", 0.0)),  # type: ignore[union-attr]
        float(eta["a_k_last"]),  # type: ignore[index]
        float(eta["c_k_last"]),  # type: ignore[index]
    ]
    return np.asarray(values, dtype=float)


def action_to_index(candidates: Sequence[Intervention]) -> Dict[str, int]:
    return {candidate.name: index for index, candidate in enumerate(candidates)}


@dataclass
class LogisticPolicy:
    """Multinomial logistic regression over the intervention grid."""

    candidates: List[Intervention]
    n_features: int = len(FEATURE_NAMES)
    weights: Optional[np.ndarray] = None  # shape (n_classes, n_features + 1)
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    classes: Optional[List[int]] = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            self.feature_std[self.feature_std == 0.0] = 1.0
        assert self.feature_mean is not None and self.feature_std is not None
        return (X - self.feature_mean) / self.feature_std

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[int],
        epochs: int = 800,
        lr: float = 0.2,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> "LogisticPolicy":
        rng = np.random.default_rng(seed)
        Xs = self._standardize(np.asarray(X, dtype=float), fit=True)
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        self.classes = sorted({int(label) for label in y})
        weights = 0.01 * rng.standard_normal((len(self.classes), Xb.shape[1]))
        y_arr = np.asarray([self.classes.index(int(label)) for label in y], dtype=int)
        one_hot = np.zeros((y_arr.size, len(self.classes)))
        one_hot[np.arange(y_arr.size), y_arr] = 1.0

        for _ in range(epochs):
            logits = Xb @ weights.T
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = (probs - one_hot).T @ Xb / Xb.shape[0] + l2 * weights
            weights -= lr * grad
        self.weights = weights
        return self

    def predict_index(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None and self.classes is not None
        Xs = self._standardize(np.atleast_2d(np.asarray(X, dtype=float)))
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        logits = Xb @ self.weights.T
        return np.asarray([self.classes[i] for i in logits.argmax(axis=1)])

    def predict(self, window: Dict[str, object]) -> Intervention:
        index = int(self.predict_index(window_features(window))[0])
        return self.candidates[index]

    def score(self, X: np.ndarray, y: Sequence[int]) -> float:
        preds = self.predict_index(X)
        return float(np.mean(preds == np.asarray(y)))

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "weights": None if self.weights is None else self.weights.tolist(),
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": None if self.feature_std is None else self.feature_std.tolist(),
            "classes": self.classes,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        candidates = [Intervention(**item) for item in payload["candidates"]]
        policy = cls(candidates=candidates)
        policy.weights = None if payload["weights"] is None else np.asarray(payload["weights"], dtype=float)
        policy.feature_mean = None if payload["feature_mean"] is None else np.asarray(payload["feature_mean"], dtype=float)
        policy.feature_std = None if payload["feature_std"] is None else np.asarray(payload["feature_std"], dtype=float)
        policy.classes = payload["classes"]
        return policy


@dataclass
class MajorityPolicy:
    """Predicts the most frequent action of the training set (sanity baseline)."""

    candidates: List[Intervention]
    majority_index: int = 0

    def fit(self, y: Sequence[int]) -> "MajorityPolicy":
        values, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
        self.majority_index = int(values[int(np.argmax(counts))])
        return self

    def predict(self, window: Dict[str, object]) -> Intervention:  # noqa: ARG002 - telemetry unused
        return self.candidates[self.majority_index]
