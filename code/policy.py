"""Classical controllers that consume the same telemetry window as the LLM.

``window_features`` flattens exactly the JSON window the language model reads,
so every classical controller works from the same information. Two model
classes are provided:

- ``LogisticPolicy``: multinomial logistic regression on the argmax label.
- ``EffectRidgePolicy``: one ridge regression per candidate action that
  predicts the counterfactual effect (baseline gap minus candidate gap) from
  the window context. This is the offline contextual-bandit estimator, the
  natural classical model class for a choose-one-of-k decision from a
  telemetry vector, and it uses the effect magnitudes, not just the argmax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .labeler import Intervention

# Flattening order of the window the LLM sees. Keep in sync with
# telemetry.build_window: same fields, same order.
SCALAR_FIELDS = (
    "energy.first",
    "energy.last",
    "energy.min",
    "energy.mean",
    "energy.std",
    "energy.slope_per_step",
    "improvement",
    "grad_norm.last",
    "grad_norm.mean",
    "grad_norm.max",
    "eta.a_k_last",
    "eta.c_k_last",
    "eta.eta_scale",
    "theta.var",
    "theta.mean_abs",
    "theta.max_abs",
    "theta.dim",
    "progress.drop_from_start",
    "progress.gap_above_best_so_far",
)


def _nested_get(window: Dict[str, Any], dotted: str) -> float:
    node: Any = window
    for part in dotted.split("."):
        node = node[part]
    return float(node)


def window_features(window: Dict[str, Any]) -> np.ndarray:
    """Flatten the full telemetry window into a fixed-order feature vector.

    The energy series is included verbatim, so the vector carries exactly the
    same information the language model receives (the LLM window minus the
    purely descriptive n_window and step_end fields).
    """
    series = [float(v) for v in window.get("energy_series", [])]
    scalars = [_nested_get(window, key) for key in SCALAR_FIELDS]
    return np.asarray(series + scalars, dtype=float)


def feature_names(max_series: int = 64) -> List[str]:
    return [f"energy_series[{i}]" for i in range(max_series)] + list(SCALAR_FIELDS)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


class LogisticPolicy:
    """Multinomial logistic policy trained on counterfactual labels."""

    def __init__(self, candidates: Optional[Sequence[Intervention]] = None) -> None:
        self.candidates: List[Intervention] = list(candidates) if candidates else [
            Intervention(),
            Intervention(eta_scale=0.5),
            Intervention(eta_scale=2.0),
            Intervention(noise_sigma=0.05),
            Intervention(noise_sigma=0.15),
            Intervention(eta_scale=2.0, noise_sigma=0.05),
            Intervention(restart=True),
        ]
        self.weights: Optional[np.ndarray] = None  # shape (n_actions, n_features)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.5,
        epochs: int = 200,
        seed: Optional[int] = None,
    ) -> "LogisticPolicy":
        # seed is accepted for API stability; training is deterministic from a
        # zero-initialized weight matrix.
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        n_actions = len(self.candidates)
        w = np.zeros((n_actions, x.shape[1]))
        eye = np.eye(n_actions)
        for _ in range(int(epochs)):
            probs = _softmax_batch(x @ w.T)
            grad = (probs - eye[y]) .T @ x / max(len(x), 1)
            w -= lr * grad
        self.weights = w
        return self

    def predict(self, window: Dict[str, Any]) -> Intervention:
        if self.weights is None:
            raise ValueError("LogisticPolicy.predict called before fit")
        x = window_features(window)
        if x.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"feature dimension {x.shape[0]} does not match the trained model {self.weights.shape[1]}"
            )
        logits = self.weights @ x
        return self.candidates[int(np.argmax(logits))]

    def score(self, features: np.ndarray, labels: Sequence[int]) -> float:
        """Fraction of argmax predictions that match the labels."""
        if self.weights is None:
            raise ValueError("LogisticPolicy.score called before fit")
        x = np.asarray(features, dtype=float)
        probs = _softmax_batch(x @ self.weights.T)
        return float(np.mean(np.argmax(probs, axis=1) == np.asarray(labels, dtype=int)))

    def save(self, path: Path) -> None:
        if self.weights is None:
            raise ValueError("LogisticPolicy.save called before fit")
        payload = {
            "kind": "logistic",
            "weights": self.weights.tolist(),
            "candidates": [c.to_dict() for c in self.candidates],
            "feature_names": feature_names(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = [Intervention.from_dict(a) for a in payload["candidates"]]
        policy = cls(candidates=candidates)
        policy.weights = np.asarray(payload["weights"], dtype=float)
        return policy


def _softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MajorityPolicy:
    """Baseline controller that always plays the most common training action."""

    def __init__(self, candidates: Optional[Sequence[Intervention]] = None) -> None:
        self.candidates: List[Intervention] = list(candidates) if candidates else LogisticPolicy().candidates
        self.index_: Optional[int] = None

    def fit(self, labels: Sequence[int]) -> "MajorityPolicy":
        counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(self.candidates))
        self.index_ = int(np.argmax(counts))
        return self

    def predict(self, window: Dict[str, Any]) -> Intervention:
        if self.index_ is None:
            raise ValueError("MajorityPolicy.predict called before fit")
        return self.candidates[self.index_]


class EffectRidgePolicy:
    """Offline contextual bandit: per-action ridge models of the counterfactual effect.

    Fits one ridge regression per candidate action predicting the effect of
    that action (baseline gap minus candidate gap) from the window context.
    At decision time it picks the action with the largest predicted effect.
    Trained offline on the counterfactual labels, it is the classical
    counterpart of the language-model controller under identical information
    and an identical action grid.
    """

    def __init__(
        self,
        candidates: Optional[Sequence[Intervention]] = None,
        alpha: float = 1.0,
    ) -> None:
        self.candidates: List[Intervention] = list(candidates) if candidates else LogisticPolicy().candidates
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None  # shape (n_actions, n_features + 1) with intercept last

    def fit(self, features: np.ndarray, effects: np.ndarray) -> "EffectRidgePolicy":
        """effects[i, a] is baseline_gap - candidate_gap for sample i, action a."""
        x = np.asarray(features, dtype=float)
        e = np.asarray(effects, dtype=float)
        n_actions = len(self.candidates)
        x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
        reg = self.alpha * np.eye(x_aug.shape[1])
        coef = np.zeros((n_actions, x_aug.shape[1]))
        for a in range(n_actions):
            coef[a] = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ e[:, a])
        self.coef_ = coef
        return self

    def predict(self, window: Dict[str, Any]) -> Intervention:
        if self.coef_ is None:
            raise ValueError("EffectRidgePolicy.predict called before fit")
        x = window_features(window)
        if x.shape[0] != self.coef_.shape[1] - 1:
            raise ValueError(
                f"feature dimension {x.shape[0]} does not match the trained model {self.coef_.shape[1] - 1}"
            )
        x_aug = np.append(x, 1.0)
        return self.candidates[int(np.argmax(self.coef_ @ x_aug))]

    def save(self, path: Path) -> None:
        if self.coef_ is None:
            raise ValueError("EffectRidgePolicy.save called before fit")
        payload = {
            "kind": "effect_ridge",
            "coef": self.coef_.tolist(),
            "alpha": self.alpha,
            "candidates": [c.to_dict() for c in self.candidates],
            "feature_names": feature_names(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EffectRidgePolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = [Intervention.from_dict(a) for a in payload["candidates"]]
        policy = cls(candidates=candidates, alpha=float(payload.get("alpha", 1.0)))
        policy.coef_ = np.asarray(payload["coef"], dtype=float)
        return policy


def load_policy(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "effect_ridge":
        return EffectRidgePolicy.load(path)
    return LogisticPolicy.load(path)
