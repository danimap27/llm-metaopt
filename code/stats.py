"""Statistics used in the paper tables.

Self-contained implementations (no SciPy dependency): bootstrap confidence
intervals, sign-flip permutation tests for paired comparisons and
Holm-Bonferroni correction for the family of comparisons against the baseline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval of ``statistic``."""
    data = _as_array(values)
    if data.size == 0:
        raise ValueError("bootstrap_ci received an empty sample")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(n_boot, data.size))
    estimates = np.asarray([statistic(data[draw]) for draw in draws], dtype=float)
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n": int(data.size),
        "mean": float(statistic(data)),
        "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": float(alpha),
    }


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sided sign-flip permutation test for paired samples.

    Assumption-free alternative to the paired t-test: under the null hypothesis
    the sign of every paired difference is exchangeable.
    """
    a_arr, b_arr = _as_array(a), _as_array(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired_permutation_test requires equally sized samples")
    diffs = a_arr - b_arr
    n = diffs.size
    if n == 0:
        raise ValueError("paired_permutation_test received empty samples")
    observed = abs(float(np.mean(diffs)))
    if observed == 0.0:
        return {"n": int(n), "observed_diff": 0.0, "p_value": 1.0, "n_perm": int(n_perm)}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    permuted = np.abs((signs * diffs).mean(axis=1))
    p_value = float((np.sum(permuted >= observed) + 1) / (n_perm + 1))
    return {
        "n": int(n),
        "observed_diff": float(np.mean(diffs)),
        "p_value": p_value,
        "n_perm": int(n_perm),
    }


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference for paired samples (Cohen's d_z)."""
    diffs = _as_array(a) - _as_array(b)
    if diffs.size < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs) / sd)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict[str, object]]:
    """Holm-Bonferroni step-down correction over a family of p-values."""
    p = _as_array(p_values)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return [
        {"index": int(i), "p_value": float(p[i]), "p_adjusted": float(adjusted[i]), "reject": bool(adjusted[i] <= alpha)}
        for i in range(m)
    ]
