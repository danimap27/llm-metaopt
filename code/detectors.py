"""Classical drift detectors used as baselines for the slow loop.

Two dependency-free detectors:

- ``PageHinkley``: the two-sided cumulative-sum test of Page (1954) as used for
  concept drift by Gama et al. (2004), with an exponentially weighted mean. Both
  directions are tracked so a downward shift in the energy is caught as fast as
  an upward one.
- ``WindowedMeanShift``: a two-sample mean comparison between the most recent
  window and the window before it, with a normal-approximation statistic. It is
  the simple and interpretable challenger to the language model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class PageHinkley:
    """Two-sided cumulative-sum change detector."""

    delta: float = 0.005
    threshold: float = 0.5
    alpha: float = 0.999
    _mean: float = 0.0
    _cumulative_up: float = 0.0
    _cumulative_down: float = 0.0
    _n: int = 0

    def update(self, value: float) -> bool:
        """Feed one sample and report whether the detector alarms."""
        self._n += 1
        if self._n == 1:
            self._mean = float(value)
        self._mean = self.alpha * self._mean + (1.0 - self.alpha) * float(value)
        self._cumulative_up = max(0.0, self._cumulative_up + (float(value) - self._mean - self.delta))
        self._cumulative_down = max(0.0, self._cumulative_down + (self._mean - float(value) - self.delta))
        return self._cumulative_up > self.threshold or self._cumulative_down > self.threshold


@dataclass
class WindowedMeanShift:
    """Two-sample mean comparison between adjacent windows."""

    window: int = 20
    threshold: float = 3.0
    min_samples: int = 20
    cooldown: int = 10
    _history: List[float] = field(default_factory=list)
    _last_alarm: int = -10**9

    def update(self, value: float) -> bool:
        """Feed one sample and report whether the detector alarms."""
        self._history.append(float(value))
        if len(self._history) < max(self.min_samples, 2 * self.window):
            return False
        recent = np.asarray(self._history[-self.window :], dtype=float)
        previous = np.asarray(self._history[-2 * self.window : -self.window], dtype=float)
        pooled = float(
            np.sqrt(recent.var(ddof=1) / self.window + previous.var(ddof=1) / self.window)
        ) or 1e-12
        statistic = abs(float(recent.mean() - previous.mean())) / pooled
        index = len(self._history)
        if statistic > self.threshold and index - self._last_alarm > self.cooldown:
            self._last_alarm = index
            return True
        return False
