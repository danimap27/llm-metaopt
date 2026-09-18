"""Classical drift detectors used as interpretable challengers.

The Page-Hinkley test follows Gama et al. (2004) but runs on residuals from a
local least-squares slope instead of the raw series, so ordinary monotone
convergence is not a change point. The cumulative deviation of the residual
from its running mean, minus a small allowed drift per sample, is tracked
against its historical minimum in both directions. A sustained level shift up
or down alarms; a smooth exponential descent does not accumulate enough
deviation to cross the threshold (adversarial review finding B2).
"""

from __future__ import annotations

from typing import List, Optional


class PageHinkley:
    """Two-sided Page-Hinkley on residuals from a local linear slope."""

    def __init__(
        self,
        threshold: float = 0.05,
        delta: float = 0.005,
        min_count: int = 5,
        window: int = 8,
    ) -> None:
        self.threshold = float(threshold)
        self.delta = float(delta)
        self.min_count = int(min_count)
        self.window = int(window)
        self._values: List[float] = []
        self._mean = 0.0
        self._count = 0
        self._m_up = 0.0
        self._min_up = 0.0
        self._m_down = 0.0
        self._min_down = 0.0

    def _residual(self, x: float) -> Optional[float]:
        self._values.append(float(x))
        if len(self._values) < self.window:
            return None
        tail = self._values[-self.window :]
        n = len(tail)
        idx = list(range(n))
        mean_i = (n - 1) / 2.0
        var_i = sum((i - mean_i) ** 2 for i in idx)
        mean_x = sum(tail) / n
        slope = sum((i - mean_i) * (x_i - mean_x) for i, x_i in zip(idx, tail)) / var_i
        predicted = mean_x + slope * (n - 1 - mean_i)
        return float(x) - predicted

    def update(self, x: float) -> bool:
        residual = self._residual(x)
        if residual is None:
            return False
        self._count += 1
        self._mean += (residual - self._mean) / self._count
        self._m_up += residual - self._mean - self.delta
        if self._m_up < self._min_up:
            self._min_up = self._m_up
        self._m_down += -residual + self._mean - self.delta
        if self._m_down < self._min_down:
            self._min_down = self._m_down
        if self._count < self.min_count:
            return False
        return (self._m_up - self._min_up) > self.threshold or (self._m_down - self._min_down) > self.threshold


class WindowedMeanShift:
    """Z-scored mean-shift detector over a sliding window with a cooldown.

    Once at least ``min_samples + window`` values exist, compares the mean of
    the last ``window`` values against the mean of everything before them,
    standardized by the standard deviation of the preceding block. The alarm
    respects a cooldown measured in samples.
    """

    def __init__(
        self,
        window: int = 10,
        threshold: float = 3.0,
        min_samples: int = 10,
        cooldown: int = 10,
    ) -> None:
        self.window = int(window)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.cooldown = int(cooldown)
        self._values: List[float] = []
        self._last_alarm: int = -self.cooldown

    def update(self, x: float, step: Optional[int] = None) -> bool:
        self._values.append(float(x))
        n = len(self._values)
        if n < self.min_samples + self.window:
            return False
        recent = self._values[-self.window :]
        previous = self._values[: -self.window]
        scale = float(np_std(previous)) + 1e-12
        z = (sum(recent) / len(recent) - sum(previous) / len(previous)) / scale
        if z > self.threshold and (n - self._last_alarm) >= self.cooldown:
            self._last_alarm = n
            return True
        return False


def np_std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
