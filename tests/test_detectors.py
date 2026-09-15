"""Tests for the classical drift detectors used as fair baselines."""

from __future__ import annotations

import numpy as np

from code.detectors import PageHinkley, WindowedMeanShift


def test_page_hinkley_ignores_a_stationary_series():
    detector = PageHinkley(delta=0.01, threshold=0.5)
    rng = np.random.default_rng(0)
    alarms = [detector.update(float(value)) for value in rng.normal(0.0, 1e-3, size=50)]
    assert not any(alarms)


def test_page_hinkley_flags_a_downward_level_shift():
    detector = PageHinkley(delta=0.01, threshold=0.5)
    rng = np.random.default_rng(0)
    for value in rng.normal(0.0, 1e-3, size=20):
        detector.update(float(value))
    alarms = [detector.update(float(value)) for value in rng.normal(-2.0, 1e-3, size=20)]
    assert any(alarms)


def test_page_hinkley_flags_an_upward_level_shift():
    detector = PageHinkley(delta=0.01, threshold=0.5)
    rng = np.random.default_rng(3)
    for value in rng.normal(0.0, 1e-3, size=20):
        detector.update(float(value))
    alarms = [detector.update(float(value)) for value in rng.normal(2.0, 1e-3, size=20)]
    assert any(alarms)


def test_windowed_mean_shift_detects_a_shift_and_respects_the_cooldown():
    detector = WindowedMeanShift(window=10, threshold=3.0, min_samples=10, cooldown=10)
    rng = np.random.default_rng(1)
    series = list(rng.normal(0.0, 0.02, size=30)) + list(rng.normal(1.0, 0.02, size=30))
    alarms = [index for index, value in enumerate(series) if detector.update(float(value))]
    assert alarms, "the detector should fire after the shift"
    assert 30 <= alarms[0] <= 35
    assert len(alarms) <= 3


def test_windowed_mean_shift_stays_silent_on_a_stationary_series():
    detector = WindowedMeanShift(window=10, threshold=3.0, min_samples=10)
    rng = np.random.default_rng(2)
    assert not any(detector.update(float(value)) for value in rng.normal(0.0, 0.02, size=80))
