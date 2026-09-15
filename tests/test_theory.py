"""Tests for the cost-benefit model of the slow loop."""

from __future__ import annotations

import pytest

from code.theory import CostModel, break_even_gain, supervision_pays, worst_case_loss


def test_break_even_gain_is_the_gap_the_inner_loop_earns_during_one_call():
    model = CostModel(fast_step_s=0.002, call_latency_s=2.0, baseline_gap_rate=0.01)
    assert break_even_gain(model) == pytest.approx(0.01 * 2.0 / 0.002)


def test_break_even_gain_grows_with_latency_and_with_the_baseline_rate():
    base = CostModel(fast_step_s=0.002, call_latency_s=2.0, baseline_gap_rate=0.01)
    slower_calls = CostModel(fast_step_s=0.002, call_latency_s=12.0, baseline_gap_rate=0.01)
    faster_inner_loop = CostModel(fast_step_s=0.002, call_latency_s=2.0, baseline_gap_rate=0.05)
    assert break_even_gain(slower_calls) > break_even_gain(base)
    assert break_even_gain(faster_inner_loop) > break_even_gain(base)


def test_supervision_pays_only_above_the_break_even_gain():
    model = CostModel(fast_step_s=0.002, call_latency_s=2.0, baseline_gap_rate=0.01)
    threshold = break_even_gain(model)
    assert supervision_pays(gain_per_call=threshold * 1.2, model=model) is True
    assert supervision_pays(gain_per_call=threshold * 0.8, model=model) is False


def test_worst_case_loss_is_bounded_by_calls_and_epsilon():
    assert worst_case_loss(n_calls=10, epsilon=0.01) == pytest.approx(0.1)
    assert worst_case_loss(n_calls=0, epsilon=0.5) == pytest.approx(0.0)
