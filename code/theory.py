"""Cost-benefit model of the slow loop and the safeguard bound.

Supervision costs wall-clock time per call and pays back gap reduction. Writing
the trade-off explicitly turns "is this worth it" into a measurable condition,
reported in the paper as a break-even surface, and gives the worst-case bound
for the safeguard that reverts a harmful intervention.

The key observation is that the call frequency cancels out of the rate
comparison. A call occupies ``call_latency_s`` seconds, during which the inner
loop would have reduced the gap by

    break_even_gain = baseline_gap_rate * call_latency_s / fast_step_s

so a call pays for itself whenever the gap it recovers exceeds that amount,
regardless of how often the loop calls the controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Measured constants of one experimental configuration."""

    fast_step_s: float
    call_latency_s: float
    baseline_gap_rate: float  # gap reduction per fast step of the unmodified optimizer


def break_even_gain(model: CostModel) -> float:
    """Gap reduction one call must produce to pay for the time the call costs."""
    return model.baseline_gap_rate * model.call_latency_s / model.fast_step_s


def supervision_pays(gain_per_call: float, model: CostModel) -> bool:
    """Whether a call that recovers ``gain_per_call`` beats the unmodified loop."""
    return float(gain_per_call) > break_even_gain(model)


def worst_case_loss(n_calls: int, epsilon: float) -> float:
    """Bound on the total damage when every intervention is reverted after epsilon.

    With the no-intervention safeguard, a call can degrade the gap by at most
    ``epsilon`` before the revert fires, so the loss over a run is bounded by
    ``n_calls * epsilon``.
    """
    return float(n_calls) * float(epsilon)
