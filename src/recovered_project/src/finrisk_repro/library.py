"""Candidate-policy library for the final FinRisk-OPE reconstruction.

The final manuscript gives two count constraints that the earlier clean-room
runner did not implement:

* 4,384 risk-OPE cases = 8 logging policies x 548 candidate policies;
* 30,688 return-OPE cases = 4,384 risk cases x 7 estimators.

The 548-policy total is exactly ``sum(actions + 6)`` over the 15 Table-I
tasks.  The manuscript names the candidate-policy families, but the original
policy-construction source is not present.  This module therefore restores
the count-preserving library and labels its six non-constant templates as
reconstruction proxies.  It must not be read as recovery of the deleted
policy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import PolicySpec, target_actions


@dataclass(frozen=True)
class CandidatePolicy:
    """One named target policy and its deterministic action sequence."""

    name: str
    family: str
    actions: np.ndarray
    provenance: str


# The paper lists constant sleeves plus six named non-constant families in
# the count-compatible candidate universe.  The exact mapping from these
# names to the deleted source code is unresolved; the windows below are
# explicit reconstruction defaults rather than historical hyperparameters.
PROXY_DYNAMIC_SPECS: tuple[tuple[str, str, int, float], ...] = (
    ("momentum_rotation", "momentum", 3, 1.0),
    ("momentum_rotation_long", "momentum", 60, 1.0),
    ("low_volatility_rotation", "low_vol", 12, 1.0),
    ("ridge_return_prediction_proxy", "momentum", 12, 1.0),
    ("ridge_risk_adjusted_prediction_proxy", "risk_adjusted", 12, 1.0),
    ("risk_parity_proxy", "low_vol", 60, 1.0),
)


def build_candidate_library(values: np.ndarray) -> list[CandidatePolicy]:
    """Build the count-preserving candidate library for one task.

    There is one constant-sleeve candidate for every action column and six
    dynamic proxy candidates.  Candidates are intentionally not deduplicated:
    the paper's case counts count declared policy instances, and distinct
    policy definitions remain distinct even when a short panel produces the
    same action sequence.
    """

    rewards = np.asarray(values, dtype=float)
    if rewards.ndim != 2:
        raise ValueError("values must have shape (periods, actions)")
    periods, action_count = rewards.shape
    candidates: list[CandidatePolicy] = []

    for action in range(action_count):
        candidates.append(
            CandidatePolicy(
                name=f"constant_sleeve_{action:03d}",
                family="constant sleeve",
                actions=np.full(periods, action, dtype=int),
                provenance="paper-named family; one instance per action column",
            )
        )

    for name, kind, window, alpha in PROXY_DYNAMIC_SPECS:
        spec = PolicySpec(name=name, kind=kind, window=window, alpha=alpha)
        candidates.append(
            CandidatePolicy(
                name=name,
                family=name.removesuffix("_proxy"),
                actions=target_actions(rewards, spec),
                provenance=(
                    "paper-named family with clean-room proxy; exact source "
                    "construction unresolved"
                ),
            )
        )

    expected = action_count + len(PROXY_DYNAMIC_SPECS)
    if len(candidates) != expected:
        raise AssertionError("candidate library count invariant failed")
    return candidates


def candidate_count_by_task(tasks: list[tuple[object, object]]) -> int:
    """Return the total candidate-policy count implied by Table-I actions."""

    return sum(int(panel.n_actions) + len(PROXY_DYNAMIC_SPECS) for _, panel in tasks)

