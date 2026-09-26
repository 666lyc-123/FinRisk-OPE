"""Deterministic target-policy templates preserved by the clean-room run.

The paper text names a broader target-policy library, but the deleted source
does not preserve its implementation.  The templates below are the concrete
ones recorded in the manifest's later executable reconstruction.  They are
kept separate from any claim that the original paper used exactly these
policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    window: int = 12
    alpha: float = 1.0


# TODO: the paper-level policy names (constant sleeves, 60/40, equity-heavy,
# risk-parity, momentum rotation, low-volatility rotation, ridge return, and
# ridge risk-adjusted prediction) do not have recoverable executable details.
# These are the eight templates actually present in the clean-room run.
POLICY_SPECS: tuple[PolicySpec, ...] = (
    PolicySpec("constant_first", "constant", 1),
    PolicySpec("constant_last", "constant", 1),
    PolicySpec("constant_mean", "constant", 1),
    PolicySpec("momentum_3", "momentum", 3),
    PolicySpec("momentum_12", "momentum", 12),
    PolicySpec("momentum_60", "momentum", 60),
    PolicySpec("low_volatility_12", "low_vol", 12),
    PolicySpec("mean_risk_adjusted_12", "risk_adjusted", 12, 1.0),
)


def target_actions(rewards: np.ndarray, policy: PolicySpec) -> np.ndarray:
    """Generate deterministic actions using only rewards before each period."""

    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (periods, actions)")
    periods, actions_count = rewards.shape
    actions = np.zeros(periods, dtype=int)

    for index in range(periods):
        if policy.kind == "constant":
            constant_index = {
                "constant_first": 0,
                "constant_last": actions_count - 1,
                "constant_mean": actions_count // 2,
            }
            if policy.name not in constant_index:
                raise ValueError(f"unknown constant policy: {policy.name}")
            actions[index] = constant_index[policy.name]
            continue

        history = rewards[max(0, index - policy.window) : index]
        if history.size == 0:
            actions[index] = 0
            continue

        means = np.nan_to_num(np.nanmean(history, axis=0), nan=0.0)
        volatility = np.nan_to_num(np.nanstd(history, axis=0), nan=0.0) + 1e-8
        if policy.kind == "momentum":
            score = means
        elif policy.kind == "low_vol":
            score = -volatility
        elif policy.kind == "risk_adjusted":
            score = policy.alpha * means / volatility
        else:
            raise ValueError(f"unknown target-policy kind: {policy.kind}")
        actions[index] = int(np.nanargmax(score))

    return actions
