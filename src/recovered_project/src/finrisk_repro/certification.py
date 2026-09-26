"""Selective risk certification and false-safe diagnostics.

The paper defines ACCEPT/REJECT/ABSTAIN using return, risk, and support
evidence.  The deleted implementation of its calibrated confidence bounds is
not available; this file restores the transparent empirical approximation
that was used in the later clean-room run and labels its thresholds as TODO
assumptions.
"""

from __future__ import annotations

import numpy as np


def classify_selector(
    estimates: list[dict[str, float]],
    full_rewards: np.ndarray,
    target_actions: list[np.ndarray],
    tau: float = 0.10,
    eta: float = 10.0,
    omega: float = 50.0,
    gamma: float = 0.995,
) -> dict[str, float | int | str]:
    """Select one policy and classify it as ACCEPT, REJECT, or ABSTAIN.

    TODO: the original numerical values for ``tau``, ``eta``, ``omega``, and
    ``gamma`` are not preserved.  The defaults are the clean-room values.  The
    return and risk confidence margins are empirical approximations rather
    than the paper's unrecovered paired/bootstrap calibration.
    """

    if not estimates:
        raise ValueError("estimates must not be empty")
    if len(estimates) != len(target_actions):
        raise ValueError("estimates and target_actions must have equal length")

    candidates: list[tuple[int, float, float, bool, bool]] = []
    for index, estimate in enumerate(estimates):
        return_lcb = estimate["DR"] - 1.96 * max(
            abs(estimate["DR"] - estimate["DM"]), 1e-4
        )
        risk_ucb = estimate["CVaR95"] + 1.96 * max(
            abs(estimate["CVaR95"]) * 0.10, 1e-4
        )
        support_ok = (
            estimate["ESS"] >= eta
            and estimate["max_weight"] <= omega
            and estimate["support_gap"] <= gamma
        )
        risk_ok = risk_ucb <= tau
        candidates.append((index, return_lcb, risk_ucb, support_ok, risk_ok))

    accepted = [candidate for candidate in candidates if candidate[3] and candidate[4]]
    if accepted:
        selected, _, risk_ucb, _, _ = max(accepted, key=lambda item: item[1])
        decision = "ACCEPT"
    elif any(candidate[3] for candidate in candidates):
        supported = [candidate for candidate in candidates if candidate[3]]
        selected, _, risk_ucb, _, _ = max(supported, key=lambda item: item[1])
        decision = "REJECT"
    else:
        selected, _, risk_ucb, _, _ = max(candidates, key=lambda item: item[1])
        decision = "ABSTAIN"

    chosen_actions = np.asarray(target_actions[selected], dtype=int)
    full_rewards = np.asarray(full_rewards, dtype=float)
    realized = full_rewards[np.arange(len(chosen_actions)), chosen_actions]
    realized_cvar = float(-np.nanquantile(realized, 0.05))
    false_safe = int(decision == "ACCEPT" and realized_cvar > tau)
    return {
        "decision": decision,
        "selected_policy": selected,
        "estimated_cvar_ucb": float(risk_ucb),
        "realized_cvar": realized_cvar,
        "false_safe": false_safe,
    }
