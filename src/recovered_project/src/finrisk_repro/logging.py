"""Known-propensity behavior policies and logged bandit samples.

The eight behavior-policy labels are preserved in the paper and manifest.
The exact temperature, exploration, concentration, and random-seed settings
were not preserved in the original source, so the clean-room defaults below
are documented as TODO assumptions rather than historical facts.
"""

from __future__ import annotations

import numpy as np


DEFAULT_SEED = 7  # TODO: exact original seed is not recoverable from the paper.
DEFAULT_LOOKBACK = 12  # TODO: exact behavior-policy window is not documented.


def _normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    weights = np.exp(np.clip(scores, -50, 50))
    return weights / weights.sum(axis=1, keepdims=True)


def behavior_probabilities(
    rewards: np.ndarray,
    name: str,
    seed: int = DEFAULT_SEED,
    lookback: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """Construct known propensities for one behavior-policy label.

    ``seed`` is accepted for API compatibility with the recovered workflow;
    the probability construction itself is deterministic.  The names and
    qualitative overlap ordering are evidence-backed, while numeric mixture
    weights and temperatures are reconstructed defaults.
    """

    del seed
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (periods, actions)")
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    periods, actions = rewards.shape
    probabilities = np.full((periods, actions), 1.0 / actions)
    for index in range(periods):
        history = rewards[max(0, index - lookback) : index]
        if history.size:
            mean = np.nan_to_num(np.nanmean(history, axis=0), nan=0.0)
            volatility = np.nan_to_num(np.nanstd(history, axis=0), nan=0.0) + 1e-6
        else:
            mean = np.zeros(actions)
            volatility = np.ones(actions)

        if name == "uniform_high_overlap":
            probabilities[index] = np.full(actions, 1.0 / actions)
        elif name == "softmax_high_entropy":
            # TODO: the original temperature is not available; 0.05 is the
            # clean-room value recorded in the prior reconstruction.
            base = _normalize(mean[None, :] / 0.05)[0]
            probabilities[index] = 0.75 * base + 0.25 / actions
        elif name == "epsilon_greedy_momentum":
            # TODO: exact epsilon is not preserved; 0.10 is the prior value.
            probabilities[index] = np.full(actions, 0.10 / actions)
            probabilities[index, int(np.nanargmax(mean))] += 0.90
        elif name == "risk_averse_low_volatility":
            base = _normalize((-volatility)[None, :] / 0.05)[0]
            probabilities[index] = 0.90 * base + 0.10 / actions
        elif name == "return_seeking_recent_winner":
            probabilities[index] = np.full(actions, 0.02 / actions)
            probabilities[index, int(np.nanargmax(mean))] += 0.98
        elif name == "market_state_mixed":
            signal = mean / volatility
            base = _normalize(signal[None, :] / 0.10)[0]
            probabilities[index] = 0.60 * base + 0.40 / actions
        elif name == "low_overlap_concentration":
            probabilities[index] = np.full(actions, 0.01 / actions)
            probabilities[index, int(np.nanargmax(mean))] += 0.99
        elif name == "adversarial_support_mismatch":
            probabilities[index] = np.full(actions, 0.005 / actions)
            probabilities[index, int(np.nanargmin(mean))] += 0.995
        else:
            raise ValueError(f"unknown behavior policy: {name}")

    probabilities = np.maximum(probabilities, 1e-8)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def sample_log(
    rewards: np.ndarray, behavior: str, seed: int = DEFAULT_SEED
) -> dict[str, np.ndarray]:
    """Sample actions and observed rewards while retaining propensities."""

    probabilities = behavior_probabilities(rewards, behavior, seed=seed)
    rng = np.random.default_rng(seed)
    actions = np.array(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=int,
    )
    observed = rewards[np.arange(len(actions)), actions]
    return {
        "actions": actions,
        "propensities": probabilities[np.arange(len(actions)), actions],
        "rewards": observed,
        "probs": probabilities,
    }
