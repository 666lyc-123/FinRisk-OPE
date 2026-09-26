"""Return- and risk-oriented off-policy evaluation utilities.

The estimator names and core formulas are explicit in the paper.  Exact
reward-model fitting, clipping, bootstrap, and calibrated interval details
were not preserved, so those choices are isolated and marked as reconstructed
defaults below.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def _weights(log: dict[str, np.ndarray], target_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(log["actions"])
    propensities = np.asarray(log["propensities"], dtype=float)
    return (actions == target_actions).astype(float) / np.maximum(propensities, 1e-8)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def cvar95_loss(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Return weighted empirical 95% CVaR of the loss ``-values``."""

    values = np.asarray(values, dtype=float)
    losses = -values
    finite = np.isfinite(losses)
    losses = losses[finite]
    if weights is None:
        weights_array = np.ones_like(losses)
    else:
        weights_array = np.asarray(weights, dtype=float)[finite]
    if len(losses) == 0 or np.sum(weights_array) <= 0:
        return float("nan")

    order = np.argsort(losses)
    losses = losses[order]
    weights_array = weights_array[order]
    weights_array = weights_array / weights_array.sum()

    cutoff = 0.95
    tail_mass = 1.0 - cutoff
    tail_sum = 0.0
    cumulative = 0.0
    for loss, weight in zip(losses, weights_array):
        if cumulative + weight <= cutoff:
            cumulative += weight
            continue
        included = min(weight, cumulative + weight - cutoff)
        tail_sum += loss * included
        cumulative += weight
        if cumulative >= 1.0:
            break
    return float(tail_sum / tail_mass)


def bootstrap_weighted_cvar_ci(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a paired row-bootstrap interval for weighted CVaR95.

    The same sampled row indices are applied to the observed values and their
    importance weights.  Sorting is performed once on the original loss
    values; bootstrap multiplicities are represented by frequency weights.
    This is computationally equivalent to resampling rows and then computing
    the weighted empirical tail mean, while avoiding repeated large sorts.

    TODO: the manuscript does not preserve the original replicate count,
    random seed, or block structure.  Those values remain explicit inputs.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have equal shape")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[finite]
    weights = weights[finite]
    if len(values) == 0 or np.sum(weights) <= 0:
        return float("nan"), float("nan")

    losses = -values
    order = np.argsort(losses)
    sorted_losses = losses[order]
    sorted_weights = weights[order]
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(values), size=len(values))
        frequencies = np.bincount(sampled, minlength=len(values)).astype(float)
        bootstrap_weights = sorted_weights * frequencies[order]
        total = float(np.sum(bootstrap_weights))
        if total <= 0:
            continue
        tail_mass = 0.05 * total
        reverse_losses = sorted_losses[::-1]
        reverse_weights = bootstrap_weights[::-1]
        cumulative_before = np.cumsum(reverse_weights) - reverse_weights
        included = np.minimum(
            reverse_weights,
            np.maximum(0.0, tail_mass - cumulative_before),
        )
        if included.sum() > 0:
            estimates.append(float(np.sum(reverse_losses * included) / tail_mass))

    if not estimates:
        return float("nan"), float("nan")
    draws = np.asarray(estimates, dtype=float)
    return (
        float(np.nanquantile(draws, alpha / 2)),
        float(np.nanquantile(draws, 1 - alpha / 2)),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a paired-row bootstrap interval for a mean."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, len(values), size=len(values))
        draws[index] = float(np.mean(values[sampled]))
    return (
        float(np.nanquantile(draws, alpha / 2)),
        float(np.nanquantile(draws, 1 - alpha / 2)),
    )


def fit_reward_model(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Fit one action-conditional Ridge model and return all predictions.

    TODO: the paper does not expose the original model family, alpha, split,
    or cross-fitting scheme.  The alpha and sparse-action fallback below are
    retained from the clean-room implementation and must not be read as
    historical hyperparameters.
    """

    contexts = np.asarray(contexts, dtype=float)
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    predictions = np.zeros((len(contexts), n_actions), dtype=float)
    global_mean = float(np.nanmean(rewards))
    for action in range(n_actions):
        mask = actions == action
        if mask.sum() < max(5, contexts.shape[1] + 1):
            predictions[:, action] = global_mean
            continue
        model = Ridge(alpha=1.0)
        model.fit(contexts[mask], rewards[mask])
        predictions[:, action] = model.predict(contexts)
    return predictions


def estimate_all(
    contexts: np.ndarray,
    log: dict[str, np.ndarray],
    target_actions: np.ndarray,
    n_actions: int,
    clip: float = 20.0,
    dm_preds: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the recovered return/risk diagnostic metrics.

    The estimator names follow the paper: DM, IPS, SNIPS, clipped IPS, DR,
    WDR, and Switch-DR, with ESS, support gap, weighted CVaR95, and normal
    approximation intervals.  TODO: exact clip, bootstrap, cross-fitting,
    and calibrated-UCB rules are not recoverable from the source history.
    """

    actions = np.asarray(log["actions"], dtype=int)
    rewards = np.asarray(log["rewards"], dtype=float)
    target_actions = np.asarray(target_actions, dtype=int)
    weights = _weights(log, target_actions)
    clipped = np.minimum(weights, clip)

    ips = float(np.mean(weights * rewards))
    snips = float(np.sum(weights * rewards) / np.sum(weights)) if np.sum(weights) > 0 else 0.0
    clipped_ips = float(np.mean(clipped * rewards))
    if dm_preds is None:
        dm_preds = fit_reward_model(contexts, actions, rewards, n_actions)

    index = np.arange(len(target_actions))
    target_prediction = dm_preds[index, target_actions]
    logged_prediction = dm_preds[index, actions]
    dr_contribution = target_prediction + weights * (rewards - logged_prediction)
    dr = float(np.mean(dr_contribution))
    residual = weights * (rewards - logged_prediction)
    wdr = (
        float(np.mean(target_prediction) + np.sum(residual) / np.sum(weights))
        if np.sum(weights) > 0
        else float(np.mean(target_prediction))
    )
    switch = float(
        np.mean(
            np.where(
                weights <= clip,
                target_prediction + weights * (rewards - logged_prediction),
                target_prediction,
            )
        )
    )

    dr_sd = float(np.nanstd(dr_contribution, ddof=1)) if len(dr_contribution) > 1 else 0.0
    dr_se = dr_sd / np.sqrt(max(1, len(dr_contribution)))
    ess = effective_sample_size(weights)
    support_se = dr_sd / np.sqrt(max(1.0, ess))
    return {
        "DM": float(np.mean(target_prediction)),
        "IPS": ips,
        "SNIPS": snips,
        "ClippedIPS": clipped_ips,
        "DR": dr,
        "WDR": wdr,
        "SwitchDR": switch,
        "ESS": ess,
        "max_weight": float(np.max(weights)) if len(weights) else 0.0,
        "support_gap": float(1.0 - min(1.0, ess / max(1, len(weights)))),
        "CVaR95": cvar95_loss(rewards, weights),
        "DR_se": dr_se,
        "DR_lcb": dr - 1.96 * dr_se,
        "DR_ucb": dr + 1.96 * dr_se,
        "support_cal_lcb": dr - 1.96 * support_se,
        "support_cal_ucb": dr + 1.96 * support_se,
    }


def bootstrap_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 7,
) -> tuple[float, float]:
    """Return an empirical bootstrap interval for a mean.

    TODO: the paper says paired bootstrap, but the original resampling unit,
    block structure, replicate count, and seed are not preserved.  This is the
    iid bootstrap used by the clean-room implementation and is not claimed to
    reproduce the paper's calibrated interval.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        draws[index] = np.nanmean(sample)
    return (
        float(np.nanquantile(draws, alpha / 2)),
        float(np.nanquantile(draws, 1 - alpha / 2)),
    )
