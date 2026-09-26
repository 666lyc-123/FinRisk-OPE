"""Lagged context construction for the reconstructed benchmark.

The paper reports 20 monthly features and 21 daily features, but the complete
feature formula and rolling-window specification were not preserved.  This
module therefore exposes the clean-room feature builder with those facts
clearly separated from its reconstruction assumptions.
"""

from __future__ import annotations

import numpy as np


def build_contexts(
    rewards: np.ndarray, n_features: int = 20, lookback: int = 60
) -> np.ndarray:
    """Build finite-dimensional contexts using rewards strictly before time t.

    The returned matrix has shape ``(periods, n_features)``.  The first row is
    zero because it has no prior observations.  The 60-period lookback and the
    exact feature list are reconstruction choices: TODO, the deleted feature
    pipeline was not shown in the chat record.
    """

    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (periods, actions)")
    if n_features <= 0 or lookback <= 0:
        raise ValueError("n_features and lookback must be positive")

    periods, _ = rewards.shape
    contexts = np.zeros((periods, n_features), dtype=float)
    for index in range(periods):
        history = rewards[max(0, index - lookback) : index]
        if history.size == 0:
            continue

        last = history[-1]
        recent3 = history[-3:]
        recent12 = history[-12:]
        finite = history.reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue

        means = np.nan_to_num(np.nanmean(history, axis=0), nan=0.0)
        volatilities = np.nan_to_num(np.nanstd(history, axis=0), nan=0.0)
        summary = np.array(
            [
                np.nanmean(last),
                np.nanstd(last),
                np.nanmin(last),
                np.nanmax(last),
                np.nanmean(means),
                np.nanstd(means),
                np.nanmean(volatilities),
                np.nanstd(volatilities),
                np.nanmean(recent3),
                np.nanstd(recent3),
                np.nanmean(recent12),
                np.nanstd(recent12),
                np.nanmedian(last),
                np.nanpercentile(last, 25),
                np.nanpercentile(last, 75),
                np.nanmean(last > 0),
                np.nanmean(last < 0),
                np.nanmean(finite),
                np.nanstd(finite),
                np.nanmax(np.abs(finite)),
            ],
            dtype=float,
        )
        contexts[index, : min(n_features, len(summary))] = summary[:n_features]

    contexts[~np.isfinite(contexts)] = 0.0
    return contexts
