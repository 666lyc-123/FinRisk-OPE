"""Task cards and reward-panel construction for the 15-task benchmark.

The task names, periods, actions, and cell counts come from the final paper's
Table I.  The source-column map was not preserved, so the selection and
missing-value rules below remain explicitly marked reconstruction assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import Panel, read_factor_momentum, read_first_return_block


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    source: tuple[str, ...]
    frequency: str
    target_actions: int
    expected_periods: int
    expected_cells: int


# These values are the hard Table I acceptance checks recorded in the manifest.
TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("monthly_industry_17", "monthly industry", ("17_Industry_Portfolios_CSV.zip",), "monthly", 18, 753, 753 * 18),
    TaskSpec("monthly_industry_30", "monthly industry", ("30_Industry_Portfolios_CSV.zip",), "monthly", 31, 753, 753 * 31),
    TaskSpec("monthly_industry_48", "monthly industry", ("48_Industry_Portfolios_CSV.zip",), "monthly", 44, 753, 753 * 44),
    TaskSpec("monthly_size_bm_25", "monthly size--B/M", ("25_Portfolios_5x5_CSV.zip",), "monthly", 26, 753, 753 * 26),
    TaskSpec("monthly_size_bm_100", "monthly size--B/M", ("100_Portfolios_10x10_CSV.zip",), "monthly", 93, 723, 723 * 93),
    TaskSpec("monthly_beme_inv", "monthly size/profit/invest", ("25_Portfolios_BEME_INV_5x5_CSV.zip",), "monthly", 26, 728, 728 * 26),
    TaskSpec("monthly_beme_op", "monthly size/profit/invest", ("25_Portfolios_BEME_OP_5x5_CSV.zip",), "monthly", 26, 728, 728 * 26),
    TaskSpec("monthly_me_inv", "monthly two-characteristic", ("25_Portfolios_ME_INV_5x5_CSV.zip",), "monthly", 26, 728, 728 * 26),
    TaskSpec("monthly_me_op", "monthly two-characteristic", ("25_Portfolios_ME_OP_5x5_CSV.zip",), "monthly", 26, 728, 728 * 26),
    TaskSpec("monthly_op_inv", "monthly two-characteristic", ("25_Portfolios_OP_INV_5x5_CSV.zip",), "monthly", 26, 728, 728 * 26),
    TaskSpec("daily_bm_25", "daily size characteristic", ("25_Portfolios_5x5_Daily_CSV.zip",), "daily", 26, 15787, 15787 * 26),
    TaskSpec("daily_me_inv", "daily size characteristic", ("25_Portfolios_ME_INV_5x5_daily_CSV.zip",), "daily", 26, 15787, 15787 * 26),
    TaskSpec("daily_me_op", "daily size characteristic", ("25_Portfolios_ME_OP_5x5_daily_CSV.zip",), "daily", 26, 15812, 15812 * 26),
    TaskSpec("daily_industry_30", "daily industry/factor", ("30_Industry_Portfolios_Daily_CSV.zip",), "daily", 31, 15812, 15812 * 31),
    TaskSpec("daily_factor_momentum", "daily industry/factor", ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"), "daily", 7, 15787, 15787 * 7),
)


def _select_actions(panel: Panel, target_actions: int) -> np.ndarray:
    """Map source columns to the reported number of actions.

    TODO: the exact source-column map is absent from the recovered project.
    The fallback preserves all raw columns and appends a zero-return cash
    action only in the raw+1 case; otherwise it selects evenly spaced columns.
    """

    raw_actions = panel.n_actions
    if target_actions == raw_actions + 1:
        return np.column_stack([panel.values, np.zeros(panel.n_periods)])
    if target_actions <= raw_actions:
        indices = np.linspace(0, raw_actions - 1, target_actions, dtype=int)
        return panel.values[:, indices]
    raise ValueError(f"Cannot map {raw_actions} raw actions to {target_actions}")


def _trim_periods(
    values: np.ndarray, dates: pd.Index, n_periods: int
) -> tuple[np.ndarray, pd.Index]:
    if len(dates) < n_periods:
        raise ValueError(f"Need {n_periods} periods but source has {len(dates)}")
    return values[-n_periods:], dates[-n_periods:]


def _impute_missing(values: np.ndarray) -> np.ndarray:
    """Fill sparse missing returns with the same-date cross-sectional median.

    TODO: the manuscript does not expose the original missing-value rule.  The
    row-wise median is retained from the clean-room implementation because it
    keeps the recorded period/action dimensions without forward interpolation.
    """

    result = np.asarray(values, dtype=float).copy()
    for row in result:
        finite = row[np.isfinite(row)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        row[~np.isfinite(row)] = fill
    return result


def load_task(spec: TaskSpec, data_dir: str | Path, freeze: str = "2026-04-30") -> Panel:
    data_dir = Path(data_dir)
    if spec.name == "daily_factor_momentum":
        panel = read_factor_momentum(data_dir, freeze)
    else:
        panel = read_first_return_block(data_dir / spec.source[0], freeze)
    values = _select_actions(panel, spec.target_actions)
    values, dates = _trim_periods(values, panel.dates, spec.expected_periods)
    values = _impute_missing(values)
    columns = [f"a{i:03d}" for i in range(values.shape[1])]
    return Panel(spec.name, dates, values, columns, spec.frequency, ";".join(spec.source))


def load_all_tasks(
    data_dir: str | Path, freeze: str = "2026-04-30"
) -> list[tuple[TaskSpec, Panel]]:
    return [(spec, load_task(spec, data_dir, freeze)) for spec in TASK_SPECS]


def benchmark_summary(tasks: Iterable[tuple[TaskSpec, Panel]]) -> pd.DataFrame:
    rows = []
    for spec, panel in tasks:
        rows.append(
            {
                **asdict(spec),
                "actual_periods": panel.n_periods,
                "actual_actions": panel.n_actions,
                "actual_cells": int(panel.values.size),
                "period_match": panel.n_periods == spec.expected_periods,
                "action_match": panel.n_actions == spec.target_actions,
                "cell_match": panel.values.size == spec.expected_cells,
            }
        )
    return pd.DataFrame(rows)


def grouped_summary(tasks: Iterable[tuple[TaskSpec, Panel]]) -> pd.DataFrame:
    frame = benchmark_summary(tasks)
    return frame.groupby("family", as_index=False).agg(
        tasks=("name", "count"),
        periods_min=("actual_periods", "min"),
        periods_max=("actual_periods", "max"),
        actions_min=("actual_actions", "min"),
        actions_max=("actual_actions", "max"),
        cells=("actual_cells", "sum"),
    )
