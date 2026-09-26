"""Readers for the public Kenneth French ZIP files.

This module is reconstructed from the manifest and the later clean-room
implementation.  It is not a byte-for-byte recovery of the deleted source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


_DATE_RE = re.compile(r"^(?:19|20)\d{4}(?:\d{2})?$")


@dataclass
class Panel:
    """A date-indexed reward panel with one column per action."""

    name: str
    dates: pd.Index
    values: np.ndarray
    columns: list[str]
    frequency: str
    source: str

    @property
    def n_periods(self) -> int:
        return int(len(self.dates))

    @property
    def n_actions(self) -> int:
        return int(self.values.shape[1])


def _date_token(line: str) -> str | None:
    token = line.strip().split(",", 1)[0].strip()
    if _DATE_RE.fullmatch(token) and len(token) in (6, 8):
        return token
    return None


def _parse_date(token: str) -> pd.Timestamp:
    if len(token) == 6:
        return pd.Timestamp(year=int(token[:4]), month=int(token[4:]), day=1)
    return pd.Timestamp(year=int(token[:4]), month=int(token[4:6]), day=int(token[6:8]))


def read_first_return_block(path: str | Path, freeze: str = "2026-04-30") -> Panel:
    """Read the first contiguous return block in a French ZIP file.

    The audit trail confirms the public ZIP source, a date freeze of
    2026-04-30, and decimal return scaling in the clean-room implementation.
    TODO: the deleted project did not preserve the exact table-block selector,
    so "first contiguous block" remains a reconstruction assumption.
    """

    path = Path(path)
    freeze_date = pd.Timestamp(freeze)
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if not members:
            raise ValueError(f"No file in {path}")
        text = archive.read(members[0]).decode("latin1", errors="replace")

    lines = text.splitlines()
    rows: list[tuple[str, list[float]]] = []
    started = False
    expected_width: int | None = None
    for line in lines:
        token = _date_token(line)
        if token is None:
            if started:
                break
            continue
        raw = line.split(",")
        try:
            values = [float(value.strip()) for value in raw[1:]]
        except ValueError:
            if started:
                break
            continue
        if expected_width is None:
            expected_width = len(values)
            started = True
        if len(values) != expected_width:
            break
        timestamp = _parse_date(token)
        if timestamp <= freeze_date:
            rows.append((token, values))

    if not rows:
        raise ValueError(f"No return rows found in {path}")

    # The public French files report percentage returns; the reconstruction
    # converts them to decimal returns before OPE calculations.
    values = np.asarray([row[1] for row in rows], dtype=float) / 100.0
    # TODO: the original missing-value sentinel policy is not recoverable.
    # This sentinel handling is retained from the clean-room implementation.
    values[values <= -0.999] = np.nan
    dates = pd.Index([_parse_date(row[0]) for row in rows], name="date")
    frequency = "daily" if len(rows[0][0]) == 8 else "monthly"
    columns = [f"a{i:03d}" for i in range(values.shape[1])]
    return Panel(path.stem, dates, values, columns, frequency, str(path))


def read_factor_momentum(data_dir: str | Path, freeze: str = "2026-04-30") -> Panel:
    """Combine daily Fama--French and momentum factor returns.

    Seven actions are reported for the daily factor/momentum reconstruction.
    TODO: the original source-column mapping is not preserved; this uses all
    six columns from the daily FF source plus the first momentum column.
    """

    data_dir = Path(data_dir)
    ff = read_first_return_block(
        data_dir / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", freeze
    )
    momentum = read_first_return_block(
        data_dir / "F-F_Momentum_Factor_daily_CSV.zip", freeze
    )
    common = ff.dates.intersection(momentum.dates)
    ff_index = ff.dates.get_indexer(common)
    momentum_index = momentum.dates.get_indexer(common)
    values = np.column_stack(
        [ff.values[ff_index], momentum.values[momentum_index, 0]]
    )
    columns = [f"a{i:03d}" for i in range(values.shape[1])]
    return Panel("factor_momentum_daily", common, values, columns, "daily", "FF5+MOM")
