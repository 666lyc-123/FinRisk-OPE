"""Verify data, results, generated tables, and paper/result consistency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(name: str, actual, expected, atol: float) -> None:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape or not np.allclose(a, e, rtol=0, atol=atol, equal_nan=True):
        diff = float(np.nanmax(np.abs(a - e))) if a.shape == e.shape else float("inf")
        raise AssertionError(f"{name}: mismatch (max absolute difference {diff}, tolerance {atol})")


def same_columns_and_rows(name: str, actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise AssertionError(f"{name}: structure mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs" / "reference_run")
    args = parser.parse_args()
    run = args.run.resolve()
    assets = args.assets.resolve()
    reference = args.reference.resolve()

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    ref_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    for key in ("status", "case_rows", "tasks", "cells", "seeds", "bootstrap", "schemes", "split"):
        if manifest[key] != ref_manifest[key]:
            raise AssertionError(f"manifest.{key}: {manifest[key]} != {ref_manifest[key]}")
    if manifest["source_sha256"] != ref_manifest["source_sha256"]:
        raise AssertionError("source ZIP hashes differ from the reference freeze")
    for name, expected_hash in ref_manifest["source_sha256"].items():
        if sha256(ROOT / "data" / name) != expected_hash:
            raise AssertionError(f"local source ZIP differs from manifest: {name}")
    release_manifest_path = ROOT / "artifacts" / "release_manifest.json"
    if release_manifest_path.exists():
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        for relative, expected_hash in release_manifest["sha256"].items():
            if sha256(ROOT / relative) != expected_hash:
                raise AssertionError(f"release file differs from release manifest: {relative}")

    cases = pd.read_csv(run / "all_cases.csv")
    ref_cases = pd.read_csv(reference / "all_cases.csv")
    same_columns_and_rows("all_cases.csv", cases, ref_cases)
    if len(cases) != 50784:
        raise AssertionError("all_cases.csv must contain 50,784 rows")
    for col in ("task", "family", "behavior", "seed", "split", "scheme", "policy"):
        if not cases[col].equals(ref_cases[col]):
            raise AssertionError(f"all_cases.csv key column differs: {col}")

    current_coverage = pd.read_csv(assets / "evidence" / "coverage.csv").sort_values(["scheme", "ess_bin"])
    ref_coverage = pd.read_csv(ROOT / "artifacts" / "evidence" / "coverage.csv").sort_values(["scheme", "ess_bin"])
    same_columns_and_rows("coverage.csv", current_coverage, ref_coverage)
    if not current_coverage[["scheme", "ess_bin", "n"]].reset_index(drop=True).equals(
            ref_coverage[["scheme", "ess_bin", "n"]].reset_index(drop=True)):
        raise AssertionError("coverage bin counts differ")
    close("coverage rates", current_coverage[["nominal_coverage", "calibrated_coverage"]],
          ref_coverage[["nominal_coverage", "calibrated_coverage"]], 0.001)
    close("coverage widths", current_coverage[["width", "cal_width"]],
          ref_coverage[["width", "cal_width"]], 0.0001)

    current_risk = pd.read_csv(assets / "evidence" / "risk_diagnostics.csv")
    ref_risk = pd.read_csv(ROOT / "artifacts" / "evidence" / "risk_diagnostics.csv")
    same_columns_and_rows("risk_diagnostics.csv", current_risk, ref_risk)
    if not current_risk[["family", "n", "valid"]].equals(ref_risk[["family", "n", "valid"]]):
        raise AssertionError("risk diagnostic counts differ")
    close("risk underestimation", current_risk["under"], ref_risk["under"], 0.001)
    close("risk MAE", current_risk["mae"], ref_risk["mae"], 0.00005)
    close("risk median ESS", current_risk["median_ess"], ref_risk["median_ess"], 0.02)

    current_selectors = pd.read_csv(assets / "evidence" / "matched_summary.csv")
    ref_selectors = pd.read_csv(ROOT / "artifacts" / "evidence" / "matched_summary.csv")
    same_columns_and_rows("matched_summary.csv", current_selectors, ref_selectors)
    count_cols = ["selector", "settings", "accepted", "rejected", "abstained",
                  "false_safe_count", "unsafe_accept_count"]
    if not current_selectors[count_cols].equals(ref_selectors[count_cols]):
        raise AssertionError("matched selector counts differ")
    close("matched selector rates", current_selectors[["false_safe_pct", "unsafe_accept_pct"]],
          ref_selectors[["false_safe_pct", "unsafe_accept_pct"]], 0.02)

    for table in sorted((ROOT / "artifacts" / "generated_tables").glob("*.tex")):
        reproduced = assets / "generated_tables" / table.name
        if table.read_text(encoding="utf-8") != reproduced.read_text(encoding="utf-8"):
            raise AssertionError(f"generated paper table differs: {table.name}")

    paper = (ROOT / "paper_source" / "main.tex").read_text(encoding="utf-8")
    required = ["50,784", "2,068,330", "249 ACCEPT", "24.10\\%", "59.84\\%",
                "59.44\\%", "32.13\\%", "transaction costs and market impact"]
    missing = [item for item in required if item not in paper]
    if missing:
        raise AssertionError("paper is missing required result/limitation text: " + repr(missing))
    print("VERIFIED: data hashes, case structure, metrics, tables, and paper claims are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
