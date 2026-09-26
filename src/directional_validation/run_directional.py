"""Independent, evidence-labelled validation of FinRisk-OPE directions.

Read PROTOCOL.md for historical evidence, newly chosen settings and limits.
This executable never reads submitted table values and does not modify them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROJECT = ROOT.parent / "recovered_project"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from finrisk_repro.features import build_contexts
from finrisk_repro.io import read_first_return_block
from finrisk_repro.logging import behavior_probabilities
from finrisk_repro.policies import PolicySpec, target_actions
from finrisk_repro.tasks import TASK_SPECS
from run_reproduction import BEHAVIORS

SEEDS = (7, 17, 27)
SELECTORS = ("SCROPE++", "Support only", "LCB only", "Risk only")
ESS_EDGES = np.array([2., 5., 10., 50.])
ESS_LABELS = ("0-2", "2-5", "5-10", "10-50", "50+")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_mean(returns: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(returns, dtype=float)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or w.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Expected a finite return vector and aligned weights")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    total = w.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(-values)[::-1]
    loss, weight = -values[order], w[order]
    mass = 0.05 * total
    before = np.cumsum(weight) - weight
    included = np.clip(mass - before, 0, weight)
    return float(loss @ included / mass)


def load_panel(spec, data_dir: Path) -> tuple[pd.DataFrame, int, dict]:
    raw = read_first_return_block(data_dir / spec.source[0])
    source_columns = list(raw.columns)
    frame = pd.DataFrame(raw.values, index=raw.dates, columns=source_columns)
    if spec.name == "daily_factor_momentum":
        mom = read_first_return_block(data_dir / spec.source[1])
        momentum = pd.Series(mom.values[:, 0], index=mom.dates)
        frame = frame.join(momentum.rename("momentum"), how="inner")
        if len(raw.columns) != 6:
            raise ValueError("FF5 source must have Mkt-RF, SMB, HML, RMW, CMA, RF")
        frame = pd.DataFrame({
            "market_total": frame.a000 + frame.a005,
            "SMB": frame.a001, "HML": frame.a002,
            "RMW": frame.a003, "CMA": frame.a004,
            "momentum": frame.momentum, "cash_RF": frame.a005,
        })
    frame = frame.loc["1963-07-01":"2026-04-30"]
    before_shape = frame.shape
    train = frame.loc[:"2005-12-31"]
    keep = train.notna().mean() >= .95
    removed_columns = frame.columns[~keep].tolist()
    frame = frame.loc[:, keep].ffill(limit=2).dropna()
    if spec.name != "daily_factor_momentum":
        frame["equal_weight_public"] = frame.mean(axis=1)
        baseline = frame.columns.get_loc("equal_weight_public")
    else:
        baseline = frame.columns.get_loc("cash_RF")
    info = {
        "task": spec.name, "source": list(spec.source),
        "before_rows": before_shape[0], "raw_columns": before_shape[1],
        "removed_columns_train_missing": removed_columns,
        "rows_after_cleaning": len(frame), "actions": len(frame.columns),
        "columns": frame.columns.tolist(), "baseline_index": int(baseline),
        "warmup_rows": 60,
    }
    if len(frame) <= 60:
        raise ValueError("Insufficient warmup")
    return frame, int(baseline), info


def fit_predict_train(contexts, actions, rewards, train_mask, n_actions):
    scaler = StandardScaler().fit(contexts[train_mask])
    x = scaler.transform(contexts)
    fallback = float(np.mean(rewards[train_mask]))
    predictions = np.full((len(x), n_actions), fallback)
    for action in range(n_actions):
        use = train_mask & (actions == action)
        if use.sum() >= max(5, x.shape[1] + 1):
            model = Ridge(alpha=1., solver="cholesky").fit(x[use], rewards[use])
            predictions[:, action] = model.predict(x)
    return predictions


def paired_counts(n: int, draws: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if block == 1:
        indices = rng.integers(0, n, size=(draws, n))
    else:
        starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    counts = np.zeros((draws, n), dtype=np.int32)
    for i in range(draws):
        counts[i] = np.bincount(indices[i], minlength=n)
    return counts


def estimate_window(rewards, actions, propensity, predictions, targets, counts):
    n, policies = targets.shape
    observed = rewards[np.arange(n), actions]
    weights = (targets == actions[:, None]) / propensity[:, None]
    target_mu = predictions[np.arange(n)[:, None], targets]
    observed_mu = predictions[np.arange(n), actions]
    contributions = target_mu + weights * (observed - observed_mu)[:, None]
    dr = contributions.mean(axis=0)
    draws = counts @ contributions / n
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    sum_w = weights.sum(axis=0)
    ess = np.divide(sum_w**2, (weights**2).sum(axis=0),
                    out=np.zeros(policies), where=(weights**2).sum(axis=0) > 0)
    out = []
    for p in range(policies):
        w = weights[:, p]
        matches = np.flatnonzero(w > 0)
        risk_ucb = float("inf")
        unsupported = counts.shape[0]
        if len(matches):
            order = matches[np.argsort(-observed[matches])[::-1]]
            bw = counts[:, order] * w[order]
            mass = .05 * bw.sum(axis=1)
            before = np.cumsum(bw, axis=1) - bw
            included = np.clip(mass[:, None] - before, 0, bw)
            risks = np.divide(included @ (-observed[order]), mass,
                              out=np.full(len(mass), np.inf), where=mass > 0)
            risk_ucb = float(np.quantile(risks, .95, method="higher"))
            unsupported = int((mass <= 0).sum())
        truth = rewards[np.arange(n), targets[:, p]]
        out.append({
            "DR": float(dr[p]), "return_lcb": float(lo[p]),
            "return_ucb": float(hi[p]), "risk_hat": tail_mean(observed, w),
            "risk_ucb": risk_ucb, "ESS": float(ess[p]),
            "max_weight": float(w.max()), "support_gap": float(1 - ess[p] / n),
            "truth_return": float(truth.mean()), "truth_cvar": tail_mean(truth),
            "eval_n": n, "unsupported_bootstrap_draws": unsupported,
            "ess_bin": int(np.searchsorted(ESS_EDGES, ess[p], side="right")),
        })
    return out


def calibration_offsets(cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, group in cal.groupby("scheme"):
        for b in range(5):
            sub = group[group.ess_bin == b]
            scores = {
                "return": np.maximum.reduce([
                    sub.return_lcb - sub.truth_return,
                    sub.truth_return - sub.return_ucb, np.zeros(len(sub)),
                ]),
                "risk": np.maximum(0, sub.truth_cvar - sub.risk_ucb),
            }
            for kind, values in scores.items():
                values = np.asarray(values)
                rank = int(np.ceil((len(values) + 1) * .95))
                offset = (float(np.sort(values)[rank - 1])
                          if 0 < rank <= len(values) else float("inf"))
                rows.append({"scheme": scheme, "ess_bin": b, "kind": kind,
                             "calibration_cases": len(values), "offset": offset})
    return pd.DataFrame(rows)


def evidence_decisions(group: pd.DataFrame, tau: float, risk_column: str) -> list[dict]:
    # Outcome columns are intentionally excluded from policy choice.
    evidence = group[["policy", "ESS", "max_weight", "support_gap",
                      "return_lcb", risk_column]].copy()
    evidence = evidence.rename(columns={risk_column: "bound"})
    supported = ((evidence.ESS >= 10) & (evidence.max_weight <= 50)
                 & (evidence.support_gap <= .995) & np.isfinite(evidence.bound))
    safe = evidence.bound <= tau
    decisions = []
    for selector in SELECTORS:
        decision, chosen, score = "ACCEPT", None, -np.inf
        if selector == "SCROPE++":
            eligible = evidence[supported & safe]
            if len(eligible):
                chosen = eligible.sort_values(["return_lcb", "policy"],
                                              ascending=[False, True]).iloc[0]
                support_margin = min(chosen.ESS / 10 - 1,
                                     1 - chosen.max_weight / 50,
                                     1 - chosen.support_gap / .995)
                score = min(support_margin, (tau - chosen.bound) / max(abs(tau), 1e-8))
            elif supported.any():
                decision = "REJECT"
                chosen = evidence[supported].sort_values("return_lcb", ascending=False).iloc[0]
                score = float(chosen.ESS)
            else:
                decision = "ABSTAIN"
        elif selector == "Support only":
            if supported.any():
                chosen = evidence[supported].sort_values(["ESS", "policy"], ascending=[False, True]).iloc[0]
                score = chosen.ESS
            else:
                decision = "ABSTAIN"
        elif selector == "LCB only":
            chosen = evidence.sort_values(["return_lcb", "policy"], ascending=[False, True]).iloc[0]
            score = chosen.return_lcb
        else:
            finite = evidence[np.isfinite(evidence.bound)]
            if len(finite):
                chosen = finite.sort_values(["bound", "policy"]).iloc[0]
                score = -chosen.bound
            else:
                decision = "ABSTAIN"
        row = {"selector": selector, "decision": decision,
               "chosen_policy": "" if chosen is None else chosen.policy,
               "evidence_score": float(score), "tau": tau,
               "false_safe": 0, "unsafe_accept": 0,
               "selected_truth_cvar": np.nan, "selected_truth_return": np.nan}
        if chosen is not None:
            truth = group.loc[chosen.name]
            row["selected_truth_cvar"] = float(truth.truth_cvar)
            row["selected_truth_return"] = float(truth.truth_return)
            row["unsafe_accept"] = int(decision == "ACCEPT" and truth.truth_cvar > tau)
            row["false_safe"] = int(row["unsafe_accept"] and chosen.bound <= tau)
        decisions.append(row)
    return decisions


def annotate_paper_protocol(frame: pd.DataFrame, target: float) -> pd.DataFrame:
    """Annotate natural decisions with the paper's abstention protocol.

    No decisions are removed, promoted, or demoted.  The gap is measured as a
    rate difference and the tolerance is applied only as an audit label.
    """
    result = frame.copy()
    result["mode"] = "paper_natural"
    result["target_abstain"] = target
    result["common_budget"] = np.nan
    keys = ["scheme", "risk_version", "budget", "seed", "selector"]
    for key, indices in result.groupby(keys, sort=False).groups.items():
        subset = result.loc[indices]
        total = len(subset)
        abstain_rate = float(subset.decision.eq("ABSTAIN").mean())
        accept_rate = float(subset.decision.eq("ACCEPT").mean())
        reject_rate = float(subset.decision.eq("REJECT").mean())
        gap = abs(abstain_rate - target)
        result.loc[indices, "abstention_rate"] = abstain_rate
        result.loc[indices, "accept_rate"] = accept_rate
        result.loc[indices, "reject_rate"] = reject_rate
        result.loc[indices, "abstention_gap"] = gap
        result.loc[indices, "matched"] = gap <= 0.10
        result.loc[indices, "false_safe_denominator"] = "ACCEPT decisions"
    return result


def summarize_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scheme", "risk_version", "budget", "seed", "mode", "target_abstain", "selector"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        accept = int(group.decision.eq("ACCEPT").sum())
        reject = int(group.decision.eq("REJECT").sum())
        abstain = int(group.decision.eq("ABSTAIN").sum())
        false_safe = int(group.false_safe.sum())
        rows.append(dict(zip(keys, key), settings=len(group), accepted=accept,
                         rejected=reject, abstained=abstain,
                         accept_pct=100 * accept / len(group),
                         abstain_pct=100 * abstain / len(group),
                         reject_pct=100 * reject / len(group),
                         abstention_gap=float(group.abstention_gap.iloc[0]),
                         abstention_gap_pct=100 * float(group.abstention_gap.iloc[0]),
                         matched=bool(group.matched.iloc[0]),
                         false_safe_count=false_safe,
                         false_safe_accepted_pct=100 * false_safe / accept if accept else np.nan,
                         false_safe_nonabstained_pct=100 * false_safe / (accept + reject) if accept + reject else np.nan,
                         false_safe_all_pct=100 * false_safe / len(group),
                         false_safe_denominator="ACCEPT decisions",
                         unsafe_accept_count=int(group.unsafe_accept.sum())))
    return pd.DataFrame(rows)


def paired_uncertainty(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scheme", "risk_version", "budget", "mode", "target_abstain"]
    for key, group in frame.groupby(keys, dropna=False):
        if key[3] != "paper_natural":
            continue
        tasks = sorted(group.task.unique())
        rng = np.random.default_rng(1729)
        indices = rng.integers(0, len(tasks), (2000, len(tasks)))
        counts = {}
        for selector in SELECTORS:
            x = group[group.selector == selector].copy()
            x["accept_indicator"] = x.decision.eq("ACCEPT").astype(int)
            counts[selector] = x.groupby("task")[["false_safe", "accept_indicator"]].sum().reindex(tasks).values
        for baseline in SELECTORS[1:]:
            draws, points = [], []
            for selector in ("SCROPE++", baseline):
                x = counts[selector]
                totals = x[indices].sum(axis=1)
                draws.append(np.divide(totals[:, 0], totals[:, 1],
                                       out=np.full(len(indices), np.nan), where=totals[:, 1] > 0))
                points.append(x[:, 0].sum() / x[:, 1].sum() if x[:, 1].sum() else np.nan)
            difference = 100 * (draws[0] - draws[1])
            valid = difference[np.isfinite(difference)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
            rows.append(dict(zip(keys, key), baseline=baseline,
                             difference_pp=100 * (points[0] - points[1]),
                             ci_low_pp=float(low), ci_high_pp=float(high),
                             tasks=len(tasks), ci_draws=len(valid)))
    return pd.DataFrame(rows)


def aggregate(cases: pd.DataFrame, output: Path):
    cal, test = cases[cases.split == "calibration"], cases[cases.split == "test"].copy()
    offsets = calibration_offsets(cal)
    offsets.to_csv(output / "calibration_offsets.csv", index=False)
    for kind in ("return", "risk"):
        lookup = offsets[offsets.kind == kind].set_index(["scheme", "ess_bin"]).offset
        test[kind + "_offset"] = [lookup.loc[(s, b)] for s, b in zip(test.scheme, test.ess_bin)]
    test["calibrated_risk_ucb"] = test.risk_ucb + test.risk_offset
    test["nominal_covered"] = (test.return_lcb <= test.truth_return) & (test.truth_return <= test.return_ucb)
    test["calibrated_covered"] = ((test.return_lcb - test.return_offset <= test.truth_return)
                                 & (test.truth_return <= test.return_ucb + test.return_offset))
    test["nominal_width"] = test.return_ucb - test.return_lcb
    test["calibrated_width"] = test.nominal_width + 2 * test.return_offset
    test["risk_under"] = (test.risk_hat < test.truth_cvar - 1e-12).where(test.risk_hat.notna())
    test["risk_abs_error"] = abs(test.risk_hat - test.truth_cvar)
    test.to_csv(output / "test_cases.csv", index=False)
    coverage = test.groupby(["scheme", "ess_bin"]).agg(
        cases=("task", "size"), nominal_coverage=("nominal_covered", "mean"),
        calibrated_coverage=("calibrated_covered", "mean"),
        mean_nominal_width=("nominal_width", "mean"),
        mean_calibrated_width=("calibrated_width", "mean"))
    coverage.to_csv(output / "coverage.csv")
    risk = test.groupby(["scheme", "family"]).agg(
        cases=("task", "size"), evaluable_risk=("risk_hat", "count"),
        underestimation=("risk_under", "mean"), mae=("risk_abs_error", "mean"), median_ess=("ESS", "median"))
    risk.to_csv(output / "risk_diagnostics.csv")
    natural = []
    for key, group in test.groupby(["task", "family", "behavior", "seed", "scheme"]):
        for version, column in (("bootstrap", "risk_ucb"), ("calibrated_extension", "calibrated_risk_ucb")):
            for budget, tau in (("baseline_linked", float(group.baseline_tau.iloc[0])), ("fixed_0.10_sensitivity", .10)):
                for row in evidence_decisions(group, tau, column):
                    row.update(dict(zip(["task", "family", "behavior", "seed", "scheme"], key)))
                    row.update(risk_version=version, budget=budget, mode="natural", target_abstain=np.nan, common_budget=np.nan)
                    natural.append(row)
    decisions = pd.DataFrame(natural)
    all_decisions = annotate_paper_protocol(decisions, target=.25)
    all_decisions.to_csv(output / "decisions.csv", index=False)
    summary = summarize_decisions(all_decisions)
    summary.to_csv(output / "selector_summary.csv", index=False)
    uncertainty = paired_uncertainty(all_decisions)
    uncertainty.to_csv(output / "paired_task_uncertainty.csv", index=False)
    primary = summary[(summary.scheme == "block") & (summary.risk_version == "bootstrap")
                      & (summary.budget == "baseline_linked") & (summary["mode"] == "paper_natural")]
    pooled = primary.groupby("selector")[["settings", "accepted", "rejected", "abstained", "false_safe_count", "unsafe_accept_count"]].sum()
    pooled["false_safe_accepted_pct"] = 100 * pooled.false_safe_count / pooled.accepted.replace(0, np.nan)
    pooled["accept_pct"] = 100 * pooled.accepted / pooled.settings
    pooled["abstain_pct"] = 100 * pooled.abstained / pooled.settings
    pooled["abstention_gap_pct"] = abs(pooled.abstain_pct - 25.0)
    pooled["matched"] = pooled.abstention_gap_pct <= 10.0
    primary_ci = uncertainty[(uncertainty.scheme == "block") & (uncertainty.risk_version == "bootstrap")
                             & (uncertainty.budget == "baseline_linked") & (uncertainty.target_abstain == .25)]
    with (output / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write("# Directional revalidation with natural Table IV protocol\n\nSee ../PROTOCOL.md for the fixed current protocol.\n\n")
        f.write("## Table IV protocol comparison\n\nBlock bootstrap, baseline-linked risk budget, natural decisions.\n")
        f.write("The target is 25% ABSTAIN. The reported gap is the absolute abstention-rate difference; matched means gap <= 0.10. No accepted-count matching or artificial abstention is applied.\n\n")
        f.write(pooled.to_markdown() + "\n\nNegative paired differences favor SCROPE++.\n\n" + primary_ci.to_markdown(index=False))
        f.write("\n\n## Return coverage\n\n" + coverage.to_markdown())
        f.write("\n\n## Tail risk\n\n" + risk.to_markdown())
        f.write("\n\n## Boundaries\n\nAll selector settings and seed outcomes are in selector_summary.csv, all decisions in decisions.csv. ")
        f.write("Task-cluster intervals capture one level of dependence; they are not full market-shock uncertainty. ")
        f.write("Risk-calibration extension and fixed-tau sensitivity must not replace the primary row by outcome preference. ")
        f.write("A zero false-safe point with no acceptance is undefined, not success. ")
        f.write("Case counts, source release and policy constructors are documented in the current protocol. ")
        f.write("These are controlled semi-synthetic benchmark results without transaction costs or market impact.\n")
    print("PRIMARY (all seeds)", flush=True)
    print(pooled.to_string(), flush=True)
    print(primary_ci.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "run_v1"))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    if args.aggregate_only:
        aggregate(pd.read_csv(output / "all_cases.csv"), output)
        return
    output.mkdir(parents=True, exist_ok=False)
    data_dir = REPOSITORY_ROOT / "data"
    sources = {p.name: sha256(p) for p in sorted(data_dir.glob("*.zip"))}
    package_root = REPOSITORY_ROOT
    code_paths = [Path(__file__), package_root / "artifacts" / "docs" / "PROTOCOL.md",
                  package_root / "src" / "export_paper_evidence.py"] + list((PROJECT / "src" / "finrisk_repro").glob("*.py"))
    manifest = {"status": "RUNNING", "seeds": SEEDS, "bootstrap": 200,
                "schemes": {"iid": 1, "block": {"monthly": 3, "daily": 20}},
                "source_sha256": sources,
                "code_sha256": {p.relative_to(package_root).as_posix(): sha256(p) for p in code_paths},
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "threads": {name: os.environ[name] for name in
                                ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
                },
                "split": {"train_end": "2005-12-31", "calibration_end": "2014-12-31", "test_start": "2015-01-01"}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    rows, inventory = [], []
    for ti, spec in enumerate(TASK_SPECS):
        frame, baseline, info = load_panel(spec, data_dir)
        x = build_contexts(frame.values, n_features=20)
        values, x, dates = frame.values[60:], x[60:], frame.index[60:]
        train = np.asarray(dates <= pd.Timestamp("2005-12-31"))
        cal = np.asarray((dates >= pd.Timestamp("2006-01-01")) & (dates <= pd.Timestamp("2014-12-31")))
        test = np.asarray(dates >= pd.Timestamp("2015-01-01"))
        if min(train.sum(), cal.sum(), test.sum()) < 10:
            raise ValueError("Insufficient chronological split: " + spec.name)
        tau = tail_mean(values[cal, baseline])
        info.update(periods=len(values), train_n=int(train.sum()), calibration_n=int(cal.sum()), test_n=int(test.sum()),
                    first_date=str(dates.min().date()), last_date=str(dates.max().date()), baseline_tau=tau,
                    cells=int(values.size))
        inventory.append(info)
        targets = [np.full(len(values), a, dtype=int) for a in range(values.shape[1])]
        names = [f"constant_{name}" for name in frame.columns]
        for kind in ("momentum", "low_vol", "risk_adjusted"):
            targets.append(target_actions(frame.values, PolicySpec(kind + "_12", kind, 12))[60:])
            names.append(kind + "_12")
        task_rows = []
        for bi, behavior in enumerate(BEHAVIORS):
            probabilities = behavior_probabilities(frame.values, behavior)[60:]
            for seed in SEEDS:
                key_seed = seed + ti * 101 + bi
                rng = np.random.default_rng(key_seed)
                actions = (rng.random(len(values))[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
                actions = np.minimum(actions, values.shape[1] - 1)
                propensity = probabilities[np.arange(len(values)), actions]
                observed = values[np.arange(len(values)), actions]
                predictions = fit_predict_train(x, actions, observed, train, values.shape[1])
                policy_targets = np.column_stack(targets + [predictions.argmax(axis=1)])
                policy_names = names + ["ridge_return_train_only"]
                for split, mask in (("calibration", cal), ("test", test)):
                    for scheme, block in (("iid", 1), ("block", 20 if spec.frequency == "daily" else 3)):
                        counts = paired_counts(int(mask.sum()), 200, block,
                                               key_seed + (10000 if split == "test" else 0))
                        estimates = estimate_window(values[mask], actions[mask], propensity[mask],
                                                    predictions[mask], policy_targets[mask], counts)
                        for name, estimate in zip(policy_names, estimates):
                            task_rows.append(dict(estimate, task=spec.name, family=spec.family,
                                                  behavior=behavior, seed=seed, split=split,
                                                  scheme=scheme, policy=name, baseline_tau=tau))
            print(f"[{ti+1:02d}/15] {spec.name} / {behavior}", flush=True)
        pd.DataFrame(task_rows).to_csv(output / (spec.name + "_cases.csv"), index=False)
        rows.extend(task_rows)
        print(f"task complete; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    pd.DataFrame(inventory).to_json(output / "task_inventory.json", orient="records", indent=2)
    cases = pd.DataFrame(rows)
    all_cases_path = output / "all_cases.csv"
    cases.to_csv(all_cases_path, index=False)
    # Re-read the persisted case file before deriving decisions. This makes
    # the on-disk CSV the canonical source and avoids in-memory/CSV float
    # rounding changing a near-tied selected policy.
    aggregate(pd.read_csv(all_cases_path), output)
    manifest.update(status="COMPLETE", runtime_seconds=time.perf_counter() - started,
                    case_rows=len(cases), tasks=len(inventory), cells=sum(x["cells"] for x in inventory))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
