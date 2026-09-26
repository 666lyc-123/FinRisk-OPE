"""Run the clean-room OPE and selective-certification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk_repro.certification import classify_selector
from finrisk_repro.features import build_contexts
from finrisk_repro.logging import sample_log
from finrisk_repro.ope import estimate_all, fit_reward_model
from finrisk_repro.policies import POLICY_SPECS, target_actions
from finrisk_repro.tasks import load_all_tasks


BEHAVIORS = (
    "uniform_high_overlap",
    "softmax_high_entropy",
    "epsilon_greedy_momentum",
    "risk_averse_low_volatility",
    "return_seeking_recent_winner",
    "market_state_mixed",
    "low_overlap_concentration",
    "adversarial_support_mismatch",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT.parent / "data_raw"))
    parser.add_argument("--freeze", default="2026-04-30")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--clip", type=float, default=20.0)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Reserved for the unrecovered calibrated-CI pass.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    tasks = load_all_tasks(args.data_dir, args.freeze)

    for task_index, (spec, panel) in enumerate(tasks):
        n_features = 21 if spec.frequency == "daily" else 20
        contexts = build_contexts(panel.values, n_features=n_features)
        target_list = [target_actions(panel.values, policy) for policy in POLICY_SPECS]
        for behavior_index, behavior in enumerate(BEHAVIORS):
            # The seed offset is a clean-room reproducibility convention; TODO,
            # the original per-task/per-policy seed schedule is not preserved.
            log_seed = args.seed + task_index * 101 + behavior_index
            log = sample_log(panel.values, behavior, seed=log_seed)
            dm_preds = fit_reward_model(
                contexts, log["actions"], log["rewards"], panel.n_actions
            )
            estimates: list[dict[str, float]] = []
            for policy_index, (policy, target) in enumerate(
                zip(POLICY_SPECS, target_list)
            ):
                estimate = estimate_all(
                    contexts,
                    log,
                    target,
                    panel.n_actions,
                    clip=args.clip,
                    dm_preds=dm_preds,
                )
                truth = panel.values[np.arange(len(target)), target]
                truth_cvar = float(-np.nanquantile(truth, 0.05))
                metric_rows.append(
                    {
                        "task": spec.name,
                        "family": spec.family,
                        "behavior": behavior,
                        "policy": policy.name,
                        "policy_index": policy_index,
                        "periods": panel.n_periods,
                        "actions": panel.n_actions,
                        "truth_return": float(np.nanmean(truth)),
                        "truth_cvar95": truth_cvar,
                        "return_error_DR": float(
                            estimate["DR"] - np.nanmean(truth)
                        ),
                        "cvar95_ope": estimate["CVaR95"],
                        "cvar95_error": float(estimate["CVaR95"] - truth_cvar),
                        **estimate,
                    }
                )
                estimates.append(estimate)

            selected = classify_selector(
                estimates, panel.values, target_list, tau=args.tau
            )
            selector_rows.append(
                {
                    "task": spec.name,
                    "family": spec.family,
                    "behavior": behavior,
                    **selected,
                }
            )
            print(f"[{task_index + 1:02d}/15] {spec.name} / {behavior}")

    metrics = pd.DataFrame(metric_rows)
    selectors = pd.DataFrame(selector_rows)
    metrics.to_csv(out_dir / "reproduction_ope_metrics.csv", index=False)
    selectors.to_csv(out_dir / "reproduction_selector_metrics.csv", index=False)
    summary = {
        "tasks": int(metrics.task.nunique()),
        "behaviors": int(metrics.behavior.nunique()),
        "policies": int(metrics.policy.nunique()),
        "ope_rows": int(len(metrics)),
        "selector_rows": int(len(selectors)),
        "false_safe_rate": float(selectors.false_safe.mean()),
        "abstain_rate": float((selectors.decision == "ABSTAIN").mean()),
        "note": (
            "Clean-room reconstruction; unspecified paper hyperparameters and "
            "the --bootstrap pass remain TODO."
        ),
    }
    (out_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
