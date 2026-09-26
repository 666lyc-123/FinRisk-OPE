# FinRisk-OPE experiment protocol

## Data and task construction

- Source: frozen public ZIP files from the Kenneth R. French Data Library.
- Freeze: observations through 2026-04-30; no later observation is used.
- Tasks: 15 monthly and daily portfolio-bandit tasks.
- Columns: retain columns with at least 95% observed training returns, forward-fill at most two dates, drop remaining incomplete rows, and append an equal-weight public action except for the factor/momentum task, whose baseline is cash RF.
- Warmup: 60 dates. Every context uses information strictly before its reward date.
- Context: 20 lagged reward-panel summaries, including recent means, volatilities, extrema, quantiles, sign rates, and magnitude summaries.

## Chronological split

- Training: through 2005-12-31.
- Calibration: 2006-01-01 through 2014-12-31.
- Test: from 2015-01-01.

## Logged feedback and policies

- Behaviors: uniform high overlap, softmax high entropy, epsilon-greedy momentum, risk-averse low volatility, return-seeking recent winner, market-state mixed, low-overlap concentration, and adversarial support mismatch.
- Seeds: 7, 17, and 27, with deterministic task/behavior offsets.
- Targets: one constant policy per action, momentum-12, low-volatility-12, mean/volatility-12, and a train-only ridge-return policy.
- Reward model: action-conditional ridge regression with `alpha=1` and the explicit Cholesky solver; features are standardized using training dates only. Sparse actions use the training global mean.

## OPE, uncertainty, and risk

- Primary return estimator: doubly robust (DR).
- Bootstrap: 200 paired date resamples shared across policies and metrics.
- Schemes: IID; circular block length 3 for monthly tasks and 20 for daily tasks.
- Return interval: 2.5th and 97.5th bootstrap percentiles.
- Risk: exact weighted mean of the worst 5% loss mass, including fractional atoms.
- Risk bound: one-sided 95th bootstrap percentile; unsupported draws remain infinite.
- Calibration: additive support-bin nonconformity offsets learned from calibration dates only.

## Certification and matched acceptance

- Risk budget: CVaR95 of each task's calibration-window baseline action.
- Support gates: ESS >= 10, maximum importance weight <= 50, and support gap <= 0.995.
- SCROPE++: maximize return LCB among supported candidates passing the risk bound; REJECT when support is adequate but no candidate passes; ABSTAIN when support is inadequate.
- Baselines: support-only maximizes ESS, LCB-only maximizes the return LCB, and risk-only minimizes the finite risk bound.
- Primary comparison: for each seed, use the largest common ACCEPT budget available across selectors, capped at 75% of the 120 task-behavior groups. This yields 249 ACCEPT decisions per selector across 360 groups.
- Report both strict false-safe and broader unsafe-acceptance rates among ACCEPT decisions.

## Reproducibility boundary

Source-file hashes, code hashes, dependency versions, seeds, split dates, bootstrap settings, and thread counts are recorded. Counts and decision totals are deterministic. Ridge fitting and floating-point reductions may vary at the last decimal across operating systems or BLAS builds, so verification uses explicit tolerances while requiring the published rounded results to remain stable.
