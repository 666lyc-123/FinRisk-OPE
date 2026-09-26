# Directional revalidation with natural Table IV protocol

See ../PROTOCOL.md for the fixed current protocol.

## Table IV protocol comparison

Block bootstrap, baseline-linked risk budget, natural decisions.
The target is 25% ABSTAIN. The reported gap is the absolute abstention-rate difference; matched means gap <= 0.10. No accepted-count matching or artificial abstention is applied.

| selector     |   settings |   accepted |   rejected |   abstained |   false_safe_count |   unsafe_accept_count |   false_safe_accepted_pct |   accept_pct |   abstain_pct |   abstention_gap_pct | matched   |
|:-------------|-----------:|-----------:|-----------:|------------:|-------------------:|----------------------:|--------------------------:|-------------:|--------------:|---------------------:|:----------|
| LCB only     |        360 |        360 |          0 |           0 |                100 |                   184 |                  27.7778  |     100      |        0      |              25      | False     |
| Risk only    |        360 |        360 |          0 |           0 |                104 |                   104 |                  28.8889  |     100      |        0      |              25      | False     |
| SCROPE++     |        360 |        249 |         62 |          49 |                 60 |                    60 |                  24.0964  |      69.1667 |       13.6111 |              11.3889 | False     |
| Support only |        360 |        311 |          0 |          49 |                 11 |                   160 |                   3.53698 |      86.3889 |       13.6111 |              11.3889 | False     |

Negative paired differences favor SCROPE++.

| scheme   | risk_version   | budget          | mode          |   target_abstain | baseline     |   difference_pp |   ci_low_pp |   ci_high_pp |   tasks |   ci_draws |
|:---------|:---------------|:----------------|:--------------|-----------------:|:-------------|----------------:|------------:|-------------:|--------:|-----------:|
| block    | bootstrap      | baseline_linked | paper_natural |             0.25 | Support only |        20.5594  |     10.4765 |     33.4789  |      15 |       2000 |
| block    | bootstrap      | baseline_linked | paper_natural |             0.25 | LCB only     |        -3.68139 |    -11.974  |      3.91151 |      15 |       2000 |
| block    | bootstrap      | baseline_linked | paper_natural |             0.25 | Risk only    |        -4.7925  |    -12.8161 |      2.5096  |      15 |       2000 |

## Return coverage

|              |   cases |   nominal_coverage |   calibrated_coverage |   mean_nominal_width |   mean_calibrated_width |
|:-------------|--------:|-------------------:|----------------------:|---------------------:|------------------------:|
| ('block', 0) |    4798 |           0.274489 |              0.946019 |           0.168789   |              0.1942     |
| ('block', 1) |    2856 |           0.683473 |              0.960434 |           0.108127   |              0.132513   |
| ('block', 2) |    1930 |           0.697409 |              0.931606 |           0.0805158  |              0.1233     |
| ('block', 3) |    1049 |           0.589133 |              0.914204 |           0.0299365  |              0.0932333  |
| ('block', 4) |    2063 |           0.839069 |              0.962191 |           0.00632729 |              0.00960056 |
| ('iid', 0)   |    4798 |           0.262818 |              0.949979 |           0.17294    |              0.200081   |
| ('iid', 1)   |    2856 |           0.679972 |              0.964286 |           0.108663   |              0.136091   |
| ('iid', 2)   |    1930 |           0.687565 |              0.932124 |           0.0804578  |              0.131322   |
| ('iid', 3)   |    1049 |           0.577693 |              0.914204 |           0.0280395  |              0.097629   |
| ('iid', 4)   |    2063 |           0.824043 |              0.958313 |           0.00626775 |              0.00985661 |

## Tail risk

|                                         |   cases |   evaluable_risk |   underestimation |       mae |   median_ess |
|:----------------------------------------|--------:|-----------------:|------------------:|----------:|-------------:|
| ('block', 'daily industry/factor')      |    1104 |             1096 |          0.664234 | 0.0112008 |     56.0736  |
| ('block', 'daily size characteristic')  |    2160 |             2135 |          0.745199 | 0.0122827 |     78.5832  |
| ('block', 'monthly industry')           |    2616 |             2013 |          0.869846 | 0.0849144 |      2.34246 |
| ('block', 'monthly size--B/M')          |    3216 |             1960 |          0.942347 | 0.105111  |      1       |
| ('block', 'monthly size/profit/invest') |    1440 |             1218 |          0.894089 | 0.0752064 |      3.9718  |
| ('block', 'monthly two-characteristic') |    2160 |             1902 |          0.888538 | 0.0715273 |      4       |
| ('iid', 'daily industry/factor')        |    1104 |             1096 |          0.664234 | 0.0112008 |     56.0736  |
| ('iid', 'daily size characteristic')    |    2160 |             2135 |          0.745199 | 0.0122827 |     78.5832  |
| ('iid', 'monthly industry')             |    2616 |             2013 |          0.869846 | 0.0849144 |      2.34246 |
| ('iid', 'monthly size--B/M')            |    3216 |             1960 |          0.942347 | 0.105111  |      1       |
| ('iid', 'monthly size/profit/invest')   |    1440 |             1218 |          0.894089 | 0.0752064 |      3.9718  |
| ('iid', 'monthly two-characteristic')   |    2160 |             1902 |          0.888538 | 0.0715273 |      4       |

## Boundaries

All selector settings and seed outcomes are in selector_summary.csv, all decisions in decisions.csv. Task-cluster intervals capture one level of dependence; they are not full market-shock uncertainty. Risk-calibration extension and fixed-tau sensitivity must not replace the primary row by outcome preference. A zero false-safe point with no acceptance is undefined, not success. Case counts, source release and policy constructors are documented in the current protocol. These are controlled semi-synthetic benchmark results without transaction costs or market impact.
