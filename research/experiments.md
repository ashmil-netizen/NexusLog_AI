# Experiments

## Dataset cohort

The verified Olist snapshot contains 99,441 orders. Preparation retained 96,221 delivered orders with valid target timestamps: 7,821 were late (8.13%) and 88,400 were on time. Exclusions were 625 canceled orders, 2,338 other non-delivered orders, 8 delivered orders missing an actual or estimated delivery timestamp, and 249 delivered rows with invalid lifecycle timestamps. There were no duplicate order IDs or missing purchase timestamps in this snapshot. Stage eligibility before chronological splitting was 96,221 (A), 96,206 (B), and 96,205 (C).

## Experiment configuration and validation selection

All stages use the oldest 70% of eligible orders for training, the next 15% for validation, and the newest 15% for final testing. Preprocessing is fitted only on training data. Logistic regression and random forest were compared by validation PR-AUC; the selected fitted model was sigmoid-calibrated on validation data. The alert threshold minimized validation cost under illustrative costs of 1 per false positive and 3 per false negative.

| Stage | Selected model | Logistic validation PR-AUC | Random forest validation PR-AUC | Selected validation PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|
| A — Order placement | Logistic Regression | 0.0807 | 0.0793 | 0.0807 | 0.95 |
| B — Payment approval | Logistic Regression | 0.0813 | 0.0791 | 0.0813 | 0.95 |
| C — Carrier handover | Random Forest | 0.1430 | 0.1474 | 0.1474 | 0.25 |

| Stage | Training period | Validation period |
|---|---|---|
| A | 2016-09-15 12:16:38 – 2018-04-15 08:25:08 | 2018-04-15 08:33:12 – 2018-06-20 16:55:30 |
| B | 2016-10-03 09:44:50 – 2018-04-15 09:53:19 | 2018-04-15 10:17:36 – 2018-06-20 17:18:04 |
| C | 2016-10-03 09:44:50 – 2018-04-15 09:53:19 | 2018-04-15 10:17:36 – 2018-06-20 17:18:04 |

## Final test results

| Stage | Train / validation / test N | Test period | Test PR-AUC | Brier score | Precision | Recall | Confusion matrix (actual rows: on-time, delayed; predicted columns: on-time, delayed) | Test cost |
|---|---|---|---:|---:|---:|---:|---|---:|
| A | 67,354 / 14,433 / 14,434 | 2018-06-20 17:02:01 – 2018-08-29 15:00:37 | 0.0569 | 0.0629 | 0.0000 | 0.0000 | `[[13480, 1], [953, 0]]` | 2,860 |
| B | 67,344 / 14,431 / 14,431 | 2018-06-20 17:18:05 – 2018-08-29 15:00:37 | 0.0590 | 0.0629 | 0.0000 | 0.0000 | `[[13477, 1], [953, 0]]` | 2,860 |
| C | 67,343 / 14,431 / 14,431 | 2018-06-20 17:18:05 – 2018-08-29 15:00:37 | 0.1959 | 0.0583 | 0.4857 | 0.1070 | `[[13370, 108], [851, 102]]` | 2,661 |

Training and validation date boundaries, calibration-curve points, the full threshold curves, validation-selected costs, confusion matrices, permutation importance, and test-set order probabilities are recorded in the generated `backend/reports/` outputs. The final test-period cost at the selected threshold is 2,860 (A), 2,860 (B), and 2,661 (C). These use the hypothetical costs above and were not used to select thresholds.

## Interpretation

Under these configurations and cost assumptions, stages A and B select a high threshold and detect no delayed orders in the final test period; their zero recall is a material limitation, not an acceptable alerting result. Stage C has higher test PR-AUC and detects 102 of 953 delayed test orders, with 108 false alerts. The observed difference is consistent with more predictive information being available at handover, but it does not establish causality or guarantee similar results elsewhere. The reports should be inspected before any operational use.

The experiment is reproducible from the checked-in scripts and fixed random seed (`42`). The downloaded data and model/report artifacts are local generated files and are not committed.
