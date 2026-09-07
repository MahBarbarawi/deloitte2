# BankSim temporal-model comparison

The experiment exactly reproduced the frozen baseline and verified that all files under `notebooks/fraud_dashboard_bundle/artifacts/` retained their pre-run SHA-256 hashes.

These saved tables were produced before the structural refactor from `data/transactions_clean.csv`. That source is byte-identical to the cleaned dashboard CSV (SHA-256 `4abbfd8d...e64df`). The current reproducible entry point starts from `data/fraud.csv` and delegates cleaning, history features, and chronological splitting to `fraud_pipeline`; the saved historical metrics have not been relabeled as a new run.

## Held-out test results

| Variant | Selected features | Threshold | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy baseline | 57 | 0.793550 | 0.924274 | 0.825000 | 0.871820 | 0.843111 | 0.948422 | 0.999042 | 98,964 | 73 | 189 | 891 |
| Corrected simulation day | 56 | 0.786255 | 0.922919 | 0.831481 | 0.874817 | 0.848290 | 0.948600 | 0.999032 | 98,962 | 75 | 182 | 898 |
| No direct time | 56 | 0.729922 | 0.898635 | 0.853704 | 0.875594 | 0.862327 | 0.948155 | 0.999017 | 98,933 | 104 | 158 | 922 |

All three variants selected tuning candidate 1: 200 trees, maximum depth 15, minimum split size 10, minimum leaf size 10, `max_features=0.5`, and `max_samples=0.6`.

## Temporal-feature selection

The legacy `day` and `hour_of_day` fields were not selected by the baseline Random Forest selector. Their selector-stage impurity importances were 0.000619 and 0.000836 respectively, but neither appears in the final fitted model and therefore neither has a final-model importance.

The corrected `simulation_day` feature was also not selected. Its selector-stage importance was 0.001109.

## Interpretation

Correcting the temporal representation did not materially change ranking performance: PR-AUC changed by +0.000178 and ROC-AUC by -0.000010 relative to baseline. At the validation-selected operating thresholds, the corrected model gained seven true positives and two false positives, increasing F1 by 0.002997.

The no-direct-time variant gained 31 true positives and 31 false positives, increasing F1 by 0.003774 and F2 by 0.019216 while reducing precision by 0.025638. Its PR-AUC changed by -0.000267.

These small differences arise from rerunning train-only feature selection and threshold optimization in feature spaces of different sizes, not from direct use of the invalid temporal fields by the frozen final Random Forest. The frozen mask excludes both invalid fields.

## Recommendation

Do not replace the frozen model solely because of the two invalid temporal fields: they are excluded from its final feature mask, and the controlled alternatives do not materially improve discrimination. Keep the UI and documentation corrections. Before any future retraining, choose deliberately between corrected `simulation_day` and no direct time based on the desired precision/recall tradeoff, then repeat the complete validation and test process as one versioned artifact set. The retired Logistic Regression confidence pipeline must not be regenerated as an active dependency.

## Legacy artifact note

The extracted `artifacts/rf_feature_mask.npy` has an invalid NumPy magic prefix and `np.load` cannot read it. The copy inside `notebooks/fraud_dashboard_bundle.zip` is valid, selects 57 features, and exactly matches `selected_feature_names.json`. This pre-existing packaging corruption remains unchanged. Active runtime now derives the mask from the authoritative selected-feature names, validates all 57 names and their order against the fitted Random Forest, and does not depend on the corrupt file.
