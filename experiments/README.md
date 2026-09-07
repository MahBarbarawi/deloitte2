# Temporal semantics comparison

BankSim's `step` column is the zero-based simulated day index. The dataset contains 180 simulated days (`0` through `179`), approximately six months. There is no intraday timestamp, so hour-of-day cannot be recovered.

The frozen model was trained after deriving `day = step // 24` and `hour_of_day = step % 24`. Those legacy fields are semantically incorrect. This experiment compares three otherwise identical pipelines without replacing anything under `notebooks/fraud_dashboard_bundle/artifacts/`:

1. `legacy_baseline`: retains the two legacy fields solely to reproduce the frozen methodology.
2. `corrected_simulation_day`: removes both legacy fields and adds `simulation_day = step`.
3. `no_direct_time`: removes both legacy fields and supplies no direct simulation-position feature.

All variants reuse the notebook's leakage-safe historical feature engineering, chronological train/validation/test split (`0–125`, `126–152`, `153–179`), train-fitted standard scaling and one-hot encoding, train-only median Random Forest feature selection, 25-candidate seeded hyperparameter search, validation PR-AUC/generalization selection rule, and validation-only F1 threshold optimization. The held-out test labels are used once per variant after model and threshold selection.

Run the complete comparison from the repository root:

```bash
.venv/bin/python experiments/temporal_model_comparison.py
```

Results are written only to `experiments/results/temporal_model_comparison/`. The script hashes the frozen artifact directory before and after execution and fails if any artifact changes.
