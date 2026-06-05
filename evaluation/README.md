# Evaluation Entry Points

This directory provides stable public-facing evaluation entrypoints.

- `threshold_selection.py`: wrapper around the validation-threshold sweep workflow.
- `evaluate_fixed_test.py`: wrapper around fixed-split checkpoint evaluation.

The underlying implementation remains in the main repository scripts:

- `scripts/reports/run_threshold_sweep_report.py`
- `test.py`
