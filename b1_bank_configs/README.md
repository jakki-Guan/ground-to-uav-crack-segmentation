# B1 Bank Configs

This directory collects public-facing configuration files and wrapper scripts for the `B1` hard-negative bank workflow.

- `b1_candidate_bank_parameters.yaml`: candidate mining and filtering presets that match the repository's current paper-facing bank family.
- `b1_mining_and_filtering.py`: wrapper that reads the YAML presets and dispatches to the underlying mining and filtering scripts in `scripts/banks/`.

The wrapper keeps the public entrypoint stable while the lower-level implementation stays under `scripts/banks/`.
