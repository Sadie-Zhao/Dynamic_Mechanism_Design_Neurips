# Dynamic Mechanism Design

Code and saved results for our NeurIPS 2026 paper on learning dynamic mechanisms.

- **Budget auctions:** principal-side history representations, training, refinement, and distillation.
- **Bandit auctions:** deviation-oracle comparisons across information, network architectures, and optimization methods.

## Contents

- `budget/`: budget-auction implementation, configurations, checkpoints, results, and figures.
- `bandit/`: bandit-auction implementation, deviation oracles, checkpoints, and results.
- `scripts/`: result verification, table export, and plotting.

## Getting started

Verify the saved results and export the experiment tables (Python 3.10+, standard library only):

```bash
python scripts/verify_and_export.py
```

For dependencies and training/evaluation commands, see [Environment and usage](docs/ENVIRONMENT.md).
For result definitions and selection details, see [Results](docs/RESULTS.md).
The budget and bandit implementations use separate environments.
