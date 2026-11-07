# Software and numerical entry points

## Offline verification and plots

Python 3.10 or newer suffices for `scripts/verify_and_export.py`. Plotting additionally needs NumPy and Matplotlib. These commands do not need JAX and do not run experiments.

```sh
python scripts/verify_and_export.py
python scripts/plot_distillation.py
python scripts/plot_budget_training.py
```

## Budget

The preserved `requirements-lock.txt` identifies JAX/JAXlib 0.4.38, Haiku 0.0.13, Optax 0.2.4, Flax 0.10.2, and NumPy 1.26.4. CPU and GPU requirement files are included separately. Use an isolated environment matching the intended platform; no package installation was done during assembly.

One cold trial, if later requested, can be launched from `budget/`:

```sh
python train_trial.py --config configs/e1_A_s02_t02.json --output new_runs/e1_A_seed2
```

This invokes the exact archived training schedule and random stream. It produces training state and checkpoints only. It does not reproduce the strong selection audits, relabel the run as a paper result, or overwrite an existing directory. Full selection uses the preserved capacity/adaptive engines and evaluation kernels; archived registration paths must be remapped to a new run directory and the relevant historical source snapshot. Auxiliary studies additionally require their recorded teacher and supervised-fit state.

## Bandit

The historical source README documents a Linux GPU environment with JAX 0.4.28, Haiku 0.0.12, Optax 0.2.2, and Flax 0.8.5, on an eight-L40S host. Later diagnostic identity records also retain source/version information. These records are not a freshly verified dependency lock; in particular, historical Haiku development-version provenance is not fully resolved. Preserve checkpoint identities when choosing a compatible runtime.

Scientific dependencies are JAX/JAXlib, dm-haiku, Optax, Flax, NumPy, PyYAML, and Matplotlib. GPU wheel selection depends on CUDA compatibility. Do not combine the budget lock with the historical bandit runtime without checking compatibility.

Examples from `bandit/`, for later numerical reproduction:

```sh
# The selected training configuration, not the driver's obsolete default policy.
python audit/retrain_discrete_single.py --env multi --policy m15 --goals 15.25 --out_dir new_runs/discrete_multi_m1
python audit/retrain_discrete_single.py --env multi --policy infostate --goals 15.25 --out_dir new_runs/discrete_multi_m2

# Continuous training from the saved configuration, writing somewhere new.
python -m bandit_auction.train.run_acceptance --config mechanisms/continuous/cont_single/ckpt_best/config.yaml --output_dir new_runs/continuous_single

# One fixed-mechanism discrete oracle cell, with explicit historical controls.
python audit/belief_npg_oracle.py --env multi --input infostate --run_dir mechanisms/discrete/dM_B_m2/goal_15.250 --out new_runs/m2_npg.json --hidden 32 --restarts 6 --iters 300 --episodes 512 --eta 0.3 --lam 0.001 --eval_episodes 20000
```

The architecture-specific and larger-budget controls are in `results/evaluation_grid.json`; the examples above do not reproduce the entire grid. Numerical entry points and GPU compatibility were not executed or tested during this assembly. The successful checks concern artifact integrity, table reconstruction, source syntax, and manuscript compilation.

## Ex-ante DIR evaluation

The saved forward-only evaluation used JAX/JAXlib 0.4.28, Haiku 0.0.12, Flax 0.8.4, Optax 0.2.2, and NumPy 1.26.4. Package versions and checkpoint hashes are recorded in `bandit/results/ex_ante_dir/results.json`.

To perform a new truthful-rollout evaluation, run from the repository root in the bandit environment:

```sh
python scripts/evaluate_ex_ante_dir.py --output bandit/new_runs/ex_ante_dir
```

The output directory must not exist. This performs numerical evaluation, without training or deviation search. The portable wrapper changes paths and output handling only; it has been syntax-checked, not numerically rerun during publication.
