# Results and provenance

## Budget study

`budget/results/main_reporting_freeze.json` fixes the main selections. `selected_checkpoints.json` resolves the 24 exact mechanism files, targets, selected epochs, and hashes. Main revenues are selected validation results. Means and sample standard deviations use the three prescribed seeds; they are not held-out test estimates or significance tests.

The primary implementation snapshot is `capacity_selection_20261103_v04/source`. All copied numerical files retain their bytes. `source_overrides/` records differences found in other executed snapshots; these are historical alternatives, not patches automatically applied to the primary package. The original per-trial `identity.json` records identify the source/configuration used by each run. Operational absolute paths inside these records are historical provenance; they are not runnable destinations.

The initial 72-trial grid, shared extensions, lower-target compact trials, five-checkpoint validation selection, and auxiliary studies are retained separately in `results.json` and `results/trials/`. Equal budgets per cold trial do not imply equal total selection opportunities. The main three-seed table remains frozen; later higher-valued explorations do not replace it.

The DIC selection quantity is the sum of positive per-buyer payoff differences. Its sampling upper bound is computed for the selected finite collection of deviation policies. It is not a bound over all policies. `config_index.json` maps the archived configurations.

The paper reports the 1000+1000 distillation runs. Earlier 500+500 records remain archived separately. The latter continues supervised fitting but restarts acceptance training from the selected fit. Expanded e2 runs also lower the target and increase the DIC penalty. The studies do not contain matched fresh-A controls. Teacher identities may differ from the main-table checkpoint: e2 B's distillation teacher has revenue 3.44171309. None of these records changes the main selection.

## Bandit study

Paper M1 maps to code `m15`; paper M2 maps to `infostate` (or `m2` in ES). The mechanisms are:

| Paper setting | Artifact identity |
| --- | --- |
| Discrete one good, M1 trained | `dS_Ap_m15/goal_7.617` |
| Discrete two goods, M1 trained | `dM_Ap_m15/goal_15.250` |
| Discrete two goods, M2 trained | `dM_B_m2/goal_15.250` |
| Continuous one/two goods | `cont_single/ckpt_best`, `cont_multi/ckpt_best` |

`evaluation_grid.json` contains 96 original cells plus all 32 retrained-mechanism cells reconstructed from individual files. Some older aggregate summaries omit two completed GRU records; the individual files are authoritative. The original discrete base NPG+ table cells use the larger `sanity` runs (24 restarts and 100,000 evaluation trajectories), not a lower-budget alternative in `dm15`.

The common paper heading is **evaluated DIC loss**. The saved calculation is recorded per cell: discrete NPG is buyer 1's gain; discrete PW/ES and continuous evaluations are signed averages across buyers. Negative estimates are retained. Symmetry does not equate these realized finite-search estimates or turn an average into a maximum. The finite single-good DP evaluates both buyers: `noncla_regret_max = 0.0870164633`, with the exhaustive public-outcome reconstruction check supplied separately. Do not substitute the larger clairvoyant field named `exact_regret_max_over_buyers`.

The raw continuous two-good M1/NPG/Wide result is 0.0014720452 and rounds to 0.001, as in the updated manuscript. Older rebuttal text rounded it to 0.002. The saved continuous initial sampler is Gaussian without truncation; clipping occurs in the later logit transition/report transformation. The original continuous PW records retain the stopped derivative through the truthful residual component; the corrected diagnostic is separate. Its single-good configuration is M2/Wide and its two-good configuration is M1/Wide. The latter is not an M2/GRU run.

Discrete saved DIR diagnostics are trajectory-level negative-utility proxies, divided by two for per-buyer training selection. They are not a statement that every mechanism's summed ex-ante DIR loss is below 0.1. Exact single-good truthful expected utilities are positive. Continuous historical rounded payoffs do not match the later frozen-checkpoint forward checks; the updated appendix therefore does not reuse those historical payoff claims.

## What can be reconstructed offline

The table exporter checks and regenerates all main budget selections and all main bandit cells from copied JSON. The compact auxiliary plot reads saved trajectories only. Most historical bandit maximum-search records do not include the winning oracle parameter trees. Thus a fresh independent confidence interval for those exact historical selected policies cannot be manufactured from this package. Numerical reruns would be new experiments and should be stored separately.

## Updated training figure and ex-ante DIR evaluation

`budget/figures/budget_revenue_compact.pdf` shows mean training revenue plus/minus one sample standard deviation across the three seeds, after smoothing each run with a trailing 25-epoch average. Regenerate it with `python scripts/plot_budget_training.py` (NumPy and Matplotlib required). Earlier interval plots remain archived; their bands have not been relabeled.

`bandit/results/ex_ante_dir/` contains the forward-only evaluation of all five frozen mechanisms on 65,536 truthful trajectories each. Each buyer has positive estimated expected utility, giving estimated summed ex-ante DIR loss 0.0 for every mechanism. This evaluates the negative part of each buyer's expected utility, rather than averaging trajectory-level negative utilities. These records supersede the earlier uncertainty about ex-ante DIR above.

The single-good truthful-only exact cross-check agrees with this Monte Carlo evaluation but differs from the historical DP record's truthful utility. The historical deviation-gain result was not recomputed by this DIR evaluation; the discrepancy is retained in the saved records.
