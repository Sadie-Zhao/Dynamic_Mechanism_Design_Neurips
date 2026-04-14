"""Helper script to generate the two Colab notebooks."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(BASE, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines if isinstance(lines, list) else [lines]}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text if isinstance(text, list) else [text]}


def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# 3-stage HP search + final training cells
# ---------------------------------------------------------------------------

def hp_search_cells(exp_label, hp_save_dir_expr, final_save_dir_expr,
                    profit_goal_values):
    """
    exp_label          : 'Exp 1' or 'Exp 2'
    hp_save_dir_expr   : Python expression string, e.g. "'outputs/public_budget/hparam_search'"
    final_save_dir_expr: Python expression string, e.g. "f'outputs/public_budget/seed{SEED}/optimal'"
    profit_goal_values : list of floats to search over, e.g. [0.8, 1.2, 1.5, 2.0]

    Stage design
    ------------
    Stage 1  Anchor lr_theta × lr_w            (12 trials, fast)
             Fixed: alpha_weight=1e5, beta_weight=1e3,
                    profit_goal=cfg default, num_inner_steps=150
             Rationale: Adam's adaptive scaling makes LRs semi-independent of
             loss magnitudes — good to lock down first without the confound of
             varying weights.

    Stage 2  Joint search of profit_goal × alpha_weight × beta_weight  (36 trials)
             Fixed: lr_theta*, lr_w* from Stage 1, num_inner_steps=200
             Rationale: these three are tightly coupled — alpha_weight must
             scale with profit_goal to keep the IC penalty competitive with the
             revenue-gap term.  Must be searched together.

    Stage 3  Fine-tune num_inner_steps          (4 trials, cheap)
             Fixed: everything from Stage 1 + 2
             Rationale: inner-step count depends on how hard the resulting
             mechanism is to break (determined by profit_goal + weights).
             Fine-tuning last avoids confounds.

    Total ≈ 52 trials vs 108 in a flat search, with better coverage of
    the interactions that actually matter.
    """
    pg_list_repr = repr(profit_goal_values)   # e.g. "[0.8, 1.2, 1.5, 2.0]"

    return [

        # ================================================================
        # MD header
        # ================================================================
        md(f"## Hyperparameter Search — 3-Stage Staged Pipeline\n\n"
           "Hyperparameters are tuned in three sequential stages, each building on "
           "the previous, so that only the parameters relevant to each stage are "
           "free while the others are held fixed. Results are written to disk after "
           "every trial; re-running the cell resumes from where it left off.\n\n"
           "| Stage | Free parameters | Fixed | Trials |\n"
           "|-------|----------------|-------|--------|\n"
           "| **1** | `lr_theta`, `lr_w` | moderate weights, default profit_goal | 12 |\n"
           "| **2** | `profit_goal`, `alpha_weight`, `beta_weight` | lr* from S1 | 36 |\n"
           "| **3** | `num_inner_steps` | all S1+S2 best values | 4 |\n\n"
           "**Scoring** (all stages): `revenue − 10·max(0, IC−ε) − 10·max(0, IR−ε) "
           "− 10·max(0, IR_viol−ε) − 10·max(0, BV−ε)`  \n"
           "Revenue is the objective; IC, IR, and feasibility are explicit constraints."),

        # ================================================================
        # Configuration + shared helper functions
        # ================================================================
        code([
            "import itertools, random, time, math\n",
            "\n",
            "# ---- Global time budget ----\n",
            "HP_TIME_BUDGET_HOURS = 12\n",
            "\n",
            "# ---- Per-stage epoch budgets ----\n",
            "HP_S1_EPOCHS = 300   # Stage 1: LR anchor  (fast, just need stability signal)\n",
            "HP_S2_EPOCHS = 500   # Stage 2: weights + profit_goal\n",
            "HP_S3_EPOCHS = 500   # Stage 3: inner steps\n",
            "HP_SAMPLES   = 64    # MC samples during training (all stages)\n",
            "HP_EVAL_KEYS = 16    # eval keys per trial\n",
            "HP_EVAL_SAMPS = 128  # eval MC samples per trial\n",
            "\n",
            "# ---- Constraint tolerances ----\n",
            "# IC and IR are *constraints*, not objective terms.\n",
            "# Revenue is maximised freely; violations are penalised once they\n",
            "# exceed tolerance (10 units penalty per unit of excess).\n",
            "HP_IC_LOSS_THRESH = 0.05   # cumulative regret\n",
            "HP_IR_LOSS_THRESH = 0.05   # sum |reward|\n",
            "HP_IR_VIOL_THRESH = 0.05   # IR violation rate\n",
            "HP_BV_THRESH      = 0.05   # budget violation rate\n",
            "\n",
            f"hp_save_dir = {hp_save_dir_expr}\n",
            "os.makedirs(hp_save_dir, exist_ok=True)\n",
            "\n",
            "# ---- Scoring function ----\n",
            "def _hp_score(r):\n",
            "    bv_pen = 10.0 * max(0.0, r['budget_viol_mean'] - HP_BV_THRESH)\n",
            "    iv_pen = 10.0 * max(0.0, r['ir_viol_mean']    - HP_IR_VIOL_THRESH)\n",
            "    ir_pen = 10.0 * max(0.0, r['ir_loss_mean']    - HP_IR_LOSS_THRESH)\n",
            "    _rg = r.get('regret_mean', float('nan'))\n",
            "    ic_pen = 0.0 if (_rg != _rg) else 10.0 * max(0.0, _rg - HP_IC_LOSS_THRESH)\n",
            "    return r['revenue_mean'] - bv_pen - iv_pen - ir_pen - ic_pen\n",
            "\n",
            "# ---- Single-trial runner ----\n",
            "# trial_cfg must contain all keys needed by nested_GD_budget.\n",
            "def _run_trial(trial_cfg, trial_seed, n_epochs, n_samples):\n",
            "    _ni = trial_cfg['num_inner_steps']\n",
            "    _pg = trial_cfg['profit_goal']\n",
            "    _mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
            "    _pnet = build_bounded_policy_net(\n",
            "        d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
            "        G=auction.num_goods, hidden=cfg['policy_hidden'],\n",
            "        residual_init_std=cfg['residual_init_std'],\n",
            "    )\n",
            "    _tl = nested_GD_budget(\n",
            "        auction=auction,\n",
            "        mechanism_network=_mnet, policy_network=_pnet,\n",
            "        profit_goal=_pg,\n",
            "        num_outer_epochs=n_epochs,\n",
            "        lr_theta=trial_cfg['lr_theta'],\n",
            "        num_inner_steps=_ni,\n",
            "        lr_w=trial_cfg['lr_w'],\n",
            "        num_samples_train=n_samples,\n",
            "        alpha_weight=trial_cfg['alpha_weight'],\n",
            "        beta_weight=trial_cfg['beta_weight'],\n",
            "        print_iter=10**7,\n",
            "        seed=trial_seed,\n",
            "    )\n",
            "    _mp = _tl.pop('mechanism_params'); _pp = _tl.pop('policy_params')\n",
            "    _ek = jax.random.split(jax.random.PRNGKey(trial_seed + 99999), HP_EVAL_KEYS)\n",
            "    _ev = eval_budget_mechanism_multikey(\n",
            "        auction=auction, mechanism_network=_mnet, mechanism_params=_mp,\n",
            "        profit_goal=_pg, num_samples=HP_EVAL_SAMPS, keys=_ek,\n",
            "        policy_network=_pnet, policy_params=_pp,\n",
            "    )\n",
            "    return {\n",
            "        'config':           trial_cfg,\n",
            "        'revenue_mean':     float(_ev['profit_stats']['mean']),\n",
            "        'revenue_std':      float(_ev['profit_stats']['std']),\n",
            "        'ir_loss_mean':     float(_ev['ir_loss_stats']['mean']),\n",
            "        'budget_viol_mean': float(_ev['budget_viol_stats']['mean']),\n",
            "        'ir_viol_mean':     float(_ev['ir_viol_stats']['mean']),\n",
            "        'gap_plus_mean':    float(_ev['gap_plus_stats']['mean']),\n",
            "        'regret_mean':      float(_ev.get('regret_stats', {}).get('mean', float('nan'))),\n",
            "        'final_reg_train':  float(_tl['reg'][-1]) if _tl.get('reg') else float('nan'),\n",
            "        'final_ir_train':   float(_tl['ir'][-1])  if _tl.get('ir')  else float('nan'),\n",
            "        'seed':             trial_seed,\n",
            "    }\n",
            "\n",
            "# ---- Stage runner (handles resumption, time budget, fault tolerance) ----\n",
            "def _run_stage(stage_name, combos, fixed_cfg, n_epochs, n_samples,\n",
            "               results_path, search_start, budget_s):\n",
            "    if os.path.exists(results_path):\n",
            "        with open(results_path) as _f:\n",
            "            _results = json.load(_f)\n",
            "        print(f'  [{stage_name}] loaded {len(_results)} prior results (resuming)')\n",
            "        _done = {str(r['config']) for r in _results}\n",
            "    else:\n",
            "        _results = []; _done = set()\n",
            "    for _i, _tcfg in enumerate(combos):\n",
            "        _full = {**fixed_cfg, **_tcfg}   # search overrides fixed\n",
            "        if str(_full) in _done:\n",
            "            continue\n",
            "        _el = time.time() - search_start\n",
            "        if _el >= budget_s:\n",
            "            print(f'  [{stage_name}] time budget reached at trial {_i}'); break\n",
            "        _rem = (budget_s - _el) / 3600\n",
            "        print(f'  [{stage_name}] {_i+1}/{len(combos)} rem={_rem:.2f}h | {_tcfg}')\n",
            "        _tseed = SEED + abs(hash(stage_name)) % 9999 + _i\n",
            "        try:\n",
            "            _rec = _run_trial(_full, _tseed, n_epochs, n_samples)\n",
            "            _rec['score'] = _hp_score(_rec)\n",
            "            _results.append(_rec); _done.add(str(_full))\n",
            "            with open(results_path, 'w') as _f:\n",
            "                json.dump(_results, _f, indent=2)\n",
            "            _rv = _rec['revenue_mean']; _sc = _rec['score']\n",
            "            _rg = _rec.get('regret_mean', float('nan'))\n",
            "            _ir = _rec['ir_loss_mean']\n",
            "            print(f'    => rev={_rv:.4f} regret={_rg:.4f} ir={_ir:.4f} score={_sc:.4f}')\n",
            "        except Exception as _exc:\n",
            "            print(f'    [FAILED] {_exc}')\n",
            "    return sorted(_results, key=lambda r: r['score'], reverse=True)\n",
            "\n",
            "# Shared timer (covers all three stages)\n",
            "_search_start = time.time()\n",
            "_budget_s     = HP_TIME_BUDGET_HOURS * 3600\n",
            "print(f'Search started. Budget: {HP_TIME_BUDGET_HOURS}h')\n",
            "print(f'Thresholds — IC<={HP_IC_LOSS_THRESH} IR<={HP_IR_LOSS_THRESH} '\n",
            "      f'IV<={HP_IR_VIOL_THRESH} BV<={HP_BV_THRESH}')\n",
        ]),

        # ================================================================
        # Stage 1
        # ================================================================
        md("### Stage 1 — Anchor Learning Rates\n\n"
           "Adam's adaptive gradient scaling makes `lr_theta` and `lr_w` partially "
           "independent of loss weight magnitudes, so they can be anchored before the "
           "weights are tuned. We hold `alpha_weight`, `beta_weight`, and `profit_goal` "
           "at moderate defaults and search a 4×3 = 12-trial grid.\n\n"
           "Criterion: score (revenue subject to IC/IR constraints). "
           "A large spread in scores here indicates strong LR sensitivity; "
           "a flat landscape means any LR in the range is acceptable."),

        code([
            "_s1_search = {\n",
            "    'lr_theta': [3e-5, 1e-4, 3e-4, 1e-3],\n",
            "    'lr_w':     [1e-4, 3e-4, 1e-3],\n",
            "}\n",
            "_s1_fixed = {\n",
            "    'alpha_weight':    1e5,\n",
            "    'beta_weight':     1e3,\n",
            "    'profit_goal':     cfg['profit_goal'],\n",
            "    'num_inner_steps': 150,\n",
            "}\n",
            "_s1_combos = [dict(zip(_s1_search, c))\n",
            "              for c in itertools.product(*_s1_search.values())]\n",
            "random.seed(SEED); random.shuffle(_s1_combos)\n",
            "print(f'Stage 1: {len(_s1_combos)} combos x {HP_S1_EPOCHS} epochs')\n",
            "\n",
            "_s1_results = _run_stage(\n",
            "    'S1-LR', _s1_combos, _s1_fixed, HP_S1_EPOCHS, HP_SAMPLES,\n",
            "    os.path.join(hp_save_dir, 's1_lr.json'), _search_start, _budget_s,\n",
            ")\n",
            "\n",
            "lr_theta_star = _s1_results[0]['config']['lr_theta']\n",
            "lr_w_star     = _s1_results[0]['config']['lr_w']\n",
            "print(f'\\nStage 1 best: lr_theta={lr_theta_star:.1e}  lr_w={lr_w_star:.1e}')\n",
            "print(f'  score={_s1_results[0][\"score\"]:.4f}  revenue={_s1_results[0][\"revenue_mean\"]:.4f}')\n",
            "\n",
            "# Sensitivity plots\n",
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n",
            "for _ax, _key, _star in zip(axes, ['lr_theta', 'lr_w'], [lr_theta_star, lr_w_star]):\n",
            "    _vals = sorted(set(r['config'][_key] for r in _s1_results))\n",
            "    _means = []\n",
            "    for _v in _vals:\n",
            "        _sub = [r['score'] for r in _s1_results if r['config'][_key] == _v]\n",
            "        _ax.scatter([_v]*len(_sub), _sub, alpha=0.6, s=40)\n",
            "        _means.append(sum(_sub)/len(_sub))\n",
            "    _ax.plot(_vals, _means, 'k--', alpha=0.4, label='mean')\n",
            "    _ax.axvline(_star, color='red', linestyle='--', label=f'best={_star:.0e}')\n",
            "    _ax.set_xscale('log'); _ax.set_xlabel(_key)\n",
            "    _ax.set_ylabel('Score'); _ax.legend(); _ax.set_title(f'Stage 1 — {_key}')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),

        # ================================================================
        # Stage 2
        # ================================================================
        md("### Stage 2 — Joint Search: Profit Goal × Loss Weights\n\n"
           "`profit_goal`, `alpha_weight`, and `beta_weight` are tightly coupled: "
           "raising `profit_goal` increases the gap-term magnitude, making the IC penalty "
           "relatively weaker unless `alpha_weight` is raised proportionally. "
           "Searching them jointly (with LRs anchored) reveals the correct trade-off surface "
           "and identifies the highest revenue target that still converges to a feasible mechanism.\n\n"
           "**Fixed:** `lr_theta*`, `lr_w*` from Stage 1, `num_inner_steps=200`  \n"
           "**Search:** `profit_goal` × `alpha_weight` × `beta_weight` (3×3×3 = **27 trials**)"),

        code([
            f"_s2_pg = {pg_list_repr}   # candidate revenue targets\n",
            "_s2_search = {\n",
            "    'profit_goal':  _s2_pg,\n",
            "    'alpha_weight': [1e4, 1e5, 1e6],\n",
            "    'beta_weight':  [5e2, 1e3, 5e3],\n",
            "}\n",
            "_s2_fixed = {\n",
            "    'lr_theta':        lr_theta_star,\n",
            "    'lr_w':            lr_w_star,\n",
            "    'num_inner_steps': 200,\n",
            "}\n",
            "_s2_combos = [dict(zip(_s2_search, c))\n",
            "              for c in itertools.product(*_s2_search.values())]\n",
            "random.seed(SEED + 1); random.shuffle(_s2_combos)\n",
            "print(f'Stage 2: {len(_s2_combos)} combos x {HP_S2_EPOCHS} epochs')\n",
            "\n",
            "_s2_results = _run_stage(\n",
            "    'S2-Weights', _s2_combos, _s2_fixed, HP_S2_EPOCHS, HP_SAMPLES,\n",
            "    os.path.join(hp_save_dir, 's2_weights.json'), _search_start, _budget_s,\n",
            ")\n",
            "\n",
            "profit_goal_star  = _s2_results[0]['config']['profit_goal']\n",
            "alpha_weight_star = _s2_results[0]['config']['alpha_weight']\n",
            "beta_weight_star  = _s2_results[0]['config']['beta_weight']\n",
            "print(f'\\nStage 2 best:')\n",
            "print(f'  profit_goal={profit_goal_star}  '\n",
            "      f'alpha_weight={alpha_weight_star:.0e}  beta_weight={beta_weight_star:.0e}')\n",
            "print(f'  score={_s2_results[0][\"score\"]:.4f}  revenue={_s2_results[0][\"revenue_mean\"]:.4f}')\n",
            "\n",
            "# Sensitivity plots\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "for _ax, _key, _star, _log in zip(\n",
            "    axes,\n",
            "    ['profit_goal', 'alpha_weight', 'beta_weight'],\n",
            "    [profit_goal_star, alpha_weight_star, beta_weight_star],\n",
            "    [False, True, True],\n",
            "):\n",
            "    _vals  = sorted(set(r['config'][_key] for r in _s2_results))\n",
            "    _means = []\n",
            "    for _v in _vals:\n",
            "        _sub = [r['score'] for r in _s2_results if r['config'][_key] == _v]\n",
            "        _ax.scatter([_v]*len(_sub), _sub, alpha=0.6, s=40)\n",
            "        _means.append(sum(_sub)/len(_sub))\n",
            "    _ax.plot(_vals, _means, 'k--', alpha=0.4, label='mean')\n",
            "    _ax.axvline(_star, color='red', linestyle='--', label=f'best={_star}')\n",
            "    if _log: _ax.set_xscale('log')\n",
            "    _ax.set_xlabel(_key); _ax.set_ylabel('Score')\n",
            "    _ax.legend(); _ax.set_title(f'Stage 2 — {_key}')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),

        # ================================================================
        # Stage 3
        # ================================================================
        md("### Stage 3 — Fine-Tune Inner Oracle Steps\n\n"
           "With the mechanism target locked in (from Stage 2), the difficulty of the IC "
           "problem is fixed. Too few inner steps and the oracle underestimates regret, "
           "producing a falsely optimistic IC loss; too many wastes compute with diminishing "
           "returns. We identify the knee of the IC-quality vs compute curve.\n\n"
           "**Fixed:** all Stage 1 + 2 best values  \n"
           "**Search:** `num_inner_steps` ∈ {100, 200, 300, 500} (**4 trials**)"),

        code([
            "_s3_search = {'num_inner_steps': [100, 200, 300, 500]}\n",
            "_s3_fixed = {\n",
            "    'lr_theta':     lr_theta_star,\n",
            "    'lr_w':         lr_w_star,\n",
            "    'profit_goal':  profit_goal_star,\n",
            "    'alpha_weight': alpha_weight_star,\n",
            "    'beta_weight':  beta_weight_star,\n",
            "}\n",
            "_s3_combos = [{'num_inner_steps': n} for n in _s3_search['num_inner_steps']]\n",
            "print(f'Stage 3: {len(_s3_combos)} combos x {HP_S3_EPOCHS} epochs')\n",
            "\n",
            "_s3_results = _run_stage(\n",
            "    'S3-InnerSteps', _s3_combos, _s3_fixed, HP_S3_EPOCHS, HP_SAMPLES,\n",
            "    os.path.join(hp_save_dir, 's3_inner.json'), _search_start, _budget_s,\n",
            ")\n",
            "\n",
            "inner_steps_star = _s3_results[0]['config']['num_inner_steps']\n",
            "print(f'\\nStage 3 best: num_inner_steps={inner_steps_star}')\n",
            "print(f'  score={_s3_results[0][\"score\"]:.4f}  revenue={_s3_results[0][\"revenue_mean\"]:.4f}')\n",
            "\n",
            "# IC regret vs inner steps (the key diagnostic)\n",
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
            "_ns = [r['config']['num_inner_steps']      for r in _s3_results]\n",
            "_sc = [r['score']                           for r in _s3_results]\n",
            "_rg = [r.get('regret_mean', float('nan'))  for r in _s3_results]\n",
            "axes[0].plot(_ns, _sc, 'o-b'); axes[0].set_xlabel('num_inner_steps')\n",
            "axes[0].set_ylabel('Score');   axes[0].set_title('Stage 3 — Score')\n",
            "axes[0].axvline(inner_steps_star, color='red', linestyle='--')\n",
            "axes[1].plot(_ns, _rg, 's-', color='darkorange')\n",
            "axes[1].set_xlabel('num_inner_steps')\n",
            "axes[1].set_ylabel('IC Regret (eval)'); axes[1].set_title('IC Regret vs inner steps')\n",
            "axes[1].axhline(HP_IC_LOSS_THRESH, color='red', linestyle=':', label=f'thresh={HP_IC_LOSS_THRESH}')\n",
            "axes[1].legend()\n",
            "plt.tight_layout(); plt.show()\n",
        ]),

        # ================================================================
        # Consolidate
        # ================================================================
        code([
            "# ---- Consolidate optimal configuration from all 3 stages ----\n",
            "best_cfg_hp = {\n",
            "    'lr_theta':        lr_theta_star,\n",
            "    'lr_w':            lr_w_star,\n",
            "    'profit_goal':     profit_goal_star,\n",
            "    'alpha_weight':    alpha_weight_star,\n",
            "    'beta_weight':     beta_weight_star,\n",
            "    'num_inner_steps': inner_steps_star,\n",
            "}\n",
            "\n",
            "print('=' * 62)\n",
            f"print('OPTIMAL HYPERPARAMETERS — {exp_label}')\n",
            "print('=' * 62)\n",
            "for _k, _v in best_cfg_hp.items():\n",
            "    print(f'  {_k:20s}: {_v}')\n",
            "_tot = len(_s1_results) + len(_s2_results) + len(_s3_results)\n",
            "_th  = (time.time() - _search_start) / 3600\n",
            "print(f'\\nTotal: {_tot} trials in {_th:.2f}h')\n",
            "\n",
            "_search_summary = {\n",
            "    'optimal_config':  best_cfg_hp,\n",
            "    'stage1_best':     _s1_results[0] if _s1_results else None,\n",
            "    'stage2_best':     _s2_results[0] if _s2_results else None,\n",
            "    'stage3_best':     _s3_results[0] if _s3_results else None,\n",
            "    'total_trials':    _tot,\n",
            "    'total_time_h':    _th,\n",
            "    'thresholds':      {'IC': HP_IC_LOSS_THRESH, 'IR': HP_IR_LOSS_THRESH,\n",
            "                        'IR_viol': HP_IR_VIOL_THRESH, 'BV': HP_BV_THRESH},\n",
            "}\n",
            "with open(os.path.join(hp_save_dir, 'best_hparams.json'), 'w') as _f:\n",
            "    json.dump(_search_summary, _f, indent=2)\n",
            "print(f'Saved: {hp_save_dir}/best_hparams.json')\n",
        ]),

        # ================================================================
        # Final training
        # ================================================================
        md("## Final Training with Optimal Hyperparameters\n\n"
           "Retrain for the full epoch budget using the staged-search optimum. "
           "Note that `profit_goal` is now set to the value found in Stage 2, "
           "not the configuration default."),

        code([
            "print(f'Retraining — {cfg[\"num_outer_epochs\"]} epochs with optimal config:')\n",
            "for _k, _v in best_cfg_hp.items():\n",
            "    print(f'  {_k:20s}: {_v}')\n",
            "\n",
            "final_mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
            "final_pnet = build_bounded_policy_net(\n",
            "    d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
            "    G=auction.num_goods, hidden=cfg['policy_hidden'],\n",
            "    residual_init_std=cfg['residual_init_std'],\n",
            ")\n",
            "final_logs = nested_GD_budget(\n",
            "    auction=auction,\n",
            "    mechanism_network=final_mnet, policy_network=final_pnet,\n",
            "    profit_goal=best_cfg_hp['profit_goal'],\n",
            "    num_outer_epochs=cfg['num_outer_epochs'],\n",
            "    lr_theta=best_cfg_hp['lr_theta'],\n",
            "    num_inner_steps=best_cfg_hp['num_inner_steps'],\n",
            "    lr_w=best_cfg_hp['lr_w'],\n",
            "    num_samples_train=cfg['num_samples_train'],\n",
            "    alpha_weight=best_cfg_hp['alpha_weight'],\n",
            "    beta_weight=best_cfg_hp['beta_weight'],\n",
            "    print_iter=cfg['print_iter'],\n",
            "    seed=SEED,\n",
            ")\n",
            "final_mechanism_params = final_logs.pop('mechanism_params')\n",
            "final_policy_params    = final_logs.pop('policy_params')\n",
            "print('Final training complete.')\n",
        ]),

        code([
            "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
            "axes[0].plot(final_logs['reg_plus']); axes[0].set_title('Regret+');    axes[0].set_xlabel('Epoch')\n",
            "axes[1].plot(final_logs['gap_plus']); axes[1].set_title('Profit Gap+'); axes[1].set_xlabel('Epoch')\n",
            "axes[2].plot(final_logs['ir']);        axes[2].set_title('IR Loss');     axes[2].set_xlabel('Epoch')\n",
            "axes[3].plot(final_logs['loss']);      axes[3].set_title('Total Loss');  axes[3].set_xlabel('Epoch')\n",
            f"plt.suptitle(f'{exp_label} — Variant {{VARIANT}} (optimal hparams)', y=1.02)\n",
            "plt.tight_layout(); plt.show()\n",
        ]),

        code([
            "final_eval_keys = jax.random.split(jax.random.PRNGKey(88888), 32)\n",
            "final_results = eval_budget_mechanism_multikey(\n",
            "    auction=auction, mechanism_network=final_mnet,\n",
            "    mechanism_params=final_mechanism_params,\n",
            "    profit_goal=best_cfg_hp['profit_goal'],\n",
            "    num_samples=256, keys=final_eval_keys,\n",
            "    policy_network=final_pnet, policy_params=final_policy_params,\n",
            ")\n",
            f"print_eval_report(final_results, prefix=f'{exp_label} Variant {{VARIANT}} (optimal): ')\n",
        ]),

        # ================================================================
        # Cross-variant comparison
        # ================================================================
        md("## Cross-Variant Comparison\n\n"
           "Train A, B, C under identical optimal hyperparameters "
           "(including the `profit_goal` found in Stage 2). "
           "This is the core experimental comparison."),

        code([
            "variant_results = {\n",
            "    VARIANT: {'logs': final_logs, 'eval': final_results,\n",
            "              'mech_params': final_mechanism_params,\n",
            "              'pol_params':  final_policy_params}\n",
            "}\n",
            "for _v in ['A', 'B', 'C']:\n",
            "    if _v == VARIANT:\n",
            "        continue\n",
            "    print(f'\\n--- Variant {_v} ---')\n",
            "    _vnet  = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[_v]))\n",
            "    _vpnet = build_bounded_policy_net(\n",
            "        d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
            "        G=auction.num_goods, hidden=cfg['policy_hidden'],\n",
            "        residual_init_std=cfg['residual_init_std'],\n",
            "    )\n",
            "    _vl = nested_GD_budget(\n",
            "        auction=auction,\n",
            "        mechanism_network=_vnet, policy_network=_vpnet,\n",
            "        profit_goal=best_cfg_hp['profit_goal'],\n",
            "        num_outer_epochs=cfg['num_outer_epochs'],\n",
            "        lr_theta=best_cfg_hp['lr_theta'],\n",
            "        num_inner_steps=best_cfg_hp['num_inner_steps'],\n",
            "        lr_w=best_cfg_hp['lr_w'],\n",
            "        num_samples_train=cfg['num_samples_train'],\n",
            "        alpha_weight=best_cfg_hp['alpha_weight'],\n",
            "        beta_weight=best_cfg_hp['beta_weight'],\n",
            "        print_iter=cfg['print_iter'],\n",
            "        seed=SEED,\n",
            "    )\n",
            "    _vmp = _vl.pop('mechanism_params'); _vpp = _vl.pop('policy_params')\n",
            "    _vek = jax.random.split(jax.random.PRNGKey(88888), 32)\n",
            "    _vev = eval_budget_mechanism_multikey(\n",
            "        auction=auction, mechanism_network=_vnet, mechanism_params=_vmp,\n",
            "        profit_goal=best_cfg_hp['profit_goal'],\n",
            "        num_samples=256, keys=_vek,\n",
            "        policy_network=_vpnet, policy_params=_vpp,\n",
            "    )\n",
            "    print_eval_report(_vev, prefix=f'Variant {_v}: ')\n",
            "    variant_results[_v] = {'logs': _vl, 'eval': _vev,\n",
            "                            'mech_params': _vmp, 'pol_params': _vpp}\n",
            "\n",
            "# Summary table\n",
            "print('\\n' + '='*72)\n",
            f"print(f'{exp_label} — A vs B vs C  (profit_goal={{profit_goal_star}}, optimal hparams)')\n",
            "print('='*72)\n",
            "print(f'  {\"Var\":>4} | {\"Revenue\":>9} | {\"Gap+\":>7} | {\"IC Reg\":>8} | {\"IR Loss\":>8} | {\"BV\":>6} | {\"IV\":>6}')\n",
            "print('  ' + '-'*68)\n",
            "for _v in ['A', 'B', 'C']:\n",
            "    if _v not in variant_results: continue\n",
            "    _ev = variant_results[_v]['eval']\n",
            "    _rg = _ev.get('regret_stats', {}).get('mean', float('nan'))\n",
            "    print(f'  {_v:>4} | {_ev[\"profit_stats\"][\"mean\"]:>9.4f} | '\n",
            "          f'{_ev[\"gap_plus_stats\"][\"mean\"]:>7.4f} | '\n",
            "          f'{_rg:>8.4f} | '\n",
            "          f'{_ev[\"ir_loss_stats\"][\"mean\"]:>8.4f} | '\n",
            "          f'{_ev[\"budget_viol_stats\"][\"mean\"]:>6.4f} | '\n",
            "          f'{_ev[\"ir_viol_stats\"][\"mean\"]:>6.4f}')\n",
            "\n",
            "# Learning-curve comparison\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "for _v, _col in zip(['A','B','C'], ['tab:blue','tab:orange','tab:green']):\n",
            "    if _v not in variant_results: continue\n",
            "    _l = variant_results[_v]['logs']\n",
            "    axes[0].plot(_l['reg_plus'], color=_col, label=_v, alpha=0.85)\n",
            "    axes[1].plot(_l['gap_plus'], color=_col, label=_v, alpha=0.85)\n",
            "    axes[2].plot(_l['ir'],       color=_col, label=_v, alpha=0.85)\n",
            "for _ax, _t in zip(axes, ['Regret+','Profit Gap+','IR Loss']):\n",
            "    _ax.set_title(_t); _ax.set_xlabel('Epoch'); _ax.legend()\n",
            f"plt.suptitle(f'{exp_label} — A vs B vs C (optimal hparams)', y=1.02)\n",
            "plt.tight_layout(); plt.show()\n",
        ]),

        # ================================================================
        # Save
        # ================================================================
        md("## Save All Outputs"),

        code([
            f"final_save_dir = {final_save_dir_expr}\n",
            "os.makedirs(final_save_dir, exist_ok=True)\n",
            "\n",
            "variant_summary = {}\n",
            "for _v, _vr in variant_results.items():\n",
            "    with open(f'{final_save_dir}/mech_params_{_v}.pkl', 'wb') as _f:\n",
            "        pickle.dump(_vr['mech_params'], _f)\n",
            "    with open(f'{final_save_dir}/pol_params_{_v}.pkl', 'wb') as _f:\n",
            "        pickle.dump(_vr['pol_params'], _f)\n",
            "    with open(f'{final_save_dir}/logs_{_v}.pkl', 'wb') as _f:\n",
            "        pickle.dump(_vr['logs'], _f)\n",
            "    _ev = _vr['eval']\n",
            "    variant_summary[_v] = {\n",
            "        'revenue':        _ev['profit_stats'],\n",
            "        'gap_plus':       _ev['gap_plus_stats'],\n",
            "        'ir_loss':        _ev['ir_loss_stats'],\n",
            "        'budget_viol':    _ev['budget_viol_stats'],\n",
            "        'ir_viol':        _ev['ir_viol_stats'],\n",
            "        'regret':         _ev.get('regret_stats', {}),\n",
            "        'optimal_config': best_cfg_hp,\n",
            "    }\n",
            "with open(f'{final_save_dir}/variant_comparison.json', 'w') as _f:\n",
            "    json.dump(variant_summary, _f, indent=2)\n",
            "with open(f'{final_save_dir}/optimal_hparams.json', 'w') as _f:\n",
            "    json.dump({'optimal_config': best_cfg_hp,\n",
            f"               'exp': '{exp_label}', 'variant': VARIANT}}, _f, indent=2)\n",
            "\n",
            "print(f'Saved to: {final_save_dir}/')\n",
            "print(f'  mech/pol/logs_{{A,B,C}}.pkl — model artefacts per variant')\n",
            "print(f'  variant_comparison.json   — cross-variant eval summary')\n",
            "print(f'  optimal_hparams.json      — best hyperparameter config')\n",
            "print(f'HP search logs: {hp_save_dir}/')\n",
            "print(f'  s1_lr.json / s2_weights.json / s3_inner.json — per-stage results')\n",
            "print(f'  best_hparams.json — consolidated search summary')\n",
        ]),
    ]


# ===========================================================================
# Experiment 1: Public-state budget auction
# ===========================================================================

_EXP1_SETUP = [
    md("# Budget Auction — Experiment 1: Public-State Budget Auction\n\n"
       "Compares mechanism input variants **A** (GBAM/public-balance), **B** (richer history), "
       "**C** (full history) in a Deng-2021-style setting where the GBAM state should be "
       "theoretically sufficient.\n\n"
       "Run in Google Colab with GPU runtime for best performance."),

    code(["import os\n",
          "# 1. Force remove existing JAX/PJRT installations to ensure a clean state\n",
          "!pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt\n",
          "\n",
          "# 2. Install the latest JAX with full CUDA 12 support and updated libraries\n",
          "!pip install -U \"jax[cuda12]\" dm-haiku optax flax\n",
          "\n",
          "print('\\n--- UPGRADE FINISHED ---')\n",
          "print('IMPORTANT: Please go to Runtime -> Restart session now to activate the GPU drivers.')\n"]),

    code(["# Mount Google Drive after restarting the session\n",
          "try:\n",
          "    from google.colab import drive\n",
          "    drive.mount('/content/drive')\n",
          "    %cd /content/drive/MyDrive/My\\ Research/Dynamic\\ Mechanism\\ Design/Budget\\ Auction\n",
          "except ImportError:\n",
          "    pass  # running locally\n"]),

    code(["from typing import Dict\n",
          "import jax\n",
          "import jax.numpy as jnp\n",
          "import haiku as hk\n",
          "import optax\n",
          "import matplotlib.pyplot as plt\n",
          "import pickle, json, os\n",
          "\n",
          "from budget_auction.core import (\n",
          "    BudgetSimulator, BoundedContPolicy,\n",
          "    build_bounded_policy_net, plot_loss_dict,\n",
          ")\n",
          "from budget_auction.mechanisms.budget_mechanism import BudgetMechanism\n",
          "from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK\n",
          "from budget_auction.envs.public_budget_env import create_public_budget_auction\n",
          "from budget_auction.baselines.static_myopic import run_static_myopic_baseline\n",
          "from budget_auction.baselines.reserve_budget import run_reserve_budget_baseline\n",
          "from budget_auction.train.run_acceptance import nested_GD_budget\n",
          "from budget_auction.train.evaluate import eval_budget_mechanism_multikey, print_eval_report\n",
          "\n",
          "print('JAX devices:', jax.devices())\n"]),

    md("## Configuration"),

    code(["VARIANT = 'A'   # 'A' | 'B' | 'C'\n",
          "SEED    = 42\n",
          "\n",
          "cfg = dict(\n",
          "    n_buyers=2, n_goods=1, horizon=5, discount=0.9,\n",
          "    initial_budget=3.0,\n",
          "    profit_goal=1.5,      # default; Stage 2 search may revise this\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=300,\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=64,\n",
          "    alpha_weight=1e5, beta_weight=1e3,\n",
          "    print_iter=100,\n",
          "    policy_hidden=128, residual_init_std=1.0,\n",
          ")\n",
          "print('Config:', cfg)\n"]),

    md("## Build Auction Environment"),

    code(["auction = create_public_budget_auction(\n",
          "    n_buyers=cfg['n_buyers'], n_goods=cfg['n_goods'],\n",
          "    horizon=cfg['horizon'], discount=cfg['discount'],\n",
          "    initial_budget=cfg['initial_budget'], seed=SEED,\n",
          ")\n",
          "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
          "print(f'Initial budgets: {auction.initial_budgets}')\n"]),

    code(["key = jax.random.PRNGKey(0)\n",
          "s0 = auction.get_initial_state(key)\n",
          "print(f'alpha={s0.alpha}  beta={s0.beta}  types={s0.current_type}')\n"]),

    md("## Evaluate Baselines (No Training)"),

    code(["print('--- Baseline B1: Static Myopic ---')\n",
          "b1 = run_static_myopic_baseline(auction, num_samples=256, num_keys=16, base_seed=1)\n",
          "for k, v in b1.items():\n",
          "    if isinstance(v, float): print(f'  {k}: {v:.4f}')\n",
          "print('--- Baseline B2: Reserve Budget ---')\n",
          "b2 = run_reserve_budget_baseline(auction, reserve_multiplier=0.5,\n",
          "                                  num_samples=256, num_keys=16, base_seed=1)\n",
          "for k, v in b2.items():\n",
          "    if isinstance(v, float): print(f'  {k}: {v:.4f}')\n",
          "print('\\n(Use baseline revenue to calibrate profit_goal search range above.)')\n"]),

    md("## Quick Single-Variant Training\n\n"
       "Sanity check with default hyperparameters. "
       "Skip to **Hyperparameter Search** for the full pipeline."),

    code(["_mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
          "_pnet = build_bounded_policy_net(\n",
          "    d_priv=auction.d_priv, d_pub=auction.d_pub, G=auction.num_goods,\n",
          "    hidden=cfg['policy_hidden'], residual_init_std=cfg['residual_init_std'],\n",
          ")\n",
          "logs = nested_GD_budget(\n",
          "    auction=auction, mechanism_network=_mnet, policy_network=_pnet,\n",
          "    profit_goal=cfg['profit_goal'], num_outer_epochs=cfg['num_outer_epochs'],\n",
          "    lr_theta=cfg['lr_theta'], num_inner_steps=cfg['num_inner_steps'],\n",
          "    lr_w=cfg['lr_w'], num_samples_train=cfg['num_samples_train'],\n",
          "    alpha_weight=cfg['alpha_weight'], beta_weight=cfg['beta_weight'],\n",
          "    print_iter=cfg['print_iter'], seed=SEED,\n",
          ")\n",
          "mechanism_params = logs.pop('mechanism_params')\n",
          "policy_params    = logs.pop('policy_params')\n",
          "print('Quick training complete.')\n"]),

    code(["fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
          "axes[0].plot(logs['reg_plus']); axes[0].set_title('Regret+ ' + VARIANT)\n",
          "axes[1].plot(logs['gap_plus']); axes[1].set_title('Profit Gap+')\n",
          "axes[2].plot(logs['ir']);        axes[2].set_title('IR Loss')\n",
          "for _ax in axes: _ax.set_xlabel('Epoch')\n",
          "plt.suptitle(f'Exp 1 — Quick check, Variant {VARIANT}', y=1.02)\n",
          "plt.tight_layout(); plt.show()\n"]),
]

EXP1_CELLS = _EXP1_SETUP + hp_search_cells(
    exp_label="Exp 1",
    hp_save_dir_expr="'outputs/public_budget/hparam_search'",
    final_save_dir_expr="f'outputs/public_budget/seed{SEED}/optimal'",
    profit_goal_values=[0.8, 1.2, 1.5, 2.0],
)


# ===========================================================================
# Experiment 2: Multi-good privately correlated budget auction
# ===========================================================================

_EXP2_SETUP = [
    md("# Budget Auction — Experiment 2: Multi-Good Privately Correlated Budget Auction\n\n"
       "Compares variants **A** / **B** / **C** in a setting *beyond* Deng 2021: "
       "types are multi-dimensional and transitions are privately correlated. "
       "The key hypothesis: richer history compression should outperform GBAM-style "
       "compressed state here."),

    code(["import os\n",
          "# 1. Force remove existing JAX/PJRT installations to ensure a clean state\n",
          "!pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt\n",
          "\n",
          "# 2. Install the latest JAX with full CUDA 12 support and updated libraries\n",
          "!pip install -U \"jax[cuda12]\" dm-haiku optax flax\n",
          "\n",
          "print('\\n--- UPGRADE FINISHED ---')\n",
          "print('IMPORTANT: Please go to Runtime -> Restart session now to activate the GPU drivers.')\n"]),

    code(["# Mount Google Drive after restarting the session\n",
          "try:\n",
          "    from google.colab import drive\n",
          "    drive.mount('/content/drive')\n",
          "    %cd /content/drive/MyDrive/My\\ Research/Dynamic\\ Mechanism\\ Design/Budget\\ Auction\n",
          "except ImportError:\n",
          "    pass\n"]),

    code(["from typing import Dict\n",
          "import jax\n",
          "import jax.numpy as jnp\n",
          "import haiku as hk\n",
          "import optax\n",
          "import matplotlib.pyplot as plt\n",
          "import pickle, json, os\n",
          "\n",
          "from budget_auction.core import (\n",
          "    BudgetSimulator, BoundedContPolicy,\n",
          "    build_bounded_policy_net, plot_loss_dict,\n",
          ")\n",
          "from budget_auction.mechanisms.budget_mechanism import BudgetMechanism\n",
          "from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK\n",
          "from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction\n",
          "from budget_auction.baselines.static_myopic import run_static_myopic_baseline\n",
          "from budget_auction.train.run_acceptance import nested_GD_budget\n",
          "from budget_auction.train.evaluate import eval_budget_mechanism_multikey, print_eval_report\n",
          "\n",
          "print('JAX devices:', jax.devices())\n"]),

    md("## Configuration"),

    code(["VARIANT = 'A'   # 'A' | 'B' | 'C'\n",
          "SEED    = 42\n",
          "\n",
          "cfg = dict(\n",
          "    n_buyers=2, n_goods=2, horizon=5, discount=0.9,\n",
          "    initial_budget=3.0,\n",
          "    profit_goal=2.0,      # default; Stage 2 search may revise this\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=300,\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=64,\n",
          "    alpha_weight=1e5, beta_weight=1e3,\n",
          "    print_iter=100,\n",
          "    policy_hidden=128, residual_init_std=1.0,\n",
          ")\n",
          "print('Config:', cfg)\n"]),

    md("## Build Auction Environment"),

    code(["auction = create_private_budget_multigood_auction(\n",
          "    n_buyers=cfg['n_buyers'], n_goods=cfg['n_goods'],\n",
          "    horizon=cfg['horizon'], discount=cfg['discount'],\n",
          "    initial_budget=cfg['initial_budget'], seed=SEED,\n",
          ")\n",
          "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
          "print(f'Initial budgets: {auction.initial_budgets}')\n",
          "print(f'Type space: [{auction.types_low}, {auction.types_high}]')\n"]),

    code(["key = jax.random.PRNGKey(0)\n",
          "s0 = auction.get_initial_state(key)\n",
          "print(f'types={s0.current_type}  alpha={s0.alpha}  beta={s0.beta}')\n"]),

    md("## Check Type Transition Dynamics"),

    code(["from budget_auction.envs.private_budget_multigood_env import private_budget_transition_fn\n",
          "alloc_hist = jnp.zeros((auction.num_horizons, auction.num_buyers, auction.num_goods))\n",
          "keys_t = jax.random.split(jax.random.PRNGKey(1), 20)\n",
          "samples = jnp.stack([auction.transition_function(auction, 0, s0.current_type, alloc_hist, k)\n",
          "                     for k in keys_t])\n",
          "print(f'Current types: {s0.current_type}')\n",
          "print(f'Next types mean: {jnp.mean(samples, 0)}')\n",
          "print(f'Next types std:  {jnp.std(samples, 0)}')\n",
          "print('(Should be near current types; sigma_no=0.01 → small drift without allocation)')\n"]),

    md("## Evaluate Myopic Baseline"),

    code(["print('--- Baseline B1: Static Myopic ---')\n",
          "b1 = run_static_myopic_baseline(auction, num_samples=256, num_keys=16, base_seed=1)\n",
          "for k, v in b1.items():\n",
          "    if isinstance(v, float): print(f'  {k}: {v:.4f}')\n",
          "print('\\n(Use baseline revenue to calibrate profit_goal search range above.)')\n"]),

    md("## Quick Single-Variant Training\n\n"
       "Sanity check with default hyperparameters. "
       "Skip to **Hyperparameter Search** for the full pipeline."),

    code(["_mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
          "_pnet = build_bounded_policy_net(\n",
          "    d_priv=auction.d_priv, d_pub=auction.d_pub, G=auction.num_goods,\n",
          "    hidden=cfg['policy_hidden'], residual_init_std=cfg['residual_init_std'],\n",
          ")\n",
          "logs = nested_GD_budget(\n",
          "    auction=auction, mechanism_network=_mnet, policy_network=_pnet,\n",
          "    profit_goal=cfg['profit_goal'], num_outer_epochs=cfg['num_outer_epochs'],\n",
          "    lr_theta=cfg['lr_theta'], num_inner_steps=cfg['num_inner_steps'],\n",
          "    lr_w=cfg['lr_w'], num_samples_train=cfg['num_samples_train'],\n",
          "    alpha_weight=cfg['alpha_weight'], beta_weight=cfg['beta_weight'],\n",
          "    print_iter=cfg['print_iter'], seed=SEED,\n",
          ")\n",
          "mechanism_params = logs.pop('mechanism_params')\n",
          "policy_params    = logs.pop('policy_params')\n",
          "print('Quick training complete.')\n"]),

    code(["fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
          "axes[0].plot(logs['reg_plus']); axes[0].set_title('Regret+ ' + VARIANT)\n",
          "axes[1].plot(logs['gap_plus']); axes[1].set_title('Profit Gap+')\n",
          "axes[2].plot(logs['ir']);        axes[2].set_title('IR Loss')\n",
          "for _ax in axes: _ax.set_xlabel('Epoch')\n",
          "plt.suptitle(f'Exp 2 — Quick check, Variant {VARIANT} ({auction.num_goods} goods)', y=1.02)\n",
          "plt.tight_layout(); plt.show()\n"]),
]

EXP2_CELLS = _EXP2_SETUP + hp_search_cells(
    exp_label="Exp 2",
    hp_save_dir_expr="'outputs/private_budget_multigood/hparam_search'",
    final_save_dir_expr="f'outputs/private_budget_multigood/seed{SEED}/optimal'",
    profit_goal_values=[1.5, 2.0, 2.5, 3.0],
)


# Write notebooks
nb1_path = os.path.join(NB_DIR, "budget_public_colab.ipynb")
with open(nb1_path, "w") as f:
    json.dump(notebook(EXP1_CELLS), f, indent=1)
print(f"Written: {nb1_path}")

nb2_path = os.path.join(NB_DIR, "budget_private_multigood_colab.ipynb")
with open(nb2_path, "w") as f:
    json.dump(notebook(EXP2_CELLS), f, indent=1)
print(f"Written: {nb2_path}")
