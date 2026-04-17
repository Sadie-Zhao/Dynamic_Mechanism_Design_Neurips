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


def env_setup_cells():
    """Two cells that replace the old hard-coded !pip-install + drive-mount pair.

    Cell 1 — user sets RUN_ENV ('colab' | 'local') and LOCAL_PROJECT_DIR.
    Cell 2 — dispatches: installs packages + mounts Drive for Colab;
              optionally changes directory for local runs.
    """
    return [
        code([
            "# ── Environment Selection ────────────────────────────────────────────────────\n",
            "# 'colab'  → installs packages (JAX CUDA 12) and mounts Google Drive\n",
            "# 'local'  → skips install; optionally changes working directory\n",
            "# ─────────────────────────────────────────────────────────────────────────────\n",
            "RUN_ENV = 'colab'   # ← change to 'local' for a local GPU run\n",
            "\n",
            "# For local runs: absolute path to the Budget Auction project root.\n",
            "# Set to None if you launch Jupyter from that directory.\n",
            "LOCAL_PROJECT_DIR = None\n",
            "# Windows example:\n",
            "#   LOCAL_PROJECT_DIR = r'E:\\Research\\code\\Dynamic Mechanism Design\\Budget Auction'\n",
            "# Linux / Mac example:\n",
            "#   LOCAL_PROJECT_DIR = '/home/user/research/budget_auction'\n",
        ]),
        code([
            "import os, sys, subprocess\n",
            "\n",
            "if RUN_ENV == 'colab':\n",
            "    print('Installing JAX CUDA 12 + dependencies ...')\n",
            "    subprocess.check_call(\n",
            "        [sys.executable, '-m', 'pip', 'uninstall', '-y',\n",
            "         'jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt'],\n",
            "        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
            "    subprocess.check_call(\n",
            "        [sys.executable, '-m', 'pip', 'install', '-qU',\n",
            "         'jax[cuda12]', 'dm-haiku', 'optax', 'flax'])\n",
            "    print('Install finished.')\n",
            "    print('IMPORTANT: Runtime → Restart session, then re-run from this cell.')\n",
            "    try:\n",
            "        from google.colab import drive\n",
            "        drive.mount('/content/drive')\n",
            "        os.chdir('/content/drive/MyDrive/My Research/Dynamic Mechanism Design/Budget Auction')\n",
            "        print(f'Working directory: {os.getcwd()}')\n",
            "    except ImportError:\n",
            "        print('(Drive mount skipped — restart session first if on Colab.)')\n",
            "elif RUN_ENV == 'local':\n",
            "    if LOCAL_PROJECT_DIR is not None:\n",
            "        os.chdir(LOCAL_PROJECT_DIR)\n",
            "    print(f'Working directory: {os.getcwd()}')\n",
            "    print('Local run — package install skipped.')\n",
            "    print('One-time install:  pip install \"jax[cuda12]\" dm-haiku optax flax')\n",
            "    print('GPU check:         python -c \"import jax; print(jax.devices())\"')\n",
            "else:\n",
            "    raise ValueError(f'Unknown RUN_ENV: {RUN_ENV!r}  — set to \"colab\" or \"local\"')\n",
        ]),
    ]


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
       "Set `RUN_ENV = 'colab'` for Google Colab GPU or `'local'` for a local GPU machine."),

    *env_setup_cells(),

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
          "    profit_goal=1.5,         # default; Stage 2 search may revise this\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=600,     # 600 steps for budget IC (multi-stage oracle)\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=128,   # 128 samples to reduce regret-estimate variance\n",
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

    md("## Test: Oracle Inner Loop (Maximize Cumulative Regret)\n\n"
       "Run the inner optimization loop against a **fixed** mechanism to verify that "
       "the oracle converges and finds deviations. This is a sanity check before "
       "full training.\n\n"
       "- `TEST_MECHANISM_PARAMS = None` uses a **randomly initialized** mechanism.\n"
       "- Set it to trained params (e.g. `mechanism_params` from a previous run) "
       "to measure how much regret a trained mechanism actually allows.\n\n"
       "The regret curve should climb from near-0 and plateau. "
       "If it stays flat at 0 throughout, the oracle is not finding deviations — "
       "increase `ORACLE_TEST_STEPS` or check the policy initialization."),

    code(["# ---- Oracle test configuration ----\n",
          "ORACLE_TEST_STEPS    = 600    # inner gradient-ascent steps\n",
          "ORACLE_TEST_RESTARTS = 3      # independent random restarts\n",
          "ORACLE_TEST_LR       = 3e-3   # Adam lr for the oracle\n",
          "ORACLE_TEST_SAMPLES  = 128    # MC samples per regret estimate\n",
          "TEST_MECHANISM_PARAMS = None  # None → random init; or set to trained params\n",
          "\n",
          "# ---- Build test networks ----\n",
          "_oracle_mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
          "_oracle_pnet = build_bounded_policy_net(\n",
          "    d_priv=auction.d_priv, d_pub=auction.d_pub, G=auction.num_goods,\n",
          "    hidden=cfg['policy_hidden'], residual_init_std=1.0,\n",
          ")\n",
          "\n",
          "# ---- Initialize mechanism params ----\n",
          "_oracle_prng = hk.PRNGSequence(77)\n",
          "_oracle_s0   = auction.get_initial_state(next(_oracle_prng))\n",
          "if TEST_MECHANISM_PARAMS is None:\n",
          "    _test_mparams = _oracle_mnet.init(\n",
          "        next(_oracle_prng),\n",
          "        _oracle_s0,\n",
          "        jnp.zeros((auction.num_buyers, auction.num_goods)),\n",
          "        auction,\n",
          "    )\n",
          "    print('Using randomly initialized mechanism params.')\n",
          "else:\n",
          "    _test_mparams = TEST_MECHANISM_PARAMS\n",
          "    print('Using provided mechanism params.')\n",
          "_test_mechanism = BudgetMechanism(_oracle_mnet, _test_mparams)\n",
          "\n",
          "# ---- Inner-loop step (JIT-compiled once per session) ----\n",
          "_oracle_opt = optax.adam(ORACLE_TEST_LR)\n",
          "\n",
          "@jax.jit\n",
          "def _oracle_step(policy_params, w_opt_state, key):\n",
          "    _reg, _grad = jax.value_and_grad(\n",
          "        lambda pp: BudgetSimulator.estimate_cumulative_regret(\n",
          "            auction, _test_mechanism,\n",
          "            BoundedContPolicy(_oracle_pnet, pp),\n",
          "            ORACLE_TEST_SAMPLES, key,\n",
          "        )\n",
          "    )(policy_params)\n",
          "    _grad_asc = jax.tree_util.tree_map(lambda g: -g, _grad)\n",
          "    _upd, w_opt_state = _oracle_opt.update(_grad_asc, w_opt_state, policy_params)\n",
          "    policy_params = optax.apply_updates(policy_params, _upd)\n",
          "    return float(_reg), policy_params, w_opt_state\n",
          "\n",
          "# ---- Run oracle for each restart, track best regret ----\n",
          "_oracle_curves   = []\n",
          "_oracle_best_reg = -float('inf')\n",
          "\n",
          "for _restart in range(ORACLE_TEST_RESTARTS):\n",
          "    _b0  = auction.get_initial_belief_state(_oracle_s0)\n",
          "    _pp  = _oracle_pnet.init(next(_oracle_prng), _b0, auction)\n",
          "    _ws  = _oracle_opt.init(_pp)\n",
          "    _curve = []\n",
          "    print(f'Restart {_restart+1}/{ORACLE_TEST_RESTARTS}')\n",
          "    for _step in range(ORACLE_TEST_STEPS):\n",
          "        _key = jax.random.PRNGKey(_restart * 100000 + _step)\n",
          "        _reg, _pp, _ws = _oracle_step(_pp, _ws, _key)\n",
          "        _curve.append(_reg)\n",
          "        if _step % 100 == 0:\n",
          "            print(f'  step {_step:4d}: regret = {_reg:.6f}')\n",
          "    _oracle_curves.append(_curve)\n",
          "    _oracle_best_reg = max(_oracle_best_reg, max(_curve))\n",
          "    print(f'  → restart best regret: {max(_curve):.6f}')\n",
          "\n",
          "# ---- Plot ----\n",
          "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n",
          "for _r, _curve in enumerate(_oracle_curves):\n",
          "    axes[0].plot(_curve, alpha=0.8, label=f'restart {_r+1}')\n",
          "axes[0].axhline(0, color='red', linestyle='--', lw=1, label='0')\n",
          "axes[0].set_xlabel('Inner Step'); axes[0].set_ylabel('Regret')\n",
          "axes[0].set_title('Oracle Convergence (all restarts)'); axes[0].legend()\n",
          "_best_curve = _oracle_curves[int(max(range(len(_oracle_curves)),\n",
          "                                       key=lambda r: max(_oracle_curves[r])))]\n",
          "axes[1].plot(_best_curve, color='tab:red')\n",
          "axes[1].axhline(0, color='k', linestyle='--', lw=1)\n",
          "axes[1].set_xlabel('Inner Step'); axes[1].set_ylabel('Regret')\n",
          "axes[1].set_title('Best Restart')\n",
          f"plt.suptitle(f'Variant {{VARIANT}} — Oracle Inner Loop Test (fixed mechanism)', y=1.02)\n",
          "plt.tight_layout(); plt.show()\n",
          "print(f'\\nBest regret across all restarts: {_oracle_best_reg:.6f}')\n",
          "print('(If this stays near 0, the oracle cannot find deviations — '\n",
          "      'mechanism may already be approximately IC, or oracle is too weak)')\n"]),

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
       "compressed state here.\n\n"
       "Set `RUN_ENV = 'colab'` for Google Colab GPU or `'local'` for a local GPU machine."),

    *env_setup_cells(),

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
          "    profit_goal=2.0,         # default; Stage 2 search may revise this\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=600,     # 600 steps for budget IC (multi-stage oracle)\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=128,   # 128 samples to reduce regret-estimate variance\n",
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


# ===========================================================================
# Binary Search Notebook
# ===========================================================================

def binary_search_cells(exp_label, exp_env_import, create_auction_call,
                        auction_info_lines, hp_save_dir_expr, bs_save_dir_expr,
                        default_pg_lo, default_pg_hi, default_pg_base,
                        default_profit_goal_init):
    """
    Generate cells for the binary search notebook.

    The binary search protocol:
      1. Load hyperparameters from HP search (or use defaults seeded from HP search)
      2. For each bisection step:
         - pg_mid = (pg_lo + pg_hi) / 2
         - Train with full epochs, recording all checkpoints where train IC+IR ≤ thresh
         - Evaluate each checkpoint on held-out keys
         - If any checkpoint passes IC ≤ 0.05 AND IR ≤ 0.05: pg_lo = pg_mid (feasible)
         - Else: pg_hi = pg_mid (infeasible)
      3. Stop when pg_hi - pg_lo < pg_tol
      4. Report the highest confirmed feasible profit_goal and save best mechanism
    """
    return [

        # ================================================================
        # Header
        # ================================================================
        md(f"# Budget Auction — Binary Search for Optimal Profit Goal ({exp_label})\n\n"
           "This notebook implements a binary search over `profit_goal` to find the highest "
           "revenue target that yields a mechanism satisfying **IC regret ≤ 0.05** and "
           "**IR loss ≤ 0.05** on held-out evaluation keys.\n\n"
           "**Key idea:** at each bisection step, we train with full epochs and record ALL "
           "mechanism checkpoints during training (not just the final one). We then evaluate "
           "every checkpoint on held-out keys. If *any* checkpoint passes both thresholds, "
           "the profit_goal is declared feasible.\n\n"
           "This avoids discarding good intermediate solutions that the optimizer may "
           'subsequently "over-train away".\n\n'
           "**Run one variant (A, B, or C) per session.** Set `VARIANT` below, "
           "then run all cells top-to-bottom."),

        # ================================================================
        # Install
        # ================================================================
        code(["import os\n",
              "!pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt\n",
              "!pip install -U \"jax[cuda12]\" dm-haiku optax flax\n",
              "print('\\n--- UPGRADE FINISHED ---')\n",
              "print('IMPORTANT: Please go to Runtime -> Restart session now.')\n"]),

        # ================================================================
        # Mount Drive
        # ================================================================
        code(["try:\n",
              "    from google.colab import drive\n",
              "    drive.mount('/content/drive')\n",
              "    %cd /content/drive/MyDrive/My\\ Research/Dynamic\\ Mechanism\\ Design/Budget\\ Auction\n",
              "except ImportError:\n",
              "    pass\n"]),

        # ================================================================
        # Imports
        # ================================================================
        code(["import jax\n",
              "import jax.numpy as jnp\n",
              "import haiku as hk\n",
              "import optax\n",
              "import matplotlib.pyplot as plt\n",
              "import pickle, json, os, time\n",
              "\n",
              "from budget_auction.core import (\n",
              "    BudgetSimulator, BoundedContPolicy,\n",
              "    build_bounded_policy_net, plot_loss_dict,\n",
              ")\n",
              "from budget_auction.mechanisms.budget_mechanism import BudgetMechanism\n",
              "from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK\n",
              f"{exp_env_import}\n",
              "from budget_auction.train.run_acceptance import nested_GD_budget\n",
              "from budget_auction.train.evaluate import eval_budget_mechanism_multikey, print_eval_report\n",
              "from budget_auction.train.binary_search import (\n",
              "    DEFAULT_BS_CFG, train_with_checkpoints,\n",
              "    evaluate_checkpoint_list, run_binary_search,\n",
              ")\n",
              "\n",
              "print('JAX devices:', jax.devices())\n"]),

        # ================================================================
        # Variant selector + config
        # ================================================================
        md("## Configuration\n\n"
           "Set `VARIANT` to the mechanism input variant you want to search over. "
           "Run one variant per session (each takes ~6-10 hours for a full binary search).\n\n"
           "The base hyperparameters below are seeded from the HP search results "
           "(Stage 1 winner: `lr_theta=3e-4`, `lr_w=1e-3`; Stage 2: moderate weights). "
           "Override `hp_json_path` to load your own HP search results."),

        code(["VARIANT  = 'A'   # 'A' | 'B' | 'C'  ← change this per run\n",
              "SEED     = 42\n",
              "TIME_BUDGET_HOURS = 11.5   # total wall-clock budget\n",
              "\n",
              "# ---- Set FORCE_RESTART=True to ignore existing results and start fresh ----\n",
              "# Use a different bs_save_dir below when changing configs, so old results\n",
              "# are not overwritten.  FORCE_RESTART=True with the same dir discards them.\n",
              "FORCE_RESTART = False\n",
              "\n",
              "# ---- Load HP search best config (optional) ----\n",
              f"hp_json_path = os.path.join({hp_save_dir_expr}, 'best_hparams.json')\n",
              "if os.path.exists(hp_json_path):\n",
              "    with open(hp_json_path) as _f:\n",
              "        _hp = json.load(_f)\n",
              "    _opt = _hp.get('optimal_config', {})\n",
              "    print(f'Loaded HP search config from: {hp_json_path}')\n",
              "    print('  Optimal config:', _opt)\n",
              "else:\n",
              "    _opt = {}\n",
              "    print(f'HP search file not found at {hp_json_path} — using defaults.')\n",
              "\n",
              "# ---- Binary search config ----\n",
              "# Seeded from HP search; override any key as needed.\n",
              "bs_cfg = {\n",
              "    **DEFAULT_BS_CFG,\n",
              "    # Learning rates (from HP search Stage 1)\n",
              "    'lr_theta':        _opt.get('lr_theta',        3e-4),\n",
              "    'lr_w':            _opt.get('lr_w',            1e-3),\n",
              "    # Loss weights at pg_base (from HP search Stage 2)\n",
              "    'alpha_base':      _opt.get('alpha_weight',    1e5),\n",
              "    'beta_weight':     _opt.get('beta_weight',     500.0),\n",
              "    # Inner oracle steps (from HP search Stage 3)\n",
              "    'num_inner_steps': _opt.get('num_inner_steps', 300),\n",
              "    # Full training epochs per binary-search trial\n",
              "    'num_outer_epochs': 2000,\n",
              "    # Checkpoint recording\n",
              "    'checkpoint_thresh':    0.15,   # record when train IC+IR both ≤ this\n",
              "    'checkpoint_interval':  50,     # check every N outer epochs\n",
              "    # Binary search range\n",
              f"    'pg_lo':   {default_pg_lo},\n",
              f"    'pg_hi':   {default_pg_hi},\n",
              f"    'pg_base': {default_pg_base},   # reference point for alpha scaling\n",
              "    'pg_tol':  0.1,\n",
              "    # Feasibility thresholds (eval loss)\n",
              "    'ic_thresh': 0.05,\n",
              "    'ir_thresh': 0.05,\n",
              "    # IC re-optimisation at eval time.\n",
              "    # The oracle is re-solved from scratch against each checkpoint mechanism\n",
              "    # to get an honest (upper-bound) regret rather than reusing the\n",
              "    # underconverged training oracle.\n",
              "    'ic_reopt_steps':    1000,   # inner steps per restart\n",
              "    'ic_reopt_restarts': 3,      # independent random restarts (take max)\n",
              "    'ic_reopt_lr':       3e-3,   # Adam lr for reopt (independent of training lr_w)\n",
              "    'ic_reopt_samples':  128,    # MC samples during reopt ascent\n",
              "    # Evaluation (revenue, IR, budget-viol, final IC)\n",
              "    'eval_num_keys':    32,\n",
              "    'eval_num_samples': 256,\n",
              "}\n",
              "\n",
              "print(f'Binary search config for Variant {VARIANT}  (FORCE_RESTART={FORCE_RESTART}):')\n",
              "for _k, _v in bs_cfg.items():\n",
              "    print(f'  {_k:25s}: {_v}')\n"]),

        # ================================================================
        # Build auction
        # ================================================================
        md("## Build Auction Environment"),

        code([f"auction = {create_auction_call}\n",
              *auction_info_lines]),

        # ================================================================
        # Build networks
        # ================================================================
        md("## Build Networks"),

        code(["mechanism_network = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
              "policy_network = build_bounded_policy_net(\n",
              "    d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
              "    G=auction.num_goods, hidden=128,\n",
              "    residual_init_std=1.0,\n",
              ")\n",
              "print(f'Variant {VARIANT} networks built.')\n",
              "print(f'  d_priv={auction.d_priv}  d_pub={auction.d_pub}  G={auction.num_goods}')\n"]),

        # ================================================================
        # Set save dir
        # ================================================================
        code([f"# To start a brand-new search with a different config, change the RUN_TAG below.\n",
              "# Each unique tag gets its own subdirectory, so old results are never overwritten.\n",
              "# Leave RUN_TAG='' to use the default directory.\n",
              "RUN_TAG = ''   # e.g. 'run2', 'higherLR', 'moreSteps'\n",
              f"_bs_base = {bs_save_dir_expr}\n",
              "bs_save_dir = f'{_bs_base}/{RUN_TAG}' if RUN_TAG else _bs_base\n",
              "os.makedirs(bs_save_dir, exist_ok=True)\n",
              "print(f'Save directory: {bs_save_dir}')\n",
              "print(f'Results will be written to: {bs_save_dir}/bs_results.json (fault-tolerant)')\n",
              "print(f'Best mechanisms: {bs_save_dir}/best_mech_pg<X>.pkl')\n",
              "print(f'(Set FORCE_RESTART=True above to discard existing results in this directory)')\n"]),

        # ================================================================
        # Run binary search
        # ================================================================
        md("## Run Binary Search\n\n"
           "This cell runs the full binary search. It will:\n"
           "1. Bisect the `[pg_lo, pg_hi]` interval until `pg_hi - pg_lo < pg_tol`\n"
           "2. For each midpoint, train a full mechanism and record checkpoints\n"
           "3. Evaluate all checkpoints on 32 held-out keys × 256 samples\n"
           "4. Declare feasibility if any checkpoint achieves IC ≤ 0.05 AND IR ≤ 0.05\n"
           "5. Save results after every trial (safe to interrupt and resume)\n\n"
           "Expected number of bisection steps: `ceil(log2((pg_hi-pg_lo)/pg_tol))` ≈ 4-5 trials.\n"
           "Each trial takes ~30-60 min on GPU. Total wall time: up to 12h."),

        code(["bs_summary = run_binary_search(\n",
              "    auction           = auction,\n",
              "    mechanism_network = mechanism_network,\n",
              "    policy_network    = policy_network,\n",
              "    variant           = VARIANT,\n",
              "    save_dir          = bs_save_dir,\n",
              "    cfg               = bs_cfg,\n",
              "    base_seed         = SEED,\n",
              "    verbose           = True,\n",
              "    time_budget_hours = TIME_BUDGET_HOURS,\n",
              "    force_restart     = FORCE_RESTART,\n",
              ")\n",
              "\n",
              "print('\\n' + '='*60)\n",
              f"print(f'Binary search complete — {exp_label} Variant {{VARIANT}}')\n",
              "print(f'  Highest confirmed feasible profit_goal: {bs_summary[\"pg_lo\"]:.3f}')\n",
              "print(f'  pg_hi (infeasible upper bound): {bs_summary[\"pg_hi\"]:.3f}')\n",
              "print(f'  Converged: {bs_summary[\"converged\"]}')\n",
              "if bs_summary['best_mechanism']:\n",
              "    _bm = bs_summary['best_mechanism']\n",
              "    print(f'  Best mechanism: rev={_bm[\"revenue_eval\"]:.4f}  '\n",
              "          f'IC={_bm[\"regret_eval\"]:.4f}  IR={_bm[\"ir_loss_eval\"]:.4f}')\n",
              "    print(f'  Saved to: {_bm[\"mech_pkl\"]}')\n"]),

        # ================================================================
        # Load results and analyse
        # ================================================================
        md("## Analyse Results\n\n"
           "Load the saved trial records and visualise the binary search progression."),

        code(["_bs_path = os.path.join(bs_save_dir, 'bs_results.json')\n",
              "with open(_bs_path) as _f:\n",
              "    _bs_trials = json.load(_f)\n",
              "_bs_trials_sorted = sorted(_bs_trials, key=lambda t: t['profit_goal'])\n",
              "\n",
              "print(f'Loaded {len(_bs_trials)} trial records from {_bs_path}')\n",
              "print(f'\\n{\"profit_goal\":>12} | {\"feasible\":>8} | {\"best_rev\":>9} | {\"best_IC\":>8} | {\"best_IR\":>8} | {\"ckpts\":>5}')\n",
              "print('-' * 65)\n",
              "for _t in _bs_trials_sorted:\n",
              "    _flag = 'PASS' if _t.get('any_feasible') else 'fail'\n",
              "    _rev  = _t.get('best_revenue_eval') or float('nan')\n",
              "    _ic   = _t.get('best_regret_eval')  or float('nan')\n",
              "    _ir   = _t.get('best_ir_loss_eval') or float('nan')\n",
              "    _nc   = _t.get('num_checkpoints', 0)\n",
              "    print(f'{_t[\"profit_goal\"]:>12.3f} | {_flag:>8} | {_rev:>9.4f} | {_ic:>8.4f} | {_ir:>8.4f} | {_nc:>5}')\n"]),

        code(["# Plot binary search progression (one point per bisection trial)\n",
              "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
              "\n",
              "_pgs = [t['profit_goal'] for t in _bs_trials_sorted]\n",
              "_revs = [t.get('best_revenue_eval') or 0.0 for t in _bs_trials_sorted]\n",
              "_ics  = [t.get('best_regret_eval')  or 1.0 for t in _bs_trials_sorted]\n",
              "_irs  = [t.get('best_ir_loss_eval') or 1.0 for t in _bs_trials_sorted]\n",
              "_cols = ['tab:green' if t.get('any_feasible') else 'tab:red' for t in _bs_trials_sorted]\n",
              "\n",
              "axes[0].scatter(_pgs, _revs, c=_cols, s=80, zorder=3)\n",
              "axes[0].set_xlabel('profit_goal'); axes[0].set_ylabel('Best feasible revenue')\n",
              "axes[0].set_title('Revenue vs profit_goal')\n",
              "axes[0].axhline(0, color='k', lw=0.5)\n",
              "\n",
              "axes[1].scatter(_pgs, _ics, c=_cols, s=80, zorder=3)\n",
              "axes[1].axhline(0.05, color='red', linestyle='--', label='IC thresh')\n",
              "axes[1].set_xlabel('profit_goal'); axes[1].set_ylabel('Best IC regret (eval)')\n",
              "axes[1].set_title('IC Regret vs profit_goal'); axes[1].legend()\n",
              "\n",
              "axes[2].scatter(_pgs, _irs, c=_cols, s=80, zorder=3)\n",
              "axes[2].axhline(0.05, color='red', linestyle='--', label='IR thresh')\n",
              "axes[2].set_xlabel('profit_goal'); axes[2].set_ylabel('Best IR loss (eval)')\n",
              "axes[2].set_title('IR Loss vs profit_goal'); axes[2].legend()\n",
              "\n",
              "from matplotlib.patches import Patch\n",
              "_legend = [Patch(color='tab:green', label='feasible'), Patch(color='tab:red', label='infeasible')]\n",
              "axes[0].legend(handles=_legend)\n",
              f"plt.suptitle(f'{exp_label} — Variant {{VARIANT}} Binary Search', y=1.02)\n",
              "plt.tight_layout(); plt.show()\n"]),

        # ================================================================
        # Convergence plots
        # ================================================================
        md("## Convergence Plots per Trial\n\n"
           "For each binary-search trial, show:\n"
           "- **Full training curves** (IC regret, IR loss, profit gap) vs epoch — solid lines\n"
           "- **Checkpoint markers** — triangles at epochs where training IC+IR ≤ threshold\n"
           "- **Eval values** at checkpoints — circles (filled=feasible, open=infeasible)\n"
           "- **Dashed red lines** at the 0.05 feasibility threshold\n\n"
           "This lets you see whether the mechanism converges, when good checkpoints appear, "
           "and whether training metrics are predictive of eval metrics."),

        code(["# Load training curves for all trials\n",
              "_curves_by_pg = {}\n",
              "for _t in _bs_trials_sorted:\n",
              "    _cp = _t.get('training_curves')\n",
              "    if _cp and os.path.exists(_cp):\n",
              "        with open(_cp) as _f:\n",
              "            _curves_by_pg[_t['profit_goal']] = json.load(_f)\n",
              "    else:\n",
              "        print(f'  [missing] training_curves for pg={_t[\"profit_goal\"]:.3f}')\n",
              "print(f'Loaded training curves for {len(_curves_by_pg)} trials.')\n"]),

        code(["# One figure per trial: 3 panels (IC, IR, profit gap)\n",
              "# IC panel shows three series:\n",
              "#   - solid line  : full training regret curve\n",
              "#   - triangle (▲): training value at each checkpoint\n",
              "#   - circle (●/x) : re-optimised oracle regret (honest eval); green=pass, red=fail\n",
              "_THRESH = 0.05\n",
              "for _pg, _cv in sorted(_curves_by_pg.items()):\n",
              "    _trial = next(t for t in _bs_trials if abs(t['profit_goal'] - _pg) < 1e-6)\n",
              "    _feasible_trial = _trial.get('any_feasible', False)\n",
              "    _title_flag = 'FEASIBLE' if _feasible_trial else 'INFEASIBLE'\n",
              "    _epochs = list(range(len(_cv['reg'])))\n",
              "\n",
              "    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
              "\n",
              "    # ---- Panel data definitions ----\n",
              "    # (label, full_train_curve, ckpt_train_key, ckpt_eval_key, reopt_key, color)\n",
              "    _panels = [\n",
              "        ('IC Regret', _cv['reg'], 'reg_train', 'regret_eval', 'regret_reopt', 'tab:blue'),\n",
              "        ('IR Loss',   _cv['ir'],  'ir_train',  'ir_loss_eval', None,          'tab:orange'),\n",
              "        ('Profit Gap', _cv['gap'], 'gap_train', None,           None,          'tab:purple'),\n",
              "    ]\n",
              "\n",
              "    for _ax, (_label, _train_curve, _ck_train_key, _ck_eval_key, _ck_reopt_key, _col) in zip(axes, _panels):\n",
              "        # Full training curve\n",
              "        _ax.plot(_epochs, _train_curve, color=_col, lw=1.2, alpha=0.75, label='train')\n",
              "\n",
              "        # Feasibility threshold line (IC and IR panels)\n",
              "        if _ck_eval_key is not None:\n",
              "            _ax.axhline(_THRESH, color='red', linestyle='--', lw=1.0,\n",
              "                        alpha=0.8, label=f'thresh={_THRESH}')\n",
              "\n",
              "        # Checkpoint training values (triangles)\n",
              "        _ck_eps   = [c['epoch']       for c in _cv['checkpoints_train']]\n",
              "        _ck_train = [c[_ck_train_key] for c in _cv['checkpoints_train']]\n",
              "        if _ck_eps:\n",
              "            _ax.scatter(_ck_eps, _ck_train, marker='^', s=60,\n",
              "                        color=_col, zorder=4, label='ckpt (train)')\n",
              "\n",
              "        # Checkpoint eval values: re-optimised IC (diamond) or multi-key mean (circle)\n",
              "        if _ck_eval_key is not None:\n",
              "            for _er in _cv['checkpoints_eval']:\n",
              "                _col_pt = 'tab:green' if _er['feasible'] else 'tab:red'\n",
              "                # Re-opt regret (diamond) — this is what determines feasibility\n",
              "                if _ck_reopt_key and _er.get(_ck_reopt_key) is not None:\n",
              "                    _ax.scatter(_er['epoch'], _er[_ck_reopt_key], marker='D', s=70,\n",
              "                                color=_col_pt, zorder=6,\n",
              "                                label='reopt IC (pass)' if _er['feasible'] else 'reopt IC (fail)')\n",
              "                # Multi-key eval mean (circle) — shown for comparison\n",
              "                if _er.get(_ck_eval_key) is not None:\n",
              "                    _ax.scatter(_er['epoch'], _er[_ck_eval_key], marker='o', s=45,\n",
              "                                color=_col_pt, alpha=0.5, zorder=5,\n",
              "                                label='eval mean')\n",
              "        elif _ck_eval_key is None and _ck_train_key == 'ir_train':\n",
              "            # IR: show eval mean circles only (no reopt)\n",
              "            for _er in _cv['checkpoints_eval']:\n",
              "                _col_pt = 'tab:green' if _er['feasible'] else 'tab:red'\n",
              "                if _er.get('ir_loss_eval') is not None:\n",
              "                    _mk = 'o' if _er['feasible'] else 'x'\n",
              "                    _ax.scatter(_er['epoch'], _er['ir_loss_eval'], marker=_mk, s=65,\n",
              "                                color=_col_pt, zorder=5, label='eval')\n",
              "\n",
              "        _ax.set_xlabel('Epoch'); _ax.set_ylabel(_label)\n",
              "        _ax.set_title(_label)\n",
              "        # Deduplicate legend entries\n",
              "        _hs, _ls = _ax.get_legend_handles_labels()\n",
              "        _seen = {}; [_seen.setdefault(_l, _h) for _h, _l in zip(_hs, _ls)]\n",
              "        _ax.legend(list(_seen.values()), list(_seen.keys()), fontsize=8)\n",
              "\n",
              f"    plt.suptitle(f'{exp_label} Variant {{VARIANT}} | pg={{_pg:.3f}} — {{_title_flag}}', y=1.02)\n",
              "    plt.tight_layout(); plt.show()\n"]),

        # ================================================================
        # Load and evaluate best mechanism
        # ================================================================
        md("## Load and Evaluate Best Mechanism"),

        code(["_bm_info = bs_summary.get('best_mechanism') if 'bs_summary' in dir() else None\n",
              "\n",
              "# Alternatively, find the best pkl from trial records\n",
              "if _bm_info is None:\n",
              "    _feas = [t for t in _bs_trials if t.get('any_feasible') and t.get('best_mech_pkl')]\n",
              "    _feas.sort(key=lambda t: t.get('best_revenue_eval') or 0.0, reverse=True)\n",
              "    _bm_info = _feas[0] if _feas else None\n",
              "\n",
              "if _bm_info is None:\n",
              "    print('No feasible mechanism found yet — search may need more iterations.')\n",
              "else:\n",
              "    _pkl_path = _bm_info.get('mech_pkl') or _bm_info.get('best_mech_pkl')\n",
              "    print(f'Loading best mechanism from: {_pkl_path}')\n",
              "    with open(_pkl_path, 'rb') as _f:\n",
              "        _best = pickle.load(_f)\n",
              "\n",
              "    _best_mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[VARIANT]))\n",
              "    _best_pnet = build_bounded_policy_net(\n",
              "        d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
              "        G=auction.num_goods, hidden=128, residual_init_std=1.0,\n",
              "    )\n",
              "\n",
              "    _eval_keys = jax.random.split(jax.random.PRNGKey(55555), 64)\n",
              "    _final_ev = eval_budget_mechanism_multikey(\n",
              "        auction=auction, mechanism_network=_best_mnet,\n",
              "        mechanism_params=_best['mechanism_params'],\n",
              "        profit_goal=_best['profit_goal'],\n",
              "        num_samples=512, keys=_eval_keys,\n",
              "        policy_network=_best_pnet, policy_params=_best['policy_params'],\n",
              "    )\n",
              f"    print_eval_report(_final_ev, prefix=f'{exp_label} Variant {{VARIANT}} Best: ')\n",
              "    print(f'  profit_goal: {_best[\"profit_goal\"]:.3f}')\n",
              "    print(f'  epoch found: {_best[\"epoch\"]}')\n"]),

        # ================================================================
        # Save summary
        # ================================================================
        md("## Save Summary"),

        code([f"_summary_path = os.path.join({bs_save_dir_expr}, f'variant_{{VARIANT}}_summary.json')\n",
              "if _bm_info is not None:\n",
              "    _out = {\n",
              "        'variant':         VARIANT,\n",
              f"        'exp':             '{exp_label}',\n",
              "        'pg_lo':           bs_summary.get('pg_lo'),\n",
              "        'pg_hi':           bs_summary.get('pg_hi'),\n",
              "        'converged':       bs_summary.get('converged'),\n",
              "        'total_time_h':    bs_summary.get('total_time_h'),\n",
              "        'best_mechanism':  {\n",
              "            'profit_goal':    _best['profit_goal'],\n",
              "            'epoch':          _best['epoch'],\n",
              "            'revenue_eval':   float(_final_ev['profit_stats']['mean']),\n",
              "            'revenue_std':    float(_final_ev['profit_stats']['std']),\n",
              "            'regret_eval':    float(_final_ev.get('regret_stats', {}).get('mean', float('nan'))),\n",
              "            'ir_loss_eval':   float(_final_ev['ir_loss_stats']['mean']),\n",
              "            'budget_viol':    float(_final_ev['budget_viol_stats']['mean']),\n",
              "            'ir_viol':        float(_final_ev['ir_viol_stats']['mean']),\n",
              "        },\n",
              "        'bs_cfg': {k: v for k, v in bs_cfg.items()},\n",
              "    }\n",
              "    with open(_summary_path, 'w') as _f:\n",
              "        json.dump(_out, _f, indent=2)\n",
              "    print(f'Summary saved to: {_summary_path}')\n",
              "else:\n",
              "    print('No feasible mechanism — nothing to save.')\n"]),
    ]


# ---- Exp 1 binary search cells ----
_BS_EXP1_CELLS = binary_search_cells(
    exp_label="Exp 1 (Public-State Budget)",
    exp_env_import="from budget_auction.envs.public_budget_env import create_public_budget_auction",
    create_auction_call=(
        "create_public_budget_auction(\n"
        "    n_buyers=2, n_goods=1, horizon=5, discount=0.9,\n"
        "    initial_budget=3.0, seed=SEED,\n"
        ")"
    ),
    auction_info_lines=[
        "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
        "print(f'Initial budgets: {auction.initial_budgets}')\n",
    ],
    hp_save_dir_expr="'outputs/public_budget/hparam_search'",
    bs_save_dir_expr="f'outputs/public_budget/binary_search/variant_{VARIANT}'",
    default_pg_lo=0.8,
    default_pg_hi=2.0,
    default_pg_base=0.8,
    default_profit_goal_init=1.5,
)

# ---- Exp 2 binary search cells ----
_BS_EXP2_CELLS = binary_search_cells(
    exp_label="Exp 2 (Multi-Good Privately Correlated)",
    exp_env_import="from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction",
    create_auction_call=(
        "create_private_budget_multigood_auction(\n"
        "    n_buyers=2, n_goods=2, horizon=5, discount=0.9,\n"
        "    initial_budget=3.0, seed=SEED,\n"
        ")"
    ),
    auction_info_lines=[
        "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
        "print(f'Initial budgets: {auction.initial_budgets}')\n",
        "print(f'Type space: [{auction.types_low}, {auction.types_high}]')\n",
    ],
    hp_save_dir_expr="'outputs/private_budget_multigood/hparam_search'",
    bs_save_dir_expr="f'outputs/private_budget_multigood/binary_search/variant_{VARIANT}'",
    default_pg_lo=1.5,
    default_pg_hi=3.5,
    default_pg_base=1.5,
    default_profit_goal_init=2.0,
)

BS_CELLS = [
    md("# Budget Auction — Binary Search Notebook\n\n"
       "This notebook covers both Experiment 1 (public-state) and Experiment 2 (multi-good). "
       "Pick the experiment section that matches your setup, set `VARIANT`, and run top-to-bottom. "
       "Results are fault-tolerantly saved after every trial.\n\n"
       "**Experiment 1** — Public-state budget auction (2 buyers, 1 good, T=5)\n\n"
       "**Experiment 2** — Multi-good privately correlated budget auction (2 buyers, 2 goods, T=5)"),
] + [
    md("---\n# Experiment 1 Section"),
] + _BS_EXP1_CELLS + [
    md("---\n# Experiment 2 Section\n\n"
       "(Clear GPU memory / restart runtime before switching experiments)"),
] + _BS_EXP2_CELLS


# Write notebooks
nb1_path = os.path.join(NB_DIR, "budget_public_colab.ipynb")
with open(nb1_path, "w") as f:
    json.dump(notebook(EXP1_CELLS), f, indent=1)
print(f"Written: {nb1_path}")

nb2_path = os.path.join(NB_DIR, "budget_private_multigood_colab.ipynb")
with open(nb2_path, "w") as f:
    json.dump(notebook(EXP2_CELLS), f, indent=1)
print(f"Written: {nb2_path}")

nb3_path = os.path.join(NB_DIR, "budget_binary_search_colab.ipynb")
with open(nb3_path, "w") as f:
    json.dump(notebook(BS_CELLS), f, indent=1)
print(f"Written: {nb3_path}")
