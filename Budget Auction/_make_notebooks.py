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


# ==========================================================================
# Experiment 1: Public-state budget auction notebook
# ==========================================================================

EXP1_CELLS = [
    md("# Budget Auction — Experiment 1: Public-State Budget Auction\n\n"
       "Compares mechanism input variants **A** (GBAM/public-balance), **B** (richer history), "
       "**C** (full history) in a Deng-2021-style setting where the GBAM state should be "
       "theoretically sufficient.\n\n"
       "Run in Google Colab with GPU runtime for best performance."),

    code(["%pip install -q git+https://github.com/deepmind/dm-haiku\n",
          "%pip install -q jaxopt optax flax pyyaml\n"]),

    code(["# Mount Google Drive (Colab only)\n",
          "try:\n",
          "    from google.colab import drive\n",
          "    drive.mount('/content/drive')\n",
          "    %cd /content/drive/MyDrive/My\\ Research/Dynamic\\ Mechanism\\ Design/Budget\\ Auction\n",
          "except ImportError:\n",
          "    pass  # Running locally\n"]),

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

    code(["# ---- Select variant and seed ----\n",
          "VARIANT = 'A'   # 'A' | 'B' | 'C'\n",
          "SEED    = 42\n",
          "\n",
          "cfg = dict(\n",
          "    n_buyers=2, n_goods=1, horizon=5, discount=0.9,\n",
          "    initial_budget=3.0,\n",
          "    profit_goal=1.5,\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=300,\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=64,\n",
          "    alpha_weight=1e5, beta_weight=1e3,\n",
          "    print_iter=100,\n",
          "    policy_hidden=128, residual_init_std=1.0,\n",
          ")\n",
          "print('Config:', cfg)\n"]),

    md("## Build auction environment"),

    code(["auction = create_public_budget_auction(\n",
          "    n_buyers=cfg['n_buyers'], n_goods=cfg['n_goods'],\n",
          "    horizon=cfg['horizon'],  discount=cfg['discount'],\n",
          "    initial_budget=cfg['initial_budget'], seed=SEED,\n",
          ")\n",
          "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
          "print(f'Initial budgets: {auction.initial_budgets}')\n"]),

    code(["# Sanity check: sample one initial state\n",
          "key = jax.random.PRNGKey(0)\n",
          "s0 = auction.get_initial_state(key)\n",
          "print('Initial BudgetState:')\n",
          "print(f'  types  = {s0.current_type}')\n",
          "print(f'  alpha  = {s0.alpha}  (should be 0)')\n",
          "print(f'  beta   = {s0.beta}   (should be {auction.initial_budgets})')\n"]),

    md("## Build mechanism network"),

    code(["raw_fn = VARIANT_NETWORK[VARIANT]\n",
          "mechanism_network = hk.without_apply_rng(hk.transform(raw_fn))\n",
          "policy_network    = build_bounded_policy_net(\n",
          "    d_priv=auction.d_priv, d_pub=auction.d_pub, G=auction.num_goods,\n",
          "    hidden=cfg['policy_hidden'], residual_init_std=cfg['residual_init_std'],\n",
          ")\n",
          "print(f'Mechanism variant: {VARIANT}')\n"]),

    md("## Evaluate baselines (no training)"),

    code(["print('--- Baseline B1: Static Myopic ---')\n",
          "b1 = run_static_myopic_baseline(auction, num_samples=128, num_keys=8, base_seed=1)\n",
          "for k, v in b1.items():\n",
          "    if isinstance(v, float):\n",
          "        print(f'  {k}: {v:.4f}')\n",
          "\n",
          "print('--- Baseline B2: Reserve Budget ---')\n",
          "b2 = run_reserve_budget_baseline(auction, reserve_multiplier=0.5,\n",
          "                                  num_samples=128, num_keys=8, base_seed=1)\n",
          "for k, v in b2.items():\n",
          "    if isinstance(v, float):\n",
          "        print(f'  {k}: {v:.4f}')\n"]),

    md("## Train learned mechanism"),

    code(["logs = nested_GD_budget(\n",
          "    auction=auction,\n",
          "    mechanism_network=mechanism_network,\n",
          "    policy_network=policy_network,\n",
          "    profit_goal=cfg['profit_goal'],\n",
          "    num_outer_epochs=cfg['num_outer_epochs'],\n",
          "    lr_theta=cfg['lr_theta'],\n",
          "    num_inner_steps=cfg['num_inner_steps'],\n",
          "    lr_w=cfg['lr_w'],\n",
          "    num_samples_train=cfg['num_samples_train'],\n",
          "    alpha_weight=cfg['alpha_weight'],\n",
          "    beta_weight=cfg['beta_weight'],\n",
          "    print_iter=cfg['print_iter'],\n",
          "    seed=SEED,\n",
          ")\n",
          "mechanism_params = logs.pop('mechanism_params')\n",
          "policy_params    = logs.pop('policy_params')\n",
          "print('Training complete.')\n"]),

    md("## Plot training curves"),

    code(["fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
          "axes[0].plot(logs['reg_plus']); axes[0].set_title('Regret+ '+ VARIANT); axes[0].set_xlabel('Outer Epoch')\n",
          "axes[1].plot(logs['gap_plus']); axes[1].set_title('Profit Gap+');        axes[1].set_xlabel('Outer Epoch')\n",
          "axes[2].plot(logs['ir']);        axes[2].set_title('IR Loss');            axes[2].set_xlabel('Outer Epoch')\n",
          "plt.suptitle(f'Exp 1 — Variant {VARIANT}', y=1.02)\n",
          "plt.tight_layout(); plt.show()\n"]),

    md("## Evaluate trained mechanism"),

    code(["eval_keys = jax.random.split(jax.random.PRNGKey(9999), 32)\n",
          "results = eval_budget_mechanism_multikey(\n",
          "    auction=auction,\n",
          "    mechanism_network=mechanism_network,\n",
          "    mechanism_params=mechanism_params,\n",
          "    profit_goal=cfg['profit_goal'],\n",
          "    num_samples=256, keys=eval_keys,\n",
          "    policy_network=policy_network,\n",
          "    policy_params=policy_params,\n",
          ")\n",
          "print_eval_report(results, prefix=f'Variant {VARIANT}: ')\n"]),

    md("## Compare A vs B vs C (optional — runs all three)"),

    code(["all_results = {}\n",
          "for v in ['A', 'B', 'C']:\n",
          "    print(f'\\n===== Variant {v} =====')\n",
          "    mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[v]))\n",
          "    pnet = build_bounded_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
          "                                    G=auction.num_goods, hidden=128, residual_init_std=1.0)\n",
          "    vl = nested_GD_budget(auction=auction, mechanism_network=mnet, policy_network=pnet,\n",
          "                          profit_goal=cfg['profit_goal'],\n",
          "                          num_outer_epochs=cfg['num_outer_epochs'],\n",
          "                          lr_theta=cfg['lr_theta'], num_inner_steps=cfg['num_inner_steps'],\n",
          "                          lr_w=cfg['lr_w'], num_samples_train=cfg['num_samples_train'],\n",
          "                          alpha_weight=cfg['alpha_weight'], beta_weight=cfg['beta_weight'],\n",
          "                          print_iter=cfg['print_iter'], seed=SEED)\n",
          "    vm = vl.pop('mechanism_params'); vp = vl.pop('policy_params')\n",
          "    vr = eval_budget_mechanism_multikey(auction=auction, mechanism_network=mnet,\n",
          "                                        mechanism_params=vm, profit_goal=cfg['profit_goal'],\n",
          "                                        num_samples=256, K=16, base_seed=9999)\n",
          "    all_results[v] = dict(logs=vl, eval=vr, mech_params=vm, pol_params=vp)\n",
          "\n",
          "print('\\n=== Comparison (Exp 1) ===')\n",
          "print(f'{\"Variant\":>8} | {\"Revenue\":>9} | {\"IR Loss\":>9} | {\"BV Rate\":>8}')\n",
          "print('-' * 45)\n",
          "for v, r in all_results.items():\n",
          "    ev = r['eval']\n",
          "    print(f'{v:>8} | {ev[\"profit_stats\"][\"mean\"]:>9.4f} | '\n",
          "          f'{ev[\"ir_loss_stats\"][\"mean\"]:>9.4f} | '\n",
          "          f'{ev[\"budget_viol_stats\"][\"mean\"]:>8.4f}')\n"]),

    md("## Save outputs"),

    code(["save_dir = f'outputs/public_budget/seed{SEED}'\n",
          "os.makedirs(save_dir, exist_ok=True)\n",
          "with open(f'{save_dir}/logs_{VARIANT}.pkl', 'wb') as f:\n",
          "    pickle.dump(logs, f)\n",
          "with open(f'{save_dir}/mech_params_{VARIANT}.pkl', 'wb') as f:\n",
          "    pickle.dump(mechanism_params, f)\n",
          "metrics_json = {k: v for k, v in results.items() if isinstance(v, dict)}\n",
          "metrics_json['variant'] = VARIANT\n",
          "with open(f'{save_dir}/eval_{VARIANT}.json', 'w') as f:\n",
          "    json.dump(metrics_json, f, indent=2)\n",
          "print(f'Saved to {save_dir}')\n"]),
]


# ==========================================================================
# Experiment 2: Multi-good privately correlated budget auction notebook
# ==========================================================================

EXP2_CELLS = [
    md("# Budget Auction — Experiment 2: Multi-Good Privately Correlated Budget Auction\n\n"
       "Compares mechanism input variants **A** (GBAM/public-balance), **B** (richer history), "
       "**C** (full history) in a setting *beyond* Deng 2021: types are multi-dimensional and "
       "transitions depend on private type history (private correlation).\n\n"
       "The key hypothesis: richer history compression should outperform the GBAM-style compressed "
       "state here, because the simple public balance is no longer theoretically sufficient."),

    code(["%pip install -q git+https://github.com/deepmind/dm-haiku\n",
          "%pip install -q jaxopt optax flax pyyaml\n"]),

    code(["try:\n",
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
          "    profit_goal=2.0,\n",
          "    num_outer_epochs=2000,\n",
          "    num_inner_steps=300,\n",
          "    lr_theta=1e-4, lr_w=3e-4,\n",
          "    num_samples_train=64,\n",
          "    alpha_weight=1e5, beta_weight=1e3,\n",
          "    print_iter=100,\n",
          "    policy_hidden=128, residual_init_std=1.0,\n",
          ")\n",
          "print('Config:', cfg)\n"]),

    md("## Build auction environment"),

    code(["auction = create_private_budget_multigood_auction(\n",
          "    n_buyers=cfg['n_buyers'], n_goods=cfg['n_goods'],\n",
          "    horizon=cfg['horizon'],  discount=cfg['discount'],\n",
          "    initial_budget=cfg['initial_budget'], seed=SEED,\n",
          ")\n",
          "print(f'Auction: {auction.num_buyers}B x {auction.num_goods}G x T={auction.num_horizons}')\n",
          "print(f'Initial budgets: {auction.initial_budgets}')\n",
          "print(f'Type space: [{auction.types_low}, {auction.types_high}]')\n"]),

    code(["# Sanity check: sample one episode trajectory\n",
          "key = jax.random.PRNGKey(0)\n",
          "s0 = auction.get_initial_state(key)\n",
          "print('Initial state:')\n",
          "print(f'  types  = {s0.current_type}   (shape: {s0.current_type.shape})')\n",
          "print(f'  alpha  = {s0.alpha}   (should be 0)')\n",
          "print(f'  beta   = {s0.beta}    (should be {auction.initial_budgets})')\n"]),

    md("## Check type transition dynamics"),

    code(["# Verify private correlation: types at t+1 should be near types at t\n",
          "from budget_auction.envs.private_budget_multigood_env import private_budget_transition_fn\n",
          "\n",
          "t0_types = s0.current_type  # (B, G)\n",
          "alloc_hist = jnp.zeros((auction.num_horizons, auction.num_buyers, auction.num_goods))\n",
          "keys_trans = jax.random.split(jax.random.PRNGKey(1), 20)\n",
          "samples = [auction.transition_function(auction, 0, t0_types, alloc_hist, k) for k in keys_trans]\n",
          "samples = jnp.stack(samples)  # (20, B, G)\n",
          "print(f'Current types: {t0_types}')\n",
          "print(f'Next types mean: {jnp.mean(samples, 0)}')\n",
          "print(f'Next types std:  {jnp.std(samples, 0)}')\n",
          "print('(Should be close to current types with small sigma_no=0.01 and no allocation)')\n"]),

    md("## Build mechanism network"),

    code(["raw_fn = VARIANT_NETWORK[VARIANT]\n",
          "mechanism_network = hk.without_apply_rng(hk.transform(raw_fn))\n",
          "policy_network    = build_bounded_policy_net(\n",
          "    d_priv=auction.d_priv, d_pub=auction.d_pub, G=auction.num_goods,\n",
          "    hidden=cfg['policy_hidden'], residual_init_std=cfg['residual_init_std'],\n",
          ")\n",
          "print(f'Mechanism variant: {VARIANT}  (n_goods={auction.num_goods})')\n"]),

    md("## Evaluate myopic baseline"),

    code(["print('--- Baseline B1: Static Myopic ---')\n",
          "b1 = run_static_myopic_baseline(auction, num_samples=128, num_keys=8, base_seed=1)\n",
          "for k, v in b1.items():\n",
          "    if isinstance(v, float): print(f'  {k}: {v:.4f}')\n"]),

    md("## Train learned mechanism"),

    code(["logs = nested_GD_budget(\n",
          "    auction=auction,\n",
          "    mechanism_network=mechanism_network,\n",
          "    policy_network=policy_network,\n",
          "    profit_goal=cfg['profit_goal'],\n",
          "    num_outer_epochs=cfg['num_outer_epochs'],\n",
          "    lr_theta=cfg['lr_theta'],\n",
          "    num_inner_steps=cfg['num_inner_steps'],\n",
          "    lr_w=cfg['lr_w'],\n",
          "    num_samples_train=cfg['num_samples_train'],\n",
          "    alpha_weight=cfg['alpha_weight'],\n",
          "    beta_weight=cfg['beta_weight'],\n",
          "    print_iter=cfg['print_iter'],\n",
          "    seed=SEED,\n",
          ")\n",
          "mechanism_params = logs.pop('mechanism_params')\n",
          "policy_params    = logs.pop('policy_params')\n",
          "print('Training complete.')\n"]),

    md("## Plot training curves"),

    code(["fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
          "axes[0].plot(logs['reg_plus']); axes[0].set_title('Regret+ ' + VARIANT); axes[0].set_xlabel('Outer Epoch')\n",
          "axes[1].plot(logs['gap_plus']); axes[1].set_title('Profit Gap+');        axes[1].set_xlabel('Outer Epoch')\n",
          "axes[2].plot(logs['ir']);        axes[2].set_title('IR Loss');            axes[2].set_xlabel('Outer Epoch')\n",
          "plt.suptitle(f'Exp 2 — Variant {VARIANT} ({auction.num_goods} goods)', y=1.02)\n",
          "plt.tight_layout(); plt.show()\n"]),

    md("## Evaluate trained mechanism"),

    code(["eval_keys = jax.random.split(jax.random.PRNGKey(9999), 32)\n",
          "results = eval_budget_mechanism_multikey(\n",
          "    auction=auction,\n",
          "    mechanism_network=mechanism_network,\n",
          "    mechanism_params=mechanism_params,\n",
          "    profit_goal=cfg['profit_goal'],\n",
          "    num_samples=256, keys=eval_keys,\n",
          "    policy_network=policy_network,\n",
          "    policy_params=policy_params,\n",
          ")\n",
          "print_eval_report(results, prefix=f'Exp2 Variant {VARIANT}: ')\n"]),

    md("## Compare A vs B vs C"),

    code(["all_results = {}\n",
          "for v in ['A', 'B', 'C']:\n",
          "    print(f'\\n===== Variant {v} =====')\n",
          "    mnet = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[v]))\n",
          "    pnet = build_bounded_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,\n",
          "                                    G=auction.num_goods, hidden=128, residual_init_std=1.0)\n",
          "    vl = nested_GD_budget(auction=auction, mechanism_network=mnet, policy_network=pnet,\n",
          "                          profit_goal=cfg['profit_goal'],\n",
          "                          num_outer_epochs=cfg['num_outer_epochs'],\n",
          "                          lr_theta=cfg['lr_theta'], num_inner_steps=cfg['num_inner_steps'],\n",
          "                          lr_w=cfg['lr_w'], num_samples_train=cfg['num_samples_train'],\n",
          "                          alpha_weight=cfg['alpha_weight'], beta_weight=cfg['beta_weight'],\n",
          "                          print_iter=cfg['print_iter'], seed=SEED)\n",
          "    vm = vl.pop('mechanism_params'); vp = vl.pop('policy_params')\n",
          "    vr = eval_budget_mechanism_multikey(auction=auction, mechanism_network=mnet,\n",
          "                                        mechanism_params=vm, profit_goal=cfg['profit_goal'],\n",
          "                                        num_samples=256, K=16, base_seed=9999)\n",
          "    all_results[v] = dict(logs=vl, eval=vr, mech_params=vm, pol_params=vp)\n",
          "\n",
          "print('\\n=== Comparison (Exp 2) ===')\n",
          "print(f'{\"Variant\":>8} | {\"Revenue\":>9} | {\"IR Loss\":>9} | {\"BV Rate\":>8}')\n",
          "print('-' * 45)\n",
          "for v, r in all_results.items():\n",
          "    ev = r['eval']\n",
          "    print(f'{v:>8} | {ev[\"profit_stats\"][\"mean\"]:>9.4f} | '\n",
          "          f'{ev[\"ir_loss_stats\"][\"mean\"]:>9.4f} | '\n",
          "          f'{ev[\"budget_viol_stats\"][\"mean\"]:>8.4f}')\n"]),

    md("## Save outputs"),

    code(["save_dir = f'outputs/private_budget_multigood/seed{SEED}'\n",
          "os.makedirs(save_dir, exist_ok=True)\n",
          "with open(f'{save_dir}/logs_{VARIANT}.pkl', 'wb') as f:\n",
          "    pickle.dump(logs, f)\n",
          "with open(f'{save_dir}/mech_params_{VARIANT}.pkl', 'wb') as f:\n",
          "    pickle.dump(mechanism_params, f)\n",
          "metrics_json = {k: v for k, v in results.items() if isinstance(v, dict)}\n",
          "metrics_json['variant'] = VARIANT\n",
          "with open(f'{save_dir}/eval_{VARIANT}.json', 'w') as f:\n",
          "    json.dump(metrics_json, f, indent=2)\n",
          "print(f'Saved to {save_dir}')\n"]),
]


# Write notebooks
nb1_path = os.path.join(NB_DIR, "budget_public_colab.ipynb")
with open(nb1_path, "w") as f:
    json.dump(notebook(EXP1_CELLS), f, indent=1)
print(f"Written: {nb1_path}")

nb2_path = os.path.join(NB_DIR, "budget_private_multigood_colab.ipynb")
with open(nb2_path, "w") as f:
    json.dump(notebook(EXP2_CELLS), f, indent=1)
print(f"Written: {nb2_path}")
