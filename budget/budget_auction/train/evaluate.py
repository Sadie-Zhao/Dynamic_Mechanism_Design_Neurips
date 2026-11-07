"""
evaluate.py — CLI entry point for evaluating a trained budget auction mechanism.

Extends eval_mechanism_multikey from the bandit notebook with budget metrics:
    - budget_violation_rate
    - ir_violation_rate

Usage:
    python -m budget_auction.train.evaluate \\
        --run_dir outputs/public_budget/A/seed0 \\
        --num_samples 256 \\
        --num_keys 32
"""

import argparse
import os
import json
import pickle
from statistics import NormalDist
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import haiku as hk
import yaml

from budget_auction.core import (
    BudgetSimulator,
    BoundedContPolicy,
    build_bounded_policy_net,
)
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK


# ---------------------------------------------------------------------------
# Evaluation function (mirrors eval_mechanism_multikey from bandit notebook)
# ---------------------------------------------------------------------------

def _summary_stats(x: jnp.ndarray) -> Dict:
    n = int(x.shape[0])
    sd = float(jnp.std(x, ddof=1)) if n > 1 else 0.0
    mean = float(jnp.mean(x))
    se = sd / n**0.5
    return {
        "mean": mean,
        "std": sd,
        "se": se,
        "ci95": [mean-1.96*se, mean+1.96*se] if n > 1 else None,
        "num_batches": n,
        "min": float(jnp.min(x)),
        "max": float(jnp.max(x)),
        "median": float(jnp.median(x)),
    }


def summarize_deviation_gains(gains):
    """IC diagnostics for fixed policies on independent evaluation batches.

    Positive parts are taken AFTER expectation and separately for each buyer.
    Bonferroni normal-approximation UCBs cover sampling uncertainty, not oracle
    optimization error. This is empirical evidence, never a global certificate.
    """
    gains = jnp.asarray(gains)
    if gains.ndim != 2 or gains.shape[0] < 2:
        raise ValueError("Need at least two independent batches of per-buyer gains")
    B = gains.shape[1]
    means = jnp.mean(gains, axis=0)
    se = jnp.std(gains, axis=0, ddof=1) / gains.shape[0]**0.5
    z = NormalDist().inv_cdf(1-0.05/B)
    positive = jax.nn.relu(means)
    return {
        "deviation_gain_per_buyer": means.tolist(),
        "deviation_se_per_buyer": se.tolist(),
        "ic_per_buyer": positive.tolist(),
        "ic_sum": float(jnp.sum(positive)),
        "ic_mean": float(jnp.mean(positive)),
        "ic_max_buyer": float(jnp.max(positive)),
        "ic_sum_sampling_ucb95": float(jnp.sum(jax.nn.relu(means+z*se))),
        "evidence_type": "empirical_fixed_policy_audit",
    }


def _stack(xs):
    return jnp.stack([jnp.asarray(x) for x in xs])


def eval_budget_mechanism_multikey(
    auction,
    mechanism_network,
    mechanism_params,
    profit_goal: float,
    num_samples: int = 256,
    keys=None,
    base_seed: int = 0,
    K: int = 32,
    policy_network=None,
    policy_params=None,
) -> Dict:
    """
    Multi-key unbiased evaluation of a trained budget mechanism.

    Evaluates:
      - profit and profit_gap
      - IR loss (sum |U_i|)
      - IC loss / regret proxy (if policy provided)
      - budget_violation_rate
      - ir_violation_rate

    Returns a dict with per-key arrays and summary stats.
    """
    if keys is None:
        keys = jax.random.split(jax.random.PRNGKey(base_seed), K)

    mechanism = BudgetMechanism(mechanism_network, mechanism_params)
    policy = None
    if (policy_network is not None) and (policy_params is not None):
        policy = BoundedContPolicy(policy_network, policy_params)

    profits, gaps, ir_losses, surpluses, regrets, bv_rates, iv_rates = [], [], [], [], [], [], []

    for k in keys:
        k1, k2, k3, k4 = jax.random.split(k, 4)

        prof = BudgetSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples=num_samples, key=k1
        )
        gap = profit_goal - prof

        rewards = BudgetSimulator.estimate_cumulative_rewards(
            auction, mechanism, num_samples=num_samples, key=k2
        )
        # IR loss for pass/fail: relu(-reward). Always 0 by construction of the
        # payment feasibility transform under truthful play.
        ir = jnp.sum(jax.nn.relu(-rewards))
        # Buyer surplus: total positive utility retained by buyers (always >= 0).
        surplus = jnp.sum(rewards)

        bv, iv = BudgetSimulator.estimate_violation_rates(
            auction, mechanism, num_samples=num_samples, key=k3
        )

        profits.append(prof)
        gaps.append(gap)
        ir_losses.append(ir)
        surpluses.append(surplus)
        bv_rates.append(bv)
        iv_rates.append(iv)

        if policy is not None:
            reg = BudgetSimulator.estimate_cumulative_deviation_gains(
                auction, mechanism, policy, num_samples=num_samples, key=k4
            )
            regrets.append(reg)

    profits = _stack(profits)
    gaps = _stack(gaps)
    ir_losses = _stack(ir_losses)
    surpluses = _stack(surpluses)
    bv_rates = jnp.array(bv_rates)
    iv_rates = jnp.array(iv_rates)

    out = {
        "profit": profits,
        "profit_stats": _summary_stats(profits),
        "gap": gaps,
        "gap_stats": _summary_stats(gaps),
        "gap_plus": jnp.maximum(0.0, gaps),
        "gap_plus_stats": _summary_stats(jnp.maximum(0.0, gaps)),
        # IR loss = relu(-reward): always 0 by construction; used for pass/fail.
        "ir_loss": ir_losses,
        "ir_loss_stats": _summary_stats(ir_losses),
        # Buyer surplus = sum(reward): positive, informative metric.
        "buyer_surplus": surpluses,
        "buyer_surplus_stats": _summary_stats(surpluses),
        "budget_viol_rate": bv_rates,
        "budget_viol_stats": _summary_stats(bv_rates),
        "ir_viol_rate": iv_rates,
        "ir_viol_stats": _summary_stats(iv_rates),
        "K": int(len(keys)),
        "num_samples": int(num_samples),
    }

    if policy is not None:
        gains = _stack(regrets)
        regrets = jnp.sum(gains, axis=1)
        clipped_batches = jnp.sum(jax.nn.relu(gains), axis=1)
        out.update({
            "deviation_gains_by_batch": gains,
            "regret": regrets,
            "regret_stats": _summary_stats(regrets),
            "ic_plus": clipped_batches,
            "ic_plus_stats": _summary_stats(clipped_batches),
        })
        out.update(summarize_deviation_gains(gains))

    return out


def print_eval_report(results: Dict, prefix: str = ""):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {prefix}Evaluation Report")
    print(f"{'='*60}")
    for key in ["profit_stats", "gap_plus_stats", "buyer_surplus_stats",
                "ir_loss_stats", "budget_viol_stats", "ir_viol_stats"]:
        if key in results:
            label = key.replace("_stats", "").replace("_", " ")
            s = results[key]
            print(f"  {label:25s}: {s['mean']:.4f} ± {s['std']:.4f}  "
                  f"[{s['min']:.4f}, {s['max']:.4f}]")
    if "regret_stats" in results:
        s = results["regret_stats"]
        print(f"  {'ic regret':25s}: {s['mean']:.4f} ± {s['std']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Select mechanism with lowest IR violation (same pattern as bandit notebook)
# ---------------------------------------------------------------------------

def select_lowest_ir(
    auction,
    mechanism_network,
    mech_params_list: List,
    num_samples: int = 256,
    num_keys: int = 32,
    seed: int = 0,
):
    keys = jax.random.split(jax.random.PRNGKey(seed), num_keys)
    best_idx, best_ir = None, float("inf")
    all_stats = []

    for i, params in enumerate(mech_params_list):
        mechanism = BudgetMechanism(mechanism_network, params)
        ir_vals = []
        for k in keys:
            rewards = BudgetSimulator.estimate_relu_cumulative_rewards(
                auction, mechanism, num_samples, k
            )
            ir_vals.append(float(jnp.sum(rewards)))
        ir_mean = float(jnp.mean(jnp.array(ir_vals)))
        all_stats.append({"idx": i, "ir_mean": ir_mean, "ir_std": float(jnp.std(jnp.array(ir_vals)))})
        if ir_mean < best_ir:
            best_ir, best_idx = ir_mean, i

    return best_idx, mech_params_list[best_idx], all_stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Budget Auction Evaluation")
    parser.add_argument("--run_dir", required=True,
                        help="Directory containing config.yaml, mechanism_params.pkl, logs.pkl")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_keys", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    run_dir = args.run_dir

    # Load config
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    variant = cfg.get("mechanism_variant", "A")
    seed = cfg.get("seed", 42)

    # Build auction
    experiment = cfg["experiment"]
    if experiment == "public_budget":
        from budget_auction.envs.public_budget_env import create_public_budget_auction
        auction = create_public_budget_auction(
            n_buyers=cfg.get("n_buyers", 2),
            n_goods=cfg.get("n_goods", 1),
            horizon=cfg.get("horizon", 5),
            discount=cfg.get("discount", 0.9),
            initial_budget=cfg.get("initial_budget", 3.0),
            seed=seed,
        )
    elif experiment == "private_budget_multigood":
        from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction
        auction = create_private_budget_multigood_auction(
            n_buyers=cfg.get("n_buyers", 2),
            n_goods=cfg.get("n_goods", 2),
            horizon=cfg.get("horizon", 5),
            discount=cfg.get("discount", 0.9),
            initial_budget=cfg.get("initial_budget", 3.0),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Build networks
    raw_network_fn = VARIANT_NETWORK[variant]
    mechanism_network = hk.without_apply_rng(hk.transform(raw_network_fn))
    policy_network = build_bounded_policy_net(
        d_priv=auction.d_priv,
        d_pub=auction.d_pub,
        G=auction.num_goods,
        hidden=cfg.get("policy_hidden", 128),
        residual_init_std=cfg.get("residual_init_std", 1.0),
    )

    # Load params
    with open(os.path.join(run_dir, "mechanism_params.pkl"), "rb") as f:
        mechanism_params = pickle.load(f)
    with open(os.path.join(run_dir, "policy_params.pkl"), "rb") as f:
        policy_params = pickle.load(f)

    # Evaluate
    profit_goal = cfg.get("profit_goal", 2.0)
    results = eval_budget_mechanism_multikey(
        auction=auction,
        mechanism_network=mechanism_network,
        mechanism_params=mechanism_params,
        profit_goal=profit_goal,
        num_samples=args.num_samples,
        K=args.num_keys,
        base_seed=args.seed,
        policy_network=policy_network,
        policy_params=policy_params,
    )

    print_eval_report(results, prefix=f"Variant {variant}, seed {seed}: ")

    # Save metrics (JSON-serializable subset)
    metrics = {
        k: v for k, v in results.items()
        if isinstance(v, dict)  # stats dicts
    }
    metrics["variant"] = variant
    metrics["seed"] = seed
    with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved eval_metrics.json to {run_dir}")


if __name__ == "__main__":
    main()
