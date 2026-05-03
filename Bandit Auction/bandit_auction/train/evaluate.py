"""
evaluate.py — Multi-key evaluation for bandit auction mechanisms.

Usage:
    python -m bandit_auction.train.evaluate \\
        --run_dir outputs/cont_single/penalty/seed0 \\
        --num_samples 256 \\
        --num_keys 32
"""

import argparse
import os
import json
import pickle
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import haiku as hk
import yaml

from bandit_auction.core import (
    AuctionSimulator, BoundedContPolicy, DiscretePolicy,
    build_bounded_policy_net, build_discrete_policy_net, SEED,
)
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network

EXPERIMENT_ENVS = {
    "cont_single":    ("bandit_auction.envs.cont_single",    "create_cont_single_auction"),
    "cont_multi":     ("bandit_auction.envs.cont_multi",     "create_cont_multi_auction"),
    "discrete_single":("bandit_auction.envs.discrete_single","create_discrete_single_auction"),
    "discrete_multi": ("bandit_auction.envs.discrete_multi", "create_discrete_multi_auction"),
}


# ---------------------------------------------------------------------------
# Core multi-key evaluation
# ---------------------------------------------------------------------------

def _summary_stats(x: jnp.ndarray) -> Dict:
    return {
        "mean":   float(jnp.mean(x)),
        "std":    float(jnp.std(x)),
        "min":    float(jnp.min(x)),
        "max":    float(jnp.max(x)),
        "median": float(jnp.median(x)),
    }


def _stack(xs):
    return jnp.stack([jnp.asarray(x) for x in xs])


def eval_mechanism_multikey(
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
    is_discrete: bool = False,
) -> Dict:
    """Multi-key unbiased evaluation of a trained bandit auction mechanism.

    Evaluates:
      - profit and profit_gap
      - IR loss (sum relu(-reward_i))
      - buyer surplus (sum reward_i)
      - IC regret (if policy provided)

    Returns a dict with per-key arrays and summary stats.
    """
    if keys is None:
        keys = jax.random.split(jax.random.PRNGKey(base_seed), K)

    mechanism = Mechanism(mechanism_network, mechanism_params)
    policy = None
    if (policy_network is not None) and (policy_params is not None):
        policy_cls = DiscretePolicy if is_discrete else BoundedContPolicy
        policy = policy_cls(policy_network, policy_params)

    update_belief_fn = (
        auction.get_next_belief_state_discrete if is_discrete
        else auction.get_next_belief_state_continuous
    )

    profits, gaps, ir_losses, surpluses, regrets = [], [], [], [], []

    for k in keys:
        k1, k2, k3 = jax.random.split(k, 3)

        prof = AuctionSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples=num_samples, key=k1
        )
        gap = profit_goal - prof

        rewards = AuctionSimulator.estimate_cumulative_rewards(
            auction, mechanism, num_samples=num_samples, key=k2
        )
        ir_per_buyer = AuctionSimulator.estimate_relu_cumulative_rewards(
            auction, mechanism, num_samples=num_samples, key=k2
        )
        ir      = jnp.sum(ir_per_buyer)   # E[relu(-U_i)], summed over buyers
        surplus = jnp.sum(rewards)        # E[U_i], summed over buyers

        profits.append(prof)
        gaps.append(gap)
        ir_losses.append(ir)
        surpluses.append(surplus)

        if policy is not None:
            reg = AuctionSimulator.estimate_cumulative_regret(
                auction, mechanism, policy, num_samples=num_samples, key=k3,
                update_belief_fn=update_belief_fn,
            )
            regrets.append(reg)

    profits  = _stack(profits)
    gaps     = _stack(gaps)
    ir_losses = _stack(ir_losses)
    surpluses = _stack(surpluses)

    out = {
        "profit":       profits,
        "profit_stats": _summary_stats(profits),
        "gap":          gaps,
        "gap_stats":    _summary_stats(gaps),
        "gap_plus":     jnp.maximum(0.0, gaps),
        "gap_plus_stats": _summary_stats(jnp.maximum(0.0, gaps)),
        "ir_loss":      ir_losses,
        "ir_loss_stats": _summary_stats(ir_losses),
        "buyer_surplus":      surpluses,
        "buyer_surplus_stats": _summary_stats(surpluses),
        "K":            int(len(keys)),
        "num_samples":  int(num_samples),
    }

    if policy is not None:
        regrets = _stack(regrets)
        out.update({
            "regret":       regrets,
            "regret_stats": _summary_stats(regrets),
            "ic_plus":      jnp.maximum(0.0, regrets),
            "ic_plus_stats": _summary_stats(jnp.maximum(0.0, regrets)),
        })

    return out


def print_eval_report(results: Dict, prefix: str = ""):
    print(f"\n{'='*60}")
    print(f"  {prefix}Evaluation Report")
    print(f"{'='*60}")
    for key in ["profit_stats", "gap_plus_stats", "buyer_surplus_stats", "ir_loss_stats"]:
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
# CLI
# ---------------------------------------------------------------------------

def _build_auction(cfg, seed):
    experiment = cfg["experiment"]
    mod_name, fn_name = EXPERIMENT_ENVS[experiment]
    import importlib
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, fn_name)
    kwargs = {k: cfg[k] for k in [
        "n_buyers", "n_goods", "horizon", "discount", "allocation_cost"
    ] if k in cfg}
    kwargs["seed"] = seed
    if experiment.startswith("cont"):
        kwargs.update({k: cfg[k] for k in [
            "type_mean", "type_std", "types_low", "types_high",
            "sigma_no", "trans_lambda",
        ] if k in cfg})
    else:
        if "type_values" in cfg:
            kwargs["type_values"] = cfg["type_values"]
        if "type_values_per_good" in cfg:
            kwargs["type_values_per_good"] = cfg["type_values_per_good"]
    return factory(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Bandit Auction Evaluation")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_keys", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    run_dir = args.run_dir

    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    experiment = cfg["experiment"]
    is_discrete = experiment.startswith("discrete")

    auction = _build_auction(cfg, seed)

    mechanism_network = build_mechanism_network()
    if is_discrete:
        policy_network = build_discrete_policy_net(
            d_priv=auction.d_priv, d_pub=auction.d_pub,
            hidden=cfg.get("policy_hidden", 128),
        )
    else:
        policy_network = build_bounded_policy_net(
            d_priv=auction.d_priv, d_pub=auction.d_pub,
            G=auction.num_goods,
            hidden=cfg.get("policy_hidden", 128),
            residual_init_std=cfg.get("residual_init_std", 1.0),
        )

    with open(os.path.join(run_dir, "mechanism_params.pkl"), "rb") as f:
        mechanism_params = pickle.load(f)
    with open(os.path.join(run_dir, "policy_params.pkl"), "rb") as f:
        policy_params = pickle.load(f)

    profit_goal = cfg.get("profit_goal", 0.0)

    results = eval_mechanism_multikey(
        auction           = auction,
        mechanism_network = mechanism_network,
        mechanism_params  = mechanism_params,
        profit_goal       = profit_goal,
        num_samples       = args.num_samples,
        K                 = args.num_keys,
        base_seed         = args.seed,
        policy_network    = policy_network,
        policy_params     = policy_params,
        is_discrete       = is_discrete,
    )

    print_eval_report(results, prefix=f"{experiment}, seed={seed}: ")

    metrics = {k: v for k, v in results.items() if isinstance(v, dict)}
    metrics["experiment"] = experiment
    metrics["seed"] = seed
    metrics["method"] = cfg.get("method", "unknown")
    with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved eval_metrics.json to {run_dir}")


if __name__ == "__main__":
    main()
