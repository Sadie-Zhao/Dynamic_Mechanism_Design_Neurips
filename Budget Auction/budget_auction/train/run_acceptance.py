"""
run_acceptance.py — CLI entry point for nested-GD training of budget auction mechanisms.

Adapts nested_GD_obj_cont_bounded from the bandit notebook for budget experiments.

Key differences from the bandit version:
  1. Uses BudgetState (includes alpha, beta) instead of State
  2. Mechanism is BudgetMechanism with payment feasibility transform
  3. Metrics include budget_viol_rate and ir_viol_rate
  4. Config loaded from YAML file; variant (A/B/C) selects mechanism network

Usage:
    python -m budget_auction.train.run_acceptance \\
        --config budget_auction/configs/public_budget_small.yaml \\
        --seed 0 \\
        --variant A
"""

import argparse
import os
import json
import pickle
import time
from typing import Dict

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import yaml

from budget_auction.core import (
    BudgetSimulator,
    BoundedContPolicy,
    build_bounded_policy_net,
    SEED,
)
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK


# ---------------------------------------------------------------------------
# Main training function (adapted from nested_GD_obj_cont_bounded)
# ---------------------------------------------------------------------------

def nested_GD_budget(
    auction,
    mechanism_network,          # hk.Transformed raw network (outputs alloc_probs, z)
    policy_network,             # hk.Transformed policy network
    profit_goal: float,

    # Outer loop (mechanism update)
    num_outer_epochs: int = 2000,
    outer_steps_per_epoch: int = 1,
    lr_theta: float = 1e-4,

    # Inner loop (oracle policy)
    num_inner_steps: int = 600,
    lr_w: float = 3e-4,
    inner_opt_reset_each_outer: bool = True,

    # Monte Carlo
    num_samples_train: int = 128,

    # Loss weights
    alpha_weight: float = 1e5,
    beta_weight: float = 1e3,

    # Diagnostics
    print_iter: int = 50,

    # Early stopping
    early_stop: bool = False,
    tol: float = 0.1,
    patience: int = 10,

    # Checkpoint recording (for binary search)
    # Record mechanism snapshots every `checkpoint_interval` epochs when
    # training reg <= checkpoint_thresh AND training ir <= checkpoint_thresh.
    # Saved to logs["checkpoint_mech"] as list of dicts.
    checkpoint_thresh: float = None,
    checkpoint_interval: int = 50,

    seed: int = SEED,
) -> Dict:
    """
    Nested gradient descent for budget auction mechanism learning.

    OUTER: minimise feasibility-style loss over mechanism params theta
        L(theta) = alpha_weight * (Regret)_+ + (profit_goal - Profit)_+ + beta_weight * IR
    INNER: approximately maximise Regret over policy params w (oracle)

    Returns: logs dict with keys:
        loss, reg, gap, ir, reg_plus, gap_plus, oracle_reg_last, good_mech,
        checkpoint_mech (list of dicts, only if checkpoint_thresh is set)
    """
    prng = hk.PRNGSequence(seed)

    # ---- Init mechanism params ----
    init_state = auction.get_initial_state(next(prng))
    mechanism_params = mechanism_network.init(
        next(prng),
        init_state,
        jnp.zeros((auction.num_buyers, auction.num_goods)),
        auction,
    )

    # ---- Init policy params ----
    init_belief = auction.get_initial_belief_state(init_state)
    policy_params = policy_network.init(next(prng), init_belief, auction)

    # ---- Optimizers ----
    theta_opt = optax.adam(lr_theta)
    theta_opt_state = theta_opt.init(mechanism_params)

    w_opt = optax.adam(lr_w)
    w_opt_state = w_opt.init(policy_params)

    # ---- Core estimators (capture via closure) ----

    @jax.jit
    def cumulative_regret(mechanism_params, policy_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        policy = BoundedContPolicy(policy_network, policy_params)
        return BudgetSimulator.estimate_cumulative_regret(
            auction, mechanism, policy, num_samples_train, key
        )

    @jax.jit
    def profit_gap(mechanism_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        prof = BudgetSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples_train, key
        )
        return profit_goal - prof

    @jax.jit
    def ir_loss(mechanism_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        rewards = BudgetSimulator.estimate_relu_cumulative_rewards(
            auction, mechanism, num_samples_train * 5, key
        )
        return jnp.sum(rewards)

    # ---- Inner: gradient ascent on w ----

    @jax.jit
    def inner_step(mechanism_params, policy_params, w_opt_state, key):
        reg, grad_w = jax.value_and_grad(cumulative_regret, argnums=1)(
            mechanism_params, policy_params, key
        )
        grad_w_ascent = jax.tree_util.tree_map(lambda g: -g, grad_w)
        updates, w_opt_state = w_opt.update(grad_w_ascent, w_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return reg, policy_params, w_opt_state

    def maximize_regret_oracle(mechanism_params, policy_params, w_opt_state, steps):
        reg_last = 0.0
        for _ in range(steps):
            key = next(prng)
            reg_last, policy_params, w_opt_state = inner_step(
                mechanism_params, policy_params, w_opt_state, key
            )
        return reg_last, policy_params, w_opt_state

    # ---- Outer: update theta ----

    @jax.jit
    def total_loss_with_oracle(mechanism_params, policy_params_star, key):
        k1, k2, k3 = jax.random.split(key, 3)
        reg = cumulative_regret(mechanism_params, policy_params_star, k1)
        gap = profit_gap(mechanism_params, k2)
        ir = ir_loss(mechanism_params, k3)

        reg_plus = jnp.maximum(0.0, reg)
        gap_plus = jnp.maximum(0.0, gap)

        loss = alpha_weight * reg_plus + gap_plus + beta_weight * ir
        return loss, (reg, gap, ir, reg_plus, gap_plus)

    @jax.jit
    def outer_step(mechanism_params, theta_opt_state, policy_params_star, key):
        (loss, aux), grad_theta = jax.value_and_grad(
            total_loss_with_oracle, argnums=0, has_aux=True
        )(mechanism_params, policy_params_star, key)
        updates, theta_opt_state = theta_opt.update(grad_theta, theta_opt_state, mechanism_params)
        mechanism_params = optax.apply_updates(mechanism_params, updates)
        return loss, aux, mechanism_params, theta_opt_state

    # ---- Logs ----
    logs = {
        "loss": [], "reg": [], "gap": [], "ir": [],
        "reg_plus": [], "gap_plus": [],
        "oracle_reg_last": [],
        "good_mech": [],
        "checkpoint_mech": [],
    }
    sat_count = 0
    _ckpt_enabled = checkpoint_thresh is not None

    # ---- Training loop ----
    t0 = time.time()
    for outer_epoch in range(num_outer_epochs):

        if inner_opt_reset_each_outer and outer_epoch > 0:
            init_s = auction.get_initial_state(next(prng))
            init_b = auction.get_initial_belief_state(init_s)
            policy_params = policy_network.init(next(prng), init_b, auction)
            w_opt_state = w_opt.init(policy_params)

        reg_star, policy_params, w_opt_state = maximize_regret_oracle(
            mechanism_params, policy_params, w_opt_state, num_inner_steps
        )
        policy_params_star = policy_params

        for _ in range(outer_steps_per_epoch):
            loss, (reg, gap, ir, reg_plus, gap_plus), mechanism_params, theta_opt_state = \
                outer_step(mechanism_params, theta_opt_state, policy_params_star, next(prng))

        logs["loss"].append(float(loss))
        logs["reg"].append(float(reg))
        logs["gap"].append(float(gap))
        logs["ir"].append(float(ir))
        logs["reg_plus"].append(float(reg_plus))
        logs["gap_plus"].append(float(gap_plus))
        logs["oracle_reg_last"].append(float(reg_star))

        if outer_epoch % print_iter == 0:
            elapsed = time.time() - t0
            print(f"=== Outer Epoch {outer_epoch} ({elapsed:.1f}s) ===")
            print(f"  oracle regret (train):  {float(reg_star):.6f}")
            print(f"  reg (in loss):          {float(reg):.6f}   reg_+={float(reg_plus):.6f}")
            print(f"  gap (goal-profit):      {float(gap):.6f}   gap_+={float(gap_plus):.6f}")
            print(f"  ir_loss:                {float(ir):.6f}")
            print(f"  total loss:             {float(loss):.6f}")

        # ---- Checkpoint recording (for binary search) ----
        if (_ckpt_enabled
                and (outer_epoch % checkpoint_interval == 0)
                and (float(reg) <= checkpoint_thresh)
                and (float(ir) <= checkpoint_thresh)):
            logs["checkpoint_mech"].append({
                "epoch": outer_epoch,
                "reg_train": float(reg),
                "ir_train": float(ir),
                "gap_train": float(gap),
                "mechanism_params": mechanism_params,
                "policy_params": policy_params_star,
            })

        if early_stop:
            ok = (float(reg) <= tol) and (float(ir) <= tol * 10) and (float(gap) <= tol)
            if ok:
                logs["good_mech"].append(
                    (mechanism_params, policy_params_star, float(reg), float(ir), float(gap))
                )
            sat_count = sat_count + 1 if ok else 0
            if sat_count >= patience:
                print(f"[EARLY STOP] at epoch {outer_epoch}")
                break

    logs["mechanism_params"] = mechanism_params
    logs["policy_params"] = policy_params
    return logs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Budget Auction Nested-GD Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["A", "B", "C"], default=None,
                        help="Mechanism input variant (overrides config)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    variant = args.variant or cfg.get("mechanism_variant", "A")
    seed = args.seed
    output_dir = args.output_dir or cfg.get("output_dir", f"outputs/{cfg['experiment']}/{variant}/seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config snapshot
    cfg_copy = dict(cfg)
    cfg_copy["seed"] = seed
    cfg_copy["mechanism_variant"] = variant
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg_copy, f)

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
        cproj_init_std=0.0,
    )

    # Train
    logs = nested_GD_budget(
        auction=auction,
        mechanism_network=mechanism_network,
        policy_network=policy_network,
        profit_goal=cfg.get("profit_goal", 2.0),
        num_outer_epochs=cfg.get("num_outer_epochs", 2000),
        outer_steps_per_epoch=cfg.get("outer_steps_per_epoch", 1),
        lr_theta=cfg.get("lr_theta", 1e-4),
        num_inner_steps=cfg.get("num_inner_steps", 300),
        lr_w=cfg.get("lr_w", 3e-4),
        num_samples_train=cfg.get("num_samples_train", 64),
        alpha_weight=cfg.get("alpha", 1e5),
        beta_weight=cfg.get("beta", 1e3),
        print_iter=cfg.get("print_iter", 50),
        early_stop=cfg.get("early_stop", False),
        tol=cfg.get("tol", 0.1),
        patience=cfg.get("patience", 10),
        seed=seed,
    )

    # Save outputs
    mechanism_params = logs.pop("mechanism_params")
    policy_params = logs.pop("policy_params")

    with open(os.path.join(output_dir, "logs.pkl"), "wb") as f:
        pickle.dump(logs, f)
    with open(os.path.join(output_dir, "mechanism_params.pkl"), "wb") as f:
        pickle.dump(mechanism_params, f)
    with open(os.path.join(output_dir, "policy_params.pkl"), "wb") as f:
        pickle.dump(policy_params, f)

    print(f"\nTraining complete. Outputs saved to: {output_dir}")

    # Quick summary metrics
    summary = {
        "final_reg": logs["reg"][-1] if logs["reg"] else None,
        "final_gap": logs["gap"][-1] if logs["gap"] else None,
        "final_ir": logs["ir"][-1] if logs["ir"] else None,
        "num_good_mech": len(logs["good_mech"]),
        "variant": variant,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
