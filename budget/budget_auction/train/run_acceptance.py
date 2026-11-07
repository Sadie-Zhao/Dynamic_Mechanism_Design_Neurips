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
from budget_auction.policies import SimpleDeviationPolicy
from budget_auction.train.revenue_objectives import (
    OBJECTIVES, objective_value, validate_resume_state,
)


# ---------------------------------------------------------------------------
# Main training function (adapted from nested_GD_obj_cont_bounded)
# ---------------------------------------------------------------------------

def nested_GD_budget(
    auction,
    mechanism_network,          # hk.Transformed raw network (outputs alloc_probs, z)
    policy_network,             # hk.Transformed policy network
    profit_goal: float | None,

    # Outer loop (mechanism update)
    num_outer_epochs: int = 2000,
    outer_steps_per_epoch: int = 1,
    lr_theta: float = 1e-4,

    # Inner loop (oracle policy)
    num_inner_steps: int = 600,
    lr_w: float = 3e-4,
    inner_opt_reset_each_outer: bool = True,
    policy_reset_interval: int = None,
    include_simple_deviations: bool = True,

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

    # Warm start: if provided, skip random init and use these params instead.
    # Useful for A_distilled warm-start fine-tuning.
    init_mechanism_params=None,
    init_policy_params=None,
    resume_state=None,
    checkpoint_callback=None,

    # Separate prospective search only; original frozen source is unchanged.
    revenue_objective: str = 'acceptance',
    resume_identity: str = None,
    extend_from_epochs: int = None,

    seed: int = SEED,
) -> Dict:
    """
    Nested gradient descent for budget auction mechanism learning.

    OUTER: acceptance uses a revenue-shortfall penalty; direct_revenue uses
        negative revenue. Both retain per-buyer positive IC penalties and IR.
    INNER: approximately maximise Regret over policy params w (oracle)

    Returns: logs dict with keys:
        loss, reg, gap, ir, reg_plus, gap_plus, oracle_reg_last, good_mech,
        checkpoint_mech (list of dicts, only if checkpoint_thresh is set)
    """
    prng = hk.PRNGSequence(seed)
    if revenue_objective not in OBJECTIVES:
        raise ValueError('Unknown revenue objective')
    if revenue_objective == 'acceptance' and profit_goal is None:
        raise ValueError('Acceptance training requires a revenue goal')
    if revenue_objective == 'direct_revenue' and early_stop:
        raise ValueError('Target-based early stopping is invalid for direct revenue')
    diagnostic_goal = 0. if profit_goal is None else profit_goal
    if policy_reset_interval is not None and policy_reset_interval < 1:
        raise ValueError("policy_reset_interval must be positive or None")
    training_contract = {
        'schema_version': 1, 'num_outer_epochs': num_outer_epochs,
        'controls': {
            'identity': resume_identity, 'revenue_objective': revenue_objective,
            'profit_goal': profit_goal, 'seed': seed,
            'outer_steps_per_epoch': outer_steps_per_epoch, 'lr_theta': lr_theta,
            'num_inner_steps': num_inner_steps, 'lr_w': lr_w,
            'inner_opt_reset_each_outer': inner_opt_reset_each_outer,
            'policy_reset_interval': policy_reset_interval,
            'include_simple_deviations': include_simple_deviations,
            'num_samples_train': num_samples_train, 'alpha_weight': alpha_weight,
            'beta_weight': beta_weight, 'early_stop': early_stop, 'tol': tol,
            'patience': patience, 'checkpoint_thresh': checkpoint_thresh,
            'checkpoint_interval': checkpoint_interval,
        },
    }
    if extend_from_epochs is not None and resume_state is None:
        raise ValueError('A phase extension requires exact saved state')
    if resume_state is not None:
        if not isinstance(resume_identity, str) or not resume_identity:
            raise ValueError('Exact resumption requires a source/config identity')
        if init_mechanism_params is not None or init_policy_params is not None:
            raise ValueError('Cannot replace parameters of an exact resumed state')
        validate_resume_state(resume_state, training_contract,
                              extend_from_epochs=extend_from_epochs)

    # ---- Init mechanism params ----
    init_state = auction.get_initial_state(next(prng))
    if init_mechanism_params is not None:
        mechanism_params = init_mechanism_params
        next(prng)  # consume key to keep PRNGSequence aligned
    else:
        mechanism_params = mechanism_network.init(
            next(prng),
            init_state,
            jnp.zeros((auction.num_buyers, auction.num_goods)),
            auction,
        )

    # ---- Init policy params ----
    init_belief = auction.get_initial_belief_state(init_state)
    policy_params = policy_network.init(next(prng), init_belief, auction)
    if init_policy_params is not None:
        policy_params = init_policy_params

    # ---- Optimizers ----
    theta_opt = optax.adam(lr_theta)
    theta_opt_state = theta_opt.init(mechanism_params)

    w_opt = optax.adam(lr_w)
    w_opt_state = w_opt.init(policy_params)

    # ---- Core estimators (capture via closure) ----

    @jax.jit
    def deviation_gains(mechanism_params, policy_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        policy = BoundedContPolicy(policy_network, policy_params)
        return BudgetSimulator.estimate_cumulative_deviation_gains(
            auction, mechanism, policy, num_samples_train, key
        )

    @jax.jit
    def cumulative_regret(mechanism_params, policy_params, key):
        return jnp.sum(deviation_gains(mechanism_params, policy_params, key))

    @jax.jit
    def training_deviation_gains(mechanism_params, policy_params, key):
        neural = deviation_gains(mechanism_params, policy_params, key)
        if not include_simple_deviations:
            return neural
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        candidates = [neural, jnp.zeros_like(neural)]
        for mode in ("low", "high", "half"):
            candidates.append(BudgetSimulator.estimate_cumulative_deviation_gains(
                auction, mechanism, SimpleDeviationPolicy(mode), num_samples_train, key))
        # Every buyer chooses independently; the fixed zero candidate is truth.
        # This sampled maximum is a training objective, not a final IC estimate.
        return jnp.max(jnp.stack(candidates), axis=0)

    @jax.jit
    def truthful_profit(mechanism_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        prof = BudgetSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples_train, key
        )
        return prof

    @jax.jit
    def ir_loss(mechanism_params, key):
        mechanism = BudgetMechanism(mechanism_network, mechanism_params)
        rewards = BudgetSimulator.estimate_cumulative_rewards(
            auction, mechanism, num_samples_train * 5, key
        )
        return jnp.sum(jax.nn.relu(-rewards))

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

    @jax.jit
    def maximize_regret_oracle(mechanism_params, policy_params, w_opt_state, key):
        # One compiled loop avoids hundreds of Python/device round trips per epoch.
        def step(_, carry):
            _, p, opt_state, k = carry
            k, sample_key = jax.random.split(k)
            gain, p, opt_state = inner_step(mechanism_params, p, opt_state, sample_key)
            return gain, p, opt_state, k
        gain, p, opt_state, _ = jax.lax.fori_loop(
            0, num_inner_steps, step, (0.0, policy_params, w_opt_state, key))
        return gain, p, opt_state

    # ---- Outer: update theta ----

    @jax.jit
    def total_loss_with_oracle(mechanism_params, policy_params_star, key):
        k1, k2, k3 = jax.random.split(key, 3)
        gains = training_deviation_gains(mechanism_params, policy_params_star, k1)
        reg = jnp.sum(gains)
        profit = truthful_profit(mechanism_params, k2)
        gap = diagnostic_goal - profit
        ir = ir_loss(mechanism_params, k3)

        # Truth is available independently to every buyer. A negative gain
        # for one buyer must never cancel a profitable deviation for another.
        reg_plus = jnp.sum(jax.nn.relu(gains))
        gap_plus = jnp.maximum(0.0, gap) if revenue_objective == 'acceptance' else 0.

        loss = objective_value(gains, profit, mode=revenue_objective,
                               goal=profit_goal, alpha=alpha_weight,
                               beta=beta_weight, ir=ir)
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
        'revenue': [],
        'revenue_objective': revenue_objective,
        'profit_goal_is_training_target': revenue_objective == 'acceptance',
        'training_identity': resume_identity,
    }
    sat_count = 0
    _ckpt_enabled = checkpoint_thresh is not None
    first_epoch = 0
    if resume_state is not None:
        mechanism_params = resume_state["mechanism_params"]
        policy_params = resume_state["policy_params"]
        theta_opt_state = resume_state["theta_opt_state"]
        w_opt_state = resume_state["w_opt_state"]
        prng.replace_internal_state(resume_state["rng_state"])
        logs = resume_state["logs"]
        first_epoch = resume_state["next_epoch"]
        sat_count = resume_state["sat_count"]

    # ---- Training loop ----
    t0 = time.time()
    for outer_epoch in range(first_epoch, num_outer_epochs):

        reset_interval = policy_reset_interval or (1 if inner_opt_reset_each_outer else None)
        if reset_interval and outer_epoch > 0 and outer_epoch % reset_interval == 0:
            init_s = auction.get_initial_state(next(prng))
            init_b = auction.get_initial_belief_state(init_s)
            policy_params = policy_network.init(next(prng), init_b, auction)
            w_opt_state = w_opt.init(policy_params)

        reg_star, policy_params, w_opt_state = maximize_regret_oracle(
            mechanism_params, policy_params, w_opt_state, next(prng)
        )
        policy_params_star = policy_params

        for _ in range(outer_steps_per_epoch):
            measured_mechanism_params = mechanism_params
            loss, (reg, gap, ir, reg_plus, gap_plus), mechanism_params, theta_opt_state = \
                outer_step(mechanism_params, theta_opt_state, policy_params_star, next(prng))

        logs["loss"].append(float(loss))
        logs["reg"].append(float(reg))
        logs["gap"].append(float(gap))
        logs["ir"].append(float(ir))
        logs["reg_plus"].append(float(reg_plus))
        logs["gap_plus"].append(float(gap_plus))
        logs['revenue'].append(float(diagnostic_goal - gap))
        logs["oracle_reg_last"].append(float(reg_star))

        if outer_epoch % print_iter == 0:
            elapsed = time.time() - t0
            print(f"=== Outer Epoch {outer_epoch} ({elapsed:.1f}s) ===")
            print(f"  oracle regret (train):  {float(reg_star):.6f}")
            print(f"  reg (in loss):          {float(reg):.6f}   reg_+={float(reg_plus):.6f}")
            if revenue_objective == 'acceptance':
                print(f"  gap (goal-profit):      {float(gap):.6f}   gap_+={float(gap_plus):.6f}")
            else:
                print(f"  direct revenue:         {float(diagnostic_goal - gap):.6f}")
            print(f"  ir_loss:                {float(ir):.6f}")
            print(f"  total loss:             {float(loss):.6f}")

        # ---- Checkpoint recording (for binary search) ----
        if (_ckpt_enabled
                and (outer_epoch % checkpoint_interval == 0)
                and (float(reg_plus) <= checkpoint_thresh)
                and (float(ir) <= checkpoint_thresh)):
            logs["checkpoint_mech"].append({
                "epoch": outer_epoch,
                "reg_train": float(reg_plus),
                "deviation_gain_sum_train": float(reg),
                "ir_train": float(ir),
                "gap_train": float(gap),
                "mechanism_params": measured_mechanism_params,
                "policy_params": policy_params_star,
                "measurement_role": "training_only_pre_update",
            })

        if early_stop:
            ok = (float(reg_plus) <= tol) and (float(ir) <= tol * 10) and (float(gap) <= tol)
            if ok:
                logs["good_mech"].append(
                    (measured_mechanism_params, policy_params_star, float(reg_plus), float(ir), float(gap))
                )
            sat_count = sat_count + 1 if ok else 0
        if checkpoint_callback is not None and (
                (outer_epoch+1) % checkpoint_interval == 0
                or outer_epoch+1 == num_outer_epochs or (early_stop and sat_count >= patience)):
            checkpoint_callback({
                "next_epoch": outer_epoch+1, "mechanism_params": mechanism_params,
                "policy_params": policy_params, "theta_opt_state": theta_opt_state,
                "w_opt_state": w_opt_state, "rng_state": prng.internal_state,
                "sat_count": sat_count, "logs": logs,
                'training_contract': training_contract,
            })
        if early_stop and sat_count >= patience:
            print(f"[EARLY STOP] at epoch {outer_epoch}")
            break

    logs["mechanism_params"] = mechanism_params
    logs["policy_params"] = policy_params
    # A final checkpoint must carry metrics evaluated at its own parameter tree.
    _, aux = total_loss_with_oracle(mechanism_params, policy_params, next(prng))
    logs["final_train_metrics"] = dict(zip(
        ("reg", "gap", "ir", "reg_plus", "gap_plus"), map(float, aux)))
    logs['final_train_metrics']['revenue'] = float(diagnostic_goal - aux[1])
    return logs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Budget Auction Nested-GD Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=list(VARIANT_NETWORK.keys()), default=None,
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
