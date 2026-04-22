"""
run_penalty.py — Direct penalty method for bandit auction mechanism learning.

Objective (nested GD, direct penalty form):
    Outer: min_theta  -Profit(theta) + alpha * [Regret(theta, w*)]_+ + beta * IR(theta)
    Inner: max_w      Regret(theta, w)

This differs from run_acceptance.py in that there is no profit_goal: the mechanism
directly maximises profit subject to soft IC and IR penalties.

Usage:
    python -m bandit_auction.train.run_penalty \\
        --config bandit_auction/configs/cont_single_penalty.yaml \\
        --seed 0
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

from bandit_auction.core import (
    AuctionSimulator, BoundedContPolicy, DiscretePolicy,
    build_bounded_policy_net, build_discrete_policy_net, SEED,
)
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network

EXPERIMENT_ENVS = {
    "cont_single": ("bandit_auction.envs.cont_single", "create_cont_single_auction"),
    "cont_multi": ("bandit_auction.envs.cont_multi", "create_cont_multi_auction"),
    "discrete_single": ("bandit_auction.envs.discrete_single", "create_discrete_single_auction"),
    "discrete_multi": ("bandit_auction.envs.discrete_multi", "create_discrete_multi_auction"),
}


def nested_GD_penalty(
    auction,
    mechanism_network,
    policy_network,

    # Outer loop
    num_outer_epochs: int = 2000,
    outer_steps_per_epoch: int = 1,
    lr_theta: float = 1e-4,

    # Inner loop
    num_inner_steps: int = 400,
    lr_w: float = 1e-3,
    inner_opt_reset_each_outer: bool = True,

    # Monte Carlo
    num_samples_train: int = 64,

    # Loss weights: L = -Profit + alpha*[Regret]+ + beta*IR
    alpha_weight: float = 1.0,
    beta_weight: float = 1.0,

    # Diagnostics
    print_iter: int = 50,

    # Checkpointing
    checkpoint_thresh: float = None,
    checkpoint_interval: int = 50,

    # Discrete vs continuous
    is_discrete: bool = False,

    seed: int = SEED,
) -> Dict:
    """
    Nested gradient descent with direct penalty objective.

    OUTER: min_theta  -Profit(theta) + alpha*[Regret(theta,w*)]_+ + beta*IR(theta)
    INNER: max_w      Regret(theta, w)

    Returns logs dict with keys:
        loss, profit, reg, ir, reg_plus, oracle_reg_last,
        checkpoint_mech (list, if checkpoint_thresh set)
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

    # ---- Belief update fn ----
    update_belief_fn = (
        auction.get_next_belief_state_discrete if is_discrete
        else auction.get_next_belief_state_continuous
    )

    # ---- Core estimators ----

    @jax.jit
    def cumulative_regret(mechanism_params, policy_params, key):
        mechanism = Mechanism(mechanism_network, mechanism_params)
        policy_cls = DiscretePolicy if is_discrete else BoundedContPolicy
        policy = policy_cls(policy_network, policy_params)
        return AuctionSimulator.estimate_cumulative_regret(
            auction, mechanism, policy, num_samples_train, key, update_belief_fn
        )

    @jax.jit
    def cumulative_profit(mechanism_params, key):
        mechanism = Mechanism(mechanism_network, mechanism_params)
        return AuctionSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples_train, key
        )

    @jax.jit
    def ir_loss(mechanism_params, key):
        mechanism = Mechanism(mechanism_network, mechanism_params)
        rewards = AuctionSimulator.estimate_relu_cumulative_rewards(
            auction, mechanism, num_samples_train * 5, key
        )
        return jnp.sum(rewards)

    # ---- Inner: gradient ascent on w ----

    @jax.jit
    def inner_step(mechanism_params, policy_params, w_opt_state, key):
        reg, grad_w = jax.value_and_grad(cumulative_regret, argnums=1)(
            mechanism_params, policy_params, key
        )
        # Ascent: negate gradient
        grad_w_ascent = jax.tree_util.tree_map(lambda g: -g, grad_w)
        updates, w_opt_state = w_opt.update(grad_w_ascent, w_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return reg, policy_params, w_opt_state

    def maximize_regret_oracle(mechanism_params, policy_params, w_opt_state, steps):
        reg_last = 0.0
        for _ in range(steps):
            reg_last, policy_params, w_opt_state = inner_step(
                mechanism_params, policy_params, w_opt_state, next(prng)
            )
        return reg_last, policy_params, w_opt_state

    # ---- Outer: update theta ----
    # Direct penalty: L = -Profit + alpha*[Regret]+ + beta*IR

    @jax.jit
    def total_loss(mechanism_params, policy_params_star, key):
        k1, k2, k3 = jax.random.split(key, 3)
        reg = cumulative_regret(mechanism_params, policy_params_star, k1)
        prof = cumulative_profit(mechanism_params, k2)
        ir = ir_loss(mechanism_params, k3)

        reg_plus = jnp.maximum(0.0, reg)
        loss = -prof + alpha_weight * reg_plus + beta_weight * ir
        return loss, (prof, reg, ir, reg_plus)

    @jax.jit
    def outer_step(mechanism_params, theta_opt_state, policy_params_star, key):
        (loss, aux), grad_theta = jax.value_and_grad(
            total_loss, argnums=0, has_aux=True
        )(mechanism_params, policy_params_star, key)
        updates, theta_opt_state = theta_opt.update(grad_theta, theta_opt_state, mechanism_params)
        mechanism_params = optax.apply_updates(mechanism_params, updates)
        return loss, aux, mechanism_params, theta_opt_state

    # ---- Logs ----
    logs = {
        "loss": [], "profit": [], "reg": [], "ir": [],
        "reg_plus": [], "oracle_reg_last": [],
        "checkpoint_mech": [],
    }
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
            loss, (prof, reg, ir, reg_plus), mechanism_params, theta_opt_state = \
                outer_step(mechanism_params, theta_opt_state, policy_params_star, next(prng))

        logs["loss"].append(float(loss))
        logs["profit"].append(float(prof))
        logs["reg"].append(float(reg))
        logs["ir"].append(float(ir))
        logs["reg_plus"].append(float(reg_plus))
        logs["oracle_reg_last"].append(float(reg_star))

        if outer_epoch % print_iter == 0:
            elapsed = time.time() - t0
            print(f"=== Outer Epoch {outer_epoch} ({elapsed:.1f}s) ===")
            print(f"  oracle regret (train):  {float(reg_star):.6f}")
            print(f"  reg (in loss):          {float(reg):.6f}   reg_+={float(reg_plus):.6f}")
            print(f"  profit:                 {float(prof):.6f}")
            print(f"  ir_loss:                {float(ir):.6f}")
            print(f"  total loss:             {float(loss):.6f}")

        if (_ckpt_enabled
                and (outer_epoch % checkpoint_interval == 0)
                and (float(reg) <= checkpoint_thresh)
                and (float(ir) <= checkpoint_thresh)):
            logs["checkpoint_mech"].append({
                "epoch": outer_epoch,
                "reg_train": float(reg),
                "ir_train": float(ir),
                "profit_train": float(prof),
                "mechanism_params": mechanism_params,
                "policy_params": policy_params_star,
            })

    logs["mechanism_params"] = mechanism_params
    logs["policy_params"] = policy_params
    return logs


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
    parser = argparse.ArgumentParser(description="Bandit Auction — Direct Penalty Training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed
    experiment = cfg["experiment"]
    is_discrete = experiment.startswith("discrete")
    output_dir = args.output_dir or cfg.get(
        "output_dir", f"outputs/{experiment}/penalty/seed{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    cfg_copy = dict(cfg)
    cfg_copy["seed"] = seed
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg_copy, f)

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

    logs = nested_GD_penalty(
        auction=auction,
        mechanism_network=mechanism_network,
        policy_network=policy_network,
        num_outer_epochs=cfg.get("num_outer_epochs", 2000),
        outer_steps_per_epoch=cfg.get("outer_steps_per_epoch", 1),
        lr_theta=cfg.get("lr_theta", 1e-4),
        num_inner_steps=cfg.get("num_inner_steps", 400),
        lr_w=cfg.get("lr_w", 1e-3),
        num_samples_train=cfg.get("num_samples_train", 64),
        alpha_weight=cfg.get("alpha", 1.0),
        beta_weight=cfg.get("beta", 1.0),
        print_iter=cfg.get("print_iter", 50),
        checkpoint_thresh=cfg.get("checkpoint_thresh", None),
        checkpoint_interval=cfg.get("checkpoint_interval", 50),
        is_discrete=is_discrete,
        seed=seed,
    )

    mechanism_params = logs.pop("mechanism_params")
    policy_params = logs.pop("policy_params")

    with open(os.path.join(output_dir, "logs.pkl"), "wb") as f:
        pickle.dump(logs, f)
    with open(os.path.join(output_dir, "mechanism_params.pkl"), "wb") as f:
        pickle.dump(mechanism_params, f)
    with open(os.path.join(output_dir, "policy_params.pkl"), "wb") as f:
        pickle.dump(policy_params, f)

    summary = {
        "final_profit": logs["profit"][-1] if logs["profit"] else None,
        "final_reg": logs["reg"][-1] if logs["reg"] else None,
        "final_ir": logs["ir"][-1] if logs["ir"] else None,
        "seed": seed,
        "experiment": experiment,
        "method": "penalty",
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining complete. Outputs saved to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
