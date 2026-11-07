"""
static_myopic.py — Baseline B1: Static myopic budget-feasible auction.

At each stage, independently of history:
  - Allocate each good to the highest bidder among buyers with beta > 0
  - Charge second-price-like payment, capped by feasibility:
      p_i = min(2nd_highest_report, beta_i, alpha_i + v_i * x_i)

This baseline requires no training. It is evaluated directly via simulation.
The allocation is deterministic (highest report wins each good).

For multi-good: each good is allocated independently to the highest bidder.
"""

import jax
import jax.numpy as jnp
import functools
from typing import Dict

from budget_auction.core import (
    BudgetAuction,
    BudgetState,
    BudgetSimulator,
)


# ---------------------------------------------------------------------------
# Myopic mechanism (non-differentiable, evaluation only)
# ---------------------------------------------------------------------------

@jax.jit
def myopic_allocate_and_pay(
    budget_state: BudgetState,
    report: jnp.ndarray,      # (B, G) — true types (baseline uses truthful)
    auction: BudgetAuction,
) -> tuple:
    """
    Allocation and payment for static myopic baseline.

    Per good g:
      - Only buyers with beta > 0 are eligible
      - Allocate to buyer with highest report among eligible buyers
      - Payment: second-price style clipped by feasibility cap

    Returns:
        allocation: (B, G)
        payment:    (B,)
    """
    B, G = auction.num_buyers, auction.num_goods

    def allocate_one_good(g):
        bids = report[:, g]                               # (B,)
        eligible = (budget_state.beta > 1e-6).astype(jnp.float32)  # (B,)
        # Set ineligible bids to -inf
        masked_bids = bids * eligible + (1.0 - eligible) * (-1e9)

        # Winner: argmax among eligible
        winner = jnp.argmax(masked_bids)                  # scalar

        # Second-price payment: highest bid among non-winners who are eligible
        # mask out winner
        non_winner_mask = jnp.ones(B).at[winner].set(0.0) * eligible
        second_price = jnp.max(bids * non_winner_mask + (1.0 - non_winner_mask) * (-1e9))
        second_price = jnp.maximum(second_price, 0.0)    # clip to 0

        # Allocation: one-hot
        alloc_g = jax.nn.one_hot(winner, B)               # (B,)
        return alloc_g, second_price * alloc_g, winner

    # Allocate each good independently
    allocs = []
    raw_pays = []
    for g in range(G):
        alloc_g, raw_pay_g, _ = allocate_one_good(g)
        allocs.append(alloc_g)
        raw_pays.append(raw_pay_g)

    allocation = jnp.stack(allocs, axis=-1)               # (B, G)
    # Sum raw payment contributions across goods
    raw_payment = jnp.sum(jnp.stack(raw_pays, axis=-1), axis=-1)  # (B,)

    # Apply feasibility cap
    utility_now = jnp.sum(allocation * report, axis=-1)  # (B,) using reports=true types
    cap = jnp.maximum(
        0.0,
        jnp.minimum(budget_state.beta, budget_state.alpha + utility_now)
    )
    payment = jnp.minimum(raw_payment, cap)

    return allocation, payment


# ---------------------------------------------------------------------------
# Evaluation via rollout
# ---------------------------------------------------------------------------

class StaticMyopicMechanism:
    """Thin wrapper to make the myopic rule compatible with BudgetSimulator."""

    def __call__(self, state: BudgetState, report: jnp.ndarray,
                 auction: BudgetAuction):
        return myopic_allocate_and_pay(state, report, auction)


# Register as an empty JAX pytree so it can be passed through jit-compiled functions
jax.tree_util.register_pytree_node(
    StaticMyopicMechanism,
    lambda m: ([], None),
    lambda aux, leaves: StaticMyopicMechanism(),
)


def run_static_myopic_baseline(
    auction: BudgetAuction,
    num_samples: int = 256,
    num_keys: int = 16,
    base_seed: int = 0,
) -> Dict:
    """
    Evaluate the static myopic baseline.

    Returns a metrics dict with profit, ir_loss, and violation rates.
    """
    mechanism = StaticMyopicMechanism()
    keys = jax.random.split(jax.random.PRNGKey(base_seed), num_keys)

    profits, ir_losses, bv_rates, iv_rates = [], [], [], []
    for k in keys:
        k1, k2, k3 = jax.random.split(k, 3)
        profit = BudgetSimulator.estimate_cumulative_profit(
            auction, mechanism, num_samples, k1
        )
        rewards = BudgetSimulator.estimate_cumulative_rewards(
            auction, mechanism, num_samples, k2
        )
        ir_loss = float(jnp.sum(jnp.abs(rewards)))
        bv, iv = BudgetSimulator.estimate_violation_rates(
            auction, mechanism, num_samples, k3
        )
        profits.append(float(profit))
        ir_losses.append(ir_loss)
        bv_rates.append(bv)
        iv_rates.append(iv)

    profits = jnp.array(profits)
    ir_losses = jnp.array(ir_losses)

    return {
        "baseline": "static_myopic",
        "profit_mean": float(jnp.mean(profits)),
        "profit_std": float(jnp.std(profits)),
        "ir_loss_mean": float(jnp.mean(jnp.array(ir_losses))),
        "budget_viol_rate": float(jnp.mean(jnp.array(bv_rates))),
        "ir_viol_rate": float(jnp.mean(jnp.array(iv_rates))),
        "num_keys": num_keys,
        "num_samples": num_samples,
    }
