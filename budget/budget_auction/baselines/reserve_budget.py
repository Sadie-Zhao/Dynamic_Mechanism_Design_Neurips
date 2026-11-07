"""
reserve_budget.py — Baseline B2: Reserve-based public-state heuristic.

At each stage, the reserve price depends on the current budget state:
    reserve_i = reserve_multiplier * (beta_i / B_i)

Higher reserve when buyer has more budget remaining; lower reserve as budget depletes.

Allocation: highest bidder above their reserve wins each good.
Payment: reserve price (or clipped to feasibility cap).

This is a simple hand-designed budget-aware dynamic baseline. No training required.
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
# Reserve-based mechanism
# ---------------------------------------------------------------------------

@jax.jit
def reserve_allocate_and_pay(
    budget_state: BudgetState,
    report: jnp.ndarray,           # (B, G) true types under truthful play
    auction: BudgetAuction,
    reserve_multiplier: float = 0.5,
) -> tuple:
    """
    Reserve-based budget-aware allocation.

    For each good g independently:
      - Compute per-buyer reserve: r_i = reserve_multiplier * (beta_i / B_i)
      - Allocate to highest bidder strictly above their reserve and with beta > 0
      - Charge the reserve price, clipped by feasibility cap

    Returns:
        allocation: (B, G)
        payment:    (B,)
    """
    B, G = auction.num_buyers, auction.num_goods

    # Per-buyer reserve in [0, reserve_multiplier]
    reserves = reserve_multiplier * (budget_state.beta / auction.initial_budgets)  # (B,)

    def allocate_one_good(g):
        bids = report[:, g]                                   # (B,)
        eligible = (
            (budget_state.beta > 1e-6) &
            (bids > reserves)
        ).astype(jnp.float32)                                 # (B,)

        masked_bids = bids * eligible + (1.0 - eligible) * (-1e9)
        winner = jnp.argmax(masked_bids)
        # Only allocate if at least one eligible buyer
        any_eligible = jnp.any(bids > reserves)
        alloc_g = jax.nn.one_hot(winner, B) * any_eligible.astype(jnp.float32)

        # Payment: the winner pays their own reserve (not VCG, but simple)
        winner_reserve = reserves[winner] * any_eligible.astype(jnp.float32)
        pay_g = alloc_g * winner_reserve
        return alloc_g, pay_g

    allocs, raw_pays = [], []
    for g in range(G):
        alloc_g, pay_g = allocate_one_good(g)
        allocs.append(alloc_g)
        raw_pays.append(pay_g)

    allocation = jnp.stack(allocs, axis=-1)                   # (B, G)
    raw_payment = jnp.sum(jnp.stack(raw_pays, axis=-1), axis=-1)  # (B,)

    # Feasibility cap
    utility_now = jnp.sum(allocation * report, axis=-1)
    cap = jnp.maximum(
        0.0,
        jnp.minimum(budget_state.beta, budget_state.alpha + utility_now)
    )
    payment = jnp.minimum(raw_payment, cap)

    return allocation, payment


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class ReserveBudgetMechanism:
    """Wrapper to make reserve-based rule compatible with BudgetSimulator."""

    def __init__(self, reserve_multiplier: float = 0.5):
        self.reserve_multiplier = reserve_multiplier

    def __call__(self, state: BudgetState, report: jnp.ndarray,
                 auction: BudgetAuction):
        return reserve_allocate_and_pay(state, report, auction, self.reserve_multiplier)


# Register as a JAX pytree (multiplier is static aux, no array leaves)
jax.tree_util.register_pytree_node(
    ReserveBudgetMechanism,
    lambda m: ([], m.reserve_multiplier),
    lambda aux, leaves: ReserveBudgetMechanism(aux),
)


def run_reserve_budget_baseline(
    auction: BudgetAuction,
    reserve_multiplier: float = 0.5,
    num_samples: int = 256,
    num_keys: int = 16,
    base_seed: int = 0,
) -> Dict:
    """
    Evaluate the reserve-based budget baseline.

    Returns a metrics dict with profit, ir_loss, and violation rates.
    """
    mechanism = ReserveBudgetMechanism(reserve_multiplier)
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

    return {
        "baseline": f"reserve_budget_{reserve_multiplier}",
        "reserve_multiplier": reserve_multiplier,
        "profit_mean": float(jnp.mean(jnp.array(profits))),
        "profit_std": float(jnp.std(jnp.array(profits))),
        "ir_loss_mean": float(jnp.mean(jnp.array(ir_losses))),
        "budget_viol_rate": float(jnp.mean(jnp.array(bv_rates))),
        "ir_viol_rate": float(jnp.mean(jnp.array(iv_rates))),
        "num_keys": num_keys,
        "num_samples": num_samples,
    }
