"""
single_buyer_indep_env.py — Experiment 0: Single buyer, independent types.

Setting:
  - 1 buyer, G goods per stage, T stages
  - Types v_t ~ F_t, drawn independently across stages (no Markov dependence)
  - Default: F_t = Uniform[0, 1] for all t (stationary i.i.d.)
  - Optional: stage-specific means for non-stationary but still independent process
  - Budget B_1 = 3.0
  - Allocation cost = 0 (single buyer: externality only from budget, not competition)

Purpose: Compare finite mechanism representations under independent private
types. A representation theorem for an unrestricted mechanism class does not
establish equal optima for these finite neural architectures. Similar observed
revenues are empirical evidence only and do not isolate the cause of gaps in
other environments.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, List

from budget_auction.core import (
    BudgetAuction,
    BudgetState,
    BeliefState,
    budget_init_state,
    budget_init_belief_state,
    build_pub_net,
    build_priv_net,
    make_box_sampler,
    SEED,
)


# ---------------------------------------------------------------------------
# Type sampler: uniform on [types_low, types_high] — independent across stages
# ---------------------------------------------------------------------------

def make_single_buyer_type_sampler(
    types_low: jnp.ndarray,   # (1, G)
    types_high: jnp.ndarray,  # (1, G)
):
    """Returns initial-stage type sampler (uniform on [low, high])."""
    def sample_types_fn(key: jax.random.PRNGKey) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=types_low.shape)
        return types_low + (types_high - types_low) * u
    return sample_types_fn


# ---------------------------------------------------------------------------
# Transition function: independent re-draw each stage (no Markov correlation)
# ---------------------------------------------------------------------------

def single_buyer_indep_transition_fn(
    auction: BudgetAuction,
    time_step: int,
    types: jnp.ndarray,              # (1, G) current types (unused)
    allocation_history: jnp.ndarray, # (T, 1, G) (unused for independent dynamics)
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Independent type re-draw. v_{t+1} ~ Uniform[low, high], independent of everything.
    This implements F_1 = F_2 = ... = F_T = Uniform[0,1] (stationary i.i.d.).
    For non-stationary F_t (e.g. Beta distributions per stage), override this function.
    """
    types_low = auction.types_low
    types_high = auction.types_high
    u = jax.random.uniform(key, shape=types.shape)
    return types_low + (types_high - types_low) * u


def get_transition_prob_uniform(
    auction: BudgetAuction,
    time_step: int,
    current_type_profile: jnp.ndarray,
    allocation_history: jnp.ndarray,
    next_type_profile: jnp.ndarray,
) -> float:
    """Uniform density: 1 / volume of type box."""
    types_low = auction.types_low
    types_high = auction.types_high
    volume = jnp.prod(types_high - types_low)
    return 1.0 / jnp.maximum(volume, 1e-12)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_single_buyer_indep_auction(
    n_goods: int = 1,
    horizon: int = 5,
    discount: float = 0.9,
    initial_budget: float = 3.0,
    types_low: float = 0.0,
    types_high: float = 1.0,
    d_pub: int = 64,
    d_priv: int = 128,
    seed: int = SEED,
) -> BudgetAuction:
    """
    Create a BudgetAuction for Experiment 0: single buyer, independent types.

    The finite Variant A network is a restricted parameterization; its learned
    revenue is not an unrestricted GBAM optimality certificate.

    Args:
        n_goods: number of goods per stage (default 1)
        horizon: number of stages T (default 5)
        discount: discount factor (default 0.9)
        initial_budget: B_1 for the single buyer (default 3.0)
        types_low: lower bound of type space (default 0.0)
        types_high: upper bound of type space (default 1.0)
        d_pub: public GRU embedding dimension
        d_priv: private GRU embedding dimension
        seed: random seed for network initialization
    """
    n_buyers = 1
    B, G = n_buyers, n_goods
    initial_budgets = jnp.ones((B,)) * initial_budget

    types_low_arr = jnp.full((B, G), types_low)
    types_high_arr = jnp.full((B, G), types_high)

    # Observation dimensions
    # Legacy public encoder: time, allocations, public payments and budget.
    d_obs_pub = 1 + B * G + 2 * B
    # d_obs_priv = 1 (t/T) + G (type) + G (report) + 1 (alpha_i) + 1 (beta_i)
    d_obs_priv = 1 + 2 * G + 2

    pub_net = build_pub_net(d_pub=d_pub)
    priv_net = build_priv_net(d_priv=d_priv, num_goods=G)

    prng = hk.PRNGSequence(seed)
    pub_params = pub_net.init(
        next(prng),
        jnp.zeros((d_pub,), jnp.float32),
        jnp.zeros((d_obs_pub,), jnp.float32),
    )
    priv_params = priv_net.init(
        next(prng),
        jnp.zeros((d_priv,), jnp.float32),
        jnp.zeros((d_obs_priv,), jnp.float32),
        jnp.zeros((d_pub,), jnp.float32),
    )

    sample_types_fn = make_single_buyer_type_sampler(types_low_arr, types_high_arr)

    return BudgetAuction(
        num_horizons=horizon,
        num_buyers=B,
        num_goods=G,
        init_state_function=budget_init_state,
        init_belief_state_function=budget_init_belief_state,
        get_transition_prob=get_transition_prob_uniform,
        transition_function=single_buyer_indep_transition_fn,
        discount=discount,
        allocation_cost=jnp.zeros((B, G)),
        sample_types_fn=sample_types_fn,
        types_low=types_low_arr,
        types_high=types_high_arr,
        initial_budgets=initial_budgets,
        belief_pub_net=pub_net,
        belief_pub_params=pub_params,
        belief_priv_net=priv_net,
        belief_priv_params=priv_params,
        d_pub=d_pub,
        d_priv=d_priv,
    )
