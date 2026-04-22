"""Experiment: Discrete types, single good (G=1). Benchmark: Pavan et al. 2014."""
import itertools
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from bandit_auction.core import (
    BanditAuction, bandit_init_state, bandit_init_belief_state_discrete,
    discrete_transition_fn, make_discrete_sampler, SEED,
)


def _discrete_transition_prob(
    auction: BanditAuction,
    time_step: int,
    current_type_i: jnp.ndarray,   # (G,) current type for buyer i
    alloc_i: jnp.ndarray,           # (G,) allocation for buyer i at time t
    next_type_i: jnp.ndarray,       # (G,) candidate next type
) -> float:
    """P(next_type_i | current_type_i, alloc_i) via allocation-dependent exponential decay.

    decay = sum(alloc_history) / 2, but at transition time we use the accumulated
    allocation count passed via alloc_i (scalar sum of alloc_i acts as proxy).
    For the per-step transition, we use alloc_i directly as the "got allocation" signal.
    """
    # Allocation count accumulator: use alloc_i as the received-allocation flag per good
    alloc_count = jnp.sum(alloc_i)   # scalar in [0, G]
    decay_rate = alloc_count / 2.0

    diff = jnp.sum(jnp.abs(next_type_i - current_type_i))
    log_prob = -decay_rate * diff
    return jnp.exp(log_prob)   # unnormalized; normalized in discrete_transition_fn


def create_discrete_single_auction(
    n_buyers: int = 2,
    n_goods: int = 1,
    horizon: int = 5,
    discount: float = 0.9,
    type_values: list = None,  # e.g. [1.0, 2.0, 3.0]
    allocation_cost: float = 0.1,
    d_pub: int = 1,    # unused in discrete (dummy); kept for BeliefState shape
    d_priv: int = 1,   # unused in discrete (dummy)
    seed: int = SEED,
) -> BanditAuction:
    if type_values is None:
        type_values = [1.0, 2.0, 3.0]

    K_per_buyer = len(type_values)
    type_vals = jnp.array(type_values, dtype=jnp.float32)  # (K,)

    # buyer_types: (B, K, G)
    buyer_types = jnp.stack([
        type_vals.reshape(K_per_buyer, n_goods) for _ in range(n_buyers)
    ])  # (B, K, 1)

    # All joint type profiles: (K^B, B, G)
    all_profiles = list(itertools.product(range(K_per_buyer), repeat=n_buyers))
    type_profiles_np = np.array([
        [[type_values[all_profiles[j][i]] for _ in range(n_goods)]
         for i in range(n_buyers)]
        for j in range(len(all_profiles))
    ], dtype=np.float32)  # (K^B, B, G)
    type_profiles = jnp.array(type_profiles_np)
    K_profiles = len(all_profiles)

    # Sample initial type profile uniformly
    sample_fn = make_discrete_sampler(type_profiles)

    alloc_cost = jnp.full((n_buyers, n_goods), allocation_cost, dtype=jnp.float32)
    low = jnp.full((n_buyers, n_goods), min(type_values), dtype=jnp.float32)
    high = jnp.full((n_buyers, n_goods), max(type_values), dtype=jnp.float32)

    return BanditAuction(
        num_horizons=horizon,
        num_buyers=n_buyers,
        num_goods=n_goods,
        init_state_function=bandit_init_state,
        init_belief_state_function=bandit_init_belief_state_discrete,
        transition_function=discrete_transition_fn,
        get_transition_prob=_discrete_transition_prob,
        discount=discount,
        allocation_cost=alloc_cost,
        sample_types_fn=sample_fn,
        types_low=low,
        types_high=high,
        buyer_types=buyer_types,
        type_profiles=type_profiles,
        num_type_profiles=K_profiles,
        belief_pub_net=None,
        belief_pub_params=None,
        belief_priv_net=None,
        belief_priv_params=None,
        d_pub=d_pub,
        d_priv=d_priv,
        trans_sigma_no=1.0,
        trans_lambda=1.0,
    )
