"""Experiment: Discrete types, multi-good (G=2). No closed-form benchmark."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np

from bandit_auction.core import (
    BanditAuction, bandit_init_state, bandit_init_belief_state_discrete,
    discrete_transition_fn, make_discrete_sampler, SEED,
)
from bandit_auction.envs.discrete_single import _discrete_transition_prob


def create_discrete_multi_auction(
    n_buyers: int = 2,
    n_goods: int = 2,
    horizon: int = 5,
    discount: float = 0.9,
    type_values: list = None,  # e.g. [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
    type_values_per_good: list = None,  # e.g. [1.0, 2.0, 3.0] repeated per good
    allocation_cost: float = 0.1,
    d_pub: int = 1,
    d_priv: int = 1,
    seed: int = SEED,
) -> BanditAuction:
    """
    For multi-good discrete, each buyer's type is a vector in R^G.
    type_values_per_good: list of values for each good (same values for both goods by default).
    All combinations across goods form the per-buyer type set.
    """
    if type_values_per_good is None:
        type_values_per_good = [1.0, 2.0, 3.0]

    K_per_good = len(type_values_per_good)

    # Per-buyer individual type space: K^G combinations
    per_buyer_types_list = list(itertools.product(type_values_per_good, repeat=n_goods))
    K_per_buyer = len(per_buyer_types_list)

    # buyer_types: (B, K_per_buyer, G)
    buyer_types = jnp.array([
        [list(t) for t in per_buyer_types_list]
        for _ in range(n_buyers)
    ], dtype=jnp.float32)  # (B, K^G, G)

    # Joint type profiles: (K_per_buyer^B, B, G)
    all_joint = list(itertools.product(range(K_per_buyer), repeat=n_buyers))
    type_profiles_np = np.array([
        [list(per_buyer_types_list[all_joint[j][i]]) for i in range(n_buyers)]
        for j in range(len(all_joint))
    ], dtype=np.float32)  # (K_per_buyer^B, B, G)
    type_profiles = jnp.array(type_profiles_np)
    K_profiles = len(all_joint)

    sample_fn = make_discrete_sampler(type_profiles)

    alloc_cost = jnp.full((n_buyers, n_goods), allocation_cost, dtype=jnp.float32)
    low = jnp.full((n_buyers, n_goods), min(type_values_per_good), dtype=jnp.float32)
    high = jnp.full((n_buyers, n_goods), max(type_values_per_good), dtype=jnp.float32)

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
