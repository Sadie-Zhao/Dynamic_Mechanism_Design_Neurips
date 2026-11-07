"""Versioned single-good correction: zero allocation freezes the buyer's type.

Historical discrete_single.py and its multi-good users remain unchanged.
This is not a claim that Pavan et al.'s optimal-index theorem applies.
"""
import numpy as np
import jax.numpy as jnp

from bandit_auction.envs.discrete_single import create_discrete_single_auction

ENVIRONMENT_VERSION = "discrete_single_frozen_v1"


def frozen_transition_probability(auction, time_step, current_type_i, alloc_i,
                                  next_type_i):
    """Normalized transition probability for a valid single-good allocation.

    P_x(v'|v) = (1-x) 1[v'=v] + x K(v'|v),
    K(v'|v) proportional to exp(-0.5 |v'-v|).

    The caller supplies x in [0,1] and on-grid current/next values. No clipping
    hides invalid allocations. For fractional x this is an explicit smooth
    marginal interpolation, not a sampled common winner. The unchanged core
    samples buyers independently conditional on their allocation intensities.
    Historical consumers may normalize this already-normalized row again.
    """
    del time_step
    values = auction.buyer_types[0, :, 0]
    current = current_type_i[0]
    following = next_type_i[0]
    x = alloc_i[0]
    winner_weights = jnp.exp(-0.5 * jnp.abs(values - current))
    winner_probability = jnp.sum(
        jnp.where(values == following, winner_weights, 0.0)
    ) / jnp.sum(winner_weights)
    stay = (following == current).astype(winner_weights.dtype)
    return (1.0 - x) * stay + x * winner_probability


def create_discrete_single_frozen_auction(
    n_buyers=2, n_goods=1, horizon=5, discount=0.9, type_values=None,
    allocation_cost=0.1, d_pub=1, d_priv=None, seed=42,
):
    """Construct the corrected environment through an explicit new entry point.

    Distribution, support, horizon, costs and winner kernel match the historical
    factory. Training and every audit must explicitly use this same factory.
    Do not use historical audit CLIs, which reconstruct the old environment.
    """
    if n_goods != 1:
        raise ValueError("discrete_single_frozen_v1 supports exactly one good")
    values = [1.0, 2.0, 3.0] if type_values is None else list(type_values)
    grid = np.asarray(values, dtype=np.float32)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or len(np.unique(grid)) != len(grid) or not np.all(np.diff(grid) > 0)):
        raise ValueError("type_values must be finite, distinct and increasing in float32")
    if n_buyers < 1 or horizon < 1 or not 0 < discount <= 1:
        raise ValueError("positive buyers/horizon and discount in (0,1] required")
    auction = create_discrete_single_auction(
        n_buyers=n_buyers, n_goods=1, horizon=horizon, discount=discount,
        type_values=values, allocation_cost=allocation_cost, d_pub=d_pub,
        d_priv=d_priv, seed=seed,
    )
    return auction.replace(get_transition_prob=frozen_transition_probability)
