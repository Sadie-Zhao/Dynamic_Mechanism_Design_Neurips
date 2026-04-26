"""Experiment: Continuous types, single good (G=1). Benchmark: Pavan et al. 2014."""
import jax.numpy as jnp
import haiku as hk

from bandit_auction.core import (
    BanditAuction, bandit_init_state, bandit_init_belief_state_continuous,
    continuous_transition_fn, make_gaussian_sampler,
    build_pub_net, build_priv_net, SEED,
)


def create_cont_single_auction(
    n_buyers: int = 2,
    n_goods: int = 1,
    horizon: int = 5,
    discount: float = 0.9,
    type_mean: float = 2.5,
    type_std: float = 0.1,
    types_low: float = 0.0,
    types_high: float = 5.0,
    allocation_cost: float = 0.1,
    sigma_no: float = 1.0,
    trans_lambda: float = 1.0,
    d_pub: int = 64,
    d_priv: int = 128,
    seed: int = SEED,
) -> BanditAuction:
    import jax
    prng = hk.PRNGSequence(seed)

    low = jnp.full((n_buyers, n_goods), types_low, dtype=jnp.float32)
    high = jnp.full((n_buyers, n_goods), types_high, dtype=jnp.float32)
    mean = jnp.full((n_buyers, n_goods), type_mean, dtype=jnp.float32)
    std = jnp.full((n_buyers, n_goods), type_std, dtype=jnp.float32)

    sample_fn = make_gaussian_sampler(mean, std)
    alloc_cost = jnp.full((n_buyers, n_goods), allocation_cost, dtype=jnp.float32)

    # Build GRU belief networks
    pub_net = build_pub_net(d_pub)
    priv_net = build_priv_net(d_priv, n_goods)

    # Dummy init to get params
    dummy_c = jnp.zeros((d_pub,))
    dummy_o_pub = jnp.zeros((1 + n_buyers * n_goods,))  # t/T + alloc_t.flatten()
    pub_params = pub_net.init(next(prng), dummy_c, dummy_o_pub)

    dummy_h = jnp.zeros((d_priv,))
    dummy_o_priv = jnp.zeros((1 + 2 * n_goods,))  # t/T + type(G) + report(G)
    priv_params = priv_net.init(next(prng), dummy_h, dummy_o_priv, dummy_c)

    return BanditAuction(
        num_horizons=horizon,
        num_buyers=n_buyers,
        num_goods=n_goods,
        init_state_function=bandit_init_state,
        init_belief_state_function=bandit_init_belief_state_continuous,
        transition_function=continuous_transition_fn,
        get_transition_prob=None,
        discount=discount,
        allocation_cost=alloc_cost,
        sample_types_fn=sample_fn,
        types_low=low,
        types_high=high,
        buyer_types=None,
        type_profiles=None,
        num_type_profiles=0,
        belief_pub_net=pub_net,
        belief_pub_params=pub_params,
        belief_priv_net=priv_net,
        belief_priv_params=priv_params,
        d_pub=d_pub,
        d_priv=d_priv,
        trans_sigma_no=sigma_no,
        trans_lambda=trans_lambda,
    )
