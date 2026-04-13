"""
private_budget_multigood_env.py — Experiment 2: Multi-good privately correlated budget auction.

Setting:
  - 2 buyers, 2 goods per stage, T=5 stages
  - Types v_t^i in [0,1]^G (multi-dimensional continuous)
  - Type transition: logit-space Gaussian reused verbatim from bandit code
    v_{t+1}^i depends on v_t^i (private) + allocation history (public)
    This "private correlation" is outside Deng 2021's public-history regime
  - Budget B_i = 3.0 per buyer
  - Budget is shared across all goods and stages (one scalar per buyer)
  - Payment is one scalar per buyer per stage (sum over goods)
  - Utility additive across goods: u_t^i = sum_g v_{t,g}^i * x_{t,g}^i - p_t^i

This setting is "beyond Deng 2021" because type transitions depend on private history
(v_t^i), meaning the GBAM public state (alpha, beta) is not theoretically sufficient.
"""

import jax
import jax.numpy as jnp
import haiku as hk

from budget_auction.core import (
    BudgetAuction,
    BudgetState,
    budget_init_state,
    budget_init_belief_state,
    build_pub_net,
    build_priv_net,
    make_box_sampler,
    _normalize_to_unit,
    _logit,
    _inv_logit,
    SEED,
)


# ---------------------------------------------------------------------------
# Type transition: logit-space Gaussian (reused from bandit notebook)
# ---------------------------------------------------------------------------

def get_transition_logprob_bounded_multigood(
    auction: BudgetAuction,
    time_step: int,
    current_type_profile: jnp.ndarray,  # (B, G)
    allocation_history: jnp.ndarray,    # (T, B, G)
    next_type_profile: jnp.ndarray,     # (B, G)
    lam: float = None,
    sigma_no: float = None,
    eps: float = 1e-6,
) -> float:
    """Log-probability of type transition (same as bandit get_transition_logprob_bounded)."""
    lam = getattr(auction, "trans_base_rate", 1.0) if lam is None else lam
    sigma_no = getattr(auction, "trans_sigma_no", 1e-2) if sigma_no is None else sigma_no

    types_low, types_high = auction.types_low, auction.types_high

    past_alloc = jnp.sum(allocation_history, axis=0)   # (B, G)
    got = allocation_history[time_step]                 # (B, G)

    alpha_param = 0.5 * past_alloc
    sigma_got = 1.0 / (lam * (1.0 + alpha_param))
    sigma = (1.0 - got) * sigma_no + got * sigma_got   # (B, G)

    p_t, width = _normalize_to_unit(current_type_profile, types_low, types_high, eps=eps)
    p_tp1, _ = _normalize_to_unit(next_type_profile, types_low, types_high, eps=eps)
    s_t = _logit(p_t)
    s_tp1 = _logit(p_tp1)

    sigma = jnp.maximum(sigma, 1e-8)
    z = (s_tp1 - s_t) / sigma
    logps = -0.5 * z**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)

    logJ = -jnp.log(width) - jnp.log(p_tp1) - jnp.log1p(-p_tp1)

    return jnp.sum(logps + logJ)


def get_transition_prob_multigood(
    auction: BudgetAuction,
    time_step: int,
    current_type_profile: jnp.ndarray,
    allocation_history: jnp.ndarray,
    next_type_profile: jnp.ndarray,
) -> float:
    logp = get_transition_logprob_bounded_multigood(
        auction, time_step, current_type_profile, allocation_history, next_type_profile
    )
    return jnp.exp(logp)


def private_budget_transition_fn(
    auction: BudgetAuction,
    time_step: int,
    types: jnp.ndarray,               # (B, G)
    allocation_history: jnp.ndarray,  # (T, B, G)
    key: jax.random.PRNGKey,
    lam: float = None,
    sigma_no: float = None,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Logit-space Gaussian transition (reused verbatim from bandit notebook).
    Provides PRIVATE correlation: v_{t+1}^i depends on v_t^i via logit mean.
    """
    lam = getattr(auction, "trans_base_rate", 1.0) if lam is None else lam
    sigma_no = getattr(auction, "trans_sigma_no", 1e-2) if sigma_no is None else sigma_no

    types_low, types_high = auction.types_low, auction.types_high

    past_alloc = jnp.sum(allocation_history, axis=0)   # (B, G)
    got = allocation_history[time_step]                 # (B, G)

    alpha_param = 0.5 * past_alloc
    sigma_got = 1.0 / (lam * (1.0 + alpha_param))
    sigma = (1.0 - got) * sigma_no + got * sigma_got   # (B, G)

    p_t, width = _normalize_to_unit(types, types_low, types_high, eps=eps)
    s_t = _logit(p_t)

    noise = jax.random.normal(key, shape=types.shape)
    s_tp1 = s_t + sigma * noise
    p_tp1 = _inv_logit(s_tp1)

    return types_low + width * p_tp1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_private_budget_multigood_auction(
    n_buyers: int = 2,
    n_goods: int = 2,
    horizon: int = 5,
    discount: float = 0.9,
    initial_budget: float = 3.0,
    d_pub: int = 64,
    d_priv: int = 128,
    trans_base_rate: float = 1.0,
    trans_sigma_no: float = 0.01,
    seed: int = SEED,
) -> BudgetAuction:
    """
    Create a BudgetAuction for Experiment 2 (multi-good privately correlated budget auction).

    Args:
        n_buyers: number of buyers (default 2)
        n_goods: number of goods per stage (default 2)
        horizon: number of stages T (default 5)
        discount: discount factor (default 0.9)
        initial_budget: B_i for each buyer (default 3.0)
        d_pub: public GRU embedding dimension
        d_priv: private GRU embedding dimension
        trans_base_rate: lambda for logit-space Gaussian (higher = tighter transitions)
        trans_sigma_no: sigma when good is NOT allocated
        seed: random seed for network initialization
    """
    B, G = n_buyers, n_goods
    initial_budgets = jnp.ones((B,)) * initial_budget

    # Types in [0, 1]^G
    types_low = jnp.zeros((B, G))
    types_high = jnp.ones((B, G))

    # Observation dimensions
    d_obs_pub = 1 + B * G + 2 * B   # t/T + alloc + alpha/B + beta/B
    d_obs_priv = 1 + 2 * G + 2      # t/T + type + report + alpha_i + beta_i

    # Build GRU encoders
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

    # Initial type sampler: uniform [0,1]^(B*G)
    sample_types_fn = make_box_sampler(types_low, types_high)

    # Create auction — attach transition hyperparams as attributes via a thin wrapper
    auction = BudgetAuction(
        num_horizons=horizon,
        num_buyers=B,
        num_goods=G,
        init_state_function=budget_init_state,
        init_belief_state_function=budget_init_belief_state,
        get_transition_prob=get_transition_prob_multigood,
        transition_function=private_budget_transition_fn,
        discount=discount,
        allocation_cost=jnp.zeros((B, G)),
        sample_types_fn=sample_types_fn,
        types_low=types_low,
        types_high=types_high,
        initial_budgets=initial_budgets,
        belief_pub_net=pub_net,
        belief_pub_params=pub_params,
        belief_priv_net=priv_net,
        belief_priv_params=priv_params,
        d_pub=d_pub,
        d_priv=d_priv,
    )

    # Attach transition hyperparams as plain Python attributes (not JAX fields)
    object.__setattr__(auction, "trans_base_rate", trans_base_rate)
    object.__setattr__(auction, "trans_sigma_no", trans_sigma_no)

    return auction
