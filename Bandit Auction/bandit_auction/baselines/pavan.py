"""
pavan.py — Pavan et al. (2014) virtual index mechanism for the single-good bandit auction.

Implements the closed-form dynamic mechanism from:
    Pavan, Segal & Toikka (2014) "Dynamic Mechanism Design: A Myersonian Approach"

This serves as the benchmark for cont_single and discrete_single experiments.

The mechanism:
  At each period t, compute the "virtual index" (generalized virtual value):
    ψ_i(v_it, t) = v_it - (1 - F(v_it|history)) / f(v_it|history)
                   - Σ_{s=t+1}^{T} γ^{s-t} * E[∂ψ_i/∂v_is | v_it, x_it]

  For the logit-space Gaussian transition used in cont_single, the impulse
  response function ∂v_{i,s}/∂v_{i,t} is approximated numerically.

Simplified implementation:
  We use the static Myerson virtual value as a lower bound / approximation:
    ψ_i(v_it) = v_it - (1 - F(v_it)) / f(v_it)
  and allocate to argmax ψ_i subject to ψ_i ≥ 0 (individual rationality).
  Payment is set to the Myerson threshold payment.

  For the full dynamic version with impulse responses, we estimate the
  continuation value numerically via Monte Carlo rollouts.

Usage:
    from bandit_auction.baselines.pavan import (
        eval_myerson_mechanism,
        eval_pavan_mechanism,
        eval_pavan_bandit_auction,
    )
    revenue, surplus = eval_myerson_mechanism(auction, num_samples=512, seed=0)
    revenue, surplus = eval_pavan_bandit_auction(auction, num_samples=500, seed=0)

NOTE on eval_pavan_mechanism vs eval_pavan_bandit_auction:
  eval_pavan_mechanism: general GVV-with-IR approach (Theorem 2), evaluates hazard
    rate at the CURRENT type -- correct for general dynamic mechanisms but
    conceptually wrong for bandit auctions where IR=1 and the IC rent is
    determined by the INITIAL type's hazard rate.
  eval_pavan_bandit_auction: bandit auction-specific implementation of Eq. (15).
    Computes inv_hazard at the initial type theta_i0 (fixed for the episode) and
    uses the Gittins stopping-time index. This is the correct benchmark for the
    bandit auction setting (PST 2014, Section 5, Proposition 2).
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: virtual value functions for cont_single (Gaussian types)
# ---------------------------------------------------------------------------

def _normal_pdf(x, mean, std):
    """Standard normal PDF evaluated at (x - mean) / std."""
    z = (x - mean) / std
    return jnp.exp(-0.5 * z**2) / (std * jnp.sqrt(2 * jnp.pi))


def _normal_cdf_approx(x, mean, std):
    """Approximate normal CDF via erf."""
    z = (x - mean) / (std * jnp.sqrt(2.0))
    return 0.5 * (1.0 + jax.lax.erf(z))


def myerson_virtual_value(v, mean, std, low, high):
    """Myerson virtual value for truncated Gaussian on [low, high].

    ψ(v) = v - (1 - F(v)) / f(v)

    where F, f are the truncated normal CDF/PDF.
    """
    # Un-truncated Gaussian CDF/PDF
    F_hi  = _normal_cdf_approx(high, mean, std)
    F_lo  = _normal_cdf_approx(low,  mean, std)
    denom = F_hi - F_lo

    F_v = (_normal_cdf_approx(v, mean, std) - F_lo) / (denom + 1e-12)
    f_v = _normal_pdf(v, mean, std) / (denom + 1e-12)

    # Clip f_v to avoid division by zero at boundaries
    f_v = jnp.maximum(f_v, 1e-8)
    return v - (1.0 - F_v) / f_v


# ---------------------------------------------------------------------------
# Static Myerson mechanism (per-period, ignoring dynamics)
# ---------------------------------------------------------------------------

def _myerson_allocation_payment(
    types: jnp.ndarray,   # (B,) scalar types per buyer (G=1)
    mean: float,
    std: float,
    low: float,
    high: float,
    alloc_cost: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single-period Myerson mechanism for B buyers, 1 good.

    Returns allocation (B,) in {0,1} and payment (B,) ≥ 0.
    """
    B = types.shape[0]
    vv = jax.vmap(lambda v: myerson_virtual_value(v, mean, std, low, high))(types)

    # Allocate to highest non-negative virtual value buyer
    winner_vv = jnp.max(vv)
    winner_idx = jnp.argmax(vv)
    allocate = (winner_vv >= 0.0) & (winner_vv > alloc_cost)

    alloc = jnp.zeros(B)
    alloc = jnp.where(allocate, alloc.at[winner_idx].set(1.0), alloc)

    # Myerson threshold payment: smallest v_i such that ψ_i(v_i) ≥ max_{j≠i} ψ_j(v_j)
    # For single buyer winning, pay = inv_ψ(max_{j≠i} ψ_j ∨ 0).
    # Simplified: pay = types[second-highest-vv-buyer] if that is the threshold.
    # We use the threshold-payment approximation via the sorted virtual values.
    sorted_vv = jnp.sort(vv)[::-1]  # descending
    # Threshold for winner is 2nd highest virtual value (or 0 if only one non-neg)
    threshold_vv = jnp.where(B > 1, sorted_vv[1], 0.0)
    threshold_vv = jnp.maximum(threshold_vv, 0.0)

    # Invert virtual value: ψ(v) = v - (1-F(v))/f(v). Numerically via bisection.
    # Closed-form approximation: for small deviations use threshold_vv ≈ threshold_v.
    # (Exact inversion requires 1D root finding; we use the simpler approximation
    # ψ^{-1}(x) ≈ x + (1-F(x))/f(x) iterated once from x itself.)
    def approx_inv_vv(x):
        F_x = _normal_cdf_approx(x, mean, std)
        f_x = jnp.maximum(_normal_pdf(x, mean, std), 1e-8)
        return x + (1.0 - F_x) / f_x

    threshold_v = approx_inv_vv(threshold_vv)
    threshold_v = jnp.clip(threshold_v, low, high)

    payment = jnp.where(allocate, threshold_v, 0.0)
    payment_vec = jnp.zeros(B).at[winner_idx].set(payment * float(allocate))

    return alloc, payment_vec


# ---------------------------------------------------------------------------
# Full trajectory rollout: Myerson (static per-period)
# ---------------------------------------------------------------------------

def _rollout_myerson(
    auction,
    key: jax.random.PRNGKey,
    num_samples: int,
    mean: float,
    std: float,
) -> Tuple[float, float]:
    """Roll out the static Myerson mechanism for `num_samples` trajectories.

    Returns (mean_revenue, mean_buyer_surplus).
    """
    B = auction.num_buyers
    G = auction.num_goods  # must be 1
    T = auction.num_horizons
    gamma = auction.discount
    low  = float(auction.types_low[0, 0])
    high = float(auction.types_high[0, 0])
    alloc_cost = float(auction.allocation_cost[0, 0])

    total_revenue = 0.0
    total_surplus = 0.0

    for _ in range(num_samples):
        key, subkey = jax.random.split(key)
        state = auction.get_initial_state(subkey)

        ep_revenue = 0.0
        ep_surplus = 0.0

        for t in range(T):
            types_t = state.current_type[:, 0]  # (B,) scalar per buyer

            alloc, pay = _myerson_allocation_payment(
                types_t, mean, std, low, high, alloc_cost
            )

            reward = types_t * alloc - pay             # (B,)
            profit = jnp.sum(pay) - alloc_cost * jnp.sum(alloc)

            ep_revenue += float(gamma**t * profit)
            ep_surplus += float(gamma**t * jnp.sum(reward))

            # Transition
            key, subkey = jax.random.split(key)
            alloc_2d = alloc.reshape(B, G)
            state = auction.get_next_state(state, types_t.reshape(B, G), alloc_2d, subkey)

        total_revenue += ep_revenue
        total_surplus += ep_surplus

    return total_revenue / num_samples, total_surplus / num_samples


def eval_myerson_mechanism(
    auction,
    num_samples: int = 512,
    seed: int = 0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[float, float]:
    """Evaluate the static Myerson mechanism on the given auction.

    Args:
        auction:     BanditAuction (cont_single or discrete_single, G=1)
        num_samples: Monte Carlo trajectories
        seed:        RNG seed
        mean:        type distribution mean (inferred from auction if None)
        std:         type distribution std  (inferred from auction if None)

    Returns:
        (revenue, buyer_surplus)
    """
    if mean is None:
        mean = float((auction.types_low[0, 0] + auction.types_high[0, 0]) / 2.0)
    if std is None:
        std  = float((auction.types_high[0, 0] - auction.types_low[0, 0]) / 6.0)

    key = jax.random.PRNGKey(seed)
    return _rollout_myerson(auction, key, num_samples, mean, std)


# ---------------------------------------------------------------------------
# Full Pavan et al. dynamic mechanism (numerical impulse response)
# ---------------------------------------------------------------------------

def _estimate_impulse_response(
    auction,
    v_it: float,
    t: int,
    s: int,
    num_rollouts: int,
    mean: float,
    std: float,
    seed: int = 0,
) -> float:
    """Estimate E[∂v_{i,s}/∂v_{i,t}] numerically via finite differences.

    Perturb v_{i,t} by ±δ and compare the expected type at time s.
    """
    delta = 0.01
    key = jax.random.PRNGKey(seed)

    def rollout_from(v_start, key):
        # Roll type from t to s assuming always allocated (IRF conditioned on allocation).
        # Uses the allocated sigma = 1 / (lambda * (1 + alpha)) where alpha accumulates.
        v   = v_start
        low  = float(auction.types_low[0, 0])
        high = float(auction.types_high[0, 0])
        lam  = getattr(auction, 'trans_lambda', 1.0)
        for step in range(t, s):
            alpha = 0.5 * (step - t)          # past allocations since t
            sigma = 1.0 / (lam * (1.0 + alpha))
            key, sk = jax.random.split(key)
            noise = float(jax.random.normal(sk))
            v_clip = max(1e-6, min(1 - 1e-6, (v - low) / (high - low)))
            s_t    = float(jnp.log(v_clip / (1 - v_clip)))
            s_next = s_t + sigma * noise
            v = low + (high - low) * float(jax.nn.sigmoid(jnp.array(s_next)))
        return v

    total_diff = 0.0
    for r in range(num_rollouts):
        key, sk1, sk2 = jax.random.split(key, 3)
        v_hi = rollout_from(v_it + delta, sk1)
        v_lo = rollout_from(v_it - delta, sk2)
        total_diff += (v_hi - v_lo) / (2 * delta)
    return total_diff / num_rollouts


def pavan_virtual_value(
    v_it: jnp.ndarray,   # scalar type
    t: int,
    T: int,
    gamma: float,
    mean: float,
    std: float,
    low: float,
    high: float,
    alloc_cost: float,
    impulse_responses=None,  # dict (t,s) -> E[∂v_s/∂v_t], precomputed
) -> float:
    """Full Pavan virtual value including future impulse responses.

    ψ_i(v_{it}, t) = ψ_myerson(v_{it}, t)
                   - Σ_{s=t+1}^{T} γ^{s-t} * IR(t,s) * (1 - F(v_{it})) / f(v_{it})

    where IR(t,s) = E[∂v_{i,s}/∂v_{i,t}] is the impulse response.
    """
    base_vv = myerson_virtual_value(v_it, mean, std, low, high)

    if impulse_responses is None:
        return float(base_vv)

    F_v = _normal_cdf_approx(v_it, mean, std)
    f_v = jnp.maximum(_normal_pdf(v_it, mean, std), 1e-8)
    hazard = (1.0 - F_v) / f_v

    correction = 0.0
    for s in range(t + 1, T):
        ir_ts = impulse_responses.get((t, s), 0.0)
        correction += (gamma ** (s - t)) * ir_ts * float(hazard)

    return float(base_vv) - correction


def eval_pavan_mechanism(
    auction,
    num_samples: int = 256,
    num_ir_rollouts: int = 50,
    seed: int = 0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[float, float]:
    """Evaluate the full Pavan et al. (2014) dynamic mechanism.

    Precomputes impulse responses for all (t, s) pairs, then simulates the
    mechanism using pavan_virtual_value at each step.

    Returns:
        (revenue, buyer_surplus)
    """
    if mean is None:
        mean = float((auction.types_low[0, 0] + auction.types_high[0, 0]) / 2.0)
    if std is None:
        std  = float((auction.types_high[0, 0] - auction.types_low[0, 0]) / 6.0)

    T         = auction.num_horizons
    gamma     = auction.discount
    low       = float(auction.types_low[0, 0])
    high      = float(auction.types_high[0, 0])
    alloc_cost= float(auction.allocation_cost[0, 0])
    B         = auction.num_buyers
    G         = auction.num_goods

    # Precompute impulse responses at the mean type
    v_ref = mean
    impulse_responses = {}
    for t in range(T):
        for s in range(t + 1, T):
            impulse_responses[(t, s)] = _estimate_impulse_response(
                auction, v_ref, t, s, num_ir_rollouts, mean, std,
                seed=seed + t * 100 + s
            )

    key = jax.random.PRNGKey(seed + 9999)

    total_revenue = 0.0
    total_surplus = 0.0

    for _ in range(num_samples):
        key, subkey = jax.random.split(key)
        state = auction.get_initial_state(subkey)

        ep_revenue = 0.0
        ep_surplus = 0.0

        for t in range(T):
            types_t = state.current_type[:, 0]  # (B,)

            # Compute Pavan virtual values
            pvv = jnp.array([
                pavan_virtual_value(
                    types_t[i], t, T, gamma, mean, std, low, high,
                    alloc_cost, impulse_responses
                )
                for i in range(B)
            ])

            winner_pvv = jnp.max(pvv)
            winner_idx = int(jnp.argmax(pvv))
            allocate = bool(winner_pvv >= 0.0 and winner_pvv > alloc_cost)

            alloc = np.zeros(B)
            if allocate:
                alloc[winner_idx] = 1.0
            alloc_j = jnp.array(alloc)

            # Threshold payment
            sorted_pvv = jnp.sort(pvv)[::-1]
            threshold_pvv = float(jnp.maximum(sorted_pvv[1] if B > 1 else 0.0, 0.0))
            # Approximate threshold type from Myerson inversion
            F_th = _normal_cdf_approx(jnp.array(threshold_pvv), mean, std)
            f_th = jnp.maximum(_normal_pdf(jnp.array(threshold_pvv), mean, std), 1e-8)
            threshold_v = float(threshold_pvv + (1.0 - F_th) / f_th)
            threshold_v = np.clip(threshold_v, low, high)

            pay = np.zeros(B)
            if allocate:
                pay[winner_idx] = threshold_v
            pay_j = jnp.array(pay)

            reward = types_t * alloc_j - pay_j
            profit = float(jnp.sum(pay_j)) - alloc_cost * float(jnp.sum(alloc_j))

            ep_revenue += gamma**t * profit
            ep_surplus += gamma**t * float(jnp.sum(reward))

            key, subkey = jax.random.split(key)
            state = auction.get_next_state(
                state,
                types_t.reshape(B, G),
                alloc_j.reshape(B, G),
                subkey,
            )

        total_revenue += ep_revenue
        total_surplus += ep_surplus

    return total_revenue / num_samples, total_surplus / num_samples


# ---------------------------------------------------------------------------
# Correct bandit auction benchmark: Gittins index (PST 2014, Eq. 15)
# ---------------------------------------------------------------------------

def _gittins_index(
    auction,
    state,
    buyer_i: int,
    inv_hz_i: float,
    c_i: float,
    num_index_samples: int,
    key: jax.random.PRNGKey,
) -> float:
    """Gittins stopping-time index for buyer i (PST 2014, Eq. 15).

    gamma_i = max_{T >= t}
        E[ sum_{tau=t}^{T} delta^{tau-t} * (theta_{i,tau} - c_i - inv_hz_i) | theta_{it}, always allocated ]
        / sum_{tau=t}^{T} delta^{tau-t}

    Flow payoff = theta - c_i - inv_hz_i, with BOTH cost and initial hazard inside numerator.
    """
    t = int(state.time_step)
    T = auction.num_horizons
    gamma = float(auction.discount)
    B = auction.num_buyers
    sample_keys = jax.random.split(key, num_index_samples)

    # Simulate num_index_samples paths always allocating to buyer i
    all_paths = []
    for sk in sample_keys:
        sim = state
        path = []
        for s in range(t, T):
            sk, step_key = jax.random.split(sk)
            path.append(float(sim.current_type[buyer_i, 0]))
            alloc = jnp.zeros((B, auction.num_goods)).at[buyer_i, 0].set(1.0)
            sim = auction.get_next_state(sim, sim.current_type, alloc, step_key)
        all_paths.append(path)

    best_ratio = float('-inf')
    for tau_stop in range(t, T):
        length = tau_stop - t + 1
        ratios = []
        for path in all_paths:
            num = sum(gamma**s * (path[s] - c_i - inv_hz_i) for s in range(length))
            den = sum(gamma**s for s in range(length))
            ratios.append(num / den)
        mean_ratio = float(np.mean(ratios))
        if mean_ratio > best_ratio:
            best_ratio = mean_ratio
    return best_ratio


def eval_pavan_bandit_auction(
    auction,
    num_samples: int = 500,
    num_index_samples: int = 50,
    seed: int = 0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[float, float]:
    """Bandit auction benchmark: Gittins index policy (PST 2014, Section 5, Eq. 15).

    Correct for the bandit auction setting where impulse responses are all 1:
    - inv_hazard_i = (1-F_0(theta_i0))/f_0(theta_i0) computed at the INITIAL type,
      fixed for the entire episode.
    - Virtual index = Gittins stopping-time formula with flow payoff
      (theta_{it} - c_i - inv_hazard_i), both cost and hazard inside numerator.
    - Revenue = E[sum_t delta^t sum_i (theta_it - inv_hazard_i - c_i) * x_it].
    - Allocation: argmax_i gamma_i if max >= 0 (seller's index = 0).

    Args:
        auction:            BanditAuction (cont_single, G=1)
        num_samples:        Monte Carlo trajectories for revenue estimation
        num_index_samples:  MC rollouts per Gittins index evaluation
        seed:               RNG seed
        mean:               F_0 mean (inferred from auction if None)
        std:                F_0 std  (inferred from auction if None)

    Returns:
        (revenue, buyer_surplus)
    """
    if mean is None:
        mean = float((auction.types_low[0, 0] + auction.types_high[0, 0]) / 2.0)
    if std is None:
        std = float((auction.types_high[0, 0] - auction.types_low[0, 0]) / 6.0)

    B = auction.num_buyers
    G = auction.num_goods
    T = auction.num_horizons
    gamma = float(auction.discount)

    key = jax.random.PRNGKey(seed)
    total_revenue = 0.0
    total_surplus = 0.0

    for ep in range(num_samples):
        key, ep_key = jax.random.split(key)
        state = auction.get_initial_state(ep_key)

        # Compute inverse hazard rates from INITIAL types — fixed for this episode.
        inv_hazard = []
        for i in range(B):
            v0 = float(state.current_type[i, 0])
            F0 = float(_normal_cdf_approx(jnp.array(v0), mean, std))
            f0 = max(float(_normal_pdf(jnp.array(v0), mean, std)), 1e-12)
            inv_hazard.append((1.0 - F0) / f0)

        ep_revenue = 0.0
        ep_surplus = 0.0

        for t in range(T):
            # Compute Gittins index for each buyer
            key, idx_key = jax.random.split(key)
            idx_keys = jax.random.split(idx_key, B)

            indices = jnp.array([
                _gittins_index(
                    auction, state, i,
                    inv_hazard[i], float(auction.allocation_cost[i, 0]),
                    num_index_samples, idx_keys[i],
                )
                for i in range(B)
            ])

            best_i = int(jnp.argmax(indices))
            alloc_2d = jnp.zeros((B, G))
            if float(indices[best_i]) >= 0.0:
                alloc_2d = alloc_2d.at[best_i, 0].set(1.0)

            types_t = state.current_type[:, 0]
            inv_hz_arr = jnp.array(inv_hazard)
            costs = auction.allocation_cost[:, 0]
            virtual_surplus = float(
                jnp.sum((types_t - inv_hz_arr - costs) * alloc_2d[:, 0])
            )
            ep_revenue += gamma**t * virtual_surplus
            ep_surplus += gamma**t * float(jnp.sum(types_t * alloc_2d[:, 0]))

            key, trans_key = jax.random.split(key)
            state = auction.get_next_state(
                state, state.current_type, alloc_2d, trans_key
            )

        total_revenue += ep_revenue
        total_surplus += ep_surplus

        if (ep + 1) % 200 == 0:
            print(f'  [{ep+1}/{num_samples}] mean rev so far: '
                  f'{total_revenue / (ep + 1):.4f}')

    return total_revenue / num_samples, total_surplus / num_samples


# ---------------------------------------------------------------------------
# Discrete Pavan: exact virtual values for 3-type case
# ---------------------------------------------------------------------------

def eval_pavan_discrete(
    auction,
    num_samples: int = 512,
    seed: int = 0,
) -> Tuple[float, float]:
    """Evaluate the Pavan mechanism for discrete_single (type_values = [1,2,3]).

    For discrete types, virtual value is:
        ψ(v_k) = v_k - (1 - F(v_k)) / f(v_k)
    where F, f are the empirical CDF/PMF over {1,2,3} (uniform).

    Returns (revenue, buyer_surplus).
    """
    type_vals = auction.buyer_types[0, :, 0]  # (K,) e.g. [1,2,3]
    K         = len(type_vals)
    B         = auction.num_buyers
    G         = auction.num_goods
    T         = auction.num_horizons
    gamma     = auction.discount
    alloc_cost= float(auction.allocation_cost[0, 0])

    # Uniform prior over K types
    pmf = jnp.ones(K) / K
    cmf = jnp.cumsum(pmf)   # F(v_k)

    def vv_discrete(v_idx):
        F_k = cmf[v_idx]
        f_k = pmf[v_idx]
        v_k = type_vals[v_idx]
        return v_k - (1.0 - F_k) / f_k

    vv_all = jnp.array([float(vv_discrete(k)) for k in range(K)])

    key = jax.random.PRNGKey(seed)
    total_revenue = 0.0
    total_surplus = 0.0

    for _ in range(num_samples):
        key, subkey = jax.random.split(key)
        state = auction.get_initial_state(subkey)

        ep_revenue = 0.0
        ep_surplus = 0.0

        for t in range(T):
            types_t = state.current_type[:, 0]  # (B,)

            # Map each buyer's type to their virtual value
            pvv = jnp.array([
                float(vv_all[jnp.argmin(jnp.abs(type_vals - types_t[i]))])
                for i in range(B)
            ])

            winner_pvv = jnp.max(pvv)
            winner_idx = int(jnp.argmax(pvv))
            allocate = bool(float(winner_pvv) >= 0.0 and float(winner_pvv) > alloc_cost)

            alloc = np.zeros(B)
            if allocate:
                alloc[winner_idx] = 1.0
            alloc_j = jnp.array(alloc)

            # Threshold: smallest type s.t. ψ(v) ≥ max_{j≠i} ψ(v_j)
            sorted_pvv = float(jnp.sort(pvv)[::-1][1]) if B > 1 else 0.0
            thresh_vv = max(sorted_pvv, 0.0)
            thresh_idx = jnp.searchsorted(vv_all, thresh_vv, side="left")
            thresh_idx = int(jnp.clip(thresh_idx, 0, K - 1))
            threshold_v = float(type_vals[thresh_idx])

            pay = np.zeros(B)
            if allocate:
                pay[winner_idx] = threshold_v
            pay_j = jnp.array(pay)

            reward = types_t * alloc_j - pay_j
            profit = float(jnp.sum(pay_j)) - alloc_cost * float(jnp.sum(alloc_j))

            ep_revenue += gamma**t * profit
            ep_surplus += gamma**t * float(jnp.sum(reward))

            key, subkey = jax.random.split(key)
            state = auction.get_next_state(
                state,
                types_t.reshape(B, G),
                alloc_j.reshape(B, G),
                subkey,
            )

        total_revenue += ep_revenue
        total_surplus += ep_surplus

    return total_revenue / num_samples, total_surplus / num_samples
