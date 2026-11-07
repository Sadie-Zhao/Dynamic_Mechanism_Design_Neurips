"""
cont_rollout.py — custom CONTINUOUS regret rollout for the ladder.

Mirrors core.AuctionSimulator.sample_cumulative_regret EXACTLY (B+1 trajectories, one SHARED
transition key k_trans per step so a truthful oracle gives regret ≡ 0, buyer-SUMMED regret) with
one addition: it threads each trajectory's own PAYMENT history to the oracle head, so the info=m2
oracle can condition on its realized payments (which core's rollout computes but discards).

The oracle head is called as  policy_apply(params, belief_state, auction, key, payment_history)
where payment_history is (T, B) with columns = per-buyer payments of THIS trajectory, steps 0..t-1
filled (past), t..T-1 zero. For info=m1 the head ignores it.

Validation anchor: with the (m1, gru) head this reproduces core.estimate_cumulative_regret to float
precision (payment ignored, identical trajectory logic) — checked in cont_smoke.py.
"""
import functools
import jax
import jax.numpy as jnp
from bandit_auction.core import AuctionSimulator


def sample_regret_det(auction, state, belief_state, mechanism, policy_apply, params, key):
    """Deterministic (pathwise / CMA-ES) regret for ONE init state. Returns buyer-SUMMED cumulative regret."""
    B, T = auction.num_buyers, auction.num_horizons
    discount = jnp.float32(auction.discount)
    ubf = auction.get_next_belief_state_continuous

    def step(t, carry):
        cumul, dev_states, dev_beliefs, dev_pays, truth_state, k = carry
        k, k_pol, k_trans = jax.random.split(k, 3)

        # Truthful trajectory (shared k_trans)
        truth_types = truth_state.current_type
        truth_alloc, truth_pay = mechanism(truth_state, truth_types, auction)
        truth_rewards = AuctionSimulator.get_rewards(auction, truth_state, truth_alloc, truth_pay)  # (B,)
        next_truth = auction.get_next_state(truth_state, truth_types, truth_alloc, k_trans)

        new_states, new_beliefs, new_pays, dev_utils = [], [], [], []
        for i in range(B):
            reports_i = policy_apply(params, dev_beliefs[i], auction, k_pol, dev_pays[i])  # (B, G)
            report_i = dev_states[i].current_type.at[i].set(reports_i[i])                  # only buyer i deviates
            alloc_i, pay_i = mechanism(dev_states[i], report_i, auction)
            util_i = jnp.sum(alloc_i[i] * dev_states[i].current_type[i]) - pay_i[i]        # TRUE-value utility
            dev_utils.append(util_i)
            ns_i = auction.get_next_state(dev_states[i], report_i, alloc_i, k_trans)        # shared k_trans
            nb_i = ubf(dev_beliefs[i], report_i, alloc_i, ns_i)
            np_i = dev_pays[i].at[t].set(pay_i)                                            # record THIS step's payments
            new_states.append(ns_i); new_beliefs.append(nb_i); new_pays.append(np_i)

        dev_rewards = jnp.stack(dev_utils)                                                 # (B,)
        cumul = cumul + jnp.power(discount, jnp.asarray(t, jnp.float32)) * jnp.sum(dev_rewards - truth_rewards)
        return (cumul, new_states, new_beliefs, new_pays, next_truth, k)

    pay0 = jnp.zeros((T, B), jnp.float32)
    init = (jnp.float32(0.0), [state] * B, [belief_state] * B, [pay0] * B, state, key)
    cumul = jax.lax.fori_loop(0, T, step, init)[0]
    return cumul


@functools.partial(jax.jit, static_argnames=("policy_apply", "num_samples"))
def estimate_regret_det(auction, mechanism, policy_apply, params, num_samples, key):
    """Mean buyer-SUMMED regret over num_samples fresh init states (÷B externally for per-buyer)."""
    k_init, k_roll = jax.random.split(key)
    init_states = AuctionSimulator.sample_init_states(auction, num_samples, k_init)
    roll_keys = jax.random.split(k_roll, num_samples)

    def fn(s, k):
        bs = auction.get_initial_belief_state(s)
        return sample_regret_det(auction, s, bs, mechanism, policy_apply, params, k)

    return jnp.mean(jax.vmap(fn)(init_states, roll_keys))
