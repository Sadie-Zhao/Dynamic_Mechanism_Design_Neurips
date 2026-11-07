"""
belief_faithful.py — the FAITHFUL discrete Bayes belief filter for the deviation oracle.

Design (per the paper's intent + the theory): b_{i,t} is agent i's posterior over the joint type
profile given agent i's information set {own type history, others' reports, public outcomes}. Because
(a) the type transition P(v'|v,x) depends only on (own type, own allocation) — both public — and
(b) the public outcome injectively identifies the opponent's report (=type under truthful play),
the belief is COMPUTABLE FROM PUBLIC OUTCOMES and is a SUFFICIENT statistic for the private history.

The shipped filter conditions the public allocation only into agent i's OWN type transition, so its
opponent-marginal is frozen at the prior (opponent-uninformative). The faithful filter conditions on
the observed OTHERS' report + their public allocation, giving the exact one-step posterior over each
opponent's current type — identical to the non-clairvoyant DP belief.

    b_{i,t+1}(k) ∝ 1[profile_k[i] = v_{i,t+1}] · Π_{j≠i} P(profile_k[j] | v_{j,t}=report[j], x_{j,t}),

i.e. own component pinned to the observed own next type; each opponent component = one-step transition
from that opponent's observed current report and public allocation.
"""
import jax
import jax.numpy as jnp
from bandit_auction.core import BeliefState, BanditAuction, State


def make_faithful_discrete_belief(auction: BanditAuction):
    """Returns update_belief_fn(belief_state, report, allocation, next_state) -> BeliefState,
    a drop-in replacement for auction.get_next_belief_state_discrete."""
    B = auction.num_buyers
    G = auction.num_goods
    K = auction.num_type_profiles
    tp = auction.type_profiles                      # (K, B, G)

    def faithful_update(belief_state, report, allocation, next_state):
        t = belief_state.time_step
        new_alloc_hist = belief_state.allocation_history.at[t].set(allocation)
        new_report_hist = belief_state.report_history.at[t].set(report)

        def update_buyer(i):
            def prof_lik(k):
                profile_k = tp[k]                    # (B, G)
                own_ok = jnp.all(profile_k[i] == next_state.current_type[i]).astype(jnp.float32)

                def other_lik(j):
                    # P(opponent j's next type = profile_k[j] | v_{j,t}=report[j], x_{j,t}=allocation[j])
                    return auction.get_transition_prob(
                        auction, t, report[j], allocation[j], profile_k[j])
                liks = jax.vmap(other_lik)(jnp.arange(B))        # (B,) unnormalized transitions
                # product over j != i  (mask own slot to 1.0)
                prod_others = jnp.prod(jnp.where(jnp.arange(B) == i, 1.0, liks))
                return own_ok * prod_others

            b = jax.vmap(prof_lik)(jnp.arange(K))                # (K,)
            return b / jnp.maximum(jnp.sum(b), 1e-12)

        new_beliefs = jax.vmap(update_buyer)(jnp.arange(B))      # (B, K)
        new_priv = belief_state.priv_embeds.at[:, :G].set(
            next_state.current_type.astype(jnp.float32))
        return belief_state.replace_state(
            t + 1, new_alloc_hist, new_report_hist,
            belief_state.pub_embed, new_priv, new_beliefs)

    return faithful_update


def make_faithful_outcome_belief(auction: BanditAuction, mechanism):
    """FAITHFUL belief updated from the PUBLIC OUTCOME ALONE (report-free) — the model-correct filter.
    For each buyer i the opponent's current type is inferred by a Bayes MEASUREMENT: run the (fixed)
    mechanism with buyer i's OWN report and each hypothesized opponent report, and compare the resulting
    PUBLIC outcome to the observed one; no opponent report is ever read. Then transition using the public
    allocation and pin the observed own next type. Requires the mechanism (audit-time, fixed mechanism).
    For an injective mechanism this reproduces make_faithful_discrete_belief exactly (verified), but under
    the correct assumption: agents observe only their own type + the public outcome."""
    B, G = auction.num_buyers, auction.num_goods
    K = auction.num_type_profiles
    tp = auction.type_profiles                         # (K, B, G)

    def update(belief_state, report, allocation, next_state):
        t = belief_state.time_step
        new_alloc_hist = belief_state.allocation_history.at[t].set(allocation)
        new_report_hist = belief_state.report_history.at[t].set(report)
        st = State(time_step=t, allocation_history=belief_state.allocation_history,
                   current_type=jnp.zeros((B, G), jnp.float32),   # mechanism ignores current_type
                   report_history=belief_state.report_history)

        def update_buyer(i):
            def meas_lik(k):                            # P(observed public outcome | opp reports = profile_k)
                prof = tp[k]
                rep_hyp = jnp.where((jnp.arange(B) == i)[:, None], report, prof)  # keep OWN report; others=prof
                alloc_hyp, _ = mechanism(st, rep_hyp, auction)
                return jnp.exp(-1e4 * jnp.sum(jnp.abs(alloc_hyp - allocation)))    # ~indicator: outcome match
            lik = jax.vmap(meas_lik)(jnp.arange(K))
            b_meas = belief_state.beliefs[i] * lik
            b_meas = b_meas / jnp.maximum(jnp.sum(b_meas), 1e-12)

            def b_next_kp(kp):                          # transition (joint, per-buyer independent) via public alloc
                prof_p = tp[kp]
                def from_k(k):
                    prof = tp[k]
                    pj = jax.vmap(lambda j: auction.get_transition_prob(
                        auction, t, prof[j], allocation[j], prof_p[j]))(jnp.arange(B))
                    return jnp.prod(pj)
                return jnp.sum(b_meas * jax.vmap(from_k)(jnp.arange(K)))
            b_pred = jax.vmap(b_next_kp)(jnp.arange(K))
            own_ok = jax.vmap(lambda kp: jnp.all(tp[kp][i] == next_state.current_type[i]).astype(jnp.float32))(jnp.arange(K))
            b_new = b_pred * own_ok
            return b_new / jnp.maximum(jnp.sum(b_new), 1e-12)

        new_beliefs = jax.vmap(update_buyer)(jnp.arange(B))
        new_priv = belief_state.priv_embeds.at[:, :G].set(next_state.current_type.astype(jnp.float32))
        return belief_state.replace_state(t + 1, new_alloc_hist, new_report_hist,
                                          belief_state.pub_embed, new_priv, new_beliefs)
    return update


def faithful_init_belief(auction: BanditAuction, initial_state):
    """Initial faithful belief: own component pinned to the observed initial own type, opponent
    component = uniform prior (no observation yet at t=0)."""
    B, G, T = auction.num_buyers, auction.num_goods, auction.num_horizons
    K = auction.num_type_profiles
    tp = auction.type_profiles

    def init_buyer(i):
        def prof(k):
            return jnp.all(tp[k][i] == initial_state.current_type[i]).astype(jnp.float32)
        b = jax.vmap(prof)(jnp.arange(K))
        return b / jnp.maximum(jnp.sum(b), 1e-12)
    beliefs = jax.vmap(init_buyer)(jnp.arange(B))                # (B, K)
    priv = jnp.zeros((B, auction.d_priv), jnp.float32).at[:, :G].set(
        initial_state.current_type.astype(jnp.float32))
    return BeliefState(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G), jnp.float32),
        report_history=jnp.zeros((T, B, G), jnp.float32),
        pub_embed=jnp.zeros((auction.d_pub,), jnp.float32),
        priv_embeds=priv,
        beliefs=beliefs,
    )
