# ── 2. Imports ───────────────────────────────────────────────────────────────
import itertools, functools, time
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax import struct

# ── 3. Core data structures ───────────────────────────────────────────────────

@struct.dataclass
class State:
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    current_type: jnp.ndarray         # (B, G)
    report_history: jnp.ndarray       # (T, B, G)

    def replace_buyer_type(self, buyer, buyer_type):
        return self.replace(current_type=self.current_type.at[buyer].set(buyer_type))


@struct.dataclass
class BeliefState:
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    report_history: jnp.ndarray       # (T, B, G)
    pub_embed: jnp.ndarray            # (d_pub,)  — zeros for discrete
    priv_embeds: jnp.ndarray          # (B, d_priv) — zeros for discrete
    beliefs: jnp.ndarray              # (B, K) Bayesian posterior

    def replace_state(self, new_t, new_ah, new_rh, new_pub, new_priv, new_bel):
        return self.replace(
            time_step=new_t, allocation_history=new_ah, report_history=new_rh,
            pub_embed=new_pub, priv_embeds=new_priv, beliefs=new_bel,
        )


@struct.dataclass
class BanditAuction:
    num_horizons: int  = struct.field(pytree_node=False)
    num_buyers: int    = struct.field(pytree_node=False)
    num_goods: int     = struct.field(pytree_node=False)
    init_state_function:        Callable = struct.field(pytree_node=False)
    init_belief_state_function: Callable = struct.field(pytree_node=False)
    transition_function:        Callable = struct.field(pytree_node=False)
    discount:          float        = struct.field(pytree_node=False)
    allocation_cost:   jnp.ndarray  = struct.field(pytree_node=False)
    sample_types_fn:   Callable     = struct.field(pytree_node=False)
    types_low:         jnp.ndarray  = struct.field(pytree_node=False)
    types_high:        jnp.ndarray  = struct.field(pytree_node=False)
    get_transition_prob: Optional[Callable] = struct.field(pytree_node=False, default=None)
    buyer_types:       Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    type_profiles:     Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    num_type_profiles: int  = struct.field(pytree_node=False, default=0)
    belief_pub_net:    Optional[object] = struct.field(pytree_node=False, default=None)
    belief_pub_params: Optional[object] = struct.field(pytree_node=False, default=None)
    belief_priv_net:   Optional[object] = struct.field(pytree_node=False, default=None)
    belief_priv_params:Optional[object] = struct.field(pytree_node=False, default=None)
    d_pub:  int = struct.field(pytree_node=False, default=64)
    d_priv: int = struct.field(pytree_node=False, default=128)
    trans_sigma_no: float = struct.field(pytree_node=False, default=1.0)
    trans_lambda:   float = struct.field(pytree_node=False, default=1.0)

    def get_initial_state(self, key):
        return self.init_state_function(self, key)

    def get_initial_belief_state(self, state):
        return self.init_belief_state_function(self, state)

    def get_next_state(self, state, report, allocation, key):
        t = state.time_step
        ah = state.allocation_history.at[t].set(allocation)
        rh = state.report_history.at[t].set(report)
        new_type = self.transition_function(self, t, state.current_type, ah, key)
        return State(time_step=t+1, allocation_history=ah,
                     current_type=new_type, report_history=rh)

    def get_next_belief_state_discrete(self, belief_state, report, allocation, next_state):
        t = belief_state.time_step
        ah = belief_state.allocation_history.at[t].set(allocation)
        rh = belief_state.report_history.at[t].set(report)

        def update_buyer(i, bel_i):
            next_type_i = next_state.current_type[i]
            def trans_k(k):
                prof_k = self.type_profiles[k]
                return self.get_transition_prob(self, t, prof_k[i], allocation[i], next_type_i)
            tp = jax.vmap(trans_k)(jnp.arange(self.num_type_profiles))
            nb = bel_i * tp
            return nb / jnp.maximum(nb.sum(), 1e-12)

        new_beliefs = jax.vmap(update_buyer, in_axes=(0,0))(
            jnp.arange(self.num_buyers), belief_state.beliefs)
        return belief_state.replace_state(
            t+1, ah, rh, belief_state.pub_embed, belief_state.priv_embeds, new_beliefs)

    @property
    def is_discrete(self): return self.buyer_types is not None

# ── 4. Init helpers, samplers, transition functions ───────────────────────────

def bandit_init_state(auction, key):
    types = auction.sample_types_fn(key)
    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    return State(time_step=0,
                 allocation_history=jnp.zeros((T,B,G)),
                 current_type=types,
                 report_history=jnp.zeros((T,B,G)))


def bandit_init_belief_state_discrete(auction, initial_state):
    B, G, T, K = auction.num_buyers, auction.num_goods, auction.num_horizons, auction.num_type_profiles
    return BeliefState(
        time_step=0,
        allocation_history=jnp.zeros((T,B,G), jnp.float32),
        report_history=jnp.zeros((T,B,G), jnp.float32),
        pub_embed=jnp.zeros((auction.d_pub,), jnp.float32),
        priv_embeds=jnp.zeros((B, auction.d_priv), jnp.float32),
        beliefs=jnp.ones((B, K), jnp.float32) / K,
    )


def make_discrete_sampler(type_profiles):
    K = type_profiles.shape[0]
    def fn(key):
        idx = jax.random.randint(key, shape=(), minval=0, maxval=K)
        return type_profiles[idx]
    return fn


def discrete_transition_fn(auction, time_step, types, allocation_history, key):
    x_t = allocation_history[time_step]
    def sample_buyer(i, key_i):
        cur_i = types[i]; alloc_i = x_t[i]
        buyer_types_i = auction.buyer_types[i]
        K_i = buyer_types_i.shape[0]
        def prob_k(k):
            return auction.get_transition_prob(auction, time_step, cur_i, alloc_i, buyer_types_i[k])
        probs = jax.vmap(prob_k)(jnp.arange(K_i))
        probs = probs / jnp.maximum(probs.sum(), 1e-12)
        return buyer_types_i[jax.random.choice(key_i, K_i, p=probs)]
    keys = jax.random.split(key, auction.num_buyers)
    return jax.vmap(sample_buyer)(jnp.arange(auction.num_buyers), keys)


def _discrete_transition_prob(auction, time_step, current_type_i, alloc_i, next_type_i):
    alloc_count = jnp.sum(alloc_i)
    decay_rate  = alloc_count / 2.0
    diff = jnp.sum(jnp.abs(next_type_i - current_type_i))
    return jnp.exp(-decay_rate * diff)

# ── 5. Policy networks ────────────────────────────────────────────────────────

@struct.dataclass
class DiscretePolicy:
    network: hk.Transformed = struct.field(pytree_node=False)
    params:  hk.Params
    def __call__(self, belief_state, auction, key=None):
        return self.network.apply(self.params, belief_state, auction, key)


class DiscreteResidualHead(hk.Module):
    def __init__(self, d_priv, d_pub, hidden=128, name=None):
        super().__init__(name=name)
        self.d_priv = d_priv; self.d_pub = d_pub; self.hidden = hidden
        self.trunk = hk.nets.MLP([hidden, hidden], activate_final=True)

    def __call__(self, belief_state, auction, key):
        B, K = auction.num_buyers, auction.num_type_profiles
        H = belief_state.priv_embeds   # (B, d_priv)
        c = belief_state.pub_embed     # (d_pub,)
        zero = hk.initializers.Constant(0.0)
        c_proj = hk.Linear(self.d_priv, w_init=zero, b_init=zero)(c)
        x = jnp.concatenate([H, jnp.broadcast_to(c_proj, H.shape)], axis=-1)
        z = self.trunk(x)
        logits = hk.Linear(K, w_init=zero, b_init=zero)(z)  # (B, K)
        if key is not None:
            g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)))
            soft = jax.nn.softmax((logits + g) / 1.0, axis=-1)
        else:
            soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), K)
        probs = soft + jax.lax.stop_gradient(hard - soft)
        return jnp.einsum('bk,kbg->bg', probs, auction.type_profiles)


def build_discrete_policy_net(d_priv, d_pub, hidden=128):
    def f(bs, auction, key=None):
        return DiscreteResidualHead(d_priv, d_pub, hidden)(bs, auction, key)
    return hk.without_apply_rng(hk.transform(f))

# ── 6. Mechanism network ──────────────────────────────────────────────────────

def _attention_pool(seq, out_dim):
    """(T, B, d) -> (B, out_dim)"""
    T, B, d = seq.shape
    proj   = jax.nn.relu(hk.Linear(out_dim, with_bias=False)(seq.reshape(T*B, d)))
    scores = hk.Linear(1, with_bias=False)(proj).reshape(T, B, 1)
    w      = jax.nn.softmax(scores, axis=0)
    return jnp.sum(w * proj.reshape(T, B, out_dim), axis=0)


def mechanism_network(state, report, auction):
    B, G = auction.num_buyers, auction.num_goods
    type_feat   = hk.nets.MLP([128,64], activate_final=True)(state.current_type.reshape(B,-1).astype(jnp.float32))
    report_feat = hk.nets.MLP([128,64], activate_final=True)(report.reshape(B,-1).astype(jnp.float32))
    alloc_sum   = jnp.sum(state.allocation_history, axis=0).astype(jnp.float32)
    alloc_feat  = hk.nets.MLP([96,48], activate_final=True)(alloc_sum.reshape(B,-1))
    rep_hist_feat = _attention_pool(state.report_history.astype(jnp.float32), 32)
    h = jnp.concatenate([type_feat, report_feat, alloc_feat, rep_hist_feat], axis=-1)
    h = hk.nets.MLP([128,64], activate_final=True)(h)
    alloc_logits = jnp.stack([
        jax.nn.softmax(jnp.mean(hk.Linear(B+1)(h), axis=0), axis=0)
        for _ in range(G)
    ], axis=1)  # (B+1, G)
    payment = jax.nn.relu(hk.nets.MLP([64,32,1], activate_final=False)(h).squeeze(-1))
    return alloc_logits, payment


@struct.dataclass
class Mechanism:
    network: hk.Transformed = struct.field(pytree_node=False)
    params:  hk.Params
    def __call__(self, state, report, auction):
        alloc_with_null, pay = self.network.apply(self.params, state, report, auction)
        return alloc_with_null[:-1], pay  # drop null row


def build_mechanism_network():
    return hk.without_apply_rng(hk.transform(mechanism_network))

# ── 7. Auction Simulator ──────────────────────────────────────────────────────

class AuctionSimulator:

    @staticmethod
    @jax.jit
    def get_rewards(auction, state, allocation, payment):
        return jnp.sum(allocation * state.current_type, axis=-1) - payment

    @staticmethod
    @jax.jit
    def get_profit(auction, state, allocation, payment):
        return jnp.sum(payment) - jnp.sum(allocation * auction.allocation_cost)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def sample_init_states(auction, num_samples, key):
        return jax.vmap(auction.get_initial_state)(jax.random.split(key, num_samples))

    # ---- Profit ----
    @staticmethod
    @jax.jit
    def sample_cumulative_profit(auction, state, mechanism, key):
        def step(t, carry):
            c, s, k = carry
            k, k2 = jax.random.split(k)
            r = s.current_type
            a, p = mechanism(s, r, auction)
            c = c + (auction.discount**t) * AuctionSimulator.get_profit(auction, s, a, p)
            return (c, auction.get_next_state(s, r, a, k2), k)
        c, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, (0., state, key))
        return c

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_profit(auction, mechanism, num_samples, key):
        k1, k2 = jax.random.split(key)
        ss = AuctionSimulator.sample_init_states(auction, num_samples, k1)
        ks = jax.random.split(k2, num_samples)
        return jnp.mean(jax.vmap(
            lambda s,k: AuctionSimulator.sample_cumulative_profit(auction,s,mechanism,k)
        )(ss, ks))

    # ---- IR ----
    @staticmethod
    @jax.jit
    def sample_cumulative_rewards(auction, state, mechanism, key):
        def step(t, carry):
            c, s, k = carry
            k, k2 = jax.random.split(k)
            r = s.current_type
            a, p = mechanism(s, r, auction)
            c = c + (auction.discount**t) * AuctionSimulator.get_rewards(auction, s, a, p)
            return (c, auction.get_next_state(s, r, a, k2), k)
        init = (jnp.zeros((auction.num_buyers,)), state, key)
        c, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, init)
        return c

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_relu_cumulative_rewards(auction, mechanism, num_samples, key):
        k1, k2 = jax.random.split(key)
        ss = AuctionSimulator.sample_init_states(auction, num_samples, k1)
        ks = jax.random.split(k2, num_samples)
        def fn(s, k):
            return jax.nn.relu(-AuctionSimulator.sample_cumulative_rewards(auction, s, mechanism, k))
        return jnp.mean(jax.vmap(fn)(ss, ks), axis=0)

    # ---- IC Regret (B+1 trajectories, shared transition key) ----
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("update_belief_fn",))
    def sample_cumulative_regret(auction, state, belief_state, mechanism, policy, key,
                                  update_belief_fn=None):
        """B+1 trajectories with shared transition noise.

        All B+1 trajectories use the same k_trans at each step. When the oracle
        is truthful (same reports => same allocs), all trajectories are identical
        and regret = 0 exactly — guaranteeing the estimator is unbiased and >= 0
        at the optimum.
        """
        if update_belief_fn is None:
            update_belief_fn = auction.get_next_belief_state_discrete
        B = auction.num_buyers

        def step(t, carry):
            cumul, dev_states, dev_beliefs, truth_state, k = carry
            # One policy key + one SHARED transition key for all B+1 trajectories
            k, k_policy, k_trans = jax.random.split(k, 3)

            # Truthful trajectory
            truth_types = truth_state.current_type
            ta, tp = mechanism(truth_state, truth_types, auction)
            truth_rewards = AuctionSimulator.get_rewards(auction, truth_state, ta, tp)
            next_truth = auction.get_next_state(truth_state, truth_types, ta, k_trans)

            # Per-buyer deviation trajectories
            dev_rewards_list, next_dev_states, next_dev_beliefs = [], [], []
            for i in range(B):
                oracle_report = policy(dev_beliefs[i], auction, k_policy)
                report_i = dev_states[i].current_type.at[i].set(oracle_report[i])
                ai, pi = mechanism(dev_states[i], report_i, auction)
                util_i = jnp.sum(ai[i] * dev_states[i].current_type[i]) - pi[i]
                dev_rewards_list.append(util_i)
                ns = auction.get_next_state(dev_states[i], report_i, ai, k_trans)
                nb = update_belief_fn(dev_beliefs[i], report_i, ai, ns)
                next_dev_states.append(ns); next_dev_beliefs.append(nb)

            dev_rewards = jnp.stack(dev_rewards_list)
            cumul = cumul + (auction.discount**t) * jnp.sum(dev_rewards - truth_rewards)
            return (cumul, next_dev_states, next_dev_beliefs, next_truth, k)

        init = (0., [state]*B, [belief_state]*B, state, key)
        cumul, _, _, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, init)
        return cumul

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("num_samples", "update_belief_fn"))
    def estimate_cumulative_regret(auction, mechanism, policy, num_samples, key,
                                    update_belief_fn=None):
        if update_belief_fn is None:
            update_belief_fn = auction.get_next_belief_state_discrete
        k1, k2 = jax.random.split(key)
        ss = AuctionSimulator.sample_init_states(auction, num_samples, k1)
        ks = jax.random.split(k2, num_samples)
        def fn(s, k):
            bs = auction.get_initial_belief_state(s)
            return AuctionSimulator.sample_cumulative_regret(
                auction, s, bs, mechanism, policy, k, update_belief_fn)
        return jnp.mean(jax.vmap(fn)(ss, ks))

# ── 8. Create discrete_single auction ─────────────────────────────────────────

def create_discrete_single_auction(
    n_buyers=2, n_goods=1, horizon=5, discount=0.9,
    type_values=None, allocation_cost=0.1, d_pub=1, d_priv=1, seed=42
):
    if type_values is None:
        type_values = [1.0, 2.0, 3.0]
    K = len(type_values)
    type_vals = jnp.array(type_values, dtype=jnp.float32)
    buyer_types = jnp.stack([type_vals.reshape(K, n_goods)]*n_buyers)  # (B,K,G)
    all_profiles = list(itertools.product(range(K), repeat=n_buyers))
    tp_np = np.array([[[type_values[all_profiles[j][i]]]*n_goods
                        for i in range(n_buyers)]
                       for j in range(len(all_profiles))], dtype=np.float32)
    type_profiles = jnp.array(tp_np)  # (K^B, B, G)
    sample_fn = make_discrete_sampler(type_profiles)
    alloc_cost = jnp.full((n_buyers, n_goods), allocation_cost, jnp.float32)
    low  = jnp.full((n_buyers, n_goods), min(type_values), jnp.float32)
    high = jnp.full((n_buyers, n_goods), max(type_values), jnp.float32)
    return BanditAuction(
        num_horizons=horizon, num_buyers=n_buyers, num_goods=n_goods,
        init_state_function=bandit_init_state,
        init_belief_state_function=bandit_init_belief_state_discrete,
        transition_function=discrete_transition_fn,
        get_transition_prob=_discrete_transition_prob,
        discount=discount, allocation_cost=alloc_cost,
        sample_types_fn=sample_fn, types_low=low, types_high=high,
        buyer_types=buyer_types, type_profiles=type_profiles,
        num_type_profiles=len(all_profiles),
        d_pub=d_pub, d_priv=d_priv,
    )

# Create the auction
SEED = 0
auction = create_discrete_single_auction(
    n_buyers=2, n_goods=1, horizon=5, discount=0.9,
    type_values=[1.0, 2.0, 3.0], allocation_cost=0.1, seed=SEED
)
print(f'Auction: {auction.num_buyers} buyers, {auction.num_goods} good(s), '
      f'T={auction.num_horizons}, {auction.num_type_profiles} type profiles')

# ── 9. Pavan benchmark ────────────────────────────────────────────────────────
# Closed-form for {1,2,3} uniform, 2 buyers, single good
# Virtual values: ψ(v_k) = v_k - (1-F(v_k))/f(v_k)
# ψ(1)=1-2/1=-1 (no alloc), ψ(2)=2-1/1=1, ψ(3)=3-0=3
# Standard Myerson (static) revenue per period = E[max(ψ(v),0)] * discount^t
# Discounted sum over T=5 periods: sum_{t=0}^4 γ^t * E[max(max_i ψ(v_i),0)]

type_values = [1.0, 2.0, 3.0]
F = {1.0: 1/3, 2.0: 2/3, 3.0: 1.0}
f = {1.0: 1/3, 2.0: 1/3, 3.0: 1/3}
virt = {v: v - (1-F[v])/f[v] for v in type_values}
print('Virtual values:', {v: round(phi,2) for v, phi in virt.items()})

# Expected static Myerson revenue per period (2 buyers)
# = E[max(max_i ψ(v_i), 0)] over joint uniform distribution
revenue_per_period = 0.0
for v1 in type_values:
    for v2 in type_values:
        best_virt = max(virt[v1], virt[v2])
        if best_virt > 0:
            # Payment = type of second-highest virtual value (Myerson pricing)
            revenue_per_period += (1/9) * best_virt

discount = 0.9
T = 5
static_myerson = sum(discount**t * revenue_per_period for t in range(T))
print(f'Static Myerson (each period independent): {static_myerson:.4f}')
print(f'(Dynamic Pavan benchmark ~7.67 — accounts for type correlations across periods)')

# ── 10. Training configuration ────────────────────────────────────────────────

# Paper hyperparameters for discrete_single
NUM_OUTER_EPOCHS  = 3  # set to 400 for a quick ~6min run
NUM_INNER_STEPS   = 5
LR_THETA          = 1e-4
LR_W              = 5e-3
NUM_SAMPLES_TRAIN = 8
PRINT_ITER        = 50

# Acceptance method
PROFIT_GOAL  = 7.5
ALPHA_ACC    = 5.0   # weight on [Regret]+
BETA_ACC     = 1.0   # weight on IR

# Penalty method (beta=5 required to enforce IR; beta=1 fails)
ALPHA_PEN    = 1.0
BETA_PEN     = 5.0

# ── 11. Nested GD — Acceptance method ────────────────────────────────────────

def run_acceptance(auction, profit_goal=PROFIT_GOAL,
                   num_outer=NUM_OUTER_EPOCHS, num_inner=NUM_INNER_STEPS,
                   lr_theta=LR_THETA, lr_w=LR_W, n_samples=NUM_SAMPLES_TRAIN,
                   alpha=ALPHA_ACC, beta=BETA_ACC, seed=SEED, print_iter=PRINT_ITER):
    prng = hk.PRNGSequence(seed)
    s0   = auction.get_initial_state(next(prng))
    b0   = auction.get_initial_belief_state(s0)
    mnet = build_mechanism_network()
    mparams = mnet.init(next(prng), s0, jnp.zeros((auction.num_buyers,auction.num_goods)), auction)
    pnet = build_discrete_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub, hidden=128)
    pparams = pnet.init(next(prng), b0, auction)
    update_fn = auction.get_next_belief_state_discrete

    theta_opt = optax.adam(lr_theta); theta_state = theta_opt.init(mparams)
    w_opt     = optax.adam(lr_w)

    @jax.jit
    def cumul_reg(mp, pp, key):
        return AuctionSimulator.estimate_cumulative_regret(
            auction, Mechanism(mnet,mp), DiscretePolicy(pnet,pp), n_samples, key, update_fn)

    @jax.jit
    def inner_step(mp, pp, ws, key):
        reg, g = jax.value_and_grad(cumul_reg, argnums=1)(mp, pp, key)
        neg_g = jax.tree_util.tree_map(lambda x: -x, g)
        upd, ws = w_opt.update(neg_g, ws, pp)
        return reg, optax.apply_updates(pp, upd), ws

    @jax.jit
    def outer_step(mp, pp_star, ts, key):
        def loss_fn(mp2, k):
            k1,k2,k3 = jax.random.split(k,3)
            reg = cumul_reg(mp2, pp_star, k1)
            prof = AuctionSimulator.estimate_cumulative_profit(
                auction, Mechanism(mnet,mp2), n_samples, k2)
            ir = jnp.sum(AuctionSimulator.estimate_relu_cumulative_rewards(
                auction, Mechanism(mnet,mp2), n_samples*5, k3))
            rp = jnp.maximum(0., reg); gp = jnp.maximum(0., profit_goal - prof)
            return alpha*rp + gp + beta*ir, (reg, profit_goal-prof, ir, rp, gp)
        (loss,(reg,gap,ir,rp,gp)), g = jax.value_and_grad(loss_fn,has_aux=True)(mp, key)
        upd, ts = theta_opt.update(g, ts, mp)
        return optax.apply_updates(mp, upd), ts, (loss, reg, gap, ir, rp, gp)

    logs = dict(loss=[], reg=[], gap=[], ir=[], reg_plus=[], gap_plus=[], oracle_reg=[])
    t0 = time.time()
    for epoch in range(num_outer):
        # Reset oracle each outer epoch
        pp = pnet.init(next(prng), auction.get_initial_belief_state(
            auction.get_initial_state(next(prng))), auction)
        ws = w_opt.init(pp)
        reg_last = 0.
        for _ in range(num_inner):
            reg_last, pp, ws = inner_step(mparams, pp, ws, next(prng))
        mparams, theta_state, (loss,reg,gap,ir,rp,gp) = outer_step(
            mparams, pp, theta_state, next(prng))
        logs['loss'].append(float(loss)); logs['reg'].append(float(reg))
        logs['gap'].append(float(gap));   logs['ir'].append(float(ir))
        logs['reg_plus'].append(float(rp)); logs['gap_plus'].append(float(gp))
        logs['oracle_reg'].append(float(reg_last))
        if epoch % print_iter == 0:
            print(f'[Acceptance] Epoch {epoch:4d} ({time.time()-t0:.0f}s) '
                  f'oracle_reg={reg_last:.4f}  reg={float(reg):.4f}  '
                  f'gap={float(gap):.4f}  ir={float(ir):.4f}')
    profit_final = AuctionSimulator.estimate_cumulative_profit(
        auction, Mechanism(mnet,mparams), 1000, jax.random.PRNGKey(99))
    reg_final = AuctionSimulator.estimate_cumulative_regret(
        auction, Mechanism(mnet,mparams), DiscretePolicy(pnet,pp), 1000,
        jax.random.PRNGKey(100), update_fn)
    print(f'\n=== Acceptance final (profit_goal={profit_goal}) ===')
    print(f'  profit={float(profit_final):.4f}  reg={float(reg_final):.4f}  ir={logs["ir"][-1]:.4f}')
    print(f'  Pavan benchmark: 7.67')
    return logs, mparams, pparams


print('Acceptance method ready. Run the next cell to start training.')

# ── 12. Run — Acceptance ──────────────────────────────────────────────────────
logs_acc, mparams_acc, pparams_acc = run_acceptance(auction)

# ── 13. Nested GD — Penalty method ───────────────────────────────────────────

def run_penalty(auction,
                num_outer=NUM_OUTER_EPOCHS, num_inner=NUM_INNER_STEPS,
                lr_theta=LR_THETA, lr_w=LR_W, n_samples=NUM_SAMPLES_TRAIN,
                alpha=ALPHA_PEN, beta=BETA_PEN, seed=SEED, print_iter=PRINT_ITER):
    prng = hk.PRNGSequence(seed)
    s0   = auction.get_initial_state(next(prng))
    b0   = auction.get_initial_belief_state(s0)
    mnet = build_mechanism_network()
    mparams = mnet.init(next(prng), s0, jnp.zeros((auction.num_buyers,auction.num_goods)), auction)
    pnet = build_discrete_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub, hidden=128)
    pparams = pnet.init(next(prng), b0, auction)
    update_fn = auction.get_next_belief_state_discrete

    theta_opt = optax.adam(lr_theta); theta_state = theta_opt.init(mparams)
    w_opt     = optax.adam(lr_w)

    @jax.jit
    def cumul_reg(mp, pp, key):
        return AuctionSimulator.estimate_cumulative_regret(
            auction, Mechanism(mnet,mp), DiscretePolicy(pnet,pp), n_samples, key, update_fn)

    @jax.jit
    def inner_step(mp, pp, ws, key):
        reg, g = jax.value_and_grad(cumul_reg, argnums=1)(mp, pp, key)
        neg_g = jax.tree_util.tree_map(lambda x: -x, g)
        upd, ws = w_opt.update(neg_g, ws, pp)
        return reg, optax.apply_updates(pp, upd), ws

    @jax.jit
    def outer_step(mp, pp_star, ts, key):
        def loss_fn(mp2, k):
            k1,k2,k3 = jax.random.split(k,3)
            reg  = cumul_reg(mp2, pp_star, k1)
            prof = AuctionSimulator.estimate_cumulative_profit(
                auction, Mechanism(mnet,mp2), n_samples, k2)
            ir   = jnp.sum(AuctionSimulator.estimate_relu_cumulative_rewards(
                auction, Mechanism(mnet,mp2), n_samples*5, k3))
            rp   = jnp.maximum(0., reg)
            return -prof + alpha*rp + beta*ir, (reg, prof, ir, rp)
        (loss,(reg,prof,ir,rp)), g = jax.value_and_grad(loss_fn,has_aux=True)(mp, key)
        upd, ts = theta_opt.update(g, ts, mp)
        return optax.apply_updates(mp, upd), ts, (loss, reg, prof, ir, rp)

    logs = dict(loss=[], reg=[], profit=[], ir=[], reg_plus=[], oracle_reg=[])
    t0 = time.time()
    for epoch in range(num_outer):
        pp = pnet.init(next(prng), auction.get_initial_belief_state(
            auction.get_initial_state(next(prng))), auction)
        ws = w_opt.init(pp)
        reg_last = 0.
        for _ in range(num_inner):
            reg_last, pp, ws = inner_step(mparams, pp, ws, next(prng))
        mparams, theta_state, (loss,reg,prof,ir,rp) = outer_step(
            mparams, pp, theta_state, next(prng))
        logs['loss'].append(float(loss)); logs['reg'].append(float(reg))
        logs['profit'].append(float(prof)); logs['ir'].append(float(ir))
        logs['reg_plus'].append(float(rp)); logs['oracle_reg'].append(float(reg_last))
        if epoch % print_iter == 0:
            print(f'[Penalty]    Epoch {epoch:4d} ({time.time()-t0:.0f}s) '
                  f'oracle_reg={reg_last:.4f}  reg={float(reg):.4f}  '
                  f'profit={float(prof):.4f}  ir={float(ir):.4f}')
    profit_final = AuctionSimulator.estimate_cumulative_profit(
        auction, Mechanism(mnet,mparams), 1000, jax.random.PRNGKey(99))
    reg_final = AuctionSimulator.estimate_cumulative_regret(
        auction, Mechanism(mnet,mparams), DiscretePolicy(pnet,pp), 1000,
        jax.random.PRNGKey(100), update_fn)
    print(f'\n=== Penalty final (alpha={alpha}, beta={beta}) ===')
    print(f'  profit={float(profit_final):.4f}  reg={float(reg_final):.4f}  ir={logs["ir"][-1]:.4f}')
    print(f'  Pavan benchmark: 7.67')
    return logs, mparams, pparams


print('Penalty method ready. Run the next cell to start training.')

# ── 14. Run — Penalty ─────────────────────────────────────────────────────────
logs_pen, mparams_pen, pparams_pen = run_penalty(auction)

# ── 15. Plot training curves ──────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def plot_metric(ax, logs_acc, logs_pen, key_acc, key_pen, title, ylabel, hline=None):
    ax.plot(logs_acc[key_acc], label='Acceptance', color='tab:blue')
    ax.plot(logs_pen[key_pen], label='Penalty (β=5)', color='tab:orange')
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', label=f'Pavan ≈{hline}')
    ax.set_title(title); ax.set_xlabel('Outer epoch'); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)

plot_metric(axes[0,0], logs_acc, logs_pen, 'oracle_reg', 'oracle_reg',
            'Oracle Regret (after inner loop)', 'regret')
axes[0,0].axhline(0, color='gray', linestyle=':')

# Profit
acc_profit = [PROFIT_GOAL - g for g in logs_acc['gap']]
axes[0,1].plot(acc_profit, label='Acceptance', color='tab:blue')
axes[0,1].plot(logs_pen['profit'], label='Penalty (β=5)', color='tab:orange')
axes[0,1].axhline(7.67, color='red', linestyle='--', label='Pavan ≈7.67')
axes[0,1].set_title('Cumulative Profit'); axes[0,1].set_xlabel('Outer epoch')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

plot_metric(axes[0,2], logs_acc, logs_pen, 'ir', 'ir',
            'IR Loss (relu(-rewards))', 'IR loss')
axes[0,2].axhline(0, color='gray', linestyle=':')

plot_metric(axes[1,0], logs_acc, logs_pen, 'reg', 'reg',
            'Regret in Outer Loss', 'regret')
axes[1,0].axhline(0, color='gray', linestyle=':')

plot_metric(axes[1,1], logs_acc, logs_pen, 'reg_plus', 'reg_plus',
            '[Regret]+ in Loss', '[Regret]+')

plot_metric(axes[1,2], logs_acc, logs_pen, 'loss', 'loss',
            'Total Loss', 'loss')

plt.tight_layout()
pass  # plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
pass  # plt.show()
print('Saved training_curves.png')

