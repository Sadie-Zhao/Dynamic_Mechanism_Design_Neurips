"""
core.py — Shared data structures, simulators, and network utilities for bandit auction experiments.

Parallel to budget_auction/core.py but without budget constraints (no alpha/beta fields).
Supports both continuous types (GRU-based belief) and discrete types (Bayesian belief).

Key structures:
  State          : auction state (time_step, allocation_history, current_type, report_history)
  BeliefState    : agent's information state; holds GRU embeds (continuous) or Bayesian
                   posteriors (discrete) — both fields present, env selects which to use
  BanditAuction  : auction configuration; buyer_types=None signals continuous case
  AuctionSimulator: static rollout methods (profit, rewards, regret)
  BoundedContPolicy: deterministic continuous report policy (logit-space residual)
  DiscretePolicy : stochastic discrete report policy (categorical over type profiles)
"""

from typing import Callable, Dict, List, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax import struct

SEED = 42

# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_loss_dict(losses: Dict, title: str = ""):
    num_losses = len(losses)
    num_rows = (num_losses - 1) // 3 + 1
    fig, axs = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    axs_flat = np.array(axs).flatten()
    for i, (name, loss) in enumerate(losses.items()):
        axs_flat[i].plot(loss)
        axs_flat[i].set_title(name, fontsize=8)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_loss_dicts(loss_list: List[Dict], dict_names: List[str] = None):
    if dict_names is None:
        dict_names = [f"Dict {i}" for i in range(len(loss_list))]
    for d, name in zip(loss_list, dict_names):
        plot_loss_dict(d, title=name)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@struct.dataclass
class State:
    """Auction state without budget tracking (bandit setting, no constraints).

    Dimensions: T = num_horizons, B = num_buyers, G = num_goods
    """
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    current_type: jnp.ndarray         # (B, G)  true private types
    report_history: jnp.ndarray       # (T, B, G)

    def replace_state(self, new_time_step, new_allocation_history,
                      new_current_type, new_report_history):
        return self.replace(
            time_step=new_time_step,
            allocation_history=new_allocation_history,
            current_type=new_current_type,
            report_history=new_report_history,
        )

    def replace_buyer_type(self, buyer, buyer_type):
        return self.replace(current_type=self.current_type.at[buyer].set(buyer_type))


@struct.dataclass
class BeliefState:
    """Agent's information state. Both continuous and discrete fields are stored;
    the environment determines which are meaningful.

    Continuous: pub_embed (GRU public hidden) + priv_embeds (GRU private hidden per buyer)
    Discrete:   beliefs (B, K) Bayesian posterior P(v_i | history)
    """
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    report_history: jnp.ndarray       # (T, B, G)
    # Continuous fields
    pub_embed: jnp.ndarray            # (d_pub,)
    priv_embeds: jnp.ndarray          # (B, d_priv)
    # Discrete field
    beliefs: jnp.ndarray              # (B, K) posterior; zeros for continuous

    def replace_state(self, new_time_step, new_allocation_history, new_report_history,
                      new_pub_embed, new_priv_embeds, new_beliefs):
        return self.replace(
            time_step=new_time_step,
            allocation_history=new_allocation_history,
            report_history=new_report_history,
            pub_embed=new_pub_embed,
            priv_embeds=new_priv_embeds,
            beliefs=new_beliefs,
        )


@struct.dataclass
class BanditAuction:
    """Auction configuration for bandit experiments.

    buyer_types is None for continuous settings; set to (B, K, G) for discrete.

    pytree_node=False is used ONLY for non-JAX fields (ints, floats, callables).
    All jnp.ndarray and hk.Params fields are pytree leaves (default pytree_node=True)
    so that JAX's JIT cache comparison never calls array __eq__ on statics.

    Field ordering: required fields (no default) must precede defaulted fields.
    """
    # ---- Static required: scalars and callables (pytree_node=False) ----
    num_horizons: int = struct.field(pytree_node=False)
    num_buyers: int = struct.field(pytree_node=False)
    num_goods: int = struct.field(pytree_node=False)
    init_state_function: Callable = struct.field(pytree_node=False)
    init_belief_state_function: Callable = struct.field(pytree_node=False)
    transition_function: Callable = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    sample_types_fn: Callable = struct.field(pytree_node=False)

    # ---- Leaf required: JAX arrays ----
    allocation_cost: jnp.ndarray
    types_low: jnp.ndarray
    types_high: jnp.ndarray

    # ---- Static optional: scalars, callables, and haiku nets (pytree_node=False) ----
    get_transition_prob: Optional[Callable] = struct.field(pytree_node=False, default=None)
    num_type_profiles: int = struct.field(pytree_node=False, default=0)
    belief_pub_net: Optional[hk.Transformed] = struct.field(pytree_node=False, default=None)
    belief_priv_net: Optional[hk.Transformed] = struct.field(pytree_node=False, default=None)
    d_pub: int = struct.field(pytree_node=False, default=64)
    d_priv: int = struct.field(pytree_node=False, default=128)
    trans_sigma_no: float = struct.field(pytree_node=False, default=1.0)
    trans_lambda: float = struct.field(pytree_node=False, default=1.0)

    # ---- Leaf optional: JAX arrays and haiku params ----
    buyer_types: Optional[jnp.ndarray] = None
    type_profiles: Optional[jnp.ndarray] = None
    belief_pub_params: Optional[hk.Params] = None
    belief_priv_params: Optional[hk.Params] = None

    # ---- Helpers ----
    def get_initial_state(self, key: jax.random.PRNGKey) -> State:
        return self.init_state_function(self, key)

    def get_initial_belief_state(self, state: State) -> BeliefState:
        return self.init_belief_state_function(self, state)

    def get_next_state(
        self,
        state: State,
        report: jnp.ndarray,
        allocation: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> State:
        t = state.time_step
        new_alloc_hist = state.allocation_history.at[t].set(allocation)
        new_report_hist = state.report_history.at[t].set(report)
        new_type = self.transition_function(self, t, state.current_type, new_alloc_hist, key)
        return State(
            time_step=t + 1,
            allocation_history=new_alloc_hist,
            current_type=new_type,
            report_history=new_report_hist,
        )

    # ---- Public observation token (continuous) ----
    def encode_obs_pub(self, t: int, allocation_t: jnp.ndarray) -> jnp.ndarray:
        T = self.num_horizons
        _t_table = jnp.arange(T + 1, dtype=jnp.float32) / jnp.float32(max(T - 1, 1))
        t_norm = jnp.reshape(_t_table[t], (1,))
        return jnp.concatenate([t_norm, allocation_t.reshape(-1).astype(jnp.float32)], axis=0)

    # ---- Private observation token (continuous) ----
    def encode_obs_priv_i(
        self,
        t: int,
        type_i_next: jnp.ndarray,
        report_it: jnp.ndarray,
    ) -> jnp.ndarray:
        T = self.num_horizons
        _t_table = jnp.arange(T + 1, dtype=jnp.float32) / jnp.float32(max(T - 1, 1))
        t_norm = jnp.reshape(_t_table[t], (1,))
        return jnp.concatenate([
            t_norm,
            type_i_next.reshape(-1).astype(jnp.float32),
            report_it.reshape(-1).astype(jnp.float32),
        ], axis=0)

    # ---- Continuous belief state update (GRU) ----
    def get_next_belief_state_continuous(
        self,
        belief_state: BeliefState,
        report: jnp.ndarray,
        allocation: jnp.ndarray,
        next_state: State,
    ) -> BeliefState:
        t = belief_state.time_step
        new_alloc_hist = belief_state.allocation_history.at[t].set(allocation)
        new_report_hist = belief_state.report_history.at[t].set(report)

        o_pub = self.encode_obs_pub(t, allocation)
        c_next = self.belief_pub_net.apply(self.belief_pub_params, belief_state.pub_embed, o_pub)

        def update_one(i, h_i):
            o_priv = self.encode_obs_priv_i(t, next_state.current_type[i], report[i])
            return self.belief_priv_net.apply(self.belief_priv_params, h_i, o_priv, c_next)

        h_next = jax.vmap(update_one, in_axes=(0, 0))(
            jnp.arange(self.num_buyers), belief_state.priv_embeds
        )
        # Store true type in first G dims of private embed (for truthful prototype)
        h_next = h_next.at[:, :self.num_goods].set(next_state.current_type)

        return belief_state.replace_state(
            t + 1, new_alloc_hist, new_report_hist, c_next, h_next, belief_state.beliefs
        )

    # ---- Discrete belief state update (Bayesian filter) ----
    def get_next_belief_state_discrete(
        self,
        belief_state: BeliefState,
        report: jnp.ndarray,
        allocation: jnp.ndarray,
        next_state: State,
    ) -> BeliefState:
        """Bayes filter update for discrete type beliefs.

        beliefs[i, k] = P(v_{i,t} = type_profiles[k][i] | public history)
        After observing next_state.current_type, update via:
          new_belief[i, k] ∝ P(v_next | v_i=profile_k[i], x_t) * belief[i, k]
        """
        t = belief_state.time_step
        new_alloc_hist = belief_state.allocation_history.at[t].set(allocation)
        new_report_hist = belief_state.report_history.at[t].set(report)

        # For each buyer i and each type profile k, compute transition probability
        # from current belief distribution to next observed type
        def update_buyer_belief(i, belief_i):
            # belief_i: (num_type_profiles,) = P(profile_k | history) for buyer i
            # next observed type for buyer i:
            next_type_i = next_state.current_type[i]  # (G,)

            def trans_prob_k(k):
                profile_k = self.type_profiles[k]   # (B, G)
                current_type_i = profile_k[i]        # (G,)
                # P(next_type_i | current_type_i, allocation_t[i])
                return self.get_transition_prob(
                    self, t, current_type_i, allocation[i], next_type_i
                )

            trans_probs = jax.vmap(trans_prob_k)(jnp.arange(self.num_type_profiles))  # (K,)
            new_belief_i = belief_i * trans_probs
            total = jnp.sum(new_belief_i)
            return new_belief_i / jnp.maximum(total, 1e-12)

        new_beliefs = jax.vmap(update_buyer_belief, in_axes=(0, 0))(
            jnp.arange(self.num_buyers), belief_state.beliefs
        )  # (B, K)

        # Embed next true type in priv_embeds[:, :G] so the deviation policy
        # can condition on the buyer's own current type (mirrors continuous case).
        G = self.num_goods
        new_priv = belief_state.priv_embeds.at[:, :G].set(
            next_state.current_type.astype(jnp.float32)
        )

        return belief_state.replace_state(
            t + 1, new_alloc_hist, new_report_hist,
            belief_state.pub_embed, new_priv, new_beliefs
        )

    @property
    def is_discrete(self) -> bool:
        return self.buyer_types is not None


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

@struct.dataclass
class BoundedContPolicy:
    """Deterministic continuous report policy: BeliefState -> report (B, G)."""
    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(self, belief_state: BeliefState, auction: BanditAuction,
                 key: jax.random.PRNGKey = None, *args):
        return self.network.apply(self.params, belief_state, auction, *args)


@struct.dataclass
class DiscretePolicy:
    """Stochastic discrete report policy: BeliefState -> report in type_profiles."""
    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(self, belief_state: BeliefState, auction: BanditAuction,
                 key: jax.random.PRNGKey = None, *args):
        return self.network.apply(self.params, belief_state, auction, key, *args)


# ---------------------------------------------------------------------------
# GRU Belief encoders (continuous case)
# ---------------------------------------------------------------------------

class PubEncoder(hk.Module):
    def __init__(self, d_pub: int = 64, name=None):
        super().__init__(name=name)
        self.inp = hk.Linear(d_pub)
        self.gru = hk.GRU(d_pub)

    def __call__(self, c: jnp.ndarray, o_pub: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.inp(o_pub))
        c_next, _ = self.gru(x, c)
        return c_next


class PrivEncoder(hk.Module):
    def __init__(self, d_priv: int = 128, num_goods: int = 1, name=None):
        super().__init__(name=name)
        self.inp = hk.Linear(d_priv)
        self.gru = hk.GRU(d_priv)
        self.c_proj = hk.Linear(d_priv)

    def __call__(self, h_i: jnp.ndarray, o_priv_i: jnp.ndarray,
                 c_next: jnp.ndarray) -> jnp.ndarray:
        c_proj = jax.nn.relu(self.c_proj(c_next))
        x = jax.nn.relu(self.inp(jnp.concatenate([o_priv_i, c_proj])))
        h_next, _ = self.gru(x, h_i)
        return h_next


def build_pub_net(d_pub: int = 64) -> hk.Transformed:
    def f(c, o_pub):
        return PubEncoder(d_pub)(c, o_pub)
    return hk.without_apply_rng(hk.transform(f))


def build_priv_net(d_priv: int = 128, num_goods: int = 1) -> hk.Transformed:
    def f(h_i, o_priv_i, c_next):
        return PrivEncoder(d_priv, num_goods)(h_i, o_priv_i, c_next)
    return hk.without_apply_rng(hk.transform(f))


# ---------------------------------------------------------------------------
# Bounded continuous policy head (reused from budget_auction)
# ---------------------------------------------------------------------------

def _normalize_to_unit(t, low, high, eps=1e-6):
    width = jnp.maximum(high - low, 1e-12)
    p = (t - low) / width
    return jnp.clip(p, eps, 1.0 - eps), width

def _logit(p):
    return jnp.log(p) - jnp.log1p(-p)

def _inv_logit(s):
    return jax.nn.sigmoid(s)


class LogitProtoResidualHead(hk.Module):
    """Bounded continuous report head. Residual in logit space around truthful prototype."""
    def __init__(self, d_priv, d_pub, G, hidden=128,
                 residual_init_std=0.0, cproj_init_std=0.0,
                 stop_grad_truth=True, eps=1e-6, name=None):
        super().__init__(name=name)
        self.d_priv = d_priv
        self.d_pub = d_pub
        self.G = G
        self.hidden = hidden
        self.residual_init_std = residual_init_std
        self.cproj_init_std = cproj_init_std
        self.stop_grad_truth = stop_grad_truth
        self.eps = eps
        self.trunk = hk.nets.MLP([hidden, hidden], activate_final=True)

    def __call__(self, belief_state: BeliefState, types_low, types_high):
        G = self.G
        H = belief_state.priv_embeds   # (B, d_priv)
        c = belief_state.pub_embed     # (d_pub,)

        H_type = H[:, :G]
        if self.stop_grad_truth:
            H_type = jax.lax.stop_gradient(H_type)

        zero = hk.initializers.Constant(0.0)
        cproj_init = (hk.initializers.RandomNormal(self.cproj_init_std)
                      if self.cproj_init_std > 0 else zero)
        c_proj = hk.Linear(self.d_priv, w_init=cproj_init, b_init=zero)(c)
        x = jnp.concatenate([H, jnp.broadcast_to(c_proj, H.shape)], axis=-1)

        z = self.trunk(x)
        res_init = (hk.initializers.RandomNormal(self.residual_init_std)
                    if self.residual_init_std > 0 else zero)
        Delta = hk.Linear(G, w_init=res_init, b_init=zero)(z)

        p0, width = _normalize_to_unit(H_type, types_low, types_high, eps=self.eps)
        s = _logit(p0) + Delta
        p = _inv_logit(s)
        return types_low + width * p   # (B, G) in [low, high]


def build_bounded_policy_net(d_priv, d_pub, G, hidden=128,
                              residual_init_std=0.0, cproj_init_std=0.0,
                              stop_grad_truth=True, eps=1e-6) -> hk.Transformed:
    def f(belief_state, auction):
        head = LogitProtoResidualHead(
            d_priv, d_pub, G, hidden, residual_init_std, cproj_init_std,
            stop_grad_truth, eps
        )
        return head(belief_state, auction.types_low, auction.types_high)
    return hk.without_apply_rng(hk.transform(f))


# ---------------------------------------------------------------------------
# Discrete policy head
# ---------------------------------------------------------------------------

class DiscreteProtoResidualHead(hk.Module):
    """Discrete report policy with truthful-prototype initialization.

    logits[b, t] = k * one_hot_true_type[b, t] + residual(priv_embeds)[b, t]

    - one_hot_true_type is derived from priv_embeds[:, :G] (stores current true type)
      matched against buyer_types (B, T, G); exact float equality holds because both
      originate from the same discrete type array.
    - k (proto_strength) >= 0 via softplus; initialized at proto_strength_init.
    - residual last layer is zero-init => exact truthful start when residual_init_std=0.
    - Output: (B, G) expected report via Gumbel-softmax over T per-buyer types.
    """
    def __init__(self, d_priv, d_pub, hidden=128,
                 proto_strength_init=10.0, residual_init_std=0.0,
                 stop_grad_truth=True, name=None):
        super().__init__(name=name)
        self.d_priv = d_priv
        self.d_pub = d_pub
        self.hidden = hidden
        self.proto_strength_init = float(proto_strength_init)
        self.residual_init_std = float(residual_init_std)
        self.stop_grad_truth = stop_grad_truth
        self.trunk = hk.nets.MLP([hidden, hidden], activate_final=True)

    def __call__(self, belief_state: BeliefState, auction: BanditAuction,
                 key: jax.random.PRNGKey):
        B = auction.num_buyers
        G = auction.num_goods
        # buyer_types: (B, T_per_buyer, G)
        buyer_types = auction.buyer_types          # (B, T, G)
        T = buyer_types.shape[1]

        # ---- Truthful proto ----
        # priv_embeds[:, :G] is set to current_type in every belief state update.
        # Values come from buyer_types[b, idx, :] so float equality holds exactly.
        H_type = belief_state.priv_embeds[:, :G]  # (B, G)
        if self.stop_grad_truth:
            H_type = jax.lax.stop_gradient(H_type)

        # match[b, t] = 1 iff buyer b's true type equals buyer_types[b, t, :]
        match = jnp.all(
            H_type[:, None, :] == buyer_types,    # (B, 1, G) broadcast vs (B, T, G)
            axis=-1,
        ).astype(jnp.float32)                     # (B, T)

        # Learnable proto strength >= 0 via softplus; init so softplus(raw) ≈ proto_strength_init
        k_raw = hk.get_parameter(
            "proto_strength_raw", shape=(1,),
            init=hk.initializers.Constant(
                float(jnp.log(jnp.expm1(jnp.array(self.proto_strength_init))))
            ),
        )
        k = jax.nn.softplus(k_raw)                # scalar >= 0
        proto_logits = k * match                  # (B, T)

        # ---- Residual ----
        H = belief_state.priv_embeds              # (B, d_priv)
        c = belief_state.pub_embed                # (d_pub,) — zeros for discrete

        zero = hk.initializers.Constant(0.0)
        c_proj = hk.Linear(self.d_priv, w_init=zero, b_init=zero)(c)
        x = jnp.concatenate([H, jnp.broadcast_to(c_proj, H.shape)], axis=-1)
        z = self.trunk(x)                         # (B, hidden)

        res_init = (hk.initializers.RandomNormal(self.residual_init_std)
                    if self.residual_init_std > 0 else zero)
        residual_logits = hk.Linear(T, w_init=res_init, b_init=zero)(z)  # (B, T)

        logits = proto_logits + residual_logits   # (B, T)

        # ---- Straight-through Gumbel-softmax ----
        if key is not None:
            gumbel_noise = -jnp.log(-jnp.log(
                jax.random.uniform(key, shape=logits.shape, minval=1e-10, maxval=1.0)
            ))
            soft = jax.nn.softmax((logits + gumbel_noise) / 1.0, axis=-1)
        else:
            soft = jax.nn.softmax(logits, axis=-1)

        hard_idx = jnp.argmax(soft, axis=-1)      # (B,)
        hard = jax.nn.one_hot(hard_idx, T)        # (B, T)
        report_probs = soft + jax.lax.stop_gradient(hard - soft)  # straight-through

        # Expected report: sum_t p_t * buyer_types[b, t, :]
        report = jnp.einsum('bt,btg->bg', report_probs, buyer_types)  # (B, G)
        return report


def build_discrete_policy_net(d_priv, d_pub, hidden=128,
                               proto_strength_init=10.0,
                               residual_init_std=0.0,
                               stop_grad_truth=True) -> hk.Transformed:
    def f(belief_state, auction, key=None):
        return DiscreteProtoResidualHead(
            d_priv, d_pub, hidden,
            proto_strength_init, residual_init_std, stop_grad_truth,
        )(belief_state, auction, key)
    return hk.without_apply_rng(hk.transform(f))


# ---------------------------------------------------------------------------
# State initialization helpers
# ---------------------------------------------------------------------------

def bandit_init_state(auction: BanditAuction, key: jax.random.PRNGKey) -> State:
    initial_types = auction.sample_types_fn(key)
    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    return State(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G)),
        current_type=initial_types,
        report_history=jnp.zeros((T, B, G)),
    )


def bandit_init_belief_state_continuous(
    auction: BanditAuction, initial_state: State
) -> BeliefState:
    B, G, T = auction.num_buyers, auction.num_goods, auction.num_horizons
    K = auction.num_type_profiles  # 0 for continuous

    c0 = jnp.zeros((auction.d_pub,), jnp.float32)

    def init_priv(i):
        h0_i = jnp.zeros((auction.d_priv,), jnp.float32)
        type0_i = initial_state.current_type[i].astype(jnp.float32)
        report0_i = jnp.zeros((G,), jnp.float32)
        tok0 = auction.encode_obs_priv_i(0, type0_i, report0_i)
        h = jnp.asarray(
            auction.belief_priv_net.apply(auction.belief_priv_params, h0_i, tok0, c0)
        )
        return h

    h0 = jax.vmap(lambda i: init_priv(i))(jnp.arange(B))
    h0 = h0.at[:, :G].set(initial_state.current_type)

    return BeliefState(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G), jnp.float32),
        report_history=jnp.zeros((T, B, G), jnp.float32),
        pub_embed=c0,
        priv_embeds=h0,
        beliefs=jnp.zeros((B, max(K, 1)), jnp.float32),
    )


def bandit_init_belief_state_discrete(
    auction: BanditAuction, initial_state: State
) -> BeliefState:
    B, G, T = auction.num_buyers, auction.num_goods, auction.num_horizons
    K = auction.num_type_profiles

    # Uniform initial beliefs
    beliefs = jnp.ones((B, K), jnp.float32) / K

    # Embed initial true type in priv_embeds[:, :G] so the deviation policy
    # can condition on the buyer's own type (mirrors continuous case).
    priv_embeds = jnp.zeros((B, auction.d_priv), jnp.float32).at[:, :G].set(
        initial_state.current_type.astype(jnp.float32)
    )

    return BeliefState(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G), jnp.float32),
        report_history=jnp.zeros((T, B, G), jnp.float32),
        pub_embed=jnp.zeros((auction.d_pub,), jnp.float32),
        priv_embeds=priv_embeds,
        beliefs=beliefs,
    )


# ---------------------------------------------------------------------------
# Type samplers
# ---------------------------------------------------------------------------

def make_box_sampler(types_low: jnp.ndarray, types_high: jnp.ndarray):
    types_low = jnp.asarray(types_low, dtype=jnp.float32)
    types_high = jnp.asarray(types_high, dtype=jnp.float32)
    def sample_types_fn(key):
        u = jax.random.uniform(key, shape=types_low.shape)
        return types_low + (types_high - types_low) * u
    return sample_types_fn


def make_gaussian_sampler(mean: jnp.ndarray, std: jnp.ndarray):
    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.asarray(std, dtype=jnp.float32)
    def sample_types_fn(key):
        return mean + std * jax.random.normal(key, shape=mean.shape)
    return sample_types_fn


def make_discrete_sampler(type_profiles: jnp.ndarray):
    """Sample a random joint type profile (uniform over all profiles)."""
    K = type_profiles.shape[0]
    def sample_types_fn(key):
        idx = jax.random.randint(key, shape=(), minval=0, maxval=K)
        return type_profiles[idx]  # (B, G)
    return sample_types_fn


# ---------------------------------------------------------------------------
# Transition functions
# ---------------------------------------------------------------------------

def continuous_transition_fn(
    auction: BanditAuction,
    time_step: int,
    types: jnp.ndarray,          # (B, G)
    allocation_history: jnp.ndarray,  # (T, B, G)
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Logit-space Gaussian transition with allocation-dependent variance.

    sigma_i = (1 - x_i) * sigma_no + x_i / (lambda * (1 + alpha_i))
    alpha_i = 0.5 * sum_{tau < t} x_{i, tau}  (allocation accumulator)

    Applied per-good independently.
    """
    low = auction.types_low    # (B, G)
    high = auction.types_high  # (B, G)
    sigma_no = auction.trans_sigma_no
    lam = auction.trans_lambda

    # Allocation accumulator: 0.5 * sum of past allocations per buyer per good
    alpha_it = 0.5 * jnp.sum(allocation_history, axis=0)  # (B, G)
    x_t = allocation_history[time_step]  # (B, G) — just-set allocation at time t

    sigma = (1.0 - x_t) * sigma_no + x_t / (lam * (1.0 + alpha_it))

    # Logit-space transform
    eps = 1e-6
    width = jnp.maximum(high - low, 1e-12)
    p = jnp.clip((types - low) / width, eps, 1.0 - eps)
    s = jnp.log(p) - jnp.log1p(-p)

    noise = jax.random.normal(key, shape=s.shape)
    s_next = s + sigma * noise
    p_next = jax.nn.sigmoid(s_next)
    return low + width * p_next   # (B, G)


def discrete_transition_fn(
    auction: BanditAuction,
    time_step: int,
    types: jnp.ndarray,               # (B, G)
    allocation_history: jnp.ndarray,  # (T, B, G)
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Sample next type from the discrete Markov chain.

    Transition probs computed per-buyer via get_transition_prob,
    then one profile is sampled.
    """
    x_t = allocation_history[time_step]  # (B, G)

    def sample_one_buyer(i, key_i):
        current_type_i = types[i]   # (G,)
        alloc_i = x_t[i]            # (G,)
        # Compute P(v_next | current, alloc) for each type value of buyer i
        buyer_types_i = auction.buyer_types[i]  # (K_i, G) individual types
        K_i = buyer_types_i.shape[0]

        def prob_k(k):
            next_type_k = buyer_types_i[k]
            return auction.get_transition_prob(
                auction, time_step, current_type_i, alloc_i, next_type_k
            )

        probs = jax.vmap(prob_k)(jnp.arange(K_i))
        probs = probs / jnp.maximum(jnp.sum(probs), 1e-12)
        idx = jax.random.choice(key_i, K_i, p=probs)
        return buyer_types_i[idx]  # (G,)

    keys = jax.random.split(key, auction.num_buyers)
    new_types = jax.vmap(sample_one_buyer)(jnp.arange(auction.num_buyers), keys)
    return new_types  # (B, G)


# ---------------------------------------------------------------------------
# Auction Simulator
# ---------------------------------------------------------------------------

class AuctionSimulator:
    """Static-method rollout utilities for bandit auction experiments."""

    @staticmethod
    @jax.jit
    def get_rewards(
        auction: BanditAuction,
        state: State,
        allocation: jnp.ndarray,
        payment: jnp.ndarray,
    ) -> jnp.ndarray:
        """Instantaneous buyer reward: v·x - p.  Shape: (B,)."""
        return jnp.sum(allocation * state.current_type, axis=-1) - payment

    @staticmethod
    @jax.jit
    def get_profit(
        auction: BanditAuction,
        state: State,
        allocation: jnp.ndarray,
        payment: jnp.ndarray,
    ) -> float:
        """Instantaneous seller profit: sum(p) - alloc_cost."""
        return jnp.sum(payment) - jnp.sum(allocation * auction.allocation_cost)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def sample_init_states(
        auction: BanditAuction, num_samples: int, key: jax.random.PRNGKey
    ) -> State:
        keys = jax.random.split(key, num_samples)
        return jax.vmap(auction.get_initial_state)(keys)

    # ---- Truthful profit rollout ----

    @staticmethod
    @jax.jit
    def sample_cumulative_profit(
        auction: BanditAuction, state: State, mechanism, key: jax.random.PRNGKey
    ) -> float:
        discount = auction.discount

        def step(t, carry):
            cumul, s, k = carry
            k, k_next = jax.random.split(k)
            report = s.current_type
            alloc, pay = mechanism(s, report, auction)
            profit = AuctionSimulator.get_profit(auction, s, alloc, pay)
            cumul = cumul + jnp.power(discount, t) * profit
            s_next = auction.get_next_state(s, report, alloc, k_next)
            return (cumul, s_next, k)

        cumul, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, (0.0, state, key))
        return cumul

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_profit(
        auction: BanditAuction, mechanism, num_samples: int, key: jax.random.PRNGKey
    ) -> float:
        k_init, k_roll = jax.random.split(key)
        init_states = AuctionSimulator.sample_init_states(auction, num_samples, k_init)
        roll_keys = jax.random.split(k_roll, num_samples)
        return jnp.mean(jax.vmap(
            lambda s, k: AuctionSimulator.sample_cumulative_profit(auction, s, mechanism, k)
        )(init_states, roll_keys))

    # ---- Truthful rewards rollout (for IR) ----

    @staticmethod
    @jax.jit
    def sample_cumulative_rewards(
        auction: BanditAuction, state: State, mechanism, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Cumulative discounted rewards per buyer under truthful play. Shape: (B,)."""
        discount = auction.discount

        def step(t, carry):
            cumul, s, k = carry
            k, k_next = jax.random.split(k)
            report = s.current_type
            alloc, pay = mechanism(s, report, auction)
            rewards = AuctionSimulator.get_rewards(auction, s, alloc, pay)
            cumul = cumul + jnp.power(discount, t) * rewards
            s_next = auction.get_next_state(s, report, alloc, k_next)
            return (cumul, s_next, k)

        init = (jnp.zeros((auction.num_buyers,)), state, key)
        cumul, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, init)
        return cumul

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_rewards(
        auction: BanditAuction, mechanism, num_samples: int, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        k_init, k_roll = jax.random.split(key)
        init_states = AuctionSimulator.sample_init_states(auction, num_samples, k_init)
        roll_keys = jax.random.split(k_roll, num_samples)
        return jnp.mean(jax.vmap(
            lambda s, k: AuctionSimulator.sample_cumulative_rewards(auction, s, mechanism, k)
        )(init_states, roll_keys), axis=0)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_relu_cumulative_rewards(
        auction: BanditAuction, mechanism, num_samples: int, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Mean of relu(-rewards): IR violation magnitude. Shape: (B,)."""
        k_init, k_roll = jax.random.split(key)
        init_states = AuctionSimulator.sample_init_states(auction, num_samples, k_init)
        roll_keys = jax.random.split(k_roll, num_samples)

        def fn(s, k):
            rewards = AuctionSimulator.sample_cumulative_rewards(auction, s, mechanism, k)
            return jax.nn.relu(-rewards)

        return jnp.mean(jax.vmap(fn)(init_states, roll_keys), axis=0)

    # ---- Regret rollout (B+1 parallel trajectories, same as budget_auction) ----

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("update_belief_fn",))
    def sample_cumulative_regret(
        auction: BanditAuction,
        state: State,
        belief_state: BeliefState,
        mechanism,
        policy,
        key: jax.random.PRNGKey,
        update_belief_fn: Callable = None,
    ) -> float:
        """Cumulative BNE regret via B+1 separate trajectories.

        Runs B+1 parallel trajectories all starting from the same initial state:
          - Truthful: all buyers report true type v_{i,t}
          - Deviation i: buyer i uses oracle policy, all others report truthfully

        Regret = sum_i [ U_i(dev_i, truth_{-i}) - U_i(truth) ]

        Key invariant: all B+1 trajectories use the SAME transition noise key at
        each step. This ensures that when the oracle is truthful (same report =>
        same allocation), the trajectories evolve identically and regret = 0 exactly.
        Variance only comes from genuine allocation differences due to deviations.

        update_belief_fn: belief update method (continuous GRU or discrete Bayes).
        """
        if update_belief_fn is None:
            update_belief_fn = auction.get_next_belief_state_continuous

        discount = auction.discount
        B = auction.num_buyers

        def step(t, carry):
            cumul, dev_states, dev_beliefs, truth_state, k = carry

            # One policy key + one SHARED transition key for all B+1 trajectories
            k, k_policy, k_trans = jax.random.split(k, 3)

            # Truthful step
            truth_types = truth_state.current_type
            truth_alloc, truth_pay = mechanism(truth_state, truth_types, auction)
            truth_rewards = AuctionSimulator.get_rewards(
                auction, truth_state, truth_alloc, truth_pay
            )
            # Shared k_trans: when oracle is truthful, truth and dev trajectories
            # have the same alloc => same sigma => same next types from the same noise
            next_truth = auction.get_next_state(truth_state, truth_types, truth_alloc, k_trans)

            # Per-buyer deviation steps (all use same k_trans)
            dev_rewards_list = []
            next_dev_states = []
            next_dev_beliefs = []

            for i in range(B):
                all_reports_i = policy(dev_beliefs[i], auction, k_policy)
                report_i = dev_states[i].current_type.at[i].set(all_reports_i[i])

                alloc_i, pay_i = mechanism(dev_states[i], report_i, auction)
                util_i = jnp.sum(alloc_i[i] * dev_states[i].current_type[i]) - pay_i[i]
                dev_rewards_list.append(util_i)

                ns_i = auction.get_next_state(dev_states[i], report_i, alloc_i, k_trans)
                nb_i = update_belief_fn(dev_beliefs[i], report_i, alloc_i, ns_i)
                next_dev_states.append(ns_i)
                next_dev_beliefs.append(nb_i)

            dev_rewards = jnp.stack(dev_rewards_list)
            cumul = cumul + jnp.power(discount, t) * jnp.sum(dev_rewards - truth_rewards)

            return (cumul, next_dev_states, next_dev_beliefs, next_truth, k)

        init_dev_states = [state for _ in range(B)]
        init_dev_beliefs = [belief_state for _ in range(B)]
        init = (0.0, init_dev_states, init_dev_beliefs, state, key)
        cumul, _, _, _, _ = jax.lax.fori_loop(0, auction.num_horizons, step, init)
        return cumul

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("num_samples", "update_belief_fn"))
    def estimate_cumulative_regret(
        auction: BanditAuction,
        mechanism,
        policy,
        num_samples: int,
        key: jax.random.PRNGKey,
        update_belief_fn: Callable = None,
    ) -> float:
        k_init, k_roll = jax.random.split(key)
        init_states = AuctionSimulator.sample_init_states(auction, num_samples, k_init)
        roll_keys = jax.random.split(k_roll, num_samples)

        if update_belief_fn is None:
            update_belief_fn = auction.get_next_belief_state_continuous

        def fn(s, k):
            bs = auction.get_initial_belief_state(s)
            return AuctionSimulator.sample_cumulative_regret(
                auction, s, bs, mechanism, policy, k, update_belief_fn
            )

        return jnp.mean(jax.vmap(fn)(init_states, roll_keys))


# ---------------------------------------------------------------------------
# Convenience: summary stats
# ---------------------------------------------------------------------------

def _summary_stats(x: jnp.ndarray) -> Dict:
    return {
        "mean": float(jnp.mean(x)),
        "std": float(jnp.std(x)),
        "min": float(jnp.min(x)),
        "max": float(jnp.max(x)),
        "median": float(jnp.median(x)),
    }
