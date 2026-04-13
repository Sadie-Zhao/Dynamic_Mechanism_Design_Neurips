"""
core.py — Shared data structures, simulators, and network utilities for budget auction experiments.

Closely mirrors the bandit-auction notebook ((final)cont+multi.ipynb), extended with:
  - BudgetState  : adds alpha (utility slack) and beta (remaining budget) to State
  - BudgetAuction: adds initial_budgets field to BanditAuction
  - BudgetMechanism: wraps a raw mechanism network with the payment feasibility transform
  - BudgetSimulator: budget-aware rollout logic (tracks alpha/beta through episodes)
  - BeliefState / PubEncoder / PrivEncoder: reused verbatim; obs encoding extended to include alpha/beta
  - Plotting utilities: reused verbatim from bandit notebook
"""

from typing import Callable, Dict, List, Tuple
import functools

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax import struct

SEED = 42
ZERO_CLIP_MIN = 1e-4

# ---------------------------------------------------------------------------
# Plotting utilities (reused verbatim from bandit notebook)
# ---------------------------------------------------------------------------

def plot_loss_dict(losses: Dict, title: str = ""):
    num_losses = len(losses)
    num_rows = (num_losses - 1) // 3 + 1
    fig, axs = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    for i, (name, loss) in enumerate(losses.items()):
        row, col = i // 3, i % 3
        ax = axs[col] if num_rows == 1 else axs[row, col]
        ax.plot(loss)
        ax.set_title(name, fontsize=8)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_loss_dicts(loss_list: List[Dict], print_iter: int = 1,
                   dict_names: List[str] = None):
    if dict_names is None:
        dict_names = [f"Dict {i}" for i in range(len(loss_list))]
    for d, name in zip(loss_list, dict_names):
        plot_loss_dict(d, title=name)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@struct.dataclass
class BudgetState:
    """Auction state extended with GBAM-style budget tracking variables.

    Dimensions:
      T = num_horizons, B = num_buyers, G = num_goods
    """
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    current_type: jnp.ndarray         # (B, G)  true private types
    report_history: jnp.ndarray       # (T, B, G)
    alpha: jnp.ndarray                # (B,)  cumulative utility slack: sum_t (v*x - p)
    beta: jnp.ndarray                 # (B,)  remaining budget: B_i - sum_t p_t
    payment_history: jnp.ndarray      # (T, B) past scalar payments per buyer

    def replace_budget_state(self, new_time_step, new_allocation_history,
                              new_current_type, new_report_history,
                              new_alpha, new_beta, new_payment_history):
        return self.replace(
            time_step=new_time_step,
            allocation_history=new_allocation_history,
            current_type=new_current_type,
            report_history=new_report_history,
            alpha=new_alpha,
            beta=new_beta,
            payment_history=new_payment_history,
        )

    def replace_buyer_type(self, buyer, buyer_type):
        new_type = self.current_type.at[buyer].set(buyer_type)
        return self.replace(current_type=new_type)


@struct.dataclass
class BeliefState:
    """Per-buyer belief/information state used by the deviation policy.
    Same structure as bandit; the GRU encoders now receive alpha/beta observations.
    """
    time_step: int
    allocation_history: jnp.ndarray   # (T, B, G)
    pub_embed: jnp.ndarray            # (d_pub,)
    priv_embeds: jnp.ndarray          # (B, d_priv)
    report_history: jnp.ndarray       # (T, B, G)

    def replace_state(self, new_time_step, new_allocation_history,
                      new_pub_embed, new_priv_embeds, new_report_history):
        return self.replace(
            time_step=new_time_step,
            allocation_history=new_allocation_history,
            pub_embed=new_pub_embed,
            priv_embeds=new_priv_embeds,
            report_history=new_report_history,
        )


@struct.dataclass
class BudgetAuction:
    """Auction configuration for budget experiments.

    Extends the BanditAuction pattern with initial_budgets (B,).
    All fields that are not JAX arrays are marked pytree_node=False.
    """
    num_horizons: int = struct.field(pytree_node=False)
    num_buyers: int = struct.field(pytree_node=False)
    num_goods: int = struct.field(pytree_node=False)

    # Callables (not JAX pytree nodes)
    init_state_function: Callable = struct.field(pytree_node=False)
    init_belief_state_function: Callable = struct.field(pytree_node=False)
    get_transition_prob: Callable = struct.field(pytree_node=False)
    transition_function: Callable = struct.field(pytree_node=False)

    discount: float = struct.field(pytree_node=False)
    allocation_cost: jnp.ndarray = struct.field(pytree_node=False)

    # Type sampler for initial state
    sample_types_fn: Callable = struct.field(pytree_node=False)
    types_low: jnp.ndarray = struct.field(pytree_node=False)
    types_high: jnp.ndarray = struct.field(pytree_node=False)

    # Budget
    initial_budgets: jnp.ndarray = struct.field(pytree_node=False)  # (B,)

    # Belief encoders (GRU-based, same as bandit)
    belief_pub_net: hk.Transformed = struct.field(pytree_node=False)
    belief_pub_params: hk.Params = struct.field(pytree_node=False)
    belief_priv_net: hk.Transformed = struct.field(pytree_node=False)
    belief_priv_params: hk.Params = struct.field(pytree_node=False)
    d_pub: int = struct.field(pytree_node=False, default=64)
    d_priv: int = struct.field(pytree_node=False, default=128)

    # Transition hyperparams (Exp 2 logit-space Gaussian; harmless defaults for Exp 1)
    trans_base_rate: float = struct.field(pytree_node=False, default=1.0)
    trans_sigma_no: float = struct.field(pytree_node=False, default=0.01)

    # ---- Initial state helpers ----
    def get_initial_state(self, key: jax.random.PRNGKey) -> BudgetState:
        return self.init_state_function(self, key)

    def get_initial_belief_state(self, state: BudgetState) -> BeliefState:
        return self.init_belief_state_function(self, state)

    # ---- State transition ----
    def get_next_budget_state(
        self,
        state: BudgetState,
        report: jnp.ndarray,    # (B, G) reports submitted to mechanism
        allocation: jnp.ndarray, # (B, G) allocation returned
        payment: jnp.ndarray,    # (B,)  payment charged
        key: jax.random.PRNGKey,
    ) -> BudgetState:
        t = state.time_step
        new_time_step = t + 1
        new_allocation_history = state.allocation_history.at[t].set(allocation)
        new_report_history = state.report_history.at[t].set(report)
        new_payment_history = state.payment_history.at[t].set(payment)

        # Type transition (same kernel as bandit)
        new_current_type = self.transition_function(
            self, t, state.current_type, new_allocation_history, key
        )

        # Budget state updates using TRUE types for accurate IR/budget tracking
        true_utility = jnp.sum(state.current_type * allocation, axis=-1)  # (B,)
        new_alpha = state.alpha + true_utility - payment
        new_beta = state.beta - payment

        return BudgetState(
            time_step=new_time_step,
            allocation_history=new_allocation_history,
            current_type=new_current_type,
            report_history=new_report_history,
            alpha=new_alpha,
            beta=new_beta,
            payment_history=new_payment_history,
        )

    # ---- Observation encoders for GRU belief state ----
    def encode_obs_pub(
        self,
        t: int,
        allocation_t: jnp.ndarray,  # (B, G)
        alpha: jnp.ndarray,          # (B,)
        beta: jnp.ndarray,           # (B,)
    ) -> jnp.ndarray:
        """Public observation token: time, allocations, normalised alpha/beta."""
        parts = [
            jnp.array([t / max(1, self.num_horizons - 1)], dtype=jnp.float32),
            allocation_t.reshape(-1).astype(jnp.float32),              # (B*G,)
            (alpha / self.initial_budgets).astype(jnp.float32),        # (B,)
            (beta / self.initial_budgets).astype(jnp.float32),         # (B,)
        ]
        return jnp.concatenate(parts, axis=0)  # (1 + B*G + 2*B,)

    def encode_obs_priv_i(
        self,
        t: int,
        type_i_next: jnp.ndarray,   # (G,)
        report_it: jnp.ndarray,      # (G,)
        alpha_i: jnp.ndarray,        # (1,) or scalar
        beta_i: jnp.ndarray,         # (1,) or scalar
    ) -> jnp.ndarray:
        """Private observation token for buyer i: time, next type, report, budget slack."""
        parts = [
            jnp.array([t / max(1, self.num_horizons - 1)], dtype=jnp.float32),
            type_i_next.reshape(-1).astype(jnp.float32),               # (G,)
            report_it.reshape(-1).astype(jnp.float32),                 # (G,)
            jnp.array([float(alpha_i) / float(self.initial_budgets[0])],
                      dtype=jnp.float32),                               # (1,)
            jnp.array([float(beta_i) / float(self.initial_budgets[0])],
                      dtype=jnp.float32),                               # (1,)
        ]
        return jnp.concatenate(parts, axis=0)  # (1 + 2G + 2,)

    def get_next_belief_state(
        self,
        belief_state: BeliefState,
        report: jnp.ndarray,
        allocation: jnp.ndarray,
        next_state: BudgetState,
    ) -> BeliefState:
        """Update GRU belief state using the new budget state's alpha/beta."""
        t = belief_state.time_step
        new_time_step = t + 1
        new_allocation_history = belief_state.allocation_history.at[t].set(allocation)
        new_report_history = belief_state.report_history.at[t].set(report)

        # Public GRU update (allocation + alpha/beta are public)
        o_pub_t = self.encode_obs_pub(t, allocation, next_state.alpha, next_state.beta)
        c_next = self.belief_pub_net.apply(
            self.belief_pub_params, belief_state.pub_embed, o_pub_t
        )

        # Per-buyer private GRU update
        def update_one(i, h_i):
            type_i_next = next_state.current_type[i]
            report_it = report[i]
            alpha_i = next_state.alpha[i]
            beta_i = next_state.beta[i]
            o_priv_i = self.encode_obs_priv_i(t, type_i_next, report_it, alpha_i, beta_i)
            return self.belief_priv_net.apply(
                self.belief_priv_params, h_i, o_priv_i, c_next
            )

        h_next = jax.vmap(update_one, in_axes=(0, 0))(
            jnp.arange(self.num_buyers), belief_state.priv_embeds
        )
        # Register true type in the first G dims of priv_embed (for policy truthful proto)
        h_next = h_next.at[:, : self.num_goods].set(next_state.current_type)

        return belief_state.replace_state(
            new_time_step, new_allocation_history, c_next, h_next, new_report_history
        )


# ---------------------------------------------------------------------------
# Policy wrappers (reused from bandit notebook)
# ---------------------------------------------------------------------------

@struct.dataclass
class BoundedContPolicy:
    """Deterministic continuous report policy: BeliefState -> report (B,G)."""
    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(self, belief_state: BeliefState, auction: BudgetAuction,
                 key: jax.random.PRNGKey, *args):
        return self.network.apply(self.params, belief_state, auction, *args)


# ---------------------------------------------------------------------------
# GRU Belief encoders (reused verbatim from bandit notebook)
# ---------------------------------------------------------------------------

class PubEncoder(hk.Module):
    """GRU encoder for the public belief state."""
    def __init__(self, d_pub: int = 64, name=None):
        super().__init__(name=name)
        self.inp = hk.Linear(d_pub)
        self.gru = hk.GRU(d_pub)

    def __call__(self, c: jnp.ndarray, o_pub: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.inp(o_pub))
        c_next, _ = self.gru(x, c)
        return c_next


class PrivEncoder(hk.Module):
    """GRU encoder for buyer i's private belief state."""
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


def build_pub_net(d_pub: int = 64):
    def f(c, o_pub):
        return PubEncoder(d_pub)(c, o_pub)
    return hk.without_apply_rng(hk.transform(f))


def build_priv_net(d_priv: int = 128, num_goods: int = 1):
    def f(h_i, o_priv_i, c_next):
        return PrivEncoder(d_priv, num_goods)(h_i, o_priv_i, c_next)
    return hk.without_apply_rng(hk.transform(f))


# ---------------------------------------------------------------------------
# Bounded policy head (reused verbatim from bandit notebook)
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
    """Bounded continuous report head. Residual in logit space around truthful prototype.
    Zero-init residual => exact truthful output at init.
    (Reused verbatim from bandit notebook.)
    """
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
        B, G = belief_state.priv_embeds.shape[0], self.G
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
        s0 = _logit(p0)
        s = s0 + Delta
        p = _inv_logit(s)
        return types_low + width * p  # (B, G) in [low, high]


def build_bounded_policy_net(d_priv, d_pub, G, hidden=128,
                              residual_init_std=0.0, cproj_init_std=0.0,
                              stop_grad_truth=True, eps=1e-6):
    def f(belief_state, auction):
        head = LogitProtoResidualHead(
            d_priv, d_pub, G, hidden, residual_init_std, cproj_init_std,
            stop_grad_truth, eps
        )
        return head(belief_state, auction.types_low, auction.types_high)
    return hk.without_apply_rng(hk.transform(f))


# ---------------------------------------------------------------------------
# State initialization helpers
# ---------------------------------------------------------------------------

def budget_init_state(auction: BudgetAuction, key: jax.random.PRNGKey) -> BudgetState:
    """Initialize BudgetState: random types, zero history, full budget."""
    initial_types = auction.sample_types_fn(key)
    B = auction.num_buyers
    T = auction.num_horizons
    G = auction.num_goods
    return BudgetState(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G)),
        current_type=initial_types,
        report_history=jnp.zeros((T, B, G)),
        alpha=jnp.zeros((B,)),
        beta=auction.initial_budgets.astype(jnp.float32),
        payment_history=jnp.zeros((T, B)),
    )


def budget_init_belief_state(auction: BudgetAuction,
                              initial_state: BudgetState) -> BeliefState:
    """Initialize BeliefState from initial BudgetState. Same logic as bandit,
    but obs encoding includes alpha/beta (both zero at t=0)."""
    B = auction.num_buyers
    G = auction.num_goods
    T = auction.num_horizons

    c0 = jnp.zeros((auction.d_pub,), jnp.float32)

    def init_priv(auction_inst, i, current_type):
        h0_i = jnp.zeros((auction_inst.d_priv,), jnp.float32)
        type0_i = current_type[i].astype(jnp.float32)
        report0_i = jnp.zeros((auction_inst.num_goods,), jnp.float32)
        alpha0_i = 0.0
        beta0_i = auction_inst.initial_budgets[i]
        tok0 = auction_inst.encode_obs_priv_i(0, type0_i, report0_i, alpha0_i, beta0_i)
        return jnp.asarray(
            auction_inst.belief_priv_net.apply(
                auction_inst.belief_priv_params, h0_i, tok0, c0
            )
        )

    h0 = jax.vmap(init_priv, in_axes=(None, 0, None))(
        auction, jnp.arange(B), initial_state.current_type
    )
    # Register true type in the first G dims of private embed
    h0 = h0.at[:, :G].set(initial_state.current_type)

    return BeliefState(
        time_step=0,
        allocation_history=jnp.zeros((T, B, G), jnp.float32),
        pub_embed=c0,
        priv_embeds=h0,
        report_history=jnp.zeros((T, B, G), jnp.float32),
    )


# ---------------------------------------------------------------------------
# Type samplers (reused verbatim from bandit notebook)
# ---------------------------------------------------------------------------

def make_box_sampler(types_low: jnp.ndarray, types_high: jnp.ndarray):
    """Uniform sampler over [types_low, types_high]."""
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


# ---------------------------------------------------------------------------
# Budget Simulator
# ---------------------------------------------------------------------------

class BudgetSimulator:
    """Static-method rollout utilities for budget auction experiments.

    Mirrors AuctionSimulator from the bandit notebook, extended to:
      - track alpha / beta through episodes
      - expose budget_violation_rate() and ir_violation_rate()
    """

    # ---- Single-step helpers ----

    @staticmethod
    @jax.jit
    def get_rewards(
        auction: BudgetAuction,
        state: BudgetState,
        allocation: jnp.ndarray,
        payment: jnp.ndarray,
    ) -> jnp.ndarray:
        """Instantaneous buyer reward: v·x - p.  Shape: (B,)."""
        return jnp.sum(allocation * state.current_type, axis=-1) - payment

    @staticmethod
    @jax.jit
    def get_profit(
        auction: BudgetAuction,
        state: BudgetState,
        allocation: jnp.ndarray,
        payment: jnp.ndarray,
    ) -> float:
        """Instantaneous seller profit: sum(p) - alloc_cost."""
        return jnp.sum(payment) - jnp.sum(allocation * auction.allocation_cost)

    @staticmethod
    @jax.jit
    def get_regret(
        auction: BudgetAuction,
        state: BudgetState,
        mechanism,
        report: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> float:
        """Single-step regret: sum_i [reward_i(deviate) - reward_i(truthful)].

        Each buyer i deviates using `report[i]` while others report truthfully.
        """
        true_types = state.current_type

        # Truthful allocation and reward
        truthful_alloc, truthful_pay = mechanism(state, true_types, auction)
        truthful_rewards = BudgetSimulator.get_rewards(
            auction, state, truthful_alloc, truthful_pay
        )

        # Deviated reports: ith row = [truth_{-i}, report_i]
        def construct_row(i):
            return true_types.at[i].set(report[i])

        deviated_reports = jax.vmap(construct_row)(jnp.arange(auction.num_buyers))
        # For each deviation, get buyer i's reward
        deviated_rewards = jax.vmap(
            lambda dr: BudgetSimulator.get_rewards(
                auction, state, *mechanism(state, dr, auction)
            )
        )(deviated_reports)
        deviated_rewards = jnp.diag(deviated_rewards)  # (B,): buyer i's reward under deviation i

        return jnp.sum(deviated_rewards) - jnp.sum(truthful_rewards)

    # ---- Init helpers ----

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def sample_init_states(
        auction: BudgetAuction,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> BudgetState:
        keys = jax.random.split(key, num_samples)
        return jax.vmap(auction.get_initial_state)(keys)

    # ---- Truthful-play profit rollout ----

    @staticmethod
    @jax.jit
    def sample_cumulative_profit(
        auction: BudgetAuction,
        state: BudgetState,
        mechanism,
        key: jax.random.PRNGKey,
    ) -> float:
        discount = auction.discount

        def episode_step(t, carry):
            cumul_profit, budget_state, key = carry
            key, k_next = jax.random.split(key)

            report = budget_state.current_type   # truthful
            allocation, payment = mechanism(budget_state, report, auction)
            profit = BudgetSimulator.get_profit(auction, budget_state, allocation, payment)
            cumul_profit = cumul_profit + (discount ** t) * profit

            next_state = auction.get_next_budget_state(
                budget_state, report, allocation, payment, k_next
            )
            return (cumul_profit, next_state, key)

        init = (0.0, state, key)
        cumul_profit, _, _ = jax.lax.fori_loop(
            0, auction.num_horizons, episode_step, init
        )
        return cumul_profit

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_profit(
        auction: BudgetAuction,
        mechanism,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> float:
        key_init, key_roll = jax.random.split(key, 2)
        init_states = BudgetSimulator.sample_init_states(auction, num_samples, key_init)
        rollout_keys = jax.random.split(key_roll, num_samples)

        def fn(state, k):
            return BudgetSimulator.sample_cumulative_profit(auction, state, mechanism, k)

        return jnp.mean(jax.vmap(fn)(init_states, rollout_keys))

    # ---- Truthful-play reward rollout (for IR tracking) ----

    @staticmethod
    @jax.jit
    def sample_cumulative_rewards(
        auction: BudgetAuction,
        state: BudgetState,
        mechanism,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Cumulative discounted rewards per buyer under truthful play. Shape: (B,)."""
        discount = auction.discount

        def episode_step(t, carry):
            cumul_rewards, budget_state, key = carry
            key, k_next = jax.random.split(key)

            report = budget_state.current_type
            allocation, payment = mechanism(budget_state, report, auction)
            rewards = BudgetSimulator.get_rewards(auction, budget_state, allocation, payment)
            cumul_rewards = cumul_rewards + (discount ** t) * rewards

            next_state = auction.get_next_budget_state(
                budget_state, report, allocation, payment, k_next
            )
            return (cumul_rewards, next_state, key)

        init = (jnp.zeros((auction.num_buyers,)), state, key)
        cumul_rewards, _, _ = jax.lax.fori_loop(
            0, auction.num_horizons, episode_step, init
        )
        return cumul_rewards

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_rewards(
        auction: BudgetAuction,
        mechanism,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        key_init, key_roll = jax.random.split(key, 2)
        init_states = BudgetSimulator.sample_init_states(auction, num_samples, key_init)
        rollout_keys = jax.random.split(key_roll, num_samples)

        def fn(state, k):
            return BudgetSimulator.sample_cumulative_rewards(auction, state, mechanism, k)

        return jnp.mean(jax.vmap(fn)(init_states, rollout_keys), axis=0)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_relu_cumulative_rewards(
        auction: BudgetAuction,
        mechanism,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Mean of relu(-rewards) per buyer: IR violation magnitude. Shape: (B,)."""
        key_init, key_roll = jax.random.split(key, 2)
        init_states = BudgetSimulator.sample_init_states(auction, num_samples, key_init)
        rollout_keys = jax.random.split(key_roll, num_samples)

        def fn(state, k):
            rewards = BudgetSimulator.sample_cumulative_rewards(auction, state, mechanism, k)
            return jax.nn.relu(-rewards)

        return jnp.mean(jax.vmap(fn)(init_states, rollout_keys), axis=0)

    # ---- Regret rollout (deviation policy) ----

    @staticmethod
    @jax.jit
    def sample_cumulative_regret(
        auction: BudgetAuction,
        state: BudgetState,
        belief_state: BeliefState,
        mechanism,
        policy: BoundedContPolicy,
        key: jax.random.PRNGKey,
    ) -> float:
        discount = auction.discount

        def episode_step(t, carry):
            cumul_regret, (budget_state, belief_state), key = carry
            key, k_report, k_regret, k_step = jax.random.split(key, 4)

            report = policy(belief_state, auction, k_report)
            allocation, payment = mechanism(budget_state, report, auction)
            regret = BudgetSimulator.get_regret(
                auction, budget_state, mechanism, report, k_regret
            )
            cumul_regret = cumul_regret + (discount ** t) * regret

            next_budget_state = auction.get_next_budget_state(
                budget_state, report, allocation, payment, k_step
            )
            next_belief_state = auction.get_next_belief_state(
                belief_state, report, allocation, next_budget_state
            )
            return (cumul_regret, (next_budget_state, next_belief_state), key)

        init = (0.0, (state, belief_state), key)
        cumul_regret, _, _ = jax.lax.fori_loop(
            0, auction.num_horizons, episode_step, init
        )
        return cumul_regret

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_cumulative_regret(
        auction: BudgetAuction,
        mechanism,
        policy: BoundedContPolicy,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> float:
        key_init, key_roll = jax.random.split(key, 2)
        init_states = BudgetSimulator.sample_init_states(auction, num_samples, key_init)
        rollout_keys = jax.random.split(key_roll, num_samples)

        def fn(state, k):
            belief_state = auction.get_initial_belief_state(state)
            return BudgetSimulator.sample_cumulative_regret(
                auction, state, belief_state, mechanism, policy, k
            )

        return jnp.mean(jax.vmap(fn)(init_states, rollout_keys))

    # ---- Budget / IR violation rates ----

    @staticmethod
    @jax.jit
    def sample_violation_flags(
        auction: BudgetAuction,
        state: BudgetState,
        mechanism,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (budget_viol, ir_viol): boolean arrays shaped (B,) for any violation in episode."""
        def episode_step(t, carry):
            budget_viol, ir_viol, budget_state, key = carry
            key, k_next = jax.random.split(key)
            report = budget_state.current_type
            allocation, payment = mechanism(budget_state, report, auction)
            next_state = auction.get_next_budget_state(
                budget_state, report, allocation, payment, k_next
            )
            # Flag if any step has beta < 0 or alpha < 0
            budget_viol = budget_viol | (next_state.beta < -1e-6)
            ir_viol = ir_viol | (next_state.alpha < -1e-6)
            return (budget_viol, ir_viol, next_state, key)

        init_bv = jnp.zeros((auction.num_buyers,), dtype=bool)
        init_iv = jnp.zeros((auction.num_buyers,), dtype=bool)
        init = (init_bv, init_iv, state, key)
        budget_viol, ir_viol, _, _ = jax.lax.fori_loop(
            0, auction.num_horizons, episode_step, init
        )
        return budget_viol, ir_viol

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def estimate_violation_rates(
        auction: BudgetAuction,
        mechanism,
        num_samples: int,
        key: jax.random.PRNGKey,
    ) -> Tuple[float, float]:
        """Returns (budget_violation_rate, ir_violation_rate) averaged over buyers and trajectories."""
        key_init, key_roll = jax.random.split(key, 2)
        init_states = BudgetSimulator.sample_init_states(auction, num_samples, key_init)
        rollout_keys = jax.random.split(key_roll, num_samples)

        def fn(state, k):
            bv, iv = BudgetSimulator.sample_violation_flags(auction, state, mechanism, k)
            return bv.astype(jnp.float32), iv.astype(jnp.float32)

        bvs, ivs = jax.vmap(fn)(init_states, rollout_keys)  # (N, B), (N, B)
        return float(jnp.mean(bvs)), float(jnp.mean(ivs))
