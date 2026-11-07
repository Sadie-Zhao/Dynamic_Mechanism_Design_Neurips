"""
input_adapters.py — Three mechanism network variants for budget auction experiments.

Each variant takes (BudgetState, report, BudgetAuction) and returns
(alloc_probs_with_null: (B+1, G), z: (B,)) where z is a raw payment score.

Variant A  — GBAM / reported-balance compression (alpha, beta, t/T + current report)
Variant B  — Richer mechanism-history compression (A + past alloc + past payments)
Variant C  — B plus masked, time-indexed attention summaries of legal history

Architecture closely mirrors bandit _mechanism_network:
  - Per-buyer feature encoding (shared MLP weights via vmap)
  - Attention pooling for sequence features in C; cumulative sums in B
  - Allocation head: softmax over (B+1, G) logits
  - Payment head: one scalar per buyer (raw z score, NOT final payment)

All three use the same allocation/payment head structure for fair comparison.
"""

import jax
import jax.numpy as jnp
import haiku as hk

from budget_auction.core import BudgetState, BudgetAuction

# ---------------------------------------------------------------------------
# Shared allocation and payment heads
# ---------------------------------------------------------------------------

def _alloc_head(h: jnp.ndarray, num_buyers: int, num_goods: int) -> jnp.ndarray:
    """Allocation head: h (B, d_hidden) -> alloc_probs (B+1, G) with softmax over buyers+null.

    Each buyer gets an independent score per good via a shared linear projection.
    A learnable null-slot score per good captures the unallocated probability mass.
    Permutation equivariant: permuting buyers permutes scores and thus permutes alloc_probs.
    """
    scores = hk.Linear(num_goods)(h)                  # (B, G): per-buyer score per good
    null_scores = hk.get_parameter(
        'null_scores', [1, num_goods],
        init=hk.initializers.Constant(0.)
    )                                                   # (1, G): learnable null-slot score
    alloc_logits = jnp.concatenate([scores, null_scores], axis=0)  # (B+1, G)
    return jax.nn.softmax(alloc_logits, axis=0)        # (B+1, G), sums to 1 over buyers+null


def _payment_head(h: jnp.ndarray) -> jnp.ndarray:
    """Payment head: h (B, d_hidden) -> z (B,) raw scores."""
    z = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(1),
    ])(h)                                              # (B, 1)
    return z.squeeze(-1)                               # (B,)


def _attention_pool(seq: jnp.ndarray, out_dim: int, time_step) -> jnp.ndarray:
    """Attention-weighted pooling over a time sequence.

    Args:
        seq: (T, B, d_in) sequence of per-buyer features
        out_dim: output feature dimension

    Returns:
        (B, out_dim) weighted sum of sequence features
    """
    T, B, _ = seq.shape
    past = jnp.arange(T) < time_step
    seq = jnp.where(past[:, None, None], seq, 0.0)
    # Time is observable. Preserve it rather than treating a history as a bag.
    positions = jnp.broadcast_to(
        (jnp.arange(T) / max(1, T - 1))[:, None, None], (T, B, 1))
    seq = jnp.concatenate([seq, positions], axis=-1)
    d = seq.shape[-1]
    # Project each timestep to out_dim
    flat = seq.reshape(T * B, d)
    proj = hk.Sequential([
        hk.Linear(out_dim), jax.nn.relu,
    ])(flat).reshape(T, B, out_dim)

    # Attention weights (scalar per timestep per buyer)
    attn_raw = hk.Linear(1)(proj)                     # (T, B, 1)
    attn_raw = jnp.where(past[:, None, None], attn_raw, -1e30)
    attn_w = jax.nn.softmax(attn_raw, axis=0) * past[:, None, None]
    return jnp.sum(proj * attn_w, axis=0)              # (B, out_dim)


# ---------------------------------------------------------------------------
# Variant A: GBAM / reported-balance compression
# ---------------------------------------------------------------------------

def mechanism_network_A(
    state: BudgetState,
    report: jnp.ndarray,       # (B, G)
    auction: BudgetAuction,
):
    """
    Mechanism input: current report + (alpha, beta, t/T).
    No explicit history. The mechanism computes alpha from reports and outcomes;
    alpha is not generally public information available to other buyers.
    """
    B, G = auction.num_buyers, auction.num_goods
    t_norm = jnp.array(
        [state.time_step / max(1, auction.num_horizons - 1)], dtype=jnp.float32
    )  # (1,)

    # Encode current report (B, G) -> (B, 64)
    h_report = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),
    ])(report.reshape(B, G))                           # (B, 64)

    # Per-buyer mechanism-ledger features: [alpha_i/B_i, beta_i/B_i, t/T]
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)  # (B, 1)
    beta_norm = (state.beta / auction.initial_budgets).reshape(B, 1)    # (B, 1)
    t_feat = jnp.broadcast_to(t_norm, (B, 1))                          # (B, 1)
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))      # (B, 32)

    h = jnp.concatenate([h_report, h_state], axis=-1)                  # (B, 96)
    h = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
    ])(h)                                                               # (B, 64)

    alloc_probs = _alloc_head(h, B, G)                                 # (B+1, G)
    z = _payment_head(h)                                               # (B,)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant B: Richer mechanism-history compression
# ---------------------------------------------------------------------------

def mechanism_network_B(
    state: BudgetState,
    report: jnp.ndarray,
    auction: BudgetAuction,
):
    """
    Mechanism input: Variant A features + aggregated past allocation + payment history.
    """
    B, G = auction.num_buyers, auction.num_goods
    t_norm = jnp.array(
        [state.time_step / max(1, auction.num_horizons - 1)], dtype=jnp.float32
    )

    # --- Report features ---
    h_report = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),
    ])(report.reshape(B, G))                                           # (B, 64)

    # --- Public balance features ---
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)
    beta_norm = (state.beta / auction.initial_budgets).reshape(B, 1)
    t_feat = jnp.broadcast_to(t_norm, (B, 1))
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))     # (B, 32)

    # --- Aggregated allocation history: sum over time -> (B, G) ---
    total_alloc = jnp.sum(state.allocation_history, axis=0)           # (B, G)
    h_alloc = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(total_alloc)                                                    # (B, 32)

    # --- Aggregated payment history: sum over time -> (B,) ---
    total_pay = jnp.sum(state.payment_history, axis=0)                # (B,)
    h_pay = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(15 if G == 1 else 18),
    ])(total_pay.reshape(B, 1))                                        # (B, 15 or 18)

    h = jnp.concatenate([h_report, h_state, h_alloc, h_pay], axis=-1) # (B, 143 or 146)
    h = hk.Sequential([
        hk.Linear(160 if G == 1 else 143), jax.nn.relu,
        hk.Linear(82 if G == 1 else 98), jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant C: Full history (designed for small T, e.g. T=5)
# ---------------------------------------------------------------------------

def mechanism_network_C(
    state: BudgetState,
    report: jnp.ndarray,
    auction: BudgetAuction,
):
    """
    Mechanism input: Variant B features plus report/allocation/payment history
    compressed with masked, time-indexed attention. Intended for T=5.
    This is a learned compression, not an injective full-history representation.
    """
    B, G = auction.num_buyers, auction.num_goods
    T = auction.num_horizons
    t_norm = jnp.array(
        [state.time_step / max(1, auction.num_horizons - 1)], dtype=jnp.float32
    )

    # --- Report features ---
    h_report = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),
    ])(report.reshape(B, G))                                           # (B, 64)

    # --- Public balance features ---
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)
    beta_norm = (state.beta / auction.initial_budgets).reshape(B, 1)
    t_feat = jnp.broadcast_to(t_norm, (B, 1))
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))     # (B, 32)

    # --- Full report history: (T, B, G) -> attention -> (B, 32) ---
    h_rh = _attention_pool(state.report_history, out_dim=32, time_step=state.time_step)

    # --- Full allocation history: (T, B, G) -> attention -> (B, 32) ---
    h_ah = _attention_pool(state.allocation_history, out_dim=32, time_step=state.time_step)

    # --- Full payment history: (T, B) -> (T, B, 1) -> attention -> (B, 16) ---
    pay_seq = state.payment_history.reshape(T, B, 1)                   # (T, B, 1)
    h_ph = _attention_pool(pay_seq, out_dim=16, time_step=state.time_step)

    # Preserve explicit aggregate history features alongside attention.
    # Capacity-matched widths do not imply containment of the resized B family.
    h_alloc = hk.Sequential([hk.Linear(64), jax.nn.relu, hk.Linear(32)])(
        jnp.sum(state.allocation_history, axis=0))
    h_pay = hk.Sequential([hk.Linear(32), jax.nn.relu, hk.Linear(16)])(
        jnp.sum(state.payment_history, axis=0).reshape(B, 1))

    h = jnp.concatenate([h_report, h_state, h_alloc, h_pay, h_rh, h_ah, h_ph], axis=-1)
    h = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant A_wide: Same information as A, deeper/wider final MLP (capacity control)
# ---------------------------------------------------------------------------

def mechanism_network_A_wide(
    state: BudgetState,
    report: jnp.ndarray,       # (B, G)
    auction: BudgetAuction,
):
    """
    Same inputs as Variant A (GBAM compression only: report + alpha + beta + t/T).
    Final MLP widths are (187,86,66) for e1 and (193,87,58) for e2.
    Total active parameters match C within each setting; see sealed capacity review.

    Capacity diagnostic only: one width comparison cannot rule out all capacity
    or optimization effects, and does not establish equality of global optima.
    """
    B, G = auction.num_buyers, auction.num_goods
    t_norm = jnp.array(
        [state.time_step / max(1, auction.num_horizons - 1)], dtype=jnp.float32
    )

    # Encode current report (B, G) -> (B, 64)
    h_report = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),
    ])(report.reshape(B, G))                                            # (B, 64)

    # Mechanism-ledger features -> (B, 32)
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)
    beta_norm = (state.beta / auction.initial_budgets).reshape(B, 1)
    t_feat = jnp.broadcast_to(t_norm, (B, 1))
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))     # (B, 32)

    h = jnp.concatenate([h_report, h_state], axis=-1)                 # (B, 96)
    # Wider 3-layer MLP: 96 -> 256 -> 128 -> 64
    h = hk.Sequential([
        hk.Linear(187 if G == 1 else 193), jax.nn.relu,
        hk.Linear(86 if G == 1 else 87), jax.nn.relu,
        hk.Linear(66 if G == 1 else 58),  jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant A_plus: A + informationally redundant sum-payment feature
# ---------------------------------------------------------------------------

def mechanism_network_A_plus(
    state: BudgetState,
    report: jnp.ndarray,       # (B, G)
    auction: BudgetAuction,
):
    """
    Variant A with one additional input: total cumulative payments sum_tau p_{i,tau},
    computed as (B_i - beta_i).  This is informationally IDENTICAL to beta_i (already
    in A's inputs) since sum_tau p = B_i - beta_i is a deterministic linear function
    of beta_i and the constant B_i.

    A redundant-feature diagnostic. Revenue differences can reflect both
    optimization and finite-network capacity; they are not a proof of sufficiency.

    The extra feature is encoded with the same MLP structure Variant B uses for its
    aggregated payment history stream, allowing a like-for-like comparison.
    """
    B, G = auction.num_buyers, auction.num_goods
    t_norm = jnp.array(
        [state.time_step / max(1, auction.num_horizons - 1)], dtype=jnp.float32
    )

    # Encode current report (B, G) -> (B, 64)
    h_report = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),
    ])(report.reshape(B, G))                                           # (B, 64)

    # Per-buyer GBAM features: [alpha_i/B_i, beta_i/B_i, t/T] -> (B, 32)
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)
    beta_norm  = (state.beta  / auction.initial_budgets).reshape(B, 1)
    t_feat     = jnp.broadcast_to(t_norm, (B, 1))
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))     # (B, 32)

    # Redundant feature: sum_tau p_{i,tau} / B_i = (B_i - beta_i) / B_i
    # Informationally equivalent to beta_norm but exposes cumulative payments directly.
    sum_pay_norm = (
        (auction.initial_budgets - state.beta) / auction.initial_budgets
    ).reshape(B, 1)                                                    # (B, 1)
    h_pay = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(16),
    ])(sum_pay_norm)                                                   # (B, 16)

    h = jnp.concatenate([h_report, h_state, h_pay], axis=-1)         # (B, 112)
    h = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

_IMPLEMENTABLE_NETWORKS = {
    "A":        mechanism_network_A,
    "B":        mechanism_network_B,
    "C":        mechanism_network_C,
    "A_wide":   mechanism_network_A_wide,
    "A_plus":   mechanism_network_A_plus,
}


def _observable_network(network):
    # Also enforce the boundary during initialization and direct network calls.
    from budget_auction.mechanisms.budget_mechanism import mechanism_observation
    def observed_network(state, report, auction):
        return network(mechanism_observation(state), report, auction)
    return observed_network


# A_cheat is excluded: private types are not legal mechanism inputs and a
# trained clairvoyant network is not a certified revenue upper bound.
VARIANT_NETWORK = {name: _observable_network(network)
                   for name, network in _IMPLEMENTABLE_NETWORKS.items()}
