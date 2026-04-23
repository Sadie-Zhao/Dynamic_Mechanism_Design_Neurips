"""
input_adapters.py — Three mechanism network variants for budget auction experiments.

Each variant takes (BudgetState, report, BudgetAuction) and returns
(alloc_probs_with_null: (B+1, G), z: (B,)) where z is a raw payment score.

Variant A  — GBAM / public-balance only (alpha, beta, t/T + current report)
Variant B  — Richer public-history compression (A + past alloc + past payments)
Variant C  — Full history (B + full report history; feasible for T=5)

Architecture closely mirrors bandit _mechanism_network:
  - Per-buyer feature encoding (shared MLP weights via vmap)
  - Attention pooling for sequence features (in B and C)
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
    """Allocation head: h (B, d_hidden) -> alloc_probs (B+1, G) with softmax over buyers."""
    # For each good, project each buyer's hidden rep to a logit
    logits_per_good = []
    for g in range(num_goods):
        raw = hk.Linear(num_buyers + 1)(h)           # (B, B+1)
        logit = jnp.mean(raw, axis=0)                 # (B+1,) average over buyers
        logits_per_good.append(logit)
    alloc_logits = jnp.stack(logits_per_good, axis=1)  # (B+1, G)
    return jax.nn.softmax(alloc_logits, axis=0)          # (B+1, G)


def _payment_head(h: jnp.ndarray) -> jnp.ndarray:
    """Payment head: h (B, d_hidden) -> z (B,) raw scores."""
    z = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(1),
    ])(h)                                              # (B, 1)
    return z.squeeze(-1)                               # (B,)


def _attention_pool(seq: jnp.ndarray, out_dim: int) -> jnp.ndarray:
    """Attention-weighted pooling over a time sequence.

    Args:
        seq: (T, B, d_in) sequence of per-buyer features
        out_dim: output feature dimension

    Returns:
        (B, out_dim) weighted sum of sequence features
    """
    T, B, d = seq.shape
    # Project each timestep to out_dim
    flat = seq.reshape(T * B, d)
    proj = hk.Sequential([
        hk.Linear(out_dim), jax.nn.relu,
    ])(flat).reshape(T, B, out_dim)

    # Attention weights (scalar per timestep per buyer)
    attn_raw = hk.Linear(1)(proj)                     # (T, B, 1)
    attn_w = jax.nn.softmax(attn_raw, axis=0)          # (T, B, 1) softmax over T
    return jnp.sum(proj * attn_w, axis=0)              # (B, out_dim)


# ---------------------------------------------------------------------------
# Variant A: GBAM / public-balance only
# ---------------------------------------------------------------------------

def mechanism_network_A(
    state: BudgetState,
    report: jnp.ndarray,       # (B, G)
    auction: BudgetAuction,
):
    """
    Mechanism input: current report + (alpha, beta, t/T).
    No past history. Corresponds to GBAM-style compressed public state.
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

    # Per-buyer public-state features: [alpha_i/B_i, beta_i/B_i, t/T]
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
# Variant B: Richer public-history compression
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
        hk.Linear(16),
    ])(total_pay.reshape(B, 1))                                        # (B, 16)

    h = jnp.concatenate([h_report, h_state, h_alloc, h_pay], axis=-1) # (B, 144)
    h = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
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
    Mechanism input: Variant A features + full report/allocation/payment history
    via attention pooling. Intended for T=5 (manageable dimension).
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
    h_rh = _attention_pool(state.report_history, out_dim=32)           # (B, 32)

    # --- Full allocation history: (T, B, G) -> attention -> (B, 32) ---
    h_ah = _attention_pool(state.allocation_history, out_dim=32)       # (B, 32)

    # --- Full payment history: (T, B) -> (T, B, 1) -> attention -> (B, 16) ---
    pay_seq = state.payment_history.reshape(T, B, 1)                   # (T, B, 1)
    h_ph = _attention_pool(pay_seq, out_dim=16)                        # (B, 16)

    h = jnp.concatenate([h_report, h_state, h_rh, h_ah, h_ph], axis=-1)  # (B, 176)
    h = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant A_cheat: Cheating oracle — observes true v_t (not implementable)
# ---------------------------------------------------------------------------

def mechanism_network_A_cheat(
    state: BudgetState,
    report: jnp.ndarray,       # (B, G)
    auction: BudgetAuction,
):
    """
    Identical to Variant A but also receives the true private type v_t = state.current_type.
    Not implementable in practice (violates IC by assumption); serves as an upper bound on
    what any mechanism knowing types exactly could achieve in the GBAM input class.

    If A_cheat ≈ A in revenue → A's inputs already saturate available information (optimization
    is the bottleneck, not information).  If A_cheat >> A → information about v_t is genuinely
    useful and A is information-limited, not just capacity-limited.
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

    # Encode true private type v_t (B, G) -> (B, 32)
    h_type = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(state.current_type.reshape(B, G))                               # (B, 32)

    # Public-state features: [alpha_i/B_i, beta_i/B_i, t/T] -> (B, 32)
    alpha_norm = (state.alpha / auction.initial_budgets).reshape(B, 1)
    beta_norm = (state.beta / auction.initial_budgets).reshape(B, 1)
    t_feat = jnp.broadcast_to(t_norm, (B, 1))
    h_state = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(32),
    ])(jnp.concatenate([alpha_norm, beta_norm, t_feat], axis=-1))     # (B, 32)

    h = jnp.concatenate([h_report, h_type, h_state], axis=-1)         # (B, 128)
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
    Final MLP is widened to 96 -> 256 -> 128 -> 64, giving parameter count comparable to
    Variant C's 176 -> 128 -> 64 final block.

    Purpose: isolates the capacity confound.  If A_wide ≈ A in revenue → more parameters
    don't help; the 31% B/C gap is not a capacity artifact.  If A_wide >> A → Variant A
    was undertrained due to insufficient model capacity, not information.
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

    # Public-state features -> (B, 32)
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
        hk.Linear(256), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),  jax.nn.relu,
    ])(h)                                                              # (B, 64)

    alloc_probs = _alloc_head(h, B, G)
    z = _payment_head(h)
    return alloc_probs, z


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANT_NETWORK = {
    "A":       mechanism_network_A,
    "B":       mechanism_network_B,
    "C":       mechanism_network_C,
    "A_cheat": mechanism_network_A_cheat,
    "A_wide":  mechanism_network_A_wide,
}
