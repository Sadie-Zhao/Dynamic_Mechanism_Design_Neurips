"""
mechanisms/network.py — Mechanism network architecture and Mechanism wrapper.

Shared across all four bandit auction experiment variants.
Architecture:
  - 3 input streams: current_report (64) | alloc_history_sum (48) | report_history_attn (32)
  - Time step scalar broadcast to (B, 1) and prepended to report features
  - Shared MLP trunk -> (128 -> 64)
  - Allocation head: per-buyer score (B,) + shared null score -> (B+1,) softmax per good
    (equivariant: permuting buyers permutes allocation probabilities accordingly)
  - Payment head: MLP -> scalar per buyer (relu for non-negativity)

The mechanism intentionally does NOT receive state.current_type (true private types).
Only public information (allocation history, report history) and the submitted reports
are used, so that IC is a non-trivial constraint and report deviations affect outcomes.
"""

import jax
import jax.numpy as jnp
import haiku as hk

from bandit_auction.core import State, BanditAuction


# ---------------------------------------------------------------------------
# Attention pooling over history sequence
# ---------------------------------------------------------------------------

def _attention_pool(seq: jnp.ndarray, out_dim: int) -> jnp.ndarray:
    """Weighted sum over T timesteps.

    seq: (T, B, d_in)  ->  (B, out_dim)
    """
    T, B, d_in = seq.shape
    W_proj = hk.Linear(out_dim, with_bias=False)
    attn_score = hk.Linear(1, with_bias=False)

    seq_flat = seq.reshape(T * B, d_in)          # (T*B, d_in)
    proj = jax.nn.relu(W_proj(seq_flat))          # (T*B, out_dim)
    scores = attn_score(proj)                     # (T*B, 1)
    scores = scores.reshape(T, B, 1)              # (T, B, 1)
    weights = jax.nn.softmax(scores, axis=0)      # softmax over T
    proj_3d = proj.reshape(T, B, out_dim)
    pooled = jnp.sum(weights * proj_3d, axis=0)  # (B, out_dim)
    return pooled


# ---------------------------------------------------------------------------
# Main mechanism network (shared across continuous + discrete, single + multi)
# ---------------------------------------------------------------------------

def mechanism_network(
    state: State,
    report: jnp.ndarray,   # (B, G) reports submitted to mechanism
    auction: BanditAuction,
) -> tuple:
    """
    Returns (alloc_probs_with_null, payment):
      alloc_probs_with_null: (B+1, G)  — row B is the null/no-allocation option
      payment:               (B,)
    """
    B = auction.num_buyers
    G = auction.num_goods
    T = auction.num_horizons

    # Time step scalar, broadcast to (B, 1)
    # jnp.full handles traced values; jnp.array([[...]]) would call float() on the tracer
    t_norm = jnp.full((B, 1), state.time_step / max(T - 1, 1), dtype=jnp.float32)

    # Stream 1: current report + time step (B, G+1) -> (B, 64)
    report_in = jnp.concatenate(
        [report.reshape(B, -1).astype(jnp.float32), t_norm], axis=-1
    )
    report_feat = hk.nets.MLP([128, 64], activate_final=True)(report_in)

    # Stream 2: cumulative allocation history sum (B, G) -> (B, 48)
    alloc_sum = jnp.sum(state.allocation_history, axis=0).astype(jnp.float32)  # (B, G)
    alloc_feat = hk.nets.MLP([96, 48], activate_final=True)(
        alloc_sum.reshape(B, -1)
    )

    # Stream 3: report history via attention pooling (T, B, G) -> (B, 32)
    rep_hist = state.report_history.astype(jnp.float32)  # (T, B, G)
    rep_hist_feat = _attention_pool(rep_hist, out_dim=32)  # (B, 32)

    # Concatenate all streams -> (B, 144)
    h = jnp.concatenate([report_feat, alloc_feat, rep_hist_feat], axis=-1)

    # Shared trunk -> (B, 64)
    h = hk.nets.MLP([128, 64], activate_final=True)(h)

    # ---- Allocation head (equivariant) ----
    # Per-buyer score for each good: (B, G)
    buyer_scores = hk.Linear(G)(h)   # (B, G)
    # Shared null score per good: (1, G)
    null_score = hk.get_parameter(
        "null_score", shape=(1, G), init=hk.initializers.Constant(0.0)
    )
    # Stack -> (B+1, G), then softmax over buyer dimension
    alloc_logits = jnp.concatenate([buyer_scores, null_score], axis=0)  # (B+1, G)
    alloc_probs_with_null = jax.nn.softmax(alloc_logits, axis=0)  # (B+1, G)

    # ---- Payment head ----
    payment = hk.nets.MLP([64, 32, 1], activate_final=False)(h).squeeze(-1)  # (B,)
    payment = jax.nn.relu(payment)  # non-negative payments (no budget cap in bandit)

    return alloc_probs_with_null, payment


# ---------------------------------------------------------------------------
# Mechanism wrapper (callable dataclass)
# ---------------------------------------------------------------------------

from flax import struct


@struct.dataclass
class Mechanism:
    """Wraps a haiku-transformed mechanism network with its params."""
    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(
        self,
        state: State,
        report: jnp.ndarray,
        auction: BanditAuction,
    ):
        """Returns (allocation, payment) with allocation (B, G), payment (B,)."""
        alloc_probs_with_null, payment = self.network.apply(
            self.params, state, report, auction
        )
        allocation = alloc_probs_with_null[:-1]  # drop null row -> (B, G)
        return allocation, payment


def build_mechanism_network() -> hk.Transformed:
    """Returns the haiku-transformed mechanism network."""
    return hk.without_apply_rng(hk.transform(mechanism_network))
