"""
budget_mechanism.py — BudgetMechanism wrapper with payment feasibility transform.

The raw network outputs (allocation_probs, z) where z is an unbounded real-valued
payment score. The BudgetMechanism applies:

    utility_now = sum_g(report_ig * alloc_ig)            per buyer i
    cap_i       = max(0, min(beta_i, alpha_i + utility_now_i))
    payment_i   = cap_i * sigmoid(z_i)

This hard-enforces by construction:
    payment_i >= 0
    payment_i <= beta_i                  (budget feasibility)
    payment_i <= alpha_i + v_i · x_i    (ex-post IR, under truthful reporting)
"""

import jax
import jax.numpy as jnp
import haiku as hk
from flax import struct

from budget_auction.core import BudgetState, BudgetAuction


@struct.dataclass
class BudgetMechanism:
    """Mechanism wrapper that applies the budget/IR payment feasibility transform.

    Usage:
        mechanism = BudgetMechanism(network, params)
        alloc, payment = mechanism(budget_state, report, auction)
    """
    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(
        self,
        state: BudgetState,
        report: jnp.ndarray,      # (B, G) reported types
        auction: BudgetAuction,
        *args,
    ):
        """
        Returns:
            allocation: (B, G)  — allocation probabilities per buyer per good
            payment:    (B,)    — feasibility-capped payments
        """
        alloc_probs_with_null, z = self.network.apply(
            self.params, state, report, auction, *args
        )
        # Drop null buyer row: (B+1, G) -> (B, G)
        allocation = alloc_probs_with_null[:-1]

        # Feasibility transform
        # Use reports as proxy for true types (exact under truthful play)
        utility_now = jnp.sum(allocation * report, axis=-1)   # (B,)
        cap = jnp.maximum(
            0.0,
            jnp.minimum(state.beta, state.alpha + utility_now)
        )                                                      # (B,)
        payment = cap * jax.nn.sigmoid(z)                     # (B,)

        return allocation, payment
