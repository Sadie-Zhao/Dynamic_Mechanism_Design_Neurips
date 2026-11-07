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
class MechanismObservation:
    """The mechanism's information: reports and public outcomes, never true types."""
    time_step: jnp.ndarray
    allocation_history: jnp.ndarray
    report_history: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray
    payment_history: jnp.ndarray


def mechanism_observation(state):
    """Enforce the information boundary and exclude future/padded history."""
    past = jnp.arange(state.allocation_history.shape[0]) < state.time_step
    return MechanismObservation(
        time_step=state.time_step,
        allocation_history=jnp.where(past[:, None, None], state.allocation_history, 0.0),
        report_history=jnp.where(past[:, None, None], state.report_history, 0.0),
        alpha=state.alpha, beta=state.beta,
        payment_history=jnp.where(past[:, None], state.payment_history, 0.0),
    )


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
        observed = mechanism_observation(state)
        alloc_probs_with_null, z = self.network.apply(
            self.params, observed, report, auction, *args
        )
        # Drop null buyer row: (B+1, G) -> (B, G)
        allocation = alloc_probs_with_null[:-1]

        # Feasibility transform
        # Use reports as proxy for true types (exact under truthful play)
        utility_now = jnp.sum(allocation * report, axis=-1)   # (B,)
        cap = jnp.maximum(
            0.0,
            jnp.minimum(observed.beta, observed.alpha + utility_now)
        )                                                      # (B,)
        payment = cap * jax.nn.sigmoid(z)                     # (B,)

        return allocation, payment
