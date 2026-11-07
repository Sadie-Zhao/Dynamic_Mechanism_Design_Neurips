"""Agent policies using exactly the observation history in paper Section 2.1."""

import haiku as hk
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SimpleDeviationPolicy:
    """Legal report rules included explicitly in the training adversary."""
    mode: str = struct.field(pytree_node=False)

    def __call__(self, belief, auction, key):
        truth = belief.type_history[belief.time_step]
        if self.mode == "low":
            return auction.types_low
        if self.mode == "high":
            return auction.types_high
        if self.mode == "half":
            return auction.types_low + .5*(truth-auction.types_low)
        if self.mode == "truth":
            return truth
        raise ValueError(f"Unknown simple deviation: {self.mode}")


@jax.custom_jvp
def truthful_logit_shift(p, delta):
    """Stable logit shift with an exactly truthful zero-residual prototype.

    The zero branch avoids roundoff from dividing two canceling expressions.
    Its analytic JVP preserves the nonzero residual gradient at initialization;
    a plain where(delta == 0, p, ...) would incorrectly suppress that gradient.
    """
    positive, negative = jax.nn.sigmoid(delta), jax.nn.sigmoid(-delta)
    shifted = p*positive / (p*positive + (1-p)*negative)
    return jnp.where(delta == 0, p, shifted)


@truthful_logit_shift.defjvp
def _truthful_logit_shift_jvp(primals, tangents):
    p, delta = primals
    dp, ddelta = tangents
    shifted = truthful_logit_shift(p, delta)
    positive, negative = jax.nn.sigmoid(delta), jax.nn.sigmoid(-delta)
    denominator = p*positive + (1-p)*negative
    derivative_p = jnp.where(delta == 0, 1.0, positive*negative/denominator**2)
    derivative_delta = shifted*(1-shifted)
    return shifted, derivative_p*dp + derivative_delta*ddelta


def agent_history_features(belief, auction):
    """Row i contains own private history and the common public outcomes only.

    Arrays store all simulated agents for efficient rollouts; no cross-buyer
    private aggregation is allowed. Padding after the current time is masked.
    """
    B, T = auction.num_buyers, auction.num_horizons
    past = jnp.arange(T) < belief.time_step
    seen_types = jnp.arange(T + 1) <= belief.time_step
    private_types = jnp.where(seen_types[:, None, None], belief.type_history, 0.0).transpose(1, 0, 2).reshape(B, -1)
    private_reports = jnp.where(past[:, None, None], belief.report_history, 0.0).transpose(1, 0, 2).reshape(B, -1)
    public = jnp.concatenate([
        jnp.where(past[:, None, None], belief.allocation_history, 0.0).reshape(-1),
        jnp.where(past[:, None], belief.payment_history, 0.0).reshape(-1),
        auction.initial_budgets, past.astype(jnp.float32),
    ])
    return jnp.concatenate([
        private_types, private_reports,
        jnp.broadcast_to(public, (B, public.size)),
        jnp.eye(B),  # known buyer identity, not opponents' private observations
        jnp.full((B, 1), belief.time_step / max(1, T - 1)),
    ], axis=-1)


def build_full_history_policy(hidden=128, residual_init_std=0.0, eps=1e-6):
    """Trainable MLP over ordered histories, with a truthful logit prototype."""
    def network(belief, auction):
        features = agent_history_features(belief, auction)
        h = hk.nets.MLP([hidden, hidden], activate_final=True)(features)
        init = (hk.initializers.RandomNormal(residual_init_std) if residual_init_std
                else hk.initializers.Constant(0.0))
        delta = hk.Linear(auction.num_goods, w_init=init,
                          b_init=hk.initializers.Constant(0.0))(h)
        truth = belief.type_history[belief.time_step]
        width = auction.types_high - auction.types_low
        p = jnp.clip((truth - auction.types_low) / width, 0.0, 1.0)
        # Algebraic logit shift, defined at the bounds and truthful when delta=0.
        # Do not detach truth: type evolution depends on earlier deviations.
        shifted = truthful_logit_shift(p, jnp.clip(delta, -20.0, 20.0))
        return auction.types_low + width * shifted
    return hk.without_apply_rng(hk.transform(network))
