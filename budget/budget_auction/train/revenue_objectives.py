"""Explicit revenue objectives and fail-closed training-state continuation.

This development module belongs to the separate prospective revenue search.
It supplies no IC, target-infeasibility, or revenue-optimality certificate.
"""
import jax
import jax.numpy as jnp

OBJECTIVES = ('acceptance', 'direct_revenue')


def objective_value(gains, revenue, *, mode, goal, alpha, beta=0., ir=0.):
    """Positive parts are per buyer after the expected-gain estimates."""
    regret = jnp.sum(jax.nn.relu(gains))
    if mode == 'acceptance':
        # Retain the frozen acceptance trainer's subgradient at an exact tie.
        revenue_term = jnp.maximum(0., goal - revenue)
    elif mode == 'direct_revenue':
        revenue_term = -revenue
    else:
        raise ValueError('Unknown revenue objective')
    return alpha * regret + revenue_term + beta * ir


def validate_resume_state(state, expected, *, extend_from_epochs=None):
    """Allow only identical controls or an explicit completed-phase extension.

    The caller's immutable identity must pin source, environment, architecture,
    and the full prospective phase schedule. An epoch extension is not an
    automatic retry and must be authorized in that schedule before execution.
    """
    required = {'next_epoch', 'mechanism_params', 'policy_params',
                'theta_opt_state', 'w_opt_state', 'rng_state', 'sat_count',
                'logs', 'training_contract'}
    if not required.issubset(state):
        raise ValueError('Incomplete exact training state')
    saved = state['training_contract']
    if saved.get('schema_version') != 1 or saved.get('controls') != expected['controls']:
        raise ValueError('Training controls or source/run identity changed')
    old_limit, limit = saved['num_outer_epochs'], expected['num_outer_epochs']
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (old_limit, limit)):
        raise ValueError('Invalid epoch budget')
    epoch = state['next_epoch']
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 < epoch <= old_limit:
        raise ValueError('Invalid next epoch')
    if len(state['logs']['loss']) != epoch:
        raise ValueError('Training log and next epoch disagree')
    if old_limit != limit:
        if (extend_from_epochs != old_limit or epoch != old_limit or limit <= old_limit):
            raise ValueError('Epoch-budget change requires explicit completed-phase extension')
    elif extend_from_epochs is not None:
        raise ValueError('Extension declaration does not match a budget change')
    return epoch
