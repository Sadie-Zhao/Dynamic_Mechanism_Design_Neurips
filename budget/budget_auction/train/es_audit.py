"""Antithetic parameter search using legal-policy forward utilities only.

This complementary audit does not differentiate through mechanisms, payments,
or transitions. It is a finite heuristic, not a global best-response oracle.
"""
import hashlib
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax

from budget_auction.core import BoundedContPolicy, BudgetSimulator


def parameter_fingerprint(params):
    leaves, structure = jax.tree_util.tree_flatten(params)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def antithetic_gradient(noise, plus, minus, sigma):
    """Unbiased Gaussian-smoothing estimator when noise rows are IID normal."""
    return jnp.mean((plus - minus)[:, None] * noise, axis=0) / (2 * sigma)


def search_es(auction, mechanism, network, initial_params, buyer, *, identity,
              seed, steps=200, pairs=16, episodes=64, validation_samples=2048,
              sigma=.01, learning_rate=.001, gradient_clip=.5, eval_every=25,
              resume_state=None, checkpoint_callback=None):
    if not isinstance(identity, str) or not identity:
        raise ValueError('ES requires a pinned source/mechanism/search identity')
    if not 0 <= buyer < auction.num_buyers:
        raise ValueError('Invalid unilateral buyer')
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
           for x in [steps, pairs, episodes, validation_samples, eval_every]):
        raise ValueError('ES budgets must be positive integers')
    if any(not math.isfinite(x) or x <= 0 for x in [sigma, learning_rate, gradient_clip]):
        raise ValueError('ES scale, rate and clip must be finite and positive')
    flat, unpack = ravel_pytree(initial_params)
    contract = {'identity': identity, 'initial_params_sha256': parameter_fingerprint(initial_params),
                'buyer': buyer, 'seed': seed, 'steps': steps, 'pairs': pairs,
                'episodes': episodes, 'validation_samples': validation_samples,
                'sigma': sigma, 'learning_rate': learning_rate,
                'gradient_clip': gradient_clip, 'eval_every': eval_every}
    optimizer = optax.chain(optax.clip_by_global_norm(gradient_clip), optax.adam(learning_rate))
    @jax.jit
    def fitness(theta, key):
        policy = BoundedContPolicy(network, unpack(theta))
        return BudgetSimulator.estimate_cumulative_deviation_gains(
            auction, mechanism, policy, episodes, key)[buyer]
    @jax.jit
    def validation(theta, key):
        policy = BoundedContPolicy(network, unpack(theta))
        return BudgetSimulator.estimate_cumulative_deviation_gains(
            auction, mechanism, policy, validation_samples, key)[buyer]
    @jax.jit
    def step(theta, opt_state, key):
        next_key, noise_key, sample_key = jax.random.split(key, 3)
        noise = jax.random.normal(noise_key, (pairs, theta.size))
        samples = jax.random.split(sample_key, pairs)
        plus = jax.vmap(fitness)(theta[None, :] + sigma * noise, samples)
        minus = jax.vmap(fitness)(theta[None, :] - sigma * noise, samples)
        gradient = antithetic_gradient(noise, plus, minus, sigma)
        updates, opt_state = optimizer.update(-gradient, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        finite = (jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(gradient))
                  & jnp.all(jnp.isfinite(plus)) & jnp.all(jnp.isfinite(minus)))
        return theta, opt_state, next_key, finite
    train_key, validation_key = jax.random.split(jax.random.PRNGKey(seed))
    if resume_state is None:
        opt_state = optimizer.init(flat)
        score = float(validation(flat, validation_key))
        best, best_score, first_step = flat, score, 0
        curve = [{'step': 0, 'selection_gain': score}]
    else:
        required = {'contract', 'next_step', 'flat', 'opt_state', 'train_key',
                    'best_flat', 'best_score', 'curve'}
        if not required.issubset(resume_state) or resume_state['contract'] != contract:
            raise ValueError('ES controls, source/mechanism identity or warm initialization changed')
        first_step = resume_state['next_step']
        if (not isinstance(first_step, int) or isinstance(first_step, bool)
                or not 0 < first_step <= steps):
            raise ValueError('ES cannot restart from zero or exceed its budget')
        flat, opt_state, train_key = resume_state['flat'], resume_state['opt_state'], resume_state['train_key']
        best, best_score, curve = resume_state['best_flat'], resume_state['best_score'], resume_state['curve']
    if not math.isfinite(best_score):
        raise ValueError('Nonfinite ES initialization/selection score')
    for index in range(first_step, steps):
        flat, opt_state, train_key, finite = step(flat, opt_state, train_key)
        if not bool(finite):
            raise FloatingPointError('ES generated a nonfinite state; preserve and review, do not retry')
        if (index + 1) % eval_every == 0 or index + 1 == steps:
            score = float(validation(flat, validation_key))
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite ES validation gain')
            curve.append({'step': index + 1, 'selection_gain': score})
            if score > best_score:
                best, best_score = flat, score
            if checkpoint_callback is not None:
                checkpoint_callback({'contract': contract, 'next_step': index + 1,
                                     'flat': flat, 'opt_state': opt_state, 'train_key': train_key,
                                     'best_flat': best, 'best_score': best_score, 'curve': curve})
    summary = {'method': 'antithetic_parameter_es', 'buyer': buyer,
               'selection_expected_gain': best_score, 'selection_curve': curve,
               'search_config': contract,
               'fitness_evaluations': steps * pairs * 2,
               'fitness_trajectories': steps * pairs * 2 * episodes,
               'validation_evaluations': len(curve),
               'fresh_final_samples_drawn': False,
               'scope': 'Finite deterministic legal-policy search;no mechanism derivatives or global-IC certificate'}
    return summary, unpack(best)
