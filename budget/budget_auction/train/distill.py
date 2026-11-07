"""Imitate a fixed teacher on truthful trajectories, then audit the student.

The objective and public function names retain the existing distillation code.
Sampling and fitting are compiled, observations exclude hidden types, and an
independent validation set selects fitted parameters. Matching sampled teacher
outputs does not prove representability on every history or incentive compatibility.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from budget_auction.mechanisms.budget_mechanism import mechanism_observation


def data_digest(data):
    """Content identity for sampled inputs and targets, including shapes/dtypes."""
    digest = hashlib.sha256()
    leaves, tree = jax.tree_util.tree_flatten(data)
    digest.update(str(tree).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def collect_distillation_data(auction, mechanism_B, num_trajectories, key, verbose=True):
    """Collect teacher outputs with the legacy trajectory-key convention.

    Return (legal observations, truthful reports, allocations including null,
    payment fractions), flattened in trajectory-major, then time-major order.
    Current true types are used only to generate truthful reports in the simulator.
    """
    if num_trajectories < 1:
        raise ValueError("At least one trajectory is required")

    def trajectory(params, trajectory_key):
        trajectory_key, init_key = jax.random.split(trajectory_key)
        initial = auction.get_initial_state(init_key)

        def step(carry, unused):
            state, sample_key = carry
            report = state.current_type
            observed = mechanism_observation(state)
            allocation, z = mechanism_B.network.apply(params, observed, report, auction)
            fraction = jax.nn.sigmoid(z)
            payment = _payments(observed, report, allocation, fraction)
            sample_key, transition_key = jax.random.split(sample_key)
            next_state = auction.get_next_budget_state(
                state, report, allocation[:-1], payment, transition_key)
            return (next_state, sample_key), (observed, report, allocation, fraction)

        _, rows = jax.lax.scan(step, (initial, trajectory_key), None, length=auction.num_horizons)
        return rows

    data = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        mechanism_B.params, jax.random.split(key, num_trajectories))
    data = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), data)
    if verbose:
        print(f"Distillation dataset: {num_trajectories} trajectories, {data[1].shape[0]} rows", flush=True)
    return data


def _payments(observation, report, allocation, fraction):
    cap = jnp.maximum(0., jnp.minimum(observation.beta,
        observation.alpha+jnp.sum(report*allocation[:-1], axis=-1)))
    return cap*fraction


def _row_errors(network, params, auction, data):
    observation, report, target_allocation, target_fraction = data

    def one(state, reports, allocation_B, fraction_B):
        # Apply the boundary even if a caller supplies full simulator states.
        observed = mechanism_observation(state)
        allocation_A, z_A = network.apply(params, observed, reports, auction)
        fraction_A = jax.nn.sigmoid(z_A)
        payment_A = _payments(observed, reports, allocation_A, fraction_A)
        payment_B = _payments(observed, reports, allocation_B, fraction_B)
        return jnp.asarray([jnp.mean((allocation_A-allocation_B)**2),
                            jnp.mean((fraction_A-fraction_B)**2),
                            jnp.mean((payment_A-payment_B)**2)])

    return jax.vmap(one)(observation, report, target_allocation, target_fraction)


def evaluate_distillation(auction, mechanism_net_A, params, data):
    """MSE on held-out teacher states; not an evaluation of student rollouts."""
    errors = jax.jit(lambda p: jnp.mean(_row_errors(mechanism_net_A, p, auction, data), axis=0))(params)
    values = [float(x) for x in errors]
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("Nonfinite distillation validation metrics")
    return dict(zip(("allocation_mse", "payment_fraction_mse", "actual_payment_mse"), values))


def distill_mechanism_A(auction, mechanism_net_A, init_params_A, states_batch,
        reports_batch, alloc_B_batch, sigmoid_z_B_batch, num_epochs=500, lr=1e-3,
        batch_size=256, alloc_weight=1., payment_weight=1., print_every=50,
        key=None, validation_data=None, eval_every=10, resume_state=None,
        checkpoint_callback=None):
    """Fit the original allocation/fraction MSE with epoch-boundary resumption.

    Independent validation chooses the best fitting parameters. With no validation
    data, return the terminal parameters for compatibility, without a held-out claim.
    The callback retains terminal Adam/RNG states and the selected parameters.
    """
    data = (states_batch, reports_batch, alloc_B_batch, sigmoid_z_B_batch)
    rows = alloc_B_batch.shape[0]
    if rows < 1 or batch_size < 1 or num_epochs < 0 or eval_every < 1 or print_every < 1:
        raise ValueError("Invalid distillation data size or schedule")
    if lr <= 0 or alloc_weight < 0 or payment_weight < 0 or alloc_weight+payment_weight <= 0:
        raise ValueError("Require positive learning rate and a nonzero nonnegative objective")
    controls = dict(num_epochs=num_epochs, lr=lr, batch_size=batch_size,
        alloc_weight=alloc_weight, payment_weight=payment_weight, eval_every=eval_every,
        data_sha256=data_digest(data),
        validation_sha256=None if validation_data is None else data_digest(validation_data))
    optimizer = optax.adam(lr)
    params, opt_state = init_params_A, optimizer.init(init_params_A)
    key = jax.random.PRNGKey(0) if key is None else key
    first_epoch, loss_history, validation_history = 0, [], []
    best_loss, best_params, best_epoch = None, params, 0
    if resume_state is not None:
        if resume_state.get("schema") != 1 or resume_state["controls"] != controls:
            raise ValueError("Distillation checkpoint controls or dataset differ")
        params, opt_state, key = resume_state["params"], resume_state["opt_state"], resume_state["rng_state"]
        first_epoch = resume_state["next_epoch"]
        loss_history, validation_history = list(resume_state["loss_history"]), list(resume_state["validation_history"])
        best_loss, best_params, best_epoch = resume_state["best_loss"], resume_state["best_params"], resume_state["best_epoch"]

    batches = (rows+batch_size-1)//batch_size
    padded_rows = batches*batch_size

    @jax.jit
    def train_epoch(p, opt, permutation):
        padded = jnp.pad(permutation, (0, padded_rows-rows)).reshape((batches, batch_size))
        valid = (jnp.arange(padded_rows) < rows).reshape((batches, batch_size))

        def step(carry, batch):
            p, opt = carry
            indices, mask = batch
            sample = jax.tree_util.tree_map(lambda x: x[indices], data)
            count = jnp.sum(mask)
            def objective(q):
                errors = _row_errors(mechanism_net_A, q, auction, sample)
                per_row = alloc_weight*errors[:, 0]+payment_weight*errors[:, 1]
                return jnp.sum(per_row*mask)/count
            value, gradient = jax.value_and_grad(objective)(p)
            updates, opt = optimizer.update(gradient, opt, p)
            return (optax.apply_updates(p, updates), opt), value*count

        (p, opt), weighted_losses = jax.lax.scan(step, (p, opt), (padded, valid))
        leaves = jax.tree_util.tree_leaves((p, opt))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        return p, opt, jnp.sum(weighted_losses)/rows, finite

    validate = None if validation_data is None else jax.jit(
        lambda p: jnp.mean(_row_errors(mechanism_net_A, p, auction, validation_data), axis=0))

    def consider(epoch):
        nonlocal best_loss, best_params, best_epoch
        if validate is None:
            best_params, best_epoch = params, epoch
            return
        metrics = [float(x) for x in validate(params)]
        if not all(math.isfinite(x) for x in metrics):
            raise FloatingPointError("Nonfinite distillation validation")
        value = alloc_weight*metrics[0]+payment_weight*metrics[1]
        validation_history.append({"epoch": epoch, "allocation_mse": metrics[0],
            "payment_fraction_mse": metrics[1], "actual_payment_mse": metrics[2], "selection_loss": value})
        if best_loss is None or value < best_loss:
            best_loss, best_params, best_epoch = value, params, epoch

    def save(epoch):
        if checkpoint_callback is not None:
            checkpoint_callback({"schema": 1, "controls": controls, "next_epoch": epoch,
                "params": params, "opt_state": opt_state, "rng_state": key,
                "loss_history": list(loss_history), "validation_history": list(validation_history),
                "best_loss": best_loss, "best_params": best_params, "best_epoch": best_epoch})

    if resume_state is None:
        consider(0)
        save(0)
    for epoch in range(first_epoch, num_epochs):
        key, permutation_key = jax.random.split(key)
        params, opt_state, value, finite = train_epoch(params, opt_state, jax.random.permutation(permutation_key, rows))
        if not bool(finite) or not math.isfinite(float(value)):
            raise FloatingPointError(f"Nonfinite distillation update at epoch {epoch+1}")
        loss_history.append(float(value))
        if validate is None or (epoch+1) % eval_every == 0 or epoch+1 == num_epochs:
            consider(epoch+1)
        save(epoch+1)
        if (epoch+1) % print_every == 0 or epoch == 0:
            print(f"distill epoch {epoch+1}/{num_epochs}: train MSE {float(value):.6f}", flush=True)
    return best_params, loss_history


def run_distillation(auction, mechanism_net_A, mechanism_B, seed=42,
        num_trajectories=2048, num_epochs=500, lr=1e-3, batch_size=256,
        alloc_weight=1., payment_weight=1., print_every=50, save_dir=None,
        num_validation_trajectories=512, eval_every=10):
    """Sample disjoint teacher trajectories and fit; IC fine-tuning is separate.

    A production caller must also record source, teacher and environment identities.
    The saved fitting checkpoint validates the exact sampled dataset and schedule.
    """
    from budget_auction.train.pilot import atomic_pickle, atomic_json
    import pickle

    key = jax.random.PRNGKey(seed)
    key, data_key = jax.random.split(key)
    data = collect_distillation_data(auction, mechanism_B, num_trajectories, data_key)
    validation_key = jax.random.fold_in(data_key, 812381)
    validation = collect_distillation_data(auction, mechanism_B, num_validation_trajectories, validation_key)
    key, init_key, parameter_key = jax.random.split(key, 3)
    state = auction.get_initial_state(init_key)
    initial = mechanism_net_A.init(parameter_key, mechanism_observation(state),
        jnp.zeros((auction.num_buyers, auction.num_goods)), auction)
    key, fit_key = jax.random.split(key)
    output = None if save_dir is None else Path(save_dir)
    resume = None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        if (output/"fit_resume.pkl").exists():
            with (output/"fit_resume.pkl").open("rb") as stream:
                resume = pickle.load(stream)
    fitted, history = distill_mechanism_A(auction, mechanism_net_A, initial, *data,
        num_epochs=num_epochs, lr=lr, batch_size=batch_size, alloc_weight=alloc_weight,
        payment_weight=payment_weight, print_every=print_every, key=fit_key,
        validation_data=validation, eval_every=eval_every, resume_state=resume,
        checkpoint_callback=None if output is None else lambda progress: atomic_pickle(output/"fit_resume.pkl", progress))
    if output is not None:
        atomic_pickle(output/"distilled_A_params.pkl", fitted)
        atomic_json(output/"fit_validation.json", {
            "train_data_sha256": data_digest(data), "validation_data_sha256": data_digest(validation),
            "validation_metrics": evaluate_distillation(auction, mechanism_net_A, fitted, validation),
            "role": "validation_selection_on_teacher_truthful_states_not_independent_final_evaluation",
            "source_and_teacher_identity_required_from_caller": True})
    return fitted, history
