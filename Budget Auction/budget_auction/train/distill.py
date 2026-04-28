"""
distill.py — Distillation of a trained Variant B mechanism into Variant A architecture.

Protocol:
  1. Roll out trained B with truthful play for N trajectories, collecting
     (state, report) -> (alloc_probs, sigmoid_z) pairs at every timestep.
  2. Train a fresh Variant A network via supervised MSE to match B's outputs.
  3. Return the distilled A params; caller can use them as a warm start for
     IC-constrained fine-tuning via nested_GD_budget.

Experiment interpretation:
  - If A_distilled alone achieves near-B revenue (evaluated as a mechanism):
      A's architecture CAN represent B's optimal function. The cold-start IC
      training simply fails to find it. This is strong direct evidence for the
      representational optimization gap.
  - If A_distilled underperforms B but warm-started fine-tuning closes the gap:
      Same conclusion — A can represent the solution, but needs a better starting
      point than random init under IC constraints.
  - If neither works: A's architecture genuinely cannot represent B's function
      (capacity or information gap remains after controlling for optimization).

Usage:
    from budget_auction.train.distill import run_distillation
    distilled_params, loss_history = run_distillation(
        auction, mechanism_net_A, mechanism_B,
        num_trajectories=2048, num_epochs=500,
        save_dir='outputs/exp1/distill',
    )
    # Then warm-start fine-tuning:
    logs = nested_GD_budget(auction, mechanism_net_A, policy_network,
                            profit_goal=..., init_mechanism_params=distilled_params, ...)
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

from budget_auction.core import BudgetAuction, BudgetState
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism


# ---------------------------------------------------------------------------
# Step 1: Collect distillation dataset from trained B
# ---------------------------------------------------------------------------

def collect_distillation_data(
    auction: BudgetAuction,
    mechanism_B: BudgetMechanism,
    num_trajectories: int,
    key: jax.random.PRNGKey,
    verbose: bool = True,
) -> Tuple[BudgetState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Roll out mechanism_B with truthful play and record (state, report, alloc_B,
    sigmoid_z_B) at every step of every trajectory.

    We match sigmoid(z_B) rather than raw z_B because z is unbounded and
    MSE on sigmoid values is numerically stable and directly interpretable
    as the payment fraction relative to the feasibility cap.

    Returns:
        states_batch      : BudgetState pytree, leading dim = (num_trajectories * T,)
        reports_batch     : (N, B, G)   truthful reports at each step
        alloc_B_batch     : (N, B+1, G) B's allocation probs incl. null row
        sigmoid_z_B_batch : (N, B)      B's payment fraction (sigmoid of raw score)
    """
    all_states: list[BudgetState] = []
    all_reports = []
    all_alloc_B = []
    all_sigmoid_z_B = []

    traj_keys = jax.random.split(key, num_trajectories)

    for i in range(num_trajectories):
        k = traj_keys[i]
        k, k_init = jax.random.split(k)
        state = auction.get_initial_state(k_init)

        for _t in range(auction.num_horizons):
            report = state.current_type                                 # truthful (B, G)

            alloc_probs_null, z = mechanism_B.network.apply(
                mechanism_B.params, state, report, auction
            )
            all_states.append(state)
            all_reports.append(report)
            all_alloc_B.append(alloc_probs_null)
            all_sigmoid_z_B.append(jax.nn.sigmoid(z))

            # Advance state using B's mechanism output
            allocation = alloc_probs_null[:-1]                         # (B, G)
            utility_now = jnp.sum(allocation * report, axis=-1)        # (B,)
            cap = jnp.maximum(
                0.0, jnp.minimum(state.beta, state.alpha + utility_now)
            )
            payment = cap * jax.nn.sigmoid(z)
            k, k_step = jax.random.split(k)
            state = auction.get_next_budget_state(
                state, report, allocation, payment, k_step
            )

    # Stack all collected data along leading dim
    states_batch      = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *all_states
    )
    reports_batch     = jnp.stack(all_reports,     axis=0)
    alloc_B_batch     = jnp.stack(all_alloc_B,     axis=0)
    sigmoid_z_B_batch = jnp.stack(all_sigmoid_z_B, axis=0)

    N = alloc_B_batch.shape[0]
    if verbose:
        print(
            f"Distillation dataset: {N} samples "
            f"({num_trajectories} trajectories × T={auction.num_horizons})"
        )
    return states_batch, reports_batch, alloc_B_batch, sigmoid_z_B_batch


# ---------------------------------------------------------------------------
# Step 2: Train A to match B's outputs via MSE
# ---------------------------------------------------------------------------

def distill_mechanism_A(
    auction: BudgetAuction,
    mechanism_net_A: hk.Transformed,
    init_params_A,
    states_batch: BudgetState,
    reports_batch: jnp.ndarray,
    alloc_B_batch: jnp.ndarray,
    sigmoid_z_B_batch: jnp.ndarray,
    num_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    alloc_weight: float = 1.0,
    payment_weight: float = 1.0,
    print_every: int = 50,
    key: Optional[jax.random.PRNGKey] = None,
) -> Tuple[dict, list]:
    """
    Supervised distillation: train A to minimize MSE between its raw outputs and B's.

    Loss = alloc_weight  * mean((alloc_A   - alloc_B)^2)
         + payment_weight * mean((sigmoid_z_A - sigmoid_z_B)^2)

    Both terms are on normalised quantities in [0,1], so alloc_weight = payment_weight = 1
    is a reasonable default.

    Returns:
        distilled_params : A mechanism params after distillation
        loss_history     : list of mean per-epoch MSE loss
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    N = alloc_B_batch.shape[0]

    # Batched forward: vmap A over the dataset batch dimension.
    # auction is captured in the closure (not vmapped — it is constant across all samples).
    def _batched_forward(params, sb: BudgetState, rb: jnp.ndarray):
        def _single(state, report):
            alloc, z = mechanism_net_A.apply(params, state, report, auction)
            return alloc, jax.nn.sigmoid(z)
        return jax.vmap(_single)(sb, rb)

    @jax.jit
    def _loss_and_grad(params, sb, rb, alloc_B_b, sig_z_B_b):
        def _loss(p):
            alloc_A, sig_z_A = _batched_forward(p, sb, rb)
            l_alloc   = jnp.mean((alloc_A  - alloc_B_b)  ** 2)
            l_payment = jnp.mean((sig_z_A  - sig_z_B_b)  ** 2)
            return alloc_weight * l_alloc + payment_weight * l_payment
        return jax.value_and_grad(_loss)(params)

    optimizer = optax.adam(lr)
    params    = init_params_A
    opt_state = optimizer.init(params)
    loss_history = []

    for epoch in range(num_epochs):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, N)

        batch_losses = []
        for start in range(0, N, batch_size):
            idx  = perm[start : start + batch_size]
            sb   = jax.tree_util.tree_map(lambda x: x[idx], states_batch)
            rb   = reports_batch[idx]
            ab   = alloc_B_batch[idx]
            szb  = sigmoid_z_B_batch[idx]

            loss, grad = _loss_and_grad(params, sb, rb, ab, szb)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            batch_losses.append(float(loss))

        epoch_loss = float(np.mean(batch_losses))
        loss_history.append(epoch_loss)
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  distill epoch {epoch+1:4d}/{num_epochs}  loss={epoch_loss:.6f}")

    return params, loss_history


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def run_distillation(
    auction: BudgetAuction,
    mechanism_net_A: hk.Transformed,
    mechanism_B: BudgetMechanism,
    seed: int = 42,
    num_trajectories: int = 2048,
    num_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    alloc_weight: float = 1.0,
    payment_weight: float = 1.0,
    print_every: int = 50,
    save_dir: Optional[str] = None,
) -> Tuple[dict, list]:
    """
    Full pipeline: collect B's outputs → train A to imitate → optionally save params.

    The returned params are ready to use as `init_mechanism_params` in nested_GD_budget
    for warm-start fine-tuning under IC constraints.

    Returns:
        distilled_params : A mechanism params after distillation
        loss_history     : per-epoch MSE loss
    """
    key = jax.random.PRNGKey(seed)

    # --- Collect dataset ---
    key, data_key = jax.random.split(key)
    states_batch, reports_batch, alloc_B_batch, sigmoid_z_B_batch = \
        collect_distillation_data(auction, mechanism_B, num_trajectories, data_key)

    # --- Init fresh A params on a dummy state ---
    key, init_key, param_key = jax.random.split(key, 3)
    s0 = auction.get_initial_state(init_key)
    init_params_A = mechanism_net_A.init(
        param_key,
        s0,
        jnp.zeros((auction.num_buyers, auction.num_goods)),
        auction,
    )

    # --- Distill ---
    key, distill_key = jax.random.split(key)
    distilled_params, loss_history = distill_mechanism_A(
        auction=auction,
        mechanism_net_A=mechanism_net_A,
        init_params_A=init_params_A,
        states_batch=states_batch,
        reports_batch=reports_batch,
        alloc_B_batch=alloc_B_batch,
        sigmoid_z_B_batch=sigmoid_z_B_batch,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        alloc_weight=alloc_weight,
        payment_weight=payment_weight,
        print_every=print_every,
        key=distill_key,
    )

    # --- Save ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "distilled_A_params.pkl")
        with open(path, "wb") as f:
            pickle.dump(distilled_params, f)
        print(f"Distilled A params saved to: {path}")

    return distilled_params, loss_history
