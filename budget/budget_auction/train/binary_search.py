"""
binary_search.py — Binary search over profit_goal for budget auction mechanisms.

Protocol:
  Given a fixed hyperparameter config (lr_theta, lr_w, alpha_base, beta_weight,
  num_inner_steps), find the highest profit_goal p* such that a mechanism trained
  with that goal achieves IC regret ≤ ic_thresh AND IR loss ≤ ir_thresh on
  held-out evaluation keys.

  alpha is scaled proportionally: alpha(p) = alpha_base * (p / pg_base)
  so that IC enforcement remains competitive with the profit-gap term at all levels.

Key idea: record ALL mechanism checkpoints during training (not just the final one).
  After training, evaluate every checkpoint on held-out keys.  A profit_goal is
  "feasible" if at least one checkpoint passes both IC and IR thresholds.
  This avoids discarding good intermediate solutions that may be over-trained away.

Usage (programmatic):
    from budget_auction.train.binary_search import run_binary_search
    result = run_binary_search(auction, mechanism_network, policy_network,
                               variant="A", save_dir="outputs/bs_A")
"""

from __future__ import annotations

import json
import os
import pickle
import time
import functools
import math
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from budget_auction.core import (
    BudgetSimulator,
    BoundedContPolicy,
    SEED,
)
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.train.run_acceptance import nested_GD_budget
from budget_auction.train.evaluate import eval_budget_mechanism_multikey


# ---------------------------------------------------------------------------
# Default binary-search hyperparameters (seeded from HP search results)
# ---------------------------------------------------------------------------

DEFAULT_BS_CFG = dict(
    # Learning rates (anchored from HP search: Stage 1 winner)
    lr_theta        = 3e-4,
    lr_w            = 1e-3,

    # IC/IR loss weights at base profit_goal
    alpha_base      = 1e5,   # alpha(p) = alpha_base * (p / pg_base)
    beta_weight     = 500.0,

    # Inner oracle steps
    num_inner_steps = 600,

    # Training epochs per binary-search trial
    num_outer_epochs = 2000,

    # Batch size
    num_samples_train = 128,

    # Checkpoint recording: save when train reg AND train ir both ≤ this
    checkpoint_thresh    = 0.15,
    checkpoint_interval  = 50,

    # Binary search range and tolerance
    pg_lo   = 0.8,
    pg_hi   = 2.0,
    pg_base = 0.8,   # reference profit_goal for alpha_base
    pg_tol  = 0.1,

    # Feasibility thresholds (eval loss)
    ic_thresh  = 0.05,
    # Revenue gap threshold: a checkpoint is feasible only if
    #   revenue_eval >= profit_goal - gap_thresh
    # Prevents a low-revenue IC solution from falsely pushing pg_lo upward.
    # Natural default: same as pg_tol so feasibility means "within search resolution".
    gap_thresh = 0.1,
    # Checkpoints with gap_train > skip_gap_mult * gap_thresh are skipped before
    # the expensive IC re-opt + eval pass. Raise if good checkpoints are being
    # skipped; lower to save more compute.
    skip_gap_mult = 2.0,

    # IC re-optimisation at eval time (separate from the training inner loop).
    # For each checkpoint, the oracle policy is re-solved from scratch against
    # the fixed mechanism params before IC regret is measured.
    # This searches for stronger deviations; it is not an upper bound on regret.
    ic_reopt_steps     = 1000,   # inner gradient steps per restart
    ic_reopt_restarts  = 3,      # independent random restarts (take the max)
    ic_reopt_lr        = 3e-3,   # Adam lr for the reopt inner loop
    ic_reopt_samples   = 128,    # MC samples used during reopt ascent

    # Evaluation keys / samples (for revenue, IR, budget-viol, and final IC estimate)
    eval_num_keys    = 32,
    eval_num_samples = 256,

    # Print frequency inside nested_GD_budget
    print_iter = 100,
)


# ---------------------------------------------------------------------------
# Step 1 — Train with checkpoint recording
# ---------------------------------------------------------------------------

def train_with_checkpoints(
    auction,
    mechanism_network,
    policy_network,
    profit_goal: float,
    cfg: Dict,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Train a mechanism for `profit_goal` with checkpoint recording.

    alpha is scaled from cfg['alpha_base'] proportionally to profit_goal.

    Returns:
        checkpoints : list of dicts (from logs["checkpoint_mech"])
        logs        : full training logs dict (mechanism_params / policy_params included)
    """
    alpha = cfg["alpha_base"] * (profit_goal / cfg["pg_base"])
    if verbose:
        print(f"  [train] profit_goal={profit_goal:.3f}  alpha={alpha:.1e}  "
              f"beta_weight={cfg['beta_weight']:.1f}  "
              f"lr_theta={cfg['lr_theta']:.1e}  lr_w={cfg['lr_w']:.1e}")

    logs = nested_GD_budget(
        auction            = auction,
        mechanism_network  = mechanism_network,
        policy_network     = policy_network,
        profit_goal        = profit_goal,
        num_outer_epochs   = cfg["num_outer_epochs"],
        lr_theta           = cfg["lr_theta"],
        num_inner_steps    = cfg["num_inner_steps"],
        lr_w               = cfg["lr_w"],
        num_samples_train  = cfg["num_samples_train"],
        alpha_weight       = alpha,
        beta_weight        = cfg["beta_weight"],
        print_iter         = cfg.get("print_iter", 100),
        checkpoint_thresh  = cfg["checkpoint_thresh"],
        checkpoint_interval= cfg["checkpoint_interval"],
        seed               = seed,
    )

    checkpoints = logs["checkpoint_mech"]

    # Always include the final mechanism as a candidate (even if it didn't
    # pass the training threshold — the eval pass may still accept it)
    final_reg = logs["final_train_metrics"]["reg_plus"]
    final_ir  = logs["final_train_metrics"]["ir"]
    final_gap = logs["final_train_metrics"]["gap"]
    checkpoints_with_final = checkpoints + [{
        "epoch": cfg["num_outer_epochs"] - 1,
        "reg_train": final_reg,
        "ir_train": final_ir,
        "gap_train": final_gap,
        "mechanism_params": logs["mechanism_params"],
        "policy_params": logs["policy_params"],
    }]

    if verbose:
        print(f"  [train] done — {len(checkpoints)} threshold-passing checkpoints "
              f"(+ final); total candidates: {len(checkpoints_with_final)}")

    return checkpoints_with_final, logs


# ---------------------------------------------------------------------------
# IC re-optimisation — empirical lower bound on best-response regret
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["policy_network", "steps", "buyer", "num_samples"])
def _reopt_chunk(auction, mechanism, policy_network, params, opt_state, key,
                 lr, steps, buyer, num_samples):
    """Compile a block of ascent steps; reuse it across mechanism checkpoints."""
    optimizer = optax.adam(lr)
    def gain(p, k):
        return BudgetSimulator.estimate_cumulative_deviation_gains(
            auction, mechanism, BoundedContPolicy(policy_network, p), num_samples, k)[buyer]
    def step(_, carry):
        p, state, k = carry
        k, sample_key = jax.random.split(k)
        grad = jax.grad(gain)(p, sample_key)
        updates, state = optimizer.update(jax.tree_util.tree_map(lambda x: -x, grad), state, p)
        return optax.apply_updates(p, updates), state, k
    return jax.lax.fori_loop(0, steps, step, (params, opt_state, key))


@jax.jit
def _finite_reopt_state(params, optimizer_state):
    leaves = jax.tree_util.tree_leaves((params, optimizer_state))
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def reopt_cumulative_ic(
    auction,
    mechanism_network,
    mechanism_params,
    policy_network,
    num_reopt_steps:    int   = 1000,
    num_reopt_restarts: int   = 3,
    lr_reopt:           float = 3e-3,
    num_samples:        int   = 128,
    seed:               int   = SEED,
    resume_state=None,
    checkpoint_callback=None,
) -> Tuple[float, object]:
    """Search independently per buyer and select on a separate validation stream.

    Return validation IC sum and a tuple of the actual selected policy parameters.
    The caller must evaluate them on fresh test samples. Optimization uncertainty
    is not bounded by this search. Keep the truthful initialization as a candidate.
    Optional checkpoints resume at 50-step boundaries. The caller must bind a
    checkpoint to the same mechanism, environment and policy architecture.
    """
    if num_reopt_steps < 0 or num_reopt_restarts < 1:
        raise ValueError("Require nonnegative steps and at least one restart")
    mechanism = BudgetMechanism(mechanism_network, mechanism_params)
    w_opt     = optax.adam(lr_reopt)
    prng      = hk.PRNGSequence(seed)
    selected, scores, terminal_states = [], [], []
    controls = {"seed": seed, "steps": num_reopt_steps, "restarts": num_reopt_restarts,
                "lr": lr_reopt, "samples": num_samples, "buyers": auction.num_buyers}
    start_buyer, start_restart, start_offset = 0, 0, 0
    if resume_state is not None:
        if resume_state.get("schema") != 1 or resume_state["controls"] != controls:
            raise ValueError("Pathwise checkpoint controls differ")
        selected, scores = list(resume_state["selected"]), list(resume_state["scores"])
        terminal_states = list(resume_state["terminal_states"])
        start_buyer, start_restart, start_offset = resume_state["cursor"]
        prng.replace_internal_state(resume_state["rng_state"])

    def save(buyer, restart, offset, params=None, opt_state=None,
             best_score=-float("inf"), best_params=None):
        if checkpoint_callback is not None:
            checkpoint_callback({"schema": 1, "controls": controls,
                "cursor": (buyer, restart, offset), "selected": list(selected),
                "scores": list(scores), "params": params, "opt_state": opt_state,
                "best_score": best_score, "best_params": best_params,
                "rng_state": prng.internal_state, "terminal_states": list(terminal_states)})

    validation_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1000003)
    for buyer in range(start_buyer, auction.num_buyers):
        best_score, best_params = -float("inf"), None
        continuing_buyer = resume_state is not None and buyer == start_buyer
        if continuing_buyer:
            best_score, best_params = resume_state["best_score"], resume_state["best_params"]
        for _restart in range(start_restart if continuing_buyer else 0, num_reopt_restarts):
            continuing_restart = continuing_buyer and _restart == start_restart and resume_state["params"] is not None
            if continuing_restart:
                params, opt_state, offset = resume_state["params"], resume_state["opt_state"], start_offset
                if not bool(_finite_reopt_state(params, opt_state)):
                    raise FloatingPointError("Nonfinite resumed pathwise state")
            else:
                initial = auction.get_initial_belief_state(auction.get_initial_state(next(prng)))
                params = policy_network.init(next(prng), initial, auction)
                opt_state, offset = w_opt.init(params), 0
            while True:
                # Evaluate the parameter tree that will actually be saved.
                score = float(BudgetSimulator.estimate_cumulative_deviation_gains(
                    auction, mechanism, BoundedContPolicy(policy_network, params),
                    max(256, num_samples), validation_key)[buyer])
                if not math.isfinite(score):
                    raise FloatingPointError(f"Nonfinite pathwise validation score: buyer={buyer}, restart={_restart}, step={offset}")
                if score > best_score:
                    best_score, best_params = score, params
                steps = min(50, num_reopt_steps-offset)
                if not steps:
                    # Preserve the final optimizer/RNG state as well as the
                    # winning parameters, which may come from an earlier step.
                    save(buyer, _restart, offset, params, opt_state, best_score, best_params)
                    break
                params, opt_state, _ = _reopt_chunk(
                    auction, mechanism, policy_network, params, opt_state,
                    next(prng), lr_reopt, steps, buyer, num_samples)
                offset += steps
                if not bool(_finite_reopt_state(params, opt_state)):
                    raise FloatingPointError(f"Nonfinite pathwise optimizer state: buyer={buyer}, restart={_restart}, step={offset}")
                save(buyer, _restart, offset, params, opt_state, best_score, best_params)
            terminal_states.append({"buyer": buyer, "restart": _restart, "steps": offset,
                "params": params, "opt_state": opt_state, "rng_state": prng.internal_state})
        selected.append(best_params)
        scores.append(max(0.0, best_score))
        # A completed buyer needs no reoptimization after resumption.
        save(buyer+1, 0, 0)
    return sum(scores), tuple(selected)


# ---------------------------------------------------------------------------
# Step 2 — Evaluate a list of checkpoints on held-out keys
# ---------------------------------------------------------------------------

def evaluate_checkpoint_list(
    auction,
    mechanism_network,
    policy_network,
    checkpoints: List[Dict],
    profit_goal: float,
    cfg: Dict,
    seed: int = SEED,
    verbose: bool = True,
) -> List[Dict]:
    """
    Evaluate every checkpoint in `checkpoints` on held-out eval keys.

    IC regret is measured by re-optimising the oracle policy from scratch
    against each fixed mechanism (see `reopt_cumulative_ic`).  This avoids
    reusing the training oracle; the result remains an empirical search.

    Revenue, IR loss, and violation rates are measured with `eval_budget_mechanism_multikey`
    using the re-optimised oracle as the deviation policy.

    Adds keys 'revenue_eval', 'regret_eval', 'ir_loss_eval',
    'budget_viol_eval', 'ir_viol_eval', 'feasible' to each checkpoint dict.

    Returns the annotated list sorted by revenue_eval (descending among feasible
    ones first, then rest).
    """
    eval_keys = jax.random.split(
        jax.random.PRNGKey(seed + 77777), cfg["eval_num_keys"]
    )
    ic_thresh  = cfg["ic_thresh"]
    gap_thresh = cfg.get("gap_thresh", 0.1)

    results = []
    for i, ckpt in enumerate(checkpoints):
        mparams = ckpt["mechanism_params"]

        # Skip expensive IC re-opt + eval if training gap is already too large.
        # gap_train ≈ profit_goal - revenue_train; if it exceeds 2×gap_thresh the
        # checkpoint cannot pass the feasibility check (which requires eval gap ≤ gap_thresh).
        gap_train     = ckpt.get("gap_train", float("inf"))
        skip_gap_mult = cfg.get("skip_gap_mult", 2.0)
        if gap_train > skip_gap_mult * gap_thresh:
            if verbose:
                print(f"    [skip] epoch={ckpt['epoch']:4d} — "
                      f"gap_train={gap_train:.4f} > {skip_gap_mult}×gap_thresh={skip_gap_mult*gap_thresh:.4f}")
            rec = dict(ckpt)
            rec["revenue_eval"]       = float("nan")
            rec["regret_reopt"]       = float("nan")
            rec["regret_eval"]        = float("nan")
            rec["ir_loss_eval"]       = float("nan")
            rec["buyer_surplus_eval"] = float("nan")
            rec["budget_viol_eval"]   = float("nan")
            rec["ir_viol_eval"]       = float("nan")
            rec["feasible"]           = False
            results.append(rec)
            continue

        if verbose:
            print(f"    [ckpt {i+1}/{len(checkpoints)}] epoch={ckpt['epoch']:4d} — "
                  f"re-optimising oracle ({cfg['ic_reopt_restarts']} restarts × "
                  f"{cfg['ic_reopt_steps']} steps) ...")

        # --- Step 1: re-optimise oracle from scratch against fixed mechanism ---
        _reopt_seed = seed + 88888 + i
        best_regret, best_pparams = reopt_cumulative_ic(
            auction            = auction,
            mechanism_network  = mechanism_network,
            mechanism_params   = mparams,
            policy_network     = policy_network,
            num_reopt_steps    = cfg["ic_reopt_steps"],
            num_reopt_restarts = cfg["ic_reopt_restarts"],
            lr_reopt           = cfg["ic_reopt_lr"],
            num_samples        = cfg["ic_reopt_samples"],
            seed               = _reopt_seed,
        )

        # --- Step 2: evaluate revenue / IR / violations + IC with re-opt oracle ---
        ev = eval_budget_mechanism_multikey(
            auction           = auction,
            mechanism_network = mechanism_network,
            mechanism_params  = mparams,
            profit_goal       = profit_goal,
            num_samples       = cfg["eval_num_samples"],
            keys              = eval_keys,
            policy_network    = policy_network,
            policy_params     = best_pparams,   # re-optimised oracle
        )

        rec = dict(ckpt)
        rec["training_policy_params"] = ckpt.get("policy_params")
        rec["policy_params"] = best_pparams
        rec["policy_params_reopt"] = best_pparams
        rec["revenue_eval"]        = ev["profit_stats"]["mean"]         if "profit_stats"         in ev else float("nan")
        # Use the max regret from reopt (highest across restarts) as the IC estimate.
        # Also record the multi-key regret mean from eval for comparison.
        rec["regret_reopt"]        = best_regret
        rec["regret_eval"]         = ev["ic_sum"]
        rec["ic_per_buyer"]        = ev["ic_per_buyer"]
        rec["ic_sampling_ucb95"]   = ev["ic_sum_sampling_ucb95"]
        rec["evaluation_role"]     = "checkpoint_selection_validation"
        # ir_loss = relu(-reward): always 0 by payment-feasibility construction; used for pass/fail.
        rec["ir_loss_eval"]        = ev["ir_loss_stats"]["mean"]        if "ir_loss_stats"        in ev else float("nan")
        # buyer_surplus = sum(reward): informative positive metric shown in plots.
        rec["buyer_surplus_eval"]  = ev["buyer_surplus_stats"]["mean"]  if "buyer_surplus_stats"  in ev else float("nan")
        rec["budget_viol_eval"]    = ev["budget_viol_stats"]["mean"]    if "budget_viol_stats"    in ev else float("nan")
        rec["ir_viol_eval"]        = ev["ir_viol_stats"]["mean"]        if "ir_viol_stats"        in ev else float("nan")
        # Feasibility: IC must be small AND revenue must be within gap_thresh of profit_goal.
        # The gap check prevents a low-revenue IC solution from falsely marking this
        # profit_goal feasible and inflating pg_lo.
        _ic_for_check  = rec["ic_sampling_ucb95"]
        _revenue_gap   = profit_goal - rec["revenue_eval"]
        rec["feasible"] = ((_ic_for_check <= ic_thresh) and (_revenue_gap <= gap_thresh)
                           and rec["ir_loss_eval"] <= 1e-6
                           and rec["budget_viol_eval"] == 0 and rec["ir_viol_eval"] == 0)

        results.append(rec)
        if verbose:
            flag = "PASS" if rec["feasible"] else "fail"
            print(f"    [{flag}] epoch={rec['epoch']:4d}  "
                  f"rev={rec['revenue_eval']:.4f}  gap={_revenue_gap:.4f}  "
                  f"IC_reopt={best_regret:.4f}  IC_eval={rec['regret_eval']:.4f}  "
                  f"surplus={rec.get('buyer_surplus_eval', float('nan')):.4f}")

    # Sort: feasible first (by revenue desc), then infeasible
    results.sort(key=lambda r: (not r["feasible"], -r["revenue_eval"]))
    return results


# ---------------------------------------------------------------------------
# Step 3 — Binary search over profit_goal
# ---------------------------------------------------------------------------

def run_binary_search(
    auction,
    mechanism_network,
    policy_network,
    variant: str             = "A",
    save_dir: str            = "outputs/binary_search",
    cfg: Optional[Dict]      = None,
    base_seed: int           = SEED,
    verbose: bool            = True,
    time_budget_hours: float = 11.5,
    force_restart: bool      = False,
) -> Dict:
    """
    Binary search over profit_goal.

    At each bisection step:
      pg_mid = (pg_lo + pg_hi) / 2
      Train mechanism with profit_goal = pg_mid (full epochs, with checkpoints)
      Evaluate all checkpoints on held-out keys
      If any checkpoint is feasible (IC ≤ ic_thresh AND IR ≤ ir_thresh):
          pg_lo = pg_mid   (feasible — push higher)
          Record best mechanism for this level
      Else:
          pg_hi = pg_mid   (infeasible — reduce target)
    Terminate when pg_hi - pg_lo < pg_tol.

    All trial results are saved incrementally to `save_dir/bs_results.json`.
    Best mechanism per trial is saved to `save_dir/best_mech_<pg>.pkl`.

    Args:
        force_restart : if True, ignore any existing bs_results.json and start
                        from scratch.  Use a different save_dir to keep old results
                        alongside the new run (recommended when changing configs).

    Returns:
        summary dict with pg_lo (highest confirmed feasible), best mechanism info,
        and full trial records.
    """
    cfg = {**DEFAULT_BS_CFG, **(cfg or {})}
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "bs_results.json")
    summary_path = os.path.join(save_dir, "bs_summary.json")

    # Load prior results for resumption (skipped if force_restart=True)
    all_trials: List[Dict] = []
    if not force_restart and os.path.exists(results_path):
        with open(results_path) as f:
            all_trials = json.load(f)
        if verbose:
            print(f"[binary_search] Resuming from {results_path} "
                  f"({len(all_trials)} prior trials)")
    elif force_restart and verbose:
        print(f"[binary_search] force_restart=True — ignoring any existing results in {save_dir}")

    pg_lo   = cfg["pg_lo"]
    pg_hi   = cfg["pg_hi"]
    pg_tol  = cfg["pg_tol"]

    # Restore pg_lo/pg_hi from prior trials
    for t in all_trials:
        if t.get("any_feasible"):
            pg_lo = max(pg_lo, t["profit_goal"])
        else:
            pg_hi = min(pg_hi, t["profit_goal"])

    best_mech_info = None  # track overall best feasible mechanism
    for t in all_trials:
        if t.get("any_feasible") and t.get("best_revenue_eval") is not None:
            if best_mech_info is None or t["best_revenue_eval"] > best_mech_info.get("revenue_eval", -1):
                best_mech_info = {
                    "profit_goal":        t["profit_goal"],
                    "revenue_eval":       t["best_revenue_eval"],
                    "regret_eval":        t["best_regret_eval"],
                    "buyer_surplus_eval": t.get("best_buyer_surplus_eval"),
                    "mech_pkl":           t.get("best_mech_pkl"),
                }

    t0_global = time.time()
    trial_seed = base_seed

    if verbose:
        print(f"[binary_search] Starting: pg_lo={pg_lo:.3f}  pg_hi={pg_hi:.3f}  "
              f"pg_tol={pg_tol:.3f}  variant={variant}")

    iteration = 0
    while pg_hi - pg_lo > pg_tol:
        elapsed_h = (time.time() - t0_global) / 3600.0
        if elapsed_h >= time_budget_hours:
            if verbose:
                print(f"[binary_search] Time budget {time_budget_hours}h reached — stopping.")
            break

        pg_mid = (pg_lo + pg_hi) / 2.0
        iteration += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"[binary_search] Iteration {iteration}: pg_mid={pg_mid:.3f}  "
                  f"[pg_lo={pg_lo:.3f}, pg_hi={pg_hi:.3f}]")
            print(f"  Elapsed: {elapsed_h:.2f}h / {time_budget_hours}h")

        # Skip if already evaluated (exact match)
        already_done = [t for t in all_trials if abs(t["profit_goal"] - pg_mid) < 1e-6]
        if already_done:
            if verbose:
                print(f"  [binary_search] pg_mid={pg_mid:.3f} already evaluated — skipping.")
            t_rec = already_done[0]
            if t_rec.get("any_feasible"):
                pg_lo = max(pg_lo, pg_mid)
            else:
                pg_hi = min(pg_hi, pg_mid)
            trial_seed += 1
            continue

        t_trial_start = time.time()

        # Train
        checkpoints, logs = train_with_checkpoints(
            auction           = auction,
            mechanism_network = mechanism_network,
            policy_network    = policy_network,
            profit_goal       = pg_mid,
            cfg               = cfg,
            seed              = trial_seed,
            verbose           = verbose,
        )

        # Evaluate all checkpoints
        if verbose:
            print(f"  [binary_search] Evaluating {len(checkpoints)} checkpoints ...")
        eval_results = evaluate_checkpoint_list(
            auction           = auction,
            mechanism_network = mechanism_network,
            policy_network    = policy_network,
            checkpoints       = checkpoints,
            profit_goal       = pg_mid,
            cfg               = cfg,
            seed              = trial_seed,
            verbose           = verbose,
        )

        # Is this profit_goal feasible?
        feasible_ckpts = [r for r in eval_results if r["feasible"]]
        any_feasible   = len(feasible_ckpts) > 0

        # Save best mechanism for this trial
        best_pkl     = None
        best_rev     = None
        best_reg     = None
        best_surplus = None
        if any_feasible:
            best_ckpt    = feasible_ckpts[0]   # sorted by revenue desc
            best_rev     = best_ckpt["revenue_eval"]
            best_reg     = best_ckpt["regret_eval"]
            best_surplus = best_ckpt.get("buyer_surplus_eval")
            best_pkl     = os.path.join(save_dir, f"best_mech_pg{pg_mid:.3f}.pkl")
            with open(best_pkl, "wb") as f_pkl:
                pickle.dump({
                    "profit_goal":         pg_mid,
                    "mechanism_params":    best_ckpt["mechanism_params"],
                    "policy_params":       best_ckpt["policy_params"],
                    "epoch":               best_ckpt["epoch"],
                    "revenue_eval":        best_rev,
                    "regret_eval":         best_reg,
                    "buyer_surplus_eval":  best_surplus,
                }, f_pkl)
            if verbose:
                print(f"  [binary_search] FEASIBLE  rev={best_rev:.4f}  "
                      f"IC={best_reg:.4f}  surplus={best_surplus:.4f}  "
                      f"→ saved {best_pkl}")
            pg_lo = pg_mid

            if (best_mech_info is None
                    or best_rev > best_mech_info.get("revenue_eval", -1)):
                best_mech_info = {
                    "profit_goal":        pg_mid,
                    "revenue_eval":       best_rev,
                    "regret_eval":        best_reg,
                    "buyer_surplus_eval": best_surplus,
                    "mech_pkl":           best_pkl,
                }
        else:
            if verbose:
                print(f"  [binary_search] INFEASIBLE — best eval: "
                      f"rev={eval_results[0]['revenue_eval']:.4f}  "
                      f"IC={eval_results[0]['regret_eval']:.4f}  "
                      f"surplus={eval_results[0].get('buyer_surplus_eval', float('nan')):.4f}")
            pg_hi = pg_mid

        # Save full per-epoch training curves (reg, ir, gap, loss) as JSON
        _curves_path = os.path.join(
            save_dir, f"training_curves_pg{pg_mid:.3f}.json"
        )
        _curves = {
            "profit_goal": pg_mid,
            "reg":         [float(x) for x in logs.get("reg",      [])],
            "ir":          [float(x) for x in logs.get("ir",       [])],
            "gap":         [float(x) for x in logs.get("gap",      [])],
            "reg_plus":    [float(x) for x in logs.get("reg_plus", [])],
            "gap_plus":    [float(x) for x in logs.get("gap_plus", [])],
            "loss":        [float(x) for x in logs.get("loss",     [])],
            # Sparse checkpoint annotations
            "checkpoints_train": [
                {
                    "epoch":     c["epoch"],
                    "reg_train": c["reg_train"],
                    "ir_train":  c["ir_train"],
                    "gap_train": c["gap_train"],
                }
                for c in checkpoints
            ],
            # Eval results per checkpoint (aligned to checkpoints list order)
            "checkpoints_eval": [
                {
                    "epoch":               r["epoch"],
                    "revenue_eval":        r["revenue_eval"],
                    "regret_reopt":        r.get("regret_reopt"),
                    "regret_eval":         r["regret_eval"],
                    "ir_loss_eval":        r["ir_loss_eval"],        # relu(-reward), always 0
                    "buyer_surplus_eval":  r.get("buyer_surplus_eval"),  # sum(reward), informative
                    "feasible":            r["feasible"],
                }
                for r in eval_results
            ],
        }
        with open(_curves_path, "w") as f:
            json.dump(_curves, f, indent=2)

        # Serialisable trial record (no JAX arrays)
        trial_record = {
            "profit_goal":             pg_mid,
            "any_feasible":            any_feasible,
            "best_revenue_eval":       best_rev,
            "best_regret_eval":        best_reg,
            "best_buyer_surplus_eval": best_surplus,
            "best_mech_pkl":           best_pkl,
            "training_curves":    _curves_path,
            "num_checkpoints":    len(checkpoints),
            "num_feasible_ckpts": len(feasible_ckpts),
            "trial_time_s":       time.time() - t_trial_start,
            "iteration":          iteration,
            "seed":               trial_seed,
        }
        all_trials.append(trial_record)

        # Fault-tolerant save after every trial
        with open(results_path, "w") as f:
            json.dump(all_trials, f, indent=2)

        trial_seed += 1

    # Final summary
    elapsed_total = (time.time() - t0_global) / 3600.0
    summary = {
        "variant":          variant,
        "pg_lo":            pg_lo,
        "pg_hi":            pg_hi,
        "pg_tol":           pg_tol,
        "converged":        (pg_hi - pg_lo) <= pg_tol,
        "num_trials":       iteration,
        "total_time_h":     elapsed_total,
        "best_mechanism":   best_mech_info,
        "cfg":              {k: v for k, v in cfg.items()
                             if not callable(v)},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[binary_search] DONE — {elapsed_total:.2f}h total")
        print(f"  Highest confirmed feasible profit_goal: {pg_lo:.3f}")
        print(f"  pg_hi (infeasible): {pg_hi:.3f}")
        if best_mech_info:
            print(f"  Best mechanism: rev={best_mech_info['revenue_eval']:.4f}  "
                  f"IC={best_mech_info['regret_eval']:.4f}  "
                  f"surplus={best_mech_info.get('buyer_surplus_eval', float('nan')):.4f}")
            print(f"  Saved to: {best_mech_info['mech_pkl']}")
        print(f"  Summary: {summary_path}")

    return summary
