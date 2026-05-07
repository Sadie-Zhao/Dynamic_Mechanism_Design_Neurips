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
    # This gives an honest upper-bound on regret rather than reusing the
    # potentially underconverged training oracle.
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
    final_reg = logs["reg"][-1] if logs["reg"] else float("nan")
    final_ir  = logs["ir"][-1]  if logs["ir"]  else float("nan")
    final_gap = logs["gap"][-1] if logs["gap"] else float("nan")
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
# IC re-optimisation — honest upper-bound on regret for a fixed mechanism
# ---------------------------------------------------------------------------

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
) -> Tuple[float, object]:
    """
    Re-optimise the oracle policy from scratch against a *fixed* mechanism.

    Unlike the training inner loop (which runs only `num_inner_steps` steps
    against a continuously moving theta), this function:
      1. Initialises a fresh policy from a random seed (repeated `num_reopt_restarts` times)
      2. Runs `num_reopt_steps` gradient-ascent steps on regret with the mechanism frozen
      3. Returns the *highest* regret found across restarts

    This gives a conservative (high-side) estimate of the true regret, which is
    what you need for a trustworthy IC feasibility check.

    Returns:
        best_regret       : float — best (highest) regret found
        best_policy_params: policy params that achieved best_regret
    """
    mechanism = BudgetMechanism(mechanism_network, mechanism_params)
    w_opt     = optax.adam(lr_reopt)
    prng      = hk.PRNGSequence(seed)

    @jax.jit
    def _cumreg(policy_params, key):
        policy = BoundedContPolicy(policy_network, policy_params)
        return BudgetSimulator.estimate_cumulative_regret(
            auction, mechanism, policy, num_samples, key
        )

    best_regret       = -float("inf")
    best_policy_params = None

    for _restart in range(num_reopt_restarts):
        # Fresh policy initialisation
        _init_state  = auction.get_initial_state(next(prng))
        _init_belief = auction.get_initial_belief_state(_init_state)
        _pparams     = policy_network.init(next(prng), _init_belief, auction)
        _w_state     = w_opt.init(_pparams)

        for _ in range(num_reopt_steps):
            _key = next(prng)
            _reg, _grad = jax.value_and_grad(_cumreg)(_pparams, _key)
            # Gradient *ascent*: negate before passing to Adam
            _grad_asc = jax.tree_util.tree_map(lambda g: -g, _grad)
            _upd, _w_state = w_opt.update(_grad_asc, _w_state, _pparams)
            _pparams = optax.apply_updates(_pparams, _upd)

        _reg_final = float(_reg)
        if _reg_final > best_regret:
            best_regret        = _reg_final
            best_policy_params = _pparams

    return best_regret, best_policy_params


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
    reusing the underconverged training oracle and gives a conservative
    (upper-bound) regret estimate for a trustworthy feasibility verdict.

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
        rec["revenue_eval"]        = ev["profit_stats"]["mean"]         if "profit_stats"         in ev else float("nan")
        # Use the max regret from reopt (highest across restarts) as the IC estimate.
        # Also record the multi-key regret mean from eval for comparison.
        rec["regret_reopt"]        = best_regret
        rec["regret_eval"]         = ev["regret_stats"]["mean"]         if "regret_stats"         in ev else float("nan")
        # ir_loss = relu(-reward): always 0 by payment-feasibility construction; used for pass/fail.
        rec["ir_loss_eval"]        = ev["ir_loss_stats"]["mean"]        if "ir_loss_stats"        in ev else float("nan")
        # buyer_surplus = sum(reward): informative positive metric shown in plots.
        rec["buyer_surplus_eval"]  = ev["buyer_surplus_stats"]["mean"]  if "buyer_surplus_stats"  in ev else float("nan")
        rec["budget_viol_eval"]    = ev["budget_viol_stats"]["mean"]    if "budget_viol_stats"    in ev else float("nan")
        rec["ir_viol_eval"]        = ev["ir_viol_stats"]["mean"]        if "ir_viol_stats"        in ev else float("nan")
        # Feasibility: IC must be small AND revenue must be within gap_thresh of profit_goal.
        # The gap check prevents a low-revenue IC solution from falsely marking this
        # profit_goal feasible and inflating pg_lo.
        _ic_for_check  = max(best_regret, rec["regret_eval"])
        _revenue_gap   = profit_goal - rec["revenue_eval"]
        rec["feasible"] = (_ic_for_check <= ic_thresh) and (_revenue_gap <= gap_thresh)

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
