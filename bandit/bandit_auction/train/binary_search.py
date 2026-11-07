"""
binary_search.py — Binary search over profit_goal for bandit auction mechanisms.

Protocol:
  Given a fixed hyperparameter config, find the highest profit_goal p* such that a
  mechanism trained with that goal achieves IC regret ≤ ic_thresh on held-out evaluation
  keys.  (No budget constraints or IR violation rates — bandit setting has soft IR only.)

  alpha is scaled proportionally: alpha(p) = alpha_base * (p / pg_base)

Key idea: record ALL mechanism checkpoints during training. After training, evaluate
  every checkpoint on held-out keys. A profit_goal is "feasible" if at least one
  checkpoint passes both IC and IR thresholds.

Usage (programmatic):
    from bandit_auction.train.binary_search import run_binary_search
    result = run_binary_search(auction, mechanism_network, policy_network,
                               experiment="cont_single",
                               save_dir="outputs/bs/cont_single")
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

from bandit_auction.core import (
    AuctionSimulator, BoundedContPolicy, DiscretePolicy, SEED,
)
from bandit_auction.mechanisms.network import Mechanism
from bandit_auction.train.run_acceptance import nested_GD_acceptance
from bandit_auction.train.evaluate import eval_mechanism_multikey


# ---------------------------------------------------------------------------
# Default binary-search hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_BS_CFG = dict(
    lr_theta        = 1e-4,
    lr_w            = 1e-3,

    alpha_base      = 5.0,
    beta_weight     = 1.0,

    num_inner_steps  = 400,
    num_outer_epochs = 2000,
    num_samples_train = 64,
    n_inner_restarts = 3,

    checkpoint_thresh   = 0.15,
    checkpoint_interval = 50,

    pg_lo   = 0.0,
    pg_hi   = 10.0,
    pg_base = 1.0,
    pg_tol  = 0.1,

    ic_thresh  = 0.05,
    gap_thresh = 0.1,

    ic_reopt_steps    = 600,
    ic_reopt_restarts = 3,
    ic_reopt_lr       = 5e-4,
    ic_reopt_samples  = 128,

    eval_num_keys    = 32,
    eval_num_samples = 256,

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
    is_discrete: bool = False,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """Train a mechanism for `profit_goal` and return checkpoints + full logs."""
    alpha = cfg["alpha_base"] * (profit_goal / cfg["pg_base"])
    if verbose:
        print(f"  [train] profit_goal={profit_goal:.3f}  alpha={alpha:.2f}  "
              f"lr_theta={cfg['lr_theta']:.1e}  lr_w={cfg['lr_w']:.1e}")

    logs = nested_GD_acceptance(
        auction             = auction,
        mechanism_network   = mechanism_network,
        policy_network      = policy_network,
        profit_goal         = profit_goal,
        num_outer_epochs    = cfg["num_outer_epochs"],
        lr_theta            = cfg["lr_theta"],
        num_inner_steps     = cfg["num_inner_steps"],
        lr_w                = cfg["lr_w"],
        num_samples_train   = cfg["num_samples_train"],
        alpha_weight        = alpha,
        beta_weight         = cfg["beta_weight"],
        print_iter          = cfg.get("print_iter", 100),
        checkpoint_thresh   = cfg["checkpoint_thresh"],
        checkpoint_interval = cfg["checkpoint_interval"],
        n_inner_restarts    = cfg.get("n_inner_restarts", 3),
        is_discrete         = is_discrete,
        seed                = seed,
    )

    checkpoints = logs["checkpoint_mech"]

    final_reg = logs["reg"][-1] if logs["reg"] else float("nan")
    final_ir  = logs["ir"][-1]  if logs["ir"]  else float("nan")
    final_gap = logs["gap"][-1] if logs["gap"] else float("nan")
    checkpoints_with_final = checkpoints + [{
        "epoch":            cfg["num_outer_epochs"] - 1,
        "reg_train":        final_reg,
        "ir_train":         final_ir,
        "gap_train":        final_gap,
        "mechanism_params": logs["mechanism_params"],
        "policy_params":    logs["policy_params"],
    }]

    if verbose:
        print(f"  [train] done — {len(checkpoints)} threshold-passing checkpoints "
              f"(+ final); total candidates: {len(checkpoints_with_final)}")

    return checkpoints_with_final, logs


# ---------------------------------------------------------------------------
# IC re-optimisation
# ---------------------------------------------------------------------------

def reopt_cumulative_ic(
    auction,
    mechanism_network,
    mechanism_params,
    policy_network,
    is_discrete:        bool  = False,
    num_reopt_steps:    int   = 600,
    num_reopt_restarts: int   = 3,
    lr_reopt:           float = 5e-4,
    num_samples:        int   = 128,
    seed:               int   = SEED,
) -> Tuple[float, object]:
    """Re-optimise oracle policy from scratch against a fixed mechanism.

    Returns:
        best_regret:        highest regret found across restarts
        best_policy_params: params achieving best_regret
    """
    mechanism = Mechanism(mechanism_network, mechanism_params)
    w_opt     = optax.adam(lr_reopt)
    prng      = hk.PRNGSequence(seed)

    update_belief_fn = (
        auction.get_next_belief_state_discrete if is_discrete
        else auction.get_next_belief_state_continuous
    )

    @jax.jit
    def _cumreg(policy_params, key):
        policy_cls = DiscretePolicy if is_discrete else BoundedContPolicy
        policy = policy_cls(policy_network, policy_params)
        return AuctionSimulator.estimate_cumulative_regret(
            auction, mechanism, policy, num_samples, key, update_belief_fn
        )

    best_regret       = -float("inf")
    best_policy_params = None

    for _restart in range(num_reopt_restarts):
        _init_state  = auction.get_initial_state(next(prng))
        _init_belief = auction.get_initial_belief_state(_init_state)
        _pparams     = policy_network.init(next(prng), _init_belief, auction)
        _w_state     = w_opt.init(_pparams)

        for _ in range(num_reopt_steps):
            _key = next(prng)
            _reg, _grad = jax.value_and_grad(_cumreg)(_pparams, _key)
            _grad_asc  = jax.tree_util.tree_map(lambda g: -g, _grad)
            _upd, _w_state = w_opt.update(_grad_asc, _w_state, _pparams)
            _pparams = optax.apply_updates(_pparams, _upd)

        _reg_final = float(_reg)
        if _reg_final > best_regret:
            best_regret        = _reg_final
            best_policy_params = _pparams

    return best_regret, best_policy_params


# ---------------------------------------------------------------------------
# Step 2 — Evaluate a list of checkpoints
# ---------------------------------------------------------------------------

def evaluate_checkpoint_list(
    auction,
    mechanism_network,
    policy_network,
    checkpoints: List[Dict],
    profit_goal: float,
    cfg: Dict,
    is_discrete: bool = False,
    seed: int = SEED,
    verbose: bool = True,
    skip_thresh: float = 2.0,
) -> List[Dict]:
    """Evaluate checkpoints on held-out eval keys, skipping obvious losers.

    IC regret is measured via reopt_cumulative_ic (conservative upper bound).
    Adds keys 'revenue_eval', 'regret_eval', 'regret_reopt', 'ir_loss_eval',
    'buyer_surplus_eval', 'feasible', 'skipped' to each checkpoint dict.

    Checkpoints whose training gap exceeds skip_thresh * gap_thresh are marked
    infeasible immediately without running the expensive IC reopt oracle.
    Set skip_thresh=float('inf') to disable skipping entirely.

    Training gap is read from 'gap_train' (acceptance checkpoints) or computed
    as profit_goal - profit_train (penalty checkpoints). If neither key is
    present the checkpoint is always evaluated.
    """
    eval_keys = jax.random.split(
        jax.random.PRNGKey(seed + 77777), cfg["eval_num_keys"]
    )
    ic_thresh  = cfg["ic_thresh"]
    gap_thresh = cfg.get("gap_thresh", 0.1)
    _skip_gap  = skip_thresh * gap_thresh

    results  = []
    n_skip   = 0
    n_eval   = 0
    for i, ckpt in enumerate(checkpoints):
        mparams = ckpt["mechanism_params"]

        # ---- Fast pre-screen on training gap ----
        if "gap_train" in ckpt:
            _train_gap = float(ckpt["gap_train"])
        elif "profit_train" in ckpt:
            _train_gap = profit_goal - float(ckpt["profit_train"])
        else:
            _train_gap = float("nan")  # unknown — don't skip

        if _train_gap > _skip_gap:
            rec = dict(ckpt)
            rec["revenue_eval"]       = float("nan")
            rec["regret_reopt"]       = float("nan")
            rec["regret_eval"]        = float("nan")
            rec["ir_loss_eval"]       = float("nan")
            rec["buyer_surplus_eval"] = float("nan")
            rec["feasible"]           = False
            rec["skipped"]            = True
            results.append(rec)
            n_skip += 1
            if verbose:
                print(f"    [skip] epoch={ckpt['epoch']:4d}  "
                      f"train_gap={_train_gap:.4f} > {_skip_gap:.3f} "
                      f"({skip_thresh}×gap_thresh) — IC eval skipped")
            continue

        # ---- Full IC reopt + eval ----
        n_eval += 1
        if verbose:
            print(f"    [ckpt {i+1}/{len(checkpoints)}] epoch={ckpt['epoch']:4d} — "
                  f"re-optimising oracle ({cfg['ic_reopt_restarts']} restarts × "
                  f"{cfg['ic_reopt_steps']} steps) ...")

        _reopt_seed = seed + 88888 + i
        best_regret, best_pparams = reopt_cumulative_ic(
            auction             = auction,
            mechanism_network   = mechanism_network,
            mechanism_params    = mparams,
            policy_network      = policy_network,
            is_discrete         = is_discrete,
            num_reopt_steps     = cfg["ic_reopt_steps"],
            num_reopt_restarts  = cfg["ic_reopt_restarts"],
            lr_reopt            = cfg["ic_reopt_lr"],
            num_samples         = cfg["ic_reopt_samples"],
            seed                = _reopt_seed,
        )

        ev = eval_mechanism_multikey(
            auction           = auction,
            mechanism_network = mechanism_network,
            mechanism_params  = mparams,
            profit_goal       = profit_goal,
            num_samples       = cfg["eval_num_samples"],
            keys              = eval_keys,
            policy_network    = policy_network,
            policy_params     = best_pparams,
            is_discrete       = is_discrete,
        )

        rec = dict(ckpt)
        rec["revenue_eval"]       = ev["profit_stats"]["mean"]        if "profit_stats"        in ev else float("nan")
        rec["regret_reopt"]       = best_regret
        rec["regret_eval"]        = ev["regret_stats"]["mean"]        if "regret_stats"        in ev else float("nan")
        rec["ir_loss_eval"]       = ev["ir_loss_stats"]["mean"]       if "ir_loss_stats"       in ev else float("nan")
        rec["buyer_surplus_eval"] = ev["buyer_surplus_stats"]["mean"] if "buyer_surplus_stats" in ev else float("nan")
        rec["skipped"]            = False

        _ic_for_check = max(best_regret, rec["regret_eval"])
        _revenue_gap  = profit_goal - rec["revenue_eval"]
        _ir_for_check = rec["ir_loss_eval"]
        rec["feasible"] = (
            (_ic_for_check <= ic_thresh)
            and (_revenue_gap  <= gap_thresh)
            and (_ir_for_check <= ic_thresh)
        )

        results.append(rec)
        if verbose:
            flag = "PASS" if rec["feasible"] else "fail"
            print(f"    [{flag}] epoch={rec['epoch']:4d}  "
                  f"rev={rec['revenue_eval']:.4f}  gap={_revenue_gap:.4f}  "
                  f"IC_reopt={best_regret:.4f}  IC_eval={rec['regret_eval']:.4f}  "
                  f"IR_eval={_ir_for_check:.4f}  "
                  f"surplus={rec.get('buyer_surplus_eval', float('nan')):.4f}")

    if verbose and n_skip:
        print(f"  — skipped {n_skip}/{len(checkpoints)} checkpoints "
              f"(train gap > {_skip_gap:.3f}); ran IC eval on {n_eval}.")

    results.sort(key=lambda r: (not r["feasible"], -(r["revenue_eval"] or 0.0)))
    return results


# ---------------------------------------------------------------------------
# Step 3 — Binary search over profit_goal
# ---------------------------------------------------------------------------

def run_binary_search(
    auction,
    mechanism_network,
    policy_network,
    experiment: str          = "cont_single",
    save_dir: str            = "outputs/binary_search",
    cfg: Optional[Dict]      = None,
    is_discrete: bool        = False,
    base_seed: int           = SEED,
    verbose: bool            = True,
    time_budget_hours: float = 11.5,
    force_restart: bool      = False,
) -> Dict:
    """Binary search over profit_goal for the bandit auction setting.

    Returns summary dict with pg_lo (highest confirmed feasible) and best mechanism info.
    """
    cfg = {**DEFAULT_BS_CFG, **(cfg or {})}
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "bs_results.json")
    summary_path = os.path.join(save_dir, "bs_summary.json")

    all_trials: List[Dict] = []
    if not force_restart and os.path.exists(results_path):
        with open(results_path) as f:
            all_trials = json.load(f)
        if verbose:
            print(f"[binary_search] Resuming from {results_path} "
                  f"({len(all_trials)} prior trials)")
    elif force_restart and verbose:
        print(f"[binary_search] force_restart=True — ignoring any existing results in {save_dir}")

    pg_lo  = cfg["pg_lo"]
    pg_hi  = cfg["pg_hi"]
    pg_tol = cfg["pg_tol"]

    for t in all_trials:
        if t.get("any_feasible"):
            pg_lo = max(pg_lo, t["profit_goal"])
        else:
            pg_hi = min(pg_hi, t["profit_goal"])

    best_mech_info = None
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

    t0_global  = time.time()
    trial_seed = base_seed

    if verbose:
        print(f"[binary_search] Starting: pg_lo={pg_lo:.3f}  pg_hi={pg_hi:.3f}  "
              f"pg_tol={pg_tol:.3f}  experiment={experiment}")

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

        checkpoints, logs = train_with_checkpoints(
            auction           = auction,
            mechanism_network = mechanism_network,
            policy_network    = policy_network,
            profit_goal       = pg_mid,
            cfg               = cfg,
            is_discrete       = is_discrete,
            seed              = trial_seed,
            verbose           = verbose,
        )

        if verbose:
            print(f"  [binary_search] Evaluating {len(checkpoints)} checkpoints ...")
        eval_results = evaluate_checkpoint_list(
            auction           = auction,
            mechanism_network = mechanism_network,
            policy_network    = policy_network,
            checkpoints       = checkpoints,
            profit_goal       = pg_mid,
            cfg               = cfg,
            is_discrete       = is_discrete,
            seed              = trial_seed,
            verbose           = verbose,
        )

        feasible_ckpts = [r for r in eval_results if r["feasible"]]
        any_feasible   = len(feasible_ckpts) > 0

        best_pkl     = None
        best_rev     = None
        best_reg     = None
        best_surplus = None
        if any_feasible:
            best_ckpt    = feasible_ckpts[0]
            best_rev     = best_ckpt["revenue_eval"]
            best_reg     = best_ckpt["regret_eval"]
            best_surplus = best_ckpt.get("buyer_surplus_eval")
            best_pkl     = os.path.join(save_dir, f"best_mech_pg{pg_mid:.3f}.pkl")
            with open(best_pkl, "wb") as f_pkl:
                pickle.dump({
                    "profit_goal":        pg_mid,
                    "mechanism_params":   best_ckpt["mechanism_params"],
                    "policy_params":      best_ckpt["policy_params"],
                    "epoch":              best_ckpt["epoch"],
                    "revenue_eval":       best_rev,
                    "regret_eval":        best_reg,
                    "buyer_surplus_eval": best_surplus,
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
                      f"IC={eval_results[0]['regret_eval']:.4f}")
            pg_hi = pg_mid

        _curves_path = os.path.join(save_dir, f"training_curves_pg{pg_mid:.3f}.json")
        _curves = {
            "profit_goal": pg_mid,
            "reg":         [float(x) for x in logs.get("reg",      [])],
            "ir":          [float(x) for x in logs.get("ir",       [])],
            "gap":         [float(x) for x in logs.get("gap",      [])],
            "reg_plus":    [float(x) for x in logs.get("reg_plus", [])],
            "gap_plus":    [float(x) for x in logs.get("gap_plus", [])],
            "loss":        [float(x) for x in logs.get("loss",     [])],
            "checkpoints_train": [
                {"epoch": c["epoch"], "reg_train": c["reg_train"],
                 "ir_train": c["ir_train"], "gap_train": c["gap_train"]}
                for c in checkpoints
            ],
            "checkpoints_eval": [
                {"epoch": r["epoch"], "revenue_eval": r["revenue_eval"],
                 "regret_reopt": r.get("regret_reopt"), "regret_eval": r["regret_eval"],
                 "ir_loss_eval": r["ir_loss_eval"],
                 "buyer_surplus_eval": r.get("buyer_surplus_eval"),
                 "feasible": r["feasible"]}
                for r in eval_results
            ],
        }
        with open(_curves_path, "w") as f:
            json.dump(_curves, f, indent=2)

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

        with open(results_path, "w") as f:
            json.dump(all_trials, f, indent=2)

        trial_seed += 1

    elapsed_total = (time.time() - t0_global) / 3600.0
    summary = {
        "experiment":       experiment,
        "pg_lo":            pg_lo,
        "pg_hi":            pg_hi,
        "pg_tol":           pg_tol,
        "converged":        (pg_hi - pg_lo) <= pg_tol,
        "num_trials":       iteration,
        "total_time_h":     elapsed_total,
        "best_mechanism":   best_mech_info,
        "cfg":              {k: v for k, v in cfg.items() if not callable(v)},
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
