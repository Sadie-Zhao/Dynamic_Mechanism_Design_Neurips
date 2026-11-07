"""
retrain_discrete_single.py — retrain the discrete single-good mechanism with the FIXED
pathwise deviation oracle (proto_strength lowered so the Gumbel-softmax gradient is not
saturated). Per instruction, we DIRECTLY train at fixed profit goals = {reported profit,
benchmark}, rather than a binary search.

For each goal we run the paper's own pipeline (train_with_checkpoints + evaluate_checkpoint_list)
with proto=3 threaded into BOTH the training inner oracle and the post-training reopt selection.
The chosen mechanism is saved for the downstream exact-BR (memoryless / non-clairvoyant /
clairvoyant) and NPG audits.

Config note: matches the Drive run that produced the REPORTED mechanism (alpha_base=5, beta=5,
seed=42, lr_w=5e-3, 2000 epochs, 500 inner steps) — NOT the degenerate local seed-0 smoke test.

Blocker fixes applied vs. the repo pipeline:
  - skip_thresh=inf  : never skip the fixed-oracle reopt (we WANT every checkpoint evaluated).
  - NaN-safe select  : pick best non-skipped checkpoint by revenue (NaN -> -inf); never a skipped one.
  - allow_nan=False  : summaries never contain bare NaN tokens.

Usage:
  python retrain_discrete_single.py --goals 7.617 7.562 --proto 3.0 --out_dir mechanisms/discrete/ds_fixed
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, math, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
import jax
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.mechanisms.network import build_mechanism_network
from bandit_auction.core import build_discrete_policy_net
from bandit_auction.train.binary_search import train_with_checkpoints, evaluate_checkpoint_list, DEFAULT_BS_CFG
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
from belief_policy import (build_discrete_belief_policy_net, build_discrete_infostate_policy_net,
                           build_discrete_m15_policy_net)
import faithful_patch


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _fin(v):  # NaN/inf -> None for strict JSON
    return v if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else (None if v is None or (isinstance(v,float) and not math.isfinite(v)) else v)
def _rev(r):
    v = r.get("revenue_eval")
    return v if (isinstance(v, (int, float)) and v == v) else -float("inf")   # NaN -> worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="mechanisms/discrete/ds_fixed")
    ap.add_argument("--goals", type=float, nargs="+", default=[7.617, 7.562])  # reported, benchmark
    ap.add_argument("--proto", type=float, default=3.0)
    ap.add_argument("--policy", default="belief", choices=["belief", "memoryless", "infostate", "m15"])
    ap.add_argument("--env", default="single", choices=["single", "multi"])   # discrete single- or multi-good
    ap.add_argument("--seed", type=int, default=42)          # matches Drive/reported run
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--inner_steps", type=int, default=500)
    ap.add_argument("--n_inner_restarts", type=int, default=3)
    ap.add_argument("--lr_theta", type=float, default=1e-4)
    ap.add_argument("--lr_w", type=float, default=5e-3)
    ap.add_argument("--alpha_base", type=float, default=5.0)
    ap.add_argument("--beta_weight", type=float, default=5.0)
    ap.add_argument("--num_samples_train", type=int, default=64)
    ap.add_argument("--faithful_belief", type=int, default=1)  # 1 = use faithful (opponent-informative) filter
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.faithful_belief:
        faithful_patch.apply()
    log(f"jax {jax.__version__} devices={jax.devices()}  proto={args.proto} goals={args.goals} "
        f"faithful_belief={bool(args.faithful_belief)}")

    auction = (create_discrete_multi_auction(seed=args.seed) if args.env == "multi"
               else create_discrete_single_auction(seed=args.seed))
    log(f"env={args.env}  n_goods={auction.num_goods} K_per_buyer={auction.buyer_types.shape[1]} "
        f"num_type_profiles={auction.num_type_profiles}")
    if args.env == "multi" and abs(float(args.goals[0]) - 7.617) < 1e-6:
        log(f"WARNING: --env multi but --goals left at single-good default {args.goals} (multi goal ~15.25)")
    mech_net = build_mechanism_network()
    if args.policy == "infostate":
        policy_net = build_discrete_infostate_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,
                                                         hidden=128, proto_strength_init=args.proto)
    elif args.policy == "m15":     # A' oracle: belief + projected public history, NO own-report history
        policy_net = build_discrete_m15_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,
                                                   hidden=128, proto_strength_init=args.proto)
    elif args.policy == "belief":
        policy_net = build_discrete_belief_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,
                                                      hidden=128, proto_strength_init=args.proto)
    else:
        policy_net = build_discrete_policy_net(d_priv=auction.d_priv, d_pub=auction.d_pub,
                                               hidden=128, proto_strength_init=args.proto)
    log(f"policy class = {args.policy} (belief-conditioned fix + proto={args.proto})")

    index = []
    for goal in args.goals:
        gdir = os.path.join(args.out_dir, f"goal_{goal:.3f}")
        os.makedirs(gdir, exist_ok=True)
        cfg = {**DEFAULT_BS_CFG,
               "lr_theta": args.lr_theta, "lr_w": args.lr_w,
               "alpha_base": args.alpha_base, "beta_weight": args.beta_weight,
               "num_inner_steps": args.inner_steps, "num_outer_epochs": args.epochs,
               "num_samples_train": args.num_samples_train, "n_inner_restarts": args.n_inner_restarts,
               "pg_base": goal,                 # alpha = alpha_base * (goal/pg_base) = alpha_base
               "checkpoint_thresh": 0.15, "checkpoint_interval": 50,
               "ic_thresh": 0.05, "gap_thresh": 0.1,
               "ic_reopt_steps": 600, "ic_reopt_restarts": 3, "ic_reopt_lr": 5e-4, "ic_reopt_samples": 128,
               "eval_num_keys": 32, "eval_num_samples": 256, "print_iter": 100}
        log(f"=== GOAL {goal:.3f}: train with fixed oracle ===")
        checkpoints, logs = train_with_checkpoints(
            auction, mech_net, policy_net, profit_goal=goal, cfg=cfg,
            is_discrete=True, seed=args.seed, verbose=True)
        log(f"=== GOAL {goal:.3f}: evaluate all checkpoints (fixed reopt, skip disabled) ===")
        results = evaluate_checkpoint_list(
            auction, mech_net, policy_net, checkpoints, profit_goal=goal, cfg=cfg,
            is_discrete=True, seed=args.seed, verbose=True,
            skip_thresh=float("inf"))            # BLOCKER 3 fix: evaluate every checkpoint
        # SELECTION: maximize profit subject to per-buyer DIC <= 0.05 AND DIR <= 0.05.
        # DIC = regret_reopt (pathwise reopt) is a SUM over buyers -> per-buyer = /num_buyers.
        # DIR = ir_loss_eval. NaN-safe: treat missing as +inf (fails the constraint).
        # SELECTION: highest profit s.t. PER-BUYER DIC <= 0.1 AND PER-BUYER DIR <= 0.1.
        # DIC = regret_reopt (SUM over buyers) -> per-buyer /Bn. DIR (ex-ante, per spec) = max(0,-U_truth);
        # truthful utility is > 0 here so DIR = 0 (<= 0.1 always) -> DIC is the binding constraint. The
        # pipeline ir_loss_eval (interim/per-type IR, also a sum) is reported PER-BUYER as a DIAGNOSTIC.
        Bn = auction.num_buyers
        SEL_TOL = 0.1
        def _dic_pb(r):
            v = r.get("regret_reopt")
            return (v / Bn) if (isinstance(v, (int, float)) and v == v) else float("inf")
        def _irloss_pb(r):
            v = r.get("ir_loss_eval")
            return (v / Bn) if (isinstance(v, (int, float)) and v == v) else float("inf")
        pool = sorted([r for r in results if not r.get("skipped")], key=_rev, reverse=True)
        # ENFORCE both constraints per spec: per-buyer DIC <= tol AND per-buyer DIR <= tol.
        # DIR proxy = ir_loss_eval/Bn (ex-post E[max(0,-U_i)] per buyer; a conservative over-approx of the
        # ex-ante max(0,-E[U_i])). Do NOT assume DIR=0 -- relu payments are unbounded so U_truth can go < 0.
        passing = [r for r in pool if _dic_pb(r) <= SEL_TOL and _irloss_pb(r) <= SEL_TOL]
        chosen = passing[0] if passing else (pool[0] if pool else results[0])   # highest-profit passer
        feasible = passing
        log(f"selection: {len(passing)}/{len(pool)} ckpts pass per-buyer DIC<={SEL_TOL} AND DIR<={SEL_TOL}; "
            f"chosen epoch={chosen['epoch']} rev={chosen.get('revenue_eval')} "
            f"DIC_pb={_dic_pb(chosen):+.4f} irloss_pb(diag)={_irloss_pb(chosen):.4f}")
        log(f"GOAL {goal:.3f} chosen: epoch={chosen['epoch']} feasible={chosen.get('feasible')} "
            f"rev={chosen.get('revenue_eval')} reopt={chosen.get('regret_reopt')} "
            f"eval={chosen.get('regret_eval')} ir={chosen.get('ir_loss_eval')}")
        with open(os.path.join(gdir, "mechanism_params.pkl"), "wb") as f:
            pickle.dump(chosen["mechanism_params"], f)
        with open(os.path.join(gdir, "policy_params.pkl"), "wb") as f:
            pickle.dump(chosen["policy_params"], f)
        summ = {"experiment": f"discrete_{args.env}", "method": f"acceptance_fixed_oracle_{args.policy}",
                "proto_strength": args.proto, "profit_goal": goal, "seed": args.seed,
                "best_epoch": int(chosen["epoch"]), "feasible": bool(chosen.get("feasible")),
                "revenue_eval": _fin(chosen.get("revenue_eval")),
                "regret_reopt": _fin(chosen.get("regret_reopt")),
                "regret_eval": _fin(chosen.get("regret_eval")),
                "ir_loss_eval": _fin(chosen.get("ir_loss_eval")),
                "num_feasible": len(feasible), "num_candidates": len(results),
                "cfg": {k: _fin(v) for k, v in cfg.items()}}
        with open(os.path.join(gdir, "summary.json"), "w") as f:
            json.dump(summ, f, indent=2, allow_nan=False)
        index.append({"goal": goal, "dir": gdir, "revenue_eval": summ["revenue_eval"],
                      "regret_reopt": summ["regret_reopt"], "regret_eval": summ["regret_eval"],
                      "feasible": summ["feasible"]})
        log(f"saved {gdir}")
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2, allow_nan=False)
    log("ALL GOALS DONE")
    for e in index:
        log(f"  goal={e['goal']:.3f} rev={e['revenue_eval']} reopt={e['regret_reopt']} "
            f"eval={e['regret_eval']} feasible={e['feasible']}")


if __name__ == "__main__":
    main()
