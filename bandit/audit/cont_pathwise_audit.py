"""
cont_pathwise_audit.py — PATHWISE (reparameterization / BPTT) continuous deviation oracle.

The bounded-logit report is a smooth function of the head params, so estimate_regret_det is fully
differentiable; we ascend it with Adam. Records the MAX regret over optimisation steps (params captured
BEFORE the update that produced the score, matching the discrete fix), then reports the HELD-OUT regret
of that argmax policy, PER BUYER (estimate_regret_det returns the buyer-SUM, so /B). Supports the info
and arch axes via cont_oracle. No Gumbel (continuous reparam); no faithful_patch (continuous belief is
report-free by construction).

Usage: python cont_pathwise_audit.py --run_dir <ckpt> --env {single,multi} --info {m1,m2}
                                     --arch {gru,wide,attn,mlp} --out results/cont/<name>.json
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
import jax, jax.numpy as jnp, numpy as np, optax
from bandit_auction.envs.cont_single import create_cont_single_auction
from bandit_auction.envs.cont_multi import create_cont_multi_auction
from bandit_auction.mechanisms.network import (Mechanism, build_mechanism_network,
    build_mechanism_network_attn)
from cont_oracle import build_cont_oracle_net
from cont_rollout import estimate_regret_det
from cont_cfg import load_cont_env, EVAL_SEED


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="single", choices=["single", "multi"])
    ap.add_argument("--info", default="m1", choices=["m1", "m2"])
    ap.add_argument("--arch", default="gru", choices=["gru", "wide", "attn", "mlp"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--eval_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"jax {jax.__version__} devices={jax.devices()}  env={args.env} info={args.info} arch={args.arch}")

    ekw, use_attn, cfg, missing = load_cont_env(args.run_dir, args.seed)
    if missing:
        log(f"WARNING: env keys {missing} not in config.yaml/summary -> constructor DEFAULTS for those "
            f"(env may be WRONG; ensure run_dir has config.yaml).")
    auction = (create_cont_multi_auction(**ekw) if args.env == "multi"
               else create_cont_single_auction(**ekw))
    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    dpr, dpu = auction.d_priv, auction.d_pub
    mech_net = build_mechanism_network_attn() if use_attn else build_mechanism_network()
    with open(os.path.join(args.run_dir, "mechanism_params.pkl"), "rb") as f:
        mech_params = pickle.load(f)
    mechanism = Mechanism(mech_net, mech_params)
    log(f"resolved env kwargs: {ekw}  use_attn={use_attn}")

    net = build_cont_oracle_net(dpr, dpu, G, T, B, args.hidden, args.info, args.arch)
    pay0 = jnp.zeros((T, B), jnp.float32)

    def cumreg_train(params, key):   # samples baked in as a static Python int
        return estimate_regret_det(auction, mechanism, net.apply, params, args.samples, key)
    def cumreg_eval(params, key):
        return estimate_regret_det(auction, mechanism, net.apply, params, args.eval_samples, key)
    grad_fn = jax.jit(jax.value_and_grad(cumreg_train))
    opt = optax.adam(args.lr)

    key = jax.random.PRNGKey(args.seed)
    best_ho, per_restart = -1e9, []
    for r in range(args.restarts):
        key, ki, ks = jax.random.split(key, 3)
        s0 = auction.get_initial_state(ki); bs0 = auction.get_initial_belief_state(s0)
        pp = net.init(ks, bs0, auction, None, pay0); ws = opt.init(pp)
        rmax, pmax = -1e9, pp
        for _ in range(args.steps):
            key, kk = jax.random.split(key)
            reg, grad = grad_fn(pp, kk)
            if float(reg) > rmax:                      # MAX over steps; pmax = params that PRODUCED reg
                rmax = float(reg); pmax = pp           # (captured BEFORE the update)
            upd, ws = opt.update(jax.tree_util.tree_map(lambda g: -g, grad), ws, pp)
            pp = optax.apply_updates(pp, upd)
        ho = float(cumreg_eval(pmax, jax.random.PRNGKey(EVAL_SEED))) / B   # held-out PER BUYER; FIXED eval seed (paired across oracles)
        per_restart.append({"restart": r, "train_max_sum": rmax, "heldout_per_buyer": ho})
        best_ho = max(best_ho, ho)
        log(f"restart {r}: train_max(sum)={rmax:+.5f} heldout(per-buyer)={ho:+.5f}")

    out = {"experiment": f"cont_{args.env}", "oracle": f"pathwise-{args.info}-{args.arch}",
           "info": args.info, "arch": args.arch, "run_dir": args.run_dir,
           "restarts": args.restarts, "steps": args.steps,
           "best_heldout_regret_per_buyer": best_ho, "per_restart": per_restart,
           "note": "continuous pathwise (reparam BPTT) + Adam; MAX regret over steps; held-out; PER BUYER"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE pathwise-{args.info}-{args.arch} best held-out regret/buyer = {best_ho:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
