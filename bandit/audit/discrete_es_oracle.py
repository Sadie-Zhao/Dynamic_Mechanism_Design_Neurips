"""
discrete_es_oracle.py — OpenAI-ES gradient-free deviation oracle for the DISCRETE settings.

The gradient-free rung of the algorithm axis where exactBR is intractable (discrete MULTI-good). Optimizes
the discrete deviation-policy head (M1 belief head / M2 infostate head) by BLACK-BOX return — no gradients —
so it is an independent check that the first-order oracles (pathwise, NPG) are not jointly stuck in the
optimization landscape. Mirrors cont_es_oracle.py (antithetic sampling + centered-rank shaping + Adam);
only the env / policy / regret-eval are discrete.

Fitness = AuctionSimulator.estimate_cumulative_regret with the FAITHFUL (report-based == report-free for
injective mechanisms) discrete belief filter, per buyer (÷ num_buyers). The head's straight-through forward
is the argmax report, so the evaluated policy is the deterministic misreport.

Usage: python discrete_es_oracle.py --run_dir <ckpt> --env {single,multi} --info {m1,m2}
                                    --arch {base,wide,attn,gru} --out results/.../es_<info>_<arch>.json
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
import jax, jax.numpy as jnp, numpy as np, optax
from jax.flatten_util import ravel_pytree
import faithful_patch; faithful_patch.apply()                      # report-based (=report-free for injective) belief
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.mechanisms.network import (Mechanism, build_mechanism_network,
    build_mechanism_network_attn)
from bandit_auction.core import DiscretePolicy, AuctionSimulator
from belief_policy import build_discrete_belief_policy_net, build_discrete_infostate_policy_net
from cont_cfg import EVAL_SEED                                     # shared fixed held-out seed (paired across oracles)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="single", choices=["single", "multi"])
    ap.add_argument("--info", default="m1", choices=["m1", "m15", "m2"])
    ap.add_argument("--arch", default="base", choices=["base", "wide", "attn", "gru"])
    ap.add_argument("--proto", type=float, default=3.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pophalf", type=int, default=64)            # K antithetic pairs -> 2K candidates/gen
    ap.add_argument("--sigma", type=float, default=0.05)         # perturbation std in param space
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--samples", type=int, default=128)          # fitness eval samples (ranking under CRN; noise ok)
    ap.add_argument("--eval_samples", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"jax {jax.__version__} devices={jax.devices()}  env={args.env} info={args.info} arch={args.arch}")

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summ = json.load(f)
    cfg = summ.get("cfg", {}) or {}
    ekw = dict(seed=args.seed)
    for k in ("n_buyers", "n_goods", "horizon", "discount", "allocation_cost", "type_values", "type_values_per_good"):
        if k in cfg: ekw[k] = cfg[k]
    auction = (create_discrete_multi_auction(**ekw) if args.env == "multi"
               else create_discrete_single_auction(**ekw))
    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    dpr, dpu = auction.d_priv, auction.d_pub
    mech_net = build_mechanism_network_attn() if cfg.get("use_attn_mechanism") else build_mechanism_network()
    with open(os.path.join(args.run_dir, "mechanism_params.pkl"), "rb") as f:
        mech_params = pickle.load(f)
    mechanism = Mechanism(mech_net, mech_params)
    ubf = auction.get_next_belief_state_discrete                  # patched -> faithful
    log(f"resolved env kwargs: {ekw}")

    # ---- discrete policy net (M1 belief head / M2 infostate head), arch mapped to head's vocabulary ----
    if args.info in ("m2", "m15"):
        arch_h = "flatten" if args.arch == "base" else args.arch   # infostate head: {flatten,wide,attn,gru}
        net = build_discrete_infostate_policy_net(dpr, dpu, args.hidden, args.proto, arch=arch_h,
                                                  use_own_reports=(args.info == "m2"))  # m15 => drop own reports
    else:  # m1
        if args.arch in ("attn", "gru"):
            raise SystemExit("M1 belief head has no history sequence; use --arch base|wide")
        arch_h = "base" if args.arch == "base" else "wide"
        net = build_discrete_belief_policy_net(dpr, dpu, args.hidden, args.proto, arch=arch_h)

    k0 = jax.random.PRNGKey(args.seed)
    s0 = auction.get_initial_state(k0); bs0 = auction.get_initial_belief_state(s0)
    flat_dim = ravel_pytree(net.init(k0, bs0, auction))[0].shape[0]
    _, unflatten = ravel_pytree(net.init(k0, bs0, auction))
    log(f"policy P={flat_dim}  popsize=2x{args.pophalf}  sigma={args.sigma}")

    def regret_of(flat, key, n):     # buyer-SUMMED regret of the deterministic discrete policy
        pol = DiscretePolicy(net, unflatten(flat))
        return AuctionSimulator.estimate_cumulative_regret(auction, mechanism, pol, n, key, ubf)

    @jax.jit
    def es_grad(theta, key):
        k_eps, k_fit = jax.random.split(key)
        eps = jax.random.normal(k_eps, (args.pophalf, flat_dim))                     # (K, P)
        cands = jnp.concatenate([theta + args.sigma * eps, theta - args.sigma * eps], 0)
        fits = jax.vmap(lambda c: regret_of(c, k_fit, args.samples))(cands)          # (2K,) CRN fitness
        ranks = jnp.argsort(jnp.argsort(fits)).astype(jnp.float32)                   # 0..2K-1 by fitness
        u = ranks / (2 * args.pophalf - 1) - 0.5                                     # centered ranks
        eps_all = jnp.concatenate([eps, -eps], 0)
        g = (u @ eps_all) / (2 * args.pophalf * args.sigma)                          # (P,) ascent direction
        return g

    key = jax.random.PRNGKey(args.seed)
    best_ho, per_restart = -1e9, []
    for r in range(args.restarts):
        key, ks = jax.random.split(key)
        theta = ravel_pytree(net.init(ks, bs0, auction))[0]
        opt = optax.adam(args.lr); ostate = opt.init(theta)
        theta_best = theta
        r_best = float(regret_of(theta, jax.random.PRNGKey(EVAL_SEED), args.samples * 8)) / B  # seed with init (truthful)
        for gen in range(args.gens):
            key, kg = jax.random.split(key)
            g = es_grad(theta, kg)
            upd, ostate = opt.update(-g, ostate, theta)                             # -g: optax minimizes -> ascend regret
            theta = optax.apply_updates(theta, upd)
            if (gen + 1) % args.eval_every == 0:
                rmid = float(regret_of(theta, jax.random.PRNGKey(EVAL_SEED), args.samples * 8)) / B  # CRN mid-eval
                if rmid > r_best:
                    r_best, theta_best = rmid, theta
        ho = float(regret_of(theta_best, jax.random.PRNGKey(EVAL_SEED), args.eval_samples)) / B      # held-out per buyer
        per_restart.append({"restart": r, "heldout_per_buyer": ho})
        best_ho = max(best_ho, ho)
        log(f"restart {r}: ES held-out regret/buyer = {ho:+.5f}")

    out = {"experiment": f"discrete_{args.env}", "oracle": f"ES-{args.info}-{args.arch}",
           "info": args.info, "arch": args.arch, "run_dir": args.run_dir,
           "sigma": args.sigma, "popsize": 2 * args.pophalf, "gens": args.gens,
           "best_regret_per_buyer": best_ho, "per_restart": per_restart,
           "config": {k: v for k, v in vars(args).items()},
           "note": "discrete OpenAI-ES (gradient-free, antithetic + centered-rank); faithful belief; PER BUYER"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE ES-{args.info}-{args.arch} best held-out regret/buyer = {best_ho:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
