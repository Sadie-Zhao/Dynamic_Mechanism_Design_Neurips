"""
cont_es_oracle.py — CONTINUOUS gradient-free deviation oracle: OpenAI-ES (Salimans et al. 2017).

The landscape-robustness rung of the algorithm axis: it optimizes the SAME full deterministic policy
that pathwise/NPG use, but with NO gradients — so if it finds more regret than the first-order methods,
they were stuck in the optimization landscape; if it agrees, it corroborates them.

Each generation (mirrored / antithetic sampling, common random numbers across the population):
  eps_k ~ N(0, I_P), k=1..K;  evaluate deterministic per-buyer regret f of theta ± sigma*eps_k
  (SAME fitness key for all 2K candidates -> low-variance comparison);
  centered-rank fitness shaping u in [-0.5, 0.5] over the 2K fitnesses (higher regret -> higher u);
  ES gradient (ascent)  g = (1/(2K*sigma)) * sum_m u_m * signed_eps_m,  signed = concat(eps, -eps);
  Adam step on theta with -g (optax minimizes, so -g ascends regret).
theta (the mean) IS the deterministic policy; we track the best theta by held-out regret and report it,
so the number is comparable to pathwise/NPG (per buyer, via estimate_regret_det /B).
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
import jax, jax.numpy as jnp, numpy as np, optax
from jax.flatten_util import ravel_pytree
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
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pophalf", type=int, default=64)       # K antithetic pairs -> 2K candidates/gen
    ap.add_argument("--sigma", type=float, default=0.05)     # perturbation std in param space
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--samples", type=int, default=128)      # fitness eval samples (noisy ok)
    ap.add_argument("--eval_samples", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=15)
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
    k0 = jax.random.PRNGKey(args.seed)
    s0 = auction.get_initial_state(k0); bs0 = auction.get_initial_belief_state(s0)
    flat_dim = ravel_pytree(net.init(k0, bs0, auction, None, pay0))[0].shape[0]
    _, unflatten = ravel_pytree(net.init(k0, bs0, auction, None, pay0))
    log(f"policy params P={flat_dim}  popsize=2x{args.pophalf}  sigma={args.sigma}")

    def regret_of(flat, key, n):
        return estimate_regret_det(auction, mechanism, net.apply, unflatten(flat), n, key)  # buyer-SUM

    @jax.jit
    def es_grad(theta, key):
        k_eps, k_fit = jax.random.split(key)
        eps = jax.random.normal(k_eps, (args.pophalf, flat_dim))                      # (K, P)
        cands = jnp.concatenate([theta + args.sigma * eps, theta - args.sigma * eps], 0)  # (2K, P)
        fits = jax.vmap(lambda c: regret_of(c, k_fit, args.samples))(cands)            # (2K,) CRN fitness
        ranks = jnp.argsort(jnp.argsort(fits)).astype(jnp.float32)                     # 0..2K-1 by fitness
        u = ranks / (2 * args.pophalf - 1) - 0.5                                       # centered ranks, higher regret -> higher
        eps_all = jnp.concatenate([eps, -eps], 0)                                      # (2K, P)
        g = (u @ eps_all) / (2 * args.pophalf * args.sigma)                            # (P,) ascent direction
        return g

    key = jax.random.PRNGKey(args.seed)
    best_ho, per_restart = -1e9, []
    for r in range(args.restarts):
        key, ki, ks = jax.random.split(key, 3)
        theta = ravel_pytree(net.init(ks, bs0, auction, None, pay0))[0]
        opt = optax.adam(args.lr); ostate = opt.init(theta)
        # seed best with an eval of the initial theta (handles gens < eval_every; theta_best always evaluated)
        theta_best = theta
        r_best = float(regret_of(theta, jax.random.PRNGKey(EVAL_SEED), args.samples * 8)) / B
        for gen in range(args.gens):
            key, kg = jax.random.split(key)
            g = es_grad(theta, kg)
            upd, ostate = opt.update(-g, ostate, theta)                                # -g: optax minimizes -> ascend regret
            theta = optax.apply_updates(theta, upd)
            if (gen + 1) % args.eval_every == 0:
                rmid = float(regret_of(theta, jax.random.PRNGKey(EVAL_SEED), args.samples * 8)) / B  # FIXED key -> CRN across gens
                if rmid > r_best:
                    r_best, theta_best = rmid, theta
        ho = float(regret_of(theta_best, jax.random.PRNGKey(EVAL_SEED), args.eval_samples)) / B      # held-out per-buyer, FIXED seed
        per_restart.append({"restart": r, "heldout_per_buyer": ho})
        best_ho = max(best_ho, ho)
        log(f"restart {r}: ES held-out regret/buyer = {ho:+.5f}")

    out = {"experiment": f"cont_{args.env}", "oracle": f"ES-{args.info}-{args.arch}",
           "info": args.info, "arch": args.arch, "run_dir": args.run_dir,
           "sigma": args.sigma, "popsize": 2 * args.pophalf, "gens": args.gens,
           "best_regret_per_buyer": best_ho, "per_restart": per_restart,
           "config": {k: v for k, v in vars(args).items()},
           "note": "continuous OpenAI-ES (gradient-free, antithetic + centered-rank shaping); PER BUYER"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE ES-{args.info}-{args.arch} best held-out regret/buyer = {best_ho:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
