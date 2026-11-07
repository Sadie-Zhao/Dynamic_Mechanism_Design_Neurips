"""
pathwise_audit.py — standalone PATHWISE deviation-oracle audit (the paper's method: truthful-prototype
+ straight-through Gumbel-softmax head, optimized by Adam) against a FIXED mechanism. Records the MAX
regret over optimization steps (not just the final step), across restarts, then reports the HELD-OUT
regret of the best policy. Uses the FAITHFUL belief filter (patched). Reports PER BUYER (the raw
cumulative-regret metric sums over buyers, so we divide by num_buyers to match exact-BR / NPG).

--head belief   -> M1 oracle (own type + belief)
--head infostate-> M2 oracle (own type + belief + own report history + cumulative allocation)

Usage: python pathwise_audit.py --run_dir <ckpt> --head {belief,infostate} --out results/pathwise/<name>.json
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
import jax, jax.numpy as jnp, numpy as np, optax, haiku as hk
import faithful_patch; faithful_patch.apply()                     # faithful (report-free) belief
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network, build_mechanism_network_attn
from bandit_auction.core import DiscretePolicy, AuctionSimulator
from belief_policy import build_discrete_belief_policy_net, build_discrete_infostate_policy_net


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="belief", choices=["belief", "m15", "infostate", "m2cum"])  # M1 / M1.5 / M2 / M2-cumulative
    ap.add_argument("--proto", type=float, default=3.0)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--eval_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", default="single", choices=["single", "multi"])
    ap.add_argument("--arch", default="default", choices=["default", "wide", "attn", "gru"])  # Axis-III
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"jax {jax.__version__} head={args.head}")

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summ = json.load(f)
    cfg = summ.get("cfg", {}) or {}
    ekw = dict(seed=args.seed)
    for k in ("n_buyers", "n_goods", "horizon", "discount", "allocation_cost", "type_values", "type_values_per_good"):
        if k in cfg: ekw[k] = cfg[k]
    auction = (create_discrete_multi_auction(**ekw) if args.env == "multi"
               else create_discrete_single_auction(**ekw))
    B = auction.num_buyers
    mech_net = build_mechanism_network_attn() if cfg.get("use_attn_mechanism") else build_mechanism_network()
    with open(os.path.join(args.run_dir, "mechanism_params.pkl"), "rb") as f:
        mech_params = pickle.load(f)
    mechanism = Mechanism(mech_net, mech_params)
    if args.head in ("infostate", "m15", "m2cum"):
        # M2 (infostate, use_own_reports=True) vs M1.5 (m15, use_own_reports=False = drop own-report history)
        # vs M2-cum (m2cum: full history but CUMULATIVE allocation -> matches exactBR-M2 info-set exactly).
        m2_arch = "flatten" if args.arch == "default" else args.arch          # {flatten,wide,attn,gru}
        policy_net = build_discrete_infostate_policy_net(auction.d_priv, auction.d_pub, 128, args.proto,
                                                         arch=m2_arch, use_own_reports=(args.head != "m15"),
                                                         cumulative_alloc=(args.head == "m2cum"))
        arch_used = m2_arch + ("+cum" if args.head == "m2cum" else "")
    else:
        if args.arch in ("attn", "gru"):
            raise SystemExit("M1 belief head has no history sequence; use --arch default|wide")
        m1_arch = "base" if args.arch == "default" else "wide"               # {base,wide}
        policy_net = build_discrete_belief_policy_net(auction.d_priv, auction.d_pub, 128, args.proto, arch=m1_arch)
        arch_used = m1_arch
    log(f"head={args.head} arch={arch_used}")

    ubf = auction.get_next_belief_state_discrete            # patched -> faithful
    w_opt = optax.adam(args.lr)
    prng = hk.PRNGSequence(args.seed)

    def cumreg_train(pp, key):     # num_samples baked in as a concrete Python int (must be static)
        pol = DiscretePolicy(policy_net, pp)
        return AuctionSimulator.estimate_cumulative_regret(auction, mechanism, pol, args.samples, key, ubf)
    def cumreg_eval(pp, key):
        pol = DiscretePolicy(policy_net, pp)
        return AuctionSimulator.estimate_cumulative_regret(auction, mechanism, pol, args.eval_samples, key, ubf)
    grad_fn = jax.jit(jax.value_and_grad(cumreg_train))

    best_ho = -1e9
    restarts = []
    for r in range(args.restarts):
        s0 = auction.get_initial_state(next(prng)); b0 = auction.get_initial_belief_state(s0)
        pp = policy_net.init(next(prng), b0, auction); ws = w_opt.init(pp)
        rmax, pmax = -1e9, pp
        for _ in range(args.steps):
            reg, grad = grad_fn(pp, next(prng))
            if float(reg) > rmax:                          # MAX regret over steps; pmax = the params that
                rmax = float(reg); pmax = pp               # PRODUCED reg (capture BEFORE the update)
            upd, ws = w_opt.update(jax.tree_util.tree_map(lambda g: -g, grad), ws, pp)
            pp = optax.apply_updates(pp, upd)
        ho = float(cumreg_eval(pmax, next(prng))) / B    # held-out, PER BUYER
        restarts.append({"restart": r, "train_max_sum": rmax, "heldout_per_buyer": ho})
        best_ho = max(best_ho, ho)
        log(f"restart {r}: train_max(sum)={rmax:+.5f} heldout(per-buyer)={ho:+.5f}")

    out = {"experiment": f"discrete_{args.env}",
           "oracle": f"pathwise-{ {'infostate':'M2','m15':'M1.5','belief':'M1','m2cum':'M2cum'}[args.head] }-{arch_used}", "head": args.head,
           "arch": arch_used,
           "run_dir": args.run_dir, "restarts": args.restarts, "steps": args.steps, "proto": args.proto,
           "best_heldout_regret_per_buyer": best_ho, "per_restart": restarts,
           "note": "pathwise Gumbel-straight-through + Adam reopt; MAX regret over steps; held-out; PER BUYER"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE pathwise-{args.head} best held-out regret/buyer = {best_ho:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
