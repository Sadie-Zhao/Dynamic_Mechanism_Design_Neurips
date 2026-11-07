"""
belief_npg_oracle.py — BELIEF-CONDITIONED stochastic Natural Policy Gradient deviation oracle.

Faithful counterpart to npg_oracle.py (which is memoryless). The policy conditions on the SAME
information the paper's belief head sees: (own current type, belief_state.beliefs[d]). The belief is
threaded through the environment's OWN Bayes filter (auction.get_next_belief_state_discrete), so the
oracle's information set is identical to the retrained DiscreteBeliefHead's.

Policy: pi(a | feat) = softmax( MLP_theta(feat) ),  feat = [one_hot(own_type, K) , beliefs[d] (K^2)].
Estimator: score-function REINFORCE (return-to-go + per-step baseline), per-step EMPIRICAL Fisher,
regularized natural step  theta += eta (F+lam I)^{-1} g. Immune to the discrete-transition
non-differentiability (no pathwise gradient), so — like npg_oracle — it isolates the INFORMATION
CLASS, not the estimator.

Purpose: pin the belief-conditioned exploitability of a FIXED mechanism, completing the ladder
  memoryless (+0.244)  <=  belief-conditioned (THIS)  <=  opponent-observed non-clairvoyant (+0.448).
NOTE (see characterize_belief.py): the discrete belief is OPPONENT-UNINFORMATIVE (types are
independent; the filter ignores the opponent), so belief-conditioning adds only a summary of the
deviator's OWN history. We therefore expect this to land at/above +0.244 and strictly below +0.448.

Usage: python belief_npg_oracle.py --run_dir <ckpt> --out results/npg/<name>_belief.json
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
import jax, jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network, build_mechanism_network_attn
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
from belief_faithful import make_faithful_discrete_belief, faithful_init_belief


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deviator", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--eval_episodes", type=int, default=20000)
    ap.add_argument("--belief", default="faithful", choices=["faithful", "shipped"])
    ap.add_argument("--input", default="belief", choices=["belief", "m15", "infostate", "m2cum"])  # M1 / M1.5 / M2 / M2-cumulative(exactBR-aligned)
    ap.add_argument("--arch", default="none", choices=["none", "flatten", "wide", "attn", "gru"])  # none=flat-MLP policy; else reuse DiscreteInfoStateHead at that arch (score-function NPG)
    ap.add_argument("--env", default="single", choices=["single", "multi"])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"jax {jax.__version__} devices={jax.devices()}")

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summ = json.load(f)
    cfg = summ.get("cfg", {}) or {}
    ekw = dict(seed=args.seed)
    for k in ("n_buyers", "n_goods", "horizon", "discount", "allocation_cost", "type_values", "type_values_per_good"):
        if k in cfg: ekw[k] = cfg[k]
    auction = (create_discrete_multi_auction(**ekw) if args.env == "multi"
               else create_discrete_single_auction(**ekw))
    log(f"env={args.env} resolved env kwargs: {ekw}")
    mech_net = build_mechanism_network_attn() if cfg.get("use_attn_mechanism") else build_mechanism_network()
    with open(os.path.join(args.run_dir, "mechanism_params.pkl"), "rb") as f:
        mech_params = pickle.load(f)
    mechanism = Mechanism(mech_net, mech_params)

    # belief filter: faithful (opponent-informative, = non-clairvoyant posterior) or shipped (buggy)
    BUP = (make_faithful_discrete_belief(auction) if args.belief == "faithful"
           else auction.get_next_belief_state_discrete)
    BINIT = ((lambda s: faithful_init_belief(auction, s)) if args.belief == "faithful"
             else auction.get_initial_belief_state)
    log(f"belief filter = {args.belief}")

    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    gamma = float(auction.discount)
    d = args.deviator; o = 1 - d
    bt = jnp.asarray(np.asarray(auction.buyer_types[d]), jnp.float32)   # (K_per_buyer, G): deviator's type/action set
    K = int(bt.shape[0]); Kp = int(auction.num_type_profiles)
    if args.input == "infostate":
        dphi = K + Kp + T * G + T * B * G       # OWN report history (T*G) + FULL public allocation history (T*B*G)
        def featfn(vd, bs):
            return jnp.concatenate([jax.nn.one_hot(vd, K), bs.beliefs[d],
                                    bs.report_history[:, d, :].reshape(-1),   # OWN reports only (private)
                                    bs.allocation_history.reshape(-1)])       # FULL public outcome history (M2 = full obs history)
    elif args.input == "m15":                    # M1.5 = M2 minus own-report history (appendix faithful oracle)
        dphi = K + Kp + T * B * G                # belief + FULL public allocation history ONLY
        def featfn(vd, bs):
            return jnp.concatenate([jax.nn.one_hot(vd, K), bs.beliefs[d],
                                    bs.allocation_history.reshape(-1)])       # public outcome history; NO own reports
    elif args.input == "m2cum":                  # M2 info-set EXACTLY matching exactBR-M2 (for clean calibration):
        dphi = K + Kp + T * G + B * G            # own type + belief + per-step OWN reports (T*G) + CUMULATIVE alloc (B*G)
        def featfn(vd, bs):
            return jnp.concatenate([jax.nn.one_hot(vd, K), bs.beliefs[d],
                                    bs.report_history[:, d, :].reshape(-1),                  # per-step own reports (= exactBR OWNRH)
                                    jnp.sum(bs.allocation_history, axis=0).reshape(-1)])     # cumulative alloc (= exactBR CUMALLOC)
    else:
        dphi = K + Kp
        def featfn(vd, bs):
            return jnp.concatenate([jax.nn.one_hot(vd, K), bs.beliefs[d]])
    log(f"K={K} num_profiles={Kp} input={args.input} feat_dim={dphi} deviator={d} G={G}")

    def type_idx(ct_d):     # ct_d: (G,) own current type -> index into the K type-pairs
        return jnp.argmin(jnp.sum((bt - ct_d) ** 2, axis=-1))

    # ---- policy params (flat) ----
    def init_params(key):
        k1, k2 = jax.random.split(key)
        return {"W1": 0.1 * jax.random.normal(k1, (dphi, args.hidden)),
                "b1": jnp.zeros((args.hidden,)),
                "W2": 0.01 * jax.random.normal(k2, (args.hidden, K)),
                "b2": jnp.zeros((K,))}
    _flat0, unravel = ravel_pytree(init_params(jax.random.PRNGKey(0)))
    P = _flat0.shape[0]
    log(f"policy params P={P}")

    def logits_of(flat, feat):
        p = unravel(flat)
        h = jnp.tanh(feat @ p["W1"] + p["b1"])
        return h @ p["W2"] + p["b2"]

    def logp(flat, feat, a):
        lg = logits_of(flat, feat)
        return lg[a] - jax.scipy.special.logsumexp(lg)
    score_batch = jax.vmap(jax.grad(logp, argnums=0), in_axes=(None, 0, 0))  # (M,dphi),(M,)->(M,P)

    # ---- one episode rollout: returns feats (T,dphi), actions (T,), rhos (T,) ----
    @jax.jit
    def rollout(flat, key):
        k0, kr = jax.random.split(key)
        s0 = auction.get_initial_state(k0)
        bs0 = BINIT(s0)

        def step(carry, t):
            dev_s, truth_s, bs, kk = carry
            kk, kpol, ktrans = jax.random.split(kk, 3)
            # truthful trajectory (shared transition noise -> CRN)
            tr_report = truth_s.current_type
            tr_alloc, tr_pay = mechanism(truth_s, tr_report, auction)
            u_truth = jnp.sum(tr_alloc[d] * truth_s.current_type[d]) - tr_pay[d]
            next_truth = auction.get_next_state(truth_s, tr_report, tr_alloc, ktrans)
            # belief-conditioned deviation
            vd = type_idx(dev_s.current_type[d])
            feat = featfn(vd, bs)                                            # (dphi,)
            lg = logits_of(flat, feat)
            a = jax.random.categorical(kpol, lg)
            dev_report = dev_s.current_type.at[d].set(bt[a])
            dev_alloc, dev_pay = mechanism(dev_s, dev_report, auction)
            u_dev = jnp.sum(dev_alloc[d] * dev_s.current_type[d]) - dev_pay[d]
            next_dev = auction.get_next_state(dev_s, dev_report, dev_alloc, ktrans)
            next_bs = BUP(bs, dev_report, dev_alloc, next_dev)
            rho = jnp.power(gamma, jnp.asarray(t, jnp.float32)) * (u_dev - u_truth)
            return (next_dev, next_truth, next_bs, kk), (feat, a, rho)

        _, (feats, actions, rhos) = jax.lax.scan(step, (s0, s0, bs0, kr), jnp.arange(T))
        return feats, actions, rhos

    rollout_batch = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))

    def npg_step(flat, key):
        keys = jax.random.split(key, args.episodes)
        feats, actions, rhos = rollout_batch(flat, keys)     # (N,T,dphi),(N,T),(N,T)
        N = args.episodes
        Gtg = jnp.flip(jnp.cumsum(jnp.flip(rhos, 1), axis=1), 1)  # return-to-go
        adv = Gtg - jnp.mean(Gtg, axis=0, keepdims=True)         # per-step baseline
        sc = score_batch(flat, feats.reshape(N * T, dphi), actions.reshape(N * T)).reshape(N, T, P)
        g = jnp.mean(jnp.sum(sc * adv[:, :, None], axis=1), axis=0) / T   # (P,)
        scf = sc.reshape(N * T, P)
        F = (scf.T @ scf) / (N * T)                             # per-step empirical Fisher (P,P)
        dvec = jnp.linalg.solve(F + args.lam * jnp.eye(P), g)
        return flat + args.eta * dvec, float(jnp.mean(jnp.sum(rhos, axis=1)))

    @jax.jit
    def eval_det(flat, key):
        # deterministic argmax policy: sharpen logits so categorical == argmax
        def rollout_det(k):
            k0, kr = jax.random.split(k)
            s0 = auction.get_initial_state(k0); bs0 = BINIT(s0)
            def step(carry, t):
                dev_s, truth_s, bs, kk = carry
                kk, ktrans = jax.random.split(kk)
                tr_report = truth_s.current_type
                tr_alloc, tr_pay = mechanism(truth_s, tr_report, auction)
                u_truth = jnp.sum(tr_alloc[d] * truth_s.current_type[d]) - tr_pay[d]
                next_truth = auction.get_next_state(truth_s, tr_report, tr_alloc, ktrans)
                vd = type_idx(dev_s.current_type[d])
                feat = featfn(vd, bs)
                a = jnp.argmax(logits_of(flat, feat))
                dev_report = dev_s.current_type.at[d].set(bt[a])
                dev_alloc, dev_pay = mechanism(dev_s, dev_report, auction)
                u_dev = jnp.sum(dev_alloc[d] * dev_s.current_type[d]) - dev_pay[d]
                next_dev = auction.get_next_state(dev_s, dev_report, dev_alloc, ktrans)
                next_bs = BUP(bs, dev_report, dev_alloc, next_dev)
                rho = jnp.power(gamma, jnp.asarray(t, jnp.float32)) * (u_dev - u_truth)
                return (next_dev, next_truth, next_bs, kk), rho
            _, rhos = jax.lax.scan(step, (s0, s0, bs0, kr), jnp.arange(T))
            return jnp.sum(rhos)
        return jnp.mean(jax.vmap(rollout_det)(jax.random.split(key, args.eval_episodes)))

    # ---- ARCHITECTURE variants (wide/attn/gru): reuse DiscreteInfoStateHead as the NPG policy, optimized
    #      by the SAME score-function natural gradient. --arch none = flat-MLP policy (default, untouched). ----
    if args.arch != "none":
        from belief_policy import build_discrete_infostate_logits_net
        uor = (args.input != "m15"); cum = (args.input == "m2cum")
        hnet = build_discrete_infostate_logits_net(auction.d_priv, auction.d_pub, args.hidden, 0.1,
                                                   arch=args.arch, use_own_reports=uor, cumulative_alloc=cum)  # proto=0.1: soft enough that the learned residual reliably flips the argmax off truthful for EVERY encoder incl. gru (0.5 left multi-gru intermittently stuck at 0.0)
        bs0 = BINIT(auction.get_initial_state(jax.random.PRNGKey(args.seed)))
        _p0 = hnet.init(jax.random.PRNGKey(0), bs0, auction)
        _flat0h, unravel_h = ravel_pytree(_p0); P = int(_flat0h.shape[0])
        log(f"HEAD-mode NPG: arch={args.arch} use_own_reports={uor} cumulative_alloc={cum} P={P}")
        def init_params(key): return hnet.init(key, bs0, auction)                # override
        def hlogits(flat, bs): return hnet.apply(unravel_h(flat), bs, auction)[d]  # (K,) deviator logits
        def hlogp(flat, bs, a):
            lg = hlogits(flat, bs); return lg[a] - jax.scipy.special.logsumexp(lg)
        hscore = jax.vmap(jax.grad(hlogp, argnums=0), in_axes=(None, 0, 0))
        def _roll(flat, key, deterministic):
            k0, kr = jax.random.split(key); s0 = auction.get_initial_state(k0); b0 = BINIT(s0)
            def step(carry, t):
                dev_s, truth_s, bs, kk = carry
                kk, kpol, ktrans = jax.random.split(kk, 3)
                tr_report = truth_s.current_type
                tr_alloc, tr_pay = mechanism(truth_s, tr_report, auction)
                u_truth = jnp.sum(tr_alloc[d] * truth_s.current_type[d]) - tr_pay[d]
                next_truth = auction.get_next_state(truth_s, tr_report, tr_alloc, ktrans)
                lg = hlogits(flat, bs)
                a = (jnp.argmax(lg) if deterministic else jax.random.categorical(kpol, lg))
                dev_report = dev_s.current_type.at[d].set(bt[a])
                dev_alloc, dev_pay = mechanism(dev_s, dev_report, auction)
                u_dev = jnp.sum(dev_alloc[d] * dev_s.current_type[d]) - dev_pay[d]
                next_dev = auction.get_next_state(dev_s, dev_report, dev_alloc, ktrans)
                next_bs = BUP(bs, dev_report, dev_alloc, next_dev)
                rho = jnp.power(gamma, jnp.asarray(t, jnp.float32)) * (u_dev - u_truth)
                return (next_dev, next_truth, next_bs, kk), (bs, a, rho)
            _, (bss, actions, rhos) = jax.lax.scan(step, (s0, s0, b0, kr), jnp.arange(T))
            return bss, actions, rhos
        rollout_batch_h = jax.jit(jax.vmap(lambda f, k: _roll(f, k, False), in_axes=(None, 0)))
        def npg_step(flat, key):                                                 # override (WOODBURY: P too large for dense Fisher)
            keys = jax.random.split(key, args.episodes)
            bss, actions, rhos = rollout_batch_h(flat, keys)
            N = args.episodes
            Gtg = jnp.flip(jnp.cumsum(jnp.flip(rhos, 1), axis=1), 1)
            adv = Gtg - jnp.mean(Gtg, axis=0, keepdims=True)
            bflat = jax.tree_util.tree_map(lambda x: x.reshape((N * T,) + x.shape[2:]), bss)
            S = hscore(flat, bflat, actions.reshape(N * T))                      # (M, P), M=N*T
            M = S.shape[0]
            g = jnp.mean(jnp.sum(S.reshape(N, T, P) * adv[:, :, None], axis=1), axis=0) / T   # (P,)
            # (S^T S / M + lam I)^{-1} g via Woodbury -> an (M x M) solve (avoids the 9GB P x P Fisher).
            # lam_h=0.1 (NOT the flat-MLP's 1e-3, which makes dvec ~1000x too large for the large-P head
            # and diverges); TRPO-bounded step (kl=0.02, eta cap 1.0) -- mirrors the fixed continuous NPG.
            lam_h = 0.1
            y = jnp.linalg.solve(lam_h * M * jnp.eye(M) + S @ S.T, S @ g)        # (M,)
            dvec = (g - S.T @ y) / lam_h                                         # (P,)
            gd = jnp.maximum(jnp.dot(g, dvec), 1e-12)
            step = jnp.minimum(1.0, jnp.sqrt(2.0 * 0.02 / gd))                   # TRPO-bounded (large-P safe)
            return flat + step * dvec, float(jnp.mean(jnp.sum(rhos, axis=1)))
        @jax.jit
        def eval_det(flat, key):                                                 # override (argmax rollout)
            return jnp.mean(jax.vmap(lambda k: jnp.sum(_roll(flat, k, True)[2]))(
                jax.random.split(key, args.eval_episodes)))

    out = {"experiment": f"discrete_{args.env}", "input": args.input, "oracle": "belief_conditioned_npg", "run_dir": args.run_dir,
           "deviator": d, "recorded": {k: summ.get(k) for k in ("regret_reopt", "regret_eval")},
           "config": {k: v for k, v in vars(args).items()}, "restarts": []}
    best_reg, best_flat = -1e9, None
    for r in range(args.restarts):
        prng = jax.random.PRNGKey(5000 + r)
        flat = ravel_pytree(init_params(prng))[0]
        for it in range(args.iters):
            prng, ks = jax.random.split(prng)
            flat, gbar = npg_step(flat, ks)
        prng, ke = jax.random.split(prng)
        reg = float(eval_det(flat, ke))
        out["restarts"].append({"restart": r, "eval_regret_per_buyer": reg})
        if reg > best_reg:
            best_reg, best_flat = reg, flat
        log(f"restart {r}: belief-NPG held-out regret(per-buyer) = {reg:+.5f}")
    out["best_regret_per_buyer"] = best_reg
    _ladder = (" Single-good ladder: memoryless +0.244 <= belief <= opponent-observed non-clairvoyant +0.448."
               if args.env == "single" else " (multi-good: single-good ladder bounds do not apply.)")
    _lvl = {'infostate': 'M2', 'm15': 'M1.5', 'belief': 'M1', 'm2cum': 'M2-cum(exactBR-aligned)'}[args.input]
    out["note"] = (f"belief-conditioned NPG (score-function), input={args.input} ({_lvl}), "
                   f"env={args.env}, belief={args.belief}." + _ladder)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE best belief-conditioned regret/buyer = {best_reg:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
