"""
cont_npg_oracle.py — CONTINUOUS Natural Policy Gradient deviation oracle (the algorithm the theory
analyzes). Single deviator d (env is buyer-symmetric, so buyer-0's regret = per-buyer regret).

POLICY (stochastic, Gaussian in logit space):
  the head gives v_true and a residual Delta_theta; the truthful logit mean is
      m_theta = logit(p0(v_true)) + Delta_theta,   p0 = clip((v_true-low)/width, eps, 1-eps)
  the sampled action in logit space is  s = stopgrad(m_theta) + eps,  eps ~ N(0, sigma^2 I_G),
  and the report is  r = low + width * sigmoid(s)  in [low, high].
  Detaching m in the SAMPLE makes the report (hence the return R) theta-independent → a pure
  score-function estimator (no pathwise leakage). log-prob of the step:
      log pi_t = sum_g [ -0.5*((s_g - m_theta,g)/sigma)^2 - log sigma - 0.5*log(2pi) ],
  so grad_theta log pi_t = sum_g (s_g - m_theta,g)/sigma^2 * grad_theta m_theta,g  (score fn).

GRADIENT + FISHER:
  L_i = sum_t log pi_t  (per rollout i, differentiable in theta);  R_i = discounted buyer-d regret
  (constant). Policy gradient  g_hat = (1/N) sum_i (R_i - b) * grad L_i = S^T w / N,  w=(R-b)/N-scaled,
  where S (N x P) stacks the per-sample scores grad L_i and b = mean_i R_i (baseline).
  Fisher  F_hat = (1/N) S^T S  is rank <= N << P, so the natural gradient
      d = (F_hat + lam I)^{-1} g_hat
  is solved EXACTLY by the Woodbury identity using only an N x N inverse (never forming the P x P F):
      d = (1/lam) g_hat - (1/lam^2) S^T ( N I_N + (1/lam) S S^T )^{-1} (S g_hat).
  Update: theta <- theta + eta * d.  Eval reports the DETERMINISTIC (mean, eps=0) policy's per-buyer
  regret via the validated det rollout, so it is comparable to pathwise/CMA-ES.
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time, math
sys.path.insert(0, CDMD_ROOT)
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
import jax, jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
from bandit_auction.envs.cont_single import create_cont_single_auction
from bandit_auction.envs.cont_multi import create_cont_multi_auction
from bandit_auction.mechanisms.network import (Mechanism, build_mechanism_network,
    build_mechanism_network_attn)
from bandit_auction.core import AuctionSimulator
from cont_oracle import build_cont_oracle_delta_net, _bounded_report
from cont_rollout import estimate_regret_det
from cont_cfg import load_cont_env, EVAL_SEED


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
_LOG2PI = math.log(2.0 * math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="single", choices=["single", "multi"])
    ap.add_argument("--info", default="m1", choices=["m1", "m2"])
    ap.add_argument("--arch", default="gru", choices=["gru", "wide", "attn", "mlp"])
    ap.add_argument("--deviator", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.3)       # exploration std in logit space
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=512)      # N per NPG step (= Fisher rank cap)
    ap.add_argument("--eval_episodes", type=int, default=20000)
    ap.add_argument("--eval_mid", type=int, default=4000)     # cheaper mid-eval for best-iterate tracking
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--eta", type=float, default=1.0)         # CAP on the (TRPO) natural-gradient step
    ap.add_argument("--lam", type=float, default=0.1)         # Fisher/Tikhonov damping (was 1e-3 -> exploded)
    ap.add_argument("--kl", type=float, default=0.02)         # TRPO target KL per step (bounds step size)
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
    d = args.deviator
    sigma = jnp.float32(args.sigma)
    low, high = auction.types_low, auction.types_high
    width = jnp.maximum(high - low, 1e-12)
    discount = jnp.float32(auction.discount)
    log(f"resolved env kwargs: {ekw}  use_attn={use_attn}")

    delta_net = build_cont_oracle_delta_net(dpr, dpu, G, T, B, args.hidden, args.info, args.arch)
    ubf = auction.get_next_belief_state_continuous

    # ---- deterministic report wrapper for eval (eps=0 mean policy) ----
    def det_report_apply(params, belief, auction_, key, payment):
        v_true, Delta = delta_net.apply(params, belief, auction_, payment)
        return _bounded_report(v_true, Delta, auction_.types_low, auction_.types_high)

    # ---- stochastic single-deviator rollout: returns (R, L) for ONE init state ----
    def rollout_RL(flat, init_state, noise_key):
        params = unflatten(flat)
        bs0 = auction.get_initial_belief_state(init_state)
        pay0 = jnp.zeros((T, B), jnp.float32)

        def step(t, carry):
            cumul, L, dev_state, dev_belief, dev_pay, truth_state, kn = carry
            kn, k_eps, k_trans = jax.random.split(kn, 3)
            # truthful buyer-d utility (shared k_trans)
            t_alloc, t_pay = mechanism(truth_state, truth_state.current_type, auction)
            t_util_d = jnp.sum(t_alloc[d] * truth_state.current_type[d]) - t_pay[d]
            next_truth = auction.get_next_state(truth_state, truth_state.current_type, t_alloc, k_trans)
            # deviator d's stochastic action
            v_true, Delta = delta_net.apply(params, dev_belief, auction, dev_pay)   # (B,G),(B,G)
            p0 = jnp.clip((v_true - low) / width, 1e-6, 1.0 - 1e-6)
            m = (jnp.log(p0) - jnp.log1p(-p0)) + Delta                              # (B,G) theta-dependent logit mean
            eps = sigma * jax.random.normal(k_eps, (B, G))
            s = jax.lax.stop_gradient(m) + eps                                      # detached action (score-fn, not pathwise)
            report_all = low + width * jax.nn.sigmoid(s)                            # (B,G)
            report_d = dev_state.current_type.at[d].set(report_all[d])              # only buyer d deviates
            # log-prob of buyer d's action (theta-dependent through m[d])
            logp = jnp.sum(-0.5 * ((s[d] - m[d]) / sigma) ** 2 - jnp.log(sigma) - 0.5 * _LOG2PI)
            L = L + logp
            alloc_d, pay_d = mechanism(dev_state, report_d, auction)
            util_d = jnp.sum(alloc_d[d] * dev_state.current_type[d]) - pay_d[d]
            cumul = cumul + jnp.power(discount, jnp.asarray(t, jnp.float32)) * (util_d - t_util_d)
            ns = auction.get_next_state(dev_state, report_d, alloc_d, k_trans)
            nb = ubf(dev_belief, report_d, alloc_d, ns)
            npay = dev_pay.at[t].set(pay_d)
            return (cumul, L, ns, nb, npay, next_truth, kn)

        init = (jnp.float32(0.0), jnp.float32(0.0), init_state, bs0, pay0, init_state, noise_key)
        cumul, L = jax.lax.fori_loop(0, T, step, init)[:2]
        return jax.lax.stop_gradient(cumul), L    # R detached (constant weight); L differentiable

    # ---- one NPG step (Woodbury natural-gradient solve) ----
    @jax.jit
    def npg_step(flat, key):
        k_init, k_roll = jax.random.split(key)
        inits = AuctionSimulator.sample_init_states(auction, args.episodes, k_init)
        nkeys = jax.random.split(k_roll, args.episodes)
        def LR(f, s, k):                                   # value = L (differentiable); aux = R (stop-grad constant)
            R, L = rollout_RL(f, s, k); return L, R
        (_, Rs), S = jax.vmap(lambda s, k: jax.value_and_grad(LR, has_aux=True)(flat, s, k))(inits, nkeys)  # Rs (N,), S (N,P) — ONE fwd+bwd
        N = args.episodes
        w = (Rs - jnp.mean(Rs)) / N                                                          # advantage (baseline=mean)
        g = S.T @ w                                                                          # (P,) policy gradient
        # d = (lam I + S^T S / N)^{-1} g  via Woodbury (N x N solve)
        Sg = S @ g                                                                           # (N,)
        Gram = S @ S.T                                                                       # (N,N)
        M = N * jnp.eye(N) + Gram / args.lam
        y = jnp.linalg.solve(M, Sg)                                                          # (N,)
        d_nat = g / args.lam - (S.T @ y) / (args.lam ** 2)                                   # (P,)
        # TRPO-style step: bound the KL per update -> step = sqrt(2*kl / g^T F^{-1} g), capped.
        # (Fixes divergence: a raw eta*d_nat exploded because d_nat can be huge on the 57k-param net.)
        gd = jnp.maximum(jnp.dot(g, d_nat), 1e-12)
        step = jnp.minimum(jnp.sqrt(2.0 * args.kl / gd), args.eta)
        return flat + step * d_nat, jnp.mean(Rs)

    # ---- deterministic held-out eval (per-buyer), n samples ----
    def eval_det(params, key, n):
        return float(estimate_regret_det(auction, mechanism, det_report_apply, params, n, key)) / B

    prng = jax.random.PRNGKey(1234)
    s0 = auction.get_initial_state(prng)
    bs0 = auction.get_initial_belief_state(s0)
    _, unflatten = ravel_pytree(delta_net.init(prng, bs0, auction, jnp.zeros((T, B))))

    best_reg, restarts = -1e9, []
    for r in range(args.restarts):
        kp = jax.random.PRNGKey(5000 + r)
        flat = ravel_pytree(delta_net.init(kp, bs0, auction, jnp.zeros((T, B))))[0]
        # seed best with the TRUTHFUL init (regret ~0) so a diverged run can never report below ~0
        best_flat = flat
        r_best = eval_det(unflatten(flat), jax.random.PRNGKey(EVAL_SEED), args.eval_mid)
        for it in range(args.iters):
            kp, ks = jax.random.split(kp)
            flat, _ = npg_step(flat, ks)
            if (it + 1) % args.eval_every == 0:                                     # best-iterate tracking (CRN mid-eval)
                rmid = eval_det(unflatten(flat), jax.random.PRNGKey(EVAL_SEED), args.eval_mid)
                if rmid > r_best:
                    r_best, best_flat = rmid, flat
        reg = eval_det(unflatten(best_flat), jax.random.PRNGKey(EVAL_SEED), args.eval_episodes)  # held-out at BEST iterate, FIXED seed
        restarts.append({"restart": r, "eval_regret_per_buyer": reg})
        best_reg = max(best_reg, reg)
        log(f"restart {r}: NPG held-out regret/buyer = {reg:+.5f}")

    out = {"experiment": f"cont_{args.env}", "oracle": f"NPG-{args.info}-{args.arch}",
           "info": args.info, "arch": args.arch, "run_dir": args.run_dir, "deviator": d,
           "sigma": args.sigma, "eta": args.eta, "lam": args.lam,
           "best_regret_per_buyer": best_reg, "per_restart": restarts,
           "config": {k: v for k, v in vars(args).items()},
           "note": "continuous NPG (score-fn, Gaussian-in-logit), exact Woodbury Fisher solve; PER BUYER"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE NPG-{args.info}-{args.arch} best held-out regret/buyer = {best_reg:+.5f}  saved {args.out}")


if __name__ == "__main__":
    main()
