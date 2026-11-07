"""
exact_br_m1.py — exactBR-M1: the exact best response in the BELIEF-restricted policy class M1
(the deviation policy may condition only on (own current type, belief over opponent current type)).

This is the decisive capacity-vs-information cell. The deviator's reward depends on the full history
(the mechanism attends to it), so (own type, belief) is NOT Markov => this is a belief-restricted POMDP
with no plain backward induction. We solve it by CONSTRAINED POLICY ITERATION on the exact
non-clairvoyant tree: exact policy EVALUATION (backward) + exact reach weights (forward) + greedy
per-info-state improvement (all nodes sharing an (own type, belief) info-state must play one action).
Multiple restarts guard against local optima; we also cross-check against M1-NPG.

Node/belief conventions match exact_br_discrete_single.py::expectimax_noncla (Phase-2 verified):
belief b = one-step transition from the (outcome-identified) opponent past type; child index of node n via
(dev action a, opp type k, own next type vd') at level t+1 is  n*K^3 + a*K^2 + k*K + vd'.

Usage: python exact_br_m1.py --run_dir <ckpt> --out results/exact_br_m1/<name>.json [--tol 1e-3 --restarts 8]
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse, os, json, pickle, sys, time
sys.path.insert(0, CDMD_ROOT)
import jax, jax.numpy as jnp, numpy as np
from bandit_auction.core import State
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.mechanisms.network import (Mechanism, build_mechanism_network,
    build_mechanism_network_attn)

CHUNK = 400_000
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--info_class", default="M1", choices=["M1", "M2"])   # M1=(own type,belief); M2 adds (own report history, cumulative allocation)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--max_iters", type=int, default=60)
    ap.add_argument("--deviator", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"jax {jax.__version__} devices={jax.devices()}")

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summ = json.load(f)
    cfg = summ.get("cfg", {}) or {}
    ekw = dict(seed=args.seed)
    for k in ("n_buyers", "n_goods", "horizon", "discount", "allocation_cost", "type_values"):
        if k in cfg: ekw[k] = cfg[k]
    auction = create_discrete_single_auction(**ekw)
    mech_net = build_mechanism_network_attn() if cfg.get("use_attn_mechanism") else build_mechanism_network()
    with open(os.path.join(args.run_dir, "mechanism_params.pkl"), "rb") as f:
        mech_params = pickle.load(f)
    mechanism = Mechanism(mech_net, mech_params)
    B, T, G = auction.num_buyers, auction.num_horizons, auction.num_goods
    gamma = float(auction.discount)
    tv = jnp.asarray(np.asarray(auction.buyer_types[0, :, 0]), jnp.float32)
    K = int(tv.shape[0]); d = args.deviator; o = 1 - d

    def mech_one(rh, ah, ct, t, report):
        st = State(time_step=t, allocation_history=ah, current_type=ct, report_history=rh)
        return mechanism(st, report, auction)
    mech_v = jax.jit(jax.vmap(mech_one, in_axes=(0, 0, 0, None, 0)), static_argnums=3)
    def mech_chunked(rh, ah, ct, t, report):
        N = rh.shape[0]; A_, P_ = [], []
        for i in range(0, N, CHUNK):
            a, p = mech_v(rh[i:i+CHUNK], ah[i:i+CHUNK], ct[i:i+CHUNK], t, report[i:i+CHUNK])
            A_.append(a); P_.append(p)
        return jnp.concatenate(A_, 0), jnp.concatenate(P_, 0)
    def trans_row(cur_val, alloc_val):
        pr = jax.vmap(lambda k: auction.get_transition_prob(
            auction, 0, jnp.reshape(cur_val, (1,)), jnp.reshape(alloc_val, (1,)),
            jnp.reshape(tv[k], (1,))))(jnp.arange(K))
        return pr / jnp.maximum(pr.sum(), 1e-12)
    trans_v = jax.jit(jax.vmap(trans_row, in_axes=(0, 0)))

    # ---- build the non-clairvoyant tree, storing u (N,K,K), Pd (N,K,K,K), b (N,K), vd (N,) per level ----
    K2, BRc = K * K, K * K * K
    vd = jnp.arange(K); voprev = -jnp.ones(K, jnp.int32)
    rh = jnp.zeros((K, T, B, G), jnp.float32); ah = jnp.zeros((K, T, B, G), jnp.float32)
    U, PD, BEL, VD, OWNRH, CUMALLOC = [], [], [], [], [], []
    for t in range(T):
        N = vd.shape[0]
        b = jnp.ones((N, K), jnp.float32) / K if t == 0 else trans_v(tv[voprev], ah[:, t - 1, o, 0])
        u = np.zeros((N, K, K), np.float32)
        Pd = None if t == T - 1 else np.zeros((N, K, K, K), np.float32)
        rh_next, ah_next = [], []
        for a in range(K):
            for k in range(K):
                report = jnp.zeros((N, B, G), jnp.float32).at[:, d, 0].set(jnp.full((N,), tv[a])).at[:, o, 0].set(jnp.full((N,), tv[k]))
                ct = jnp.zeros((N, B, G), jnp.float32).at[:, d, 0].set(tv[vd]).at[:, o, 0].set(jnp.full((N,), tv[k]))
                alloc, pay = mech_chunked(rh, ah, ct, t, report)
                u[:, a, k] = np.asarray(tv[vd] * alloc[:, d, 0] - pay[:, d])
                if t < T - 1:
                    Pd[:, a, k, :] = np.asarray(trans_v(tv[vd], alloc[:, d, 0]))
                    rh_next.append(rh.at[:, t].set(report)); ah_next.append(ah.at[:, t].set(alloc))
        U.append(u); PD.append(Pd); BEL.append(np.asarray(b)); VD.append(np.asarray(vd))
        # M2 info-state extras: deviator's OWN report history + PUBLIC cumulative allocation (mechanism inputs)
        OWNRH.append(np.asarray(rh[:, :, d, :]).reshape(u.shape[0], T * G))
        CUMALLOC.append(np.asarray(jnp.sum(ah, axis=1)).reshape(u.shape[0], B * G))
        if t < T - 1:
            rh = jnp.repeat(jnp.stack(rh_next, 1).reshape(N * K2, T, B, G), K, 0)
            ah = jnp.repeat(jnp.stack(ah_next, 1).reshape(N * K2, T, B, G), K, 0)
            idx = np.arange(N * BRc); vd = jnp.asarray(idx % K); voprev = jnp.asarray((idx // K) % K)
            log(f"  built level {t+1}: N={vd.shape[0]}")

    # ---- info-state id per node per level: (own type vd, rounded belief) ----
    inv = round(1.0 / args.tol)
    def make_info(with_m2):
        INFO, NINFO = [], []
        for t in range(T):
            parts = [VD[t].reshape(-1, 1).astype(np.int64), np.round(BEL[t] * inv).astype(np.int64)]
            if with_m2:
                parts.append(np.round(OWNRH[t] * inv).astype(np.int64))
                parts.append(np.round(CUMALLOC[t] * inv).astype(np.int64))
            comb = np.concatenate(parts, axis=1)
            _, inverse = np.unique(comb, axis=0, return_inverse=True)
            INFO.append(inverse.reshape(-1)); NINFO.append(int(inverse.max()) + 1)
        return INFO, NINFO
    INFO_M1, NINFO_M1 = make_info(False)
    INFO_T, NINFO_T = (make_info(True) if args.info_class == "M2" else (INFO_M1, NINFO_M1))
    log(f"info_class={args.info_class}  M1 info-states={NINFO_M1}  target info-states={NINFO_T}")

    def evaluate(node_action):     # node_action[t]: (N_t,) action per node -> V per level, value at root
        V = [None] * T
        a_last = node_action[T - 1]
        V[T - 1] = np.einsum('nk,nk->n', BEL[T - 1], U[T - 1][np.arange(len(a_last)), a_last, :])
        for t in range(T - 2, -1, -1):
            Nt = U[t].shape[0]; a = node_action[t]
            Vnext = V[t + 1].reshape(Nt, K, K, K)                    # [n,a,k,vd']
            cont = (PD[t] * Vnext).sum(-1)                           # (N,K,K) [n,a,k]
            Q = U[t] + gamma * cont                                  # (N,K,K)
            Qak = np.einsum('nk,nak->na', BEL[t], Q)                 # (N,K) value per action
            V[t] = Qak[np.arange(Nt), a]
        return V, float(V[0].mean())

    def reach_weights(node_action):
        R = [None] * T
        R[0] = np.full(U[0].shape[0], 1.0 / U[0].shape[0])           # 1/K per level-0 own type
        for t in range(T - 1):
            Nt = U[t].shape[0]; a = node_action[t]
            selPd = PD[t][np.arange(Nt), a, :, :]                    # (N,K,vd')
            rkv = R[t][:, None, None] * BEL[t][:, :, None] * selPd   # (N,K,vd')
            rn = np.zeros((Nt, K, K, K), np.float32)
            rn[np.arange(Nt), a, :, :] = rkv
            R[t + 1] = rn.reshape(Nt * BRc)
        return R

    def qvals(node_action, V):     # Q per (node, action): (N,K)
        Q = [None] * T
        a_last = node_action[T - 1]
        Q[T - 1] = np.einsum('nk,nak->na', BEL[T - 1], U[T - 1])
        for t in range(T - 2, -1, -1):
            Nt = U[t].shape[0]
            Vnext = V[t + 1].reshape(Nt, K, K, K)
            cont = (PD[t] * Vnext).sum(-1)
            Q[t] = np.einsum('nk,nak->na', BEL[t], U[t] + gamma * cont)
        return Q

    truth_val = evaluate([VD[t] for t in range(T)])[1]

    def group_vd(INFO, NINFO):
        G = []
        for t in range(T):
            gv = np.zeros(NINFO[t], np.int64); gv[INFO[t]] = VD[t]; G.append(gv)
        return G

    def to_node(api, INFO):
        return [api[t][INFO[t]] for t in range(T)]

    def policy_iteration(INFO, NINFO, init_api):
        api = [a.copy() for a in init_api]; prev = -1e9
        for _ in range(args.max_iters):
            na = to_node(api, INFO)
            V, val = evaluate(na); R = reach_weights(na); Q = qvals(na, V)
            new_api = []; changed = 0
            for t in range(T):
                wQ = np.stack([np.bincount(INFO[t], weights=R[t] * Q[t][:, a], minlength=NINFO[t])
                               for a in range(K)], axis=1)
                b = wQ.argmax(1); changed += int((b != api[t]).sum()); new_api.append(b)
            api = new_api
            if changed == 0 or abs(val - prev) < 1e-9:
                break
            prev = val
        return evaluate(to_node(api, INFO))[1], api

    def solve(INFO, NINFO, extra_inits, nrestart):
        GV = group_vd(INFO, NINFO); rng = np.random.RandomState(0)
        inits = list(extra_inits) + [[GV[t].copy() for t in range(T)]] + \
                [[rng.randint(0, K, NINFO[t]) for t in range(T)] for _ in range(nrestart)]
        best_v, best_api = -1e9, None
        for init in inits:
            v, api = policy_iteration(INFO, NINFO, init)
            if v > best_v: best_v, best_api = v, api
        return best_v, best_api

    # --- solve M1 first (few info-states => reliable), then WARM-START the richer class from it ---
    m1_val, m1_api = solve(INFO_M1, NINFO_M1, [], args.restarts)
    log(f"  M1 exactBR value={m1_val:.6f} regret={m1_val - truth_val:+.6f}")
    if args.info_class == "M1":
        best = m1_val
    else:
        warm = []                              # each M2 info-state inherits its M1 info-state's action
        for t in range(T):
            _, first = np.unique(INFO_T[t], return_index=True)
            warm.append(m1_api[t][INFO_M1[t][first]])
        m2_val, _ = solve(INFO_T, NINFO_T, [warm], args.restarts)
        best = max(m2_val, m1_val)             # M2 >= M1 guaranteed by the warm start
        log(f"  M2 exactBR value={best:.6f} regret={best - truth_val:+.6f}")
    vals = [m1_val, best]
    regret = best - truth_val

    out = {"experiment": "discrete_single", "oracle": f"exactBR-{args.info_class} (constrained PI)",
           "info_class": args.info_class, "info_set": (
               "(own type, belief)" if args.info_class == "M1"
               else "(own type, belief, own report history, public cumulative allocation)"),
           "run_dir": args.run_dir, "tol": args.tol, "restarts": args.restarts,
           "truthful_value": truth_val, "exactBR_value": best,
           "exactBR_regret_per_buyer": regret,
           "all_restart_values": vals, "info_states_per_level": NINFO_T,
           "note": ("belief-restricted exact BR via constrained policy iteration (exact eval + per-infostate "
                    "greedy improvement). Cross-check vs M1-NPG; if they agree, this is the M1 ceiling "
                    "(capacity-sufficient). Upper-bounded by non-clairvoyant (=M2 exactBR).")}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"DONE exactBR-M1 regret/buyer = {regret:+.6f} (value {best:.4f}, truthful {truth_val:.4f}). saved {args.out}")


if __name__ == "__main__":
    main()
