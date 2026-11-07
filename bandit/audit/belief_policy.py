"""
belief_policy.py — the CORRECTED belief-conditioned discrete deviation policy head.

The shipped DiscreteProtoResidualHead is memoryless: it reads only priv_embeds (= own current
type, since d_priv=n_goods for discrete) and pub_embed (zeros); it NEVER reads belief_state.beliefs,
despite the paper (p.32) describing a head that takes the Bayes belief b in R^{B x K}. This module
implements the paper's intended belief-CONDITIONED head: the residual trunk additionally consumes
the maintained Bayes posterior. proto_strength_init defaults to 3.0 (non-saturated gradient) so the
optimization bug is fixed too. Same truthful-prototype init and straight-through Gumbel-softmax as
the original, so it is a drop-in policy_network for train_with_checkpoints / reopt.

The belief is fed as a stop-gradient observation (standard reactive-policy convention).
"""
import math
import haiku as hk
import jax
import jax.numpy as jnp
from bandit_auction.core import BeliefState, BanditAuction


def _trunk_sizes(hidden, arch):
    """Axis-III capacity knob. 'base' = [h,h]; 'wide' = [4h,4h,4h] (deeper+wider)."""
    if arch == "wide":
        return [4 * hidden, 4 * hidden, 4 * hidden]
    return [hidden, hidden]


class DiscreteBeliefHead(hk.Module):
    # M1 head: input is the fixed vector (own type, belief) -> NO sequence, so the only architecture
    # knob is capacity (arch in {base, wide}). attn/gru are undefined here (nothing to attend/recur over).
    def __init__(self, d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                 residual_init_std=0.0, stop_grad_truth=True, arch="base", name=None):
        super().__init__(name=name)
        self.d_priv = d_priv
        self.d_pub = d_pub
        self.hidden = hidden
        self.proto_strength_init = float(proto_strength_init)
        self.residual_init_std = float(residual_init_std)
        self.stop_grad_truth = stop_grad_truth
        self.arch = arch
        if arch not in ("base", "wide"):
            raise ValueError(f"M1 belief head has no sequence; arch must be base|wide, got {arch!r}")
        self.trunk = hk.nets.MLP(_trunk_sizes(hidden, arch), activate_final=True)

    def logits(self, belief_state: BeliefState, auction: BanditAuction):
        B, G = auction.num_buyers, auction.num_goods
        buyer_types = auction.buyer_types           # (B, T, G)
        T = buyer_types.shape[1]
        H_type = belief_state.priv_embeds[:, :G]    # (B, G) = current type
        if self.stop_grad_truth:
            H_type = jax.lax.stop_gradient(H_type)
        match = jnp.all(H_type[:, None, :] == buyer_types, axis=-1).astype(jnp.float32)  # (B,T)
        # NOTE: compute the softplus-inverse init as a pure-Python constant (math, not jnp):
        # jnp here traces under the training scan -> ConcretizationTypeError (same fix as core.py:482).
        k_raw = hk.get_parameter(
            "proto_strength_raw", shape=(1,),
            init=hk.initializers.Constant(
                float(math.log(math.expm1(max(self.proto_strength_init, 1e-6))))))
        k = jax.nn.softplus(k_raw)
        proto_logits = k * match                    # (B,T)
        # ---- Residual: belief-conditioned (the fix) ----
        H = belief_state.priv_embeds                # (B, d_priv)
        c = belief_state.pub_embed                  # (d_pub,) zeros for discrete
        beliefs = jax.lax.stop_gradient(belief_state.beliefs)   # (B, K) Bayes posterior  <-- FED IN
        zero = hk.initializers.Constant(0.0)
        c_proj = hk.Linear(self.d_priv, w_init=zero, b_init=zero)(c)
        x = jnp.concatenate([H, jnp.broadcast_to(c_proj, H.shape), beliefs], axis=-1)
        z = self.trunk(x)
        res_init = (hk.initializers.RandomNormal(self.residual_init_std)
                    if self.residual_init_std > 0 else zero)
        residual_logits = hk.Linear(T, w_init=res_init, b_init=zero)(z)   # (B,T)
        return proto_logits + residual_logits, buyer_types

    def __call__(self, belief_state: BeliefState, auction: BanditAuction, key):
        logits, buyer_types = self.logits(belief_state, auction)
        T = buyer_types.shape[1]
        if key is not None:
            g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)))
            soft = jax.nn.softmax((logits + g) / 1.0, axis=-1)
        else:
            soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), T)
        report_probs = soft + jax.lax.stop_gradient(hard - soft)
        return jnp.einsum('bt,btg->bg', report_probs, buyer_types)      # (B,G)


def build_discrete_belief_policy_net(d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                                     residual_init_std=0.0, stop_grad_truth=True, arch="base"):
    def f(belief_state, auction, key=None):
        return DiscreteBeliefHead(d_priv, d_pub, hidden, proto_strength_init,
                                  residual_init_std, stop_grad_truth, arch=arch)(belief_state, auction, key)
    return hk.without_apply_rng(hk.transform(f))


class DiscreteInfoStateHead(hk.Module):
    """Theory-faithful information-state head. Same truthful-prototype + Gumbel-softmax structure, but
    the residual trunk consumes the observed history the mechanism uses. Two info levels via
    `use_own_reports` (this is the ONLY difference between them, so M2 = M1.5 + own-report-history exactly):

      use_own_reports=True   -> M2   (full observed history): own current type + belief
                                + OWN report history + FULL public allocation history.
      use_own_reports=False  -> M1.5 (appendix's faithful oracle): own current type + belief
                                + FULL public allocation history ONLY (the *projected public* outcome
                                history). This mirrors the CONTINUOUS appendix oracle, whose report-free
                                GRU belief already encodes the projected public history; here the discrete
                                belief-only M1 head lacks it, so M1.5 supplies it explicitly.

    M1 (DiscreteBeliefHead) < M1.5 (this, use_own_reports=False) < M2 (this, use_own_reports=True).
    Rendering the deviator Markov w.r.t. the observed statistic makes the belief-conditioned oracle a
    SOUND worst-case deviation oracle — unlike the type-belief-only M1 head, which is blind to the
    mechanism's history dependence."""
    def __init__(self, d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                 residual_init_std=0.0, stop_grad_truth=True, arch="flatten",
                 use_own_reports=True, cumulative_alloc=False, name=None):
        super().__init__(name=name)
        self.d_priv = d_priv; self.d_pub = d_pub; self.hidden = hidden
        self.proto_strength_init = float(proto_strength_init)
        self.residual_init_std = float(residual_init_std)
        self.stop_grad_truth = stop_grad_truth
        self.use_own_reports = use_own_reports   # False => M1.5 (drop own-report history; keep public alloc)
        self.cumulative_alloc = cumulative_alloc # True => sum-over-t allocation (matches exactBR-M2 info-set exactly)
        if cumulative_alloc and arch not in ("flatten", "wide"):
            raise ValueError("cumulative_alloc only defined for flatten/wide (no per-step sequence for attn/gru)")
        # Axis-III sequence encoder for the T-step observed history:
        #   flatten = concat all steps -> MLP (current, order-blind, fixed T)
        #   wide    = flatten -> deeper/wider MLP (capacity control)
        #   attn    = attention-pool over steps, query = (own type, belief) [mirrors mechanism Stream-3]
        #   gru     = GRU over steps [mirrors the continuous oracle]
        if arch not in ("flatten", "wide", "attn", "gru"):
            raise ValueError(f"M2 head arch must be flatten|wide|attn|gru, got {arch!r}")
        self.arch = arch
        self.trunk = hk.nets.MLP(_trunk_sizes(hidden, "wide" if arch == "wide" else "base"),
                                 activate_final=True)

    def _encode_history(self, seq, ctx0):
        """seq: (B, Th, feat) per-step observed history; ctx0: (B, d0) = (own type, belief).
        Returns a (B, hid) recurrent/attention summary. Used only for arch in {attn, gru}."""
        B = seq.shape[0]
        if self.arch == "attn":                                                # query = (own type, belief)
            q = hk.Linear(self.hidden, name="attn_q")(ctx0)                     # (B, hid)
            K_ = hk.Linear(self.hidden, name="attn_k")(seq)                     # (B, Th, hid)
            V_ = hk.Linear(self.hidden, name="attn_v")(seq)                     # (B, Th, hid)
            scores = jnp.einsum('bh,bth->bt', q, K_) / jnp.sqrt(float(self.hidden))
            w = jax.nn.softmax(scores, axis=-1)                                 # (B, Th)
            return jnp.einsum('bt,bth->bh', w, V_)                              # (B, hid) context
        core = hk.GRU(self.hidden)                                             # gru: recurrent encode
        seq_tm = jnp.transpose(seq, (1, 0, 2))                                  # (Th, B, feat) time-major
        _, final = hk.dynamic_unroll(core, seq_tm, core.initial_state(B))
        return final                                                           # (B, hid)

    def logits(self, belief_state: BeliefState, auction: BanditAuction):
        B, G = auction.num_buyers, auction.num_goods
        Th = auction.num_horizons                 # HORIZON (for report/alloc history), distinct from
        buyer_types = auction.buyer_types; T = buyer_types.shape[1]   # T = K (num type values)
        H_type = belief_state.priv_embeds[:, :G]
        if self.stop_grad_truth:
            H_type = jax.lax.stop_gradient(H_type)
        match = jnp.all(H_type[:, None, :] == buyer_types, axis=-1).astype(jnp.float32)
        k_raw = hk.get_parameter("proto_strength_raw", shape=(1,),
            init=hk.initializers.Constant(float(math.log(math.expm1(max(self.proto_strength_init, 1e-6))))))
        k = jax.nn.softplus(k_raw)
        proto_logits = k * match
        # ---- residual: M2 = FULL OBSERVED HISTORY (matches the non-clairvoyant / M2-exactBR info set) ----
        # own type (H) + belief (helper) + OWN report history (private; opponents' reports NOT observed)
        # + the FULL per-step public allocation history (the public outcome, from which the opponent's
        # reports are recoverable by injectivity). This is exactly what the deviator observes.
        H = belief_state.priv_embeds
        beliefs = jax.lax.stop_gradient(belief_state.beliefs)                  # (B,K)
        if self.arch in ("flatten", "wide"):
            # ORIGINAL M2 layout: [own reports (all steps), full public allocation (all steps)].
            # M1.5 (use_own_reports=False) DROPS own reports, keeping only the public allocation history.
            # cumulative_alloc=True uses sum_t allocation (B*G) instead of per-step (Th*B*G) -> matches exactBR-M2.
            if self.cumulative_alloc:
                alloc_hist = jax.lax.stop_gradient(jnp.sum(belief_state.allocation_history, axis=0)).reshape(-1)  # (B*G,)
            else:
                alloc_hist = jax.lax.stop_gradient(belief_state.allocation_history).reshape(-1)  # (Th*B*G,)
            alloc_b = jnp.broadcast_to(alloc_hist, (B, alloc_hist.shape[0]))
            if self.use_own_reports:
                own_rh = jax.lax.stop_gradient(
                    jnp.transpose(belief_state.report_history, (1, 0, 2))).reshape(B, Th * G)
                x = jnp.concatenate([H, beliefs, own_rh, alloc_b], axis=-1)
            else:
                x = jnp.concatenate([H, beliefs, alloc_b], axis=-1)          # M1.5: belief + projected public history
        else:
            # per-step observed history seq_t = [ (own report_t (G)) , FULL public allocation_t (B*G) ]
            alloc_seq = jax.lax.stop_gradient(belief_state.allocation_history).reshape(Th, B * G)      # (Th,B*G)
            alloc_seq_b = jnp.broadcast_to(alloc_seq[None], (B, Th, B * G))    # public outcome -> all buyers
            if self.use_own_reports:
                own_rh_seq = jax.lax.stop_gradient(jnp.transpose(belief_state.report_history, (1, 0, 2)))  # (B,Th,G)
                seq = jnp.concatenate([own_rh_seq, alloc_seq_b], axis=-1)     # (B, Th, G+B*G)
            else:
                seq = alloc_seq_b                                            # M1.5: public alloc history only
            ctx0 = jnp.concatenate([H, beliefs], axis=-1)                      # query/context = (own type, belief)
            enc = self._encode_history(seq, ctx0)                             # (B, hid)
            x = jnp.concatenate([H, beliefs, enc], axis=-1)
        z = self.trunk(x)
        res_init = (hk.initializers.RandomNormal(self.residual_init_std)
                    if self.residual_init_std > 0 else hk.initializers.Constant(0.0))
        residual_logits = hk.Linear(T, w_init=res_init, b_init=hk.initializers.Constant(0.0))(z)
        return proto_logits + residual_logits, buyer_types

    def __call__(self, belief_state: BeliefState, auction: BanditAuction, key):
        logits, buyer_types = self.logits(belief_state, auction)
        T = buyer_types.shape[1]
        if key is not None:
            g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)))
            soft = jax.nn.softmax((logits + g) / 1.0, axis=-1)
        else:
            soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), T)
        report_probs = soft + jax.lax.stop_gradient(hard - soft)
        return jnp.einsum('bt,btg->bg', report_probs, buyer_types)


def build_discrete_infostate_policy_net(d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                                        residual_init_std=0.0, stop_grad_truth=True, arch="flatten",
                                        use_own_reports=True, cumulative_alloc=False):
    """use_own_reports=True -> M2 (full observed history); False -> M1.5 (belief + projected public
    allocation history, i.e. the appendix's faithful oracle). Same head otherwise, so the two info
    levels differ by EXACTLY the own-report history. cumulative_alloc=True (flatten/wide only) uses
    sum-over-t allocation to match exactBR-M2's info-set exactly (for clean calibration)."""
    def f(belief_state, auction, key=None):
        return DiscreteInfoStateHead(d_priv, d_pub, hidden, proto_strength_init, residual_init_std,
                                     stop_grad_truth, arch=arch, use_own_reports=use_own_reports,
                                     cumulative_alloc=cumulative_alloc)(belief_state, auction, key)
    return hk.without_apply_rng(hk.transform(f))


def build_discrete_m15_policy_net(d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                                  residual_init_std=0.0, stop_grad_truth=True, arch="flatten"):
    """M1.5 convenience wrapper: the appendix's faithful oracle = own type + belief + projected public
    allocation history (NO own-report history). Equivalent to the M2 head with use_own_reports=False."""
    return build_discrete_infostate_policy_net(d_priv, d_pub, hidden, proto_strength_init,
                                               residual_init_std, stop_grad_truth, arch=arch,
                                               use_own_reports=False)


def build_discrete_belief_logits_net(d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                                     residual_init_std=0.0, stop_grad_truth=True):
    """Debug/verification: returns the pre-Gumbel logits (B,T) so belief-dependence is visible
    without the discrete argmax masking it."""
    def f(belief_state, auction):
        logits, _ = DiscreteBeliefHead(d_priv, d_pub, hidden, proto_strength_init,
                                       residual_init_std, stop_grad_truth).logits(belief_state, auction)
        return logits
    return hk.without_apply_rng(hk.transform(f))


def build_discrete_infostate_logits_net(d_priv, d_pub, hidden=128, proto_strength_init=3.0,
                                        stop_grad_truth=True, arch="flatten",
                                        use_own_reports=True, cumulative_alloc=False):
    """Pre-Gumbel logits (B,T) of the M1.5/M2 head at a given ARCHITECTURE, for score-function NPG.
    Reuses the SAME DiscreteInfoStateHead as pathwise/ES, so NPG-arch-X is the identical function class
    optimized by natural gradient instead of pathwise/ES. arch in {flatten,wide,attn,gru}."""
    def f(belief_state, auction):
        logits, _ = DiscreteInfoStateHead(d_priv, d_pub, hidden, proto_strength_init, 0.0,
                                          stop_grad_truth, arch=arch, use_own_reports=use_own_reports,
                                          cumulative_alloc=cumulative_alloc).logits(belief_state, auction)
        return logits                                # (B, T)
    return hk.without_apply_rng(hk.transform(f))
