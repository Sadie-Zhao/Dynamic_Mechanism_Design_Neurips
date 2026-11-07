"""
cont_oracle.py — CONTINUOUS deviation-oracle heads for the 3-axis ladder.

Two orthogonal knobs (algorithm is handled by the driver, not here):
  INFO  ∈ {m1, m2}
    m1 = the appendix oracle's information set: own current type + own report history
         + full public allocation history (the report/alloc history live in belief_state).
    m2 = m1 + the agent's OWN payment history (the one observable the appendix oracle omits).
         Payment is not in belief_state, so the rollout threads it in as `payment_history`.
  ARCH ∈ {gru, wide, attn, mlp}  (the oracle's HISTORY ENCODER)
    gru  = BASE = the shipped fixed belief-GRU embedding (priv_embeds, pub_embed) -> trunk.
           (gru + m1 reproduces core.LogitProtoResidualHead exactly.)
    wide = base features, but a wider/deeper trunk (report-head capacity knob).
    attn = the oracle RE-ENCODES the raw observed history with its own trainable attention
           (query = current observation: own type + cumulative public allocation).
    mlp  = the oracle re-encodes the raw history by flatten -> MLP.

Report head (shared across ALL cells): the appendix bounded logit-residual map
    p0 = clip((v_true - low)/width, eps, 1-eps);  s = logit(p0) + Delta;  r = low + width*sigmoid(s)
so at Delta=0 the report is truthful and it is always in [low, high].

`payment_history` convention: (T, B) — payment of each buyer at each step of THIS trajectory.
Column i is buyer i's own payment; the head uses row j's payment for its row-j output, so buyer i's
report (the one the rollout keeps) conditions on buyer i's own payment. Zeros for future steps.
"""
import math
import haiku as hk
import jax
import jax.numpy as jnp
from bandit_auction.core import BeliefState, BanditAuction


def _trunk_sizes(hidden, arch):
    """Report-head trunk. 'wide' = 3x[4h] (deeper+wider); else 2x[h]."""
    return [4 * hidden, 4 * hidden, 4 * hidden] if arch == "wide" else [hidden, hidden]


def _bounded_report(v_true, Delta, low, high, eps=1e-6):
    """Shared logit-residual bounded map — identical to core.LogitProtoResidualHead (core.py:411-414)."""
    width = jnp.maximum(high - low, 1e-12)
    p0 = jnp.clip((v_true - low) / width, eps, 1.0 - eps)
    s = (jnp.log(p0) - jnp.log1p(-p0)) + Delta
    return low + width * jax.nn.sigmoid(s)


class ContOracleHead(hk.Module):
    def __init__(self, d_priv, d_pub, G, T, B, hidden=128, info="m1", arch="gru",
                 residual_init_std=0.0, stop_grad_truth=True, eps=1e-6, name=None):
        super().__init__(name=name)
        assert info in ("m1", "m2"), info
        assert arch in ("gru", "wide", "attn", "mlp"), arch
        self.d_priv, self.d_pub, self.G, self.T, self.B = d_priv, d_pub, G, T, B
        self.hidden, self.info, self.arch = hidden, info, arch
        self.residual_init_std, self.stop_grad_truth, self.eps = residual_init_std, stop_grad_truth, eps
        self.trunk = hk.nets.MLP(_trunk_sizes(hidden, arch), activate_final=True)

    # ---- raw per-step observed-history sequence (for attn / mlp), (B, T, feat) ----
    def _raw_seq(self, belief_state: BeliefState, payment_history):
        T, B, G = self.T, self.B, self.G
        rh = belief_state.report_history        # (T, B, G)
        ah = belief_state.allocation_history    # (T, B, G)
        own_rep = jnp.transpose(rh, (1, 0, 2))                                  # (B, T, G) OWN reports (row i = buyer i)
        alloc_pub = jnp.broadcast_to(ah.reshape(T, B * G)[None], (B, T, B * G)) # (B, T, B*G) FULL public alloc (shared)
        parts = [own_rep, alloc_pub]
        if self.info == "m2":
            pay = jnp.transpose(payment_history, (1, 0))[:, :, None]            # (B, T, 1) OWN payment (row i = buyer i)
            parts.append(pay)
        # NOTE: no stop_gradient here — the observed history is theta-dependent (own past reports/allocs),
        # so pathwise BPTT should flow through it (matches gru/wide, which flow through priv_embeds). In
        # NPG the history is already detached (actions sampled via stop_gradient(m)+eps); ES is gradient-free.
        return jnp.concatenate(parts, axis=-1)                                 # (B, T, feat)

    def _features(self, belief_state: BeliefState, payment_history):
        """Return (v_true (B,G), feats (B,d_feat)) for the chosen (info, arch)."""
        B, G = self.B, self.G
        H = belief_state.priv_embeds            # (B, d_priv) fixed belief-GRU private embedding
        c = belief_state.pub_embed              # (d_pub,)   fixed belief-GRU public embedding
        v_true = H[:, :G]                       # own current type (exact, stored in first G dims)
        if self.stop_grad_truth:
            v_true = jax.lax.stop_gradient(v_true)
        zero = hk.initializers.Constant(0.0)

        if self.arch in ("gru", "wide"):
            # ---- BASE features = appendix: [priv_embeds, zero-init proj of pub_embed] ----
            c_proj = hk.Linear(self.d_priv, w_init=zero, b_init=zero)(c)        # zero-init (matches shipped)
            feats = jnp.concatenate([H, jnp.broadcast_to(c_proj, H.shape)], axis=-1)  # (B, 2*d_priv)
            if self.info == "m2":
                own_pay = jnp.transpose(payment_history, (1, 0))                          # (B, T) own payment (theta-dep; see _raw_seq note)
                pay_enc = hk.nets.MLP([self.hidden], activate_final=True)(own_pay)        # (B, hidden)
                feats = jnp.concatenate([feats, pay_enc], axis=-1)
            return v_true, feats

        # ---- trainable encoder over the raw observed history (attn / mlp) ----
        seq = self._raw_seq(belief_state, payment_history)                      # (B, T, feat)
        if self.arch == "mlp":
            enc = seq.reshape(B, -1)                                            # flatten (order-agnostic, fixed T)
        else:  # attn: query = current observation (own type + cumulative public allocation), no dynamic index
            cum_alloc = jax.lax.stop_gradient(
                jnp.sum(belief_state.allocation_history, axis=0).reshape(-1))   # (B*G,) observed cumulative alloc
            cum_alloc_b = jnp.broadcast_to(cum_alloc, (B, cum_alloc.shape[0]))  # (B, B*G) public
            q_in = jnp.concatenate([v_true, cum_alloc_b], axis=-1)              # (B, G + B*G)
            q = hk.Linear(self.hidden, name="attn_q")(q_in)                     # (B, hid)
            K = hk.Linear(self.hidden, name="attn_k")(seq)                      # (B, T, hid)
            V = hk.Linear(self.hidden, name="attn_v")(seq)                      # (B, T, hid)
            scores = jnp.einsum('bh,bth->bt', q, K) / jnp.sqrt(float(self.hidden))
            w = jax.nn.softmax(scores, axis=-1)                                 # (B, T)
            enc = jnp.einsum('bt,bth->bh', w, V)                                # (B, hid) attention context
        feats = jnp.concatenate([v_true, enc], axis=-1)
        return v_true, feats

    def report_and_delta(self, belief_state: BeliefState, auction: BanditAuction, payment_history=None):
        """Returns (report (B,G), v_true (B,G), Delta (B,G)). Delta is the pre-bounded logit residual
        (exposed so the NPG driver can add Gaussian noise in logit space and score it)."""
        v_true, feats = self._features(belief_state, payment_history)
        z = self.trunk(feats)
        res_init = (hk.initializers.RandomNormal(self.residual_init_std)
                    if self.residual_init_std > 0 else hk.initializers.Constant(0.0))
        Delta = hk.Linear(self.G, w_init=res_init, b_init=hk.initializers.Constant(0.0))(z)  # (B, G)
        report = _bounded_report(v_true, Delta, auction.types_low, auction.types_high, self.eps)
        return report, v_true, Delta

    def __call__(self, belief_state: BeliefState, auction: BanditAuction, key=None, payment_history=None):
        report, _, _ = self.report_and_delta(belief_state, auction, payment_history)
        return report                                                          # (B, G) — deterministic (pathwise/CMA-ES)


def build_cont_oracle_net(d_priv, d_pub, G, T, B, hidden=128, info="m1", arch="gru",
                          residual_init_std=0.0, stop_grad_truth=True, eps=1e-6):
    """Deterministic policy net: (belief_state, auction, key, payment_history) -> report (B,G)."""
    def f(belief_state, auction, key=None, payment_history=None):
        return ContOracleHead(d_priv, d_pub, G, T, B, hidden, info, arch,
                              residual_init_std, stop_grad_truth, eps)(belief_state, auction, key, payment_history)
    return hk.without_apply_rng(hk.transform(f))


def build_cont_oracle_delta_net(d_priv, d_pub, G, T, B, hidden=128, info="m1", arch="gru",
                                residual_init_std=0.0, stop_grad_truth=True, eps=1e-6):
    """Exposes (v_true, Delta) for the NPG driver (stochastic logit-space Gaussian policy)."""
    def f(belief_state, auction, payment_history=None):
        _, v_true, Delta = ContOracleHead(d_priv, d_pub, G, T, B, hidden, info, arch,
                                          residual_init_std, stop_grad_truth, eps
                                          ).report_and_delta(belief_state, auction, payment_history)
        return v_true, Delta
    return hk.without_apply_rng(hk.transform(f))
