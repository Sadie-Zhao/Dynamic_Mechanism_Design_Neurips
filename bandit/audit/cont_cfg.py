"""
cont_cfg.py — shared config loader + eval seed for the continuous ladder drivers.

FIX (audit-critical): the continuous trainer (run_acceptance/run_penalty) writes the env config to
config.yaml in the run_dir, NOT into summary.json. The drivers previously read summary.json["cfg"] (which
does not exist), silently fell back to create_cont_*_auction constructor DEFAULTS (type_std=0.1 vs the
configs' 0.8) and never read use_attn_mechanism (so attn checkpoints got the wrong mechanism net). This
loader reads config.yaml (fallback summary.json["cfg"]), extracts the env kwargs + use_attn_mechanism, and
returns the list of MISSING critical keys so the driver can WARN loudly (never a silent default).

EVAL_SEED: a fixed held-out seed shared by all three oracle drivers so estimate_regret_det draws the SAME
init states for every oracle -> paired, strictly-comparable per-buyer regrets across the algorithm axis.
"""
import os, json

EVAL_SEED = 20260726

_ENV_KEYS = ("n_buyers", "n_goods", "horizon", "discount", "type_mean", "type_std",
             "types_low", "types_high", "allocation_cost", "sigma_no", "trans_lambda", "d_pub", "d_priv")
_CRITICAL = ("type_std", "sigma_no", "trans_lambda")


def load_cont_env(run_dir, seed):
    """Returns (ekw, use_attn_mechanism, cfg, missing_critical_keys)."""
    cfg = {}
    cy = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cy):
        import yaml
        cfg = yaml.safe_load(open(cy)) or {}
    if not cfg:
        try:
            summ = json.load(open(os.path.join(run_dir, "summary.json")))
            cfg = summ.get("cfg", {}) or {}
        except Exception:
            cfg = {}
    ekw = dict(seed=seed)
    for k in _ENV_KEYS:
        if k in cfg:
            ekw[k] = cfg[k]
    use_attn = bool(cfg.get("use_attn_mechanism", False))
    missing = [k for k in _CRITICAL if k not in cfg]
    return ekw, use_attn, cfg, missing
