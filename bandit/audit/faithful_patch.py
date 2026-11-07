"""
faithful_patch.py — replace BanditAuction's discrete belief filter with the FAITHFUL one
(opponent-informative, = exact non-clairvoyant one-step posterior) for training/eval/audit.

The shipped get_next_belief_state_discrete conditions the public allocation only into each buyer's
OWN type transition, so its opponent-marginal is frozen at the prior (opponent-uninformative). The
faithful filter conditions each opponent component on that opponent's observed report + public
allocation, giving the correct posterior (verified in verify_faithful_belief.py). We patch the class
method so the entire pipeline (compute_ic_regret rollouts, reopt, eval) uses the faithful belief with
no other code change. This is the permanent fix's stand-in; the fix belongs in core.py.
"""
import os, sys  # --- CDMD path shim (repo-relative, env-overridable) ---
CDMD_ROOT = os.environ.get("CDMD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os, sys
sys.path.insert(0, os.path.join(CDMD_ROOT, "audit"))
from bandit_auction.core import BanditAuction
from belief_faithful import make_faithful_discrete_belief


def _faithful_update(self, belief_state, report, allocation, next_state):
    return make_faithful_discrete_belief(self)(belief_state, report, allocation, next_state)


def apply():
    BanditAuction._get_next_belief_state_discrete_shipped = BanditAuction.get_next_belief_state_discrete
    BanditAuction.get_next_belief_state_discrete = _faithful_update
    print("[faithful_patch] BanditAuction.get_next_belief_state_discrete -> FAITHFUL "
          "(opponent-informative)", flush=True)
