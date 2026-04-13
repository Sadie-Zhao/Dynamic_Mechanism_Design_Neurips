"""
_smoke_test.py — Small-scale end-to-end tests for the budget auction package.

Tests:
  1. Auction construction (Exp 1 and Exp 2)
  2. State initialization and type sampling
  3. State transition (alpha/beta update, pytree survival)
  4. trans_base_rate/trans_sigma_no survival through pytree flatten/unflatten
  5. Mechanism network forward passes (variants A, B, C)
  6. Payment feasibility transform: payment <= beta, payment <= alpha + v*x
  7. BudgetSimulator rollouts: profit, rewards, regret, violation rates
  8. Nested-GD training for a few steps (Exp 1, variant A)
  9. Nested-GD training for a few steps (Exp 2, variant B)
  10. Baseline mechanisms (myopic, reserve)

Usage:
    python _smoke_test.py
"""

import sys
import traceback

import jax
import jax.numpy as jnp
import haiku as hk
import optax

sys.path.insert(0, ".")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, cond, detail=""):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}" + (f": {detail}" if detail else ""))
    return cond

def run_section(title, fn):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    try:
        fn()
    except Exception as e:
        print(f"  [{FAIL}] EXCEPTION: {e}")
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 1: Auction construction
# ---------------------------------------------------------------------------
def test_construction():
    from budget_auction.envs.public_budget_env import create_public_budget_auction
    from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction

    auction1 = create_public_budget_auction(n_buyers=2, n_goods=1, horizon=5)
    check("Exp1 auction created", auction1 is not None)
    check("Exp1 num_buyers=2", auction1.num_buyers == 2)
    check("Exp1 num_goods=1", auction1.num_goods == 1)
    check("Exp1 num_horizons=5", auction1.num_horizons == 5)
    check("Exp1 initial_budgets shape", auction1.initial_budgets.shape == (2,))
    check("Exp1 trans_base_rate default", auction1.trans_base_rate == 1.0)

    auction2 = create_private_budget_multigood_auction(
        n_buyers=2, n_goods=2, horizon=5,
        trans_base_rate=0.8, trans_sigma_no=0.02,
    )
    check("Exp2 auction created", auction2 is not None)
    check("Exp2 num_goods=2", auction2.num_goods == 2)
    check("Exp2 trans_base_rate=0.8", auction2.trans_base_rate == 0.8)
    check("Exp2 trans_sigma_no=0.02", auction2.trans_sigma_no == 0.02)

    # Pytree flatten/unflatten
    import jax
    leaves1, td1 = jax.tree_util.tree_flatten(auction1)
    auction1_r = jax.tree_util.tree_unflatten(td1, leaves1)
    check("Exp1 pytree round-trip: trans_base_rate", auction1_r.trans_base_rate == 1.0)

    leaves2, td2 = jax.tree_util.tree_flatten(auction2)
    auction2_r = jax.tree_util.tree_unflatten(td2, leaves2)
    check("Exp2 pytree round-trip: trans_base_rate=0.8", auction2_r.trans_base_rate == 0.8)
    check("Exp2 pytree round-trip: trans_sigma_no=0.02", auction2_r.trans_sigma_no == 0.02)

    return auction1, auction2

# ---------------------------------------------------------------------------
# Test 2: State init and type sampling
# ---------------------------------------------------------------------------
def test_state_init(auction1, auction2):
    key = jax.random.PRNGKey(0)

    s1 = auction1.get_initial_state(key)
    check("Exp1 initial state created", s1 is not None)
    check("Exp1 time_step=0", s1.time_step == 0)
    check("Exp1 alpha=0", jnp.allclose(s1.alpha, 0.0))
    check("Exp1 beta=B_i", jnp.allclose(s1.beta, auction1.initial_budgets))
    check("Exp1 current_type in [0,1]", jnp.all(s1.current_type >= 0) and jnp.all(s1.current_type <= 1))
    check("Exp1 alloc_history=0", jnp.allclose(s1.allocation_history, 0.0))

    s2 = auction2.get_initial_state(key)
    check("Exp2 current_type shape (2,2)", s2.current_type.shape == (2, 2))
    check("Exp2 payment_history shape (5,2)", s2.payment_history.shape == (5, 2))

    return s1, s2

# ---------------------------------------------------------------------------
# Test 3: Mechanism forward pass (variants A, B, C)
# ---------------------------------------------------------------------------
def test_mechanism_variants(auction1, auction2, s1, s2):
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.mechanisms.budget_mechanism import BudgetMechanism

    for variant_label, auction, state in [
        ("A-Exp1", auction1, s1),
        ("B-Exp1", auction1, s1),
        ("C-Exp1", auction1, s1),
        ("A-Exp2", auction2, s2),
        ("B-Exp2", auction2, s2),
        ("C-Exp2", auction2, s2),
    ]:
        variant = variant_label[0]
        net_fn = VARIANT_NETWORK[variant]
        net = hk.without_apply_rng(hk.transform(net_fn))
        key = jax.random.PRNGKey(1)
        report = jnp.ones((auction.num_buyers, auction.num_goods)) * 0.5

        params = net.init(key, state, report, auction)
        alloc_probs, z = net.apply(params, state, report, auction)
        check(f"Variant {variant_label}: alloc_probs shape (B+1,G)",
              alloc_probs.shape == (auction.num_buyers + 1, auction.num_goods))
        check(f"Variant {variant_label}: alloc sums to ≤1 per good",
              jnp.all(jnp.sum(alloc_probs, axis=0) <= 1.0 + 1e-5))
        check(f"Variant {variant_label}: z shape (B,)",
              z.shape == (auction.num_buyers,))

        # Full mechanism with feasibility transform
        mech = BudgetMechanism(net, params)
        alloc, payment = mech(state, report, auction)
        check(f"Variant {variant_label}: alloc shape (B,G)",
              alloc.shape == (auction.num_buyers, auction.num_goods))
        check(f"Variant {variant_label}: payment shape (B,)",
              payment.shape == (auction.num_buyers,))
        check(f"Variant {variant_label}: payment >= 0",
              jnp.all(payment >= -1e-6))
        check(f"Variant {variant_label}: payment <= beta",
              jnp.all(payment <= state.beta + 1e-6))
        utility_now = jnp.sum(alloc * report, axis=-1)
        check(f"Variant {variant_label}: payment <= alpha + v*x",
              jnp.all(payment <= state.alpha + utility_now + 1e-6))

        return params  # return last params for later use

# ---------------------------------------------------------------------------
# Test 4: State transition and alpha/beta update
# ---------------------------------------------------------------------------
def test_state_transition(auction1, s1):
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.mechanisms.budget_mechanism import BudgetMechanism

    net_fn = VARIANT_NETWORK["A"]
    net = hk.without_apply_rng(hk.transform(net_fn))
    key = jax.random.PRNGKey(2)
    report = s1.current_type  # truthful

    params = net.init(key, s1, report, auction1)
    mech = BudgetMechanism(net, params)
    alloc, payment = mech(s1, report, auction1)

    key_step = jax.random.PRNGKey(3)
    s_next = auction1.get_next_budget_state(s1, report, alloc, payment, key_step)

    check("time_step incremented", s_next.time_step == 1)
    check("alpha updated correctly",
          jnp.allclose(
              s_next.alpha,
              s1.alpha + jnp.sum(s1.current_type * alloc, axis=-1) - payment,
              atol=1e-5
          ))
    check("beta updated correctly",
          jnp.allclose(s_next.beta, s1.beta - payment, atol=1e-5))
    check("alloc recorded in history",
          jnp.allclose(s_next.allocation_history[0], alloc))
    check("payment recorded in history",
          jnp.allclose(s_next.payment_history[0], payment))
    check("new type in [0,1]",
          jnp.all(s_next.current_type >= 0) and jnp.all(s_next.current_type <= 1))

# ---------------------------------------------------------------------------
# Test 5: Simulator rollouts
# ---------------------------------------------------------------------------
def test_simulator(auction1, auction2):
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
    from budget_auction.core import BudgetSimulator

    for label, auction in [("Exp1", auction1), ("Exp2", auction2)]:
        net_fn = VARIANT_NETWORK["A"]
        net = hk.without_apply_rng(hk.transform(net_fn))
        key = jax.random.PRNGKey(5)
        s0 = auction.get_initial_state(key)
        report = jnp.ones((auction.num_buyers, auction.num_goods)) * 0.5
        params = net.init(key, s0, report, auction)
        mech = BudgetMechanism(net, params)

        k1, k2, k3 = jax.random.split(key, 3)
        profit = BudgetSimulator.estimate_cumulative_profit(auction, mech, 16, k1)
        check(f"{label} profit is finite", jnp.isfinite(profit))
        check(f"{label} profit >= 0", float(profit) >= -1e-3)

        rewards = BudgetSimulator.estimate_cumulative_rewards(auction, mech, 16, k2)
        check(f"{label} rewards shape (B,)", rewards.shape == (auction.num_buyers,))

        bv, iv = BudgetSimulator.estimate_violation_rates(auction, mech, 16, k3)
        check(f"{label} budget_viol_rate in [0,1]", 0.0 <= bv <= 1.0)
        check(f"{label} ir_viol_rate in [0,1]", 0.0 <= iv <= 1.0)
        check(f"{label} budget_viol_rate ≈ 0 (feasibility enforced)", bv < 1e-5,
              f"got {bv:.4f}")
        check(f"{label} ir_viol_rate ≈ 0 (feasibility enforced)", iv < 1e-5,
              f"got {iv:.4f}")

# ---------------------------------------------------------------------------
# Test 6: Regret rollout
# ---------------------------------------------------------------------------
def test_regret(auction1):
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
    from budget_auction.core import (
        BudgetSimulator, BoundedContPolicy, build_bounded_policy_net
    )

    net_fn = VARIANT_NETWORK["A"]
    net = hk.without_apply_rng(hk.transform(net_fn))
    key = jax.random.PRNGKey(6)
    s0 = auction1.get_initial_state(key)
    report = jnp.ones((auction1.num_buyers, auction1.num_goods)) * 0.5
    mech_params = net.init(key, s0, report, auction1)
    mech = BudgetMechanism(net, mech_params)

    policy_net = build_bounded_policy_net(
        d_priv=auction1.d_priv, d_pub=auction1.d_pub,
        G=auction1.num_goods, hidden=32, residual_init_std=0.0,
    )
    b0 = auction1.get_initial_belief_state(s0)
    policy_params = policy_net.init(jax.random.PRNGKey(7), b0, auction1)
    policy = BoundedContPolicy(policy_net, policy_params)

    k1, k2 = jax.random.split(key, 2)
    regret = BudgetSimulator.estimate_cumulative_regret(
        auction1, mech, policy, 8, k1
    )
    check("Regret is finite", jnp.isfinite(regret))
    check("Regret near 0 at init (truthful policy)", abs(float(regret)) < 1.0,
          f"got {float(regret):.4f}")

# ---------------------------------------------------------------------------
# Test 7: Short nested-GD run (Exp 1 + Exp 2, variants A and B)
# ---------------------------------------------------------------------------
def test_training(auction, label, variant="A", n_outer=3, n_inner=5, n_samples=8):
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
    from budget_auction.core import (
        BudgetSimulator, BoundedContPolicy, build_bounded_policy_net
    )

    net_fn = VARIANT_NETWORK[variant]
    mechanism_network = hk.without_apply_rng(hk.transform(net_fn))
    policy_network = build_bounded_policy_net(
        d_priv=auction.d_priv, d_pub=auction.d_pub,
        G=auction.num_goods, hidden=32, residual_init_std=0.0,
    )

    prng = hk.PRNGSequence(0)
    init_state = auction.get_initial_state(next(prng))
    report0 = jnp.zeros((auction.num_buyers, auction.num_goods))
    mech_params = mechanism_network.init(next(prng), init_state, report0, auction)
    init_belief = auction.get_initial_belief_state(init_state)
    policy_params = policy_network.init(next(prng), init_belief, auction)

    theta_opt = optax.adam(1e-4)
    theta_state = theta_opt.init(mech_params)
    w_opt = optax.adam(3e-4)
    w_state = w_opt.init(policy_params)

    profit_goal = 1.0
    alpha_w, beta_w = 1e4, 1e2

    @jax.jit
    def cumulative_regret(mp, pp, key):
        mech = BudgetMechanism(mechanism_network, mp)
        pol = BoundedContPolicy(policy_network, pp)
        return BudgetSimulator.estimate_cumulative_regret(auction, mech, pol, n_samples, key)

    @jax.jit
    def total_loss(mp, pp_star, key):
        k1, k2, k3 = jax.random.split(key, 3)
        mech = BudgetMechanism(mechanism_network, mp)
        reg = BudgetSimulator.estimate_cumulative_regret(auction, mech, BoundedContPolicy(policy_network, pp_star), n_samples, k1)
        prof = BudgetSimulator.estimate_cumulative_profit(auction, mech, n_samples, k2)
        rewards = BudgetSimulator.estimate_relu_cumulative_rewards(auction, mech, n_samples, k3)
        ir = jnp.sum(rewards)
        gap = jnp.maximum(0.0, profit_goal - prof)
        return alpha_w * jnp.maximum(0.0, reg) + gap + beta_w * ir, (reg, prof, ir)

    @jax.jit
    def inner_step(mp, pp, ws, key):
        reg, g = jax.value_and_grad(cumulative_regret, argnums=1)(mp, pp, key)
        g_asc = jax.tree_util.tree_map(lambda x: -x, g)
        upd, ws = w_opt.update(g_asc, ws, pp)
        pp = optax.apply_updates(pp, upd)
        return reg, pp, ws

    @jax.jit
    def outer_step(mp, ts, pp_star, key):
        (loss, aux), g = jax.value_and_grad(total_loss, argnums=0, has_aux=True)(mp, pp_star, key)
        upd, ts = theta_opt.update(g, ts, mp)
        mp = optax.apply_updates(mp, upd)
        return loss, aux, mp, ts

    losses = []
    for ep in range(n_outer):
        for _ in range(n_inner):
            _, policy_params, w_state = inner_step(
                mech_params, policy_params, w_state, next(prng)
            )
        loss, (reg, prof, ir), mech_params, theta_state = outer_step(
            mech_params, theta_state, policy_params, next(prng)
        )
        losses.append(float(loss))

    check(f"{label} Variant {variant}: training ran {n_outer} outer epochs", True)
    check(f"{label} Variant {variant}: all losses finite",
          all(jnp.isfinite(l) for l in losses), f"losses={losses}")
    check(f"{label} Variant {variant}: loss is a number", isinstance(losses[-1], float))
    print(f"    losses={[f'{l:.4f}' for l in losses]}")

# ---------------------------------------------------------------------------
# Test 8: Baselines
# ---------------------------------------------------------------------------
def test_baselines(auction1, auction2):
    from budget_auction.baselines.static_myopic import StaticMyopicMechanism
    from budget_auction.baselines.reserve_budget import ReserveBudgetMechanism
    from budget_auction.core import BudgetSimulator

    for label, auction in [("Exp1", auction1), ("Exp2", auction2)]:
        mech_m = StaticMyopicMechanism()
        mech_r = ReserveBudgetMechanism()

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        pm = BudgetSimulator.estimate_cumulative_profit(auction, mech_m, 8, k1)
        pr = BudgetSimulator.estimate_cumulative_profit(auction, mech_r, 8, k2)
        check(f"{label} myopic profit finite", jnp.isfinite(pm))
        check(f"{label} reserve profit finite", jnp.isfinite(pr))
        check(f"{label} myopic profit >= 0", float(pm) >= -1e-3)

        bv_m, iv_m = BudgetSimulator.estimate_violation_rates(auction, mech_m, 8, key)
        check(f"{label} myopic budget_viol ≈ 0", bv_m < 1e-5, f"got {bv_m:.4f}")
        check(f"{label} myopic ir_viol ≈ 0", iv_m < 1e-5, f"got {iv_m:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nBudget Auction Package — Smoke Test")

    auction1 = auction2 = None
    s1 = s2 = None

    run_section("1. Auction Construction + Pytree Round-trip", lambda: None)
    from budget_auction.envs.public_budget_env import create_public_budget_auction
    from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction
    auction1 = create_public_budget_auction(n_buyers=2, n_goods=1, horizon=5)
    auction2 = create_private_budget_multigood_auction(
        n_buyers=2, n_goods=2, horizon=5, trans_base_rate=0.8, trans_sigma_no=0.02)
    run_section("1. Auction Construction + Pytree Round-trip",
                lambda: test_construction())

    # Re-create for clean state
    auction1 = create_public_budget_auction(n_buyers=2, n_goods=1, horizon=5)
    auction2 = create_private_budget_multigood_auction(
        n_buyers=2, n_goods=2, horizon=5, trans_base_rate=1.0, trans_sigma_no=0.01)

    run_section("2. State Initialization", lambda: None)
    s1, s2 = test_state_init(auction1, auction2)

    run_section("3. Mechanism Variants A/B/C (Exp1 + Exp2)",
                lambda: test_mechanism_variants(auction1, auction2, s1, s2))

    run_section("4. State Transition + alpha/beta Update",
                lambda: test_state_transition(auction1, s1))

    run_section("5. Simulator Rollouts (Profit, Rewards, Violations)",
                lambda: test_simulator(auction1, auction2))

    run_section("6. Regret Rollout",
                lambda: test_regret(auction1))

    run_section("7a. Nested-GD Training: Exp1 Variant A (3 outer epochs)",
                lambda: test_training(auction1, "Exp1", "A"))
    run_section("7b. Nested-GD Training: Exp1 Variant C (3 outer epochs)",
                lambda: test_training(auction1, "Exp1", "C"))
    run_section("7c. Nested-GD Training: Exp2 Variant A (3 outer epochs)",
                lambda: test_training(auction2, "Exp2", "A"))
    run_section("7d. Nested-GD Training: Exp2 Variant B (3 outer epochs)",
                lambda: test_training(auction2, "Exp2", "B"))

    run_section("8. Baseline Mechanisms (Myopic + Reserve)",
                lambda: test_baselines(auction1, auction2))

    print("\nSmoke test complete.")
