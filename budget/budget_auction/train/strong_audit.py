"""Independent full-information-history deviation search for budget mechanisms.

Reuses the rebuttal's score-function / Woodbury natural-gradient method, with
budget-state dynamics and the paper's observation boundary. Both stochastic and
deterministic policies are considered. Selection and final testing use disjoint
random streams; all selected policy parameters are returned for preservation.
"""

import math
import haiku as hk
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import struct

from budget_auction.core import BudgetSimulator, build_bounded_policy_net
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.train.binary_search import reopt_cumulative_ic, _finite_reopt_state
from budget_auction.train.evaluate import summarize_deviation_gains


@struct.dataclass
class AuditPolicy:
    network: object = struct.field(pytree_node=False)
    params: tuple
    sigmas: jnp.ndarray
    modes: tuple = struct.field(pytree_node=False)

    def __call__(self, belief, auction, key):
        truth = belief.type_history[belief.time_step]
        noise = jax.random.normal(key, truth.shape)
        rows = []
        for i, mode in enumerate(self.modes):
            if mode == "truth":
                row = truth[i]
            elif mode == "low":
                row = auction.types_low[i]
            elif mode == "high":
                row = auction.types_high[i]
            elif mode == "half":
                row = auction.types_low[i] + 0.5*(truth[i]-auction.types_low[i])
            else:
                # Haiku's Transformed is itself a namedtuple; distinguish it
                # from the tuple of independently selected buyer networks.
                network = self.network if hasattr(self.network, "apply") else self.network[i]
                mean_report = network.apply(self.params[i], belief, auction)[i]
                width = auction.types_high[i]-auction.types_low[i]
                p = jnp.clip((mean_report-auction.types_low[i])/width, 1e-6, 1-1e-6)
                logit = jnp.log(p)-jnp.log1p(-p)
                stochastic = auction.types_low[i]+width*jax.nn.sigmoid(logit+self.sigmas[i]*noise[i])
                row = jnp.where(self.sigmas[i] == 0, mean_report, stochastic)
            rows.append(row)
        return jnp.stack(rows)


def natural_direction(scores, returns, damping=0.1):
    """Solve (lambda I + S' S/N) d = S' advantage/N in sample space."""
    n = scores.shape[0]
    # Leave-one-out mean baseline: avoid the (N-1)/N bias of centering with self.
    advantage = (returns-jnp.mean(returns)) * n/max(1, n-1)
    gradient = scores.T @ (advantage/n)
    matrix = n*jnp.eye(n) + (scores @ scores.T)/damping
    correction = jnp.linalg.solve(matrix, scores @ gradient)
    direction = gradient/damping - scores.T @ correction/damping**2
    return gradient, direction


def make_npg_step(auction, mechanism, network, initial_params, buyer,
                  episodes=128, sigma=0.3, damping=0.1, kl=0.02):
    """Pure score-function optimization: sampled actions and returns detached."""
    if episodes < 2 or sigma <= 0 or damping <= 0 or kl <= 0:
        raise ValueError("NPG needs >=2 episodes and positive sigma, damping and KL limit")
    _, unpack = ravel_pytree(initial_params)
    width = auction.types_high-auction.types_low

    def trajectory(flat, state, key):
        params = unpack(flat)
        belief = auction.get_initial_belief_state(state)
        def step(t, carry):
            gain, logprob, dev, obs, truth, k = carry
            k, action_key, transition_key = jax.random.split(k, 3)
            mean_report = network.apply(params, obs, auction)
            p = jnp.clip((mean_report-auction.types_low)/width, 1e-6, 1-1e-6)
            mean = jnp.log(p)-jnp.log1p(-p)
            action = jax.lax.stop_gradient(mean[buyer]) + sigma*jax.random.normal(
                action_key, (auction.num_goods,))
            logprob += jnp.sum(-0.5*((action-mean[buyer])/sigma)**2
                               -jnp.log(sigma)-0.5*jnp.log(2*jnp.pi))
            report = dev.current_type.at[buyer].set(
                auction.types_low[buyer]+width[buyer]*jax.nn.sigmoid(action))
            allocation, payment = mechanism(dev, report, auction)
            truth_alloc, truth_pay = mechanism(truth, truth.current_type, auction)
            difference = (BudgetSimulator.get_rewards(auction, dev, allocation, payment)[buyer]
                          - BudgetSimulator.get_rewards(auction, truth, truth_alloc, truth_pay)[buyer])
            gain += auction.discount**t * difference
            next_dev = auction.get_next_budget_state(dev, report, allocation, payment, transition_key)
            next_obs = auction.get_next_belief_state(obs, report, allocation, next_dev)
            next_truth = auction.get_next_budget_state(truth, truth.current_type,
                                                       truth_alloc, truth_pay, transition_key)
            return gain, logprob, next_dev, next_obs, next_truth, k
        gain, logprob, *_ = jax.lax.fori_loop(0, auction.num_horizons, step,
            (0.0, 0.0, state, belief, state, key))
        return logprob, jax.lax.stop_gradient(gain)

    @jax.jit
    def update(flat, key):
        init_key, roll_key = jax.random.split(key)
        states = BudgetSimulator.sample_init_states(auction, episodes, init_key)
        keys = jax.random.split(roll_key, episodes)
        (_, gains), scores = jax.vmap(lambda s,k: jax.value_and_grad(
            trajectory, has_aux=True)(flat, s, k))(states, keys)
        gradient, direction = natural_direction(scores, gains, damping)
        scale = jnp.minimum(1.0, jnp.sqrt(2*kl/jnp.maximum(gradient @ direction, 1e-12)))
        return flat+scale*direction, jnp.mean(gains)
    return update, unpack


def audit_mechanism(auction, mechanism_network, mechanism_params, policy_network,
                    seed=4201, pathwise_steps=600, pathwise_restarts=3,
                    npg_steps=300, npg_restarts=3, episodes=128,
                    validation_samples=2048, test_batches=32, test_samples=256,
                    sigma=0.3, eval_every=25, ic_threshold=0.05,
                    resume_state=None, checkpoint_callback=None):
    """Select one strongest policy per buyer on validation, then freeze and test.

    The returned UCB addresses evaluation sampling error only. A pass is evidence
    against this declared search family, not proof that the global best response
    was found. All candidate selection occurs before any final test evaluation.
    """
    mechanism = BudgetMechanism(mechanism_network, mechanism_params)
    key = jax.random.PRNGKey(seed)
    validation_key = jax.random.fold_in(key, 11001)
    test_key = jax.random.fold_in(key, 22002)
    init_state = auction.get_initial_state(jax.random.fold_in(key, 33003))
    belief = auction.get_initial_belief_state(init_state)
    initial = policy_network.init(jax.random.fold_in(key, 44004), belief, auction)
    selected = [initial for _ in range(auction.num_buyers)]
    modes = ["truth"] * auction.num_buyers
    sigmas = [0.0] * auction.num_buyers
    best = [0.0] * auction.num_buyers
    winners = ["truth"] * auction.num_buyers
    history = []
    controls = {"seed": seed, "pathwise_steps": pathwise_steps,
        "pathwise_restarts": pathwise_restarts, "npg_steps": npg_steps,
        "npg_restarts": npg_restarts, "episodes": episodes,
        "validation_samples": validation_samples, "test_batches": test_batches,
        "test_samples": test_samples, "sigma": sigma, "eval_every": eval_every,
        "ic_threshold": ic_threshold, "buyers": auction.num_buyers}
    if min(pathwise_steps, npg_steps) < 0 or npg_restarts < 1 or eval_every < 1:
        raise ValueError("Audit requires nonnegative steps and positive restarts/eval interval")
    state = {"schema": 1, "controls": controls, "phase": "simple", "pathwise_state": None,
             "pathwise": None, "cursor": (0, 0, 0), "flat": None, "train_key": None,
             "npg_terminal_states": [], "family_best": {}}
    if resume_state is not None:
        if resume_state.get("schema") != 1 or resume_state["controls"] != controls:
            raise ValueError("Audit checkpoint controls differ")
        state = dict(resume_state)
        selected, modes, sigmas = list(state["selected"]), list(state["modes"]), list(state["sigmas"])
        best, winners, history = list(state["best"]), list(state["winners"]), list(state["history"])
        state["npg_terminal_states"] = list(state["npg_terminal_states"])
        if state["phase"] == "complete":
            return state["summary"], AuditPolicy(policy_network, tuple(selected), jnp.asarray(sigmas), tuple(modes))

    def save():
        if checkpoint_callback is not None:
            checkpoint_callback({**state, "selected": list(selected), "modes": list(modes),
                "sigmas": list(sigmas), "best": list(best), "winners": list(winners),
                "history": list(history), "npg_terminal_states": list(state["npg_terminal_states"])})

    def consider(buyer, params, mode, noise_sigma, label):
        candidate = AuditPolicy(policy_network, tuple([params]*auction.num_buyers),
                                jnp.full((auction.num_buyers,), noise_sigma),
                                tuple([mode]*auction.num_buyers))
        gain = float(BudgetSimulator.estimate_cumulative_deviation_gains(
            auction, mechanism, candidate, validation_samples, validation_key)[buyer])
        if not math.isfinite(gain):
            raise FloatingPointError(f"Nonfinite audit validation: buyer={buyer}, candidate={label}")
        history.append({"buyer": buyer, "candidate": label, "validation_gain": gain})
        family = ("pathwise" if label == "pathwise" else
                  "npg_stochastic" if label.endswith("_stochastic") else
                  "npg_mean" if label.endswith("_mean") else None)
        if family is not None:
            index = f"{buyer}/{family}"
            previous = state["family_best"].get(index)
            if previous is None or gain > previous["validation_gain"]:
                state["family_best"][index] = {"buyer": buyer, "family": family,
                    "params": params, "mode": mode, "sigma": noise_sigma,
                    "validation_gain": gain, "label": label}
        if gain > best[buyer]:
            best[buyer], selected[buyer] = gain, params
            modes[buyer], sigmas[buyer], winners[buyer] = mode, noise_sigma, label

    if state["phase"] == "simple":
        for buyer in range(auction.num_buyers):
            for mode in ("low", "high", "half"):
                consider(buyer, initial, mode, 0.0, mode)
        state["phase"] = "pathwise"
        save()

    if state["phase"] == "pathwise":
        def save_pathwise(progress):
            state["pathwise_state"] = progress
            save()
        _, pathwise = reopt_cumulative_ic(auction, mechanism_network, mechanism_params,
            policy_network, num_reopt_steps=pathwise_steps, num_reopt_restarts=pathwise_restarts,
            num_samples=episodes, seed=seed+100, resume_state=state["pathwise_state"],
            checkpoint_callback=save_pathwise if checkpoint_callback is not None else None)
        state["pathwise"] = pathwise
        state["phase"] = "npg"
        save()
    pathwise = state["pathwise"]
    start_buyer, start_restart, start_step = state["cursor"]
    for buyer in range(start_buyer, auction.num_buyers):
        continuing_buyer = buyer == start_buyer
        if not continuing_buyer or (start_restart == 0 and state["flat"] is None):
            consider(buyer, pathwise[buyer], "network", 0.0, "pathwise")
        if npg_steps == 0:
            state.update(cursor=(buyer+1, 0, 0), flat=None, train_key=None)
            save()
            continue
        update, unpack = make_npg_step(auction, mechanism, policy_network, initial,
                                      buyer, episodes=episodes, sigma=sigma)
        for restart in range(start_restart if continuing_buyer else 0, npg_restarts):
            if continuing_buyer and restart == start_restart and state["flat"] is not None:
                flat, train_key, first_step = state["flat"], state["train_key"], start_step
            else:
                train_key = jax.random.fold_in(key, 55005+buyer*100+restart)
                # Include a warm start from the independent pathwise best response.
                params = pathwise[buyer] if restart == 0 else policy_network.init(train_key, belief, auction)
                flat, _ = ravel_pytree(params)
                first_step = 0
            for step in range(first_step, npg_steps):
                train_key, step_key = jax.random.split(train_key)
                flat, _ = update(flat, step_key)
                if (step+1) % eval_every == 0 or step+1 == npg_steps:
                    if not bool(_finite_reopt_state(flat, ())):
                        raise FloatingPointError(f"Nonfinite NPG state: buyer={buyer}, restart={restart}, step={step+1}")
                    params = unpack(flat)
                    label = f"npg_restart{restart}_step{step+1}"
                    consider(buyer, params, "network", 0.0, label+"_mean")
                    consider(buyer, params, "network", sigma, label+"_stochastic")
                    state.update(cursor=(buyer, restart, step+1), flat=flat, train_key=train_key)
                    save()
            state["npg_terminal_states"].append({"buyer": buyer, "restart": restart,
                "steps": npg_steps, "flat": flat, "train_key": train_key})
            state.update(cursor=(buyer, restart+1, 0), flat=None, train_key=None)
            save()
        state.update(cursor=(buyer+1, 0, 0), flat=None, train_key=None)
        save()

    policy = AuditPolicy(policy_network, tuple(selected), jnp.asarray(sigmas), tuple(modes))
    gains = jnp.stack([BudgetSimulator.estimate_cumulative_deviation_gains(
        auction, mechanism, policy, test_samples, k) for k in jax.random.split(test_key, test_batches)])
    summary = summarize_deviation_gains(gains)
    summary.update({
        "selected_oracles": winners, "selected_modes": modes, "selected_sigmas": sigmas,
        "validation_gain_per_buyer": best, "selection_history": history,
        "test_gains_by_batch": gains.tolist(), "seed": seed,
        "test_batches": test_batches, "test_samples_per_batch": test_samples,
        "ic_threshold_sum": ic_threshold,
        "passes_empirical_ic_audit": summary["ic_sum_sampling_ucb95"] <= ic_threshold,
        "search_config": {"pathwise_steps": pathwise_steps, "pathwise_restarts": pathwise_restarts,
            "npg_steps": npg_steps, "npg_restarts": npg_restarts, "episodes": episodes,
            "validation_samples": validation_samples, "sigma": sigma, "eval_every": eval_every},
    })
    state.update(phase="complete", summary=summary)
    save()
    return summary, policy


def audit_capacity_ladder(auction, mechanism_network, mechanism_params,
                          hidden_sizes=(128, 256), seed=4201,
                          resume_state=None, checkpoint_callback=None, **search_config):
    """Search each width, select a width per buyer, then test on fresh samples.

    Each rung's evaluation is used ONLY for width selection. A separate stream
    evaluates the combined frozen policy, so capacity selection does not reuse
    the reported final samples. Truthful reporting remains a candidate.
    """
    if not hidden_sizes or any(size < 1 for size in hidden_sizes):
        raise ValueError("At least one positive policy width is required")
    rungs, policies, networks = [], [], []
    controls = {"hidden_sizes": list(hidden_sizes), "seed": seed, "search_config": search_config}
    state = {"schema": 1, "controls": controls, "completed_rungs": [], "active_rung": None}
    if resume_state is not None:
        if resume_state.get("schema") != 1 or resume_state["controls"] != controls:
            raise ValueError("Capacity checkpoint controls differ")
        state = {**resume_state, "completed_rungs": list(resume_state["completed_rungs"])}
    def save():
        if checkpoint_callback is not None:
            checkpoint_callback({**state, "completed_rungs": list(state["completed_rungs"])})
    for index, hidden in enumerate(hidden_sizes):
        network = build_bounded_policy_net(auction.d_priv, auction.d_pub,
                                            auction.num_goods, hidden=hidden)
        if index < len(state["completed_rungs"]):
            cached = state["completed_rungs"][index]
            result = dict(cached["result"])
            policy = AuditPolicy(network, **cached["policy"])
        else:
            def save_rung(progress):
                state["active_rung"] = progress
                save()
            result, policy = audit_mechanism(auction, mechanism_network, mechanism_params,
                network, seed=seed+7919*(index+1), resume_state=state["active_rung"],
                checkpoint_callback=save_rung if checkpoint_callback is not None else None, **search_config)
            state["completed_rungs"].append({"result": result,
                "policy": {"params": policy.params, "sigmas": policy.sigmas, "modes": policy.modes},
                "search_state": state["active_rung"]})
            state["active_rung"] = None
            save()
        result["role"] = "capacity_selection_validation"
        rungs.append(result)
        policies.append(policy)
        networks.append(network)
    chosen, params, sigmas, modes, widths = [], [], [], [], []
    for buyer in range(auction.num_buyers):
        index = max(range(len(rungs)), key=lambda j: rungs[j]["deviation_gain_per_buyer"][buyer])
        positive = rungs[index]["deviation_gain_per_buyer"][buyer] > 0
        chosen.append(index)
        params.append(policies[index].params[buyer])
        sigmas.append(float(policies[index].sigmas[buyer]) if positive else 0.0)
        modes.append(policies[index].modes[buyer] if positive else "truth")
        widths.append(hidden_sizes[index])
    policy = AuditPolicy(tuple(networks[j] for j in chosen), tuple(params),
                         jnp.asarray(sigmas), tuple(modes))
    batches = search_config.get("test_batches", 32)
    samples = search_config.get("test_samples", 256)
    test_key = jax.random.fold_in(jax.random.PRNGKey(seed), 66006)
    mechanism = BudgetMechanism(mechanism_network, mechanism_params)
    gains = jnp.stack([BudgetSimulator.estimate_cumulative_deviation_gains(
        auction, mechanism, policy, samples, k) for k in jax.random.split(test_key, batches)])
    summary = summarize_deviation_gains(gains)
    threshold = search_config.get("ic_threshold", .05)
    summary.update({
        "capacity_rungs": rungs, "hidden_sizes": list(hidden_sizes),
        "selected_hidden_per_buyer": widths, "selected_modes": modes,
        "selected_sigmas": sigmas, "selected_rung_per_buyer": chosen,
        "test_gains_by_batch": gains.tolist(), "seed": seed,
        "test_batches": batches, "test_samples_per_batch": samples,
        "ic_threshold_sum": threshold,
        "passes_empirical_ic_audit": summary["ic_sum_sampling_ucb95"] <= threshold,
        "search_config": search_config,
        "role": "independent_test_after_capacity_selection",
    })
    return summary, policy
