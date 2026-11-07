"""Resumable per-buyer pathwise checks of frozen continuous mechanisms.

Preserves the original GRU/wide/MLP forward policy classes but retains the
endogenous-type derivative. Attention is deliberately unsupported here because
its additional query detach requires a separate correction. This is an added
empirical audit, not a replacement for the preserved rebuttal ladder.
"""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import pickle
import sys
import time
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"audit"))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from bandit_auction.core import AuctionSimulator
from bandit_auction.envs.cont_single import create_cont_single_auction
from bandit_auction.envs.cont_multi import create_cont_multi_auction
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network
from cont_cfg import load_cont_env
from cont_oracle import build_cont_oracle_net


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def atomic_pickle(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    temp.replace(path)


def sample_buyer_gain(auction, state, mechanism, policy_apply, params, key, buyer, mode="network"):
    """One unilateral deviation and its truthful counterfactual, with shared noise."""
    belief = auction.get_initial_belief_state(state)
    payments = jnp.zeros((auction.num_horizons, auction.num_buyers))
    def step(t, carry):
        gain, deviating, observation, past_payments, truthful, rng = carry
        rng, policy_key, transition_key = jax.random.split(rng, 3)
        if mode == "network":
            own_report = policy_apply(params, observation, auction, policy_key, past_payments)[buyer]
        elif mode == "truth":
            own_report = deviating.current_type[buyer]
        elif mode == "low":
            own_report = auction.types_low[buyer]
        elif mode == "high":
            own_report = auction.types_high[buyer]
        elif mode == "half":
            own_report = auction.types_low[buyer]+.5*(deviating.current_type[buyer]-auction.types_low[buyer])
        else:
            raise ValueError(f"Unknown reporting mode: {mode}")
        report = deviating.current_type.at[buyer].set(own_report)
        allocation, payment = mechanism(deviating, report, auction)
        truth_allocation, truth_payment = mechanism(truthful, truthful.current_type, auction)
        difference = (AuctionSimulator.get_rewards(auction, deviating, allocation, payment)[buyer]
                      - AuctionSimulator.get_rewards(auction, truthful, truth_allocation, truth_payment)[buyer])
        next_deviation = auction.get_next_state(deviating, report, allocation, transition_key)
        next_observation = auction.get_next_belief_state_continuous(
            observation, report, allocation, next_deviation)
        next_truth = auction.get_next_state(truthful, truthful.current_type, truth_allocation, transition_key)
        return (gain+auction.discount**t*difference, next_deviation, next_observation,
                past_payments.at[t].set(payment), next_truth, rng)
    return jax.lax.fori_loop(0, auction.num_horizons, step,
        (jnp.float32(0), state, belief, payments, state, key))[0]


@functools.partial(jax.jit, static_argnames=("policy_apply", "num_samples", "buyer", "mode"))
def estimate_buyer_gain(auction, mechanism, policy_apply, params, num_samples, key, buyer, mode="network"):
    initial_key, rollout_key = jax.random.split(key)
    states = AuctionSimulator.sample_init_states(auction, num_samples, initial_key)
    keys = jax.random.split(rollout_key, num_samples)
    return jnp.mean(jax.vmap(lambda state, rng: sample_buyer_gain(
        auction, state, mechanism, policy_apply, params, rng, buyer, mode))(states, keys))


def source_files():
    return sorted((ROOT/"bandit_auction").rglob("*.py"))+[
        ROOT/"audit/cont_cfg.py", ROOT/"audit/cont_oracle.py", Path(__file__).resolve()]


def run_check(config, output, stop_after_updates=None):
    if config["arch"] not in ("gru", "wide", "mlp"):
        raise ValueError("Attention query correction is outside this isolated check")
    if config["steps"] < 0 or config["restarts"] < 1 or config["eval_every"] < 1:
        raise ValueError("Invalid search budget")
    if config["final_batches"] < 2 or min(config["samples"], config["validation_samples"], config["final_samples"]) < 1:
        raise ValueError("Invalid evaluation sample budget")
    folder = ROOT/f"mechanisms/continuous/cont_{config['setting']}/ckpt_best"
    environment, attention, _, missing = load_cont_env(str(folder), 42)
    if attention or missing:
        raise ValueError("Expected the complete saved baseline mechanism configuration")
    files = source_files()
    identity = {"config": config, "environment": environment,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "inputs_sha256": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ("mechanism_params.pkl", "config.yaml")},
        "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "dm-haiku", "flax", "optax", "numpy")}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"identity.json").exists():
        if json.loads((output/"identity.json").read_text()) != identity:
            raise ValueError("Source, input, environment or configuration changed; use a new output directory")
    else:
        atomic_json(output/"identity.json", identity)
        for path in files:
            destination = output/"source"/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    if (output/"final_result.json").exists():
        return json.loads((output/"final_result.json").read_text())
    factory = create_cont_single_auction if config["setting"] == "single" else create_cont_multi_auction
    auction = factory(**environment)
    mechanism = Mechanism(build_mechanism_network(), pickle.loads((folder/"mechanism_params.pkl").read_bytes()))
    network = build_cont_oracle_net(auction.d_priv, auction.d_pub, auction.num_goods,
        auction.num_horizons, auction.num_buyers, hidden=config["hidden"], info=config["info"],
        arch=config["arch"], stop_grad_truth=False)
    optimizer = optax.adam(config["lr"])
    base = jax.random.PRNGKey(config["seed"])
    evaluation = jax.random.PRNGKey(config["evaluation_seed"])
    validation_key = jax.random.fold_in(evaluation, 11001)
    final_key = jax.random.fold_in(evaluation, 22002)
    initial = auction.get_initial_belief_state(auction.get_initial_state(jax.random.fold_in(base, 1)))
    payments = jnp.zeros((auction.num_horizons, auction.num_buyers))
    start, updates_done, selected, records = time.perf_counter(), 0, [], []
    for buyer in range(auction.num_buyers):
        initial_params = network.init(jax.random.fold_in(base, 100+buyer), initial, auction, None, payments)
        def score(params, mode="network"):
            value = float(estimate_buyer_gain(auction, mechanism, network.apply, params,
                config["validation_samples"], validation_key, buyer, mode))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation gain")
            return value
        choices = [{"params": initial_params, "mode": "truth", "validation_gain": 0., "label": "truth"}]
        for mode in ("low", "high", "half"):
            choices.append({"params": initial_params, "mode": mode,
                            "validation_gain": score(initial_params, mode), "label": mode})
        @jax.jit
        def update(params, optimizer_state, rng):
            rng, sample_key = jax.random.split(rng)
            def objective(p):
                return estimate_buyer_gain(auction, mechanism, network.apply, p,
                    config["samples"], sample_key, buyer)
            value, gradient = jax.value_and_grad(objective)(params)
            delta, optimizer_state = optimizer.update(jax.tree_util.tree_map(lambda g: -g, gradient), optimizer_state, params)
            params = optax.apply_updates(params, delta)
            finite = jnp.isfinite(value) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(params)]))
            return params, optimizer_state, rng, finite
        for restart in range(config["restarts"]):
            run = output/f"buyer_{buyer}_restart_{restart}"
            run.mkdir(exist_ok=True)
            if (run/"complete.pkl").exists():
                saved = pickle.loads((run/"complete.pkl").read_bytes())
            else:
                if (run/"resume.pkl").exists():
                    saved = pickle.loads((run/"resume.pkl").read_bytes())
                else:
                    rng, init_key = jax.random.split(jax.random.fold_in(base, 1000+100*buyer+restart))
                    params = network.init(init_key, initial, auction, None, payments)
                    saved = {"params": params, "optimizer_state": optimizer.init(params), "rng": rng,
                        "next_step": 0, "best_params": params, "best_gain": score(params),
                        "best_step": 0, "history": []}
                for step in range(saved["next_step"], config["steps"]):
                    params, opt_state, rng, finite = update(saved["params"], saved["optimizer_state"], saved["rng"])
                    if not bool(finite):
                        raise FloatingPointError("Nonfinite pathwise update; partial checkpoint preserved")
                    saved.update(params=params, optimizer_state=opt_state, rng=rng, next_step=step+1)
                    updates_done += 1
                    if (step+1) % config["eval_every"] == 0 or step+1 == config["steps"]:
                        gain = score(params)
                        saved["history"].append({"step": step+1, "validation_gain": gain})
                        if gain > saved["best_gain"]:
                            saved.update(best_params=params, best_gain=gain, best_step=step+1)
                        atomic_pickle(run/"resume.pkl", saved)
                    if stop_after_updates is not None and updates_done >= stop_after_updates:
                        atomic_pickle(run/"resume.pkl", saved)
                        return None
                atomic_pickle(run/"complete.pkl", saved)
                atomic_json(run/"selection.json", {"best_validation_gain": saved["best_gain"],
                    "best_step": saved["best_step"], "history": saved["history"]})
            choices.append({"params": saved["best_params"], "mode": "network",
                "validation_gain": saved["best_gain"], "label": f"restart{restart}/step{saved['best_step']}"})
        best = max(choices, key=lambda choice: choice["validation_gain"])
        selected.append(best)
        records.append({"buyer": buyer, "selected": best["label"], "mode": best["mode"],
            "validation_gain": best["validation_gain"],
            "candidates": [{k: v for k, v in choice.items() if k != "params"} for choice in choices]})
        print(json.dumps(records[-1]), flush=True)
    atomic_pickle(output/"frozen_policies.pkl", selected)
    atomic_json(output/"frozen_selection.json", records)
    # All selection is complete. Only frozen policies are evaluated below.
    keys = jax.random.split(final_key, config["final_batches"])
    gains = np.asarray([[float(estimate_buyer_gain(auction, mechanism, network.apply,
        choice["params"], config["final_samples"], key, buyer, choice["mode"]))
        for buyer, choice in enumerate(selected)] for key in keys])
    means, se = gains.mean(axis=0), gains.std(axis=0, ddof=1)/len(gains)**.5
    upper = means+NormalDist().inv_cdf(1-.05/auction.num_buyers)*se
    result = {"identity": identity, "role": "independent_test_of_frozen_corrected_pathwise_policies",
        "phase": config["phase"], "gains_by_batch": gains.tolist(),
        "gain_per_buyer": means.tolist(), "se_per_buyer": se.tolist(),
        "signed_buyer_mean_gain": float(means.mean()),
        "positive_gain_sum": float(np.maximum(means, 0).sum()),
        "positive_gain_mean": float(np.maximum(means, 0).mean()),
        "positive_gain_max_buyer": float(np.maximum(means, 0).max()),
        "gain_sampling_upper_per_buyer": upper.tolist(),
        "selected_modes": [choice["mode"] for choice in selected],
        "selected_validation_gains": [choice["validation_gain"] for choice in selected],
        "seconds_this_invocation": time.perf_counter()-start,
        "devices": [str(device) for device in jax.devices()],
        "claim": "Additional empirical pathwise search, not a global IC certificate or replacement of the original ladder. Sampling bounds do not cover missed deviations. Rebuttal values remain unchanged."}
    atomic_json(output/"final_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_check(json.loads(Path(args.config).read_text()), args.output)
    print(json.dumps({"output": args.output, "gains": result["gain_per_buyer"],
                      "upper": result["gain_sampling_upper_per_buyer"]}), flush=True)


if __name__ == "__main__":
    main()
