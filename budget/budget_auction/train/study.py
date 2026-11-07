"""Resumable trial search, validation selection, then one frozen final test.

The JSON configuration declares the training grid and audit budget before a run.
Search failures are not certificates of target infeasibility. Outputs describe
the best mechanism found by this finite protocol, never a global optimum.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp

from budget_auction.core import BudgetSimulator, build_bounded_policy_net
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
from budget_auction.train.evaluate import eval_budget_mechanism_multikey, summarize_deviation_gains
from budget_auction.train.pilot import FACTORIES, atomic_json, atomic_pickle, source_digest
from budget_auction.train.run_acceptance import nested_GD_budget
from budget_auction.train.diagnostic_artifacts import load_study_initialization
from budget_auction.train.strong_audit import AuditPolicy, audit_capacity_ladder


def read_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def descriptor(policy):
    return {"params": policy.params, "sigmas": policy.sigmas, "modes": policy.modes}


def select_audit_candidates(candidates, config):
    """Declare target coverage before using strong-audit validation scores.

    The legacy global rule remains available for existing single-target configs.
    Production per-trial mode reserves the same number of slots for every target.
    """
    key = lambda item: (item[0]["ic_sum"] > config.get("screen_ic_threshold_sum", .1),
                        -item[0]["revenue"])
    mode = config.get("shortlist_mode", "global")
    if mode == "global":
        count = config.get("audit_candidates", 3)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("audit_candidates must be a positive integer")
        return sorted(candidates, key=key)[:count]
    if mode != "per_trial":
        raise ValueError("Unknown shortlist_mode")
    count = config.get("audit_candidates_per_trial")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("per_trial selection requires positive audit_candidates_per_trial")
    groups = {}
    for item in candidates:
        groups.setdefault(item[0]["trial"], []).append(item)
    return [item for trial in sorted(groups)
            for item in sorted(groups[trial], key=key)[:count]]


def run_study(config, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    identity = {"config_sha256": fingerprint, "source_sha256": source_digest(), "config": config}
    initial_params = load_study_initialization(config, root, identity["source_sha256"])
    if (root/"identity.json").exists():
        if json.loads((root/"identity.json").read_text()) != identity:
            raise ValueError("Configuration or source differs from this run; use a new output directory")
    else:
        atomic_json(root/"identity.json", identity)
    if (root/"final_result.json").exists():
        return json.loads((root/"final_result.json").read_text())

    seed = config["seed"]
    # Tuning and the main study must use different evaluation namespaces even
    # when they intentionally share training seeds and comparison streams.
    evaluation_seed = config.get("evaluation_seed", seed)
    auction = FACTORIES[config["setting"]](seed=seed, **config.get("environment", {}))
    network = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[config["variant"]]))
    hidden = config.get("policy_hidden", 128)
    policy_network = build_bounded_policy_net(auction.d_priv, auction.d_pub, auction.num_goods, hidden=hidden)
    ic_threshold = config.get("ic_threshold_sum", .05)
    candidates = []
    start = time.perf_counter()
    for index, trial in enumerate(config["trials"]):
        folder = root/f"trial_{index:02d}"
        folder.mkdir(exist_ok=True)
        training = {**config["training"], **trial, "seed": seed + 1009*index,
                    "checkpoint_thresh": 1e9}
        atomic_json(folder/"training_config.json", training)
        if (folder/"trained.pkl").exists():
            logs = read_pickle(folder/"trained.pkl")
        else:
            resume = read_pickle(folder/"resume.pkl") if (folder/"resume.pkl").exists() else None
            training_start = time.perf_counter()
            logs = nested_GD_budget(auction, network, policy_network, **training,
                init_mechanism_params=initial_params,
                resume_state=resume, checkpoint_callback=lambda state: atomic_pickle(folder/"resume.pkl", state))
            atomic_pickle(folder/"trained.pkl", logs)
            atomic_json(folder/"training_timing.json", {
                "seconds_this_invocation_including_compile": time.perf_counter()-training_start,
                "resumed_from_epoch": 0 if resume is None else resume["next_epoch"],
                "completed_epochs": len(logs["loss"]),
                "mechanism_parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(logs["mechanism_params"])),
                "policy_parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(logs["policy_params"]))})
        checkpoints = logs["checkpoint_mech"] + [{
            "epoch": "final", "mechanism_params": logs["mechanism_params"],
            "policy_params": logs["policy_params"],
        }]
        for number, checkpoint in enumerate(checkpoints):
            destination = folder/f"screen_{number:03d}.json"
            if destination.exists():
                record = json.loads(destination.read_text())
            else:
                # Screening policies were optimized during training. These scores
                # can shortlist candidates but cannot be their final IC evidence.
                evaluation = eval_budget_mechanism_multikey(auction, network,
                    checkpoint["mechanism_params"], training["profit_goal"],
                    policy_network=policy_network, policy_params=checkpoint["policy_params"],
                    K=config.get("screen_batches", 4), num_samples=config.get("screen_samples", 128),
                    base_seed=evaluation_seed+100000)
                record = {"trial": index, "checkpoint": number, "epoch": checkpoint["epoch"],
                    "revenue": evaluation["profit_stats"]["mean"],
                    "ic_sum": evaluation["ic_sum"], "profit_goal": training["profit_goal"],
                    "role": "checkpoint_screening_with_training_policy"}
                atomic_json(destination, record)
            candidates.append((record, checkpoint["mechanism_params"]))

    # Apply the same shortlist rule to every representation. The looser screen
    # is only a compute-saving heuristic; selection below uses fresh strong search.
    shortlist = select_audit_candidates(candidates, config)
    if not shortlist:
        raise ValueError("The configuration must contain at least one training trial")
    atomic_json(root/"shortlist.json", [record for record, _ in shortlist])
    audited = []
    for index, (record, mechanism_params) in enumerate(shortlist):
        folder = root/f"candidate_{index:02d}"
        folder.mkdir(exist_ok=True)
        if (folder/"selection.json").exists():
            selection = json.loads((folder/"selection.json").read_text())
            saved = read_pickle(folder/"selected.pkl")
        else:
            audit_resume = read_pickle(folder/"audit_resume.pkl") if (folder/"audit_resume.pkl").exists() else None
            audit, policy = audit_capacity_ladder(auction, network, mechanism_params,
                hidden_sizes=config.get("audit_hidden_sizes", [hidden, 2*hidden]),
                seed=evaluation_seed+200000+1009*index, ic_threshold=ic_threshold,
                resume_state=audit_resume,
                checkpoint_callback=lambda state: atomic_pickle(folder/"audit_resume.pkl", state),
                **config["audit"])
            revenue = eval_budget_mechanism_multikey(auction, network, mechanism_params,
                record["profit_goal"], K=config.get("screen_batches", 4),
                num_samples=config.get("screen_samples", 128), base_seed=evaluation_seed+300000)
            saved = {"mechanism_params": mechanism_params, "policy": descriptor(policy),
                     "policy_hidden_per_buyer": audit["selected_hidden_per_buyer"]}
            selection = {"candidate": record, "audit": audit, "revenue": revenue["profit_stats"],
                "role": "mechanism_selection_validation_not_final_test",
                "passes_selection_audit": audit["passes_empirical_ic_audit"]}
            atomic_pickle(folder/"selected.pkl", saved)
            atomic_json(folder/"selection.json", selection)
        audited.append((selection, saved))
    # Prefer a validated mechanism with high revenue. If none passed, preserve
    # the least exploitable candidate and clearly report the failed search.
    audited.sort(key=lambda pair: (
        not pair[0]["passes_selection_audit"],
        -pair[0]["revenue"]["mean"] if pair[0]["passes_selection_audit"]
        else pair[0]["audit"]["ic_sum_sampling_ucb95"]))
    selection, frozen = audited[0]
    atomic_pickle(root/"frozen_selection.pkl", frozen)
    atomic_json(root/"frozen_selection.json", selection)

    # Nothing below this line may change the selected mechanism or policy.
    mechanism = BudgetMechanism(network, frozen["mechanism_params"])
    selected_networks = tuple(build_bounded_policy_net(auction.d_priv, auction.d_pub,
        auction.num_goods, hidden=size) for size in frozen["policy_hidden_per_buyer"])
    policy = AuditPolicy(selected_networks, **frozen["policy"])
    batches, samples = config.get("final_batches", 32), config.get("final_samples", 256)
    gains = jnp.stack([BudgetSimulator.estimate_cumulative_deviation_gains(
        auction, mechanism, policy, samples, key)
        for key in jax.random.split(jax.random.PRNGKey(evaluation_seed+400000), batches)])
    audit = summarize_deviation_gains(gains)
    evaluation = eval_budget_mechanism_multikey(auction, network, frozen["mechanism_params"],
        selection["candidate"]["profit_goal"], K=batches, num_samples=samples, base_seed=evaluation_seed+500000)
    result = {"identity": identity, "role": "independent_final_test_of_frozen_selection",
        "selection_passed": selection["passes_selection_audit"],
        "final_empirical_ic_passed": audit["ic_sum_sampling_ucb95"] <= ic_threshold,
        "ic_threshold_sum": ic_threshold, "audit": audit, "test_gains_by_batch": gains.tolist(),
        "revenue": evaluation["profit_stats"], "ir_loss": evaluation["ir_loss_stats"],
        "test_revenue_by_batch": evaluation["profit"].tolist(),
        "test_feasibility_by_batch": {
            "ir_loss": evaluation["ir_loss"].tolist(),
            "budget_violation": evaluation["budget_viol_rate"].tolist(),
            "ir_violation": evaluation["ir_viol_rate"].tolist()},
        "sampling_uncertainty": {
            "independent_units": "evaluation batch means",
            "revenue_interval": "two-sided 95 percent normal approximation",
            "ic_upper_bound": "one-sided 95 percent Bonferroni normal approximation across buyers",
            "scope": "fixed mechanism and selected policies within this run",
            "excludes": ["best-response search error", "training-seed variability",
                         "simultaneous coverage across the full study"]},
        "final_streams": {"deviation_base_seed": evaluation_seed+400000,
                          "truthful_base_seed": evaluation_seed+500000},
        "budget_violation": evaluation["budget_viol_stats"], "ir_violation": evaluation["ir_viol_stats"],
        "num_batches": batches, "samples_per_batch": samples,
        "evaluation_seed": evaluation_seed,
        "phase": config.get("phase", "unspecified"),
        "publication_eligible_phase": config.get("phase") == "main_study",
        "devices": [str(x) for x in jax.devices()], "jax": jax.__version__,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "wall_seconds_this_invocation": time.perf_counter()-start,
        "claim": "Best mechanism found by the declared finite search; empirical audit, not global optimality or exact IC"}
    atomic_json(root/"final_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU allocation required; refusing CPU fallback")
    result = run_study(json.loads(Path(args.config).read_text()), args.output)
    print(json.dumps({"STUDY_COMPLETE": args.output, "revenue": result["revenue"],
                      "audit": result["audit"]}), flush=True)


if __name__ == "__main__":
    main()
