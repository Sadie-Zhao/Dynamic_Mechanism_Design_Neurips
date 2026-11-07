"""Short, instrumented camera-ready pilot. Outputs are not paper results."""

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

import haiku as hk
import jax
import numpy as np

from budget_auction.core import build_bounded_policy_net
from budget_auction.envs.single_buyer_indep_env import create_single_buyer_indep_auction
from budget_auction.envs.public_budget_env import create_public_budget_auction
from budget_auction.envs.private_budget_multigood_env import create_private_budget_multigood_auction
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
from budget_auction.train.run_acceptance import nested_GD_budget
from budget_auction.train.evaluate import eval_budget_mechanism_multikey
from budget_auction.train.strong_audit import audit_mechanism


FACTORIES = {"e0": create_single_buyer_indep_auction, "e1": create_public_budget_auction,
             "e2": create_private_budget_multigood_auction}
PILOT_GOALS = {"e0": 1.5, "e1": 2.0, "e2": 3.0}


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as output:
        pickle.dump(jax.device_get(value), output)
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", nargs="+", choices=list(FACTORIES), default=list(FACTORIES))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_NETWORK), default=["A", "C"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--audit-steps", type=int, default=20)
    parser.add_argument("--npg-steps", type=int, default=5)
    parser.add_argument("--audit-episodes", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU allocation is required; refusing CPU fallback")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 2, "purpose": "pilot_only_not_paper_results",
        "source_sha256": source_digest(), "arguments": vars(args), "jax": jax.__version__,
        "python": platform.python_version(), "devices": [str(x) for x in jax.devices()],
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "runs": []}
    atomic_json(root/"manifest.json", manifest)
    for setting in args.settings:
        auction = FACTORIES[setting](seed=42)
        for variant in args.variants:
            run = root/f"{setting}_{variant}"
            run.mkdir(exist_ok=True)
            network = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[variant]))
            policy = build_bounded_policy_net(auction.d_priv, auction.d_pub, auction.num_goods)
            start = time.perf_counter()
            checkpoint_times = []
            def checkpoint(state):
                checkpoint_times.append(time.perf_counter()-start)
                atomic_pickle(run/"resume.pkl", state)
            config = dict(profit_goal=PILOT_GOALS[setting], num_outer_epochs=args.epochs,
                num_inner_steps=args.inner_steps, num_samples_train=args.batch,
                lr_theta=1e-4, lr_w=3e-4, alpha_weight=5.0, beta_weight=0.0,
                checkpoint_thresh=0.15, checkpoint_interval=1, print_iter=5, seed=42)
            atomic_json(run/"config.json", {**config, "setting": setting,"variant":variant,
                "horizon":auction.num_horizons,"discount":auction.discount,
                "initial_budgets":np.asarray(auction.initial_budgets).tolist(),
                "type_low":np.asarray(auction.types_low).tolist(),
                "type_high":np.asarray(auction.types_high).tolist(),
                "sigma_no":auction.trans_sigma_no,"sigma_got":1/auction.trans_base_rate,
                "agent_representation":"full_history_mlp128","mechanism_schema_version":2,
                "source_sha256":manifest["source_sha256"]})
            logs = nested_GD_budget(auction,network,policy,**config,checkpoint_callback=checkpoint)
            training_seconds = time.perf_counter()-start
            atomic_pickle(run/"trained.pkl", {"mechanism_params":logs["mechanism_params"],
                "policy_params":logs["policy_params"],"config":config,
                "source_sha256":manifest["source_sha256"],"schema_version":2})
            eval_start = time.perf_counter()
            revenue = eval_budget_mechanism_multikey(auction,network,logs["mechanism_params"],
                config["profit_goal"],K=4,num_samples=args.test_samples,base_seed=901001)
            audit, selected = audit_mechanism(auction,network,logs["mechanism_params"],policy,
                seed=902002,pathwise_steps=args.audit_steps,pathwise_restarts=1,
                npg_steps=args.npg_steps,npg_restarts=1,episodes=args.audit_episodes,
                validation_samples=args.validation_samples,test_batches=4,test_samples=args.test_samples)
            # Save parameters and static descriptors, not nonportable network closures.
            atomic_pickle(run/"audit_policy.pkl", {"params":selected.params,
                "sigmas":selected.sigmas,"modes":selected.modes,
                "policy_representation":"full_history_mlp128"})
            result = {"setting":setting,"variant":variant,"pilot_only":True,
                "mechanism_parameter_count":sum(x.size for x in jax.tree_util.tree_leaves(logs["mechanism_params"])),
                "policy_parameter_count":sum(x.size for x in jax.tree_util.tree_leaves(logs["policy_params"])),
                "training_seconds_including_compile":training_seconds,
                "first_epoch_seconds_including_compile":checkpoint_times[0],
                "steady_epoch_seconds":float(np.mean(np.diff(checkpoint_times))) if len(checkpoint_times)>1 else None,
                "audit_seconds_including_compile":time.perf_counter()-eval_start,
                "profit_stats":revenue["profit_stats"],"ir_loss_stats":revenue["ir_loss_stats"],
                "budget_viol_stats":revenue["budget_viol_stats"],"audit":audit,
                "final_training_metrics":logs["final_train_metrics"]}
            atomic_json(run/"result.json", result)
            manifest["runs"].append(result)
            atomic_json(root/"manifest.json", manifest)
            print(json.dumps({k:v for k,v in result.items() if k not in ("audit",)},allow_nan=False),flush=True)
    print(f"PILOT_COMPLETE {root}",flush=True)


if __name__ == "__main__":
    main()
