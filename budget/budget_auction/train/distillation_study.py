"""Validation-bound B-to-A imitation, IC fine-tuning and independent audit.

All fit and fine-tuning budgets must be supplied before running. No teacher final
test is read. Extra teacher/fit compute makes this a diagnostic, not a matched
total-compute comparison with from-scratch A. Use Slurm on the cluster.
"""
import argparse
import json
import os
import time
from pathlib import Path

import haiku as hk
import jax

from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
from budget_auction.train.diagnostic_artifacts import (
    config_digest, load_teacher, read_verified, sha256_bytes)
from budget_auction.train.distill import run_distillation
from budget_auction.train.pilot import FACTORIES, atomic_json, source_digest
from budget_auction.train.study import run_study


FIT_FIELDS = {"seed", "num_trajectories", "num_validation_trajectories", "num_epochs",
    "lr", "batch_size", "alloc_weight", "payment_weight", "eval_every", "print_every"}


def preserve_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing immutable artifact changed: {path}")
    else:
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)


def run_distillation_study(config, teacher_directory, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/"diagnostic.lock"
    # A stale lock is deliberately not auto-deleted: first reconcile the process.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    try:
        return _run(config, Path(teacher_directory), root)
    finally:
        lock.unlink()


def _run(config, teacher_directory, root):
    source = source_digest()
    student = config["fine_tune"]
    fit = config["fit"]
    if set(fit) != FIT_FIELDS or "initialization" in student:
        raise ValueError("Declare all fit budgets explicitly; the driver supplies initialization")
    if not 11000000 <= fit["seed"] < 12000000:
        raise ValueError("Reserve 11000000..11999999 for diagnostic imitation data")
    teacher_identity, selection, teacher_params, teacher_raw = load_teacher(
        teacher_directory, config["teacher_sha256"], student, source)
    identity = {"config": config, "config_sha256": config_digest(config), "source_sha256": source,
        "teacher_source_sha256": teacher_identity["source_sha256"],
        "role": "B_to_A_diagnostic_with_additional_imitation_compute"}
    identity_path = root/"diagnostic_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Diagnostic source or configuration changed; use a new directory")
    else:
        atomic_json(identity_path, identity)
    for name, data in teacher_raw.items():
        preserve_bytes(root/"teacher_input"/name, data)

    folder = root/"fit"
    folder.mkdir(exist_ok=True)
    receipt = folder/"complete.json"
    if receipt.exists():
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if recorded["diagnostic_config_sha256"] != identity["config_sha256"]:
            raise ValueError("Fit receipt belongs to another diagnostic")
        for name, digest in recorded["files_sha256"].items():
            read_verified(folder/name, digest)
    else:
        auction = FACTORIES["e1"](seed=student["seed"], **student.get("environment", {}))
        teacher_net = hk.without_apply_rng(hk.transform(VARIANT_NETWORK["B"]))
        student_net = hk.without_apply_rng(hk.transform(VARIANT_NETWORK["A"]))
        started = time.perf_counter()
        timing_path = folder/"invocations.json"
        invocations = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else []
        finished = False
        try:
            run_distillation(auction, student_net, BudgetMechanism(teacher_net, teacher_params),
                **fit, save_dir=folder)
            finished = True
        finally:
            invocations.append({"seconds_including_compile": time.perf_counter()-started,
                "completed": finished})
            atomic_json(timing_path, invocations)
        atomic_json(receipt, {"diagnostic_config_sha256": identity["config_sha256"],
            "files_sha256": {name: sha256_bytes((folder/name).read_bytes()) for name in
                ("distilled_A_params.pkl", "fit_validation.json", "fit_resume.pkl")}})

    params_bytes = (folder/"distilled_A_params.pkl").read_bytes()
    preserve_bytes(root/"initialization/params.pkl", params_bytes)
    initialization = {"role": "validation_selected_distilled_A_initialization", "variant": "A",
        "setting": "e1", "seed": student["seed"], "environment": student.get("environment", {}),
        "source_sha256": source, "params_sha256": sha256_bytes(params_bytes),
        "teacher_sha256": config["teacher_sha256"], "fit_config": fit,
        "fit_validation_sha256": sha256_bytes((folder/"fit_validation.json").read_bytes()),
        "diagnostic_config_sha256": identity["config_sha256"]}
    metadata_bytes = (json.dumps(initialization, indent=2, sort_keys=True)+"\n").encode()
    preserve_bytes(root/"initialization/manifest.json", metadata_bytes)
    fine_tune = {**student, "initialization": {"params": "../initialization/params.pkl",
        "params_sha256": sha256_bytes(params_bytes), "manifest": "../initialization/manifest.json",
        "manifest_sha256": sha256_bytes(metadata_bytes)}}
    result = run_study(fine_tune, root/"fine_tune")
    summary = {"role": "distillation_diagnostic_not_main_representation_comparison",
        "identity": identity, "teacher_validation": selection,
        "fit_validation": json.loads((folder/"fit_validation.json").read_text(encoding="utf-8")),
        "fit_timing": json.loads((folder/"invocations.json").read_text(encoding="utf-8")),
        "fine_tune_result_sha256": sha256_bytes((root/"fine_tune/final_result.json").read_bytes()),
        "fine_tune_revenue": result["revenue"], "fine_tune_audit": result["audit"],
        "selection_passed": result["selection_passed"],
        "final_empirical_ic_passed": result["final_empirical_ic_passed"],
        "compute_accounting": "Report teacher training/search, imitation and fine-tuning/audit separately; teacher timings remain in the pinned teacher study directory.",
        "limits": "Teacher-state imitation does not prove equality of classes or optimal revenues. The student receives a separate empirical IC audit."}
    atomic_json(root/"diagnostic_result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU allocation required; refusing CPU fallback")
    result = run_distillation_study(json.loads(Path(args.config).read_text(encoding="utf-8")),
        args.teacher, args.output)
    print(json.dumps({"DIAGNOSTIC_COMPLETE": args.output, "revenue": result["fine_tune_revenue"]}), flush=True)


if __name__ == "__main__":
    main()
