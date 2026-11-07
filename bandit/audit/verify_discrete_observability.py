"""Enumerate report histories to check whether public outcomes reveal past reports.

No policies are trained. At each prefix and own current report, compare all
opponent reports. Separated public outcomes permit inductive reconstruction of
past reports. This is a finite-precision check on the discrete report grid,
not a claim for arbitrary continuous reports or unenumerated prefixes.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from bandit_auction.core import State
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.mechanisms.network import Mechanism, build_mechanism_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=1000000)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--separation-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    checkpoint = ROOT/args.checkpoint
    summary = json.loads((checkpoint/"summary.json").read_text())
    config = summary.get("cfg", {})
    if config.get("use_attn_mechanism", False):
        raise ValueError("This verification currently supports the shipped baseline architecture only")
    kwargs = {key: config[key] for key in ("n_buyers", "n_goods", "horizon", "discount", "allocation_cost") if key in config}
    factory = create_discrete_multi_auction if args.multi else create_discrete_single_auction
    auction = factory(seed=42, **kwargs)
    if auction.num_buyers != 2 or not 1 <= args.depth <= auction.num_horizons:
        raise ValueError("Need two buyers and a depth within the actual horizon")
    profiles = np.asarray(auction.type_profiles)
    joint, buyers, goods = profiles.shape
    total = sum(joint**(t+1) for t in range(args.depth))
    if total > args.max_evaluations:
        raise ValueError(f"Requested {total} evaluations exceeds the explicit cap; reduce depth")
    params_path = checkpoint/"mechanism_params.pkl"
    with params_path.open("rb") as stream:
        params = pickle.load(stream)
    mechanism = Mechanism(build_mechanism_network(), params)

    @jax.jit
    def evaluate(report_history, allocation_history, reports, t):
        def one(rh, ah, report):
            return mechanism(State(t, ah, jnp.zeros((buyers, goods)), rh), report, auction)
        return jax.vmap(one)(report_history, allocation_history, reports)

    rh = np.zeros((1, auction.num_horizons, buyers, goods), np.float32)
    ah = np.zeros_like(rh)
    levels = []
    started = time.perf_counter()
    for t in range(args.depth):
        count = len(rh)
        repeated_rh = np.repeat(rh, joint, axis=0)
        repeated_ah = np.repeat(ah, joint, axis=0)
        reports = np.tile(profiles, (count, 1, 1))
        allocations, payments = [], []
        for start in range(0, len(reports), args.chunk):
            n = min(args.chunk, len(reports)-start)
            def chunk(array):
                part = array[start:start+n]
                return np.pad(part, [(0,args.chunk-n)]+[(0,0)]*(part.ndim-1))
            alloc, pay = evaluate(chunk(repeated_rh), chunk(repeated_ah), chunk(reports), jnp.asarray(t))
            allocations.append(np.asarray(alloc)[:n])
            payments.append(np.asarray(pay)[:n])
        allocations, payments = np.concatenate(allocations), np.concatenate(payments)
        outcomes = np.concatenate([allocations.reshape(len(reports),-1), payments],axis=1).reshape(count,joint,-1)
        minima, witnesses = [], []
        for buyer in range(buyers):
            minimum, witness = float("inf"), None
            for own in np.unique(profiles[:,buyer],axis=0):
                indices = np.flatnonzero(np.all(profiles[:,buyer] == own,axis=1))
                for a_index, first in enumerate(indices):
                    for second in indices[a_index+1:]:
                        distances = np.max(np.abs(outcomes[:,first]-outcomes[:,second]),axis=1)
                        parent = int(np.argmin(distances))
                        if float(distances[parent]) < minimum:
                            minimum = float(distances[parent])
                            witness = {"history": rh[parent,:t].tolist(), "own_report": own.tolist(),
                                "opponent_reports": [profiles[first,1-buyer].tolist(),profiles[second,1-buyer].tolist()]}
            minima.append(minimum)
            witnesses.append(witness)
        level = {"time":t,"prefixes":count,"evaluations":len(reports),
                 "minimum_public_outcome_separation_per_buyer":minima,
                 "smallest_separation_witnesses":witnesses}
        levels.append(level)
        print(json.dumps({key:value for key,value in level.items() if key != "smallest_separation_witnesses"}),flush=True)
        if t+1 < args.depth:
            repeated_rh[:,t] = reports
            repeated_ah[:,t] = allocations
            rh, ah = repeated_rh, repeated_ah
    passed = all(min(level["minimum_public_outcome_separation_per_buyer"]) > args.separation_tolerance for level in levels)
    result = {"checkpoint":args.checkpoint,"mechanism_sha256":hashlib.sha256(params_path.read_bytes()).hexdigest(),
        "verification_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax":jax.__version__,"devices":[str(device) for device in jax.devices()],
        "horizon":auction.num_horizons,"enumerated_depth":args.depth,"evaluations":total,
        "covers_all_discrete_report_histories":args.depth == auction.num_horizons,
        "separated_above_tolerance":passed,"tolerance":args.separation_tolerance,"levels":levels,
        "seconds":time.perf_counter()-started,
        "interpretation":"Finite-precision report-grid observability check; no retraining or regret optimization. Full-depth separation supports inductive report reconstruction from public allocations and payments, conditional on the declared discrete report space."}
    output = ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"passed":passed,"seconds":result["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
