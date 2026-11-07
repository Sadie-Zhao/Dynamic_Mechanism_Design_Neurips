"""One owned, prospectively registered numerical invocation; no retry loop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Load the standard-library guard helper without initializing package __init__.
import importlib.util
_spec=importlib.util.spec_from_file_location('revenue_execution_preimport',Path(__file__).with_name('revenue_execution.py'))
execution=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(execution)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def atomic_record(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    os.replace(temporary, path)


class CooperativeStop(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume-review', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve(); folder = args.registration.resolve()
    plan = json.loads((folder/'registration.json').read_text())
    seal = json.loads((folder/'seal.json').read_text())
    if sha(folder/'registration.json') != seal['registration_sha256']:
        raise ValueError('Experimental seal differs')
    config = json.loads(args.config.read_text()); case = config['case_id']
    source = Path(__file__).resolve().parents[2]
    local = 'microsoft' in platform.release().lower()
    backend = 'local_wsl' if local else 'slurm'
    output,fingerprint=execution.validate_worker_plan(plan,seal,config,source,root,backend)
    lock=execution.guard_path(root,case,backend)
    marker = {'pid': os.getpid(), 'process_start_ticks': Path('/proc/self/stat').read_text().split()[21],
              'started_at': datetime.now(timezone.utc).isoformat(), 'case_id': case,
              'registration_sha256': seal['registration_sha256'], 'state': 'REGISTERED_REVENUE_SEARCH',
              'attempt': str(output), 'project_root': str(root),'backend':backend,'automatic_retry': False}
    execution.acquire_guard(lock,marker)
    started = time.perf_counter(); receipt = None
    try:
        controls = output.parent/(output.name+'.controls'); controls.mkdir(parents=True, exist_ok=True)
        review = json.loads(args.resume_review.read_text()) if args.resume_review else None
        invocation=execution.validate_resume(output,controls,review,config['max_invocations'])
        control = controls/f'invocation_{invocation:02d}'; control.mkdir(exist_ok=False)
        receipt = control/'worker.json'; record = dict(marker, state='running')
        if review is not None:
            write_new(control/'resume_review.json',review)
            record['resume_review_sha256']=sha(control/'resume_review.json')
            record['supplied_resume_review_sha256']=sha(args.resume_review)
        write_new(receipt, record)
        os.environ.update(plan['numerical_settings'])
        packages = json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'], text=True))
        installed,tooling=execution.verify_packages(packages,plan['packages'])
        # Imports that can initialize the numerical backend occur only under the lock.
        import jax
        from budget_auction.train.revenue_backend import RevenueBackend
        from budget_auction.train.revenue_flow import run_search
        runtime = {'started_at': marker['started_at'], 'python': sys.version,
                   'platform': platform.platform(), 'packages': installed,
                   'devices': [{'kind': d.device_kind,'platform': d.platform} for d in jax.devices()],
                   'backend': backend, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                   'numerical_settings': {k:os.environ.get(k) for k in plan['numerical_settings']},
                   'registration_sha256': seal['registration_sha256'], 'config_sha256':fingerprint}
        runtime.update(source_sha256=execution.package_digest(source),
                       source_files_verified_count=len(plan['source_files_sha256']),
                       tooling_packages=tooling,package_policy='pip recorded separately;all59 registered package versions remain exact')
        write_new(control/'runtime.json', runtime)
        record['runtime_sha256']=sha(control/'runtime.json')
        def checkpoint(name, path):
            if time.perf_counter()-started >= plan['chunk_seconds']:
                raise CooperativeStop('Declared resource chunk ended after an exact saved checkpoint/stage')
        result = run_search(config, output, RevenueBackend(), lock_path=lock,
                            registration=folder, resume_review=review, checkpoint_hook=checkpoint)
        record.update(state='complete', result_state=result['state'], result_sha256=sha(output/'result.json'))
    except CooperativeStop as exc:
        record.update(state='cooperative_stop_requires_review', exception_type=type(exc).__name__, error=str(exc))
    except BaseException as exc:
        if receipt is not None:
            record.update(state='stopped_or_error_requires_review', exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if receipt is not None:
            record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                          seconds_this_invocation=time.perf_counter()-started)
            atomic_record(receipt, record)
        execution.release_guard(lock,marker)


if __name__ == '__main__':
    main()
