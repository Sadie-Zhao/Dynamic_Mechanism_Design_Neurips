"""Standard-library-only ownership/continuation checks before numerical imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_digest(source):
    root=Path(source)/'budget_auction';value=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        value.update(path.relative_to(root).as_posix().encode());value.update(path.read_bytes())
    return value.hexdigest()


def verify_packages(installed,expected):
    numerical=sorted((p['name'],p['version']) for p in installed if p['name'].lower()!='pip')
    tooling=sorted((p['name'],p['version']) for p in installed if p['name'].lower()=='pip')
    if numerical!=sorted(tuple(p) for p in expected):
        raise ValueError('Numerical package versions differ from registration')
    return numerical,tooling


def verify_checkpoint_capture(config):
    if (config.get('epoch_phases')!=[500,1000]
            or config.get('candidate_count_per_phase')!=2
            or config.get('training',{}).get('checkpoint_interval')!=50
            or config.get('training',{}).get('checkpoint_thresh')!=1e9):
        raise ValueError('Production search must explicitly capture every50-epoch candidate snapshot')


def guard_path(project_root, case, backend):
    if not re.fullmatch(r'[A-Za-z0-9_]+',case):
        raise ValueError('Unsafe case identity')
    root=Path(project_root).resolve()
    if backend=='local_wsl':return root/'outputs/local_main_speculation.lock'
    if backend=='slurm':return root/'outputs/revenue_search_execution_locks'/(case+'.lock')
    raise ValueError('Unknown registered backend')


def process_alive(pid, ticks):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[21]==str(ticks)


def acquire_guard(path, marker):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as out:
        json.dump(marker,out,indent=2,allow_nan=False);out.write('\n')


def release_guard(path, marker):
    if json.loads(Path(path).read_text())!=marker:
        raise ValueError('Guard changed owner;retain it for review')
    Path(path).unlink()


def file_manifest(folder):
    root=Path(folder).resolve();result={}
    if not root.exists():return result
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        if not path.resolve().is_relative_to(root):
            raise ValueError('Case artifact resolves outside its output')
        result[path.relative_to(root).as_posix()]=sha(path)
    return result


def validate_resume(output, controls, review, maximum):
    """No changed/omitted partial artifacts or live prior worker may be reused."""
    output,controls=Path(output),Path(controls)
    previous=sorted(controls.glob('invocation_*/worker.json'))
    if not previous:
        if review is not None:raise ValueError('First invocation cannot carry an unrelated resume review')
        if output.exists() and any(output.iterdir()):
            raise ValueError('Existing numerical output has no prior worker identity')
        return 1
    prior=previous[-1];old=json.loads(prior.read_text())
    if old['state']=='complete' or (output/'completion.json').exists():
        raise ValueError('Case is already complete;never launch it again')
    if old['state'] not in ['cooperative_stop_requires_review','stopped_or_error_requires_review']:
        raise ValueError('Prior worker receipt is not terminal;explicit interruption review required')
    if process_alive(old['pid'],old['process_start_ticks']):
        raise ValueError('Prior exact worker remains alive')
    if review is None or review.get('prior_worker_sha256')!=sha(prior):
        raise ValueError('Prior terminal worker receipt was not reviewed')
    if (not output.exists() and not (prior.parent/'runtime.json').exists()
            and old.get('error')=='Numerical package versions differ from registration'):
        # This exact failure precedes JAX imports,case initialization or samples.
        # Preserve its receipt/cost and consume an invocation;do not call it a
        # numerical retry or silently accept an unknown partial output.
        if (review.get('pre_numerical_package_failure_verified') is not True
                or review.get('files_sha256')!={}
                or review.get('control_files_sha256')!=file_manifest(controls)
                or review.get('checkpoints')!={}):
            raise ValueError('Pre-numerical package failure needs its complete explicit review')
        if len(previous)>=maximum:raise ValueError('Declared worker invocation cap exhausted')
        return len(previous)+1
    if review.get('prior_runtime_sha256')!=sha(prior.parent/'runtime.json'):
        raise ValueError('Prior invocation runtime record was not pinned')
    if review.get('control_files_sha256')!=file_manifest(controls):
        raise ValueError('All prior worker/runtime/continuation records must be pinned')
    actual=file_manifest(output)
    if review.get('files_sha256')!=actual:
        raise ValueError('Full saved-artifact review is missing,changed or incomplete')
    invocations=sorted(output.glob('invocation_*.json'))
    if not invocations or review.get('prior_invocation_sha256')!=sha(invocations[-1]):
        raise ValueError('Prior search invocation receipt was not reviewed')
    for name,expected in review.get('checkpoints',{}).items():
        if actual.get(name)!=expected:
            raise ValueError('Reviewed exact numerical checkpoint differs')
    if len(previous)>=maximum:
        raise ValueError('Declared worker invocation cap exhausted')
    return len(previous)+1


def validate_worker_plan(plan,seal,config,source,project_root,backend):
    """All backend and budget identities are fixed before touching the GPU."""
    case=config['case_id'];root=Path(project_root).resolve();source=Path(source).resolve()
    verify_checkpoint_capture(config)
    guard_path(root,case,backend)  # Also rejects path-shaped identities.
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if config['phase']!='revenue_search' or plan['cases'][case]!=fingerprint:
        raise ValueError('Case configuration differs from the experimental registration')
    if plan.get('automatic_retry') is not False or plan.get('registered_before_new_samples') is not True:
        raise ValueError('Missing prospective execution controls')
    registered=datetime.fromisoformat(plan['registered_at'])
    if registered.tzinfo is None or registered>datetime.now(timezone.utc):
        raise ValueError('Registration timestamp is not a completed prospective declaration')
    if plan['execution_backend_by_case'][case]!=backend:
        raise ValueError('Case is reserved for another backend;no duplicate local/cluster copy')
    if backend=='slurm' and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Remote numerical work needs an actual Slurm allocation')
    if (plan['chunk_seconds']!=7200 or config['max_invocations']!=6
            or plan.get('max_invocations_per_training_phase')!=2):
        raise ValueError('Resource controls differ from the common declared case budget')
    for name,expected in plan['source_files_sha256'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or sha(path)!=expected:
            raise ValueError('Registered source artifact changed')
    if package_digest(source)!=plan['source_sha256']:
        raise ValueError('Numerical package has unregistered or changed Python source')
    output=(root/plan['outputs'][case]).resolve()
    if not output.is_relative_to(root/'outputs'):
        raise ValueError('Registered numerical output escapes project outputs')
    return output,fingerprint
