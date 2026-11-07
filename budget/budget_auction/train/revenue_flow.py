"""Validation-only search orchestration; final failures never trigger adaptation.

Production entry requires a sealed case registration and exclusive global owner.
The pipeline-verification mode is CPU-only and never publication eligible.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import jax

from budget_auction.train.pilot import atomic_json, source_digest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(row, threshold):
    return (math.isfinite(row['ic_upper']) and row['ic_upper'] <= threshold
            and row['observed_feasible'] and math.isfinite(row['revenue']['mean']))


def choose_continuation(rows, threshold):
    passing = [r for r in rows if eligible(r, threshold)]
    if passing:
        chosen = min(passing, key=lambda r: (-r['revenue']['mean'], r['candidate_index']))
    else:
        chosen = min(rows, key=lambda r: (r['ic_upper'], -r['revenue']['mean'], r['candidate_index']))
    return {'trial_index': chosen['trial_index'], 'candidate_index_used_for_choice': chosen['candidate_index'],
            'repair_initialization': not bool(passing), 'initialization': 'selected_trial_final_first_phase_state',
            'old_candidate_checkpoint_is_not_optimizer_initialization': True}


def select_stress(rows, threshold, count):
    return sorted((r for r in rows if eligible(r, threshold)),
                  key=lambda r: (-r['revenue']['mean'], r['candidate_index']))[:count]


class StageStore:
    def __init__(self, root, identity, hook=None):
        self.root, self.identity, self.hook = Path(root), identity, hook
        (self.root / 'stages').mkdir(exist_ok=True)

    def stage(self, name, inputs, operation):
        path = self.root / 'stages' / (name + '.json')
        fingerprint = digest({'identity': self.identity, 'inputs': inputs})
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['inputs_digest'] != fingerprint or digest(saved['data']) != saved['data_digest']:
                raise ValueError('Committed stage identity/data changed: ' + name)
            for relative, expected in saved['artifact_sha256'].items():
                artifact = (self.root / relative).resolve()
                if not artifact.is_relative_to(self.root.resolve()) or sha(artifact) != expected:
                    raise ValueError('Committed stage artifact changed: ' + relative)
            return saved['data']
        result = operation()
        pins = {}
        for artifact in result.get('artifacts', []):
            artifact = Path(artifact).resolve()
            if not artifact.is_relative_to(self.root.resolve()):
                raise ValueError('Stage artifact escapes case output')
            pins[artifact.relative_to(self.root.resolve()).as_posix()] = sha(artifact)
        data = result['data']
        atomic_json(path, {'inputs_digest': fingerprint, 'data_digest': digest(data),
                           'data': data, 'artifact_sha256': pins})
        if self.hook:
            self.hook(name, path)
        return data


def verify_execution_authority(config, registration, lock_path):
    marker = json.loads(Path(lock_path).read_text())
    proc = Path(f"/proc/{marker['pid']}/stat")
    if not proc.exists() or proc.read_text().split()[21] != marker['process_start_ticks']:
        raise ValueError('Numerical owner creation time differs from its guard')
    phase = config['phase']
    if phase == 'pipeline_verification':
        if any(device.platform != 'cpu' for device in jax.devices()):
            raise ValueError('Functional search verification must use CPU')
        if marker['pid'] not in [os.getpid(), os.getppid()]:
            raise ValueError('Functional verifier does not own the global lock')
        return {'phase': phase, 'registration_sha256': None}
    if phase != 'revenue_search' or marker['pid'] != os.getpid():
        raise ValueError('Production search requires its own exclusive numerical owner')
    folder = Path(registration)
    plan = json.loads((folder / 'registration.json').read_text())
    seal = json.loads((folder / 'seal.json').read_text())
    if sha(folder / 'registration.json') != seal['registration_sha256']:
        raise ValueError('Experimental registration seal changed')
    if plan['source_sha256'] != source_digest() or plan['cases'][config['case_id']] != digest(config):
        raise ValueError('Case/source differs from experimental registration')
    if plan.get('automatic_retry') is not False or not plan.get('registered_before_new_samples'):
        raise ValueError('Registration lacks fixed prospective execution controls')
    from budget_auction.train.revenue_execution import guard_path
    backend=plan['execution_backend_by_case'][config['case_id']]
    if (marker.get('backend')!=backend or marker.get('case_id')!=config['case_id']
            or marker.get('registration_sha256')!=seal['registration_sha256']
            or guard_path(marker['project_root'],config['case_id'],backend).resolve()!=Path(lock_path).resolve()):
        raise ValueError('Numerical guard does not own this registered case/backend')
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise ValueError('Production search refuses a CPU fallback')
    return {'phase': phase, 'registration_sha256': seal['registration_sha256']}


def run_search(config, output, backend, *, lock_path, registration=None,
               resume_review=None, checkpoint_hook=None):
    authority = verify_execution_authority(config, registration, lock_path)
    if len(config['trials']) != 5 or config['candidate_count_per_phase'] != 2:
        raise ValueError('Five cold trials and two candidates per phase are required')
    if len(config['epoch_phases']) != 2 or config['stress_shortlist_count'] != 3:
        raise ValueError('Declare two phases and at most three stress candidates')
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'source_sha256': source_digest(), 'authority': authority}
    identity_path = root / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Case configuration, source or registration changed')
    elif any(root.iterdir()):
        raise ValueError('Unidentified search directory; inspect before reuse')
    else:
        atomic_json(identity_path, identity)
    terminal = root / 'completion.json'
    if terminal.exists():
        done = json.loads(terminal.read_text())
        if (done['identity_digest'] != digest(identity)
                or sha(root / 'result.json') != done['result_sha256']):
            raise ValueError('Terminal search result changed')
        if done.get('frozen_sha256') and sha(root / 'stages/frozen.json') != done['frozen_sha256']:
            raise ValueError('Frozen mechanism choice changed')
        for name, expected in done['files_sha256'].items():
            if sha(root / name) != expected:
                raise ValueError('Completed search artifact changed: ' + name)
        return json.loads((root / 'result.json').read_text())
    previous = sorted(root.glob('invocation_*.json'))
    reviewed = {} if resume_review is None else resume_review.get('checkpoints', {})
    if previous:
        old = json.loads(previous[-1].read_text())
        if (resume_review is None or resume_review['prior_invocation_sha256'] != sha(previous[-1])
                or old['state'] not in ['stopped_or_error_requires_review']):
            raise ValueError('Unfinished search requires an explicit prior-invocation review')
        if config['phase'] != 'pipeline_verification':
            proc = Path(f"/proc/{old['pid']}/stat")
            if proc.exists() and proc.read_text().split()[21] == old['process_start_ticks']:
                raise ValueError('Prior search owner is still alive')
        for relative, expected in reviewed.items():
            p = (root / relative).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p) != expected:
                raise ValueError('Reviewed checkpoint changed')
    if len(previous) >= config['max_invocations']:
        raise ValueError('Declared search invocation cap exhausted')
    invocation = root / f'invocation_{len(previous)+1:02d}.json'
    record = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'process_start_ticks': Path('/proc/self/stat').read_text().split()[21],
              'identity_digest': digest(identity), 'state': 'running', 'automatic_retry': False}
    atomic_json(invocation, record)
    started = time.perf_counter()
    store = StageStore(root, digest(identity), checkpoint_hook)
    def decision(name, inputs, value):
        saved = store.stage(name, inputs, lambda: {'data': value})
        if saved != value:
            raise ValueError('Committed validation decision disagrees with the fixed rule')
        return saved
    try:
        backend.bind(root, config, reviewed, checkpoint_hook)
        candidates, audits = [], []
        first, last = config['epoch_phases']
        for trial in range(5):
            trained = store.stage(f'train_{trial}_first', [trial, first],
                lambda trial=trial: backend.train(trial, first))
            screened = store.stage(f'screen_{trial}_first', trained,
                lambda trial=trial, trained=trained: backend.screen(trial, first, trained, 2*trial, 0))
            if len(screened) != 2:
                raise ValueError('A cold trial did not supply its two reserved candidates')
            for candidate in screened:
                audit = store.stage(f"audit_{candidate['candidate_index']}", candidate,
                                    lambda candidate=candidate: backend.audit(candidate))
                candidates.append(candidate); audits.append(audit)
        choice = decision('continuation_choice', audits,
                          choose_continuation(audits, config['selection_ic_threshold']))
        trial = choice['trial_index']
        trained = store.stage('train_chosen_continuation', [choice, last], lambda: backend.train(trial, last))
        continued = store.stage('screen_continued', [choice, trained],
            lambda: backend.screen(trial, last, trained, 10, first + config['training']['checkpoint_interval']))
        if len(continued) != 2:
            raise ValueError('Continuation did not supply its two reserved candidates')
        for candidate in continued:
            audit = store.stage(f"audit_{candidate['candidate_index']}", candidate,
                                lambda candidate=candidate: backend.audit(candidate))
            candidates.append(candidate); audits.append(audit)
        shortlisted = decision('stress_shortlist', audits,
                               select_stress(audits, config['selection_ic_threshold'], 3))
        stressed = []
        for selected in shortlisted:
            candidate = next(c for c in candidates if c['candidate_index'] == selected['candidate_index'])
            stressed.append(store.stage(f"stress_{selected['candidate_index']}", [selected, candidate],
                lambda candidate=candidate, selected=selected: backend.stress(candidate, selected)))
        passing = select_stress(stressed, config['selection_ic_threshold'], 1)
        if not passing:
            result = {'state': 'no_candidate_after_stress' if shortlisted else 'no_candidate',
                      'candidate_audits': audits, 'stress_results': stressed,
                      'final_samples_drawn': False, 'main_result_replacement': False}
        else:
            selected = decision('final_choice', stressed, passing[0])
            frozen = store.stage('frozen', selected, lambda: backend.freeze(selected))
            final = store.stage('final', frozen, lambda: backend.final(frozen))
            result = {'state': 'passed' if eligible(final, config['final_ic_threshold']) else 'failed_final',
                      'selected_candidate_index': selected['candidate_index'],
                      'candidate_audits': audits, 'stress_results': stressed, 'final': final,
                      'final_samples_drawn': True, 'main_result_replacement': False,
                      'claim': 'Best found by the fixed finite search;not global IC or optimal revenue'}
        result.update(phase=config['phase'], publication_eligible_phase=False,
                      identity_digest=digest(identity), continuation_choice=choice)
        atomic_json(root / 'result.json', result)
        atomic_json(terminal, {'identity_digest': digest(identity),
                              'result_sha256': sha(root / 'result.json'),
                              'frozen_sha256': sha(root / 'stages/frozen.json') if passing else None,
                              'files_sha256': {p.relative_to(root).as_posix(): sha(p)
                                  for p in sorted(root.rglob('*')) if p.is_file()
                                  and p != invocation and p != terminal
                                  and not (p.parent == root and p.name.startswith('invocation_'))}})
        record['state'] = 'complete'
        return result
    except BaseException as exc:
        record.update(state='stopped_or_error_requires_review', exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        record['seconds_this_invocation'] = time.perf_counter() - started
        atomic_json(invocation, record)
