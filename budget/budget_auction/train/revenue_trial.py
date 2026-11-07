"""Source-pinned phases for the prospective search, without audit/final testing.

The external registered worker must hold the global numerical-work lock. Calling
this helper is not registration or evidence of IC feasibility/optimal revenue.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import pickle
from pathlib import Path
import time

import haiku as hk

from budget_auction.core import build_bounded_policy_net
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
from budget_auction.train.pilot import FACTORIES, atomic_json, atomic_pickle, source_digest
from budget_auction.train.run_acceptance import nested_GD_budget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def train_phase(config, output, target_epoch, *, reviewed_resume_sha256=None,
                checkpoint_hook=None):
    """Explicit scheduled extension or a separately reviewed interrupted resume.

    A completed phase is reused without training. No first phase is launched
    implicitly to satisfy a second-phase request, and no partial run restarts.
    """
    phases = config['epoch_phases']
    if (not phases or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in phases)
            or sorted(set(phases)) != phases or len(phases) > 2):
        raise ValueError('Declare one or two strictly increasing positive epoch phases')
    if target_epoch not in phases or isinstance(target_epoch, bool):
        raise ValueError('Epoch budget is outside the prospective schedule')
    if 'num_outer_epochs' in config['training']:
        raise ValueError('Epoch budget must be declared only in epoch_phases')
    identity = {'config': config, 'config_sha256': digest(config), 'source_sha256': source_digest()}
    identity_digest = digest(identity)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'identity.json').exists():
        if json.loads((root / 'identity.json').read_text()) != identity:
            raise ValueError('Configuration or numerical source changed')
    elif any(root.iterdir()):
        raise ValueError('Unidentified output directory; review before reuse')
    else:
        atomic_json(root / 'identity.json', identity)
    phase = root / f'phase_{target_epoch:06d}'
    phase.mkdir(exist_ok=True)
    trained, receipt, checkpoint = phase / 'trained.pkl', phase / 'completion.json', phase / 'resume.pkl'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved['identity_digest'] != identity_digest or saved['target_epoch'] != target_epoch
                or sha(trained) != saved['trained_sha256'] or sha(checkpoint) != saved['checkpoint_sha256']
                or sha(phase/'training_evidence.json') != saved['training_evidence_sha256']):
            raise ValueError('Completed phase evidence changed')
        if any(json.loads(p.read_text())['state'] != 'complete'
               and json.loads(p.read_text())['state'] != 'interrupted_or_error_requires_review'
               for p in phase.glob('invocation_*.json')):
            raise ValueError('Prior invocation is not terminal; review its ownership')
        return read_pickle(trained)
    if trained.exists():
        raise ValueError('Unknown outcome: trained artifact lacks completion receipt')
    state, extension, state_sha = None, None, None
    if checkpoint.exists():
        state_sha = sha(checkpoint)
        if reviewed_resume_sha256 != state_sha:
            raise ValueError('Interrupted phase requires explicit reviewed checkpoint hash')
        state = read_pickle(checkpoint)
    elif reviewed_resume_sha256 is not None:
        raise ValueError('Reviewed checkpoint is missing; refuse initialization')
    elif target_epoch != phases[0]:
        prior_epoch = phases[phases.index(target_epoch) - 1]
        prior = root / f'phase_{prior_epoch:06d}'
        if not (prior / 'completion.json').exists():
            raise ValueError('Scheduled continuation requires a completed prior phase')
        previous = json.loads((prior / 'completion.json').read_text())
        if (previous['identity_digest'] != identity_digest
                or previous['target_epoch'] != prior_epoch
                or sha(prior / 'trained.pkl') != previous['trained_sha256']
                or sha(prior / 'resume.pkl') != previous['checkpoint_sha256']
                or sha(prior / 'training_evidence.json') != previous['training_evidence_sha256']):
            raise ValueError('Completed prior phase changed')
        state_sha = sha(prior / 'resume.pkl')
        state, extension = read_pickle(prior / 'resume.pkl'), prior_epoch
    invocations = sorted(phase.glob('invocation_*.json'))
    if invocations and state is None:
        raise ValueError('Prior invocation has no exact checkpoint; refuse restart')
    if len(invocations) >= 2:
        raise ValueError('At most two reviewed invocations per phase')
    invocation = phase / f'invocation_{len(invocations) + 1:02d}.json'
    record = {'started_at': datetime.now(timezone.utc).isoformat(),
              'identity_digest': identity_digest, 'target_epoch': target_epoch,
              'resumed_from_epoch': 0 if state is None else state['next_epoch'],
              'loaded_checkpoint_sha256': state_sha, 'scheduled_extension_from': extension,
              'state': 'running', 'automatic_retry': False}
    atomic_json(invocation, record)
    started = time.perf_counter()
    def save(current):
        atomic_pickle(checkpoint, current)
        if checkpoint_hook is not None:
            checkpoint_hook(current)
    try:
        auction = FACTORIES[config['setting']](seed=config.get('environment_seed', config['seed']),
                                              **config.get('environment', {}))
        network = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[config['variant']]))
        policy = build_bounded_policy_net(auction.d_priv, auction.d_pub, auction.num_goods,
                                          hidden=config['policy_hidden'])
        logs = nested_GD_budget(auction, network, policy, **config['training'],
                               num_outer_epochs=target_epoch, seed=config['seed'],
                               resume_identity=identity_digest, resume_state=state,
                               extend_from_epochs=extension, checkpoint_callback=save)
        if len(logs['loss']) != target_epoch:
            raise ValueError('Full declared training budget did not finish')
        import jax
        import numpy as np
        final_state=read_pickle(checkpoint)
        leaves=jax.tree_util.tree_leaves((logs['mechanism_params'],logs['policy_params'],
            final_state['theta_opt_state'],final_state['w_opt_state'],final_state['rng_state'],
            [(p['mechanism_params'],p['policy_params']) for p in logs['checkpoint_mech']]))
        if not all(np.isfinite(np.asarray(value)).all() for value in leaves):
            raise FloatingPointError('Nonfinite final/checkpoint parameters,optimizer or RNG state')
        scalar_names=['loss','reg','gap','ir','reg_plus','gap_plus','revenue','oracle_reg_last']
        if any(len(logs[name])!=target_epoch or not all(math.isfinite(x) for x in logs[name]) for name in scalar_names):
            raise FloatingPointError('Incomplete or nonfinite scalar training history')
        evidence={'identity_digest':identity_digest,'completed_epochs':target_epoch,
            'num_inner_steps_per_epoch':config['training']['num_inner_steps'],
            'objective':config['training']['revenue_objective'],'training_config':config['training'],
            'scalar_logs':{name:logs[name] for name in scalar_names},
            'checkpoint_epochs':[p['epoch'] for p in logs['checkpoint_mech']],
            'finite_numeric_leaves_checked':len(leaves),'all_checked_numeric_leaves_finite':True,
            'mechanism_parameter_count':sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(logs['mechanism_params'])),
            'mechanism_optimizer_policy_and_rng_preserved':True}
        atomic_json(phase/'training_evidence.json',evidence)
        atomic_pickle(trained, logs)
        atomic_json(receipt, {'completed_at': datetime.now(timezone.utc).isoformat(),
                             'target_epoch': target_epoch, 'identity_digest': identity_digest,
                             'trained_sha256': sha(trained), 'checkpoint_sha256': sha(checkpoint),
                             'completed_epochs': len(logs['loss']),
                             'training_evidence_sha256':sha(phase/'training_evidence.json'),
                             'objective': config['training']['revenue_objective']})
        record['state'] = 'complete'
        return logs
    except BaseException as exc:
        record.update(state='interrupted_or_error_requires_review',
                      exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      seconds_this_invocation=time.perf_counter() - started)
        atomic_json(invocation, record)
