"""Numerical stages for the separately registered finite revenue search."""
import json
from pathlib import Path
import shutil

import haiku as hk
import jax
import jax.numpy as jnp

from budget_auction.core import BudgetSimulator, build_bounded_policy_net
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
from budget_auction.train.es_audit import search_es
from budget_auction.train.evaluate import eval_budget_mechanism_multikey, summarize_deviation_gains, _summary_stats
from budget_auction.train.pilot import FACTORIES, atomic_json, atomic_pickle, source_digest
from budget_auction.train.revenue_flow import digest, sha
from budget_auction.train.revenue_trial import train_phase, read_pickle
from budget_auction.train.strong_audit import AuditPolicy, audit_capacity_ladder


def descriptor(policy):
    return {'params': policy.params, 'sigmas': policy.sigmas, 'modes': policy.modes}


class RevenueBackend:
    def bind(self, root, config, reviewed, hook):
        self.root, self.config, self.reviewed, self.hook = Path(root), config, reviewed, hook
        self.identity = digest({'config': config, 'source': source_digest()})
        self.auction = FACTORIES[config['setting']](seed=config['environment_seed'], **config['environment'])
        self.network = hk.without_apply_rng(hk.transform(VARIANT_NETWORK[config['variant']]))

    def seed(self, label):
        value = self.config['streams'][label]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
            raise ValueError('Invalid declared stream seed: ' + label)
        return value

    def checkpoint(self, path, state):
        atomic_pickle(path, state)
        if self.hook:
            self.hook('checkpoint:' + path.relative_to(self.root).as_posix(), path)

    def resume(self, path):
        if not path.exists():
            return None
        relative = path.relative_to(self.root).as_posix()
        if self.reviewed.get(relative) != sha(path):
            raise ValueError('Active numerical checkpoint requires explicit reviewed hash: ' + relative)
        return read_pickle(path)

    def files(self, folder):
        return [p for p in sorted(folder.rglob('*')) if p.is_file()]

    def component(self, folder, inputs, operation):
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / 'receipt.json'
        expected = digest({'identity': self.identity, 'inputs': inputs})
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['inputs_digest'] != expected:
                raise ValueError('Numerical component controls changed')
            for name, value in saved['files_sha256'].items():
                if sha(folder / name) != value:
                    raise ValueError('Completed numerical component changed')
            return json.loads((folder / 'summary.json').read_text()), read_pickle(folder / 'selected.pkl')
        if (folder / 'selected.pkl').exists() or (folder / 'summary.json').exists():
            raise ValueError('Unknown component commit outcome;inspect before continuation')
        summary, selected = operation()
        atomic_pickle(folder / 'selected.pkl', selected)
        atomic_json(folder / 'summary.json', summary)
        atomic_json(receipt, {'inputs_digest': expected,
                             'files_sha256': {p.relative_to(folder).as_posix(): sha(p) for p in self.files(folder)}})
        return summary, selected

    def train(self, trial, limit):
        entry = self.config['trials'][trial]
        options = {**self.config['training'], **{k: v for k, v in entry.items() if k not in ['seed', 'name']}}
        spec = {'setting': self.config['setting'], 'variant': self.config['variant'],
                'seed': entry['seed'], 'environment_seed': self.config['environment_seed'],
                'environment': self.config['environment'], 'policy_hidden': self.config['policy_hidden'],
                'epoch_phases': self.config['epoch_phases'], 'training': options,
                'search_identity': self.identity, 'trial_index': trial}
        folder = self.root / f'trial_{trial:02d}'
        checkpoint = folder / f'phase_{limit:06d}/resume.pkl'
        reviewed = self.reviewed.get(checkpoint.relative_to(self.root).as_posix())
        train_phase(spec, folder, limit, reviewed_resume_sha256=reviewed,
                    checkpoint_hook=lambda state: self.hook('checkpoint:' + checkpoint.relative_to(self.root).as_posix(), checkpoint) if self.hook else None)
        phase = folder / f'phase_{limit:06d}'
        return {'data': {'trial_index': trial, 'limit': limit,
                         'trained': (phase / 'trained.pkl').relative_to(self.root).as_posix()},
                'artifacts': self.files(phase) + [folder / 'identity.json']}

    def screen(self, trial, limit, trained, first_index, minimum_epoch):
        logs = read_pickle(self.root / trained['trained'])
        checkpoints = [p for p in logs['checkpoint_mech'] if p['epoch'] >= minimum_epoch]
        checkpoints += [{'epoch': 'final', 'mechanism_params': logs['mechanism_params'],
                         'policy_params': logs['policy_params']}]
        folder = self.root / f'screens/trial_{trial:02d}_{limit:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        policy = build_bounded_policy_net(self.auction.d_priv, self.auction.d_pub,
                                          self.auction.num_goods, hidden=self.config['policy_hidden'])
        rows = []
        for index, point in enumerate(checkpoints):
            path = folder / f'screen_{index:03d}.json'
            if path.exists():
                record = json.loads(path.read_text())
            else:
                value = eval_budget_mechanism_multikey(self.auction, self.network, point['mechanism_params'], 0.,
                    policy_network=policy, policy_params=point['policy_params'],
                    K=self.config['screen_batches'], num_samples=self.config['screen_samples'],
                    base_seed=self.seed(f'screen/{trial}/{limit}'))
                record = {'epoch': point['epoch'], 'revenue': value['profit_stats']['mean'],
                          'ic_sum': value['ic_sum'], 'role': 'training_policy_screen_not_IC_eligibility'}
                atomic_json(path, record)
            rows.append((index, record, point))
        rows.sort(key=lambda row: (row[1]['ic_sum'] > self.config['screen_ic_threshold'],
                                   -row[1]['revenue'], row[0]))
        result = []
        for offset, (_, record, point) in enumerate(rows[:2]):
            artifact = folder / f'candidate_{offset}.pkl'
            if not artifact.exists():
                atomic_pickle(artifact, {'mechanism_params': point['mechanism_params']})
            result.append({'candidate_index': first_index + offset, 'trial_index': trial,
                           'limit': limit, 'epoch': point['epoch'], 'screen': record,
                           'mechanism_artifact': artifact.relative_to(self.root).as_posix()})
        return {'data': result, 'artifacts': self.files(folder)}

    def policy(self, saved):
        networks = tuple(build_bounded_policy_net(self.auction.d_priv, self.auction.d_pub,
                         self.auction.num_goods, hidden=w) for w in saved['widths'])
        return AuditPolicy(networks, **saved['policy'])

    def ladder(self, candidate, folder, stress=False):
        params = read_pickle(self.root / candidate['mechanism_artifact'])['mechanism_params']
        index = candidate['candidate_index']; controls = self.config['stress_audit' if stress else 'audit']
        checkpoint = folder / 'resume.pkl'
        def execute():
            summary, policy = audit_capacity_ladder(self.auction, self.network, params,
                hidden_sizes=self.config['audit_hidden_sizes'],
                seed=self.seed(f"{'stress_ladder' if stress else 'audit'}/{index}"),
                ic_threshold=self.config['selection_ic_threshold'],
                resume_state=self.resume(checkpoint),
                checkpoint_callback=lambda state: self.checkpoint(checkpoint, state), **controls)
            state = read_pickle(checkpoint)
            return summary, {'mechanism_params': params, 'policy': descriptor(policy),
                             'widths': summary['selected_hidden_per_buyer'],
                             'rung_policies': [p['policy'] for p in state['completed_rungs']],
                             'rung_family_best': [p['search_state']['family_best']
                                                  for p in state['completed_rungs']]}
        return self.component(folder, ['ladder', candidate, stress, controls], execute)

    def audit(self, candidate):
        index = candidate['candidate_index']; folder = self.root / f'candidate_{index:02d}/regular'
        summary, saved = self.ladder(candidate, folder)
        revenue = eval_budget_mechanism_multikey(self.auction, self.network, saved['mechanism_params'], 0.,
                    K=self.config['screen_batches'], num_samples=self.config['screen_samples'],
                    base_seed=self.seed(f'candidate_revenue/{index}'))
        feasible = (revenue['budget_viol_stats']['mean'] == 0. and revenue['ir_viol_stats']['mean'] == 0.
                    and revenue['ir_loss_stats']['mean'] <= 1e-6)
        row = {**candidate, 'ic_upper': summary['ic_sum_sampling_ucb95'],
               'observed_feasible': feasible, 'revenue': revenue['profit_stats'],
               'raw_validation_revenue_batches': revenue['profit'].tolist(),
               'raw_validation_feasibility':{'budget_violation':revenue['budget_viol_rate'].tolist(),
                   'ir_violation':revenue['ir_viol_rate'].tolist(),'ir_loss':revenue['ir_loss'].tolist()},
               'regular_artifact': (folder / 'selected.pkl').relative_to(self.root).as_posix()}
        return {'data': row, 'artifacts': self.files(folder)}

    def evaluate_fixed(self, folder, saved, ic_seed, revenue_seed, threshold):
        """Persist each fixed batch; unknown begun batches never silently repeat."""
        folder.mkdir(parents=True, exist_ok=True)
        policy = self.policy(saved); mechanism = BudgetMechanism(self.network, saved['mechanism_params'])
        batches, samples = self.config['final_batches'], self.config['final_samples']
        rows = []
        for index in range(batches):
            batch = folder / f'batch_{index:03d}.json'; intent = folder / f'intent_{index:03d}.json'
            receipt = folder / f'completed_{index:03d}.json'
            identity = {'search_identity': self.identity, 'policy_sha256': digest({'artifact': saved['frozen_artifact_sha256']}),
                        'ic_seed': ic_seed, 'revenue_seed': revenue_seed, 'batch_index': index, 'samples': samples}
            if batch.exists():
                result = json.loads(batch.read_text())
                if json.loads(intent.read_text()) != identity or result['identity'] != identity:
                    raise ValueError('Frozen batch identity changed')
                if not receipt.exists() or json.loads(receipt.read_text())['batch_sha256'] != sha(batch):
                    raise ValueError('Frozen batch commit is incomplete or changed')
            else:
                if intent.exists():
                    raise ValueError('Unknown evaluation-batch outcome;no automatic repeat or new samples')
                atomic_json(intent, identity)
                key = jax.random.fold_in(jax.random.PRNGKey(ic_seed), index)
                gains = BudgetSimulator.estimate_cumulative_deviation_gains(self.auction, mechanism, policy, samples, key)
                key = jax.random.fold_in(jax.random.PRNGKey(revenue_seed), index)
                k1, k2, k3 = jax.random.split(key, 3)
                profit = BudgetSimulator.estimate_cumulative_profit(self.auction, mechanism, samples, k1)
                rewards = BudgetSimulator.estimate_cumulative_rewards(self.auction, mechanism, samples, k2)
                bv, iv = BudgetSimulator.estimate_violation_rates(self.auction, mechanism, samples, k3)
                result = {'identity': identity, 'gains': gains.tolist(), 'revenue': float(profit),
                          'ir_loss': float(jnp.sum(jax.nn.relu(-rewards))),
                          'budget_violation': float(bv), 'ir_violation': float(iv)}
                atomic_json(batch, result)
                atomic_json(receipt, {'batch_sha256': sha(batch)})
                if self.hook:
                    self.hook('completed_batch:' + batch.relative_to(self.root).as_posix(), batch)
            rows.append(result)
        audit = summarize_deviation_gains(jnp.asarray([r['gains'] for r in rows]))
        summary = {'ic_upper': audit['ic_sum_sampling_ucb95'], 'audit': audit,
                   'revenue': _summary_stats(jnp.asarray([r['revenue'] for r in rows])),
                   'observed_feasible': all(r['budget_violation'] == 0 and r['ir_violation'] == 0
                                            and r['ir_loss'] <= 1e-6 for r in rows),
                   'batches': batches, 'samples_per_batch': samples, 'ic_threshold': threshold,
                   'raw_batches': rows}
        return summary

    def stress(self, candidate, regular):
        index = candidate['candidate_index']; folder = self.root / f'candidate_{index:02d}/stress'
        _, stronger = self.ladder(candidate, folder / 'ladder', True)
        original = read_pickle(self.root / regular['regular_artifact'])
        old_policy, new_policy = self.policy(original), self.policy(stronger)
        mechanism = BudgetMechanism(self.network, stronger['mechanism_params'])
        controls = self.config['es']; width = controls['hidden']; initializations = []
        warm_index = self.config['audit_hidden_sizes'].index(width)
        state = self.auction.get_initial_state(jax.random.fold_in(
            jax.random.PRNGKey(self.seed(f'es_init/{index}/0/0')), 0))
        belief = self.auction.get_initial_belief_state(state)
        for buyer in range(self.auction.num_buyers):
            choices = []
            for restart in range(3):
                std = .1 if restart == 2 else 0.
                network = build_bounded_policy_net(self.auction.d_priv, self.auction.d_pub,
                    self.auction.num_goods, hidden=width, residual_init_std=std)
                warm = original['rung_policies'][warm_index]
                if restart == 0 and warm['modes'][buyer] not in ['truth', 'low', 'high', 'half']:
                    params = warm['params'][buyer]
                else:
                    params = network.init(jax.random.fold_in(
                        jax.random.PRNGKey(self.seed(f'es_init/{index}/{buyer}/{restart}')), 1), belief, self.auction)
                part = folder / f'es_b{buyer}_r{restart}'; checkpoint = part / 'resume.pkl'
                def execute(network=network, params=params, checkpoint=checkpoint, buyer=buyer, restart=restart):
                    return search_es(self.auction, mechanism, network, params, buyer,
                        identity=digest([self.identity, candidate, buyer, restart]),
                        seed=self.seed(f'es_fit/{index}/{buyer}/{restart}'),
                        resume_state=self.resume(checkpoint),
                        checkpoint_callback=lambda value: self.checkpoint(checkpoint, value),
                        **{k: v for k, v in controls.items() if k != 'hidden'})
                _, params = self.component(part, ['es', candidate, buyer, restart, controls], execute)
                choices.append((network, params, 0., 'network', width, f'es_restart{restart}'))
            initializations.append(choices)
        def build_merge():
            merge_key = jax.random.PRNGKey(self.seed(f'merge/{index}'))
            rows, trace = [], []
            for buyer in range(self.auction.num_buyers):
                default_network, default_params, _, _, _, _ = initializations[buyer][0]
                options = [(default_network, default_params, 0., mode, width, mode)
                           for mode in ['truth', 'low', 'high', 'half']]
                for name, policy, saved in [('retained', old_policy, original), ('expanded', new_policy, stronger)]:
                    options.append((policy.network[buyer], policy.params[buyer], float(policy.sigmas[buyer]),
                                    policy.modes[buyer], saved['widths'][buyer], name))
                    for rung, families in enumerate(saved['rung_family_best']):
                        hidden = self.config['audit_hidden_sizes'][rung]
                        family_network = build_bounded_policy_net(self.auction.d_priv, self.auction.d_pub,
                                                                  self.auction.num_goods, hidden=hidden)
                        for family in ['pathwise', 'npg_mean', 'npg_stochastic']:
                            entry = families.get(f'{buyer}/{family}')
                            if entry is not None:
                                options.append((family_network, entry['params'], entry['sigma'],
                                                entry['mode'], hidden, f'{name}_w{hidden}_{family}'))
                options.extend(initializations[buyer]); measured = []
                for option in options:
                    network, params, sigma, mode, _, name = option
                    policy = AuditPolicy(network, tuple([params]*self.auction.num_buyers),
                                         jnp.full((self.auction.num_buyers,), sigma), tuple([mode]*self.auction.num_buyers))
                    gain = float(BudgetSimulator.estimate_cumulative_deviation_gains(self.auction, mechanism,
                        policy, self.config['merge_validation_samples'], merge_key)[buyer])
                    trace.append({'buyer': buyer, 'family': name, 'gain': gain})
                    measured.append(gain)
                winner = max(range(len(options)), key=lambda k: measured[k])
                rows.append(options[winner])
            policy = AuditPolicy(tuple(r[0] for r in rows), tuple(r[1] for r in rows),
                                 jnp.asarray([r[2] for r in rows]), tuple(r[3] for r in rows))
            saved = {'mechanism_params': stronger['mechanism_params'], 'policy': descriptor(policy),
                     'widths': [r[4] for r in rows], 'families': [r[5] for r in rows]}
            return {'trace': trace,'selected_families':saved['families'],
                    'selected_widths':saved['widths'],'selected_modes':list(policy.modes),
                    'selected_sigmas':[float(x) for x in policy.sigmas]}, saved
        merge_folder = folder / 'merge'
        merge_inputs = ['merge', candidate, sha(folder / 'ladder/selected.pkl'),
                        [sha(folder / f'es_b{buyer}_r{restart}/selected.pkl')
                         for buyer in range(self.auction.num_buyers) for restart in range(3)]]
        merged, saved = self.component(merge_folder, merge_inputs, build_merge)
        trace = merged['trace']
        artifact = merge_folder / 'selected.pkl'
        saved['frozen_artifact_sha256'] = sha(artifact)
        summary = self.evaluate_fixed(folder / 'evaluation', saved, self.seed(f'stress_ic/{index}'),
                                      self.seed(f'stress_revenue/{index}'), self.config['selection_ic_threshold'])
        summary.update(candidate_index=index, trial_index=candidate['trial_index'], merge_trace=trace,
                       stress_artifact=artifact.relative_to(self.root).as_posix())
        return {'data': summary, 'artifacts': self.files(folder)}

    def freeze(self, selected):
        source = self.root / selected['stress_artifact']; destination = self.root / 'frozen_selection.pkl'
        if destination.exists():
            if sha(destination) != sha(source): raise ValueError('Frozen selection changed')
        else:
            shutil.copy2(source, destination)
        return {'data': {'candidate_index': selected['candidate_index'],
                         'artifact': destination.relative_to(self.root).as_posix(), 'sha256': sha(destination)},
                'artifacts': [destination]}

    def final(self, frozen):
        artifact = self.root / frozen['artifact']
        if sha(artifact) != frozen['sha256']: raise ValueError('Frozen policy/mechanism changed')
        saved = read_pickle(artifact); saved['frozen_artifact_sha256'] = sha(artifact)
        folder = self.root / 'final_evaluation'
        summary = self.evaluate_fixed(folder, saved, self.seed('final_ic'), self.seed('final_revenue'),
                                      self.config['final_ic_threshold'])
        return {'data': summary, 'artifacts': self.files(folder) + [artifact]}
