"""Approved bounded auxiliary paths; all numerical kernels are pinned v04 imports."""
from pathlib import Path
import json
import shutil
from adaptive_engine import Engine as Warm, now
from capacity_engine_original import Engine as Cold
from capacity_engine import EPOCH_INDEX, ranked
from adaptive_rules import eligible, accepted
from focused_engine import release_compilation_cache
from budget_auction.train.pilot import atomic_json
from budget_auction.train.revenue_flow import sha


class Engine(Warm):
    def stage(self, name, inputs, operation):
        result = super().stage(name, inputs, operation)
        release_compilation_cache()
        return result

    def cold_train(self):
        result = Cold.train(self, 0, self.c['target'])
        for candidate in result['data']:
            candidate['candidate_index'] = EPOCH_INDEX[candidate['epoch']]
        return result

    def incumbent(self):
        c = self.c
        source = self.reg / c['incumbent_path']
        assert sha(source) == c['incumbent_sha256']
        row = json.loads(source.read_text())
        assert eligible(row, .04)
        artifact = self.reg / c['incumbent_artifact']
        assert sha(artifact) == c['incumbent_artifact_sha256']
        dest = self.root / 'incumbent.pkl'
        shutil.copyfile(artifact, dest)
        row['stress_artifact'] = self.artifact(dest)
        return {'data': row, 'artifacts': [dest]}

    def run(self):
        c = self.c
        assert c['auxiliary_kind'] in ['lower_target', 'refinement', 'distillation']
        assert c['unit_kind'] == 'target' and c['max_attempts'] == 1
        incumbent = None
        if c['auxiliary_kind'] == 'lower_target':
            candidates = self.stage('train_00', {'cold': True, 'target': c['target']}, self.cold_train)
        else:
            initial = self.initial()
            if c['auxiliary_kind'] == 'refinement':
                incumbent = self.stage('incumbent', c['incumbent_sha256'], self.incumbent)
            candidates = self.stage('train_00', {'initial': initial, 'sha256': sha(self.root / initial), 'target': c['target']},
                                    lambda: Warm.train(self, 0, c['target'], initial))
        regular = [self.stage(f"regular_{x['candidate_index']:02d}", x,
                              lambda x=x: self.backend.audit(x)) for x in candidates]
        order = ranked(regular)
        attempts, selected = [], None
        for rank, row in enumerate(order[:2]):
            candidate = next(x for x in candidates if x['candidate_index'] == row['candidate_index'])
            strong = self.stage(f'strong_{rank:02d}', [candidate, row],
                                lambda candidate=candidate, row=row: self.backend.stress(candidate, row))
            attempts.append(strong)
            if eligible(strong, .04):
                selected = strong
                break
        retained = selected
        retained_origin = 'new_candidate' if selected else None
        if incumbent is not None and (retained is None or incumbent['revenue']['mean'] >= retained['revenue']['mean']):
            retained, retained_origin = incumbent, 'incumbent'
        result = {'state': 'target_complete', 'case_id': c['case_id'], 'logical_case': c['logical_case'],
                  'original_unit': c['original_unit'], 'unit_kind': 'target', 'target': c['target'],
                  'auxiliary_kind': c['auxiliary_kind'], 'strong': selected, 'regular': regular,
                  'strong_attempts': attempts, 'screen_ranking': [x['candidate_index'] for x in order],
                  'attained': bool(selected and accepted(selected, c['target'])),
                  'incumbent': incumbent, 'retained': retained, 'retained_origin': retained_origin,
                  'final': None, 'independent_final_tests': 0,
                  'cold_start': c['auxiliary_kind'] == 'lower_target', 'epochs': c['epochs'],
                  'selection_mode': 'auxiliary_validation_only', 'checkpoint_epochs': c['checkpoint_epochs'],
                  'phase': 'approved_auxiliary_v01', 'matched_control': False,
                  'claim': 'Selected validation performance; exploratory, no independent test or causal distillation claim.'}
        atomic_json(self.root / 'result.json', result)
        pins = {p.relative_to(self.root).as_posix(): sha(p) for p in sorted(self.root.rglob('*'))
                if p.is_file() and p.name != 'completion.json'}
        atomic_json(self.root / 'completion.json', {'identity': self.identity, 'result_sha256': sha(self.root / 'result.json'),
                                                   'files_sha256': pins, 'completed_at': now()})
        return result
