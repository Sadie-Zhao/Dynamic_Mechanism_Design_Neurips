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


from adaptive_engine import hk,jax,jnp,time,finite,read_pickle,data_digest,distill_mechanism_A,evaluate_distillation,eval_budget_mechanism_multikey,mechanism_observation,atomic_pickle
from continuation import extend_supervised_state

class Engine(Warm):
    def initial(self):
        src=self.reg/self.c['input_path'];assert sha(src)==self.c['input_sha256']
        if 'toA' not in self.c['arm']:
            def copy():
                p=self.root/'initial.pkl';atomic_pickle(p,{'mechanism_params':read_pickle(src)['mechanism_params']})
                return {'data':{'artifact':self.artifact(p)},'artifacts':[p]}
            return self.stage('initial',self.c['input_sha256'],copy)['artifact']
        b=self.backend;out=self.root/'distillation';out.mkdir(exist_ok=True)
        def data():
            paths=[]
            for role in ['data','validation']:
                src500=self.reg/self.c['supervised_continuation']['path']/(role+'.pkl')
                assert sha(src500)==self.c['supervised_continuation']['files_sha256'][role+'.pkl']
                p=out/(role+'.pkl');assert not p.exists()
                shutil.copyfile(src500,p);paths.append(p)
            return {'data':{'train_sha':sha(paths[0]),'validation_sha':sha(paths[1])},'artifacts':paths}
        dataset=self.stage('distill_data',self.c['input_sha256'],data)
        def fit():
            train=jax.tree_util.tree_map(jnp.asarray,read_pickle(out/'data.pkl'))
            validation=jax.tree_util.tree_map(jnp.asarray,read_pickle(out/'validation.pkl'))
            key=jax.random.PRNGKey(b.seed('distill_init'));k1,k2=jax.random.split(key)
            state=b.auction.get_initial_state(k1)
            initial=b.network.init(k2,mechanism_observation(state),jnp.zeros((b.auction.num_buyers,b.auction.num_goods)),b.auction)
            checkpoint=out/'fit_resume.pkl';assert not checkpoint.exists(), 'No automatic numerical retry'
            origin=self.reg/self.c['supervised_continuation']['path']
            pins=self.c['supervised_continuation']['files_sha256']
            assert all(sha(origin/n)==h for n,h in pins.items())
            original_state=read_pickle(origin/'fit_resume.pkl')
            expected=dict(num_epochs=500,lr=.001,batch_size=256,alloc_weight=1.,payment_weight=1.,eval_every=10,
                data_sha256=data_digest(train),validation_sha256=data_digest(validation))
            original_history=json.loads((origin/'loss_trajectory.json').read_text())
            resume=extend_supervised_state(original_state,expected,original_history)
            finite((resume['params'],resume['opt_state'],resume['best_params'],resume['rng_state']))
            atomic_json(out/'continuation_proof.json',{'origin_checkpoint_sha256':pins['fit_resume.pkl'],
                'original_controls':original_state['controls'],'extended_controls':resume['controls'],
                'from_epoch':500,'to_epoch':1000,'preserved_state_fields':['params','opt_state','rng_state','loss_history','validation_history','best_loss','best_params','best_epoch'],
                'original_history_sha256':pins['loss_trajectory.json'],'post_training_restart':True})
            inv,record,started=self.phase_invocation(out,resume)
            def save(s):
                finite((s['params'],s['opt_state'],s['best_params']))
                atomic_pickle(checkpoint,s)
                if s['next_epoch']%10==0:
                    atomic_json(out/'loss_trajectory.json',{'loss':s['loss_history'],'validation':s['validation_history'],'epoch':s['next_epoch']})
                self.hook('distill_checkpoint',checkpoint)
            try:
                fitted,loss=distill_mechanism_A(b.auction,b.network,initial,*train,num_epochs=self.c['distill']['epochs'],
                    lr=self.c['distill']['learning_rate'],batch_size=self.c['distill']['batch_size'],
                    key=jax.random.PRNGKey(b.seed('distill_fit')),validation_data=validation,resume_state=resume,checkpoint_callback=save)
                assert len(loss)==1000
                atomic_pickle(out/'student.pkl',{'mechanism_params':fitted})
                atomic_json(out/'fit_validation.json',evaluate_distillation(b.auction,b.network,fitted,validation))
                record['state']='complete'
            except BaseException as exc:
                record.update(state='interrupted_requires_review',error=str(exc));raise
            finally:
                record.update(finished_at=now(),seconds=time.perf_counter()-started);atomic_json(inv,record)
            return {'data':{'artifact':self.artifact(out/'student.pkl')},'artifacts':[p for p in out.iterdir() if p.is_file()]}
        result=self.stage('distill_fit',dataset,fit)
        def diagnostic():
            params=read_pickle(self.root/result['artifact'])['mechanism_params']
            measured=eval_budget_mechanism_multikey(b.auction,b.network,params,0.,K=4,num_samples=128,base_seed=b.seed('distill_diagnostic'))
            p=out/'pure_imitation_diagnostic.json';atomic_json(p,{'revenue':measured['profit_stats'],'IC_certified':False,'role':'pure_imitation_diagnostic_only'})
            return {'data':{'path':self.artifact(p)},'artifacts':[p]}
        self.stage('distill_diagnostic',result,diagnostic)
        return result['artifact']

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
                  'phase': 'approved_distillation1000', 'matched_control': False,
                  'claim': 'Selected validation performance; exploratory, no independent test or causal distillation claim.'}
        atomic_json(self.root / 'result.json', result)
        pins = {p.relative_to(self.root).as_posix(): sha(p) for p in sorted(self.root.rglob('*'))
                if p.is_file() and p.name != 'completion.json'}
        atomic_json(self.root / 'completion.json', {'identity': self.identity, 'result_sha256': sha(self.root / 'result.json'),
                                                   'files_sha256': pins, 'completed_at': now()})
        return result
