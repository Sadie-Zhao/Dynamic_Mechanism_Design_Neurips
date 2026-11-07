"""New warm-start adaptive study; old numerical components imported unchanged."""
from pathlib import Path
import json, time, math
from datetime import datetime,timezone
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from adaptive_rules import decision,accepted,eligible,select,lower_bound
from budget_auction.train.revenue_backend import RevenueBackend
from budget_auction.train.revenue_flow import StageStore,digest,sha
from budget_auction.train.revenue_trial import read_pickle
from budget_auction.train.pilot import atomic_json,atomic_pickle,source_digest
from budget_auction.train.run_acceptance import nested_GD_budget
from budget_auction.train.distill import collect_distillation_data,distill_mechanism_A,evaluate_distillation,data_digest
from budget_auction.train.evaluate import eval_budget_mechanism_multikey
from budget_auction.core import build_bounded_policy_net
from budget_auction.mechanisms.budget_mechanism import BudgetMechanism,mechanism_observation
from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK

SCALARS=['loss','reg','gap','ir','reg_plus','gap_plus','revenue','oracle_reg_last']
def now(): return datetime.now(timezone.utc).isoformat()
def finite(tree):
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree)):
        raise FloatingPointError('Nonfinite numerical state')
class Engine:
    def __init__(self,config,root,registration,reviewed,hook,spent,authority):
        self.c,self.root,self.reg,self.reviewed,self.hook,self.spent=config,Path(root),Path(registration),reviewed,hook,spent
        self.root.mkdir(parents=True,exist_ok=True)
        identity={'config':config,'authority':authority,'source':source_digest()}
        path=self.root/'identity.json'
        if path.exists(): assert json.loads(path.read_text())==identity
        else: atomic_json(path,identity)
        self.identity=digest(identity)
        self.backend=RevenueBackend();self.backend.bind(self.root,config,reviewed,hook)
        self.store=StageStore(self.root,self.identity,hook)
    def stage(self,name,inputs,operation):return self.store.stage(name,inputs,operation)
    def resume(self,p):
        if not p.exists():return None
        assert self.reviewed.get(p.relative_to(self.root).as_posix())==sha(p),'Explicit checkpoint review required'
        return read_pickle(p)
    def artifact(self,p):return p.relative_to(self.root).as_posix()
    def phase_invocation(self,folder,resume):
        prior=list(folder.glob('invocation_*.json'))
        assert len(prior)<self.c['max_phase_invocations'],'Phase invocation cap'
        assert not prior or resume is not None,'Refuse restart without exact state'
        p=folder/f'invocation_{len(prior)+1:02d}.json'
        record={'started_at':now(),'state':'running','resume_epoch':None if resume is None else resume['next_epoch']}
        atomic_json(p,record);return p,record,time.perf_counter()
    def initial(self):
        src=self.reg/self.c['input_path'];assert sha(src)==self.c['input_sha256']
        if 'toA' not in self.c['arm']:
            def copy():
                p=self.root/'initial.pkl';atomic_pickle(p,{'mechanism_params':read_pickle(src)['mechanism_params']})
                return {'data':{'artifact':self.artifact(p)},'artifacts':[p]}
            return self.stage('initial',self.c['input_sha256'],copy)['artifact']
        b=self.backend;out=self.root/'distillation';out.mkdir(exist_ok=True)
        def data():
            teacher_net=hk.without_apply_rng(hk.transform(VARIANT_NETWORK[self.c['teacher_variant']]))
            teacher=BudgetMechanism(teacher_net,read_pickle(src)['mechanism_params'])
            paths=[]
            for role,n in [('data',self.c['distill']['train_trajectories']),('validation',self.c['distill']['validation_trajectories'])]:
                p=out/(role+'.pkl')
                if p.exists(): raise ValueError('Uncommitted distillation dataset needs review')
                values=collect_distillation_data(b.auction,teacher,n,jax.random.PRNGKey(b.seed('distill_'+role)))
                atomic_pickle(p,values);paths.append(p)
            return {'data':{'train_sha':sha(paths[0]),'validation_sha':sha(paths[1])},'artifacts':paths}
        dataset=self.stage('distill_data',self.c['input_sha256'],data)
        def fit():
            train=jax.tree_util.tree_map(jnp.asarray,read_pickle(out/'data.pkl'))
            validation=jax.tree_util.tree_map(jnp.asarray,read_pickle(out/'validation.pkl'))
            key=jax.random.PRNGKey(b.seed('distill_init'));k1,k2=jax.random.split(key)
            state=b.auction.get_initial_state(k1)
            initial=b.network.init(k2,mechanism_observation(state),jnp.zeros((b.auction.num_buyers,b.auction.num_goods)),b.auction)
            checkpoint=out/'fit_resume.pkl';resume=self.resume(checkpoint)
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
    def train(self,index,target,initial):
        folder=self.root/f'target_{index:02d}';folder.mkdir(exist_ok=True)
        checkpoint=folder/'resume.pkl';resume=self.resume(checkpoint)
        inv,record,started=self.phase_invocation(folder,resume)
        identity=digest([self.identity,index,target,sha(self.root/initial)])
        def save(s):
            finite((s['mechanism_params'],s['policy_params'],s['theta_opt_state'],s['w_opt_state']))
            atomic_pickle(checkpoint,s)
            atomic_json(folder/'training_evidence.json',{'epoch':s['next_epoch'],'target':target,'scalar_logs':{k:s['logs'][k] for k in SCALARS},'training_only':True})
            if s['next_epoch'] in self.c['checkpoint_epochs']:
                atomic_pickle(folder/f"candidate_epoch{s['next_epoch']:04d}.pkl",{'mechanism_params':s['mechanism_params']})
            self.hook('training_checkpoint',checkpoint)
        try:
            options=dict(self.c['training'],profit_goal=target,alpha_weight=self.c['alpha_weight'],revenue_objective='acceptance')
            policy=build_bounded_policy_net(self.backend.auction.d_priv,self.backend.auction.d_pub,self.backend.auction.num_goods,hidden=self.c['policy_hidden'])
            logs=nested_GD_budget(self.backend.auction,self.backend.network,policy,**options,
              num_outer_epochs=self.c['epochs'],seed=self.backend.seed(f'train/{index}'),resume_identity=identity,resume_state=resume,
              init_mechanism_params=None if resume else read_pickle(self.root/initial)['mechanism_params'],checkpoint_callback=save)
            assert len(logs['loss'])==self.c['epochs']
            assert all(math.isfinite(v) for k in SCALARS for v in logs[k])
            atomic_pickle(folder/'trained.pkl',logs);record['state']='complete'
        except BaseException as exc:
            record.update(state='interrupted_requires_review',error=str(exc));raise
        finally:
            record.update(finished_at=now(),seconds=time.perf_counter()-started);atomic_json(inv,record)
        candidates=[{'candidate_index':2*index+j,'trial_index':index,'epoch':epoch,'target':target,
                     'mechanism_artifact':self.artifact(folder/f'candidate_epoch{epoch:04d}.pkl')}
                    for j,epoch in enumerate(self.c['checkpoint_epochs'])]
        return {'data':candidates,'artifacts':[p for p in folder.iterdir() if p.is_file()]}
    def run(self):
        initial=self.initial();history=[];strong_rows=[];incumbent=initial
        for index in range(self.c['max_attempts']):
            d=decision(history,self.c['interval'],self.c['universal_revenue_ceiling'],self.c['max_attempts'])
            if d['next'] is None:break
            # A previously begun target may finish; new targets obey the training cutoff.
            if not (self.root/f'target_{index:02d}').exists() and (self.spent()>=self.c['stop_training_seconds'] or now()>=self.c['train_cutoff_utc']):break
            target=d['next'];warm=incumbent
            candidates=self.stage(f'train_{index:02d}',[index,target,warm,sha(self.root/warm)],lambda:self.train(index,target,warm))
            regular=[self.stage(f'regular_{c["candidate_index"]:02d}',c,lambda c=c:self.backend.audit(c)) for c in candidates]
            chosen=select(regular);strong=None
            if chosen is not None:
                candidate=next(c for c in candidates if c['candidate_index']==chosen['candidate_index'])
                strong=self.stage(f'strong_{index:02d}',[candidate,chosen],lambda:self.backend.stress(candidate,chosen))
                strong_rows.append(strong)
                if eligible(strong):
                    # Keep highest eligible validation-revenue incumbent even when target was missed.
                    best=select(strong_rows);incumbent=best['stress_artifact']
            history.append({'attempt':index,'target':target,'accepted':accepted(strong,target) if strong else False,'strong':strong})
            atomic_json(self.root/'target_history.json',history)
            atomic_json(self.root/'bracket.json',decision(history,self.c['interval'],self.c['universal_revenue_ceiling'],self.c['max_attempts']))
        bracket=decision(history,self.c['interval'],self.c['universal_revenue_ceiling'],self.c['max_attempts'])
        if bracket['next'] is not None:bracket['status']='time_cap_unresolved'
        selected=select(strong_rows)
        if selected is None:
            result={'state':'no_candidate','final':None,'bracket':bracket,'attempts':len(history)}
        else:
            frozen=self.stage('freeze',selected,lambda:self.backend.freeze(selected))
            final=self.stage('final',frozen,lambda:self.backend.final(frozen))
            passed=eligible(final,self.c['final_ic_threshold'])
            result={'state':'passed' if passed else 'failed_final','final':final,'bracket':bracket,'attempts':len(history),
                    'selected_candidate_index':selected['candidate_index'],
                    'final_confirms_lower_target':bool(passed and bracket['lower'] is not None and lower_bound(final)>=bracket['lower'])}
        result.update(case_id=self.c['case_id'],phase='adaptive_warm_refinement',independent_final_tests=0 if selected is None else 1,
                      claim='Finite budget empirical search;not certified IC,optimality or infeasibility',history=history)
        atomic_json(self.root/'result.json',result)
        pins={p.relative_to(self.root).as_posix():sha(p) for p in sorted(self.root.rglob('*')) if p.is_file() and p.name!='completion.json'}
        atomic_json(self.root/'completion.json',{'identity':self.identity,'result_sha256':sha(self.root/'result.json'),'files_sha256':pins,'completed_at':now()})
        return result
