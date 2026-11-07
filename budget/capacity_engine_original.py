"""Cold target units and a separate single frozen final; no cross-target warm start."""
from pathlib import Path
import json,time,math,shutil
import jax
from adaptive_engine import Engine as BaseEngine,now,finite,SCALARS
from adaptive_rules import select,eligible,accepted
from focused_engine import release_compilation_cache
from budget_auction.core import build_bounded_policy_net
from budget_auction.train.run_acceptance import nested_GD_budget
from budget_auction.train.revenue_flow import digest,sha
from budget_auction.train.pilot import atomic_json,atomic_pickle
from capacity_rules import validate_decision,selected_row
class Engine(BaseEngine):
    def stage(self,name,inputs,operation):
        result=super().stage(name,inputs,operation)
        release_compilation_cache()
        return result
    def train(self,index,target):
        folder=self.root/f'target_{index:02d}';folder.mkdir(exist_ok=True)
        checkpoint=folder/'resume.pkl';resume=self.resume(checkpoint)
        inv,record,started=self.phase_invocation(folder,resume)
        identity=digest([self.identity,index,target,'fresh_initialization'])
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
              init_mechanism_params=None,checkpoint_callback=save)
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
        c=self.c
        if c['unit_kind']=='target':
            target=c['target']
            candidates=self.stage('train_00',{'target':target,'cold':True},lambda:self.train(0,target))
            regular=[self.stage(f'regular_{x["candidate_index"]:02d}',x,lambda x=x:self.backend.audit(x)) for x in candidates]
            chosen=select(regular);strong=None
            if chosen is not None:
                candidate=next(x for x in candidates if x['candidate_index']==chosen['candidate_index'])
                strong=self.stage('strong_00',[candidate,chosen],lambda:self.backend.stress(candidate,chosen))
            result={'state':'target_complete','target':target,'strong':strong,'attained':accepted(strong,target) if strong else False,
                    'regular':regular,'final':None,'independent_final_tests':0,'cold_start':True,'epochs':c['epochs']}
        else:
            decision=validate_decision(self.reg,c['decision_path'])
            selected=selected_row(decision,c['logical_case'])
            atomic_json(self.root/'selection_decision.json',decision)
            if selected is None:
                result={'state':'no_candidate','final':None,'independent_final_tests':0}
            else:
                origin=self.reg/'outputs'/selected['unit']/selected['row']['stress_artifact']
                assert sha(origin)==selected['artifact_sha256']
                def imported():
                    dest=self.root/'selected_input.pkl';shutil.copyfile(origin,dest)
                    row=dict(selected['row'],stress_artifact='selected_input.pkl')
                    return {'data':row,'artifacts':[dest]}
                row=self.stage('import_selection',selected,imported)
                frozen=self.stage('freeze',row,lambda:self.backend.freeze(row))
                final=self.stage('final',frozen,lambda:self.backend.final(frozen))
                result={'state':'passed' if eligible(final,c['final_ic_threshold']) else 'failed_final','final':final,
                    'independent_final_tests':1,'selected_candidate_index':row['candidate_index'],
                    'selected_unit':selected['unit'],'selected_validation_mean':row['revenue']['mean']}
        result.update(case_id=c['case_id'],logical_case=c['logical_case'],unit_kind=c['unit_kind'],phase='capacity_matched_cold_main',
                      claim='Finite development-informed target search; not certified IC, optimality or infeasibility')
        atomic_json(self.root/'result.json',result)
        pins={p.relative_to(self.root).as_posix():sha(p) for p in sorted(self.root.rglob('*')) if p.is_file() and p.name!='completion.json'}
        atomic_json(self.root/'completion.json',{'identity':self.identity,'result_sha256':sha(self.root/'result.json'),'files_sha256':pins,'completed_at':now()})
        return result
