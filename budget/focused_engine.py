"""Four best-start follow-ups. Existing scientific numerical components unchanged."""
import json,shutil,gc,ctypes
import jax
from adaptive_engine import Engine as BaseEngine,now
from adaptive_rules import eligible,lower_bound,select
from focused_rules import decision
from budget_auction.train.revenue_trial import read_pickle
from budget_auction.train.revenue_flow import sha
from budget_auction.train.pilot import atomic_json

def release_compilation_cache():
    jax.clear_caches();gc.collect()
    try:ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError,AttributeError):pass

class Engine(BaseEngine):
    def stage(self,name,inputs,operation):
        result=super().stage(name,inputs,operation)
        # Compilation caches are execution resources, not optimizer/policy/RNG state.
        release_compilation_cache()
        return result
    def initial(self):
        src=self.reg/self.c['input_path'];assert sha(src)==self.c['input_sha256']
        def copy():
            p=self.root/'initial.pkl';shutil.copyfile(src,p)
            return {'data':{'artifact':'initial.pkl'},'artifacts':[p]}
        return self.stage('initial',self.c['input_sha256'],copy)['artifact']
    def baseline(self,initial):
        def evaluate():
            p=self.root/initial;saved=read_pickle(p);saved['frozen_artifact_sha256']=sha(p)
            folder=self.root/'baseline_validation'
            row=self.backend.evaluate_fixed(folder,saved,self.backend.seed('baseline_ic'),self.backend.seed('baseline_revenue'),self.c['selection_ic_threshold'])
            row.update(candidate_index=12,stress_artifact=initial,role='retained_incumbent_fresh_fixed_policy_validation_after_prior_strong_search')
            return {'data':row,'artifacts':self.backend.files(folder)+[p]}
        return self.stage('baseline',self.c['input_sha256'],evaluate)
    def decide(self,history,baseline):
        return decision(history,baseline,self.c['universal_revenue_ceiling'],self.c['max_attempts'],self.c['initial_increment'],self.c['tolerance'])
    def run(self):
        history_path=self.root/'target_history.json'
        already_completed=len(json.loads(history_path.read_text())) if history_path.exists() else 0
        initial=self.initial();baseline=self.baseline(initial);history=[];rows=[baseline] if eligible(baseline) else []
        if not rows:
            result={'state':'baseline_failed_validation','final':None,'attempts':0,'independent_final_tests':0,'baseline':baseline,'history':[]}
        else:
            incumbent=baseline
            for index in range(self.c['max_attempts']):
                d=self.decide(history,baseline)
                if d['next'] is None:break
                if not (self.root/f'target_{index:02d}').exists() and (self.spent()>=self.c['stop_training_seconds'] or now()>=self.c['train_cutoff_utc']):break
                target=d['next'];warm=incumbent['stress_artifact']
                candidates=self.stage(f'train_{index:02d}',[index,target,warm,sha(self.root/warm)],lambda:self.train(index,target,warm))
                regular=[self.stage(f'regular_{c["candidate_index"]:02d}',c,lambda c=c:self.backend.audit(c)) for c in candidates]
                chosen=select(regular);strong=None
                if chosen is not None:
                    candidate=next(c for c in candidates if c['candidate_index']==chosen['candidate_index'])
                    strong=self.stage(f'strong_{index:02d}',[candidate,chosen],lambda:self.backend.stress(candidate,chosen))
                    rows.append(strong)
                incumbent=select(rows)
                witness=max(lower_bound(r) for r in rows if eligible(r))
                history.append({'attempt':index,'target':target,'accepted':witness>=target,'strong':strong,
                    'incumbent_candidate_index':incumbent['candidate_index'],'incumbent_lower':witness})
                atomic_json(self.root/'target_history.json',history)
                bracket=self.decide(history,baseline);atomic_json(self.root/'bracket.json',bracket)
                if index>=already_completed and bracket['next'] is not None and index<self.c['max_attempts']-1:
                    self.hook('planned_target_boundary',self.root/'target_history.json')
            bracket=self.decide(history,baseline)
            if bracket['next'] is not None:bracket['status']='time_cap_unresolved'
            selected=select(rows)
            frozen=self.stage('freeze',selected,lambda:self.backend.freeze(selected))
            final=self.stage('final',frozen,lambda:self.backend.final(frozen))
            passed=eligible(final,self.c['final_ic_threshold'])
            result={'state':'passed' if passed else 'failed_final','baseline':baseline,'final':final,'bracket':bracket,
                'attempts':len(history),'history':history,'selected_candidate_index':selected['candidate_index'],
                'independent_final_tests':1,'final_confirms_lower_target':passed and lower_bound(final)>=bracket['lower'],
                'selected_validation_mean':selected['revenue']['mean'],'baseline_validation_mean':baseline['revenue']['mean']}
            assert selected['revenue']['mean']>=baseline['revenue']['mean']
        result.update(case_id=self.c['case_id'],phase='focused_best_start_refinement',
          claim='Best-start extra-compute case study; empirical target interval, not global optimality or infeasibility proof',
          historical_exact_cost_unknown=self.c['prior_measured_seconds'] is None,operational_prior_seconds=self.c['operational_prior_seconds'])
        atomic_json(self.root/'result.json',result)
        pins={p.relative_to(self.root).as_posix():sha(p) for p in sorted(self.root.rglob('*')) if p.is_file() and p.name!='completion.json'}
        atomic_json(self.root/'completion.json',{'identity':self.identity,'result_sha256':sha(self.root/'result.json'),'files_sha256':pins,'completed_at':now()})
        return result
