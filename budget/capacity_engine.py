"""Five-checkpoint selection; unchanged training and numerical audit kernels."""
from pathlib import Path
import json,shutil
from capacity_engine_original import Engine as Original
from adaptive_rules import eligible,accepted
from budget_auction.train.pilot import atomic_json,atomic_pickle
from budget_auction.train.revenue_trial import read_pickle
from budget_auction.train.revenue_flow import sha
from adaptive_engine import now
EPOCH_INDEX={500:2,750:0,1000:3,1250:4,1500:1}

def ranked(rows):
    return sorted((x for x in rows if eligible(x,.04)),key=lambda x:(-x['revenue']['mean'],x['epoch']))

class Engine(Original):
    def train(self,index,target):
        result=super().train(index,target)
        for c in result['data']:c['candidate_index']=EPOCH_INDEX[c['epoch']]
        return result

    def import_origin(self):
        manifest=json.loads((self.reg/'origins'/f'{self.c["case_id"]}.json').read_text())
        src=Path(manifest['output_root']);cr=manifest['review']
        assert manifest['review_sha256']==sha(Path(manifest['review_path']))
        for n,h in cr['files_sha256'].items():assert sha(src/n)==h
        old=json.loads((src/'result.json').read_text())
        assert old['final'] is None and old['independent_final_tests']==0
        logs=read_pickle(src/'target_00/trained.pkl')
        assert len(logs['loss'])==1500
        snapshots={x['epoch']:x for x in logs['checkpoint_mech']}
        folder=self.root/'target_00';folder.mkdir(exist_ok=True)
        candidates=[];paths=[]
        for epoch,idx in EPOCH_INDEX.items():
            dest=folder/f'candidate_epoch{epoch:04d}.pkl'
            if epoch in [750,1500]:shutil.copyfile(src/f'target_00/candidate_epoch{epoch:04d}.pkl',dest)
            else:
                point=snapshots[epoch];assert point['measurement_role']=='training_only_pre_update'
                # epoch e stores parameters before update e+1: exactly e updates completed.
                atomic_pickle(dest,{'mechanism_params':point['mechanism_params']})
            paths.append(dest);candidates.append({'candidate_index':idx,'trial_index':0,'epoch':epoch,'target':self.c['target'],'mechanism_artifact':self.artifact(dest)})
        # Copy committed audit subtrees, preserving their original receipts and seeds.
        for row in old['regular']:
            idx=row['candidate_index'];assert row['epoch'] in [750,1500] and idx==EPOCH_INDEX[row['epoch']]
            part=Path(f'candidate_{idx:02d}/regular');shutil.copytree(src/part,self.root/part)
            paths.extend(p for p in (self.root/part).rglob('*') if p.is_file())
        oldstrong=old.get('strong')
        if oldstrong is not None:
            part=Path(f'candidate_{oldstrong["candidate_index"]:02d}/stress');shutil.copytree(src/part,self.root/part)
            paths.extend(p for p in (self.root/part).rglob('*') if p.is_file())
        return {'data':{'candidates':candidates,'regular':old['regular'],'strong':oldstrong,'origin_review_sha256':manifest['review_sha256']},'artifacts':paths}

    def run(self):
        c=self.c;assert c['unit_kind']=='target','Finals require a separate reviewed shared-schedule decision'
        imported=None
        if c['selection_mode']=='reselect':
            imported=self.stage('import_origin',c['original_unit'],self.import_origin)
            candidates=imported['candidates']
        else:candidates=self.stage('train_00',{'target':c['target'],'cold':True},lambda:self.train(0,c['target']))
        oldrows={} if imported is None else {x['candidate_index']:x for x in imported['regular']}
        regular=[]
        for candidate in candidates:
            idx=candidate['candidate_index']
            def audit(candidate=candidate,idx=idx):
                if idx in oldrows:
                    row=dict(oldrows[idx]);assert row['epoch']==candidate['epoch']
                    return {'data':row,'artifacts':[]}
                return self.backend.audit(candidate)
            regular.append(self.stage(f'regular_{idx:02d}',candidate,audit))
        order=ranked(regular);attempts=[];selected=None
        for rank,row in enumerate(order[:2]):
            candidate=next(x for x in candidates if x['candidate_index']==row['candidate_index'])
            def stress(candidate=candidate,row=row):
                old=None if imported is None else imported['strong']
                if old is not None and old['candidate_index']==candidate['candidate_index']:
                    return {'data':old,'artifacts':[]}
                return self.backend.stress(candidate,row)
            strong=self.stage(f'strong_{rank:02d}',[candidate,row],stress);attempts.append(strong)
            if eligible(strong,.04):selected=strong;break
        result={'state':'target_complete','case_id':c['case_id'],'logical_case':c['logical_case'],'original_unit':c['original_unit'],'unit_kind':'target','target':c['target'],'strong':selected,'attained':accepted(selected,c['target']) if selected else False,'regular':regular,'strong_attempts':attempts,'screen_ranking':[x['candidate_index'] for x in order],'final':None,'independent_final_tests':0,'cold_start':True,'epochs':1500,'selection_mode':c['selection_mode'],'checkpoint_epochs':list(EPOCH_INDEX),'phase':'capacity_matched_five_checkpoint_selection','claim':'Finite adaptive validation selection; not certified IC, optimality or infeasibility'}
        atomic_json(self.root/'result.json',result)
        pins={p.relative_to(self.root).as_posix():sha(p) for p in sorted(self.root.rglob('*')) if p.is_file() and p.name!='completion.json'}
        atomic_json(self.root/'completion.json',{'identity':self.identity,'result_sha256':sha(self.root/'result.json'),'files_sha256':pins,'completed_at':now()})
        return result
