"""Metadata-only shared schedule and validation-only selection."""
from pathlib import Path
import json,hashlib,math
METHODS=('A','A_wide','B','C')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eligible(r):
    return bool(r and r['observed_feasible'] and math.isfinite(r['ic_upper']) and r['ic_upper']<=.04 and math.isfinite(r['revenue']['mean']))
def logical(setting,method,block):return f'{setting}_{method}_s{block:02d}'
def unit(setting,method,block,index):return logical(setting,method,block)+f'_t{index:02d}'
def schedule(setting,block,results):
    """Return next shared target count; never consult independent finals."""
    count=3
    while True:
        needed=[unit(setting,m,block,i) for i in range(count) for m in METHODS]
        missing=[u for u in needed if u not in results]
        if missing:return {'ready':False,'target_count':count,'missing':missing,'stop_reason':None}
        assert all(results[u]['state']=='target_complete' and results[u]['final'] is None for u in needed)
        highest=round((2.1 if setting=='e1' else 3.2)+.1*(count-1),1)
        attained=any(eligible(results[u]['strong']) and results[u]['strong']['revenue']['ci95'][0]>=highest for u in needed)
        if not attained or count==5:
            return {'ready':True,'target_count':count,'missing':[],
                    'stop_reason':'extension_cap_unresolved' if attained else 'highest_target_not_attained'}
        count+=1
def selected_row(decision,case):
    rows=[x for x in decision['targets'] if x['logical_case']==case and eligible(x['row'])]
    return min(rows,key=lambda x:(-x['row']['revenue']['mean'],x['unit'],x['row']['candidate_index'])) if rows else None
def make_decision(reg,setting,block):
    results={};entries=[]
    for i in range(5):
      for m in METHODS:
        u=unit(setting,m,block,i);rp=reg/'outputs'/u/'result.json';review=reg/'dispatch'/u/'completion_review.json'
        if not review.exists():continue
        rr=read(review);assert rr['state']=='COMPLETE_REVIEW_PASS' and sha(rp)==rr['files_sha256']['result.json']
        r=read(rp);results[u]=r;row=r['strong']
        ap=reg/'outputs'/u/row['stress_artifact'] if row else None
        entries.append({'unit':u,'logical_case':logical(setting,m,block),'row':row,'result_sha256':sha(rp),
                        'review_sha256':sha(review),'artifact_sha256':sha(ap) if ap else None})
    s=schedule(setting,block,results);assert s['ready'],'Shared schedule must finish before any final'
    assert len(entries)==4*s['target_count'],'Unexpected or out-of-schedule trial'
    return {'setting':setting,'seed_block':block,'schedule':s,'targets':entries,
            'selection':'highest eligible strong-validation mean, ties by unit then candidate; no final data',
            'registration_sha256':sha(reg/'registration.json')}
def validate_decision(reg,path):
    reg=Path(reg);d=read(reg/path)
    assert d==make_decision(reg,d['setting'],d['seed_block']),'Selection inputs/decision changed'
    return d
