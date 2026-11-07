"""Verify saved artifacts and export paper tables. Standard library only; no experiments."""
import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'derived_tables')
    parser.add_argument('--paper',type=Path,help='Optional manuscript folder to check the main 128-cell table')
    args=parser.parse_args()
    failures=[]
    manifest=read(ROOT/'SOURCE_MANIFEST.json')['files']
    for name,item in manifest.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            failures.append('Artifact hash mismatch: '+name)
    parsed=0
    for path in ROOT.rglob('*.py'):
        try: ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));parsed+=1
        except SyntaxError as e: failures.append(str(e))
    budget=read(ROOT/'budget/results/results.json')
    selected=read(ROOT/'budget/results/selected_checkpoints.json')
    assert len(selected)==24
    for row in budget['main_table']:
        values=[r['strong_validation_mean'] for r in selected if (r['setting'],r['variant'])==(row['setting'],row['variant'])]
        assert values==row['seed_revenues']
        assert abs(statistics.mean(values)-row['mean'])<1e-12
        assert abs(statistics.stdev(values)-row['sample_sd'])<1e-12
    assert all(r['ic_upper']<=0.04 for r in selected)
    cells=read(ROOT/'bandit/results/evaluation_grid.json')['cells']
    assert len(cells)==128
    keys=[(r['mechanism'],r['paper_information_label'],r['algorithm'],r['architecture']) for r in cells]
    assert len(set(keys))==128
    for row in cells:
        raw=read(ROOT/row['release_artifact'])
        candidates=[raw.get(k) for k in ['best_regret_per_buyer','best_heldout_regret_per_buyer']]
        assert row['raw_value'] in candidates,(row['release_artifact'],row['raw_value'])
        assert row['mechanism'] in raw['run_dir'],row['release_artifact']
    if args.paper:
        text=(args.paper/'Neurips_sections/new_exp.tex').read_text(encoding='utf-8')
        table=text[text.index('& & \\multicolumn{4}{c}{Discrete, two goods}'):text.index('\\label{tab:bandit_results}')]
        actual=[]
        for match in re.finditer(r'M[12] & (?:PW|ES|NPG\+?)\s*(.*?)\\\\',table,re.S):
            actual += [float(x) for x in re.findall(r'[-+]?\d+\.\d+',match.group(1))]
        expected=[]
        groups=[('dM_Ap_m15',['base','wide','attn','gru']),('dM_B_m2',['base','wide','attn','gru']),('cont_single',['mlp','wide','attn','gru']),('cont_multi',['mlp','wide','attn','gru'])]
        bykey=dict(zip(keys,cells))
        for info in ['M1','M2']:
            for alg in ['pathwise','ES','NPG','NPG-ceiling']:
                for mech,arches in groups:
                    for arch in arches: expected.append(float(f"{bykey[mech,info,alg,arch]['raw_value']:.3f}"))
        assert len(actual)==128,len(actual)
        assert actual==expected,[(i,a,e) for i,(a,e) in enumerate(zip(actual,expected)) if a!=e]
    args.output.mkdir(parents=True,exist_ok=True)
    fields=['setting','variant','seed','selected_unit','selected_epoch','target','strong_validation_mean','ic_upper','checkpoint','checkpoint_sha256']
    with (args.output/'budget_selections.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(selected)
    fields=['mechanism','paper_information_label','algorithm','architecture','raw_value','metric','release_artifact','artifact_sha256']
    with (args.output/'bandit_evaluation_grid.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(cells)
    result=dict(status='passed' if not failures else 'failed',source_artifacts=len(manifest),python_files_parsed=parsed,budget_selections=len(selected),bandit_cells=len(cells),main_table_checked=bool(args.paper),failures=failures,scope='Read-only artifact checks and table export; no numerical experiments or checkpoint loading.')
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
