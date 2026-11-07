"""Pure target decisions. Failed optimization is never an infeasibility proof."""
import math
def eligible(row, threshold=.04):
    return bool(row and row['observed_feasible'] and math.isfinite(row['ic_upper'])
                and row['ic_upper']<=threshold and math.isfinite(row['revenue']['mean']))
def lower_bound(row):
    return float(row['revenue']['ci95'][0])
def accepted(row,target):
    return eligible(row) and lower_bound(row)>=target
def decision(history,interval,ceiling,maximum=6):
    lo0,hi0=interval; width=hi0-lo0
    positive=[h['target'] for h in history if h['accepted']]
    # Any later strongly eligible candidate can invalidate an earlier rejection.
    measured=[lower_bound(h['strong']) for h in history if eligible(h.get('strong'))]
    established=max(positive+measured) if positive or measured else None
    rejected=[h['target'] for h in history if not h['accepted'] and (established is None or h['target']>established)]
    upper=min(rejected) if rejected else None
    if established is not None and upper is not None and upper-established<=.1+1e-10:
        return {'next':None,'lower':established,'upper_not_found':upper,'status':'empirical_gap_le_0.1'}
    if len(history)>=maximum:
        return {'next':None,'lower':established,'upper_not_found':upper,'status':'attempt_cap_unresolved'}
    if established is None:
        target=lo0 if not history else history[-1]['target']-.2
    elif upper is not None:
        target=(established+upper)/2
    else:
        target=hi0 if established<hi0 and not any(h['target']==hi0 for h in history) else min(ceiling,max(established,max([h['target'] for h in history],default=lo0))+width)
        if established>=ceiling or any(h['target']==ceiling and h['accepted'] for h in history):
            return {'next':None,'lower':established,'upper_not_found':None,'status':'ceiling_boundary_requires_review'}
    return {'next':round(target,10),'lower':established,'upper_not_found':upper,'status':'searching'}
def select(rows):
    valid=[r for r in rows if eligible(r)]
    return min(valid,key=lambda r:(-r['revenue']['mean'],r['candidate_index'])) if valid else None
