"""Prespecified upward search anchored at an eligible starting mechanism."""
from adaptive_rules import eligible,lower_bound,select
def decision(history, baseline, ceiling, maximum=6, increment=.4, tolerance=.1):
    anchor=lower_bound(baseline)
    lower=max([anchor]+[h['target'] for h in history if h['accepted']])
    witnesses=[anchor]+[h['incumbent_lower'] for h in history]
    rejected=[h['target'] for h in history if not h['accepted'] and h['target']>max(witnesses)]
    upper=min(rejected) if rejected else None
    if upper is not None and upper-lower<=tolerance+1e-9:
        return {'next':None,'lower':lower,'upper_not_found':upper,'status':'empirical_gap_le_0.1','anchor':anchor}
    if len(history)>=maximum:
        return {'next':None,'lower':lower,'upper_not_found':upper,'status':'attempt_cap_unresolved','anchor':anchor}
    if lower>=ceiling:
        return {'next':None,'lower':lower,'upper_not_found':None,'status':'ceiling_boundary','anchor':anchor}
    if upper is not None:target=(lower+upper)/2
    else:target=min(ceiling,anchor+increment if not history else anchor+2*max(increment,max(h['target']-anchor for h in history)))
    return {'next':round(target,10),'lower':lower,'upper_not_found':upper,'status':'searching','anchor':anchor}
