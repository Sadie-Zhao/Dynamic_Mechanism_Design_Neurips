"""Explicit completed500-to1000 extension; never changes the saved predecessor."""
import math
def extend_supervised_state(state, expected, history):
    assert state['schema']==1 and state['controls']==expected
    assert expected['num_epochs']==500 and expected['lr']==.001 and expected['batch_size']==256
    assert expected['alloc_weight']==expected['payment_weight']==1. and expected['eval_every']==10
    assert state['next_epoch']==500 and len(state['loss_history'])==500
    assert history['epoch']==500 and history['loss']==state['loss_history'] and history['validation']==state['validation_history']
    assert all(math.isfinite(v) for v in state['loss_history'])
    validation=state['validation_history']
    assert [r['epoch'] for r in validation]==list(range(0,501,10))
    assert all(math.isfinite(v) for r in validation for v in r.values())
    best=min(validation,key=lambda r:(r['selection_loss'],r['epoch']))
    assert state['best_epoch']==best['epoch'] and state['best_loss']==best['selection_loss']
    assert all(k in state for k in ['params','opt_state','rng_state','best_params'])
    result=dict(state);result['controls']=dict(expected,num_epochs=1000)
    assert all(result[k] is state[k] for k in state if k!='controls')
    return result
