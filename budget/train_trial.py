"""Explicit cold training of one archived budget configuration into a new directory.

This entry point does not change frozen paper selections or perform selection searches.
It is supplied for later reproduction and was not executed during release assembly.
"""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8'))
    if c.get('arm') or 'distill' in c or 'refine' in c.get('case_id',''):
        p.error('Use a main cold-training configuration, not a refinement/distillation configuration.')
    if args.output.exists():p.error('Output must be a new directory.')
    import haiku as hk
    from budget_auction.core import build_bounded_policy_net
    from budget_auction.mechanisms.input_adapters import VARIANT_NETWORK
    from budget_auction.train.pilot import FACTORIES,atomic_json,atomic_pickle
    from budget_auction.train.run_acceptance import nested_GD_budget
    auction=FACTORIES[c['setting']](seed=c['environment_seed'],**c['environment'])
    mechanism=hk.without_apply_rng(hk.transform(VARIANT_NETWORK[c['variant']]))
    policy=build_bounded_policy_net(auction.d_priv,auction.d_pub,auction.num_goods,hidden=c['policy_hidden'])
    args.output.mkdir(parents=True)
    atomic_json(args.output/'config.json',c)
    identity=hashlib.sha256(args.config.read_bytes()).hexdigest()
    logs=nested_GD_budget(auction,mechanism,policy,**c['training'],profit_goal=c['target'],alpha_weight=c['alpha_weight'],revenue_objective='acceptance',num_outer_epochs=c['epochs'],seed=c['streams']['train/0'],resume_identity=identity,checkpoint_callback=lambda state:atomic_pickle(args.output/'resume.pkl',state))
    atomic_pickle(args.output/'trained.pkl',logs)

if __name__=='__main__':main()
