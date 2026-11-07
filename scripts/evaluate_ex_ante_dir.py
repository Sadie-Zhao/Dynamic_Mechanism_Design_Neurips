"""Forward-only ex-ante DIR evaluation of the five frozen bandit mechanisms."""
from pathlib import Path
import hashlib,json,pickle,sys,time,csv
import importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'bandit'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'audit'))
import jax
from bandit_auction.core import AuctionSimulator
from bandit_auction.mechanisms.network import Mechanism,build_mechanism_network,build_mechanism_network_attn
from bandit_auction.envs.discrete_single import create_discrete_single_auction
from bandit_auction.envs.discrete_multi import create_discrete_multi_auction
from bandit_auction.envs.cont_single import create_cont_single_auction
from bandit_auction.envs.cont_multi import create_cont_multi_auction
from cont_cfg import load_cont_env
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True,help='New output directory; existing directories are rejected')
OUT=parser.parse_args().output
OUT.mkdir(parents=True,exist_ok=False)
SEED=11062026; BATCHES=64; SAMPLES=1024
specs=[('discrete_single','discrete/dS_Ap_m15/goal_7.617',create_discrete_single_auction),('discrete_multi_M1','discrete/dM_Ap_m15/goal_15.250',create_discrete_multi_auction),('discrete_multi_M2','discrete/dM_B_m2/goal_15.250',create_discrete_multi_auction),('continuous_single','continuous/cont_single/ckpt_best',create_cont_single_auction),('continuous_multi','continuous/cont_multi/ckpt_best',create_cont_multi_auction)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT.parent/'SOURCE_MANIFEST.json').read_text())['files']
results=[]
for index,(name,relative,factory) in enumerate(specs):
 folder=ROOT/'mechanisms'/relative; checkpoint=folder/'mechanism_params.pkl'
 before=sha(checkpoint)
 assert before==manifest[checkpoint.relative_to(ROOT.parent).as_posix()]['sha256']
 if name.startswith('continuous'):
  kwargs,attention,config,missing=load_cont_env(str(folder),42)
  assert not missing,missing
  config_path=folder/'config.yaml'
 else:
  config_path=folder/'summary.json'; config=json.loads(config_path.read_text());attention=False
  kwargs=dict(seed=config['seed'],n_buyers=2,n_goods=1 if name=='discrete_single' else 2,horizon=5,discount=.9,allocation_cost=.1)
 auction=factory(**kwargs)
 network=build_mechanism_network_attn() if attention else build_mechanism_network()
 with checkpoint.open('rb') as f: params=pickle.load(f)
 mechanism=Mechanism(network,params)
 keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED),index),BATCHES)
 start=time.perf_counter(); means=[]
 print('Evaluating',name,flush=True)
 for key in keys:
  means.append(np.asarray(AuctionSimulator.estimate_cumulative_rewards(auction,mechanism,num_samples=SAMPLES,key=key),dtype=np.float64))
 means=np.stack(means); mu=means.mean(0); se=means.std(0,ddof=1)/np.sqrt(BATCHES)
 assert np.isfinite(means).all()
 assert sha(checkpoint)==before
 row=dict(mechanism=name,checkpoint=str(checkpoint),checkpoint_sha256=before,checkpoint_unchanged=True,config_sha256=sha(config_path),environment=kwargs,attention=attention,seed=SEED,seed_fold_in=index,batches=BATCHES,samples_per_batch=SAMPLES,total_trajectories=BATCHES*SAMPLES,truthful_utility_batch_means=means.tolist(),truthful_utility_mean=mu.tolist(),truthful_utility_se=se.tolist(),truthful_utility_normal_ci95=np.stack([mu-1.96*se,mu+1.96*se],axis=1).tolist(),estimated_ex_ante_dir_per_buyer=np.maximum(0,-mu).tolist(),estimated_ex_ante_dir_sum=float(np.maximum(0,-mu).sum()),normal_approx_simultaneous_one_sided_95_lower_utility=(mu-1.96*se).tolist(),normal_approx_95_upper_dir_sum=float(np.maximum(0,-(mu-1.96*se)).sum()),elapsed_seconds=time.perf_counter()-start)
 if name=='discrete_single':
  row['saved_exact_truthful_utility_per_buyer']=[.8790415525436401]*2
  row['mc_minus_exact_in_standard_errors']=((mu-.8790415525436401)/se).tolist()
 (OUT/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 results.append(row)
 print(name,'utility',mu,'SE',se,'DIR',row['estimated_ex_ante_dir_sum'],flush=True)
report=dict(role='new_forward_only_evaluation_of_frozen_mechanisms_no_training_no_deviation_search',metric='sum_i max(0, -mean_truthful_discounted_utility_i), hinge after averaging all trajectories',uncertainty='Normal approximation from 64 independent batch means; 1.96 also gives Bonferroni simultaneous one-sided 95% lower bounds for two buyers. These are sampling statements, not exact integration.',packages={p:importlib.metadata.version(p) for p in ['jax','jaxlib','dm-haiku','flax','optax','numpy']},script_sha256=sha(Path(__file__)),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'bandit_auction').rglob('*.py')},results=results)
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['mechanism','buyer_1_mean','buyer_2_mean','buyer_1_se','buyer_2_se','estimated_ex_ante_DIR','approx_95_upper_DIR'])
 for r in results:w.writerow([r['mechanism'],*r['truthful_utility_mean'],*r['truthful_utility_se'],r['estimated_ex_ante_dir_sum'],r['normal_approx_95_upper_dir_sum']])
print('Saved',OUT,flush=True)
