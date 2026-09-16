"""Paired, explicitly bounded stress tests; never silently discard failures."""
import argparse,json
from pathlib import Path
import torch
from rollout import load_policy,rollout

p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--start',type=int,default=700);p.add_argument('--count',type=int,default=10);p.add_argument('--label',default='robustness');a=p.parse_args()
torch.set_num_threads(4)
policy,digest=load_policy(a.checkpoint)
out=Path(__file__).resolve().parent/'evaluations'/a.label;out.mkdir(parents=True,exist_ok=True)
results=[]
for profile in ['standard','lighting','physics','size','combined']:
    for seed in range(a.start,a.start+a.count):
        result=rollout(policy,seed,profile=profile,out=out/profile/str(seed))
        results.append(result)
        print(json.dumps({'profile':profile,'seed':seed,'success':result['success']}),flush=True)
        (out/'summary.json').write_text(json.dumps({'checkpoint_sha256':digest,'results':results,'scores':{x:sum(r['success'] for r in results if r['profile']==x) for x in set(r['profile'] for r in results)}},indent=2))
