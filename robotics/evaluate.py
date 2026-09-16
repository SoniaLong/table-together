"""Small, reproducible local robustness check, not a general success-rate claim."""
import argparse,json,time
from scene import ROOT
from episode import run
parser=argparse.ArgumentParser()
parser.add_argument('--seeds',type=int,default=5)
parser.add_argument('--jitter',type=float,default=.005)
parser.add_argument('--label',default='recovery')
parser.add_argument('--start-seed',type=int,default=0)
args=parser.parse_args()
out=ROOT/'artifacts'/f'evaluation-{args.jitter:g}-{args.label}'
out.mkdir(parents=True,exist_ok=True)
results=[]
for seed in range(args.start_seed,args.start_seed+args.seeds):
    start=time.perf_counter()
    try:
        log=run(record_video=False,seed=seed,jitter=args.jitter,out_dir=out/str(seed))
        result={'seed':seed,**log[-1],'objects':[x for x in log if 'evaluation' in x]}
    except Exception as exc:result={'seed':seed,'success':False,'error':str(exc)}
    result['wall_seconds']=time.perf_counter()-start
    results.append(result)
    report={'protocol':'Plate, cup and utensil independently shifted uniformly in x/y; same dimensions, colors, camera and lighting. No language-model variation in this controller evaluation.','jitter_m':args.jitter,'completed':len(results),'successes':sum(r['success'] for r in results),'results':results}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
