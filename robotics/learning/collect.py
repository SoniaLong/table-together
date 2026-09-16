"""Collect genuine rendered observations and expert joint commands; keep split provenance."""
from pathlib import Path
import sys,json,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from episode import run
from scene import ROOT

INSTRUCTIONS=['Set the table. Start with the cup.','Set the table. Start with the utensil.']
def collect(split,start,count,jitter):
    target=ROOT/'learning/data'/split;target.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(f.read_text()) for f in target.glob('*.json') if f.name!='manifest.json' and not start<=json.loads(f.read_text())['seed']<start+count]
    for seed in range(start,start+count):
        for task,order in enumerate(['cup_first','utensil_first']):
            name=f'{seed}-{order}';path=target/(name+'.npz')
            if path.exists():
                manifest.append(json.loads((target/(name+'.json')).read_text()));continue
            samples=[];renderer=None
            def capture(model,data,command,seconds):
                nonlocal renderer
                if renderer is None:renderer=mujoco.Renderer(model,96,128)
                renderer.update_scene(data,camera='overhead')
                samples.append((renderer.render().copy(),data.qpos[:12].copy(),np.r_[command,seconds]))
            started=time.perf_counter()
            record={'seed':seed,'split':split,'task':task,'instruction':INSTRUCTIONS[task],'jitter_m':jitter}
            try:
                log=run(record_video=False,seed=seed,jitter=jitter,max_retries=0,order=order,demonstration=capture,out_dir=target/'logs'/name)
                record.update(success=bool(log[-1]['success']),steps=len(samples))
                if record['success'] and len(samples)==18:
                    np.savez_compressed(path,images=np.stack([s[0] for s in samples]),state=np.stack([s[1] for s in samples]).astype('float32'),actions=np.stack([s[2] for s in samples]).astype('float32'),task=np.int64(task),seed=np.int64(seed))
            except Exception as exc:record.update(success=False,error=str(exc))
            finally:
                if renderer is not None:renderer.close()
            record['seconds']=time.perf_counter()-started
            (target/(name+'.json')).write_text(json.dumps(record,indent=2))
            manifest.append(record)
            (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(record),flush=True)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'split':split,'successful_demos':sum(r.get('success',False) and r.get('steps')==18 for r in manifest),'attempts':len(manifest)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='train');p.add_argument('--start',type=int,default=100);p.add_argument('--count',type=int,default=24);p.add_argument('--jitter',type=float,default=.008);a=p.parse_args()
    collect(a.split,a.start,a.count,a.jitter)
