"""Learned-policy rollout. No expert controller, IK, segmentation or scripted target lookup."""
from pathlib import Path
import sys,argparse,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
import torch
from PIL import Image
from scene import build,initial_data,ROOT
from policy import Policy,tokenize,INSTRUCTIONS

def publish_frame(frame,out):
    """A Windows HTTP reader can briefly lock the destination JPEG."""
    out.mkdir(parents=True,exist_ok=True)
    frame.save(out/'live.tmp',format='JPEG')
    for attempt in range(10):
        try:
            (out/'live.tmp').replace(out/'live.jpg')
            return
        except PermissionError:
            time.sleep(.01)
    # A dropped preview frame must not interrupt physical control.

def load_policy(path=None):
    path=Path(path) if path else ROOT/'learning/checkpoints/policy.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=Policy(checkpoint['means'],checkpoint['scales'],checkpoint.get('version',1))
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,hashlib.sha256(path.read_bytes()).hexdigest()

def rollout(policy,seed=300,task=0,jitter=.008,ablation='none',video=False,out=None,on_event=None,instruction=None,profile='standard',video_size=(640,480)):
    m=build(seed,jitter,profile);d=initial_data(m)
    # Ground-truth poses are accessed only by this evaluator, not by policy inputs.
    plate_start=d.body('plate').xpos.copy()
    instruction=instruction or INSTRUCTIONS[task]
    tokens=torch.from_numpy(tokenize(instruction)).unsqueeze(0)
    report={'seed':seed,'instruction':instruction,'ablation':ablation,'jitter_m':jitter,'profile':profile,'actions':[],'inference_ms':[]}
    frames=[];first_arm=None;started=time.perf_counter()
    with mujoco.Renderer(m,96,128) as camera:
        movie=mujoco.Renderer(m,video_size[1],video_size[0]) if video else None
        try:
            for step in range(18):
                if on_event:on_event({'object':'Learned policy','stage':f'action {step+1}/18'})
                camera.update_scene(d,camera='overhead');rgb=camera.render().copy()
                if ablation=='blank_image':rgb[:]=0
                image=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255
                state=torch.from_numpy(d.qpos[:12].copy()).unsqueeze(0).float()
                t=time.perf_counter()
                with torch.no_grad():action,logits=policy(image,state,tokens,torch.tensor([step]))
                report['inference_ms'].append((time.perf_counter()-t)*1000)
                action=action[0].numpy()
                target=np.clip(action[:12],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
                duration=float(np.clip(action[12],.5,2.5))
                start=d.ctrl.copy()
                changes=[np.linalg.norm(target[:6]-start[:6]),np.linalg.norm(target[6:]-start[6:])]
                if first_arm is None and max(changes)>.05:first_arm=['left','right'][int(np.argmax(changes))]
                report['actions'].append({'step':step,'joint_targets':target.tolist(),'seconds':duration,'language_probabilities':torch.softmax(logits,dim=-1)[0].tolist()})
                for frame in range(round(duration/m.opt.timestep)):
                    t=(frame+1)/round(duration/m.opt.timestep)
                    d.ctrl[:]=start+(target-start)*(t*t*(3-2*t))
                    mujoco.mj_step(m,d)
                    for contact in d.contact:
                        names=[m.body(m.geom_bodyid[int(g)]).name for g in contact.geom]
                        if any(n.startswith('left_') for n in names) and any(n.startswith('right_') for n in names):raise RuntimeError('Arm-to-arm contact; stopped')
                    if movie and frame%100==0:
                        movie.update_scene(d,camera='overview');frames.append(Image.fromarray(movie.render()))
                        if out and len(frames)%5==0:
                            out.mkdir(parents=True,exist_ok=True)
                            publish_frame(frames[-1],out)
            report['objects']={}
            for name,delta in [('cup',.14),('utensil',-.14)]:
                target=plate_start[:2]+np.array([delta,0]);pos=d.body(name).xpos.copy()
                upright=float(d.body(name).xmat.reshape(3,3)[2,2])
                error=float(np.linalg.norm(pos[:2]-target))
                report['objects'][name]={'position':pos.tolist(),'error_m':error,'upright_cosine':upright,'passed':bool(error<.03 and pos[2]<.05 and (name!='cup' or upright>.9))}
            report['plate_undisturbed']=bool(np.linalg.norm(d.body('plate').xpos[:2]-plate_start[:2])<.015)
            report['first_arm']=first_arm
            report['instruction_order_correct']=first_arm==('right' if task==0 else 'left')
            report['success']=all(o['passed'] for o in report['objects'].values()) and report['plate_undisturbed'] and report['instruction_order_correct']
            camera.update_scene(d,camera='overhead');final=Image.fromarray(camera.render())
            if movie and out:
                movie.update_scene(d,camera='overview')
                publish_frame(Image.fromarray(movie.render()),out)
        except Exception as exc:report.update(success=False,error=str(exc));final=None
        finally:
            if movie:movie.close()
    report['wall_seconds']=time.perf_counter()-started
    if out:
        out.mkdir(parents=True,exist_ok=True)
        (out/'rollout.json').write_text(json.dumps(report,indent=2))
        if final:final.resize((640,480)).save(out/'final.png')
        if frames:frames[0].save(out/'rollout.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=300);p.add_argument('--count',type=int,default=10);p.add_argument('--task',type=int,default=0);p.add_argument('--ablation',choices=['none','blank_image'],default='none');p.add_argument('--video',action='store_true');p.add_argument('--label',default='test');p.add_argument('--checkpoint');p.add_argument('--runtime',choices=['torch','openvino'],default='torch');p.add_argument('--precision',default='fp32');args=p.parse_args()
    torch.set_num_threads(4)
    if args.runtime=='openvino':
        from openvino_policy import OpenVINOPolicy
        policy=OpenVINOPolicy(precision=args.precision);digest=policy.digest
    else:policy,digest=load_policy(args.checkpoint)
    results=[]
    out=ROOT/'learning/evaluations'/args.label
    for seed in range(args.start,args.start+args.count):
        result=rollout(policy,seed,args.task,ablation=args.ablation,video=args.video,out=out/str(seed))
        results.append(result)
        print(json.dumps({'seed':seed,'success':result['success'],'objects':result.get('objects'),'first_arm':result.get('first_arm')}),flush=True)
        (out/'summary.json').write_text(json.dumps({'checkpoint_sha256':digest,'runtime':args.runtime,'precision':args.precision if args.runtime=='openvino' else 'fp32','successes':sum(r['success'] for r in results),'episodes':len(results),'results':results},indent=2))
