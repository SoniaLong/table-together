"""Camera-driven manipulation baseline. Simulator state is used only for evaluation."""
import json
import numpy as np
import mujoco
from PIL import Image
from scene import build,initial_data,ROOT
from control import Controller
from perception import observe
POSITION_TOLERANCE=.03

def run(on_event=None,record_video=True,seed=None,jitter=.005,out_dir=None,max_retries=1,demonstration=None,order='cup_first'):
    out=ROOT/'artifacts' if out_dir is None else out_dir
    out.mkdir(parents=True,exist_ok=True)
    m=build(seed,jitter);d=initial_data(m);c=Controller(m,d)
    log=[];frames=[]
    if demonstration is not None:
        c.on_action=lambda command,seconds:demonstration(m,d,command,seconds)
    with mujoco.Renderer(m,480,640) as renderer:
        def record():
            if not record_video:return
            renderer.update_scene(d,camera='overview')
            frames.append(Image.fromarray(renderer.render()).resize((640,480)))
            if len(frames)%10==0:
                frames[-1].save(out/'live.tmp',format='JPEG')
                (out/'live.tmp').replace(out/'live.jpg')
        def camera():
            renderer.update_scene(d,camera='overhead')
            return observe(renderer.render())
        seen=camera();log.append({'observation':seen})
        plate=np.array(seen['plate']['xy'])
        objects=[('cup','right',.14,.32,.035),('utensil','left',-.14,-.04,.017)]
        if order=='utensil_first':objects.reverse()
        elif order!='cup_first':raise ValueError('Unknown demonstration order')
        for name,side,delta,grip,z in objects:
            c.jaw_axis=np.array([1,0,0])
            target=plate+np.array([delta,0])
            # Calibrated grasp-frame offsets for these two proxy geometries.
            offset=np.array([-.008,0]) if name=='cup' else np.array([-.010,0])
            place_offset=np.array([0,0]) if name=='cup' else np.array([.013,0])
            correction=np.zeros(2)
            for attempt in range(max_retries+1):
                seen=camera()
                if name not in seen:
                    log.append({'object':name,'error':'Object not visible; no blind retry.'});break
                xy=np.array(seen[name]['xy']);destination=target+place_offset+correction
                # Release the cup slightly above the table: pressing it into the
                # table while still gripped produces lateral skidding.
                release_z=.048 if name=='cup' else z
                tasks=[('approach',xy+offset,.07,.9),('lower',xy+offset,z,.9),('grasp',xy+offset,z,grip),('lift',xy+offset,.07,grip),('transfer',destination,.07,grip),('place',destination,release_z,grip),('release',destination,release_z,.9),('retreat',destination,.07,.9)]
                failed=False
                for stage,point,height,closing in tasks:
                    if on_event:on_event({'object':name,'stage':f'{stage} (attempt {attempt+1})'})
                    try:
                        result=c.move(side,[*point,height],closing,seconds=1.3,record=record)
                        log.append({'object':name,'stage':stage,'attempt':attempt+1,**result,'evaluation_object_position':d.body(name).xpos.tolist()})
                    except ValueError as exc:
                        log.append({'object':name,'stage':stage,'error':str(exc)});failed=True;break
                c.park(side,record=record)
                observed=camera()
                if name not in observed:break
                error=target-np.array(observed[name]['xy'])
                log.append({'object':name,'camera_error_m':float(np.linalg.norm(error)),'attempt':attempt+1})
                if failed or np.linalg.norm(error)<POSITION_TOLERANCE:break
                # Bounded visual correction. No simulator position is used here.
                correction=np.clip(correction+error,-.03,.03)
            # Evaluation-only: these values do not drive control.
            actual=d.body(name).xpos.copy()
            upright=float(d.body(name).xmat.reshape(3,3)[2,2])
            log.append({'evaluation':name,'target':target.tolist(),'actual':actual.tolist(),'upright_cosine':upright,'position_error_m':float(np.linalg.norm(actual[:2]-target)),'placed':bool(np.linalg.norm(actual[:2]-target)<POSITION_TOLERANCE and actual[2]<.05 and (name!='cup' or upright>.9))})
        final_seen=camera()
        Image.fromarray(renderer.render()).save(out/'final-overhead.png')
        camera_result={name:bool(name in final_seen and np.linalg.norm(np.array(final_seen[name]['xy'])-(plate+np.array([delta,0])))<POSITION_TOLERANCE) for name,delta in [('cup',.14),('utensil',-.14)]}
        plate_undisturbed=bool('plate' in final_seen and np.linalg.norm(np.array(final_seen['plate']['xy'])-plate)<.015)
        log.append({'camera_verification':camera_result,'plate_undisturbed':plate_undisturbed,'success':all(camera_result.values()) and plate_undisturbed and all(x['placed'] for x in log if 'evaluation' in x),'final_observation':final_seen,'evaluation_plate_position':d.body('plate').xpos.tolist()})
        renderer.update_scene(d,camera='overview')
        Image.fromarray(renderer.render()).save(out/'live.tmp',format='JPEG')
        (out/'live.tmp').replace(out/'live.jpg')
    (out/'episode.json').write_text(json.dumps(log,indent=2))
    if frames: frames[0].save(out/'episode.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
    print(json.dumps({'seed':seed,'success':log[-1]['success'],'output':str(out)}),flush=True)
    return log

if __name__=='__main__':run()
