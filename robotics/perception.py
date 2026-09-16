"""Calibrated RGB color segmentation baseline, not a learned vision model.

Uses pixels and known camera calibration/object dimensions, never object body poses.
Designed for the deliberately color-coded tableware proxies in this first scene.
"""
import numpy as np
from scipy.ndimage import label

def observe(rgb, fovy=48, camera_height=1.1):
    rgb=rgb.astype(float)/255
    r,g,b=rgb.transpose(2,0,1)
    masks={'plate':(b>r*1.5)&(b>g*.95)&(b>.3),
           'cup':(r>g*1.25)&(r>b*2)&(r>.3),
           'utensil':(g>r*1.3)&(g>b*1.1)&(g>.25)}
    tops={'plate':.024,'cup':.056,'utensil':.020}
    h,w=r.shape; focal=.5*h/np.tan(np.deg2rad(fovy/2))
    results={}
    for name,mask in masks.items():
        components,n=label(mask)
        if n==0: continue
        areas=np.bincount(components.ravel());areas[0]=0
        largest=int(areas.argmax())
        if areas[largest]<20: continue
        yy,xx=np.where(components==largest)
        # Bounding-box center limits side-face perspective bias for cylinders.
        u=(xx.min()+xx.max())*.5;v=(yy.min()+yy.max())*.5
        depth=camera_height-tops[name]
        results[name]={'xy':[(u-(w-1)/2)*depth/focal, -(v-(h-1)/2)*depth/focal],
                       'pixel_center':[float(u),float(v)],'visible_pixels':int(areas[largest]),'method':'calibrated_rgb_color_baseline'}
    return results

if __name__=='__main__':
    from scene import build,ROOT,initial_data
    import mujoco,json
    m=build();d=initial_data(m)
    with mujoco.Renderer(m,720,960) as renderer:
        renderer.update_scene(d,camera='overhead')
        results=observe(renderer.render())
    # Evaluation-only ground truth; not passed to observe() or used by its output.
    evaluation={name:{**value,'evaluation_error_m':float(np.linalg.norm(np.array(value['xy'])-d.body(name).xpos[:2]))} for name,value in results.items()}
    (ROOT/'artifacts/perception.json').write_text(json.dumps(evaluation,indent=2))
    print(json.dumps(evaluation,indent=2))
