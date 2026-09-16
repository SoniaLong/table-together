from scene import build,initial_data,ROOT
from control import Controller
import mujoco,json
import numpy as np
from PIL import Image
m=build();d=initial_data(m);c=Controller(m,d);c.jaw_axis=np.array([1,0,0])
report=[]
for stage,z,grip in [('approach',.07,.9),('lower',.017,.9),('close',.017,-.04),('lift',.07,-.04)]:
    result=c.move('left',[-.260,-.15,z],grip,seconds=1.5)
    report.append({'stage':stage,**result,'object':d.body('utensil').xpos.tolist(),'gripper':float(d.qpos[5])})
with mujoco.Renderer(m,720,960) as r:
    r.update_scene(d,camera='overview');Image.fromarray(r.render()).save(ROOT/'artifacts/utensil-probe.png')
print(json.dumps(report,indent=2))
