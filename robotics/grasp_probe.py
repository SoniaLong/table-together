"""Development contact-grasp experiment; failure is reported, never hidden."""
from scene import build,ROOT
from control import Controller
import mujoco
import json
from PIL import Image

m=build();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=Controller(m,d)
out=ROOT/'artifacts';out.mkdir(exist_ok=True)
report=[]
with mujoco.Renderer(m,720,960) as renderer:
    for label,target,grip in [('approach',[.156,-.122,.10],.9),('lower',[.156,-.122,.035],.9),('close',[.156,-.122,.035],.32),('lift',[.156,-.122,.10],.32)]:
        try: result=c.move('right',target,grip,seconds=1.5)
        except ValueError as exc:
            report.append({'stage':label,'error':str(exc)});break
        contacts=[]
        for contact in d.contact:
            names=[m.geom(int(i)).name for i in contact.geom]
            if 'cup_geom' in names: contacts.append(names)
        report.append({'stage':label,**result,'cup_position':d.body('cup').xpos.tolist(),'contacts':contacts})
        renderer.update_scene(d,camera='overview')
        Image.fromarray(renderer.render()).save(out/('grasp_'+label+'.png'))
report.append({'lift_success':bool(d.body('cup').xpos[2]>.065)})
(out/'grasp-probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))


