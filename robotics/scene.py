"""Two real SO-101 MJCF models; free objects move only through MuJoCo physics."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor/mujoco_menagerie/robotstudio_so101'

def build(seed=None,jitter=.005,profile='standard'):
    original = ET.parse(SOURCE / 'so101.xml').getroot()
    root = ET.Element('mujoco', model='TableTogether dual SO101')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', meshdir=str(SOURCE / 'assets'))
    ET.SubElement(root, 'option', timestep='0.002', integrator='implicitfast', cone='elliptic')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    root.append(copy.deepcopy(original.find('default')))
    # Rubber fingertip contact baseline; no welds, grasp constraints or teleports.
    for node in root.findall('.//default'):
        if node.get('class') in ('collision_gripper','collision_gripper_mesh'):
            node.find('geom').set('friction','2 .01 .001')
    root.append(copy.deepcopy(original.find('asset')))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 1.5', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='table', type='box', pos='0 0 -.035', size='.55 .4 .035', rgba='.16 .20 .24 1')
    ET.SubElement(world, 'camera', name='overhead', pos='0 0 1.1', quat='1 0 0 0', fovy='48')
    ET.SubElement(world, 'camera', name='overview', pos='.85 -.9 .8', xyaxes='.727 .687 0 -.40 .424 .812', fovy='48')
    actuators = ET.SubElement(root, 'actuator')
    for side, x in [('left', -.26), ('right', .26)]:
        mount = ET.SubElement(world, 'body', name=side+'_mount', pos=f'{x} .13 0', quat='.70710678 0 0 -.70710678')
        for body in original.find('worldbody'):
            element = copy.deepcopy(body)
            for node in element.iter():
                if 'name' in node.attrib: node.set('name', side+'_'+node.get('name'))
            mount.append(element)
        for act in original.find('actuator'):
            element = copy.deepcopy(act)
            for attr in ['name', 'joint']: element.set(attr, side+'_'+element.get(attr))
            actuators.append(element)
    objects = [('plate',(0,-.13,.013),'cylinder','.055 .012','.12 .55 .95 1'),
               ('cup',(.20,-.13,.029),'cylinder','.022 .028','.98 .35 .12 1'),
               ('utensil',(-.25,-.15,.011),'box','.008 .042 .01','.25 .9 .45 1')]
    if seed is not None:
        import numpy as np
        rng=np.random.default_rng(seed)
        objects=[(name,(pos[0]+rng.uniform(-jitter,jitter),pos[1]+rng.uniform(-jitter,jitter),pos[2]),kind,size,color) for name,pos,kind,size,color in objects]
    for name, pos, kind, size, color in objects:
        body = ET.SubElement(world,'body', name=name, pos=' '.join(map(str,pos)))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'geom',name=name+'_geom',type=kind,size=size,rgba=color,mass='.035',friction='1 .01 .001',condim='4')
    if profile not in ('standard','lighting','physics','size','combined'):
        raise ValueError('Unknown randomization profile')
    if profile!='standard':
        import numpy as np
        variation=np.random.default_rng((seed or 0)+100000)
        if profile in ('lighting','combined'):
            world.find('light').set('diffuse',' '.join(map(str,variation.uniform(.55,1,3))))
            world.find("geom[@name='table']").set('rgba',' '.join(map(str,[*variation.uniform(.10,.28,3),1])))
        for name,_,_,_,_ in objects:
            geom=world.find(f"body[@name='{name}']/geom")
            if profile in ('physics','combined'):
                geom.set('mass',str(.035*variation.uniform(.7,1.3)))
                geom.set('friction',f'{variation.uniform(.7,1.3)} .01 .001')
            if profile in ('size','combined'):
                sizes=list(map(float,geom.get('size').split()))
                # Vary horizontal dimensions only; preserve initial table clearance.
                sizes[0]*=variation.uniform(.9,1.1)
                if geom.get('type')=='box':sizes[1]*=variation.uniform(.9,1.1)
                geom.set('size',' '.join(map(str,sizes)))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))

def initial_data(model):
    data=mujoco.MjData(model)
    for side,pan,offset in [('left',1.6,0),('right',-1.6,6)]:
        data.qpos[model.joint(side+'_shoulder_pan').qposadr]=pan
        data.ctrl[offset]=pan
        data.qpos[model.joint(side+'_gripper').qposadr]=.9
        data.ctrl[offset+5]=.9
    mujoco.mj_forward(model,data)
    return data

if __name__ == '__main__':
    from PIL import Image
    import json
    model = build()
    data = initial_data(model)
    mujoco.mj_forward(model,data)
    for _ in range(500): mujoco.mj_step(model,data)
    out = ROOT/'artifacts'; out.mkdir(exist_ok=True)
    with mujoco.Renderer(model,720,960) as renderer:
        for camera in ['overhead','overview']:
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(out/(camera+'.png'))
    print(json.dumps({'actuators':model.nu,'cameras':model.ncam,'simulation_seconds':data.time,'objects':[data.body(n).xpos.tolist() for n in ['plate','cup','utensil']]}))
