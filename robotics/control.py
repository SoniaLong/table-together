"""Joint-limited numerical IK and actuator motion. Object state is never overwritten."""
import numpy as np
import mujoco
from scipy.optimize import least_squares

class Controller:
    def __init__(self, model, data):
        self.model,self.data=model,data
        self.ikdata=mujoco.MjData(model)
        self.frames=[]
        self.jaw_axis=None
        self.on_action=None

    def check_arm_contacts(self):
        for contact in self.data.contact:
            bodies=[self.model.body(self.model.geom_bodyid[int(i)]).name for i in contact.geom]
            if any(n.startswith('left_') for n in bodies) and any(n.startswith('right_') for n in bodies):
                raise RuntimeError('Arm-to-arm contact detected; episode stopped.')

    def solve(self,side,target):
        m,d=self.model,self.ikdata
        joints=[m.joint(side+'_'+name).id for name in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']]
        addresses=m.jnt_qposadr[joints]
        limits=m.jnt_range[joints]
        d.qpos[:]=self.data.qpos
        def residual(q):
            d.qpos[addresses]=q
            mujoco.mj_forward(m,d)
            site=d.site(side+'_gripperframe')
            direction=site.xmat.reshape(3,3)[:,0]
            error=np.r_[site.xpos-target, .10*(direction-np.array([0,0,-1]))]
            if self.jaw_axis is not None:error=np.r_[error,.03*(site.xmat.reshape(3,3)[:,2]-self.jaw_axis)]
            return error
        guesses=[self.data.qpos[addresses],np.array([0,.5,.5,.5,0]),np.array([0,-.5,1,1,0])]
        solutions=[least_squares(residual,np.clip(g,limits[:,0]+1e-5,limits[:,1]-1e-5),bounds=(limits[:,0],limits[:,1]),max_nfev=150) for g in guesses]
        best=min(solutions,key=lambda r:np.linalg.norm(r.fun))
        error=float(np.linalg.norm(best.fun[:3]))
        if error>.008: raise ValueError(f'{side} unreachable target {target}; IK error {error:.3f}m')
        return best.x,error

    def move(self,side,target,grip=.8,seconds=.8,record=None):
        q,error=self.solve(side,np.asarray(target))
        offset=0 if side=='left' else 6
        start=self.data.ctrl[offset:offset+6].copy()
        end=np.r_[q,grip]
        if self.on_action:
            command=self.data.ctrl.copy();command[offset:offset+6]=end
            self.on_action(command,seconds)
        steps=max(1,int(seconds/self.model.opt.timestep))
        for step in range(steps):
            t=(step+1)/steps; blend=t*t*(3-2*t)
            self.data.ctrl[offset:offset+6]=start+(end-start)*blend
            mujoco.mj_step(self.model,self.data)
            self.check_arm_contacts()
            if record and step%100==0: record()
        return {'ik_error_m':error,'actual_site':self.data.site(side+'_gripperframe').xpos.tolist()}

    def park(self,side,record=None):
        offset=0 if side=='left' else 6
        start=self.data.ctrl[offset:offset+6].copy()
        end=np.array([1.6 if side=='left' else -1.6,0,0,0,0,.9])
        if self.on_action:
            command=self.data.ctrl.copy();command[offset:offset+6]=end
            self.on_action(command,2.0)
        for step in range(1000):
            t=(step+1)/1000;blend=t*t*(3-2*t)
            self.data.ctrl[offset:offset+6]=start+(end-start)*blend
            mujoco.mj_step(self.model,self.data)
            self.check_arm_contacts()
            if record and step%100==0:record()

if __name__=='__main__':
    from scene import build,ROOT
    import json
    from PIL import Image
    m=build(); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    c=Controller(m,d)
    results={}
    for side,x in [('left',-.26),('right',.26)]:
        results[side]=c.move(side,[x,-.15,.08],seconds=2)
    with mujoco.Renderer(m,720,960) as r:
        r.update_scene(d,camera='overview')
        Image.fromarray(r.render()).save(ROOT/'artifacts/reach.png')
    print(json.dumps(results,indent=2))
