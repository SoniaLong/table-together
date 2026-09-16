import unittest
import numpy as np
import mujoco
from scene import build,initial_data,ROOT
from perception import observe
from control import Controller

class RoboticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=build()
    def test_two_arms_and_free_objects(self):
        self.assertEqual(self.m.nu,12)
        self.assertEqual(self.m.ncam,4)
        for name in ['plate','cup','utensil']:
            self.assertEqual(self.m.joint(name+'_free').type,mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(self.m.neq,0,'No grasp welds should be present')
    def test_rgb_localization(self):
        d=initial_data(self.m)
        with mujoco.Renderer(self.m,480,640) as renderer:
            renderer.update_scene(d,camera='overhead');seen=observe(renderer.render())
        self.assertEqual(set(seen),{'plate','cup','utensil'})
        for name,result in seen.items():
            self.assertLess(np.linalg.norm(np.array(result['xy'])-d.body(name).xpos[:2]),.005)
    def test_unreachable_target_rejected(self):
        c=Controller(self.m,initial_data(self.m))
        with self.assertRaises(ValueError):c.solve('left',[2,2,2])

if __name__=='__main__':unittest.main()
