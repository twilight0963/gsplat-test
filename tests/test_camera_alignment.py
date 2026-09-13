import unittest
import numpy as np
from src.gsplat_viewer import OrbitCamera

class CameraAlignmentTests(unittest.TestCase):
    def camera(self):
        a=.6
        axes=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],np.float32)
        return OrbitCamera(np.array([2.,3.,4.]),5,axes),axes

    def test_startup_tracks_box_and_projects_up_upward(self):
        cam,axes=self.camera()
        np.testing.assert_allclose(cam.orbit_frame[:,1],axes[:,1],atol=1e-6)
        view=cam.viewmat()
        np.testing.assert_allclose(view[:3,:3]@view[:3,:3].T,np.eye(3),atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(view[:3,:3]),1.,places=6)
        self.assertLess((view[:3,:3]@(cam.orbit_frame[:,1]*cam.up_sign))[1],0)
        np.testing.assert_allclose(view[:3,:3]@cam.eye()+view[:3,3],0,atol=1e-6)

    def test_straighten_preserves_orbit_and_pan_but_removes_roll_and_flip(self):
        cam,_=self.camera()
        cam.azimuth=.8;cam.elevation=.4;cam.pan(20,10)
        eye=cam.eye().copy();target=cam.target.copy()
        cam.roll=.7;cam.up_sign=1
        cam.straighten()
        np.testing.assert_allclose(cam.eye(),eye)
        np.testing.assert_allclose(cam.target,target)
        self.assertEqual(cam.roll,0);self.assertEqual(cam.up_sign,-1)
        self.assertLess((cam.viewmat()[:3,:3]@(cam.orbit_frame[:,1]*cam.up_sign))[1],0)

    def test_reset_restores_box_aligned_start(self):
        cam,_=self.camera();initial=cam.viewmat().copy()
        cam.roll=1;cam.up_sign=1;cam.radius=10;cam.azimuth=1;cam.pan(10,10)
        cam.reset()
        np.testing.assert_allclose(cam.viewmat(),initial)
