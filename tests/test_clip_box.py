import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.clip_box import estimate_box, box_mask, validate_box, estimate_oriented_box, validate_axes, box_view
from src.gltf_gsplat import write_gsplat_glb, read_gsplat_glb

class ClipTests(unittest.TestCase):
    def test_orientation_tracks_rotated_cloud(self):
        from itertools import product
        points = np.array(list(product(np.linspace(-4,4,13), np.linspace(-2,2,9), np.linspace(-.5,.5,5))))
        angle = .6
        rotation = np.array([[np.cos(angle),-np.sin(angle),0],
                             [np.sin(angle),np.cos(angle),0],[0,0,1]])
        cloud = points @ rotation.T + [12, -7, 3]
        bounds, axes = estimate_oriented_box(cloud)
        validate_axes(axes)
        np.testing.assert_allclose(np.abs(rotation.T @ axes), np.eye(3), atol=1e-5)
        np.testing.assert_allclose(bounds.mean(axis=0) @ axes.T, [12,-7,3], atol=1e-5)
        # The same box-local transform is used for drawing and ray clipping.
        camera = np.eye(4, dtype=np.float32)
        camera[:3,3] = [0,0,15]
        local = cloud @ axes
        np.testing.assert_allclose(local @ box_view(camera, axes)[:3,:3].T + camera[:3,3],
                                   cloud + camera[:3,3], atol=1e-5)

    def test_rotation_metadata_roundtrip(self):
        bounds, axes = estimate_oriented_box(np.array([[0,0,0],[2,1,0],[4,2,1],[1,0,2]]))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rotated.glb'
            write_gsplat_glb(p,np.zeros((1,3)),np.ones((1,3)),np.array([[1,0,0,0]]),np.ones(1),np.ones((1,3)),clip_bounds=bounds,clip_axes=axes)
            result=read_gsplat_glb(p)
            np.testing.assert_array_equal(result['clip_axes'],axes)
            np.testing.assert_array_equal(result['clip_bounds'],bounds)

    def test_cube_rejects_outlier_influence(self):
        points = np.random.default_rng(4).uniform(-1, 1, (1000, 3))
        bounds = estimate_box(np.vstack([points, [1000, 1000, 1000]]))
        np.testing.assert_allclose(np.diff(bounds, axis=0)[0], np.repeat(bounds[1,0]-bounds[0,0],3))
        self.assertLess(bounds.max(), 2)

    def test_ray_mask_outside_inside_and_behind(self):
        K = np.array([[10,0,10],[0,10,10],[0,0,1]],np.float32)
        view = np.eye(4,dtype=np.float32)
        mask=box_mask(np.array([[-1,-1,2],[1,1,4]]),view,K,20,20)
        self.assertTrue(mask[10,10]);self.assertFalse(mask[0,0])
        self.assertFalse(box_mask(np.array([[-1,-1,-4],[1,1,-2]]),view,K,20,20).any())
        self.assertTrue(box_mask(np.array([[-1,-1,-1],[1,1,1]]),view,K,20,20).all())

    def test_bounds_persist_in_glb(self):
        bounds=np.array([[-2,-2,-2],[2,2,2]],np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.glb'
            write_gsplat_glb(p,np.zeros((1,3)),np.ones((1,3)),np.array([[1,0,0,0]]),np.ones(1),np.ones((1,3)),clip_bounds=bounds)
            np.testing.assert_array_equal(read_gsplat_glb(p)['clip_bounds'],bounds)
        with self.assertRaises(ValueError): validate_box([[0,0,0],[0,0,0]])


class BoxMaskTorchTests(unittest.TestCase):
    def test_matches_the_numpy_mask(self):
        from src.clip_box import box_mask_torch
        from src.gsplat_viewer import OrbitCamera, make_K
        rng = np.random.default_rng(0)
        bounds = np.array([[-1., -.5, -2.], [1.5, .5, 1.]])
        K = make_K(96, 64, 60)
        for azimuth, elevation, radius in ((0., .5, 6.), (2., -.3, 3.), (4., 1.2, .8), (1., 0., .2)):
            cam = OrbitCamera(bounds.mean(0), radius)
            cam.azimuth, cam.elevation = azimuth, elevation
            view = cam.viewmat()
            with self.subTest(azimuth=azimuth, radius=radius):
                expected = box_mask(bounds, view, K, 96, 64)
                actual = box_mask_torch(bounds, view, K, 96, 64, 'cpu').numpy()
                self.assertLessEqual((expected != actual).sum(), 2)  # float rounding at the very edge only
        view = rng.normal(size=(4, 4)).astype(np.float32)
        view[:3, :3] = np.linalg.qr(view[:3, :3])[0]
        self.assertEqual(box_mask_torch(bounds, view, K, 96, 64, 'cpu').shape, (64, 96))
