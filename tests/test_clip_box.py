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
