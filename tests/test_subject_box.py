import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.clip_box import fit_scene_box
from src.gltf_gsplat import read_gsplat_glb, write_gsplat_glb


def inside(points, bounds, axes):
    local = np.asarray(points) @ axes
    return ((local >= bounds[0]) & (local <= bounds[1])).all(axis=1)


class SceneBoxTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        # A wide, flat site (40 x 30 x ~1) with a thin 12-unit tower standing on it.
        self.ground = np.column_stack([rng.uniform(-20, 20, 20000), rng.uniform(-15, 15, 20000), rng.normal(0, .2, 20000)])
        self.tower = np.column_stack([rng.normal(12, .15, 600), rng.normal(-6, .15, 600), rng.uniform(0, 12, 600)])

    def test_keeps_the_whole_site_including_thin_connected_structures(self):
        points = np.vstack([self.ground, self.tower])
        bounds, axes = fit_scene_box(points)
        self.assertGreater(inside(self.ground, bounds, axes).mean(), .995)
        self.assertGreater(inside(self.tower, bounds, axes).mean(), .99)  # the tower top is not cut off
        size = np.sort(bounds[1] - bounds[0])
        self.assertLess(size[0], size[2] / 2.5)  # flat box for a flat scene, not a cube

    def test_leaves_out_isolated_floaters_and_transparent_splats(self):
        rng = np.random.default_rng(3)
        floaters = rng.normal([200, 150, 90], 1, (80, 3))  # a small clump far away
        haze = rng.uniform(-400, 400, (300, 3))  # near-transparent splats everywhere
        points = np.vstack([self.ground, floaters, haze])
        opacity = np.concatenate([np.full(len(self.ground), .9), np.full(80, .9), np.full(300, .01)])
        bounds, axes = fit_scene_box(points, opacity)
        self.assertGreater(inside(self.ground, bounds, axes).mean(), .995)
        self.assertFalse(inside(floaters, bounds, axes).any())
        self.assertLess((bounds[1] - bounds[0]).max(), 60)

    def test_keeps_every_large_part_of_a_split_scene(self):
        rng = np.random.default_rng(5)
        east = rng.normal([30, 0, 0], [4, 4, .3], (8000, 3))
        west = rng.normal([-30, 0, 0], [4, 4, .3], (6000, 3))
        bounds, axes = fit_scene_box(np.vstack([east, west]))
        self.assertGreater(inside(east, bounds, axes).mean(), .99)
        self.assertGreater(inside(west, bounds, axes).mean(), .99)

    def test_tiny_or_duplicated_inputs(self):
        points = np.random.default_rng(1).normal(size=(10, 3))
        bounds, axes = fit_scene_box(points)
        self.assertTrue(inside(points, bounds, axes).all())
        many = np.random.default_rng(2).normal(size=(500, 3))
        b1, _ = fit_scene_box(many)
        b2, _ = fit_scene_box(np.vstack([many, np.repeat(many[:1], 1000, axis=0)]))
        self.assertTrue(inside(many, b2, _).mean() > .99)
        np.testing.assert_allclose((b1[1] - b1[0]).max(), (b2[1] - b2[0]).max(), rtol=.3)

    def test_final_box_metadata_is_preserved(self):
        bounds, axes = fit_scene_box(np.random.default_rng(3).normal(size=(100, 3)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.glb'
            write_gsplat_glb(path, np.zeros((1, 3)), np.ones((1, 3)), np.array([[1, 0, 0, 0]]), np.ones(1), np.ones((1, 3)),
                             clip_bounds=bounds, clip_axes=axes, clip_final=True)
            data = read_gsplat_glb(path)
            self.assertTrue(data['clip_final'])
            np.testing.assert_array_equal(data['clip_bounds'], bounds)
            np.testing.assert_array_equal(data['clip_axes'], axes)


class FramingTests(unittest.TestCase):
    def test_camera_shows_every_corner_and_is_not_too_far(self):
        from itertools import product
        from src.gsplat_viewer import OrbitCamera, framing_radius, make_K
        bounds = np.array([[-25, -2, -18], [25, 2, 18]], np.float32)  # wide and flat
        axes = np.eye(3, dtype=np.float32)
        width, height = 1280, 720
        K = make_K(width, height, 60)
        cam = OrbitCamera(bounds.mean(0), 1.0, box_axes=axes)
        radius = framing_radius(cam, bounds, axes, K, width, height)
        cam.radius = radius

        def corner_pixels(camera):
            corners = np.array(list(product(*zip(bounds[0], bounds[1]))), np.float64)
            view = camera.viewmat()
            c = corners @ view[:3, :3].T + view[:3, 3]
            p = c @ K.T
            return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2], c[:, 2]

        x, y, z = corner_pixels(cam)
        self.assertTrue((z > 0).all() and (x >= 0).all() and (x <= width).all() and (y >= 0).all() and (y <= height).all())
        # Tight: one corner reaches the 90% frame margin, and the box diagonal would be much farther.
        margin = min(x.min(), width - x.max(), y.min() * width / height, (height - y.max()) * width / height)
        self.assertLess(margin, width * .06)
        self.assertLess(radius, .9 * float(np.linalg.norm(bounds[1] - bounds[0])))
