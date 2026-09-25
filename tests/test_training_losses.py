import unittest

import torch

from src import engine


class TrainingLossTests(unittest.TestCase):
    def test_ssim_is_one_for_identical_images_and_drops_with_noise(self):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(2, 32, 40, 3, generator=g)
        self.assertAlmostEqual(float(engine.ssim(x, x)), 1.0, places=5)
        noisy = (x + 0.2 * torch.randn(x.shape, generator=g)).clamp(0, 1)
        self.assertLess(float(engine.ssim(x, noisy)), 0.9)

    def test_scene_scale_uses_camera_centers(self):
        viewmats = torch.eye(4).repeat(2, 1, 1)
        viewmats[0, :3, 3] = torch.tensor([2., 0., 0.])   # center at (-2, 0, 0)
        viewmats[1, :3, 3] = torch.tensor([-2., 0., 0.])  # center at (2, 0, 0)
        self.assertAlmostEqual(engine._scene_scale(viewmats), 2.2, places=5)
        self.assertEqual(engine._scene_scale(torch.eye(4)[None]), 1.0)

    def test_target_chunks_shrink_for_large_frames(self):
        from pathlib import Path
        from unittest.mock import patch
        sizes = []
        def load(paths):
            sizes.append(len(paths))
            return torch.zeros((len(paths), 1080, 1920, 3), dtype=torch.uint8)
        with patch.object(engine, '_image_size', return_value=(1080, 1920)), \
             patch.object(engine, '_load_targets_parallel', side_effect=load):
            out = engine.build_targets_gpu([Path(f'{i}.jpg') for i in range(20)], 0, 1, False, .5, 'cpu')
        self.assertEqual(tuple(out.shape), (20, 1080, 1920, 3))
        self.assertEqual(sizes, [7, 7, 6])  # 16M-pixel budget, not a fixed 32 frames
