import unittest
from unittest.mock import patch

import torch

from src import engine


def fake_render(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kwargs):
    # Differentiable in every parameter, so all optimizers receive gradients.
    value = sum(t.mean() for t in (means, quats, scales, opacities, colors))
    return value.expand(len(viewmats), height, width, 3), None, {}


def scene(n=64):
    g = torch.Generator().manual_seed(0)
    return {'means': torch.rand(n, 3, generator=g), 'colors': torch.rand(n, 3, generator=g),
            'viewmats': torch.eye(4).repeat(2, 1, 1), 'Ks': torch.eye(3).repeat(2, 1, 1)}


class DensifyTests(unittest.TestCase):
    def train(self, device='cpu', **kwargs):
        with patch.object(engine, 'rasterization', side_effect=fake_render):
            return engine.train_splats(scene(), [], 4, 3, 40, device, 1, targets=torch.zeros(2, 3, 4, 3, dtype=torch.uint8),
                                       mixed_precision=False, ssim_weight=0, **kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "gsplat's MCMC relocation kernels need CUDA")
    def test_mcmc_grows_up_to_the_cap(self):
        grown = self.train('cuda', densify='mcmc', refine_every=5, max_gaussians=1000)
        self.assertGreater(len(grown['means']), 64)
        capped = self.train('cuda', densify='mcmc', refine_every=5, max_gaussians=70)
        self.assertEqual(len(capped['means']), 70)
        for key in ('means', 'colors', 'scales', 'quats', 'opacities'):
            self.assertEqual(len(capped[key]), 70)
            self.assertTrue(torch.isfinite(capped[key]).all())

    def test_invalid_densify_settings(self):
        for kwargs in ({'densify': 'split'}, {'max_gaussians': 0}, {'refine_every': 0}, {'densify': 'mcmc'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.train(**kwargs)

    def test_pose_corrections_start_at_identity_and_receive_gradients(self):
        views = torch.eye(4).repeat(3, 1, 1)
        views[:, :3, 3] = torch.randn(3, 3)
        deltas = torch.zeros(3, 6, requires_grad=True)
        corrected = engine._correct_poses(views, deltas)
        torch.testing.assert_close(corrected, views)
        corrected[:, :3, 3].sum().backward()
        self.assertTrue(torch.isfinite(deltas.grad).all())
        self.assertGreater(deltas.grad.abs().sum().item(), 0)
        result = self.train(pose_opt=True, voxel_guided=False)
        self.assertEqual(tuple(result['pose_deltas'].shape), (2, 6))
