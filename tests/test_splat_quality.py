import unittest
import torch
from src.engine import _scale_regularizer, _reset_opacities
from src.voxel_reconstruction import VoxelGuidedConfig, VoxelGuidedOptimizer


def setup(opacity):
    n = len(opacity)
    params = {name: torch.zeros((n, d) if d else (n,), requires_grad=True)
              for name, d in zip(VoxelGuidedOptimizer.PARAM_NAMES, (3, 3, 3, 4, 0))}
    with torch.no_grad():
        params['opacities'].copy_(torch.logit(torch.tensor(opacity)))
    optimizer = torch.optim.Adam([{'params': [p], 'name': name} for name, p in params.items()])
    voxel = VoxelGuidedOptimizer(params['means'].detach(), VoxelGuidedConfig(), 'cpu')
    return params, optimizer, voxel


class SplatQualityTests(unittest.TestCase):
    def test_opaque_singleton_survives_and_dense_fog_is_removed(self):
        p, opt, voxel = setup([0.8, 0.8, 0.001, 0.02])
        voxel.voxel_id = voxel.voxel_id + torch.tensor([0, 1, 1, 2])
        p = voxel._prune(opt, p)
        torch.testing.assert_close(p['opacities'].sigmoid(), torch.tensor([0.8, 0.8]))
        self.assertEqual(len(voxel.grad_count), 2)

    def test_pruning_never_erases_entire_model(self):
        p, opt, voxel = setup([0.001, 0.001])
        self.assertIs(voxel._prune(opt, p), p)

    def test_growth_cap_and_leaf_gradients(self):
        p, opt, voxel = setup([0.5, 0.5])
        sum(t.sum() for t in p.values()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        voxel.cfg.max_gaussians = 3
        voxel.grad_accum.fill_(1)
        voxel.grad_count.fill_(1)
        p = voxel._densify(opt, p)
        self.assertEqual(len(p['means']), 3)
        for _ in range(2):
            sum(t.sum() for t in p.values()).backward()
            self.assertTrue(all(t.is_leaf and t.grad is not None for t in p.values()))
            opt.step()
            opt.zero_grad(set_to_none=True)

    def test_shape_penalty_shrinks_long_axis(self):
        scales = torch.tensor([[100., 1., 1.], [2., 1., 1.]]).log().requires_grad_()
        loss = _scale_regularizer(scales, 10)
        loss.backward()
        self.assertGreater(scales.grad[0, 0].item(), 0)
        self.assertLess(scales.grad[0, 1].item(), 0)
        self.assertEqual(scales.grad[1].abs().sum().item(), 0)

    def test_damping_allows_shape_and_opacity_recovery(self):
        p, opt, voxel = setup([0.5, 0.5])
        sum(t.sum() for t in p.values()).backward()
        voxel.dampen_gradients(p['means'], p['scales'], p['colors'], p['quats'], p['opacities'])
        torch.testing.assert_close(p['scales'].grad, torch.ones(2, 3))
        torch.testing.assert_close(p['opacities'].grad, torch.ones(2))

    def test_reset_does_not_increase_fog_and_pauses_refinement(self):
        p, opt, voxel = setup([0.8, 0.001])
        sum(t.sum() for t in p.values()).backward()
        opt.step()
        _reset_opacities(p['opacities'], opt)
        self.assertLessEqual(p['opacities'].sigmoid()[1].item(), 0.001)
        self.assertAlmostEqual(p['opacities'].sigmoid()[0].item(), 0.01, places=6)
        voxel.pause_after_reset(3000, 200)
        result = voxel.maybe_densify_and_prune(3100, opt, *[p[n] for n in voxel.PARAM_NAMES])
        self.assertTrue(all(result[i] is p[n] for i, n in enumerate(voxel.PARAM_NAMES)))
        self.assertEqual(opt.state[p['opacities']]['exp_avg'].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
