import unittest
import torch
from src.gsplat_viewer import _clamp_scale_outliers, _clamp_at_box_edges


class ScaleClampTests(unittest.TestCase):
    def test_only_oversized_splats_shrink_and_preserve_shape(self):
        scales = torch.tensor([[1., .5, .25]] * 20 + [[100., 50., 25.]])
        original = scales.clone()
        result = _clamp_scale_outliers(scales)
        torch.testing.assert_close(result[:20], original[:20])
        torch.testing.assert_close(result[-1], torch.tensor([1., .5, .25]))
        torch.testing.assert_close(scales, original)
        torch.testing.assert_close(result[-1] / result[-1, 0], original[-1] / original[-1, 0])

    def test_respects_scene_scale_and_normal_spread(self):
        scales = torch.arange(1., 21.)[:, None].repeat(1, 3)
        torch.testing.assert_close(_clamp_scale_outliers(scales), scales)
        scales[-1] *= 100
        torch.testing.assert_close(_clamp_scale_outliers(scales * 7), _clamp_scale_outliers(scales) * 7)

    def test_adjustable_threshold_and_disable(self):
        scales = torch.tensor([[1., .5, .25]] * 20 + [[100., 50., 25.]])
        torch.testing.assert_close(_clamp_scale_outliers(scales, 2)[-1], torch.tensor([2., 1., .5]))
        torch.testing.assert_close(_clamp_scale_outliers(scales, 0), scales)
        torch.testing.assert_close(_clamp_scale_outliers(scales, 100), scales)
        for invalid in (-1., float('nan'), float('inf'), 1001.):
            with self.assertRaises(ValueError): _clamp_scale_outliers(scales, invalid)

    def test_edge_clamp_smooth_transition_and_disabled(self):
        scales = torch.tensor([[1., .5, .25]] * 20 + [[100., 50., 25.]] * 5)
        positions = torch.zeros(25, 3)
        positions[-5:, 0] = torch.tensor([0., .25, .5, .75, 1.])
        bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
        result = _clamp_at_box_edges(scales, positions, bounds)
        torch.testing.assert_close(result[-5], scales[-5])
        torch.testing.assert_close(result[-1], torch.tensor([1., .5, .25]))
        torch.testing.assert_close(result[-3], torch.tensor([100., 50., 25.]) / 50.5)
        self.assertTrue(torch.all(result[-5:-1, 0] > result[-4:, 0]))
        torch.testing.assert_close(result[:20], scales[:20])
        torch.testing.assert_close(_clamp_at_box_edges(scales, positions, bounds, 0), scales)
        # Translation and uniform scene scaling preserve relative behavior.
        torch.testing.assert_close(_clamp_at_box_edges(scales, positions * 3 + 7, bounds * 3 + 7), result)

    def test_huge_outliers_converge_near_edge(self):
        scales = torch.tensor([[1., .5, .25]] * 40 + [[100.,50.,25.],[10000.,5000.,2500.]])
        positions = torch.zeros(42, 3)
        positions[-2:, 0] = .9
        bounds = torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
        result = _clamp_at_box_edges(scales, positions, bounds)
        self.assertLess(result[-2:, 0].max().item(), 1.02)
        self.assertLess(abs(result[-1, 0] - result[-2, 0]).item(), .001)
        torch.testing.assert_close(result[-2:,1] / result[-2:,0], torch.full((2,),.5))
        torch.testing.assert_close(result[:40], scales[:40])

    def test_face_diagonal_transition_is_smooth_and_symmetric(self):
        scales = torch.tensor([[1.,1.,1.]] * 40 + [[100.,100.,100.]] * 3)
        positions = torch.zeros(43,3)
        positions[-3:] = torch.tensor([[.4999,.5,0],[.5,.5,0],[.5001,.5,0]])
        bounds = torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
        result = _clamp_at_box_edges(scales,positions,bounds)[-3:,0]
        self.assertLess(abs(result[0] - 2*result[1] + result[2]).item(), 1e-5)
        reflected = _clamp_at_box_edges(scales,-positions,bounds)[-3:,0]
        torch.testing.assert_close(result,reflected)

    def test_empty_tiny_zero_and_invalid_inputs(self):
        for scales in (torch.empty(0, 3), torch.zeros(10, 3), torch.ones(3, 3)):
            torch.testing.assert_close(_clamp_scale_outliers(scales), scales)
        for invalid in (float('inf'), float('nan'), -1.):
            with self.assertRaises(ValueError):
                _clamp_scale_outliers(torch.full((8, 3), invalid))
