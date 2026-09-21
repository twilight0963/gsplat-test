import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src import engine
from src.super_resolution import upscale_tiled, validate_setup


class Nearest2x(torch.nn.Module):
    def forward(self, image):
        return F.interpolate(image, scale_factor=2, mode='nearest')


class SuperResolutionTests(unittest.TestCase):
    def test_tiles_cover_odd_small_and_overlapping_images(self):
        for h, w in [(1, 1), (7, 13), (33, 65), (101, 137)]:
            with self.subTest(shape=(h, w)):
                original = torch.rand(3, h, w)
                result = upscale_tiled(Nearest2x(), original, 32, 'cpu')
                expected = F.interpolate(original[None], scale_factor=2, mode='nearest')[0]
                torch.testing.assert_close(result, expected)

    def test_consistency_loss_reaches_render_and_preserves_base_reference(self):
        rendered = torch.full((1, 4, 6, 3), 0.8, requires_grad=True)
        enhanced = torch.ones_like(rendered)
        base = torch.zeros(1, 2, 3, 3)
        loss = engine.sr_training_loss(rendered, enhanced, base, 0.25)
        self.assertAlmostEqual(loss.item(), 0.65, places=6)
        loss.backward()
        self.assertTrue(torch.all(rendered.grad > 0))
        torch.testing.assert_close(base, torch.zeros_like(base))

    def test_setup_fails_before_work_for_missing_weights_or_invalid_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                validate_setup(root, root/'missing.pth', 128)
            (root/'models').mkdir()
            (root/'models/network_swinir.py').touch()
            (root/'weights.pth').touch()
            with self.assertRaises(ValueError):
                validate_setup(root, root/'weights.pth', 31)

    def test_original_extraction_bypasses_all_enhancement(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.full((8, 8, 3), 100, np.uint8)
            with patch.object(cv2, 'VideoCapture') as capture, patch.object(engine, 'apply_cas') as cas:
                reader = capture.return_value
                reader.isOpened.return_value = True
                reader.get.return_value = 30
                reader.read.side_effect = [(True, frame), (True, frame), (False, None)]
                output = engine.extract_frames(Path('in.mp4'), Path(tmp), 1, 0, 50, 2,
                                               original_only=True)
                self.assertEqual(int(cv2.imread(str(output/'frame_000000.jpg'))[0, 0, 0]), 100)
                cas.assert_not_called()
                reader.release.assert_called_once()

    def test_base_excludes_sharpening_and_targets_are_in_camera_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [root/'z.png', root/'a.png']
            for p, value in zip(sources, [40, 80]):
                cv2.imwrite(str(p), np.full((8, 10, 3), value, np.uint8))
            def fake_sr(command):
                import json
                manifest = Path(command[command.index('--manifest')+1])
                output = Path(command[command.index('--output')+1])
                output.mkdir()
                for path in json.loads(manifest.read_text()):
                    frame = cv2.imread(path)
                    cv2.imwrite(str(output/Path(path).name), cv2.resize(frame, (20, 16)))
            with patch.object(engine, '_run', side_effect=fake_sr), \
                 patch.object(engine, 'apply_cas', side_effect=lambda images, *_: images) as cas, \
                 patch.object(engine, 'unsharp_mask') as unsharp:
                enhanced, base = engine.prepare_sr_targets(sources, root/'sr', 10, 8, 5, 2,
                                                          .2, False, root, root/'weights', 32, 'cpu')
            self.assertEqual([int(cv2.imread(str(p))[0,0,0]) for p in base], [85,165])
            self.assertEqual([p.name for p in enhanced], [p.name for p in base])
            unsharp.assert_not_called()
            self.assertEqual(cas.call_args.args[2], .2)

    def test_sr_training_reads_paired_batches_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            high, base = root/'high.png', root/'base.png'
            cv2.imwrite(str(high), np.full((4, 6, 3), 120, np.uint8))
            cv2.imwrite(str(base), np.full((2, 3, 3), 100, np.uint8))
            data = {
                'means': torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]),
                'colors': torch.full((4, 3), .5),
                'viewmats': torch.eye(4)[None], 'Ks': torch.eye(3)[None],
            }
            def render(means, quats, scales, opacities, colors, *args, **kwargs):
                value = sum(t.mean() for t in (means, quats, scales, opacities, colors))
                return value.expand(1, 4, 6, 3), None, None
            before = data['means'].clone()
            with patch.object(engine, 'rasterization', side_effect=render), \
                 patch.object(engine, 'sr_training_loss', wraps=engine.sr_training_loss) as loss:
                result = engine.train_splats(data, [high], 6, 4, 2, 'cpu', 1,
                                            voxel_guided=False, mixed_precision=False,
                                            base_images=[base])
            self.assertEqual(loss.call_count, 2)
            self.assertEqual(loss.call_args.args[2].shape, (1, 2, 3, 3))
            self.assertFalse(torch.equal(result['means'], before))
            self.assertTrue(torch.isfinite(result['means']).all())

    def test_build_scales_intrinsics_and_uses_single_view_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {'Ks': torch.tensor([[[10.,0,4],[0,11,3],[0,0,1]]])}
            with patch('src.super_resolution.validate_setup'), \
                 patch.object(engine, 'extract_frames', return_value=root/'original') as extract, \
                 patch.object(engine, 'run_colmap') as colmap, \
                 patch.object(engine, 'load_reconstruction', return_value=(data, [root/'raw.png'], 8, 6)), \
                 patch.object(engine, 'prepare_sr_targets', return_value=([root/'high.png'], [root/'base.png'])), \
                 patch.object(engine, 'PreviewPublisher'), patch.object(engine, 'ensure_viewer'), \
                 patch.object(engine, 'train_splats', return_value={}) as train, \
                 patch.object(engine, 'export_gltf'):
                engine.build_model(root/'video.mp4', root, super_resolution=True, use_server=True,
                                   swinir_root=root, sr_checkpoint=root/'weights')
            self.assertTrue(extract.call_args.kwargs['original_only'])
            self.assertFalse(extract.call_args.kwargs['unsharp'])
            self.assertEqual(colmap.call_args.args[0], root/'original')
            self.assertEqual(train.call_args.args[2:4], (16, 12))
            self.assertEqual(train.call_args.args[6], 1)
            self.assertEqual(train.call_args.kwargs['base_images'], [root/'base.png'])
            torch.testing.assert_close(data['Ks'], torch.tensor([[[20.,0,8],[0,22,6],[0,0,1]]]))


if __name__ == '__main__':
    unittest.main()
