import contextlib
import io
import unittest
from unittest.mock import patch

import torch
import cv2
from src import gsplat_viewer as viewer

from src import engine
from src.gsplat_viewer import TrainingPreview


class TrainingPreviewTests(unittest.TestCase):
    def test_latest_snapshot_and_close(self):
        preview = TrainingPreview()
        preview.publish({'value': 1}, 1, 4)
        preview.publish({'value': 2}, 2, 4)
        data, status, error = preview.read()
        self.assertEqual(data, {'value': 2})
        self.assertIn('50.0%', status)
        self.assertIsNone(error)
        self.assertIsNone(preview.read()[0])
        preview.close()
        preview.publish({'value': 3}, 3, 4)
        self.assertIsNone(preview.read()[0])

    def test_completion_and_error(self):
        preview = TrainingPreview()
        preview.finish()
        self.assertIn('100.0% - model saved', preview.read()[1])
        error = RuntimeError('training failed')
        preview.finish(error)
        self.assertIs(preview.read()[2], error)

    def test_viewer_opens_before_snapshot_and_updates(self):
        preview = TrainingPreview()
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu', *args[1:])
            return original_to(tensor, *args, **kwargs)
        ticks = 0
        def wait(_delay):
            nonlocal ticks
            ticks += 1
            if ticks == 1:
                preview.publish({
                    'means': torch.zeros(4, 3), 'quats': torch.ones(4, 4),
                    'scales': torch.ones(4, 3), 'opacities': torch.ones(4),
                    'colors': torch.ones(4, 3),
                }, 1, 2)
            if ticks == 3:
                preview.finish()
            return ord('q') if ticks == 5 else -1
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.Tensor, 'to', cpu_to), \
             patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
             patch.object(cv2, 'setMouseCallback'), \
             patch.object(cv2, 'imshow') as show, \
             patch.object(cv2, 'getWindowProperty', return_value=1), \
             patch.object(cv2, 'waitKey', side_effect=wait), patch.object(cv2, 'destroyWindow'), \
             patch.object(viewer, 'rasterization', return_value=(torch.zeros(1, 64, 128, 3), None, None)) as raster:
            viewer.start_viewer(None, 128, 64, preview=preview)
        self.assertEqual(raster.call_count, 1)
        self.assertEqual(show.call_count, 3)  # preparation, preview, saved status
        self.assertTrue(preview.closed)

    def test_training_progress_and_renderable_snapshot(self):
        data = {
            'means': torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            'colors': torch.full((4, 3), 0.5),
            'viewmats': torch.eye(4)[None],
            'Ks': torch.eye(3)[None],
        }
        def render(means, quats, scales, opacities, colors, *args, **kwargs):
            value = sum(t.mean() for t in (means, quats, scales, opacities, colors))
            return value.expand(1, 2, 2, 3), None, None
        preview = TrainingPreview()
        output = io.StringIO()
        with patch.object(engine, '_load_targets_parallel', return_value=torch.zeros((1, 2, 2, 3), dtype=torch.uint8)), \
             patch.object(torch.Tensor, 'pin_memory', lambda self: self), \
             patch.object(engine, 'rasterization', side_effect=render), \
             contextlib.redirect_stdout(output):
            result = engine.train_splats(data, [], 2, 2, 3, 'cpu', 1,
                                         voxel_guided=False, mixed_precision=False, preview=preview)
        snapshot, status, _ = preview.read()
        self.assertIn('100.0%', status)
        self.assertIn('(3/3)', output.getvalue())
        for tensor in snapshot.values():
            self.assertFalse(tensor.requires_grad)
            self.assertEqual(tensor.device.type, 'cpu')
        torch.testing.assert_close(snapshot['means'], result['means'])
        torch.testing.assert_close(snapshot['scales'], result['scales'].exp())
        torch.testing.assert_close(snapshot['opacities'], result['opacities'].sigmoid())
        torch.testing.assert_close(snapshot['quats'].norm(dim=-1), torch.ones(4))


if __name__ == '__main__':
    unittest.main()
