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
            viewer.start_viewer(None, 128, 64, preview=preview, desktop=True)
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


class RenderSizeTests(unittest.TestCase):
    def test_render_size_fits_the_cap_and_keeps_the_aspect(self):
        self.assertEqual(viewer.render_size(1920, 1080), (1920, 1080))
        self.assertEqual(viewer.render_size(3840, 2160), (2560, 1440))
        self.assertEqual(viewer.render_size(1179, 2556), (1180, 2556))  # fits: only rounded to even
        self.assertEqual(viewer.render_size(1440, 3120), (1182, 2560))
        self.assertEqual(viewer.render_size(10, 10), (64, 64))

    def test_browser_frames_follow_the_page_size_and_drop_resolution_only_while_moving(self):
        from queue import Queue
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu', *args[1:])
            return original_to(tensor, *args, **kwargs)
        frames = []

        class FakeWeb:
            def __init__(self, *args, **kwargs):
                self.commands = Queue()
                self.shutdown_requested = False
                self.commands.put(('resize', 1000., 600.))

            def publish(self, frame):
                frames.append(frame.shape[:2])
                if len(frames) == 1:
                    self.commands.put(('orbit', 5., 0.))  # a drag: the next frame is interactive
                elif len(frames) == 2:
                    self.clock[0] += 1.0  # the drag has stopped: expect one full-resolution frame
                else:
                    self.shutdown_requested = True

            def close(self):
                pass
        clock = [100.0]
        FakeWeb.clock = clock
        def render(means, quats, scales, opacities, colors, viewmat, K, width, height, **kwargs):
            return torch.zeros(1, height, width, 3), None, None
        preview = TrainingPreview()
        preview.publish({'means': torch.zeros(4, 3), 'quats': torch.ones(4, 4), 'scales': torch.ones(4, 3),
                         'opacities': torch.ones(4), 'colors': torch.ones(4, 3)}, 1, 2)
        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.Tensor, 'to', cpu_to), \
             patch.object(viewer, 'ViewerHTTP', FakeWeb), patch.object(viewer, 'sleep'), \
             patch.object(viewer, 'monotonic', lambda: clock[0]), \
             patch.object(viewer, 'rasterization', side_effect=render):
            viewer.start_viewer(None, 461, 829, preview=preview)
        self.assertEqual(frames, [(600, 1000), (300, 500), (600, 1000)])
