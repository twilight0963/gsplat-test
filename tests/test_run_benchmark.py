import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import engine, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_nested_stages_do_not_double_count(self):
        run = {'seconds': dict.fromkeys(run_benchmark.STAGES, 0.0), 'stack': []}
        token = run_benchmark._active.set(run)
        try:
            with patch.object(run_benchmark, 'perf_counter', side_effect=[0., 2., 5., 10.]):
                with run_benchmark.stage('preprocessing'):
                    with run_benchmark.stage('cas'):
                        pass
            self.assertEqual(run['seconds']['cas'], 3.)
            self.assertEqual(run['seconds']['preprocessing'], 7.)
            self.assertEqual(sum(run['seconds'].values()), 10.)
        finally:
            run_benchmark._active.reset(token)

    def test_completed_video_logs_decoded_selected_and_registered_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = np.full((8, 10, 3), 100, np.uint8)
            engine.cv2.imwrite(str(root/'a.png'), frame)
            def check_colmap(capture_dir, *_):
                cas.assert_not_called()
                originals = sorted(capture_dir.glob('*.jpg'))
                self.assertEqual(len(originals), 3)
                self.assertEqual(int(engine.cv2.imread(str(originals[0]))[0, 0, 0]), 100)
            with patch.object(engine.cv2, 'VideoCapture') as capture, \
                 patch.object(engine, 'apply_cas', side_effect=lambda images, *_: images) as cas, \
                 patch.object(engine, 'run_colmap', side_effect=check_colmap), \
                 patch.object(engine, 'load_reconstruction', return_value=({}, [root/'a.png'], 10, 8)), \
                 patch.object(engine, 'train_splats', return_value={}), \
                 patch.object(engine, 'export_gltf'), \
                 patch.object(engine, 'PreviewPublisher'), patch.object(engine, 'ensure_viewer'):
                capture.return_value.isOpened.return_value = True
                capture.return_value.get.return_value = 30
                capture.return_value.read.side_effect = [(True, frame)] * 5 + [(False, None)]
                engine.build_model(root/'input.mp4', root/'out', every=2, use_server=True,
                                   brightness=10, contrast=2, unsharp=False)
                cas.assert_called_once()
                targets = cas.call_args.args[0]
                self.assertEqual(len(targets), 1)
                self.assertEqual(int(engine.cv2.imread(str(targets[0]))[0, 0, 0]), 210)
            report = json.loads((root/'out/benchmark.json').read_text())
            self.assertEqual(report['frame_counts'], {'decoded_video_frames': 5,
                             'frames_before_colmap': 3, 'frames_after_colmap': 1})
            self.assertEqual(report['status'], 'completed')
            self.assertGreaterEqual(report['total_seconds'], sum(report['stage_seconds'].values()))
            self.assertEqual(report['settings']['view_batch_size'], 4)
            self.assertIn('COLMAP includes', (root/'out/benchmark.log').read_text())
            self.assertIsNone(run_benchmark._active.get())

    def test_failed_run_does_not_write_completed_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(engine, 'extract_frames', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    engine.build_model(root/'input.mp4', root/'out')
            self.assertFalse((root/'out/benchmark.json').exists())
            self.assertIsNone(run_benchmark._active.get())
