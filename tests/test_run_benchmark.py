import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

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
            def check_colmap(capture_dir, *_, **kwargs):
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
                capture.return_value.grab.side_effect = [True] * 5 + [False]
                capture.return_value.retrieve.return_value = (True, frame)
                engine.build_model(root/'input.mp4', root/'out', every=2, use_server=True,
                                   brightness=10, contrast=2, unsharp=False, device='cpu')
                cas.assert_not_called()
                targets = engine.train_splats.call_args.kwargs['targets']
                self.assertEqual(len(targets), 1)
                self.assertAlmostEqual(float(targets[0, 0, 0, 0]), 210, places=5)
            report = json.loads((root/'out/benchmark.json').read_text())
            self.assertEqual(report['frame_counts'], {'decoded_video_frames': 5,
                             'frames_before_colmap': 3, 'frames_after_colmap': 1, 'keyframe_candidates': 3})
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['absolute_accuracy']['status'], 'unverified')
            self.assertTrue(report['runtime_target_met'])
            self.assertEqual(report['settings']['sequential_overlap'], 12)
            self.assertGreaterEqual(report['total_seconds'], sum(report['stage_seconds'].values()))
            self.assertEqual(report['settings']['view_batch_size'], 4)
            self.assertIn('COLMAP includes', (root/'out/benchmark.log').read_text())
            self.assertIsNone(run_benchmark._active.get())

    def test_eval_holds_out_views_and_reports_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = [root / f'{i}.png' for i in range(5)]
            data = {'viewmats': torch.eye(4).repeat(5, 1, 1), 'Ks': torch.eye(3).repeat(5, 1, 1)}
            targets = torch.arange(5, dtype=torch.uint8).view(5, 1, 1, 1).expand(5, 2, 2, 3).clone()
            with patch.object(engine, 'run_colmap'), \
                 patch.object(engine, 'extract_frames', return_value=root), \
                 patch.object(engine, 'select_keyframes', return_value=root), \
                 patch.object(engine, 'load_reconstruction', return_value=(data, images, 2, 2)), \
                 patch.object(engine, 'build_targets_gpu', return_value=targets), \
                 patch.object(engine, 'train_splats', return_value={}) as train, \
                 patch.object(engine, 'evaluate_views', return_value={'eval_views': 2, 'eval_psnr': 30.5, 'eval_ssim': .9}) as evaluate, \
                 patch.object(engine, 'export_gltf'):
                engine.build_model(root / 'input.mp4', root / 'out', headless=True, device='cpu', eval_every=4)
            trained = train.call_args
            self.assertEqual(trained.args[1], images[1:4])
            self.assertEqual(trained.kwargs['targets'][:, 0, 0, 0].tolist(), [1, 2, 3])
            self.assertEqual(evaluate.call_args.args[3][:, 0, 0, 0].tolist(), [0, 4])
            report = json.loads((root / 'out/benchmark.json').read_text())
            self.assertEqual(report['quality'], {'eval_views': 2, 'eval_psnr': 30.5, 'eval_ssim': .9})
            self.assertEqual(report['settings']['eval_every'], 4)
            self.assertIn('Quality eval_psnr: 30.5000', (root / 'out/benchmark.log').read_text())

    def test_colmap_details_do_not_double_count_parent(self):
        run = {'seconds': dict.fromkeys(run_benchmark.STAGES, 0.), 'stack': []}
        token = run_benchmark._active.set(run)
        try:
            with patch.object(run_benchmark, 'perf_counter', side_effect=[0., 2., 5., 10.]):
                with run_benchmark.stage('colmap'):
                    with run_benchmark.colmap_command('global_mapper'):
                        pass
            self.assertEqual(run['seconds']['colmap'], 10.)
            self.assertEqual(run['colmap_seconds']['global_mapper'], 3.)
            self.assertEqual(sum(run['seconds'].values()), 10.)
        finally:
            run_benchmark._active.reset(token)

    def test_failed_run_does_not_write_completed_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(engine, 'extract_frames', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    engine.build_model(root/'input.mp4', root/'out', device='cpu')
            self.assertFalse((root/'out/benchmark.json').exists())
            self.assertIsNone(run_benchmark._active.get())
