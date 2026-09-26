import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch

from src import engine, upload_server
from src.live_viewer import PreviewPublisher
from src.training_control import read_state, request_stop, training_active
from src.upload_server import Jobs, StopRejected, make_server
from src.viewer_http import ViewerHTTP, code_version


class StopChannelTests(unittest.TestCase):
    def test_requests_reach_only_the_active_session_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(request_stop(tmp, 'save'), (False, 'No training is running.'))
            first = PreviewPublisher(tmp)
            self.assertTrue(training_active(read_state(tmp)))
            accepted, _ = request_stop(tmp, 'save')
            self.assertTrue(accepted)
            self.assertEqual(first.stop_requested(), 'save')
            self.assertIsNone(first.stop_requested())  # honoured once
            second = PreviewPublisher(tmp)  # a later run in the same directory
            self.assertIsNone(second.stop_requested())  # the old request named the first session
            request_stop(tmp, 'discard')
            self.assertIsNone(first.stop_requested())
            self.assertEqual(second.stop_requested(), 'discard')
            with self.assertRaises(ValueError):
                request_stop(tmp, 'pause')

    def test_finished_or_dead_runs_cannot_be_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = PreviewPublisher(tmp)
            publisher.finish(message='Model saved')
            self.assertFalse(training_active(read_state(tmp)))
            self.assertFalse(request_stop(tmp, 'save')[0])
            dead = subprocess.Popen([sys.executable, '-c', 'pass'])
            dead.wait()
            state = read_state(tmp)
            state.update(active=True, pid=dead.pid)
            (Path(tmp) / 'state.json').write_text(json.dumps(state))
            self.assertFalse(training_active(read_state(tmp)))  # crashed runs do not look active


def fake_render(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kwargs):
    value = sum(t.mean() for t in (means, quats, scales, opacities, colors))
    return value.expand(len(viewmats), height, width, 3), None, {}


class TrainingStopTests(unittest.TestCase):
    def train(self, actions):
        preview = Mock(closed=False)
        preview.stop_requested.side_effect = actions + [None] * 100
        data = {'means': torch.rand(8, 3), 'colors': torch.rand(8, 3),
                'viewmats': torch.eye(4)[None], 'Ks': torch.eye(3)[None]}
        with patch.object(engine, 'rasterization', side_effect=fake_render):
            result = engine.train_splats(data, [], 4, 3, 10, 'cpu', 1, targets=torch.zeros(1, 3, 4, 3, dtype=torch.uint8),
                                         mixed_precision=False, voxel_guided=False, ssim_weight=0, preview=preview)
        return result, preview

    def test_stop_and_save_ends_early_and_keeps_the_model(self):
        result, preview = self.train([None, None, None, 'save'])
        self.assertEqual(int(result['steps_completed']), 3)
        self.assertEqual(preview.publish.call_args.args[1:], (3, 10))
        self.assertTrue(torch.isfinite(result['means']).all())

    def test_full_run_reports_all_steps_and_discard_raises(self):
        result, _ = self.train([])
        self.assertEqual(int(result['steps_completed']), 10)
        with self.assertRaises(engine.TrainingStopped):
            self.train([None, 'discard'])

    def test_engine_marks_discarded_runs_and_exits_with_the_stopped_code(self):
        self.assertEqual(engine.STOPPED_EXIT_CODE, upload_server.STOPPED_EXIT_CODE)
        with patch.object(engine, 'build_model', side_effect=engine.TrainingStopped('stopped at step 4/10')), \
             patch.object(sys, 'argv', ['engine', 'input.mp4']), self.assertRaises(SystemExit) as exit_info:
            engine.main()
        self.assertEqual(exit_info.exception.code, engine.STOPPED_EXIT_CODE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'original').mkdir()
            for name in ('frame_000000.jpg', 'frame_000001.jpg'):
                (root / 'original' / name).write_bytes(b'x')
            data = {'viewmats': torch.eye(4)[None], 'Ks': torch.eye(3)[None]}
            with patch.object(engine, 'extract_frames', return_value=root / 'original'), \
                 patch.object(engine, 'select_keyframes', return_value=root / 'original'), \
                 patch.object(engine, 'run_colmap'), \
                 patch.object(engine, 'load_reconstruction', return_value=(data, [root / 'a.png'], 4, 3)), \
                 patch.object(engine, 'build_targets_gpu', return_value=torch.zeros(1, 3, 4, 3, dtype=torch.uint8)), \
                 patch.object(engine, 'ensure_viewer'), patch.object(engine, 'PreviewPublisher') as publisher, \
                 patch.object(engine, 'train_splats', side_effect=engine.TrainingStopped('x')), \
                 self.assertRaises(engine.TrainingStopped):
                engine.build_model(root / 'v.mp4', root / 'out', use_server=True, device='cpu')
            publisher.return_value.finish.assert_called_once_with(message='Stopped - nothing was saved')
            self.assertFalse((root / 'out' / 'model.glb').exists())


class ViewerServerTests(unittest.TestCase):
    def serve(self, directory):
        server = ViewerHTTP('127.0.0.1', 0, preview_directory=directory)
        self.addCleanup(server.close)
        return server, f'http://127.0.0.1:{server.server.server_port}'

    def post(self, url, body, kind='application/json'):
        request = Request(url, data=body, headers={'Content-Type': kind})
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_status_and_stop_while_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, base = self.serve(tmp)
            publisher = PreviewPublisher(tmp)
            publisher.publish({'means': torch.zeros(2, 3)}, 250, 1000)
            with urlopen(base + '/status', timeout=2) as response:
                status = json.load(response)
            self.assertEqual((status['training'], status['completed'], status['total']), (True, 250, 1000))
            self.assertEqual(self.post(base + '/stop', b'{"action":"pause"}')[0], 400)
            self.assertEqual(self.post(base + '/stop', b'not json')[0], 400)
            code, body = self.post(base + '/stop', b'{"action":"save"}')
            self.assertEqual(code, 202)
            self.assertIn('saved', json.loads(body)['message'])
            self.assertEqual(publisher.stop_requested(), 'save')
            publisher.finish(message='Model saved - stopped early at step 250/1000')
            code, body = self.post(base + '/stop', b'{"action":"discard"}')
            self.assertEqual(code, 409)
            with urlopen(base + '/status', timeout=2) as response:
                status = json.load(response)
            self.assertFalse(status['training'])
            self.assertIn('stopped early', status['status'])

    def test_saved_model_viewer_has_no_training(self):
        server, base = self.serve(None)
        with urlopen(base + '/status', timeout=2) as response:
            self.assertFalse(json.load(response)['training'])
        self.assertEqual(self.post(base + '/stop', b'{"action":"save"}')[0], 409)

    def test_health_version_frame_revision_and_local_shutdown(self):
        server, base = self.serve(None)
        with urlopen(base + '/health', timeout=2) as response:
            self.assertEqual(json.load(response)['version'], code_version())
        server.publish(np.zeros((4, 4, 3), np.uint8))
        with urlopen(base + '/frame.jpg', timeout=2) as response:
            first = response.headers['X-Frame-Revision']
        server.publish(np.zeros((4, 4, 3), np.uint8))
        with urlopen(base + '/frame.jpg', timeout=2) as response:
            self.assertEqual(int(response.headers['X-Frame-Revision']), int(first) + 1)
        self.assertFalse(server.shutdown_requested)
        self.assertEqual(self.post(base + '/shutdown', b'')[0], 204)
        self.assertTrue(server.shutdown_requested)


class UploadStopTests(unittest.TestCase):
    def job_paths(self):
        runs = upload_server.ROOT / 'runs'
        (runs / '.uploads').mkdir(parents=True, exist_ok=True)
        upload = runs / '.uploads' / (uuid.uuid4().hex + '.video')
        output = runs / f'test-stop-{uuid.uuid4().hex}'
        upload.write_bytes(b'video')
        (output / 'colmap').mkdir(parents=True)
        (output / 'colmap' / 'database.db').write_bytes(b'db')
        self.addCleanup(lambda: upload.unlink(missing_ok=True))
        self.addCleanup(shutil.rmtree, output, True)
        return upload, output

    @property
    def job_output(self):
        # Created on first use so a subprocess script can be told where the job's output is.
        if not hasattr(self, '_paths'):
            self._paths = self.job_paths()
        return self._paths[1]

    def start(self, jobs, script, stage):
        upload, output = self._paths if hasattr(self, '_paths') else self.job_paths()
        jobs.reserve()
        jobs.job = (upload, output)
        jobs.process = subprocess.Popen([sys.executable, '-u', '-c', script], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        with jobs.lock:
            jobs.state['stage'] = stage
        monitor = Thread(target=jobs.monitor)
        monitor.start()
        return upload, output, monitor

    def test_discard_before_training_stops_the_engine_and_removes_the_job(self):
        jobs = Jobs()
        upload, output, monitor = self.start(jobs, 'import time; print("Extracting video frames...", flush=True); time.sleep(60)', 'reconstruct')
        with self.assertRaisesRegex(StopRejected, 'no model to save'):
            jobs.stop('save')
        self.assertIn('nothing will be saved', jobs.stop('discard'))
        with self.assertRaisesRegex(StopRejected, 'Already stopping'):
            jobs.stop('discard')
        monitor.join(timeout=10)
        state = jobs.snapshot()
        self.assertEqual((state['stage'], state['busy'], state['stopping']), ('stopped', False, None))
        self.assertIn('Nothing was saved', state['status'])
        self.assertFalse(upload.exists())
        self.assertFalse(output.exists())

    def test_stop_during_training_asks_the_engine(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(upload_server, 'VIEWER_DIR', Path(tmp)):
            publisher = PreviewPublisher(tmp)  # the engine's publisher (this process is alive)
            jobs = Jobs()
            upload, output, monitor = self.start(jobs, 'import time; time.sleep(60)', 'train')
            self.assertIn('will be saved', jobs.stop('save'))
            self.assertEqual(publisher.stop_requested(), 'save')
            self.assertTrue(jobs.process.poll() is None)  # the engine stops itself after the step
            jobs.process.terminate()
            monitor.join(timeout=10)
            self.assertTrue(upload.exists())  # a failed/unsaved save-stop keeps files for a retry

    def test_engine_exit_code_three_means_discarded(self):
        jobs = Jobs()
        upload, output, monitor = self.start(jobs, 'import sys; print("Training splats: 40.0% (40/100)"); sys.exit(3)', 'train')
        monitor.join(timeout=10)
        state = jobs.snapshot()
        self.assertEqual(state['stage'], 'stopped')
        self.assertFalse(upload.exists())
        self.assertFalse(output.exists())

    def test_stop_sent_from_the_viewer_is_reported_as_stopped_early(self):
        jobs = Jobs()
        upload, output, monitor = self.start(jobs, (
            'import pathlib, sys\n'
            'print("Stop requested at step 5/10: saving the model trained so far.", flush=True)\n'
            f'pathlib.Path({str(self.job_output)!r}).joinpath("model.glb").write_bytes(b"glb")\n'), 'train')
        monitor.join(timeout=10)
        state = jobs.snapshot()
        self.assertEqual(state['stage'], 'done')
        self.assertIn('Model saved (stopped early)', state['status'])
        self.assertTrue((output / 'model.glb').exists())
        self.assertFalse(upload.exists())

    def test_rejected_stops(self):
        jobs = Jobs()
        with self.assertRaisesRegex(StopRejected, 'No job is running'):
            jobs.stop('discard')
        jobs.reserve()
        with self.assertRaisesRegex(StopRejected, 'upload is still in progress'):
            jobs.stop('discard')
        with self.assertRaises(ValueError):
            jobs.stop('pause')

    def test_stages_and_training_progress_follow_engine_output(self):
        jobs = Jobs()
        jobs.reserve()
        jobs.process = Mock(stdout=io.StringIO(
            'Extracting video frames...\n[timing] colmap feature_extractor: 2.0s\nTraining splats...\n'
            'Training splats:  12.5% (625/5000)\rTraining splats:  50.0% (2500/5000)\n'))
        jobs.process.wait.return_value = 1
        seen = []
        original = jobs._update_stage
        def record(line):
            original(line)
            seen.append(jobs.state['stage'])
        jobs._update_stage = record
        jobs.monitor()
        self.assertEqual(seen, ['prepare', 'reconstruct', 'train', 'train', 'train'])
        state = jobs.snapshot()
        self.assertEqual((state['progress'], state['step'], state['steps']), (50.0, 2500, 5000))
        self.assertEqual(state['stage'], 'failed')

    def test_http_stop_endpoint(self):
        jobs = Jobs()
        server = make_server('127.0.0.1', 0, jobs)
        thread = Thread(target=server.serve_forever)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            def post(body, kind='application/json'):
                try:
                    with urlopen(Request(base + '/stop', data=body, headers={'Content-Type': kind}), timeout=2) as response:
                        return response.status, json.load(response)
                except HTTPError as error:
                    return error.code, json.load(error)
            self.assertEqual(post(b'{"action":"save"}'), (409, {'error': 'No job is running.'}))
            self.assertEqual(post(b'{"action":"pause"}')[0], 400)
            self.assertEqual(post(b'nope')[0], 400)
            self.assertEqual(post(b'{"action":"save"}', 'text/plain')[0], 400)
            with urlopen(base + '/status', timeout=2) as response:
                self.assertEqual(json.load(response)['stage'], 'idle')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
