import io
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
import unittest.mock
from unittest.mock import Mock

from src import upload_server
from src.upload_server import Jobs, clean_up


class CleanupTests(unittest.TestCase):
    def make_job(self, photos=True):
        runs = upload_server.ROOT / 'runs'
        (runs / '.uploads').mkdir(parents=True, exist_ok=True)
        output = runs / f'test-cleanup-{uuid.uuid4().hex}'
        upload = runs / '.uploads' / (uuid.uuid4().hex + ('.photos' if photos else '.video'))
        self.addCleanup(shutil.rmtree, output, True)
        self.addCleanup(lambda: shutil.rmtree(upload, True) if upload.is_dir() else upload.unlink(missing_ok=True))
        if photos:
            upload.mkdir()
            (upload / 'a.jpg').write_bytes(b'x' * 1000)
        else:
            upload.write_bytes(b'v' * 1000)
        for name, size in (('capture/photos/frame_000000.jpg', 2000), ('colmap/database.db', 3000),
                           ('colmap/undistorted/images/frame_000000.jpg', 4000), ('sr/base/frame_000000.png', 500),
                           ('model.glb', 7), ('benchmark.json', 8), ('benchmark.log', 9), ('photo_metadata.json', 10)):
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            (output / name).write_bytes(b'd' * size)
        (output / 'capture_subset').mkdir()
        os.link(output / 'capture/photos/frame_000000.jpg', output / 'capture_subset/frame_000000.jpg')
        return upload, output

    def run_monitor(self, upload, output, code):
        jobs = Jobs()
        jobs.reserve()
        jobs.job = (upload, output)
        jobs.process = Mock(stdout=io.StringIO(f'Model saved: {output / "model.glb"}\n'))
        jobs.process.wait.return_value = code
        jobs.monitor()
        return jobs.snapshot()

    def test_success_removes_intermediates_and_keeps_model_and_reports(self):
        for photos in (True, False):
            with self.subTest(photos=photos):
                upload, output = self.make_job(photos)
                state = self.run_monitor(upload, output, 0)
                self.assertFalse(upload.exists())
                self.assertEqual(sorted(p.name for p in output.iterdir()),
                                 ['benchmark.json', 'benchmark.log', 'model.glb', 'photo_metadata.json'])
                self.assertFalse(state['busy'])
                self.assertTrue(state['saved'])
                self.assertIn('Removed temporary files', state['status'])

    def test_freed_size_counts_hard_links_once(self):
        upload, output = self.make_job()
        self.assertEqual(clean_up(upload, output), 1000 + 2000 + 3000 + 4000 + 500)

    def test_failed_or_modelless_jobs_keep_everything(self):
        upload, output = self.make_job()
        state = self.run_monitor(upload, output, 1)
        self.assertTrue(state['status'].startswith('Failed (exit 1)'))
        (output / 'model.glb').unlink()
        self.run_monitor(upload, output, 0)
        self.assertTrue((upload / 'a.jpg').exists())
        self.assertTrue((output / 'colmap' / 'database.db').exists())
        self.assertTrue((output / 'capture_subset').exists())

    def test_launch_records_the_job_for_cleanup(self):
        jobs = Jobs()
        with unittest.mock.patch('src.upload_server.subprocess.Popen'), unittest.mock.patch('src.upload_server.Thread'):
            jobs.launch(Path('/tmp/in.video'), {'output': '/tmp/out'})
        self.assertEqual(jobs.job, (Path('/tmp/in.video'), Path('/tmp/out')))

    def test_refuses_paths_outside_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / 'capture'
            outside.mkdir()
            upload, output = self.make_job()
            for args in ((Path(tmp) / 'x.video', output), (upload, Path(tmp)), (upload, upload_server.ROOT / 'runs')):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    clean_up(*args)
            self.assertTrue(outside.exists())
            self.assertTrue(upload.exists())
