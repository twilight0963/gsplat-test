import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from src import upload_server
from src.upload_server import Jobs, parameters, make_server


def setUpModule():
    # Jobs.reserve() clears the model downloads; keep tests away from the real runs/.downloads.
    downloads = tempfile.TemporaryDirectory()
    patcher = patch.object(upload_server, 'DOWNLOADS_DIR', Path(downloads.name))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    unittest.addModuleCleanup(downloads.cleanup)


class UploadTests(unittest.TestCase):
    def test_parameters_and_paths(self):
        p = parameters('steps=45&every=2&max-width=720&brightness=0&contrast=1.2&sharpness=0.8&output=test-'+uuid.uuid4().hex)
        self.assertEqual(p['steps'], '45')
        self.assertEqual(p['brightness'], '0.0')
        for query in ('steps=0', 'every=-1', 'sharpness=NaN', 'contrast=inf', 'output=../outside', 'output=runs/x', 'output=.viewer', 'output=a/b', 'output=-x', 'output=a%00b'):
            with self.assertRaises(ValueError):
                parameters(query)

    def test_parameters_enable_persistent_viewer_flag(self):
        p = parameters('output=test-' + uuid.uuid4().hex)
        self.assertIn('use-server', p)
        self.assertEqual(p['brightness'], '0.0')  # additive offset: 0 leaves pixels unchanged
        self.assertIsNone(p['use-server'])

    def test_jobs_launch_passes_boolean_flags_without_values(self):
        jobs = Jobs()
        process = Mock()
        with patch('src.upload_server.subprocess.Popen', return_value=process) as popen, \
             patch('src.upload_server.Thread'):
            jobs.launch(Path('/tmp/input.video'), {'output': '/tmp/out', 'use-server': None})
        command = popen.call_args.args[0]
        self.assertIn('--use-server', command)
        self.assertNotIn(None, command)

    def test_status_and_carriage_returns(self):
        jobs=Jobs()
        jobs.reserve()
        with self.assertRaises(ValueError): jobs.reserve()
        jobs.process=Mock(stdout=io.StringIO('Extracting\nTraining: 10%\rTraining: 20%\rViewer ready: http://localhost:8000/\nModel saved: runs/x/model.glb\n'))
        jobs.process.wait.return_value=0
        jobs.monitor()
        self.assertEqual(jobs.snapshot()['status'], 'Model saved: runs/x/model.glb')
        self.assertTrue(jobs.snapshot()['saved'])
        self.assertFalse(jobs.snapshot()['busy'])
        self.assertTrue(jobs.snapshot()['viewer'])
        jobs.process.poll.return_value = 0
        jobs.reserve()
        self.assertTrue(jobs.snapshot()['busy'])
        self.assertFalse(jobs.snapshot()['saved'])

    def test_http_upload_stream_and_launch(self):
        jobs=Jobs()
        jobs.launch=Mock()
        server=make_server('127.0.0.1',0,jobs)
        thread=Thread(target=server.serve_forever);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base,timeout=2) as response:
                self.assertIn(b'Training steps',response.read())
            request=Request(base+'/upload?steps=7&output=test-'+uuid.uuid4().hex,
                            data=b'fake video data',headers={'Content-Type':'application/octet-stream'})
            with urlopen(request,timeout=2) as response:self.assertEqual(response.status,202)
            path,params=jobs.launch.call_args.args
            self.assertEqual(path.read_bytes(),b'fake video data')
            self.assertEqual(params['steps'],'7')
            path.unlink()
            with urlopen(base+'/status',timeout=2) as response:
                self.assertTrue(json.load(response)['busy'])
        finally:
            server.shutdown();server.server_close();thread.join()


class CustomArgumentsTests(unittest.TestCase):
    def output(self):
        return 'output=test-' + uuid.uuid4().hex

    def test_custom_parser_takes_no_paths_or_server_options(self):
        from src.upload_server import SERVER_OPTIONS, custom_parser
        parser = custom_parser()
        flags = {flag for action in parser._actions for flag in action.option_strings}
        self.assertFalse(flags & set(SERVER_OPTIONS))
        for action in parser._actions:
            if action.option_strings:
                # Only numbers, listed choices and flags: nothing that names a file.
                self.assertTrue(action.type in (int, float) or action.choices or action.nargs == 0, action.dest)

    def test_custom_arguments_are_rebuilt_from_typed_values(self):
        from src.upload_server import custom_arguments
        self.assertEqual(custom_arguments('--steps 5000 --every 5 --max-width 1920 --brightness 0 '
                                          '--contrast 1 --sharpness 0.5'), {})  # the engine defaults
        self.assertEqual(custom_arguments('--steps=7000 --brightness -12 --no-unsharp --device cpu --no-colmap-cache'),
                         {'steps': '7000', 'brightness': '-12.0', 'no-unsharp': None, 'device': 'cpu',
                          'no-colmap-cache': None})

    def test_custom_arguments_reject_anything_else(self):
        from src.upload_server import custom_arguments
        for text in ('--swin x', '--swinir-root=/tmp', '--sr-checkpoint /x', '--vocab-tree /x', '--output runs/x',
                     '--use-server', '--headless', '--step 5', '--help', '-h', 'input.mp4', '-- /etc/passwd',
                     '$(id)', '--steps 5; id', '--steps nan', '--brightness inf', '--densify other', '"unclosed',
                     '--steps', 'x' * 5000):
            with self.subTest(text=text), self.assertRaises(ValueError):
                custom_arguments(text)

    def test_parameters_with_custom_arguments(self):
        p = parameters('args=' + quote('--steps 7000 --max-width 1280') + '&' + self.output())
        self.assertEqual((p['steps'], p['max-width']), ('7000', '1280'))
        self.assertIsNone(p['use-server'])
        self.assertTrue(p['output'].startswith(str(Path(__file__).resolve().parent.parent / 'runs')))
        for query in ('args=' + quote('--steps 0'), 'args=' + quote('--sharpness 2'),
                      'args=' + quote('--steps 10') + '&steps=10', 'args=' + quote('--sr-tile 100 --super-resolution')):
            with self.subTest(query=query), self.assertRaises(ValueError):
                parameters(query + '&' + self.output())

    def test_custom_super_resolution_uses_the_server_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, checkpoint = Path(tmp) / 'SwinIR', Path(tmp) / 'x2.pth'
            with self.assertRaises(ValueError):
                parameters('args=--super-resolution&' + self.output(), root, checkpoint)
            (root / 'models').mkdir(parents=True)
            (root / 'models/network_swinir.py').write_text('')
            checkpoint.write_bytes(b'')
            p = parameters('args=--super-resolution&' + self.output(), root, checkpoint)
        self.assertEqual(p['swinir-root'], str(root.resolve()))
        self.assertEqual(p['sr-checkpoint'], str(checkpoint.resolve()))
        self.assertIn('super-resolution', p)

    def test_page_help_and_check_endpoint(self):
        jobs = Jobs()
        jobs.launch = Mock()
        server = make_server('127.0.0.1', 0, jobs, swinir_root='/nonexistent', sr_checkpoint='/nonexistent')
        thread = Thread(target=server.serve_forever); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base, timeout=2) as response:
                page = response.read().decode()
            self.assertNotIn('<!--engine-help-->', page)
            self.assertIn('usage: python -m src.engine', page)
            self.assertIn('data-sr-available="false"', page)
            self.assertNotIn('--swinir-root SWINIR_ROOT', page)
            with urlopen(base + '/check?args=' + quote('--steps 2000'), timeout=2) as response:
                self.assertEqual(response.status, 200)
            with self.assertRaises(HTTPError) as error:
                urlopen(base + '/check?args=' + quote('--swinir-root /tmp'), timeout=2)
            self.assertEqual(error.exception.code, 400)
            self.assertIn('--swinir-root', json.load(error.exception)['error'])
            jobs.launch.assert_not_called()
            self.assertFalse(jobs.snapshot()['busy'])
        finally:
            server.shutdown(); server.server_close(); thread.join()


class DownloadAndOpenModelTests(unittest.TestCase):
    def serve(self, jobs):
        server = make_server('127.0.0.1', 0, jobs)
        thread = Thread(target=server.serve_forever); thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        return f'http://127.0.0.1:{server.server_port}'

    def test_only_the_current_download_is_served(self):
        (upload_server.DOWNLOADS_DIR / 'run-a.zip').write_bytes(b'zip bytes')
        (upload_server.DOWNLOADS_DIR / 'other.zip').write_bytes(b'other')
        jobs = Jobs()
        jobs.state['download'] = 'run-a.zip'
        base = self.serve(jobs)
        with urlopen(base + '/download/run-a.zip', timeout=2) as response:
            self.assertEqual(response.read(), b'zip bytes')
            self.assertEqual(response.headers['Content-Disposition'], 'attachment; filename="run-a.zip"')
        for path in ('/download/other.zip', '/download/../runs/x', '/download/', '/download/%2e%2e%2fsrc%2fengine.py'):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(base + path, timeout=2)
            self.assertEqual(error.exception.code, 404)

    def test_open_model_upload_launches_the_viewer_job(self):
        jobs = Jobs()
        jobs.launch_open = Mock()
        jobs.state['download'] = 'kept.zip'
        base = self.serve(jobs)
        request = Request(base + '/open-model?name=' + quote('my model.glb'), data=b'glTF\x02\x00\x00\x00rest',
                          headers={'Content-Type': 'model/gltf-binary'})
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 202)
        model, name = jobs.launch_open.call_args.args
        self.addCleanup(model.unlink, missing_ok=True)
        self.assertEqual((name, model.read_bytes()[:4], model.parent.name), ('my model.glb', b'glTF', '.uploads'))
        self.assertEqual(jobs.snapshot()['download'], 'kept.zip')  # opening a model keeps the download
        with self.assertRaises(HTTPError) as error:  # one job at a time
            urlopen(Request(base + '/open-model', data=b'glTF', headers={'Content-Type': 'model/gltf-binary'}), timeout=2)
        self.assertIn('already running', json.load(error.exception)['error'])

    def test_open_model_rejects_other_files(self):
        jobs = Jobs()
        jobs.launch_open = Mock()
        base = self.serve(jobs)
        before = set((upload_server.ROOT / 'runs' / '.uploads').glob('*.glb'))
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base + '/open-model', data=b'PK\x03\x04zip', headers={'Content-Type': 'model/gltf-binary'}), timeout=2)
        self.assertIn('not a .glb', json.load(error.exception)['error'])
        self.assertEqual(set((upload_server.ROOT / 'runs' / '.uploads').glob('*.glb')), before)
        self.assertFalse(jobs.snapshot()['busy'])
        jobs.launch_open.assert_not_called()

    def test_open_job_finishes_and_removes_its_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            for code, stage in ((0, 'opened'), (2, 'failed')):
                with self.subTest(code=code):
                    model = Path(tmp) / 'upload.glb'
                    model.write_bytes(b'glTF')
                    jobs = Jobs()
                    jobs.reserve(keep_download=True)
                    jobs.kind, jobs.model = 'open', model
                    jobs.process = Mock(stdout=io.StringIO('Viewer ready: http://localhost:8000/\nModel opened: x.glb (3 splats)\n'))
                    jobs.process.wait.return_value = code
                    jobs.monitor()
                    state = jobs.snapshot()
                    self.assertEqual((state['stage'], state['busy'], state['viewer']), (stage, False, True))
                    self.assertFalse(model.exists())

    def test_launch_open_passes_the_name_as_one_option(self):
        jobs = Jobs()
        with patch('src.upload_server.subprocess.Popen') as popen, patch('src.upload_server.Thread'):
            jobs.launch_open(Path('/tmp/m.glb'), '--headless.glb')
        self.assertEqual(popen.call_args.args[0][-3:], ['src.open_model', '/tmp/m.glb', '--name=--headless.glb'])
        with self.assertRaisesRegex(Exception, 'No job is running'):
            jobs.stop('discard')  # not busy: nothing reserved in this test


class OpenModelScriptTests(unittest.TestCase):
    def test_publishes_the_model_to_the_viewer_as_a_finished_run(self):
        import numpy as np
        from src import open_model
        from src.gltf_gsplat import write_gsplat_glb
        from src.live_viewer import FilePreview
        from src.training_control import read_state, training_active
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'upload.glb'
            write_gsplat_glb(path, np.zeros((3, 3), np.float32), np.full((3, 3), .1, np.float32),
                             np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1)), np.full(3, .5, np.float32),
                             np.full((3, 3), .5, np.float32))
            viewer = Path(tmp) / 'viewer'
            with patch.object(open_model, 'VIEWER_DIR', viewer), patch.object(open_model, 'ensure_viewer') as ensure, \
                 patch('sys.argv', ['open_model', str(path), '--name=site.glb']), redirect_stdout(io.StringIO()) as out:
                open_model.main()
            ensure.assert_called_once()
            state = read_state(viewer)
            self.assertFalse(training_active(state))
            self.assertEqual(state['status'], 'Saved model: site.glb')
            data, _status, _error = FilePreview(viewer).read()
            self.assertEqual(tuple(data['means'].shape), (3, 3))
            self.assertIn('Model opened: site.glb (3 splats)', out.getvalue())
            path.write_bytes(b'glTF not really')
            with patch('sys.argv', ['open_model', str(path), '--name=bad.glb']), redirect_stdout(io.StringIO()) as out, \
                 self.assertRaises(SystemExit) as exit_info:
                open_model.main()
            self.assertEqual(exit_info.exception.code, 2)
            self.assertIn('bad.glb is not a Gaussian splat model', out.getvalue())
            self.assertNotIn(str(path), out.getvalue())
