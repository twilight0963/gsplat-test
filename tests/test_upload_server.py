import io
import json
import unittest
import uuid
from threading import Thread
from unittest.mock import Mock
from urllib.request import Request, urlopen
from src.upload_server import Jobs, parameters, make_server


class UploadTests(unittest.TestCase):
    def test_parameters_and_paths(self):
        p = parameters('steps=45&every=2&max-width=720&brightness=0&contrast=1.2&sharpness=0.8&output=runs/test-'+uuid.uuid4().hex)
        self.assertEqual(p['steps'], '45')
        self.assertEqual(p['brightness'], '0.0')
        for query in ('steps=0', 'every=-1', 'sharpness=NaN', 'contrast=inf', 'output=../outside', 'output=runs'):
            with self.assertRaises(ValueError):
                parameters(query)

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

    def test_http_upload_stream_and_launch(self):
        jobs=Jobs()
        jobs.launch=Mock()
        server=make_server('127.0.0.1',0,jobs)
        thread=Thread(target=server.serve_forever);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base,timeout=2) as response:
                self.assertIn(b'Training steps',response.read())
            request=Request(base+'/upload?steps=7&output=runs/test-'+uuid.uuid4().hex,
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
