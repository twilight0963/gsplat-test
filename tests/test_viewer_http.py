import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import numpy as np
from src.viewer_http import ViewerHTTP


class ViewerHTTPTests(unittest.TestCase):
    def test_page_frames_controls_and_shutdown(self):
        server = ViewerHTTP('127.0.0.1', 0)
        base = f'http://127.0.0.1:{server.server.server_port}'
        try:
            with urlopen(base + '/', timeout=2) as response:
                self.assertIn(b'/frame.jpg', response.read())
            server.publish(np.zeros((8, 8, 3), dtype=np.uint8))
            with urlopen(base + '/frame.jpg', timeout=2) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                self.assertTrue(response.read().startswith(b'\xff\xd8'))
            request = Request(base + '/control', data=b'{"action":"orbit","dx":12,"dy":3}',
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 204)
            self.assertEqual(server.commands.get_nowait(), ('orbit', 12., 3.))
            for path in ('/../src/engine.py', '/src/engine.py'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path, timeout=2)
                self.assertEqual(error.exception.code, 404)
            request = Request(base + '/control', data=b'{"action":"orbit","dx":"NaN"}',
                              headers={'Content-Type': 'application/json'})
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.close()
        self.assertFalse(server.thread.is_alive())
