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


class ViewerResizeTests(unittest.TestCase):
    def post(self, base, body):
        request = Request(base + '/control', data=body, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=2) as response:
                return response.status
        except HTTPError as error:
            return error.code

    def test_resize_accepts_page_sizes_only(self):
        server = ViewerHTTP('127.0.0.1', 0)
        base = f'http://127.0.0.1:{server.server.server_port}'
        try:
            self.assertEqual(self.post(base, b'{"action":"resize","dx":2560,"dy":1440}'), 204)
            self.assertEqual(server.commands.get_nowait(), ('resize', 2560., 1440.))
            for body in (b'{"action":"resize","dx":10,"dy":10}', b'{"action":"resize","dx":9000,"dy":100}',
                         b'{"action":"resize","dx":"inf","dy":100}', b'{"action":"orbit","dx":2000,"dy":0}'):
                with self.subTest(body=body):
                    self.assertEqual(self.post(base, body), 400)
        finally:
            server.close()

    def test_jpeg_keeps_full_colour_resolution(self):
        server = ViewerHTTP('127.0.0.1', 0)
        try:
            server.publish(np.zeros((16, 16, 3), dtype=np.uint8))
            frame = server.frame
            start = frame.index(b'\xff\xc0')  # baseline start-of-frame: per-component sampling factors
            components = frame[start + 9]
            factors = [frame[start + 11 + 3 * i] for i in range(components)]
            self.assertEqual(factors, [0x11] * components)  # 1x1 for Y, Cb and Cr: 4:4:4
        finally:
            server.close()
