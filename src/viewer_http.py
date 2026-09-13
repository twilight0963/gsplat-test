"""Small HTTP transport for the GPU viewer; never serves workspace files."""
import json
import math
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue, Full
from threading import Lock, Thread

import cv2


class ViewerHTTP:
    def __init__(self, host='0.0.0.0', port=8000, size_clamp_multiplier=1.0, preview_directory=None):
        self.commands = Queue(maxsize=256)
        self.lock = Lock()
        self.frame = None
        page = Path(__file__).with_name('viewer.html').read_bytes().replace(b'__CLAMP_VALUE__', str(float(size_clamp_multiplier)).encode())
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, code, body=b'', content_type='text/plain'):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path == '/':
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif path == '/health':
                    self.reply(200, json.dumps({'preview_directory': preview_directory}).encode(), 'application/json')
                elif path == '/frame.jpg':
                    with owner.lock:
                        frame = owner.frame
                    self.reply(200 if frame else 503, frame or b'Preparing viewer', 'image/jpeg')
                else:
                    self.reply(404)

            def do_POST(self):
                if self.path != '/control':
                    self.reply(404)
                    return
                try:
                    if self.headers.get('Content-Type') != 'application/json':
                        raise ValueError('Expected JSON')
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 1024:
                        raise ValueError('Invalid length')
                    command = json.loads(self.rfile.read(length))
                    action = command['action']
                    if action not in ('orbit', 'pan', 'zoom', 'reset', 'flip', 'left', 'right', 'straighten', 'size_clamp'):
                        raise ValueError('Invalid action')
                    values = [float(command.get(k, 0)) for k in ('dx', 'dy')]
                    if not all(math.isfinite(v) and abs(v) <= 1000 for v in values):
                        raise ValueError('Invalid movement')
                    if action == "size_clamp" and values[0] < 0:
                        raise ValueError("Invalid size clamp")
                    owner.commands.put_nowait((action, *values))
                except (ValueError, KeyError, TypeError, Full):
                    self.reply(400, b'Invalid control')
                    return
                self.reply(204)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_port
        print(f"Viewer ready: http://localhost:{port}/", flush=True)
        if host == '0.0.0.0':
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.connect(('192.0.2.1', 80))  # Route lookup; sends no packet.
                    address = probe.getsockname()[0]
                print(f'Join from phone: http://{address}:{port}/', flush=True)
            except OSError:
                print("Unable to find IP!")
        else:
            print(f'Listening on {host}, port {port}', flush=True)

    def publish(self, frame):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            with self.lock:
                self.frame = encoded.tobytes()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
