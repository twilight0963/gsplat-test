"""Small HTTP transport for the GPU viewer; never serves workspace files."""
import hashlib
import json
import math
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue, Full
from threading import Lock, Thread

import cv2

from src.training_control import STOP_ACTIONS, read_state, request_stop, training_active

_CODE = Path(__file__).parent
_VIEWER_FILES = ('viewer.html', 'viewer_http.py', 'gsplat_viewer.py', 'live_viewer.py', 'training_control.py')


def code_version():
    """Fingerprint of the viewer's code, so a job can tell whether a running viewer is outdated."""
    digest = hashlib.sha1()
    for name in _VIEWER_FILES:
        digest.update((_CODE / name).read_bytes())
    return digest.hexdigest()[:16]


class ViewerHTTP:
    def __init__(self, host='0.0.0.0', port=8000, size_clamp_multiplier=1.0, preview_directory=None):
        self.commands = Queue(maxsize=256)
        self.lock = Lock()
        self.frame = None
        self.revision = 0  # bumps with every published frame, so the page can skip unchanged ones
        self.shutdown_requested = False
        version = code_version()
        page = Path(__file__).with_name('viewer.html').read_bytes().replace(b'__CLAMP_VALUE__', str(float(size_clamp_multiplier)).encode())
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, code, body=b'', content_type='text/plain', headers=()):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The page navigated away mid-reply.

            def json_reply(self, code, value):
                self.reply(code, json.dumps(value).encode(), 'application/json')

            def read_json(self):
                if self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected JSON')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('Invalid length')
                return json.loads(self.rfile.read(length))

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path == '/':
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif path == '/health':
                    self.json_reply(200, {'preview_directory': preview_directory, 'version': version})
                elif path == '/status':
                    self.json_reply(200, owner.status())
                elif path == '/frame.jpg':
                    with owner.lock:
                        frame, revision = owner.frame, owner.revision
                    self.reply(200 if frame else 503, frame or b'Preparing viewer', 'image/jpeg',
                               [('X-Frame-Revision', str(revision))])
                else:
                    self.reply(404)

            def do_POST(self):
                if self.path == '/stop':
                    self.stop()
                    return
                if self.path == '/shutdown':
                    # Only the local machine may replace the viewer (see live_viewer.ensure_viewer).
                    if self.client_address[0] not in ('127.0.0.1', '::1'):
                        self.reply(403)
                        return
                    owner.shutdown_requested = True
                    self.reply(204)
                    return
                if self.path != '/control':
                    self.reply(404)
                    return
                try:
                    command = self.read_json()
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

            def stop(self):
                if preview_directory is None:
                    self.json_reply(409, {'error': 'This viewer is showing a saved model, not a training run.'})
                    return
                try:
                    action = self.read_json()['action']
                    if action not in STOP_ACTIONS:
                        raise ValueError(action)
                except (ValueError, KeyError, TypeError):
                    self.json_reply(400, {'error': "Expected {'action': 'save' | 'discard'}"})
                    return
                accepted, message = request_stop(preview_directory, action)
                self.json_reply(202 if accepted else 409, {'message': message} if accepted else {'error': message})

        self.preview_directory = preview_directory
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

    def status(self):
        """Training progress for the page: a saved model has none."""
        if self.preview_directory is None:
            return {'training': False, 'status': 'Saved model', 'completed': None, 'total': None}
        state = read_state(self.preview_directory) or {}
        return {'training': training_active(state), 'status': state.get('status', 'Waiting for training'),
                'completed': state.get('completed'), 'total': state.get('total')}

    def publish(self, frame):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            with self.lock:
                self.frame = encoded.tobytes()
                self.revision += 1

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
