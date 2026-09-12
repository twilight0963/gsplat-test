"""Browser upload interface for the existing engine CLI."""
import argparse
import json
import math
import re
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent


def parameters(query):
    values = parse_qs(query, strict_parsing=True)
    result = {}
    for name, default, lower, upper, kind in (
        ('steps', '2000', 1, 1000000, int), ('every', '5', 1, 100000, int),
        ('max-width', '1920', 0, 16384, int), ('brightness', '1', -255, 255, float),
        ('contrast', '1', 0, 10, float), ('sharpness', '.5', 0, 1, float),
    ):
        value = kind(values.get(name, [default])[0])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'Invalid {name}: expected {lower} to {upper}')
        result[name] = str(value)
    output = values.get('output', ['runs/web-' + uuid.uuid4().hex[:8]])[0]
    path = (ROOT / output).resolve()
    if not path.is_relative_to(ROOT / 'runs') or path == ROOT / 'runs':
        raise ValueError('Output must be a new folder inside runs/')
    if path.exists():
        raise ValueError('Output folder already exists; choose a new name')
    result['output'] = str(path)
    return result


class Jobs:
    def __init__(self):
        self.lock = Lock()
        self.process = None
        self.state = {'busy': False, 'status': 'Ready to upload', 'viewer': False, 'saved': False}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def reserve(self):
        with self.lock:
            if self.state['busy'] or (self.process is not None and self.process.poll() is None):
                raise ValueError('A model session is already running. Restart this server for another run.')
            self.state = {'busy': True, 'status': 'Receiving video…', 'viewer': False, 'saved': False}

    def fail(self, message):
        with self.lock:
            self.state.update(busy=False, status=message)

    def launch(self, video, params):
        command = [sys.executable, '-u', '-m', 'src.engine', str(video)]
        for key, value in params.items():
            command.extend(['--' + key, value])
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        Thread(target=self.monitor, daemon=True).start()

    def monitor(self):
        process = self.process
        line = ''
        for char in iter(lambda: process.stdout.read(1), ''):
            if char in '\r\n':
                clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
                if clean:
                    print(clean, flush=True)
                    with self.lock:
                        self.state['status'] = clean[-2000:]
                        if clean.startswith('Viewer ready:'):
                            self.state['viewer'] = True
                        if clean.startswith('Model saved:'):
                            self.state['saved'] = True
                line = ''
            else:
                line = (line + char)[-4000:]
        code = process.wait()
        with self.lock:
            self.state['busy'] = False
            self.state['viewer'] = False
            if code:
                self.state['status'] = f'Failed (exit {code}): ' + self.state['status']

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def make_server(host, port, jobs):
    page = Path(__file__).with_name('upload.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, body, kind='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self.reply(200, page, 'text/html; charset=utf-8')
            elif path == '/status':
                self.reply(200, json.dumps(jobs.snapshot()).encode())
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path != '/upload':
                self.reply(404, b'{}')
                return
            reserved = False
            video = None
            try:
                params = parameters(url.query)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20 * 1024 ** 3:
                    raise ValueError('Upload must be between 1 byte and 20 GiB')
                if self.headers.get('Content-Type') != 'application/octet-stream':
                    raise ValueError('Expected a video upload')
                jobs.reserve()
                reserved = True
                directory = ROOT / 'runs' / '.uploads'
                directory.mkdir(parents=True, exist_ok=True)
                video = directory / (uuid.uuid4().hex + '.video')
                self.connection.settimeout(120)
                with video.open('xb') as output:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Upload interrupted')
                        output.write(chunk)
                        remaining -= len(chunk)
                jobs.launch(video, params)
                self.reply(202, b'{"ok":true}')
            except (ValueError, OSError) as exc:
                if reserved:
                    jobs.fail(str(exc))
                    if video is not None:
                        video.unlink(missing_ok=True)
                self.reply(400, json.dumps({'error': str(exc)}).encode())

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Upload a drone video and build a model in your browser')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    if args.port == 8000:
        parser.error('Port 8000 is reserved for the viewer; use 8001 for uploads')
    jobs = Jobs()
    server = make_server(args.host, args.port, jobs)
    print(f'Upload interface: http://localhost:{server.server_port}/', flush=True)
    print(f'Listening on {args.host}:{server.server_port}; viewer uses port 8000', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()


if __name__ == '__main__':
    main()
