"""Browser upload interface for the existing engine CLI."""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlsplit
import socket

ROOT = Path(__file__).resolve().parent.parent
# Intermediates of an upload job, removed once its model is saved. The model,
# benchmark reports and photo_metadata.json in the output folder are kept.
TEMPORARY_OUTPUTS = ('capture', 'capture_subset', 'colmap', 'sr')


def parameters(query, swinir_root=None, sr_checkpoint=None):
    values = parse_qs(query, strict_parsing=True)
    if any(len(items) != 1 for items in values.values()):
        raise ValueError('Duplicate upload parameters are not allowed')
    result = {}
    for name, default, lower, upper, kind in (
        ('steps', '2000', 1, 1000000, int), ('every', '5', 1, 100000, int),
        ('max-width', '1920', 0, 16384, int), ('brightness', '0', -255, 255, float),
        ('contrast', '1', 0, 10, float), ('sharpness', '.5', 0, 1, float),
    ):
        value = kind(values.get(name, [default])[0])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'Invalid {name}: expected {lower} to {upper}')
        result[name] = str(value)
    sr = values.get('super-resolution', ['false'])[0]
    if sr not in ('true', 'false'):
        raise ValueError('Invalid super-resolution: expected true or false')
    unsharp = values.get('unsharp', ['auto'])[0]
    if unsharp not in ('auto', 'on', 'off'):
        raise ValueError('Invalid unsharp: expected auto, on or off')
    if unsharp != 'auto':
        result['unsharp' if unsharp == 'on' else 'no-unsharp'] = None
    if sr == 'true':
        tile = int(values.get('sr-tile', ['128'])[0])
        if tile < 32 or tile > 512 or tile % 8:
            raise ValueError('SR tile size must be a multiple of 8 between 32 and 512')
        weight = float(values.get('sr-prior-weight', ['0.5'])[0])
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('SR prior weight must be between 0 and 1')
        # These executable-source paths are configured by the server operator,
        # never taken from the browser request.
        root = Path(swinir_root) if swinir_root is not None else ROOT / 'third_party/SwinIR'
        checkpoint = Path(sr_checkpoint) if sr_checkpoint is not None else ROOT / 'weights/swinir-lightweight-x2.pth'
        if not (root / 'models/network_swinir.py').is_file() or not checkpoint.is_file():
            raise ValueError('Super-resolution is not configured. Install SwinIR and its 2x weights on the server; see README.md.')
        result.update({'super-resolution': None, 'swinir-root': str(root.resolve()),
                       'sr-checkpoint': str(checkpoint.resolve()), 'sr-tile': str(tile),
                       'sr-prior-weight': str(weight)})
    output = values.get('output', ['runs/web-' + uuid.uuid4().hex[:8]])[0]
    path = (ROOT / output).resolve()
    if not path.is_relative_to(ROOT / 'runs') or path == ROOT / 'runs':
        raise ValueError('Output must be a new folder inside runs/')
    if path.exists():
        raise ValueError('Output folder already exists; choose a new name')
    result['output'] = str(path)
    # Uploaded jobs always use the persistent browser viewer. This is fixed by
    # the server and cannot be overridden by request parameters.
    result['use-server'] = None
    return result


def clean_up(upload, output):
    """Delete a finished job's upload and intermediates; return the bytes freed.

    Only paths inside runs/ are touched: the upload must be a direct child of
    runs/.uploads and the output a folder inside runs/.
    """
    runs = (ROOT / 'runs').resolve()
    upload, output = Path(upload), Path(output)
    if upload.resolve().parent != runs / '.uploads' or not output.resolve().is_relative_to(runs) \
            or output.resolve() == runs:
        raise ValueError(f'Refusing to clean up outside runs/: {upload}, {output}')
    freed, seen = 0, set()
    for path in (upload, *(output / name for name in TEMPORARY_OUTPUTS)):
        if not path.exists() and not path.is_symlink():
            continue
        files = [path] if path.is_file() or path.is_symlink() else [f for f in path.rglob('*') if f.is_file() and not f.is_symlink()]
        for f in files:
            stat = f.lstat()
            if (stat.st_dev, stat.st_ino) not in seen:  # hard links are freed once
                seen.add((stat.st_dev, stat.st_ino))
                freed += stat.st_size
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return freed


class Jobs:
    def __init__(self):
        self.lock = Lock()
        self.process = None
        self.job = None  # (upload, output) of the running job
        self.state = {'busy': False, 'status': 'Ready to upload', 'viewer': False, 'saved': False}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def reserve(self):
        with self.lock:
            if self.state['busy'] or (self.process is not None and self.process.poll() is None):
                raise ValueError('A training job is already running. Wait for it to finish before uploading again.')
            self.state = {'busy': True, 'status': 'Receiving video…', 'viewer': False, 'saved': False}

    def fail(self, message):
        with self.lock:
            self.state.update(busy=False, status=message)

    def launch(self, video, params):
        command = [sys.executable, '-u', '-m', 'src.engine', str(video)]
        for key, value in params.items():
            command.append('--' + key)
            if value is not None:
                command.append(value)
        self.job = (Path(video), Path(params['output']))
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
        note = None
        # Failed jobs keep their upload and intermediates for inspection or a retry.
        if not code and self.job is not None and (self.job[1] / 'model.glb').is_file():
            try:
                freed = clean_up(*self.job)
                note = f'Model saved: {self.job[1] / "model.glb"}. Removed temporary files ({freed / 1024 ** 3:.2f} GiB).'
            except (OSError, ValueError) as exc:
                note = f'Model saved: {self.job[1] / "model.glb"}. Could not remove temporary files: {exc}'
            print(note, flush=True)
        with self.lock:
            self.state['busy'] = False
            # The detached viewer survives engine exit.
            if code:
                self.state['status'] = f'Failed (exit {code}): ' + self.state['status']
            elif note:
                self.state['status'] = note

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def make_server(host, port, jobs, *, swinir_root=None, sr_checkpoint=None):
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
                params = parameters(url.query, swinir_root, sr_checkpoint)
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
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--swinir-root', type=Path, default=ROOT / 'third_party/SwinIR',
                        help='Official SwinIR source checkout for browser SR jobs')
    parser.add_argument('--sr-checkpoint', type=Path, default=ROOT / 'weights/swinir-lightweight-x2.pth',
                        help='Pretrained SwinIR-S lightweight 2x checkpoint')
    args = parser.parse_args()
    if args.port == 8000:
        parser.error('Port 8000 is reserved for the viewer; use 8001 for uploads')
    jobs = Jobs()
    server = make_server(args.host, args.port, jobs,
                         swinir_root=args.swinir_root, sr_checkpoint=args.sr_checkpoint)
    print(f'Upload interface: http://localhost:{server.server_port}/', flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 80))  # Route lookup; sends no packet.
            address = probe.getsockname()[0]
        print(f'Join from phone: http://{address}:{args.port}/', flush=True)
    except OSError:
        print("Unable to find IP!")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()


if __name__ == '__main__':
    main()
