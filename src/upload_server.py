"""Browser upload interface for the existing engine CLI."""
import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread, Timer
from urllib.parse import parse_qs, urlsplit
import socket

from src.training_control import STOP_ACTIONS, request_stop

ROOT = Path(__file__).resolve().parent.parent
# Intermediates of an upload job, removed once its model is saved. The model,
# benchmark reports and photo_metadata.json in the output folder are kept.
TEMPORARY_OUTPUTS = ('capture', 'capture_subset', 'colmap', 'sr')
# Where the engine publishes live training state (its PreviewPublisher with --use-server).
VIEWER_DIR = ROOT / 'runs' / '.viewer'
# The engine's exit code for "stopped by the user, nothing saved" (src/engine.py).
STOPPED_EXIT_CODE = 3
# Engine output line -> pipeline stage shown on the upload page.
STAGE_MARKERS = (
    ('Extracting video frames', 'prepare'), ('Importing photos', 'prepare'),
    ('[timing] colmap', 'reconstruct'), ('Loading reconstructions', 'reconstruct'),
    ('Training splats', 'train'), ('Fitting final bounding box', 'save'), ('Exporting model', 'save'),
)
TRAINING_PROGRESS = re.compile(r'Training splats:\s+([\d.]+)% \((\d+)/(\d+)\)')
MAX_UPLOAD = 20 * 1024 ** 3
PHOTO_BATCH = 'application/x-photo-batch'
# Must match the engine's photo import (src/image_dataset.py).
PHOTO_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
MAX_PHOTOS = 20000


def parameters(query, swinir_root=None, sr_checkpoint=None):
    values = parse_qs(query, strict_parsing=True)
    if any(len(items) != 1 for items in values.values()):
        raise ValueError('Duplicate upload parameters are not allowed')
    result = {}
    for name, default, lower, upper, kind in (
        ('steps', '5000', 1, 1000000, int), ('every', '5', 1, 100000, int),
        ('max-width', '1920', 0, 16384, int), ('brightness', '0', -255, 255, float),
        ('contrast', '1', 0, 10, float), ('sharpness', '.5', 0, 1, float),
        ('photo-every', '2', 1, 1000, int),
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


class UploadInterrupted(ValueError):
    """The browser closed the connection before sending the whole upload."""


def _read_exact(stream, size):
    data = b''
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise UploadInterrupted('Upload interrupted')
        data += chunk
    return data


def _copy(stream, output, size):
    while size:
        chunk = stream.read(min(1024 * 1024, size))
        if not chunk:
            raise UploadInterrupted('Upload interrupted')
        output.write(chunk)
        size -= len(chunk)


def receive_photos(stream, length, directory):
    """Store a photo batch as flat files in `directory` (which must not exist).

    The body is a sequence of records, written by upload.html's photoBatch():
    uint32 big-endian name length, UTF-8 file name, uint64 big-endian size, bytes.
    Names are plain file names; anything that could escape `directory` or that
    the engine would not import is rejected rather than silently skipped.
    """
    directory.mkdir(parents=True)
    remaining, seen = length, set()
    while remaining:
        if remaining < 12:
            raise ValueError('Malformed photo upload')
        name_length = struct.unpack('>I', _read_exact(stream, 4))[0]
        if not 0 < name_length <= 255 or name_length + 12 > remaining:
            raise ValueError('Malformed photo upload')
        try:
            name = _read_exact(stream, name_length).decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError('Photo file names must be UTF-8') from None
        size = struct.unpack('>Q', _read_exact(stream, 8))[0]
        remaining -= 12 + name_length
        if (name != Path(name).name or name.startswith('.') or any(c in name for c in '/\\\0')
                or Path(name).suffix.lower() not in PHOTO_EXTS):
            raise ValueError(f'Unsupported photo file name: {name!r}')
        if name.casefold() in seen:
            raise ValueError(f'Duplicate photo file name: {name}')
        if len(seen) >= MAX_PHOTOS:
            raise ValueError(f'At most {MAX_PHOTOS} photos can be uploaded')
        if not 0 < size <= remaining:
            raise ValueError(f'Malformed photo upload at {name}')
        seen.add(name.casefold())
        with (directory / name).open('xb') as output:
            _copy(stream, output, size)
        remaining -= size
    if len(seen) < 2:
        raise ValueError('Upload at least two photos')
    return len(seen)


class StopRejected(ValueError):
    """A stop request that does not apply to the current job state."""


def discard_run(upload, output):
    """Remove everything a stopped, unsaved job created: its upload, intermediates and output folder."""
    freed = clean_up(upload, output)
    output = Path(output)
    if output.is_dir() and not (output / 'model.glb').exists():
        shutil.rmtree(output)
    return freed


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
        self.state = self._fresh_state(busy=False, status='Ready to upload', stage='idle')

    @staticmethod
    def _fresh_state(**values):
        return {'busy': False, 'status': '', 'viewer': False, 'saved': False, 'stage': 'idle',
                'progress': None, 'step': None, 'steps': None, 'stopping': None, **values}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def reserve(self):
        with self.lock:
            if self.state['busy'] or (self.process is not None and self.process.poll() is None):
                raise ValueError('A training job is already running. Wait for it to finish before uploading again.')
            self.state = self._fresh_state(busy=True, status='Receiving upload…', stage='upload')

    def fail(self, message):
        with self.lock:
            self.state.update(busy=False, status=message, stage='failed')

    def stop(self, action):
        """Stop the running job early: 'save' keeps the model trained so far, 'discard' keeps nothing."""
        if action not in STOP_ACTIONS:
            raise ValueError("Expected action 'save' or 'discard'")  # a bad request, not a conflict
        with self.lock:
            process, stage = self.process, self.state['stage']
            if self.state['stopping']:
                raise StopRejected('Already stopping.')
            if stage == 'upload':
                raise StopRejected('The upload is still in progress; cancel it on the upload page.')
            if not self.state['busy'] or process is None or process.poll() is not None:
                raise StopRejected('No job is running.')
            if action == 'save' and stage != 'train':
                raise StopRejected('There is no model to save before training starts; discard the job instead.')
            self.state['stopping'] = action
        if stage == 'train':
            # The engine checks for this between training steps and exits cleanly.
            accepted, message = request_stop(VIEWER_DIR, action)
            if not accepted and action == 'save':
                with self.lock:
                    self.state['stopping'] = None
                # Training is announced just before its first step publishes state.
                raise StopRejected('Training is starting; try again in a moment.')
        else:
            accepted, message = False, ''
        if action == 'discard':
            if not accepted:
                process.terminate()  # before training there is nothing to save
            # If the engine does not exit on its own (e.g. stuck in a long export), end it.
            watchdog = Timer(60, lambda: process.poll() is None and process.terminate())
            watchdog.daemon = True
            watchdog.start()
            message = 'Stopping; nothing will be saved.'
        with self.lock:
            self.state['status'] = message
        return message

    def _update_stage(self, line):
        # The engine announces every stop, including ones sent from the viewer page.
        if line.startswith('Stop requested at step'):
            self.state['stopping'] = 'save' if 'saving' in line else 'discard'
            return
        progress = TRAINING_PROGRESS.search(line)
        if progress:
            self.state.update(stage='train', progress=float(progress.group(1)),
                              step=int(progress.group(2)), steps=int(progress.group(3)))
            return
        for marker, stage in STAGE_MARKERS:
            if marker in line:
                self.state['stage'] = stage
                return

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
                        self._update_stage(clean)
                        if clean.startswith('Viewer ready:'):
                            self.state['viewer'] = True
                        if clean.startswith('Model saved:'):
                            self.state['saved'] = True
                line = ''
            else:
                line = (line + char)[-4000:]
        process.stdout.close()
        code = process.wait()
        with self.lock:
            stopping = self.state['stopping']
        saved = not code and self.job is not None and (self.job[1] / 'model.glb').is_file()
        stopped = not saved and (code == STOPPED_EXIT_CODE or (stopping == 'discard' and code))
        note, stage = None, 'failed' if code else 'done'
        if saved:
            # Failed jobs keep their upload and intermediates for inspection or a retry.
            early = ' (stopped early)' if stopping == 'save' else ''
            try:
                freed = clean_up(*self.job)
                note = (f'Model saved{early}: {self.job[1] / "model.glb"}. '
                        f'Removed temporary files ({freed / 1024 ** 3:.2f} GiB).')
            except (OSError, ValueError) as exc:
                note = f'Model saved{early}: {self.job[1] / "model.glb"}. Could not remove temporary files: {exc}'
        elif stopped and self.job is not None:
            stage = 'stopped'
            try:
                freed = discard_run(*self.job)
                note = f'Stopped. Nothing was saved; removed this job\'s files ({freed / 1024 ** 3:.2f} GiB).'
            except (OSError, ValueError) as exc:
                note = f'Stopped. Nothing was saved. Could not remove this job\'s files: {exc}'
        if note:
            print(note, flush=True)
        with self.lock:
            self.state.update(busy=False, stage=stage, stopping=None)
            # The detached viewer survives engine exit.
            if note:
                self.state['status'] = note
            elif code:
                self.state['status'] = f'Failed (exit {code}): ' + self.state['status']

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
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser is already gone; there is nobody to tell.

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
            if url.path == '/stop':
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if self.headers.get('Content-Type') != 'application/json' or not 0 < length <= 1024:
                        raise ValueError('bad stop request')
                    action = json.loads(self.rfile.read(length)).get('action')
                    message = jobs.stop(action)
                except StopRejected as exc:
                    self.reply(409, json.dumps({'error': str(exc)}).encode())
                except (ValueError, AttributeError):
                    self.reply(400, json.dumps({'error': "Expected a JSON body {'action': 'save' | 'discard'}"}).encode())
                else:
                    self.reply(202, json.dumps({'message': message}).encode())
                return
            if url.path != '/upload':
                self.reply(404, b'{}')
                return
            reserved = False
            video = None
            try:
                params = parameters(url.query, swinir_root, sr_checkpoint)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_UPLOAD:
                    raise ValueError('Upload must be between 1 byte and 20 GiB')
                kind = self.headers.get('Content-Type')
                if kind not in ('application/octet-stream', PHOTO_BATCH):
                    raise ValueError('Expected a video or photo upload')
                photos = kind == PHOTO_BATCH
                # --every applies to video frames, --photo-every to photo folders.
                params.pop('every' if photos else 'photo-every')
                jobs.reserve()
                reserved = True
                directory = ROOT / 'runs' / '.uploads'
                directory.mkdir(parents=True, exist_ok=True)
                self.connection.settimeout(120)
                if photos:
                    video = directory / (uuid.uuid4().hex + '.photos')
                    receive_photos(self.rfile, length, video)
                else:
                    video = directory / (uuid.uuid4().hex + '.video')
                    with video.open('xb') as output:
                        _copy(self.rfile, output, length)
                jobs.launch(video, params)
                self.reply(202, b'{"ok":true}')
            except (ValueError, OSError) as exc:
                if isinstance(exc, (UploadInterrupted, ConnectionResetError, BrokenPipeError)):
                    exc = UploadInterrupted('Upload interrupted: the browser closed the connection '
                                            '(page reloaded or closed, or the file could not be read)')
                    print(exc, flush=True)
                if reserved:
                    jobs.fail(str(exc))
                    if video is not None:
                        if video.is_dir():
                            shutil.rmtree(video, ignore_errors=True)
                        else:
                            video.unlink(missing_ok=True)
                self.reply(400, json.dumps({'error': str(exc)}).encode())

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Upload a drone video or photo folder and build a model in your browser')
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
