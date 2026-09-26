"""Persistent viewer process and an atomic, NumPy-only preview mailbox."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

import numpy as np
import torch

from src.training_control import STOP_ACTIONS, atomic_json, read_state, request_stop, training_active  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


class PreviewPublisher:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.closed = False
        self.revision = 0
        self.stop_checked = None
        self.state = {'session': self.session, 'revision': 0, 'snapshot': None, 'active': True, 'pid': os.getpid(),
                      'completed': 0, 'total': None,
                      'status': 'Training: 0.0% - preparing images and splats'}
        self._write()

    def stop_requested(self):
        """'save' or 'discard' once, if a stop was requested for this session; else None."""
        path = self.directory / 'stop.json'
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if stamp == self.stop_checked:
            return None
        self.stop_checked = stamp
        try:
            request = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if request.get('session') != self.session or request.get('action') not in STOP_ACTIONS:
            return None
        return request['action']

    def _write(self):
        atomic_json(self.directory / 'state.json', self.state, self.session)

    def publish(self, data, completed, total):
        if self.closed:
            return
        path = self.directory / 'snapshot.npz'
        temporary = self.directory / f'snapshot.{self.session}.tmp'
        with temporary.open('wb') as output:
            np.savez(output, **{key: tensor.detach().cpu().numpy() for key, tensor in data.items()},
                     _session=np.array(self.session), _revision=np.array(self.revision + 1))
        os.replace(temporary, path)
        self.revision += 1
        self.state.update(revision=self.revision, snapshot=path.name, completed=completed, total=total,
                          status=f'Training: {100 * completed / total:.1f}% ({completed}/{total})')
        self._write()

    def finish(self, error=None, message=None):
        if message is None:
            message = f'Training failed: {error}' if error else 'Training: 100.0% - model saved'
        self.state.update(status=message, active=False)
        self._write()

    def close(self):
        self.closed = True


class FilePreview:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.version = None
        self.closed = False

    def read(self):
        try:
            state = json.loads((self.directory / 'state.json').read_text())
        except FileNotFoundError:
            return None, 'Waiting for training', None
        version = (state['session'], state['revision'])
        if state['snapshot'] is None:
            return None, state['status'], None
        if version == self.version:
            return None, state['status'], None
        with np.load(self.directory / state['snapshot'], allow_pickle=False) as snapshot:
            if str(snapshot['_session']) != version[0] or int(snapshot['_revision']) != version[1]:
                return None, state['status'], None  # Publisher is between atomic replacements.
            data = {key: torch.from_numpy(snapshot[key].copy()) for key in snapshot.files if not key.startswith('_')}
        data['_new_session'] = self.version is None or self.version[0] != version[0]
        self.version = version
        return data, state['status'], None

    def close(self):
        self.closed = True


def _viewer_health():
    try:
        with urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def ensure_viewer(directory, width, height):
    from src.viewer_http import code_version
    directory = Path(directory).resolve()
    version = code_version()

    def healthy():
        health = _viewer_health()
        return bool(health) and health.get('preview_directory') == str(directory) and health.get('version') == version

    if healthy():
        print('Viewer ready: http://localhost:8000/', flush=True)
        return
    health = _viewer_health()
    if health and health.get('preview_directory') == str(directory):
        # A viewer from older code is still running: ask it to exit so this one serves the current page.
        try:
            with urlopen(Request('http://127.0.0.1:8000/shutdown', data=b'', method='POST'), timeout=2):
                pass
        except OSError:
            pass
        for _ in range(100):
            if _viewer_health() is None:
                break
            time.sleep(.1)
        else:
            print('Warning: an older viewer is running on port 8000 and could not be replaced; '
                  'restart it (stop the src.live_viewer process) to get the current viewer page.', flush=True)
            print('Viewer ready: http://localhost:8000/', flush=True)
            return
    log_path = directory / 'viewer.log'
    with log_path.open('ab') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'src.live_viewer', '--state-dir', str(directory),
             '--width', str(width), '--height', str(height)], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    for _ in range(300):
        if healthy():
            print('Viewer ready: http://localhost:8000/', flush=True)
            return
        if process.poll() is not None:
            raise RuntimeError(f'Viewer failed to start; check {log_path}. Port 8000 may already be in use.')
        time.sleep(.1)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError(f'Viewer startup timed out; check {log_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    args = parser.parse_args()
    from src.gsplat_viewer import start_viewer
    start_viewer(None, args.width, args.height, preview=FilePreview(args.state_dir))


if __name__ == '__main__':
    main()
