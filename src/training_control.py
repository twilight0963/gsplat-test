"""Training state shared by the engine, the viewer and the upload server.

Kept free of heavy imports so the upload server can use it without loading PyTorch.
"""
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value, tag=None):
    # A per-writer temporary name: two runs sharing runs/.viewer must not rename
    # each other's half-written files.
    temporary = path.with_name(f'{path.name}.{tag or uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


STOP_ACTIONS = ('save', 'discard')


def read_state(directory):
    """The training state the viewer and upload page show, or None before any run."""
    try:
        return json.loads((Path(directory) / 'state.json').read_text())
    except (FileNotFoundError, ValueError):
        return None


def training_active(state):
    """True while the run that wrote `state` is training and its process is alive."""
    if not state or not state.get('active'):
        return False
    try:
        os.kill(int(state['pid']), 0)  # signal 0: existence check only
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def request_stop(directory, action):
    """Ask the training run publishing to `directory` to stop.

    'save' ends training at the current step and exports the model; 'discard'
    ends the run without saving. The request names the active session, so it
    can never stop a later run. Returns (accepted, message).
    """
    if action not in STOP_ACTIONS:
        raise ValueError(f'Unknown stop action: {action!r}')
    state = read_state(directory)
    if not training_active(state):
        return False, 'No training is running.'
    atomic_json(Path(directory) / 'stop.json',
                {'session': state['session'], 'action': action, 'requested': time.time()})
    return True, ('Stopping after the current step; the model will be saved.' if action == 'save'
                  else 'Stopping; nothing will be saved.')
