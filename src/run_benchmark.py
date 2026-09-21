"""Per-run wall-clock timings, with nested stages counted only once."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
from pathlib import Path
from time import perf_counter


_active = ContextVar('run_benchmark', default=None)
STAGES = ('input_preparation', 'preprocessing', 'cas', 'super_resolution',
          'colmap', 'reconstruction_loading', 'viewer_startup', 'training', 'export')


@contextmanager
def stage(name):
    run = _active.get()
    if run is None:
        yield
        return
    entry = [perf_counter(), 0.0]
    run['stack'].append(entry)
    try:
        yield
    finally:
        elapsed = perf_counter() - entry[0]
        run['stack'].pop()
        run['seconds'][name] += max(0.0, elapsed - entry[1])
        if run['stack']:
            run['stack'][-1][1] += elapsed


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def frame_counts(**counts):
    run = _active.get()
    if run is not None:
        run['counts'].update(counts)


def benchmark_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        started = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        run = {'seconds': dict.fromkeys(STAGES, 0.0), 'stack': [],
               'counts': {'decoded_video_frames': None, 'frames_before_colmap': None,
                          'frames_after_colmap': None}}
        token = _active.set(run)
        try:
            result = function(*args, **kwargs)
        finally:
            _active.reset(token)
        total = perf_counter() - start
        seconds = run['seconds']
        preprocessing = sum(seconds[k] for k in ('preprocessing', 'cas', 'super_resolution'))
        report = {
            'status': 'completed', 'started_at_utc': started,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'input': str(options['video']),
            'input_type': 'images' if Path(options['video']).is_dir() else 'video',
            'model': str(result), 'frame_counts': run['counts'],
            'total_seconds': total, 'stage_seconds': seconds,
            'preprocessing_total_seconds': preprocessing,
            'other_seconds': max(0.0, total - sum(seconds.values())),
            'settings': {key: options[key] for key in ('steps', 'every', 'max_width', 'device',
                'super_resolution', 'sr_tile', 'sr_prior_weight', 'sharpness', 'brightness',
                'contrast', 'image_matching', 'spatial_neighbors', 'use_server')},
        }
        report['settings']['view_batch_size'] = options['view_batch_size'] or (1 if options['super_resolution'] else 4)
        report['settings']['unsharp'] = options['unsharp'] if options['unsharp'] is not None else not options['super_resolution']
        lines = [f"Run completed: {report['finished_at_utc']}", f"Input: {report['input']}",
                 f"Started: {started}", f"Model: {result}",
                 f"Total elapsed: {total:.2f} s ({total / 60:.2f} min)"]
        lines += [f'{key}: {value if value is not None else "N/A"}' for key, value in run['counts'].items()]
        lines += [f'{key}: {value:.2f} s' for key, value in seconds.items()]
        lines += [f'Preprocessing total (includes CAS and SR): {preprocessing:.2f} s',
                  f'Other/setup/cleanup: {report["other_seconds"]:.2f} s',
                  'Stage times are exclusive wall times; preprocessing total is a subtotal.',
                  'COLMAP includes extraction, matching, mapping and undistortion.',
                  'Training includes target loading, optimization, preview snapshots and final box fitting.',
                  'Frames after COLMAP are usable registered images in the selected reconstruction.',
                  'Settings: ' + json.dumps(report['settings'], sort_keys=True)]
        output = Path(options['output'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
        (output / 'benchmark.log').write_text('\n'.join(lines) + '\n')
        print(f'Run time: {total:.2f} s ({total / 60:.2f} min); '
              f'COLMAP: {seconds["colmap"]:.2f} s; preprocessing: {preprocessing:.2f} s; '
              f'training: {seconds["training"]:.2f} s. Log: {output / "benchmark.log"}', flush=True)
        return result
    return wrapped
