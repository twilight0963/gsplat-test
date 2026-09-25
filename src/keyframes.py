"""Conservative video keyframes using tracked motion, coverage and image sharpness.

Motion is an image-space proxy, not a measurement of metric parallax. Tracking
failure retains frames rather than silently skipping an unobserved transition.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import shutil

import cv2
import numpy as np


def _gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'Cannot read keyframe candidate: {path}')
    scale = min(1., 320 / max(image.shape))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def _motion(anchor, current):
    if anchor.shape != current.shape:
        return None
    points = cv2.goodFeaturesToTrack(anchor, 200, .01, 7)
    if points is None or len(points) < 20:
        return None
    forward, status, _ = cv2.calcOpticalFlowPyrLK(anchor, current, points, None)
    if forward is None:
        return None
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, anchor, forward, None)
    if backward is None:
        return None
    valid = ((status.ravel() != 0) & (back_status.ravel() != 0)
             & (np.linalg.norm((backward - points).reshape(-1, 2), axis=1) < 1.5))
    if valid.sum() < max(20, len(points) * .5):
        return None
    displacement = np.linalg.norm((forward - points).reshape(-1, 2)[valid], axis=1)
    return float(np.median(displacement) / math.hypot(*anchor.shape))


def _select(grays, count, max_gap, motion_threshold):
    anchor = next(grays)
    selected = [0]
    reasons = ['first']
    pending = []
    for index, current in enumerate(grays, start=1):
        motion = _motion(anchor, current)
        sharpness = float(cv2.Laplacian(current, cv2.CV_32F).var())
        pending.append((index, current, motion, sharpness))
        last = index == count - 1
        if motion is None or last:
            # Retain both sides of a tracking break, including the last good view.
            if motion is None and index - 1 > selected[-1]:
                selected.append(index - 1)
                reasons.append('before_tracking_break')
            selected.append(index)
            reasons.append('last' if last else 'tracking_break')
            anchor = current
            pending = []
        elif motion >= motion_threshold or index - selected[-1] >= max_gap:
            eligible = [p for p in pending if p[0] >= index - 1 and p[2] is not None
                        and p[2] >= .7 * motion]
            best = max(eligible, key=lambda p: (p[3], p[0]))
            selected.append(best[0])
            reasons.append('motion' if motion >= motion_threshold else 'max_gap')
            anchor = best[1]
            pending = []
    return selected, reasons


def select_keyframes(files, output: Path, *, max_gap=6, motion_threshold=.06):
    """Return an isolated subset, keeping endpoints and at most max_gap gaps.

    Once motion or the temporal limit triggers, prefer the sharpest recent view
    with at least 70% of the trigger motion. No hard count cap sacrifices coverage.
    max_gap=1 disables selection. Files must be in capture order.
    """
    if max_gap < 1 or not math.isfinite(motion_threshold) or not 0 < motion_threshold < 1:
        raise ValueError('Keyframe max gap must be >= 1 and motion must be between 0 and 1')
    if len(files) < 2:
        raise ValueError('At least two keyframe candidates are required')
    if max_gap == 1:
        return files[0].parent
    if (output / 'keyframes.json').exists():
        saved = json.loads((output / 'keyframes.json').read_text())
        if (saved.get('source_signature') == [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in files]
                and saved['max_gap'] == max_gap and saved['motion_threshold'] == motion_threshold
                and all((output / f['image']).is_file() for f in saved['frames'])):
            return output
    if output.exists():
        raise ValueError(f'Keyframe output already exists: {output}; use a new output directory')
    # Decoding releases the GIL, so read ahead while tracking runs.
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected, reasons = _select(pool.map(_gray, files), len(files), max_gap, motion_threshold)
    output.mkdir(parents=True)
    for index in selected:
        shutil.copy2(files[index], output / files[index].name)
    report = {'source_signature': [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in files],
              'candidate_count': len(files), 'selected_count': len(selected),
              'max_gap': max_gap, 'motion_threshold': motion_threshold,
              'frames': [{'candidate_index': i, 'image': files[i].name, 'reason': reason}
                         for i, reason in zip(selected, reasons)]}
    (output / 'keyframes.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Keyframes: {len(selected)}/{len(files)} candidates retained', flush=True)
    return output
