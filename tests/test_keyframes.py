import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.keyframes import _motion, select_keyframes


class KeyframeTests(unittest.TestCase):
    def candidates(self, root, count=20, blank=False):
        rng = np.random.default_rng(12)
        image = np.zeros((100, 150), np.uint8) if blank else rng.integers(0, 256, (100, 150), np.uint8)
        files = []
        for i in range(count):
            path = root / f'frame_{i:06d}.png'
            cv2.imwrite(str(path), image)
            files.append(path)
        return files

    def test_static_texture_reduces_frames_with_bounded_gaps_and_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root)
            output = select_keyframes(files, root / 'selected', max_gap=6)
            report = json.loads((output / 'keyframes.json').read_text())
            indices = [f['candidate_index'] for f in report['frames']]
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 19)
            self.assertLess(len(indices), 10)
            self.assertLessEqual(max(np.diff(indices)), 6)
            self.assertEqual(select_keyframes(files, output), output)
            for index in indices:
                self.assertEqual((output / files[index].name).read_bytes(), files[index].read_bytes())

    def test_untrackable_frames_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root, blank=True)
            output = select_keyframes(files, root / 'selected')
            self.assertEqual(len(list(output.glob('*.png'))), len(files))

    def test_tracking_break_preserves_previous_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root, count=5)
            with patch('src.keyframes._motion', side_effect=[0., 0., None, 0.]):
                output = select_keyframes(files, root / 'selected')
            indices = [f['candidate_index'] for f in json.loads((output / 'keyframes.json').read_text())['frames']]
            self.assertEqual(indices, [0, 2, 3, 4])

    def test_translation_is_measured(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, (200, 300), np.uint8)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 0]]), (300, 200))
        self.assertAlmostEqual(_motion(image, shifted), 5 / np.hypot(200, 300), delta=.003)

    def test_motion_trigger_prefers_sharper_nearby_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root, count=4)
            blurred = cv2.GaussianBlur(cv2.imread(str(files[2]), 0), (11, 11), 3)
            cv2.imwrite(str(files[2]), blurred)
            with patch('src.keyframes._motion', side_effect=[.05, .065, .01]):
                output = select_keyframes(files, root / 'selected')
            report = json.loads((output / 'keyframes.json').read_text())
            self.assertEqual([f['candidate_index'] for f in report['frames']], [0, 1, 3])
            self.assertEqual(report['frames'][1]['reason'], 'motion')

    def test_changed_input_does_not_reuse_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root)
            output = select_keyframes(files, root / 'selected')
            files[1].write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                select_keyframes(files, output)

    def test_disable_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.candidates(root, count=2)
            self.assertEqual(select_keyframes(files, root / 'selected', max_gap=1), root)
            self.assertFalse((root / 'selected').exists())
            for kwargs in ({'max_gap': 0}, {'motion_threshold': float('nan')}):
                with self.assertRaises(ValueError):
                    select_keyframes(files, root / 'selected', **kwargs)
