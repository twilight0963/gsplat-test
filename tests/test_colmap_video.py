from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import engine


class VideoColmapTests(unittest.TestCase):
    def test_overlap_loop_detection_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / 'capture'
            capture.mkdir()
            (capture / 'frame_000000.jpg').write_bytes(b'fixture')
            vocab = root / 'vocab.bin'
            vocab.write_bytes(b'fixture')
            work = root / 'colmap'

            def fake_run(command):
                if command[1] == 'global_mapper':
                    (work / 'sparse' / '0').mkdir()
                elif command[1] == 'image_undistorter':
                    (work / 'undistorted' / 'sparse').mkdir(parents=True)
                    (work / 'undistorted' / 'images').mkdir()

            with patch.object(engine, '_run', side_effect=fake_run) as run:
                engine.run_colmap(capture, work, vocab, sequential_overlap=12)
                match = next(c.args[0] for c in run.call_args_list if c.args[0][1] == 'sequential_matcher')
                self.assertEqual(match[match.index('--SequentialMatching.overlap') + 1], '12')
                self.assertIn('--SequentialMatching.loop_detection', match)
                self.assertEqual(match[match.index('--SequentialMatching.quadratic_overlap') + 1], '0')
                mapper = next(c.args[0] for c in run.call_args_list if c.args[0][1] == 'global_mapper')
                self.assertNotIn('--GlobalMapper.track_required_tracks_per_view', mapper)
                run.reset_mock()
                engine.run_colmap(capture, work, vocab, sequential_overlap=12)
                run.assert_not_called()
                engine.run_colmap(capture, work, vocab, sequential_overlap=30)
                self.assertTrue(run.called)
                match = next(c.args[0] for c in run.call_args_list if c.args[0][1] == 'sequential_matcher')
                self.assertEqual(match[match.index('--SequentialMatching.overlap') + 1], '30')

    def test_mapper_track_cap_is_passed_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / 'capture'
            capture.mkdir()
            (capture / 'frame_000000.jpg').write_bytes(b'fixture')
            work = root / 'colmap'

            def fake_run(command):
                if command[1] == 'global_mapper':
                    (work / 'sparse' / '0').mkdir()
                elif command[1] == 'image_undistorter':
                    (work / 'undistorted' / 'sparse').mkdir(parents=True)
                    (work / 'undistorted' / 'images').mkdir()

            with patch.object(engine, '_run', side_effect=fake_run) as run:
                engine.run_colmap(capture, work, None)
                run.reset_mock()
                engine.run_colmap(capture, work, None, mapper_tracks_per_view=1500)
                mapper = next(c.args[0] for c in run.call_args_list if c.args[0][1] == 'global_mapper')
                self.assertEqual(mapper[mapper.index('--GlobalMapper.track_required_tracks_per_view') + 1], '1500')
