import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src import engine
from src.image_dataset import ImageDataset, prepare_image_dataset, read_photo


class ImageDatasetTests(unittest.TestCase):
    def test_metadata_and_original_bytes_survive_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            xmp = b'''<x xmlns:d="http://www.dji.com/drone-dji/1.0/">
            <d:CalibratedFocalLength>80</d:CalibratedFocalLength>
            <d:CalibratedOpticalCenterX>50</d:CalibratedOpticalCenterX>
            <d:CalibratedOpticalCenterY>40</d:CalibratedOpticalCenterY></x>'''
            for name, time in [('a.jpg', '2020:01:01 12:02:00'), ('b.jpg', '2020:01:01 12:01:00')]:
                exif = Image.Exif()
                exif[34665] = {36867: time, 37386: 8.8}
                exif[34853] = {1: 'N', 2: (37., 59., 0.), 3: 'W', 4: (122., 3., 0.)}
                Image.new('RGB', (100, 80)).save(source / name, exif=exif, xmp=xmp)
            # No downscaling needed: originals are copied byte-for-byte.
            for max_width in (0, 100, 200):
                result = prepare_image_dataset(source, root / f'full{max_width}', max_width)
                self.assertEqual(result.camera_params, '80.0,50.0,40.0,0.0')
                self.assertEqual((result.capture / 'frame_000000.jpg').read_bytes(), (source / 'b.jpg').read_bytes())
                self.assertEqual(read_photo(result.capture / 'frame_000000.jpg').calibration, (80., 50., 40.))
            # Wider than --max-width: staged once at half size, intrinsics scaled to match.
            result = prepare_image_dataset(source, root / 'out', 50)
            self.assertEqual(result.camera_params, '40.0,25.0,20.0,0.0')
            self.assertTrue(result.spatial)
            self.assertEqual(result.max_image_size, 50)
            staged = read_photo(result.capture / 'frame_000000.jpg')
            self.assertEqual(staged.size, (50, 40))
            self.assertTrue(staged.gps)
            self.assertEqual(staged.timestamp, '2020:01:01 12:01:00')  # b.jpg sorts first by time
            self.assertNotEqual((result.capture / 'frame_000000.jpg').read_bytes(), (source / 'b.jpg').read_bytes())
            mapping = json.loads((root/'out/photo_metadata.json').read_text())
            self.assertEqual([m['scale'] for m in mapping], [.5, .5])
            self.assertEqual(mapping[0]['calibration'], [80., 50., 40.])

    def test_downscaled_png_and_tiff_keep_gps(self):
        for suffix in ('.png', '.tif'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'input'
                source.mkdir()
                exif = Image.Exif()
                exif[34853] = {1: 'N', 2: (37., 59., 0.), 3: 'W', 4: (122., 3., 0.)}
                for name in ('a', 'b'):
                    # EXIF as bytes: the form a camera file carries (and Pillow can write to TIFF).
                    Image.new('RGB', (120, 60), (10, 20, 30)).save(source / (name + suffix), exif=exif.tobytes())
                result = prepare_image_dataset(source, root / 'out', 60)
                self.assertTrue(read_photo(source / ('a' + suffix)).gps)
                staged = read_photo(result.capture / f'frame_000000{suffix}')
                self.assertEqual(staged.size, (60, 30))
                with Image.open(result.capture / f'frame_000000{suffix}') as image:
                    self.assertEqual(image.getpixel((5, 5)), (10, 20, 30))
                self.assertTrue(staged.gps)
                self.assertTrue(result.spatial)

    def test_missing_metadata_and_mixed_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            for name in ('a.png', 'b.png'):
                Image.new('RGB', (40, 80)).save(source / name)
            result = prepare_image_dataset(source, root/'out', 20)
            self.assertFalse(result.spatial)
            self.assertIsNone(result.camera_params)
            self.assertEqual(result.max_image_size, 40)
            Image.new('RGB', (40, 40)).save(source/'b.png')
            with self.assertRaisesRegex(ValueError, 'consistent dimensions'):
                prepare_image_dataset(source, root/'other', 20)

    def test_colmap_calibration_and_matching_modes(self):
        for gps, mode in [(True, 'auto'), (False, 'auto'), (True, 'exhaustive')]:
            with self.subTest(gps=gps, mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'photos').mkdir()
                for i in range(3):
                    Image.new('RGB', (100, 80)).save(root/'photos'/f'{i}.jpg')
                dataset = ImageDataset(root/'photos', '80,50,40,0', True, gps, 50, 3)
                def fake_run(command):
                    if command[1] == 'global_mapper':
                        (root/'colmap/sparse/0').mkdir(parents=True)
                    elif command[1] == 'image_undistorter':
                        (root/'colmap/undistorted').mkdir()
                with patch.object(engine, '_run', side_effect=fake_run) as run:
                    engine.run_colmap(dataset.capture, root/'colmap', None,
                                      photo_dataset=dataset, image_matching=mode)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertIn('80,50,40,0', commands[0])
                names = [c[1] for c in commands]
                if gps and mode == 'auto':
                    self.assertIn('spatial_matcher', names)
                    self.assertIn('sequential_matcher', names)
                    self.assertNotIn('exhaustive_matcher', names)
                else:
                    self.assertIn('exhaustive_matcher', names)
                    self.assertNotIn('spatial_matcher', names)
                self.assertEqual(commands[-1][-2:], ['--max_image_size', '50'])

    def test_directory_bypasses_video_and_enhances_registered_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root/'registered.png'
            Image.new('RGB', (10, 8), (50, 50, 50)).save(frame)
            Image.new('RGB', (10, 8), (50, 50, 50)).save(root/'second.png')
            dataset = ImageDataset(root, None, True, False, 10, 2)
            with patch.object(engine, 'prepare_image_dataset', return_value=dataset), \
                 patch.object(engine, 'extract_frames') as video, \
                 patch.object(engine, 'run_colmap'), \
                 patch.object(engine, 'load_reconstruction', return_value=({}, [frame], 10, 8)), \
                 patch.object(engine, 'apply_cas', side_effect=lambda images, *_: images), \
                 patch.object(engine, 'PreviewPublisher'), patch.object(engine, 'ensure_viewer'), \
                 patch.object(engine, 'train_splats', return_value={}) as train, \
                 patch.object(engine, 'export_gltf'):
                engine.build_model(root, root/'out', use_server=True, brightness=10, contrast=2, unsharp=False, device='cpu', photo_every=1)
            video.assert_not_called()
            report = json.loads((root/'out/benchmark.json').read_text())
            self.assertEqual(report['frame_counts']['frames_before_colmap'], 2)
            self.assertEqual(report['frame_counts']['frames_after_colmap'], 1)
            self.assertIsNone(report['frame_counts']['decoded_video_frames'])
            target = train.call_args.kwargs['targets']
            self.assertEqual(target[0, 0, 0].tolist(), [110, 110, 110])


if __name__ == '__main__':
    unittest.main()
