"""Read photo metadata before COLMAP or any pixel preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


@dataclass
class Photo:
    path: Path
    size: tuple[int, int]
    camera: tuple[str, str, str]
    timestamp: str | None
    calibration: tuple[float, float, float] | None
    gps: bool


def read_photo(path: Path) -> Photo:
    with Image.open(path) as image:
        exif = image.getexif()
        if exif.get(274, 1) != 1:
            raise ValueError(f"Image orientation must be upright before import: {path}")
        details = exif.get_ifd(34665)
        gps = exif.get_ifd(34853)
        values = {}
        try:
            root = ET.fromstring(image.info.get('xmp', b'<empty/>'))
            for node in root.iter():
                for key, value in [(node.tag, node.text), *node.attrib.items()]:
                    if key.startswith('{http://www.dji.com/drone-dji/1.0/}'):
                        values[key.split('}')[-1]] = value
        except ET.ParseError:
            pass  # EXIF and visual matching remain usable without DJI XMP.
        calibration = None
        try:
            f, cx, cy = (float(values[k]) for k in (
                'CalibratedFocalLength', 'CalibratedOpticalCenterX', 'CalibratedOpticalCenterY'))
            if all(math.isfinite(x) for x in (f, cx, cy)) and f > 0 and 0 < cx < image.width and 0 < cy < image.height:
                calibration = (f, cx, cy)
        except (KeyError, TypeError, ValueError):
            pass
        has_gps = False
        try:
            lat = sum(float(v) / d for v, d in zip(gps[2], (1, 60, 3600)))
            lon = sum(float(v) / d for v, d in zip(gps[4], (1, 60, 3600)))
            has_gps = (gps[1] in ('N', 'S') and gps[3] in ('E', 'W') and
                       math.isfinite(lat) and math.isfinite(lon) and
                       0 <= lat <= 90 and 0 <= lon <= 180 and (lat != 0 or lon != 0))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        return Photo(path, image.size,
                     (str(exif.get(271, '')), str(exif.get(272, '')), str(details.get(37386, ''))),
                     details.get(36867), calibration, has_gps)


@dataclass
class ImageDataset:
    capture: Path
    camera_params: str | None
    single_camera: bool
    spatial: bool
    max_image_size: int
    count: int


def prepare_image_dataset(source: Path, output: Path, max_width: int) -> ImageDataset:
    """Copy originals byte-for-byte; COLMAP resizes features and undistorted targets."""
    if max_width < 0:
        raise ValueError('--max-width must be nonnegative')
    paths = sorted(p for p in source.iterdir() if p.is_file() and
                   p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if len(paths) < 2:
        raise ValueError('Image directory must contain at least two supported images')
    photos = [read_photo(p) for p in paths]
    if len({p.size for p in photos}) != 1:
        raise ValueError('Image directory must have consistent dimensions for batched splat training')
    # Only use capture-time ordering when every timestamp is available.
    if all(p.timestamp for p in photos):
        photos.sort(key=lambda p: (p.timestamp, p.path.name))
    capture = output / 'capture' / 'photos'
    if capture.exists():
        raise ValueError('Photo capture folder already exists; use a new output directory')
    if (output / 'colmap' / 'database.db').exists():
        raise ValueError('COLMAP database already exists; use a new output directory')
    capture.mkdir(parents=True)
    mapping = []
    for i, photo in enumerate(photos):
        name = f'frame_{i:06d}{photo.path.suffix.lower()}'
        shutil.copy2(photo.path, capture / name)
        mapping.append({'image': name, 'source': str(photo.path.resolve()),
                        'timestamp': photo.timestamp, 'gps': photo.gps,
                        'calibration': photo.calibration})
    (output / 'photo_metadata.json').write_text(json.dumps(mapping, indent=2) + '\n')
    same_camera = len({p.camera for p in photos}) == 1
    calibration = photos[0].calibration
    shared = same_camera and all(p.calibration == calibration for p in photos)
    params = None
    if shared and calibration is not None:
        f, cx, cy = calibration
        # SIMPLE_RADIAL permits distortion refinement while seeding f/cx/cy.
        params = ','.join(map(str, (f, cx, cy, 0.0)))
    width, height = photos[0].size
    limit = round(max(width, height) * min(1, max_width / width)) if max_width else -1
    spatial = all(p.gps for p in photos)
    print(f'Imported {len(photos)} photos; GPS: {sum(p.gps for p in photos)}/{len(photos)}; '
          f'calibrated intrinsics: {"yes" if params else "no"}', flush=True)
    return ImageDataset(capture, params, shared, spatial, max(1, limit) if max_width else -1, len(photos))
