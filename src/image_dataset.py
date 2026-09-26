"""Read photo metadata before COLMAP or any pixel preprocessing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def _portable_exif(exif) -> bytes | None:
    """Camera make/model plus the EXIF (focal length, time) and GPS sub-IFDs."""
    clean = Image.Exif()
    for tag in (271, 272):
        if tag in exif:
            clean[tag] = exif[tag]
    for ifd in (34665, 34853):
        values = exif.get_ifd(ifd)
        if values:
            clean[ifd] = dict(values)
    return clean.tobytes() if len(clean) else None


def _stage_photo(source: Path, target: Path, size: tuple[int, int] | None) -> None:
    """Copy a photo, or write it at `size` keeping its EXIF (GPS, focal length).

    Downscaling once here spares COLMAP from decoding full-size originals in
    feature extraction, color extraction and undistortion.
    """
    if size is None:
        shutil.copy2(source, target)
        return
    with Image.open(source) as image:
        if source.suffix.lower() in ('.tif', '.tiff'):
            # A TIFF's main IFD also holds its strip layout, which must not be copied.
            exif = _portable_exif(image.getexif())
        else:
            exif = image.info.get('exif')
        image.draft(image.mode, size)  # JPEG: decode at a reduced DCT scale >= size
        small = image.resize(size, Image.Resampling.LANCZOS)
    options = {'exif': exif} if exif else {}
    if target.suffix.lower() in ('.jpg', '.jpeg'):
        options['quality'] = 95
    small.save(target, **options)


def prepare_image_dataset(source: Path, output: Path, max_width: int) -> ImageDataset:
    """Stage photos for SfM: originals are copied byte-for-byte, or written once at
    `max_width` (EXIF kept, intrinsics scaled) when they are wider."""
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
    width, height = photos[0].size
    scale = min(1, max_width / width) if max_width else 1
    size = (round(width * scale), round(height * scale)) if scale < 1 else None
    names = [f'frame_{i:06d}{photo.path.suffix.lower()}' for i, photo in enumerate(photos)]
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda item: _stage_photo(item[0].path, capture / item[1], size), zip(photos, names)))
    mapping = [{'image': name, 'source': str(photo.path.resolve()),
                'timestamp': photo.timestamp, 'gps': photo.gps,
                'calibration': photo.calibration, 'scale': scale}
               for name, photo in zip(names, photos)]
    (output / 'photo_metadata.json').write_text(json.dumps(mapping, indent=2) + '\n')
    same_camera = len({p.camera for p in photos}) == 1
    calibration = photos[0].calibration
    shared = same_camera and all(p.calibration == calibration for p in photos)
    params = None
    if shared and calibration is not None:
        # Calibration is in original pixels; staged photos may be smaller.
        sx, sy = (size[0] / width, size[1] / height) if size else (1, 1)
        f, cx, cy = calibration[0] * sx, calibration[1] * sx, calibration[2] * sy
        # SIMPLE_RADIAL permits distortion refinement while seeding f/cx/cy.
        params = ','.join(map(str, (f, cx, cy, 0.0)))
    limit = round(max(width, height) * scale) if max_width else -1
    spatial = all(p.gps for p in photos)
    print(f'Imported {len(photos)} photos; GPS: {sum(p.gps for p in photos)}/{len(photos)}; '
          f'calibrated intrinsics: {"yes" if params else "no"}'
          + (f'; staged at {size[0]}x{size[1]} (from {width}x{height})' if size else ''), flush=True)
    return ImageDataset(capture, params, shared, spatial, max(1, limit) if max_width else -1, len(photos))
