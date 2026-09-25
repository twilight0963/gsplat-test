from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from src.clip_box import estimate_oriented_box, fit_subject_box
from src.gsplat_viewer import TrainingPreview, start_viewer
from src.live_viewer import PreviewPublisher, ensure_viewer
from src.gltf_gsplat import write_gsplat_glb
from src.voxel_reconstruction import VoxelGuidedConfig, VoxelGuidedOptimizer
from src.keyframes import select_keyframes
from src.image_dataset import ImageDataset, prepare_image_dataset
from src.run_benchmark import benchmark_run, colmap_command, frame_counts, metrics, stage, timed

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _image_size(path: Path) -> tuple[int, int] | None:
    """(height, width) without a full decode when Pillow is available."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[1], im.size[0]
    except Exception:
        pass
    image = cv2.imread(str(path))
    return None if image is None else image.shape[:2]


def _run(command: list[str]) -> None:
    started = monotonic()
    try:
        with colmap_command(command[1]) if command[0] == 'colmap' else nullcontext():
            subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    print(f"[timing] {' '.join(command[:2])}: {monotonic() - started:.1f}s", flush=True)


def unsharp_mask(frame, amount=1.5, sigma=1.0, threshold=0):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(frame.astype(int) - blurred.astype(int)) < threshold
        np.copyto(sharpened, frame, where=low_contrast_mask)
    return sharpened


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #

@timed('input_preparation')
def extract_frames(video: Path, output: Path, every: int, max_width: int, brightness: float=0, contrast: float=1, sharpness: float=0.5, *, original_only: bool = False,
                   unsharp: bool = True) -> Path:
    if every < 1:
        raise ValueError("--every must be at least 1")
    output.mkdir(parents=True, exist_ok=True)
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = reader.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[Path] = []
    index = 0

    def write(path: Path, frame: np.ndarray) -> None:
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write frame: {path}")

    # JPEG encoding releases the GIL, so overlap it with video decoding.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        while reader.grab():
            # grab() skips the colour conversion/copy for frames we don't keep.
            if index % every == 0:
                ok, frame = reader.retrieve()
                if not ok:
                    break
                if max_width and frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (max_width, round(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                if not original_only:
                    with stage('preprocessing'):
                        frame = cv2.addWeighted(frame, contrast, np.zeros_like(frame), 0, brightness)
                        if unsharp:
                            frame = unsharp_mask(frame, amount=1.2, sigma=1.0)
                path = output / f"frame_{len(frames):06d}.jpg"
                pending.append(pool.submit(write, path, frame))
                frames.append(path)
                if len(pending) >= 64:
                    for future in pending:
                        future.result()
                    pending.clear()
            index += 1
        for future in pending:
            future.result()
    reader.release()
    frame_counts(decoded_video_frames=index, frames_before_colmap=len(frames))

    if len(frames) < 2:
        raise RuntimeError("The video did not produce at least two usable frames")
    if original_only:
        return output
    output_paths = apply_cas(frames, output / "sharpened", sharpness)
    (output_paths[0].parent / "capture.json").write_text(
        json.dumps({"fps": fps, "frames": [p.name for p in output_paths]}, indent=2) + "\n"
    )
    return output_paths[0].parent


def _make_photo_subset(capture: Path, output: Path, every: int) -> Path:
    """Keep every Nth photo for SfM without touching the source files.

    Hard links are used (copy fallback) so EXIF/GPS metadata is preserved and no
    extra disk space is used. The subset lives outside `capture`, because COLMAP
    scans the image directory recursively.
    """
    files = _list_images(capture)
    subset = output / "capture_subset"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True)
    for source in files[::every]:
        target = subset / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return subset


# --------------------------------------------------------------------------- #
# CAS / sharpening
# --------------------------------------------------------------------------- #

@timed('cas')
def apply_cas(images: list[Path], output: Path, sharpness: float) -> list[Path]:
    """Use lossless output and retain the exact input order for camera pairing.

    Only used by the video/SR paths now; the standard photo path runs CAS on the
    GPU in memory (see build_targets_gpu).
    """
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"frame_{i:06d}.png" for i in range(len(images))]
    executable = Path(__file__).resolve().parent.parent / "FidelityFX_CLI.exe"
    prefix = [] if os.name == "nt" else ["wine"]
    batches = [(images[s:s + 59], targets[s:s + 59]) for s in range(0, len(images), 59)]

    def run_batch(batch) -> None:
        sources, destinations = batch
        pairs = [str(p.resolve()) for pair in zip(sources, destinations) for p in pair]
        _run([*prefix, str(executable), "-Mode", "CAS", "-Sharpness", str(sharpness), *pairs])

    with ThreadPoolExecutor(max_workers=max(1, min(len(batches), 3))) as pool:
        list(pool.map(run_batch, batches))

    # Cheap verification: every output exists, and the first one matches its source.
    missing = [t for t in targets if not t.exists() or t.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"CAS did not produce an output image: {missing[0]}")
    src, dst = cv2.imread(str(images[0])), cv2.imread(str(targets[0]))
    if src is None or dst is None or src.shape != dst.shape:
        raise RuntimeError(f"CAS did not produce a matching image: {targets[0]}")
    return targets


def _blur7(x: torch.Tensor, sigma: float, r: int = 3, mode: str = "reflect") -> torch.Tensor:
    """Separable (2r+1)-tap Gaussian; r=3 matches cv2.GaussianBlur(ksize=(0,0)) at sigma=1."""
    axis = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-axis ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode=mode)
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _cas(x: torch.Tensor, sharpness: float, linear: bool = False) -> torch.Tensor:
    """AMD FidelityFX CAS on (N, 3, H, W) in [0, 1].

    `linear=True` sharpens in linear light like FidelityFX_CLI.exe does (within
    0.2-0.45/255 mean error of its output for sharpness 0.2-1.0 on SR targets).
    """
    if linear:
        x = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        y = _cas(x, sharpness)
        return torch.where(y <= 0.0031308, y * 12.92, 1.055 * y.clamp_min(0) ** (1 / 2.4) - 0.055).clamp(0, 1)
    H, W = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")

    def nb(dy: int, dx: int) -> torch.Tensor:
        return p[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

    a, b, c = nb(-1, -1), nb(-1, 0), nb(-1, 1)
    d, e, f = nb(0, -1), x, nb(0, 1)
    g, h, i = nb(1, -1), nb(1, 0), nb(1, 1)
    mn = torch.minimum(torch.minimum(torch.minimum(d, e), torch.minimum(f, b)), h)
    mn = mn + torch.minimum(mn, torch.minimum(torch.minimum(a, c), torch.minimum(g, i)))
    mx = torch.maximum(torch.maximum(torch.maximum(d, e), torch.maximum(f, b)), h)
    mx = mx + torch.maximum(mx, torch.maximum(torch.maximum(a, c), torch.maximum(g, i)))
    amp = (torch.minimum(mn, 2 - mx) / mx.clamp_min(1e-6)).clamp(0, 1).sqrt()
    w = amp * (-1.0 / (8.0 - 3.0 * sharpness))
    return (((b + d + f + h) * w + e) / (1 + 4 * w)).clamp(0, 1)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of (N, H, W, 3) images in [0, 1] (11-tap Gaussian, sigma 1.5)."""
    x, y = x.float().permute(0, 3, 1, 2), y.float().permute(0, 3, 1, 2)
    blur = lambda t: _blur7(t, 1.5, 5, "replicate")
    mx, my = blur(x), blur(y)
    vx, vy, cov = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cov + c2))
            / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


def _scene_scale(viewmats: torch.Tensor) -> float:
    """3DGS-style extent: 1.1x the largest camera distance from the camera centroid."""
    R, t = viewmats[:, :3, :3], viewmats[:, :3, 3:]
    centers = -(R.transpose(1, 2) @ t).squeeze(-1)
    radius = float((centers - centers.mean(dim=0)).norm(dim=-1).max())
    return 1.1 * radius if radius > 0 else 1.0


def _read_rgb_uint8(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read reconstructed frame: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb)


def _load_targets_parallel(images: list[Path]) -> torch.Tensor:
    with ThreadPoolExecutor() as pool:
        frames = list(pool.map(_read_rgb_uint8, images))
        return torch.stack(frames)


def _enhance(raw: torch.Tensor, brightness: float, contrast: float, unsharp: bool,
             sharpness: float, unsharp_amount: float = 1.2, linear_cas: bool = False) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> the same, on raw's device: brightness/contrast -> unsharp -> CAS."""
    x = raw.permute(0, 3, 1, 2).float()
    if contrast != 1.0 or brightness != 0.0:
        x = (x * contrast + brightness).clamp_(0, 255)
    if unsharp:
        # Same math as cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        x = (x * (1 + unsharp_amount) - _blur7(x, 1.0) * unsharp_amount).clamp_(0, 255)
    x = _cas(x / 255.0, sharpness, linear_cas) * 255.0
    return x.round().clamp_(0, 255).byte().permute(0, 2, 3, 1)


def _available_ram() -> int | None:
    try:
        with open('/proc/meminfo') as meminfo:
            for line in meminfo:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@timed('preprocessing')
def build_targets_gpu(images: list[Path], brightness: float, contrast: float,
                      unsharp: bool, sharpness: float, device: str,
                      chunk: int | None = None, unsharp_amount: float = 1.2,
                      chunk_pixels: int = 16_000_000, linear_cas: bool = False,
                      pin: bool | None = None) -> torch.Tensor:
    """Brightness/contrast -> unsharp -> CAS, entirely in memory.

    Returns an (N, H, W, 3) uint8 RGB tensor (pinned on CUDA unless `pin=False`) ready for training.
    Images are streamed in chunks, so peak RAM is one copy of the final targets.
    Unsharp + CAS keep ~7 float copies of a chunk alive on the GPU, so by default
    a chunk holds at most `chunk_pixels` pixels (~1.4 GB peak; 32 frames at 960 px,
    7 at 1920 px) rather than a fixed frame count.
    """
    n = len(images)
    if chunk is None:
        size = _image_size(images[0]) if images else None
        chunk = 32 if size is None else max(1, min(32, chunk_pixels // (size[0] * size[1])))
    out: torch.Tensor | None = None
    for start in range(0, n, chunk):
        raw = _load_targets_parallel(images[start:start + chunk])
        if out is None:
            pinned = device == "cuda" if pin is None else pin
            out = torch.empty((n, *raw.shape[1:]), dtype=torch.uint8, pin_memory=pinned)
        elif raw.shape[1:] != out.shape[1:]:
            raise ValueError("Undistorted images have different dimensions")
        out[start:start + raw.shape[0]] = _enhance(raw.to(device), brightness, contrast, unsharp,
                                                   sharpness, unsharp_amount, linear_cas).cpu()
    return out


# --------------------------------------------------------------------------- #
# Super-resolution targets
# --------------------------------------------------------------------------- #

@timed('preprocessing')
def prepare_sr_targets(images: list[Path], output: Path, width: int, height: int,
                       brightness: float, contrast: float, sharpness: float,
                       unsharp: bool, root: Path, checkpoint: Path, tile: int,
                       device: str) -> tuple[list[Path], dict]:
    """SwinIR 2x targets plus base targets, as keyword arguments for train_splats.

    Base images carry brightness/contrast only; enhanced images get optional unsharp
    and linear-light CAS on the GPU (matching FidelityFX_CLI). Both are kept in RAM
    ({'targets', 'base_targets'}) when they fit in half of the available memory;
    otherwise training reads them per step ({'base_images', 'enhance'}).
    """
    base_dir = output / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    def make_base(item) -> Path:
        i, path = item
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (height, width):
            raise ValueError(f"SR requires matching undistorted image dimensions: {path}")
        frame = cv2.addWeighted(frame, contrast, np.zeros_like(frame), 0, brightness)
        target = base_dir / f"frame_{i:06d}.png"
        if not cv2.imwrite(str(target), frame, _PNG_FAST):
            raise RuntimeError(f"Could not write {target}")
        return target

    with ThreadPoolExecutor() as pool:
        base = list(pool.map(make_base, enumerate(images)))

    manifest = output / "inputs.json"
    manifest.write_text(json.dumps([str(p.resolve()) for p in base]))
    enhanced = output / "enhanced"
    print("Generating tiled SwinIR 2x training targets...", flush=True)
    # A separate process releases the entire SR CUDA context before training starts.
    with stage('super_resolution'):
        _run([sys.executable, "-u", "-m", "src.super_resolution",
              "--root", str(root.resolve()), "--checkpoint", str(checkpoint.resolve()),
              "--manifest", str(manifest.resolve()), "--output", str(enhanced.resolve()),
              "--tile", str(tile), "--device", device])
    targets = [enhanced / p.name for p in base]
    for path in targets[:1]:
        if _image_size(path) != (height * 2, width * 2):
            raise RuntimeError(f"SwinIR did not produce a matching 2x image: {path}")
    needed = len(base) * width * height * 3 * 5  # 2x enhanced (4 px per base px) + base
    available = _available_ram()
    if available is not None and needed > available // 2:
        print(f"SR targets ({needed / 1024 ** 3:.1f} GiB) exceed half of free RAM; "
              "reading them from disk during training.", flush=True)
        enhance = lambda raw: _enhance(raw, 0.0, 1.0, unsharp, sharpness, linear_cas=True)
        return targets, {"base_images": base, "enhance": enhance}
    print("Applying CAS to super-resolution targets on the GPU...", flush=True)
    enhanced_targets = build_targets_gpu(targets, 0.0, 1.0, unsharp, sharpness, device,
                                         linear_cas=True, pin=False)
    if tuple(enhanced_targets.shape[1:3]) != (height * 2, width * 2):
        raise RuntimeError("SwinIR did not produce matching 2x images")
    return targets, {"targets": enhanced_targets, "base_targets": _load_targets_parallel(base)}


def sr_training_loss(rendered: torch.Tensor, enhanced: torch.Tensor,
                     base: torch.Tensor, prior_weight: float) -> torch.Tensor:
    """SRGS-inspired L1 objective; base targets deliberately exclude sharpening."""
    downsampled = F.interpolate(rendered.permute(0, 3, 1, 2).float(),
                               size=base.shape[1:3], mode="area").permute(0, 2, 3, 1)
    return (prior_weight * F.l1_loss(rendered, enhanced)
            + (1 - prior_weight) * F.l1_loss(downsampled, base))


# --------------------------------------------------------------------------- #
# COLMAP
# --------------------------------------------------------------------------- #

def _colmap_key(files: list[Path], params: dict) -> str:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode())
    for f in files:
        st = f.stat()
        digest.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


@timed('colmap')
def run_colmap(
    capture: Path, workdir: Path, vocab_tree: Path | None,
    *, photo_dataset: ImageDataset | None = None,
    image_matching: str = 'auto', spatial_neighbors: int = 12,
    max_features: int = 4096, gp_iterations: int = 50, ba_iterations: int = 3,
    use_cache: bool = True, sequential_overlap: int = 12,
    mapper_tracks_per_view: int | None = None,
) -> Path:
    if sequential_overlap < 1:
        raise ValueError('Sequential overlap must be >= 1')
    files = _list_images(capture)
    num_images = len(files)
    undistorted = workdir / "undistorted"

    # ---- cache: skip the whole SfM when inputs and settings are unchanged ----
    key = _colmap_key(files, {
        "video": photo_dataset is None,
        "sequential_overlap": sequential_overlap,
        "quadratic_overlap": 0,
        "mapper_tracks_per_view": mapper_tracks_per_view,
        "matching": image_matching,
        "neighbors": spatial_neighbors,
        "max_features": max_features,
        "gp_iterations": gp_iterations,
        "ba_iterations": ba_iterations,
        "vocab_tree": (str(vocab_tree), vocab_tree.stat().st_size) if vocab_tree and vocab_tree.exists() else None,
        "single_camera": getattr(photo_dataset, "single_camera", None),
        "max_image_size": getattr(photo_dataset, "max_image_size", None),
        "camera_params": getattr(photo_dataset, "camera_params", None),
        "spatial": getattr(photo_dataset, "spatial", None),
    })
    marker = undistorted / ".cache_key"
    if (use_cache and marker.exists() and marker.read_text() == key
            and (undistorted / "sparse").exists() and (undistorted / "images").exists()):
        print("Reusing cached COLMAP reconstruction (inputs and settings unchanged).", flush=True)
        return undistorted

    # Stale artefacts would be silently merged into the new run.
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in [*workdir.glob("database.db*"), workdir / "sparse", undistorted]:
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    # ---- feature extraction ----
    feature_command = [
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--ImageReader.single_camera", str(int(photo_dataset.single_camera)) if photo_dataset else "1",
        "--FeatureExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", str(max_features),
    ]
    if photo_dataset:
        feature_command += ['--FeatureExtraction.max_image_size', str(photo_dataset.max_image_size)]
        if photo_dataset.camera_params:
            feature_command += ['--ImageReader.camera_model', 'SIMPLE_RADIAL',
                                '--ImageReader.camera_params', photo_dataset.camera_params]
    _run(feature_command)

    # ---- matching ----
    spatial = photo_dataset is not None and photo_dataset.spatial and image_matching == 'auto'
    if photo_dataset is not None and image_matching == 'exhaustive':
        print('Matching photos exhaustively (exhaustive mode selected)...', flush=True)
        _run(['colmap', 'exhaustive_matcher', '--database_path', str(database)])
    elif spatial:
        print(f'Matching GPS neighbors ({spatial_neighbors} per image) plus a short sequential window...', flush=True)
        # GPS neighbors carry the load; the sequential window is only a safety net.
        _run(['colmap', 'sequential_matcher', '--database_path', str(database),
              '--SequentialMatching.overlap', '4'])
        _run(['colmap', 'spatial_matcher', '--database_path', str(database),
              '--SpatialMatching.max_num_neighbors', str(spatial_neighbors),
              '--SpatialMatching.min_num_neighbors', str(min(5, spatial_neighbors)),
              '--SpatialMatching.max_distance', '100',
              '--SpatialMatching.ignore_z', '1'])
    elif photo_dataset is not None:
        if vocab_tree is not None:
            print('No GPS: matching with vocabulary-tree retrieval...', flush=True)
            _run(['colmap', 'vocab_tree_matcher', '--database_path', str(database),
                  '--VocabTreeMatching.vocab_tree_path', str(vocab_tree),
                  '--VocabTreeMatching.num_images', str(min(30, max(num_images - 1, 1)))])
        elif num_images <= 150:
            print('No GPS and no vocab tree: matching exhaustively (small set)...', flush=True)
            _run(['colmap', 'exhaustive_matcher', '--database_path', str(database)])
        else:
            print(f'WARNING: {num_images} photos without GPS and no --vocab-tree; falling back to '
                  'sequential matching (assumes filename order follows capture order). '
                  'Pass --vocab-tree for order-independent matching.', flush=True)
            _run(['colmap', 'sequential_matcher', '--database_path', str(database),
                  '--SequentialMatching.overlap', '20'])
    else:
        match_command = ['colmap', 'sequential_matcher', '--database_path', str(database),
                         '--SequentialMatching.overlap', str(sequential_overlap),
                         # Quadratic mode replaces the window with gaps 1, 2, 4, 8, ...; a dense
                         # window verifies ~2x more pairs and roughly halves global_mapper time.
                         # Loop detection still supplies the long-range pairs.
                         '--SequentialMatching.quadratic_overlap', '0']
        if vocab_tree is not None:
            match_command += ['--SequentialMatching.loop_detection', '1',
                              '--SequentialMatching.vocab_tree_path', str(vocab_tree)]
        _run(match_command)

    # ---- mapping ----
    _run([
        "colmap", "view_graph_calibrator",
        "--database_path", str(database),
    ])

    mapper_command = [
        "colmap", "global_mapper",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--output_path", str(sparse),
        "--GlobalMapper.ba_ceres_max_num_iterations", "50",
        "--GlobalMapper.ba_num_iterations", str(ba_iterations),
        "--GlobalMapper.gp_max_num_iterations", str(gp_iterations),
        "--GlobalMapper.ba_refine_focal_length", "1",
        "--GlobalMapper.ba_refine_extra_params", "1",
    ]
    if mapper_tracks_per_view:
        mapper_command += ["--GlobalMapper.track_required_tracks_per_view", str(mapper_tracks_per_view)]
    _run(mapper_command)

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError(
            "COLMAP found no valid reconstruction; use a video with more overlap."
        )
    # Prefer the model that registered the most images.
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size if (p / "images.bin").exists() else 0)

    undistort_command = [
        "colmap", "image_undistorter",
        "--image_path", str(capture),
        "--input_path", str(best),
        "--output_path", str(undistorted),
        "--output_type", "COLMAP",
    ]
    if photo_dataset:
        undistort_command += ['--max_image_size', str(photo_dataset.max_image_size)]
    _run(undistort_command)
    marker.write_text(key)
    return undistorted


# --------------------------------------------------------------------------- #
# Reconstruction loading
# --------------------------------------------------------------------------- #

def _subsample(means: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if means.shape[0] <= max_points:
        return means, colors
    keep = np.random.choice(means.shape[0], max_points, replace=False)
    return means[keep], colors[keep]


@timed('reconstruction_loading')
def load_reconstruction(
    undistorted: Path, max_points: int
) -> tuple[dict[str, torch.Tensor], list[Path], int, int]:
    reconstruction = pycolmap.Reconstruction(str(undistorted / "sparse"))
    points = list(reconstruction.points3D.values())
    if not points:
        raise RuntimeError("COLMAP produced no sparse points.")

    means = np.stack([np.asarray(p.xyz, dtype=np.float32) for p in points])
    colors = np.stack([np.asarray(p.color, dtype=np.float32) / 255.0 for p in points])
    means, colors = _subsample(means, colors, max_points)

    image_dir = undistorted / "images"
    viewmats: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    images: list[Path] = []
    width = height = 0

    for image in reconstruction.images.values():
        frame = image_dir / image.name
        if not frame.exists():
            continue

        size = _image_size(frame)
        if size is None:
            raise RuntimeError(f'Could not read undistorted image: {frame}')
        if width == 0:
            height, width = size
        elif size != (height, width):
            raise ValueError('Undistorted images have different dimensions; use a consistent camera dataset')

        mat = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
        viewmats.append(mat)

        camera = reconstruction.cameras[image.camera_id]
        intrinsics.append(np.asarray(camera.calibration_matrix(), dtype=np.float32))
        images.append(frame)

    if not images:
        raise RuntimeError("No undistorted images matched the reconstruction.")

    data = {
        "means": torch.from_numpy(means),
        "colors": torch.from_numpy(colors),
        "viewmats": torch.from_numpy(np.stack(viewmats)),
        "Ks": torch.from_numpy(np.stack(intrinsics)),
    }
    return data, images, width, height


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _means_lr(step: int, steps: int, lr_init: float, lr_final_ratio: float) -> float:
    t = min(step / max(steps - 1, 1), 1.0)
    return lr_init * (lr_final_ratio ** t)


def _scale_regularizer(log_scales: torch.Tensor, max_ratio: float) -> torch.Tensor:
    """Soft anisotropy penalty; no hard deletion of thin surfaces."""
    spread = log_scales.amax(dim=-1) - log_scales.amin(dim=-1)
    return (spread.exp() - max_ratio).clamp_min(0).mean()


def _reset_opacities(
    opacities: torch.Tensor, optimizer: torch.optim.Optimizer, value: float = 0.01
) -> None:
    inv_sigmoid = float(np.log(value / (1 - value)))
    with torch.no_grad():
        opacities.clamp_(max=inv_sigmoid)
        for group in optimizer.param_groups:
            if group.get("name") == "opacities":
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state:
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()


@timed('training')
def train_splats(
    data: dict[str, torch.Tensor],
    images: list[Path],
    width: int,
    height: int,
    steps: int,
    device: str,
    view_batch_size: int,
    voxel_guided: bool = True,
    voxel_config: VoxelGuidedConfig | None = None,
    means_lr_init: float = 1.6e-4,
    means_lr_final_ratio: float = 0.01,
    opacity_reset_interval: int = 3000,
    mixed_precision: bool = True,
    preview: TrainingPreview | PreviewPublisher | None = None,
    base_images: list[Path] | None = None,
    sr_prior_weight: float = 0.5,
    targets: torch.Tensor | None = None,
    voxel_stride: int = 1,
    preview_interval: float = 2.0,
    ssim_weight: float = 0.2,
    scale_means_lr: bool = True,
    base_targets: torch.Tensor | None = None,
    enhance=None,
) -> dict[str, torch.Tensor]:
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if view_batch_size < 1:
        raise ValueError("--view-batch-size must be at least 1")
    if voxel_stride < 1:
        raise ValueError("--voxel-stride must be at least 1")
    if not 0 <= ssim_weight <= 1:
        raise ValueError("--ssim-weight must be between 0 and 1")
    quality_config = voxel_config or VoxelGuidedConfig()
    if quality_config.max_axis_ratio < 1 or quality_config.scale_regularization < 0:
        raise ValueError("Scale ratio must be >= 1 and regularization must be >= 0")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat training requires CUDA"
        )
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if not 0 <= sr_prior_weight <= 1:
        raise ValueError("SR prior weight must be between 0 and 1")
    if base_images is not None and len(base_images) != len(images):
        raise ValueError("SR base and enhanced targets must have matching view counts")
    if base_targets is not None and (targets is None or len(base_targets) != len(targets)
                                     or tuple(base_targets.shape[1:3]) != (height // 2, width // 2)):
        raise ValueError("SR base targets must match the enhanced targets and half the render size")

    # Targets: prebuilt in-memory tensors (standard path, or SR with base_targets)
    # > on-disk SR images read a few steps ahead > load now.
    if targets is not None:
        if tuple(targets.shape[1:3]) != (height, width):
            raise ValueError("Prebuilt targets do not match the render cameras")
        target_all = targets
    elif base_images is not None:
        target_all = None  # SR images stay on disk; only the current batch is loaded.
    else:
        target_all = _load_targets_parallel(images)
    # In-memory SR targets are several GiB; they are copied per view instead of pinned.
    if target_all is not None and device == "cuda" and not target_all.is_pinned() and base_targets is None:
        target_all = target_all.pin_memory()

    means = data["means"].to(device).requires_grad_()
    colors_init = data["colors"].numpy()
    colors_init = np.clip(colors_init, 1e-4, 1 - 1e-4)
    colors_logits = np.log(colors_init / (1 - colors_init)).astype(np.float32)
    colors = torch.from_numpy(colors_logits).to(device).requires_grad_()

    from scipy.spatial import cKDTree

    means_np = data["means"].detach().cpu().numpy()
    bounds_np, axes_np = estimate_oriented_box(means_np)
    clip_bounds = torch.from_numpy(bounds_np)
    clip_axes = torch.from_numpy(axes_np)
    tree = cKDTree(means_np)
    if len(means_np) < 2:
        raise ValueError("At least two reconstructed points are required")
    dists, _ = tree.query(means_np, k=min(4, len(means_np)))
    mean_nn_dist = np.clip(dists[:, 1:].mean(axis=1), 1e-6, None)
    init_scale = np.log(mean_nn_dist).astype(np.float32)
    scales = torch.from_numpy(init_scale).to(device).unsqueeze(-1).repeat(1, 3)
    scales = scales.detach().requires_grad_()

    quats = torch.zeros((means.shape[0], 4), device=device)
    quats[:, 0] = 1
    quats.requires_grad_()
    opacities = torch.full((means.shape[0],), 0.0, device=device, requires_grad=True)

    viewmats_all = data["viewmats"].to(device)
    Ks_all = data["Ks"].to(device)
    if scale_means_lr:
        # Positions are learned in scene units; COLMAP's scale is arbitrary.
        means_lr_init = means_lr_init * _scene_scale(data["viewmats"])
    num_views = len(images) if target_all is None else target_all.shape[0]
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": means_lr_init, "name": "means"},
        {"params": [colors], "lr": 2.5e-3, "name": "colors"},
        {"params": [scales], "lr": 5e-3, "name": "scales"},
        {"params": [quats], "lr": 1e-3, "name": "quats"},
        {"params": [opacities], "lr": 5e-2, "name": "opacities"},
    ])
    # An isolated SfM outlier must not set the scale ceiling for the whole scene.
    max_log_scale = float(np.log(np.quantile(mean_nn_dist, 0.95) * 20))

    voxel_opt = None
    if voxel_guided:
        voxel_opt = VoxelGuidedOptimizer(
            means.detach(), quality_config, device
        )

    use_amp = mixed_precision and device == "cuda"

    last_preview = 0.0
    last_percent = -1

    def report(completed: int) -> None:
        nonlocal last_preview, last_percent
        percent = int(100 * completed / steps)
        if percent != last_percent:
            print(f"\rTraining splats: {100 * completed / steps:5.1f}% ({completed}/{steps})",
                  end="\n" if completed == steps else "", flush=True)
            last_percent = percent
        if preview is None or preview.closed:
            return
        now = monotonic()
        if completed == 0 or completed == steps or now - last_preview >= preview_interval:
            with torch.no_grad():
                snapshot = {
                    "clip_bounds": clip_bounds,
                    "clip_axes": clip_axes,
                    "means": means.detach().cpu().clone(),
                    "colors": colors.detach().sigmoid().cpu(),
                    "scales": scales.detach().clamp(max=max_log_scale).exp().cpu(),
                    "quats": F.normalize(quats.detach(), dim=-1, eps=1e-8).cpu(),
                    "opacities": opacities.detach().sigmoid().cpu(),
                }
            preview.publish(snapshot, completed, steps)
            last_preview = now

    report(0)

    recovery_steps = max(quality_config.prune_interval, (num_views + view_batch_size - 1) // view_batch_size)
    # Permutations live on the CPU so indexing host-side targets never forces a GPU sync.
    def view_batches():
        while True:
            perm = torch.randperm(num_views)
            for start in range(0, num_views, view_batch_size):
                yield perm[start:start + view_batch_size]

    batches = view_batches()

    def load_views(idx: torch.Tensor):
        selected = idx.tolist()
        return (_load_targets_parallel([images[i] for i in selected]),
                _load_targets_parallel([base_images[i] for i in selected]))

    # On-disk SR targets: decode the next few steps' views in threads while this step
    # trains (one 2x PNG takes longer to decode than a training step).
    ahead = 3
    reader = ThreadPoolExecutor(max_workers=ahead) if target_all is None else None
    upcoming = deque()
    if reader is not None:
        for _ in range(min(ahead, steps)):
            view_idx = next(batches)
            upcoming.append((view_idx, reader.submit(load_views, view_idx)))
    for step in range(steps):
        if reader is None:
            idx = next(batches)
        else:
            idx, loading = upcoming.popleft()
            raw, base_raw = loading.result()
            if step + ahead < steps:
                view_idx = next(batches)
                upcoming.append((view_idx, reader.submit(load_views, view_idx)))
        idx_dev = idx.to(device, non_blocking=True)

        if target_all is not None:
            target = target_all[idx].to(device, non_blocking=True).float() / 255.0
            base_target = (None if base_targets is None else
                           base_targets[idx].to(device, non_blocking=True).float() / 255.0)
        else:
            raw = raw.to(device)
            target = (raw if enhance is None else enhance(raw)).float() / 255.0
            base_target = base_raw.to(device).float() / 255.0
            if target.shape[1:3] != (height, width) or base_target.shape[1:3] != (height // 2, width // 2):
                raise ValueError("SR target dimensions do not match the render cameras")
        viewmats = viewmats_all[idx_dev]
        Ks = Ks_all[idx_dev]

        for group in optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = _means_lr(step, steps, means_lr_init, means_lr_final_ratio)

        optimizer.zero_grad(set_to_none=True)
        quats_n = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scales_c = scales.clamp(max=max_log_scale)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            rendered, _, render_info = rasterization(
                means, quats_n, scales_c.exp(), opacities.sigmoid(), colors.sigmoid(),
                viewmats, Ks, width, height, packed=True,
            )
            loss = (torch.abs(rendered - target).mean() if base_target is None else
                    sr_training_loss(rendered, target, base_target, sr_prior_weight))
        if ssim_weight and base_target is None:
            # Half resolution keeps SSIM ~4x cheaper (full-res roughly doubles step time);
            # it runs outside autocast because it subtracts near-equal moments.
            half = lambda t: F.avg_pool2d(t.float().permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            loss = (1 - ssim_weight) * loss + ssim_weight * (1 - ssim(half(rendered), half(target)))
        if quality_config.scale_regularization:
            loss = loss + quality_config.scale_regularization * _scale_regularizer(
                scales, quality_config.max_axis_ratio
            )
        loss.backward()

        if voxel_opt is not None:
            # Visibility bookkeeping can be sampled every `voxel_stride` steps
            # (default 1 = every step, i.e. unchanged behaviour).
            if step % voxel_stride == 0:
                voxel_opt.record_visible_views(
                    render_info["gaussian_ids"], render_info["camera_ids"], idx_dev
                )
                voxel_opt.accumulate_step(means)
            voxel_opt.dampen_gradients(means, scales, colors, quats, opacities)

        optimizer.step()

        if voxel_opt is not None:
            means, colors, scales, quats, opacities = voxel_opt.maybe_densify_and_prune(
                step, optimizer, means, colors, scales, quats, opacities,
            )

        if (
            opacity_reset_interval
            and step > 0
            and step % opacity_reset_interval == 0
            and step + recovery_steps < steps - 1
        ):
            _reset_opacities(opacities, optimizer)
            if voxel_opt is not None:
                voxel_opt.pause_after_reset(step, recovery_steps)

        report(step + 1)

    if reader is not None:
        reader.shutdown()

    with torch.no_grad():
        quats.copy_(quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        scales.copy_(scales.clamp(max=max_log_scale))

    result = {
        "clip_bounds": clip_bounds,
        "clip_axes": clip_axes,
        "means": means.detach().cpu(),
        "colors": colors.detach().sigmoid().cpu(),
        "scales": scales.detach().cpu(),
        "quats": quats.detach().cpu(),
        "opacities": opacities.detach().cpu(),
    }
    print("Fitting final bounding box around the dense subject...", flush=True)
    final_bounds, final_axes = fit_subject_box(result["means"].numpy())
    result.update(clip_bounds=torch.from_numpy(final_bounds),
                  clip_axes=torch.from_numpy(final_axes), clip_final=torch.tensor(True))
    if preview is not None and not preview.closed:
        preview.publish({**result, "scales": result["scales"].exp(),
                         "opacities": result["opacities"].sigmoid()}, steps, steps)
    return result


@torch.no_grad()
def evaluate_views(result: dict[str, torch.Tensor], viewmats: torch.Tensor, Ks: torch.Tensor,
                   targets: torch.Tensor, width: int, height: int, device: str) -> dict[str, float]:
    """PSNR/SSIM of the trained splats on views that were held out of training."""
    params = (result["means"].to(device), result["quats"].to(device),
              result["scales"].to(device).exp(), result["opacities"].to(device).sigmoid(),
              result["colors"].to(device))
    psnr, similarity = [], []
    for i in range(targets.shape[0]):
        rendered, _, _ = rasterization(*params, viewmats[i:i + 1].to(device), Ks[i:i + 1].to(device),
                                       width, height, packed=True)
        rendered = rendered.clamp(0, 1)
        target = targets[i:i + 1].to(device).float() / 255.0
        psnr.append(float(-10 * torch.log10(F.mse_loss(rendered, target).clamp_min(1e-10))))
        similarity.append(float(ssim(rendered, target)))
    return {"eval_views": len(psnr), "eval_psnr": float(np.mean(psnr)), "eval_ssim": float(np.mean(similarity))}


@timed('export')
def export_gltf(result: dict[str, torch.Tensor], path: Path) -> None:
    means = result["means"].numpy().astype(np.float32)
    colors = result["colors"].numpy().astype(np.float32)
    scales = np.exp(result["scales"].numpy().astype(np.float32))
    quats = result["quats"].numpy().astype(np.float32)
    opacities = torch.sigmoid(result["opacities"]).numpy().astype(np.float32)

    write_gsplat_glb(path, means, scales, quats, opacities, colors,
                     clip_bounds=result.get("clip_bounds"), clip_axes=result.get("clip_axes"),
                     clip_final=bool(result.get("clip_final", False)))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@benchmark_run
def build_model(
    video: Path,
    output: Path,
    every: int = 5,
    max_width: int = 1920,
    steps: int = 3500,
    device: str = "cuda",
    view_batch_size: int | None = None,
    max_points: int = 150_000,
    vocab_tree: Path | None = None,
    voxel_guided: bool = True,
    voxel_config: VoxelGuidedConfig | None = None,
    means_lr_init: float = 1.6e-4,
    means_lr_final_ratio: float = 0.01,
    opacity_reset_interval: int = 3000,
    mixed_precision: bool = True,
    brightness: float = 0.0,
    contrast: float = 1.0,
    sharpness: float = 0.5,
    super_resolution: bool = False,
    swinir_root: Path | None = None,
    sr_checkpoint: Path | None = None,
    sr_tile: int = 128,
    sr_prior_weight: float = 0.5,
    unsharp: bool | None = None,
    use_server: bool = False,
    image_matching: str = 'auto',
    spatial_neighbors: int = 12,
    photo_every: int = 2,
    max_features: int = 4096,
    headless: bool = False,
    colmap_cache: bool = True,
    gp_iterations: int = 50,
    ba_iterations: int = 3,
    voxel_stride: int = 1,
    keyframe_max_gap: int = 6,
    keyframe_motion: float = .06,
    sequential_overlap: int = 12,
    mapper_tracks_per_view: int | None = 1000,
    eval_every: int = 0,
    ssim_weight: float = 0.2,
    scale_means_lr: bool = True,
) -> Path:
    if mapper_tracks_per_view is not None and mapper_tracks_per_view < 0:
        raise ValueError('--mapper-tracks-per-view must be >= 0')
    if eval_every < 0 or eval_every == 1:
        raise ValueError('--eval-every must be 0 (off) or >= 2')
    if eval_every and super_resolution:
        raise ValueError('--eval-every is not supported with super-resolution')
    if keyframe_max_gap < 1 or not np.isfinite(keyframe_motion) or not 0 < keyframe_motion < 1 or sequential_overlap < 1:
        raise ValueError('Invalid keyframe or sequential overlap settings')
    if image_matching not in ('auto', 'exhaustive') or spatial_neighbors < 1:
        raise ValueError('Invalid image matching mode or spatial neighbor count')
    if photo_every < 1 or max_features < 256:
        raise ValueError('--photo-every must be >= 1 and --max-features >= 256')
    if not np.isfinite([brightness, contrast, sharpness, sr_prior_weight]).all():
        raise ValueError("Image controls and SR weight must be finite")
    if not 0 <= sharpness <= 1 or not 0 <= sr_prior_weight <= 1:
        raise ValueError("Sharpness and SR prior weight must be between 0 and 1")
    if device == "cuda" and not torch.cuda.is_available():
        # Fail now rather than after 10+ minutes of COLMAP.
        raise RuntimeError("gsplat training requires CUDA")
    if super_resolution:
        from src.super_resolution import validate_setup
        validate_setup(swinir_root, sr_checkpoint, sr_tile)
    if view_batch_size is None:
        view_batch_size = 1 if super_resolution else 4
    if unsharp is None:
        unsharp = not super_resolution
    colmap_options = dict(
        max_features=max_features, gp_iterations=gp_iterations,
        ba_iterations=ba_iterations, use_cache=colmap_cache, sequential_overlap=sequential_overlap,
        mapper_tracks_per_view=mapper_tracks_per_view,
    )
    photo_dataset = None
    if video.is_dir():
        print('Importing photos and metadata...', flush=True)
        with stage('input_preparation'):
            photo_dataset = prepare_image_dataset(video, output, max_width)
            capture = photo_dataset.capture
            if photo_every > 1:
                capture = _make_photo_subset(photo_dataset.capture, output, photo_every)
        frame_counts(frames_before_colmap=len(_list_images(capture)))
        undistorted = run_colmap(capture, output / 'colmap', vocab_tree,
                                photo_dataset=photo_dataset, image_matching=image_matching,
                                spatial_neighbors=spatial_neighbors, **colmap_options)
    else:
        print("Extracting video frames...", flush=True)
        capture_dir = output / "capture" / "originals"
        capture = extract_frames(video, capture_dir, every, max_width, brightness, contrast,
                                 sharpness, original_only=True, unsharp=False)
        frame_counts(keyframe_candidates=len(_list_images(capture)))
        with stage('keyframe_selection'):
            candidates = _list_images(capture)
            selection_key = _colmap_key(candidates, {'gap': keyframe_max_gap, 'motion': keyframe_motion})
            capture = select_keyframes(candidates, output / 'capture' / 'keyframes' / selection_key,
                                       max_gap=keyframe_max_gap, motion_threshold=keyframe_motion)
        frame_counts(frames_before_colmap=len(_list_images(capture)))
        undistorted = run_colmap(capture, output / "colmap", vocab_tree, **colmap_options)
    print("Loading reconstructions...", flush=True)
    data, images, width, height = load_reconstruction(undistorted, max_points)
    frame_counts(frames_after_colmap=len(images))
    sr_inputs = {}
    targets = None
    if super_resolution:
        images, sr_inputs = prepare_sr_targets(
            images, output / "sr", width, height, brightness, contrast, sharpness,
            unsharp, swinir_root, sr_checkpoint, sr_tile, device)
        targets = sr_inputs.pop("targets", None)
        data["Ks"] = data["Ks"].clone()
        data["Ks"][:, :2, :] *= 2
        width, height = width * 2, height * 2
    else:
        # Keep originals and metadata intact for SfM; enhance only training targets.
        print('Preprocessing registered images on the GPU (brightness/contrast, unsharp, CAS)...', flush=True)
        targets = build_targets_gpu(images, brightness, contrast, unsharp, sharpness, device)
    held_out = None
    if eval_every:
        # Mip-NeRF 360 convention: every Nth registered view is a test view.
        test = torch.arange(len(images)) % eval_every == 0
        held_out = (data["viewmats"][test], data["Ks"][test], targets[test])
        data = {**data, "viewmats": data["viewmats"][~test], "Ks": data["Ks"][~test]}
        images = [image for image, t in zip(images, test.tolist()) if not t]
        targets = targets[~test]
    print("Training splats...")
    glb_path = output / "model.glb"
    if headless:
        preview = None
    elif use_server:
        preview = PreviewPublisher(Path(__file__).resolve().parent.parent / "runs" / ".viewer")
    else:
        preview = TrainingPreview()
    viewer_thread = None
    try:
        if preview is not None:
            if use_server:
                with stage('viewer_startup'):
                    ensure_viewer(preview.directory, width, height)
            else:
                # Keep direct engine runs self-contained: the desktop window consumes
                # the same live snapshots while training continues in this thread.
                import threading
                viewer_thread = threading.Thread(
                    target=start_viewer,
                    args=(None, width, height),
                    kwargs={"preview": preview, "desktop": True},
                    daemon=True,
                )
                with stage('viewer_startup'):
                    viewer_thread.start()
        result = train_splats(
            data, images, width, height, steps, device, view_batch_size,
            voxel_guided, voxel_config,
            means_lr_init, means_lr_final_ratio,
            opacity_reset_interval, mixed_precision, preview=preview,
            sr_prior_weight=sr_prior_weight, targets=targets, voxel_stride=voxel_stride,
            **sr_inputs,
            ssim_weight=ssim_weight, scale_means_lr=scale_means_lr,
        )
        if held_out is not None:
            quality = evaluate_views(result, *held_out, width, height, device)
            metrics(**quality)
            print(f"Held-out views: {quality['eval_views']}, PSNR {quality['eval_psnr']:.2f} dB, "
                  f"SSIM {quality['eval_ssim']:.4f}", flush=True)
        print("Exporting model...", flush=True)
        export_gltf(result, glb_path)
        if preview is not None:
            preview.finish()
        print(f"Model saved: {glb_path}", flush=True)
    except Exception as exc:
        if preview is not None:
            preview.finish(exc)
        raise
    finally:
        if preview is not None:
            preview.close()
        if viewer_thread is not None:
            viewer_thread.join(timeout=2)
    return glb_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gaussian splat model from a drone video or image directory."
    )
    parser.add_argument("video", type=Path, metavar='INPUT', help='Video file or directory of photos (non-recursive).')
    parser.add_argument("--output", type=Path, default=Path("runs/gsplat"))
    parser.add_argument("--every", type=int, default=5,
                        help="Keep every Nth video frame (videos only; see --photo-every for photo directories).")
    parser.add_argument('--keyframe-max-gap', type=int, default=6,
                        help='Video: maximum gap in candidates AFTER --every; 1 disables selection (default: 6).')
    parser.add_argument('--keyframe-motion', type=float, default=.06,
                        help='Video: tracked displacement / image diagonal triggering a keyframe (default: .06).')
    parser.add_argument('--sequential-overlap', type=int, default=12,
                        help='Video COLMAP matching overlap (default: 12).')
    parser.add_argument('--mapper-tracks-per-view', type=int, default=1000,
                        help='Tracks global_mapper keeps per image; fewer is faster (default: 1000, 0 = all).')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Hold out every Nth registered view and report PSNR/SSIM (0 = off, 8 is standard).')
    parser.add_argument('--ssim-weight', type=float, default=0.2,
                        help='D-SSIM weight in the training loss (0 = pure L1; ignored with SR).')
    parser.add_argument('--no-scale-means-lr', dest='scale_means_lr', action='store_false',
                        help='Do not scale the position learning rate by the scene extent.')
    parser.add_argument("--photo-every", type=int, default=2,
                        help="Use every Nth photo for SfM/training in image directories "
                             "(default: 2; 1 uses all photos). Matching cost grows ~quadratically with image count.")
    parser.add_argument('--image-matching', choices=('auto', 'exhaustive'), default='auto',
                        help='Photos: GPS + sequential matching when all images have GPS; otherwise vocab-tree '
                             '(if --vocab-tree is given), exhaustive for <=150 photos, or sequential.')
    parser.add_argument('--spatial-neighbors', type=int, default=12,
                        help='GPS nearest neighbors per photo in auto mode (default: 12).')
    parser.add_argument("--max-features", type=int, default=4096,
                        help="Max SIFT features per image (default: 4096).")
    parser.add_argument("--gp-iterations", type=int, default=50,
                        help="global_mapper position-solver iterations (lower = faster).")
    parser.add_argument("--ba-iterations", type=int, default=3,
                        help="global_mapper bundle-adjustment rounds (lower = faster).")
    parser.add_argument("--no-colmap-cache", dest="colmap_cache", action="store_false",
                        help="Always re-run COLMAP even if inputs and settings are unchanged.")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=None,
                        help="Defaults to 1 with SR, otherwise 4.")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--max-gaussians", type=int, default=200_000)
    parser.add_argument("--max-axis-ratio", type=float, default=10.0,
                        help="Axis ratio above which splats receive a soft shape penalty.")
    parser.add_argument("--scale-regularization", type=float, default=0.01,
                        help="Strength of the anti-streak shape penalty; 0 disables it.")
    parser.add_argument("--vocab-tree", type=Path, default="vocab_tree.bin",
                        help="COLMAP vocabulary tree (.bin). Enables retrieval matching for photos "
                             "without GPS and loop detection for videos.")
    parser.add_argument(
        "--no-voxel-guided", dest="voxel_guided", action="store_false",
        help="Disable the DroneSplat-style voxel-guided optimization (floater fix).",
    )
    parser.add_argument(
        "--voxel-n", type=int, default=80,
        help="Divide the scene's shortest bbox edge into this many voxels (paper's N).",
    )
    parser.add_argument(
        "--voxel-tau", type=float, default=3.5,
        help="Voxel-lengths a Gaussian may drift/scale before being flagged unconstrained.",
    )
    parser.add_argument(
        "--voxel-gamma1", type=float, default=1e-4,
        help="Accumulated world-space gradient norm needed to grow into an empty voxel "
        "(scene-dependent - see src/voxel_guided.py's module docstring).",
    )
    parser.add_argument(
        "--voxel-gamma2", type=int, default=2,
        help="Prune sparse voxels only when their average opacity is also below voxel-gamma3.",
    )
    parser.add_argument(
        "--voxel-gamma3", type=float, default=0.075,
        help="Average opacity threshold for pruning sparse voxels.",
    )
    parser.add_argument(
        "--voxel-stride", type=int, default=1,
        help="Record visibility/accumulate voxel statistics every N steps (1 = every step).",
    )
    parser.add_argument(
        "--means-lr", type=float, default=1.6e-4,
        help="Initial learning rate for Gaussian positions.",
    )
    parser.add_argument(
        "--means-lr-final-ratio", type=float, default=0.01,
        help="Means LR is annealed exponentially to (means-lr * this ratio) by the last step.",
    )
    parser.add_argument(
        "--opacity-reset-interval", type=int, default=3000,
        help="Reset all opacities every N steps to flush unearned floaters. 0 disables.",
    )
    parser.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false",
        help="Disable bf16 autocast during rasterization/loss (CUDA only).",
    )
    parser.add_argument(
        "--brightness", type=float, default=0.0,
        help="Added to every pixel in 0-255 units (0 = unchanged).",
    )
    parser.add_argument(
        "--contrast", type=float, default=1.0
    )
    parser.add_argument(
        "--sharpness", type=float, default=0.5
    )
    parser.add_argument("--super-resolution", action="store_true",
                        help="Use tiled SwinIR 2x, CAS and dual-resolution supervision.")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--swinir-root", type=Path,
                        default=default_root / "third_party" / "SwinIR",
                        help="Official SwinIR source checkout (same default as upload_server).")
    parser.add_argument("--sr-checkpoint", type=Path,
                        default=default_root / "weights" / "swinir-lightweight-x2.pth",
                        help="SwinIR-S lightweight 2x checkpoint (same default as upload_server).")
    parser.add_argument("--sr-tile", type=int, default=128, help="Input tile size, multiple of 8, >= 32.")
    parser.add_argument("--sr-prior-weight", type=float, default=0.5,
                        help="Enhanced target weight; remaining weight anchors to base images.")
    parser.add_argument("--unsharp", action=argparse.BooleanOptionalAction, default=None,
                        help="Unsharp masking: defaults off with SR, on otherwise.")
    parser.add_argument("--use-server", action="store_true",
                        help="Use the persistent browser viewer (used by upload_server).")
    parser.add_argument("--headless", action="store_true",
                        help="No live viewer or preview snapshots (fastest; use for benchmarking).")
    args = parser.parse_args()
    voxel_config = VoxelGuidedConfig(
        n_along_shortest=args.voxel_n,
        tau=args.voxel_tau,
        gamma1=args.voxel_gamma1,
        gamma2=args.voxel_gamma2,
        gamma3=args.voxel_gamma3,
        max_gaussians=args.max_gaussians,
        max_axis_ratio=args.max_axis_ratio,
        scale_regularization=args.scale_regularization,
    )
    print(
        build_model(
            args.video,
            args.output,
            args.every,
            args.max_width,
            args.steps,
            args.device,
            args.view_batch_size,
            args.max_points,
            args.vocab_tree,
            args.voxel_guided,
            voxel_config,
            args.means_lr,
            args.means_lr_final_ratio,
            args.opacity_reset_interval,
            args.mixed_precision,
            args.brightness,
            args.contrast,
            args.sharpness,
            super_resolution=args.super_resolution,
            swinir_root=args.swinir_root, sr_checkpoint=args.sr_checkpoint,
            sr_tile=args.sr_tile, sr_prior_weight=args.sr_prior_weight,
            unsharp=args.unsharp,
            use_server=args.use_server,
            image_matching=args.image_matching, spatial_neighbors=args.spatial_neighbors,
            photo_every=args.photo_every, max_features=args.max_features,
            headless=args.headless, colmap_cache=args.colmap_cache,
            gp_iterations=args.gp_iterations, ba_iterations=args.ba_iterations,
            voxel_stride=args.voxel_stride,
            keyframe_max_gap=args.keyframe_max_gap, keyframe_motion=args.keyframe_motion,
            sequential_overlap=args.sequential_overlap,
            mapper_tracks_per_view=args.mapper_tracks_per_view,
            eval_every=args.eval_every, ssim_weight=args.ssim_weight,
            scale_means_lr=args.scale_means_lr,
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
