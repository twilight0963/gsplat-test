from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import os
import sys
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
from src.image_dataset import ImageDataset, prepare_image_dataset
from src.run_benchmark import benchmark_run, frame_counts, stage, timed

def unsharp_mask(frame, amount=1.5, sigma=1.0, threshold=0):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(frame.astype(int) - blurred.astype(int)) < threshold
        np.copyto(sharpened, frame, where=low_contrast_mask)
    return sharpened

@timed('input_preparation')
def extract_frames(video: Path, output: Path, every: int, max_width: int, brightness: float=1, contrast: float=1, sharpness: float=0.5, *, original_only: bool = False,
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
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        if index % every == 0:
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
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write frame: {path}")
            frames.append(path)
        index += 1
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


@timed('cas')
def apply_cas(images: list[Path], output: Path, sharpness: float) -> list[Path]:
    """Use lossless output and retain the exact input order for camera pairing."""
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"frame_{i:06d}.png" for i in range(len(images))]
    executable = Path(__file__).resolve().parent.parent / "FidelityFX_CLI.exe"
    prefix = [] if os.name == "nt" else ["wine"]
    for start in range(0, len(images), 59):
        pairs = [str(p.resolve()) for pair in zip(images[start:start+59], targets[start:start+59]) for p in pair]
        _run([*prefix, str(executable), "-Mode", "CAS", "-Sharpness", str(sharpness), *pairs])
    for source, target in zip(images, targets):
        src = cv2.imread(str(source))
        dst = cv2.imread(str(target))
        if src is None or dst is None or src.shape != dst.shape:
            raise RuntimeError(f"CAS did not produce a matching image: {target}")
    return targets


@timed('preprocessing')
def prepare_sr_targets(images: list[Path], output: Path, width: int, height: int,
                       brightness: float, contrast: float, sharpness: float,
                       unsharp: bool, root: Path, checkpoint: Path, tile: int,
                       device: str) -> tuple[list[Path], list[Path]]:
    base_dir = output / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base = []
    for i, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (height, width):
            raise ValueError(f"SR requires matching undistorted image dimensions: {path}")
        frame = cv2.addWeighted(frame, contrast, np.zeros_like(frame), 0, brightness)
        target = base_dir / f"frame_{i:06d}.png"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Could not write {target}")
        base.append(target)
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
    for path in targets:
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (height * 2, width * 2):
            raise RuntimeError(f"SwinIR did not produce a matching 2x image: {path}")
        if unsharp:
            if not cv2.imwrite(str(path), unsharp_mask(frame, amount=1.2)):
                raise RuntimeError(f"Could not write {path}")
    print("Applying CAS to super-resolution targets...", flush=True)
    return apply_cas(targets, output / "cas", sharpness), base


def sr_training_loss(rendered: torch.Tensor, enhanced: torch.Tensor,
                     base: torch.Tensor, prior_weight: float) -> torch.Tensor:
    """SRGS-inspired L1 objective; base targets deliberately exclude sharpening."""
    downsampled = F.interpolate(rendered.permute(0, 3, 1, 2).float(),
                               size=base.shape[1:3], mode="area").permute(0, 2, 3, 1)
    return (prior_weight * F.l1_loss(rendered, enhanced)
            + (1 - prior_weight) * F.l1_loss(downsampled, base))


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc


@timed('colmap')
def run_colmap(
    capture: Path, workdir: Path, vocab_tree: Path | None,
    *, photo_dataset: ImageDataset | None = None,
    image_matching: str = 'auto', spatial_neighbors: int = 20,
) -> Path:
    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    feature_command = [
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--ImageReader.single_camera", str(int(photo_dataset.single_camera)) if photo_dataset else "1",
        "--FeatureExtraction.use_gpu", "1"
    ]
    if photo_dataset:
        feature_command += ['--FeatureExtraction.max_image_size', str(photo_dataset.max_image_size)]
        if photo_dataset.camera_params:
            feature_command += ['--ImageReader.camera_model', 'SIMPLE_RADIAL',
                                '--ImageReader.camera_params', photo_dataset.camera_params]
    _run(feature_command)

    spatial = photo_dataset is not None and photo_dataset.spatial and image_matching == 'auto'
    match_command = [
        "colmap", "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", "10" if spatial else "30",
    ]
    if vocab_tree is not None:
        match_command += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", str(vocab_tree),
        ]
    if photo_dataset and not spatial:
        print('Matching photos exhaustively (GPS unavailable or exhaustive mode selected)...', flush=True)
        _run(['colmap', 'exhaustive_matcher', '--database_path', str(database)])
    else:
        _run(match_command)
        if spatial:
            print(f'Matching GPS neighbors ({spatial_neighbors} per image) plus capture-order neighbors...', flush=True)
            _run(['colmap', 'spatial_matcher', '--database_path', str(database),
                  '--SpatialMatching.max_num_neighbors', str(spatial_neighbors),
                  '--SpatialMatching.min_num_neighbors', str(min(5, spatial_neighbors)),
                  '--SpatialMatching.max_distance', '100',
                  '--SpatialMatching.ignore_z', '1'])

    _run([
        "colmap", "view_graph_calibrator",
        "--database_path", str(database),
    ])

    _run([
        "colmap", "global_mapper",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--output_path", str(sparse),
        "--GlobalMapper.ba_ceres_max_num_iterations", "50",
        "--GlobalMapper.ba_num_iterations", "3",
        "--GlobalMapper.gp_max_num_iterations", "50",
        "--GlobalMapper.ba_refine_focal_length", "1",
        "--GlobalMapper.ba_refine_extra_params", "1",
    ])

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError(
            "COLMAP found no valid reconstruction; use a video with more overlap."
        )

    undistorted = workdir / "undistorted"
    undistort_command = [
        "colmap", "image_undistorter",
        "--image_path", str(capture),
        "--input_path", str(models[0]),
        "--output_path", str(undistorted),
        "--output_type", "COLMAP",
    ]
    if photo_dataset:
        undistort_command += ['--max_image_size', str(photo_dataset.max_image_size)]
    _run(undistort_command)
    return undistorted


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

        loaded = cv2.imread(str(frame))
        if loaded is None:
            raise RuntimeError(f'Could not read undistorted image: {frame}')
        if width == 0:
            height, width = loaded.shape[:2]
        elif loaded.shape[:2] != (height, width):
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
) -> dict[str, torch.Tensor]:
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if view_batch_size < 1:
        raise ValueError("--view-batch-size must be at least 1")
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
    # SR images stay on disk; transfer only the current camera batch to the GPU.
    target_all = None if base_images is not None else _load_targets_parallel(images)
    if target_all is not None and device == "cuda":
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
    num_views = len(images) if base_images is not None else target_all.shape[0]
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
        now = monotonic()
        if preview is not None and not preview.closed and (
            completed == 0 or completed == steps or now - last_preview >= 1.0
        ):
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
    perm = torch.randperm(num_views, device=device)
    cursor = 0
    for step in range(steps):
        if cursor >= num_views:
            perm = torch.randperm(num_views, device=device)
            cursor = 0
        idx = perm[cursor : cursor + view_batch_size]
        cursor += view_batch_size

        if base_images is None:
            target = target_all[idx.cpu()].to(device, non_blocking=True).float() / 255.0
            base_target = None
        else:
            selected = idx.cpu().tolist()
            target = _load_targets_parallel([images[i] for i in selected]).to(device).float() / 255.0
            base_target = _load_targets_parallel([base_images[i] for i in selected]).to(device).float() / 255.0
            if target.shape[1:3] != (height, width) or base_target.shape[1:3] != (height // 2, width // 2):
                raise ValueError("SR target dimensions do not match the render cameras")
        viewmats = viewmats_all[idx]
        Ks = Ks_all[idx]

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
        if quality_config.scale_regularization:
            loss = loss + quality_config.scale_regularization * _scale_regularizer(
                scales, quality_config.max_axis_ratio
            )
        loss.backward()

        if voxel_opt is not None:
            voxel_opt.record_visible_views(
                render_info["gaussian_ids"], render_info["camera_ids"], idx
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


@benchmark_run
def build_model(
    video: Path,
    output: Path,
    every: int = 5,
    max_width: int = 1920,
    steps: int = 5000,
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
    brightness: float = 1.0,
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
    spatial_neighbors: int = 20,
) -> Path:
    if image_matching not in ('auto', 'exhaustive') or spatial_neighbors < 1:
        raise ValueError('Invalid image matching mode or spatial neighbor count')
    if not np.isfinite([brightness, contrast, sharpness, sr_prior_weight]).all():
        raise ValueError("Image controls and SR weight must be finite")
    if not 0 <= sharpness <= 1 or not 0 <= sr_prior_weight <= 1:
        raise ValueError("Sharpness and SR prior weight must be between 0 and 1")
    if super_resolution:
        from src.super_resolution import validate_setup
        validate_setup(swinir_root, sr_checkpoint, sr_tile)
    if view_batch_size is None:
        view_batch_size = 1 if super_resolution else 4
    if unsharp is None:
        unsharp = not super_resolution
    photo_dataset = None
    if video.is_dir():
        print('Importing photos and metadata...', flush=True)
        with stage('input_preparation'):
            photo_dataset = prepare_image_dataset(video, output, max_width)
        frame_counts(frames_before_colmap=photo_dataset.count)
        undistorted = run_colmap(photo_dataset.capture, output / 'colmap', vocab_tree,
                                photo_dataset=photo_dataset, image_matching=image_matching,
                                spatial_neighbors=spatial_neighbors)
    else:
        print("Extracting video frames...", flush=True)
        capture_dir = output / "capture" / "originals"
        capture = extract_frames(video, capture_dir, every, max_width, brightness, contrast,
                                 sharpness, original_only=True, unsharp=False)
        undistorted = run_colmap(capture, output / "colmap", vocab_tree)
    print("Loading reconstructions...", flush=True)
    data, images, width, height = load_reconstruction(undistorted, max_points)
    frame_counts(frames_after_colmap=len(images))
    base_images = None
    if super_resolution:
        images, base_images = prepare_sr_targets(
            images, output / "sr", width, height, brightness, contrast, sharpness,
            unsharp, swinir_root, sr_checkpoint, sr_tile, device)
        data["Ks"] = data["Ks"].clone()
        data["Ks"][:, :2, :] *= 2
        width, height = width * 2, height * 2
    else:
        # Keep originals and metadata intact for SfM; enhance only training targets.
        print('Preprocessing registered images (brightness/contrast, unsharp and CAS)...', flush=True)
        target_dir = output / 'training_targets'
        target_dir.mkdir(parents=True, exist_ok=True)
        targets = []
        for i, path in enumerate(images):
            with stage('preprocessing'):
                frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError(f'Could not read image: {path}')
                frame = cv2.addWeighted(frame, contrast, np.zeros_like(frame), 0, brightness)
                if unsharp:
                    frame = unsharp_mask(frame, amount=1.2)
                target = target_dir / f'frame_{i:06d}.png'
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f'Could not write image: {target}')
                targets.append(target)
        images = apply_cas(targets, output / 'training_cas', sharpness)
    print("Training splats...")
    glb_path = output / "model.glb"
    preview = (PreviewPublisher(Path(__file__).resolve().parent.parent / "runs" / ".viewer")
               if use_server else TrainingPreview())
    viewer_thread = None
    try:
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
            base_images=base_images, sr_prior_weight=sr_prior_weight,
        )
        print("Exporting model...", flush=True)
        export_gltf(result, glb_path)
        preview.finish()
        print(f"Model saved: {glb_path}", flush=True)
    except Exception as exc:
        preview.finish(exc)
        raise
    finally:
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
    parser.add_argument("--every", type=int, default=5, help="Keep every Nth video frame; image directories use all photos.")
    parser.add_argument('--image-matching', choices=('auto', 'exhaustive'), default='auto',
                        help='Photos: GPS + sequential matching when all images have GPS, otherwise exhaustive.')
    parser.add_argument('--spatial-neighbors', type=int, default=20,
                        help='GPS nearest neighbors per photo in auto mode (default: 20).')
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=None,
                        help="Defaults to 1 with SR, otherwise 4.")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--max-gaussians", type=int, default=200_000)
    parser.add_argument("--max-axis-ratio", type=float, default=10.0,
                        help="Axis ratio above which splats receive a soft shape penalty.")
    parser.add_argument("--scale-regularization", type=float, default=0.01,
                        help="Strength of the anti-streak shape penalty; 0 disables it.")
    parser.add_argument("--vocab-tree", type=Path, default=None)
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
        "--brightness", type=float, default=1.0
    )
    parser.add_argument(
        "--contrast", type=float, default=1.0
    )
    parser.add_argument(
        "--sharpness",type=float, default=0.5
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
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
