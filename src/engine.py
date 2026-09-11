from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import os
import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from src.gsplat_viewer import TrainingPreview, start_viewer
from src.gltf_gsplat import write_gsplat_glb
from src.voxel_reconstruction import VoxelGuidedConfig, VoxelGuidedOptimizer

def unsharp_mask(frame, amount=1.5, sigma=1.0, threshold=0):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(frame.astype(int) - blurred.astype(int)) < threshold
        np.copyto(sharpened, frame, where=low_contrast_mask)
    return sharpened

def extract_frames(video: Path, output: Path, every: int, max_width: int, brightness: float=1, contrast: float=1, sharpness: float=0.5) -> Path:
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

            frame = cv2.addWeighted(
                frame,
                contrast,
                np.zeros(frame.shape, frame.dtype),
                0,
                brightness
            )
            frame = unsharp_mask(frame, amount=1.2, sigma=1.0)
            path = output / f"frame_{len(frames):06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write frame: {path}")
            frames.append(path)
        index += 1
    reader.release()

    sharp_output_path = (output / "sharpened")
    sharp_output_path.mkdir(parents=True, exist_ok=True)
    src_dst_pairs = []
    output_paths: list[Path] = []
    for frame in frames:
        out_path = sharp_output_path / frame.name.replace(".jpg", "_sharpened.jpg")
        output_paths.append(out_path)
        src_dst_pairs.extend([str(frame), str(out_path)])
    print(src_dst_pairs)
    for i in range(0,len(src_dst_pairs),118):
        if os.name == 'nt':
            _run([
                "FidelityFX_CLI.exe", "-Mode", "CAS", "-Sharpness", str(sharpness),
                *src_dst_pairs[i:i+118]
            ])
        else:
            _run([
                "wine", "FidelityFX_CLI.exe", "-Mode", "CAS", "-Sharpness", str(sharpness),
                *src_dst_pairs[i:i+118]
            ])
    if len(output_paths) < 2:
        raise RuntimeError("The video did not produce at least two usable frames")
    (sharp_output_path / "capture.json").write_text(
        json.dumps({"fps": fps, "frames": [p.name for p in output_paths]}, indent=2) + "\n"
    )
    return sharp_output_path


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("COLMAP is required and must be on PATH.") from exc


def run_colmap(
    capture: Path, workdir: Path, vocab_tree: Path | None,
) -> Path:
    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    _run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1"
    ])

    match_command = [
        "colmap", "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", "30",
    ]
    if vocab_tree is not None:
        match_command += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", str(vocab_tree),
        ]
    _run(match_command)

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
    _run([
        "colmap", "image_undistorter",
        "--image_path", str(capture),
        "--input_path", str(models[0]),
        "--output_path", str(undistorted),
        "--output_type", "COLMAP",
    ])
    return undistorted


def _subsample(means: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if means.shape[0] <= max_points:
        return means, colors
    keep = np.random.choice(means.shape[0], max_points, replace=False)
    return means[keep], colors[keep]


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

        if width == 0:
            loaded = cv2.imread(str(frame))
            if loaded is None:
                continue
            height, width = loaded.shape[:2]

        mat = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
        viewmats.append(mat)

        camera = reconstruction.cameras[image.camera_id]
        params = np.asarray(camera.params, dtype=np.float32)
        fx, fy, cx, cy = params[:4]
        intrinsics.append(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        )
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
    preview: TrainingPreview | None = None,
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

    target_all = _load_targets_parallel(images).pin_memory()

    means = data["means"].to(device).requires_grad_()
    colors_init = data["colors"].numpy()
    colors_init = np.clip(colors_init, 1e-4, 1 - 1e-4)
    colors_logits = np.log(colors_init / (1 - colors_init)).astype(np.float32)
    colors = torch.from_numpy(colors_logits).to(device).requires_grad_()

    from scipy.spatial import cKDTree

    means_np = data["means"].detach().cpu().numpy()
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
    num_views = target_all.shape[0]
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

        target = target_all[idx.cpu()].to(device, non_blocking=True).float() / 255.0
        viewmats = viewmats_all[idx]
        Ks = Ks_all[idx]

        for group in optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = _means_lr(step, steps, means_lr_init, means_lr_final_ratio)

        optimizer.zero_grad(set_to_none=True)
        quats_n = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scales_c = scales.clamp(max=max_log_scale)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            rendered, _, _ = rasterization(
                means, quats_n, scales_c.exp(), opacities.sigmoid(), colors.sigmoid(),
                viewmats, Ks, width, height, packed=True,
            )
            loss = torch.abs(rendered - target).mean()
        if quality_config.scale_regularization:
            loss = loss + quality_config.scale_regularization * _scale_regularizer(
                scales, quality_config.max_axis_ratio
            )
        loss.backward()

        if voxel_opt is not None:
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

    return {
        "means": means.detach().cpu(),
        "colors": colors.detach().sigmoid().cpu(),
        "scales": scales.detach().cpu(),
        "quats": quats.detach().cpu(),
        "opacities": opacities.detach().cpu(),
    }

def export_gltf(result: dict[str, torch.Tensor], path: Path) -> None:
    means = result["means"].numpy().astype(np.float32)
    colors = result["colors"].numpy().astype(np.float32)
    scales = np.exp(result["scales"].numpy().astype(np.float32))
    quats = result["quats"].numpy().astype(np.float32)
    opacities = torch.sigmoid(result["opacities"]).numpy().astype(np.float32)

    write_gsplat_glb(path, means, scales, quats, opacities, colors)


def build_model(
    video: Path,
    output: Path,
    every: int = 5,
    max_width: int = 1920,
    steps: int = 5000,
    device: str = "cuda",
    view_batch_size: int = 4,
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
    sharpness: float = 0.5
) -> Path:
    capture_dir = output / "capture"
    extract_frames(video, capture_dir, every, max_width, brightness, contrast, sharpness)
    undistorted = run_colmap(capture_dir, output / "colmap", vocab_tree)
    print("Loading reconstructions...")
    data, images, width, height = load_reconstruction(undistorted, max_points)
    print("Training splats...")
    glb_path = output / "model.glb"
    preview = TrainingPreview()

    def train_and_export():
        try:
            result = train_splats(
                data, images, width, height, steps, device, view_batch_size,
                voxel_guided, voxel_config,
                means_lr_init, means_lr_final_ratio,
                opacity_reset_interval, mixed_precision, preview=preview,
            )
            print("Exporting model...")
            export_gltf(result, glb_path)
            print("Finished export.")
        except Exception as exc:
            preview.finish(exc)
            raise
        else:
            preview.finish()

    with ThreadPoolExecutor(max_workers=1) as pool:
        training = pool.submit(train_and_export)
        try:
            start_viewer(None, width, height, preview=preview)
        finally:
            preview.close()
        training.result()
    return glb_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gaussian splat model from a drone video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/gsplat"))
    parser.add_argument("--every", type=int, default=5, help="Keep every Nth video frame.")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=4)
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
            args.sharpness
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
