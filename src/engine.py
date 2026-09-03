from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from src.gsplat_viewer import start_viewer
from src.gltf_gsplat import write_gsplat_glb
from src.voxel_reconstruction import VoxelGuidedConfig, VoxelGuidedOptimizer


def extract_frames(video: Path, output: Path, every: int, max_width: int) -> Path:
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
            path = output / f"frame_{len(frames):06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write frame: {path}")
            frames.append(path)
        index += 1
    reader.release()
    if len(frames) < 2:
        raise RuntimeError("The video did not produce at least two usable frames")
    (output / "capture.json").write_text(
        json.dumps({"fps": fps, "frames": [p.name for p in frames]}, indent=2) + "\n"
    )
    return output


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("COLMAP is required and must be on PATH.") from exc


def run_colmap(
    capture: Path, workdir: Path, vocab_tree: Path | None, use_gpu: bool = True
) -> Path:
    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    gpu_flag = "1" if use_gpu else "0"

    _run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--ImageReader.single_camera", "1",
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


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _means_lr(step: int, steps: int, lr_init: float, lr_final_ratio: float) -> float:
    
    t = min(step / max(steps - 1, 1), 1.0)
    return lr_init * (lr_final_ratio ** t)


def _reset_opacities(
    opacities: torch.Tensor, optimizer: torch.optim.Optimizer, value: float = 0.01
) -> None:
    
    
    
    inv_sigmoid = float(np.log(value / (1 - value)))
    with torch.no_grad():
        opacities.data.fill_(inv_sigmoid)
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
    ssim_weight: float = 0.2,
    opacity_reset_interval: int = 3000,
    mixed_precision: bool = True,
) -> dict[str, torch.Tensor]:
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat training requires CUDA"
        )
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    target_all = _load_targets_parallel(images)  

    means = data["means"].to(device).requires_grad_()
    colors_init = data["colors"].numpy()
    colors_init = np.clip(colors_init, 1e-4, 1 - 1e-4)
    colors_logits = np.log(colors_init / (1 - colors_init)).astype(np.float32)
    colors = torch.from_numpy(colors_logits).to(device).requires_grad_()

    from scipy.spatial import cKDTree

    means_np = data["means"].numpy()
    tree = cKDTree(means_np)
    dists, _ = tree.query(means_np, k=4)
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
    max_log_scale = float(np.log(mean_nn_dist.max() * 20))

    voxel_opt = None
    if voxel_guided:
        voxel_opt = VoxelGuidedOptimizer(
            means.detach(), voxel_config or VoxelGuidedConfig(), device
        )

    use_amp = mixed_precision and device == "cuda"

    perm = torch.randperm(num_views, device=device)
    cursor = 0
    for step in range(steps):
        if cursor + view_batch_size > num_views:
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
            l1 = torch.abs(rendered - target).mean()
            if ssim_weight > 0:
                rendered_nchw = rendered.permute(0, 3, 1, 2).float()
                target_nchw = target.permute(0, 3, 1, 2).float()
                ssim_val = _ssim(rendered_nchw, target_nchw)
                loss = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim_val)
            else:
                loss = l1
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
            and step < steps - 1
        ):
            _reset_opacities(opacities, optimizer)

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
    ssim_weight: float = 0.2,
    opacity_reset_interval: int = 3000,
    mixed_precision: bool = True,
    colmap_gpu: bool = True,
) -> Path:
    capture_dir = output / "capture"
    extract_frames(video, capture_dir, every, max_width)
    undistorted = run_colmap(capture_dir, output / "colmap", vocab_tree, colmap_gpu)
    data, images, width, height = load_reconstruction(undistorted, max_points)
    result = train_splats(
        data, images, width, height, steps, device, view_batch_size,
        voxel_guided, voxel_config,
        means_lr_init, means_lr_final_ratio, ssim_weight,
        opacity_reset_interval, mixed_precision,
    )
    glb_path = output / "model.glb"
    print("Exporting model...")
    export_gltf(result, glb_path)
    start_viewer(glb_path, width, height)
    return glb_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gaussian splat model from a drone video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/gsplat"))
    parser.add_argument("--every", type=int, default=5, help="Keep every Nth video frame.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=100_000)
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
        help="Voxels with fewer live Gaussians than this are pruned.",
    )
    parser.add_argument(
        "--voxel-gamma3", type=float, default=0.075,
        help="Voxels whose average opacity falls below this are pruned.",
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
        "--ssim-weight", type=float, default=0.2,
        help="Loss = (1 - w) * L1 + w * (1 - SSIM). Set to 0 to use pure L1.",
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
        "--no-colmap-gpu", dest="colmap_gpu", action="store_false",
        help="Force COLMAP SIFT extraction/matching onto CPU.",
    )
    args = parser.parse_args()
    voxel_config = VoxelGuidedConfig(
        n_along_shortest=args.voxel_n,
        tau=args.voxel_tau,
        gamma1=args.voxel_gamma1,
        gamma2=args.voxel_gamma2,
        gamma3=args.voxel_gamma3,
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
            args.ssim_weight,
            args.opacity_reset_interval,
            args.mixed_precision,
            args.colmap_gpu,
        )
    )


if __name__ == "__main__":
    main()
