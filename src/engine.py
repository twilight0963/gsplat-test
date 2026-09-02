from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from gsplat import rasterization
from src.gsplat_viewer import start_viewer
from src.gltf_gsplat import write_gsplat_glb

@dataclass
class Capture:
    frames: list[Path]
    fps: float


def extract_frames(video: Path, output: Path, every: int, max_width: int) -> Capture:
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
    return Capture(frames, fps)


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("COLMAP is required and must be on PATH.") from exc


def run_colmap(capture: Path, workdir: Path, vocab_tree: Path | None) -> Path:
    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    _run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(capture),
        "--ImageReader.single_camera", "1",
    ])

    match_command = [
        "colmap", "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", "15",
    ]
    if vocab_tree is not None:
        match_command += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", str(vocab_tree),
        ]
    _run(match_command)

    _run([
        "colmap", "mapper",
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

def train_splats(
    data: dict[str, torch.Tensor],
    images: list[Path],
    width: int,
    height: int,
    steps: int,
    device: str,
    view_batch_size: int,
) -> dict[str, torch.Tensor]:
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat training requres CUDA"
        )

    target_images = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read reconstructed frame: {path}")
        target_images.append(
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).float() / 255.0
        )
    
    
    target_all = torch.stack(target_images).to(device)

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
        {"params": [means], "lr": 1.6e-4},
        {"params": [colors], "lr": 2.5e-3},
        {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [opacities], "lr": 5e-2},
    ])
    max_log_scale = float(np.log(mean_nn_dist.max() * 20))

    perm = torch.randperm(num_views, device=device)
    cursor = 0
    for _ in range(steps):
        if cursor + view_batch_size > num_views:
            perm = torch.randperm(num_views, device=device)
            cursor = 0
        idx = perm[cursor : cursor + view_batch_size]
        cursor += view_batch_size

        target = target_all[idx]
        viewmats = viewmats_all[idx]
        Ks = Ks_all[idx]

        optimizer.zero_grad(set_to_none=True)
        quats_n = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scales_c = scales.clamp(max=max_log_scale)

        rendered, _, _ = rasterization(
            means, quats_n, scales_c.exp(), opacities.sigmoid(), colors.sigmoid(),
            viewmats, Ks, width, height, packed=True,
        )
        loss = torch.abs(rendered - target).mean()
        loss.backward()
        optimizer.step()

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
    steps: int = 2000,
    device: str = "cuda",
    view_batch_size: int = 4,
    max_points: int = 100_000,
    vocab_tree: Path | None = None,
) -> Path:
    capture_dir = output / "capture"
    capture = extract_frames(video, capture_dir, every, max_width)
    undistorted = run_colmap(capture_dir, output / "colmap", vocab_tree)
    data, images, width, height = load_reconstruction(undistorted, max_points)
    result = train_splats(data, images, width, height, steps, device, view_batch_size)
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
    args = parser.parse_args()
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
        )
    )


if __name__ == "__main__":
    main()
