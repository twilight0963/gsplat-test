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


@dataclass
class Capture:
    frames: list[Path]
    width: int
    height: int
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
    width = height = 0
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
            height, width = frame.shape[:2]
            path = output / f"frame_{len(frames):06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write frame: {path}")
            frames.append(path)
        index += 1
    reader.release()
    if len(frames) < 2:
        raise RuntimeError("The video did not produce at least two usable frames")
    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": [p.name for p in frames],
    }
    (output / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Capture(frames, width, height, fps)


def run_colmap(capture: Path, sparse: Path) -> Path:
    database = capture / "database.db"
    sparse.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(capture),
        ],
        ["colmap", "exhaustive_matcher", "--database_path", str(database)],
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(capture),
            "--output_path",
            str(sparse),
        ],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "COLMAP is required for camera poses. Install it and ensure `colmap` is on PATH."
            ) from exc
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError(
            "COLMAP found no valid reconstruction; use a video with more overlap."
        )
    return models[0]


def _camera_intrinsics(camera: object, width: int, height: int) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float32)
    model = str(camera.model).upper()
    if "SIMPLE_PINHOLE" in model:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif "PINHOLE" in model:
        fx, fy, cx, cy = params[:4]
    elif "SIMPLE_RADIAL" in model or "RADIAL" in model:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise RuntimeError(f"Unsupported COLMAP camera model: {camera.model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32).reshape(
        3, 3
    )


def load_reconstruction(
    model_path: Path, capture: Capture
) -> tuple[dict[str, torch.Tensor], list[Path]]:
    reconstruction = pycolmap.Reconstruction(str(model_path))
    points = list(reconstruction.points3D.values())
    if not points:
        raise RuntimeError("COLMAP produced no sparse points.")
    means = np.stack([np.asarray(point.xyz, dtype=np.float32) for point in points])
    colors = np.stack(
        [np.asarray(point.color, dtype=np.float32) / 255.0 for point in points]
    )

    viewmats: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    images: list[Path] = []
    for image in reconstruction.images.values():
        frame = capture.frames[0].parent / image.name
        if not frame.exists():
            continue
        viewmats.append(np.asarray(image.cam_from_world.matrix(), dtype=np.float32))
        camera = reconstruction.cameras[image.camera_id]
        intrinsics.append(_camera_intrinsics(camera, capture.width, capture.height))
        images.append(frame)
    if not images:
        raise RuntimeError("No reconstructed images matched the extracted frames.")
    return {
        "means": torch.from_numpy(means),
        "colors": torch.from_numpy(colors),
        "viewmats": torch.from_numpy(np.stack(viewmats)),
        "Ks": torch.from_numpy(np.stack(intrinsics)),
    }, images


def train_splats(
    data: dict[str, torch.Tensor],
    images: list[Path],
    width: int,
    height: int,
    steps: int,
    device: str,
) -> dict[str, torch.Tensor]:

    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat training requires CUDA; run COLMAP on this machine, then train on an NVIDIA GPU."
        )
    target_images = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read reconstructed frame: {path}")
        target_images.append(
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).float() / 255.0
        )
    target = torch.stack(target_images).to(device)
    means = data["means"].to(device).requires_grad_()
    colors = data["colors"].to(device).requires_grad_()
    scales = torch.full_like(means, -2.5, requires_grad=True)
    quats = torch.zeros((means.shape[0], 4), device=device)
    quats[:, 0] = 1
    quats.requires_grad_()
    opacities = torch.full((means.shape[0],), 0.0, device=device, requires_grad=True)
    viewmats = data["viewmats"].to(device)
    Ks = data["Ks"].to(device)
    optimizer = torch.optim.Adam([means, colors, scales, quats, opacities], lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rendered, _, _ = rasterization(
            means,
            quats,
            scales.exp(),
            opacities.sigmoid(),
            colors.clamp(0, 1),
            viewmats,
            Ks,
            width,
            height,
            packed=True,
        )
        loss = torch.abs(rendered - target).mean()
        loss.backward()
        optimizer.step()
    return {
        k: v.detach().cpu()
        for k, v in {
            "means": means,
            "colors": colors,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "viewmats": viewmats,
            "Ks": Ks,
        }.items()
    }


def build_model(
    video: Path,
    output: Path,
    every: int = 5,
    max_width: int = 1920,
    steps: int = 100,
    device: str = "cuda",
) -> Path:
    capture_dir = output / "capture"
    capture = extract_frames(video, capture_dir, every, max_width)
    model = run_colmap(capture_dir, output / "sparse")
    data, images = load_reconstruction(model, capture)
    result = train_splats(data, images, capture.width, capture.height, steps, device)
    checkpoint = output / "model.pt"
    torch.save(result, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gaussian splat model from a video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/gsplat"))
    parser.add_argument(
        "--every", type=int, default=5, help="Keep every Nth video frame."
    )
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(
        build_model(
            args.video, args.output, args.every, args.max_width, args.steps, args.device
        )
    )


if __name__ == "__main__":
    main()
