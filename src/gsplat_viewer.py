from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
import argparse

from src.gltf_gsplat import read_gsplat_glb


class OrbitCamera:
    def __init__(self, target: np.ndarray, radius: float):
        self.target = target.copy()
        self.default_target = target.copy()
        self.radius = radius
        self.default_radius = radius
        self.azimuth = 0.0
        self.elevation = 0.3
        self.up_sign = 1.0

    def reset(self) -> None:
        self.target = self.default_target.copy()
        self.radius = self.default_radius
        self.azimuth = 0.0
        self.elevation = 0.3

    def eye(self) -> np.ndarray:
        ce, se = np.cos(self.elevation), np.sin(self.elevation)
        ca, sa = np.cos(self.azimuth), np.sin(self.azimuth)
        offset = self.radius * np.array([ce * sa, se, ce * ca])
        return self.target + offset

    def viewmat(self) -> np.ndarray:
        eye = self.eye()
        world_up = np.array([0.0, self.up_sign, 0.0])
        z_axis = self.target - eye
        z_axis /= max(np.linalg.norm(z_axis), 1e-8)
        x_axis = np.cross(world_up, z_axis)
        x_axis /= max(np.linalg.norm(x_axis), 1e-8)
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ eye
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R
        mat[:3, 3] = t
        return mat

    def pan(self, dx: float, dy: float) -> None:
        eye = self.eye()
        world_up = np.array([0.0, self.up_sign, 0.0])
        z_axis = self.target - eye
        z_axis /= max(np.linalg.norm(z_axis), 1e-8)
        x_axis = np.cross(world_up, z_axis)
        x_axis /= max(np.linalg.norm(x_axis), 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        shift = (-dx * x_axis + dy * y_axis) * self.radius * 0.001
        self.target += shift


def make_K(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.radians(fov_deg) / 2)
    return np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32
    )


def start_viewer(
    glb: Path | str,
    width: int = 1920,
    height: int = 1080,
    fov: float = 30.0,
) -> None:
    glb = Path(glb)

    if not torch.cuda.is_available():
        raise RuntimeError("This viewer needs a CUDA GPU (gsplat's rasterizer is CUDA-only).")
    device = "cuda"

    data = read_gsplat_glb(glb)
    means = data["means"].to(device)
    quats = data["quats"].to(device)
    scales = data["scales"].to(device)
    opacities = data["opacities"].to(device)
    colors = data["colors"].to(device)

    means_np = data["means"].numpy()
    target = means_np.mean(axis=0)
    radius = float(np.linalg.norm(means_np - target, axis=1).max()) * 1.5 or 1.0
    cam = OrbitCamera(target, radius)
    K = make_K(width, height, fov)
    K_t = torch.from_numpy(K).to(device).unsqueeze(0)

    window = "gsplat viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)

    state = {"dragging": None, "last": (0, 0)}

    def on_mouse(event, x, y, flags, _param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            state["dragging"] = "orbit" if event == cv2.EVENT_LBUTTONDOWN else "pan"
            state["last"] = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP):
            state["dragging"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            lx, ly = state["last"]
            dx, dy = x - lx, y - ly
            if state["dragging"] == "orbit":
                cam.azimuth -= dx * 0.005
                cam.elevation = np.clip(cam.elevation + dy * 0.005, -1.5, 1.5)
            else:
                cam.pan(dx, dy)
            state["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            cam.radius = max(cam.radius * (0.9 ** delta), 1e-3)

    cv2.setMouseCallback(window, on_mouse)

    while True:
        viewmat = torch.from_numpy(cam.viewmat()).to(device).unsqueeze(0)
        with torch.no_grad():
            rendered, _, _ = rasterization(
                means, quats, scales, opacities, colors,
                viewmat, K_t, width, height, packed=True,
            )
        frame = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(window, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("]")):
            cam.radius *= 0.9
        elif key in (ord("-"), ord("[")):
            cam.radius *= 1.1
        elif key == ord("u"):
            cam.up_sign *= -1
        elif key == ord("r"):
            cam.reset()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show a gsplat model in the viewer"
    )
    parser.add_argument("model_path", type=Path, default=Path("runs/scene/model.glb"))
    args = parser.parse_args()
    start_viewer(args.model_path)
