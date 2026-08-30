from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

SH_C0 = 0.28209479177387814

PLY_TYPE_MAP = {
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1",
    "char": "i1", "int8": "i1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
}


def load_ply(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="ignore")
    lines = [l.strip() for l in header.splitlines()]

    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary_little_endian PLY is supported by this viewer.")

    n_vertex = 0
    fields: list[tuple[str, str]] = []
    in_vertex_element = False
    for line in lines:
        if line.startswith("element vertex"):
            n_vertex = int(line.split()[-1])
            in_vertex_element = True
        elif line.startswith("element") and not line.startswith("element vertex"):
            in_vertex_element = False
        elif line.startswith("property") and in_vertex_element:
            _, ptype, pname = line.split()
            fields.append((pname, PLY_TYPE_MAP[ptype]))

    dtype = np.dtype(fields)
    data = np.frombuffer(raw, dtype=dtype, count=n_vertex, offset=header_end)
    names = set(data.dtype.names)

    means = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    has_gaussian_fields = {"f_dc_0", "opacity", "scale_0", "rot_0"} <= names
    if has_gaussian_fields:
        f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        colors = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0).astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
        scales = np.exp(
            np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
        )
        quats = np.stack(
            [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1
        ).astype(np.float32)
        quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8, None)
    else:
        # Plain colored point cloud (e.g. a raw COLMAP sparse.ply) - fake it as
        # small, opaque, isotropic gaussians so the same viewer still works.
        if {"red", "green", "blue"} <= names:
            colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
        else:
            colors = np.ones((n_vertex, 3), dtype=np.float32) * 0.7
        nn = max(np.ptp(means, axis=0).mean() / max(n_vertex ** (1 / 3), 1.0), 1e-4)
        scales = np.full((n_vertex, 3), nn, dtype=np.float32)
        quats = np.zeros((n_vertex, 4), dtype=np.float32)
        quats[:, 0] = 1.0
        opacities = np.ones(n_vertex, dtype=np.float32)

    return {
        "means": torch.from_numpy(means),
        "colors": torch.from_numpy(colors),
        "scales": torch.from_numpy(scales),
        "quats": torch.from_numpy(quats),
        "opacities": torch.from_numpy(opacities),
    }


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
        y_axis = np.cross(z_axis, x_axis)  # "down" in camera space

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
    ply: Path | str,
    width: int = 1920,
    height: int = 1080,
    fov: float = 30.0,
) -> None:
    ply = Path(ply)

    if not torch.cuda.is_available():
        raise RuntimeError("This viewer needs a CUDA GPU (gsplat's rasterizer is CUDA-only).")
    device = "cuda"

    data = load_ply(ply)
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
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            state["dragging"] = "orbit" if event == cv2.EVENT_LBUTTONDOWN else "pan"
            state["last"] = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
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
    start_viewer("runs/scene/model.ply")
