from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from src.clip_box import validate_box, validate_axes

SH_C0 = 0.28209479177387814

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_ATTR_ROTATION = "KHR_gaussian_splatting:ROTATION"
_ATTR_SCALE = "KHR_gaussian_splatting:SCALE"
_ATTR_OPACITY = "KHR_gaussian_splatting:OPACITY"
_ATTR_SH0 = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _pad(buf: bytes, pad_byte: bytes) -> bytes:
    remainder = len(buf) % 4
    return buf if remainder == 0 else buf + pad_byte * (4 - remainder)


def write_gsplat_glb(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    color_space: str = "srgb_rec709_display",
    clip_bounds=None,
    clip_axes=None,
    clip_final=False,
) -> None:
    n = means.shape[0]
    means = means.astype(np.float32)
    scales = np.clip(scales.astype(np.float32), 0.0, None)
    opacities = np.clip(opacities.astype(np.float32), 0.0, 1.0)
    colors = np.clip(colors.astype(np.float32), 0.0, 1.0)


    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    norm = np.clip(np.linalg.norm(quats_xyzw, axis=1, keepdims=True), 1e-8, None)
    quats_xyzw = quats_xyzw / norm


    if color_space == "srgb_rec709_display":
        color0_rgb = _srgb_to_linear(colors)
    elif color_space == "lin_rec709_display":
        color0_rgb = colors
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")
    sh0 = (color0_rgb - 0.5) / SH_C0
    color0 = np.concatenate([color0_rgb, opacities[:, None]], axis=1).astype(np.float32)


    buffer = bytearray()
    accessors = []
    buffer_views = []

    def add_accessor(data: np.ndarray, accessor_type: str, with_bounds: bool = False):
        offset = len(buffer)
        payload = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        buffer.extend(payload)


        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        })
        accessor = {
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": n,
            "type": accessor_type,
        }
        if with_bounds:
            accessor["min"] = data.min(axis=0).tolist()
            accessor["max"] = data.max(axis=0).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    position_idx = add_accessor(means, "VEC3", with_bounds=True)
    color0_idx = add_accessor(color0, "VEC4")
    scale_idx = add_accessor(scales, "VEC3")
    rotation_idx = add_accessor(quats_xyzw, "VEC4")
    opacity_idx = add_accessor(opacities.reshape(-1, 1), "SCALAR")
    sh0_idx = add_accessor(sh0, "VEC3")

    gltf = {
        "asset": {"version": "2.0", "generator": "build_gsplat.py (KHR_gaussian_splatting)"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": position_idx,
                    "COLOR_0": color0_idx,
                    _ATTR_SCALE: scale_idx,
                    _ATTR_ROTATION: rotation_idx,
                    _ATTR_OPACITY: opacity_idx,
                    _ATTR_SH0: sh0_idx,
                },
                "mode": 0,
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "kernel": "ellipse",
                        "colorSpace": color_space,
                        "sortingMethod": "cameraDistance",
                        "projection": "perspective",
                    }
                },
            }]
        }],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    if clip_bounds is not None:
        gltf["extras"] = {"gsplat_clip_bounds": validate_box(clip_bounds).tolist()}
        gltf["extras"]["gsplat_clip_final"] = bool(clip_final)
        if clip_axes is not None:
            gltf["extras"]["gsplat_clip_axes"] = validate_axes(clip_axes).tolist()

    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad(bytes(buffer), b"\x00")

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as f:
        _ = f.write(struct.pack("<III", _GLB_MAGIC, 2, total_len))
        _ = f.write(struct.pack("<II", len(json_chunk), _CHUNK_JSON))
        _ = f.write(json_chunk)
        _ = f.write(struct.pack("<II", len(bin_chunk), _CHUNK_BIN))
        _ = f.write(bin_chunk)


def read_gsplat_glb(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        raw = f.read()

    magic, version, _length = struct.unpack_from("<III", raw, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"Not a .glb file: {path}")
    if version != 2:
        raise ValueError(f"Unsupported glTF binary version: {version}")

    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_data = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == _CHUNK_JSON:
            json_chunk = chunk_data
        elif chunk_type == _CHUNK_BIN:
            bin_chunk = chunk_data
    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"Malformed .glb (missing JSON or BIN chunk): {path}")

    gltf = json.loads(json_chunk.decode("utf-8"))
    if "KHR_gaussian_splatting" not in gltf.get("extensionsUsed", []):
        raise ValueError(f"{path} does not use KHR_gaussian_splatting")

    primitive = gltf["meshes"][0]["primitives"][0]
    attrs = primitive["attributes"]
    accessors = gltf["accessors"]
    buffer_views = gltf["bufferViews"]

    type_sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def read_attr(name: str) -> np.ndarray:
        accessor = accessors[attrs[name]]
        if accessor["componentType"] != 5126:
            raise ValueError(
                f"Only FLOAT accessors are supported by this reader (attribute {name})"
            )
        view = buffer_views[accessor["bufferView"]]
        start = view.get("byteOffset", 0)
        length = view["byteLength"]
        count = accessor["count"]
        n_comp = type_sizes[accessor["type"]]
        arr = np.frombuffer(bin_chunk[start : start + length], dtype="<f4", count=count * n_comp)
        return arr.reshape(count, n_comp)

    means = read_attr("POSITION")
    scales = read_attr(_ATTR_SCALE)
    quats_xyzw = read_attr(_ATTR_ROTATION)
    opacities = read_attr(_ATTR_OPACITY).reshape(-1)
    sh0 = read_attr(_ATTR_SH0)

    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
    norm = np.clip(np.linalg.norm(quats_wxyz, axis=1, keepdims=True), 1e-8, None)
    quats_wxyz = quats_wxyz / norm

    colors = np.clip(SH_C0 * sh0 + 0.5, 0.0, 1.0).astype(np.float32)
    # Undo this writer's conversion so the viewer receives the same per-splat
    # colors as the live training preview (before alpha compositing).
    color_space = primitive.get("extensions", {}).get("KHR_gaussian_splatting", {}).get("colorSpace")
    if color_space == "srgb_rec709_display":
        colors = _linear_to_srgb(colors)
    elif color_space != "lin_rec709_display":
        raise ValueError(f"Unsupported or missing color_space: {color_space}")
    opacities = np.clip(opacities, 0.0, 1.0).astype(np.float32)
    scales = np.clip(scales, 0.0, None).astype(np.float32)

    result = {
        "means": torch.from_numpy(means.astype(np.float32)),
        "colors": torch.from_numpy(colors),
        "scales": torch.from_numpy(scales),
        "quats": torch.from_numpy(quats_wxyz.astype(np.float32)),
        "opacities": torch.from_numpy(opacities),
    }
    bounds = gltf.get("extras", {}).get("gsplat_clip_bounds")
    if bounds is not None:
        result["clip_bounds"] = torch.from_numpy(validate_box(bounds))
    axes = gltf.get("extras", {}).get("gsplat_clip_axes")
    if axes is not None:
        result["clip_axes"] = torch.from_numpy(validate_axes(axes))
    if gltf.get("extras", {}).get("gsplat_clip_final", False):
        result["clip_final"] = torch.tensor(True)
    return result
