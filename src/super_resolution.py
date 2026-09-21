"""Optional SwinIR-S 2x inference worker, isolated from the training CUDA context."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def validate_setup(root: Path | None, checkpoint: Path | None, tile: int) -> None:
    if root is None or not (root / "models" / "network_swinir.py").is_file():
        raise ValueError("--swinir-root must point to an official SwinIR checkout")
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("--sr-checkpoint must point to the pretrained SwinIR-S lightweight 2x .pth file")
    if tile < 32 or tile % 8:
        raise ValueError("--sr-tile must be a multiple of 8 and at least 32")


def load_model(root: Path, checkpoint: Path, device: str) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("swinir_network", root / "models" / "network_swinir.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError("SwinIR dependencies are missing; install requirements-sr.txt") from exc
    model = module.SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
                          num_heads=[6, 6, 6, 6], mlp_ratio=2,
                          upsampler="pixelshuffledirect", resi_connection="1conv")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params", state), strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def upscale_tiled(model: torch.nn.Module, image: torch.Tensor, tile: int,
                  device: str) -> torch.Tensor:
    """CPU CHW RGB [0,1] -> CPU CHW RGB, with only one tile on the GPU."""
    if tile < 32 or tile % 8:
        raise ValueError("Tile size must be a multiple of 8 and at least 32")
    _, h, w = image.shape
    padded = F.pad(image.unsqueeze(0), (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    ph, pw = padded.shape[-2:]
    th, tw = min(tile, ph), min(tile, pw)
    overlap_h, overlap_w = min(16, th // 2), min(16, tw // 2)
    ys = list(range(0, ph - th, th - overlap_h)) + [ph - th]
    xs = list(range(0, pw - tw, tw - overlap_w)) + [pw - tw]
    output = torch.zeros(3, ph * 2, pw * 2)
    weights = torch.zeros(1, ph * 2, pw * 2)
    # Positive tapered weights suppress tile-boundary seams without zero divisions.
    wy = torch.hann_window(th * 2, periodic=False).clamp_min(0.01)
    wx = torch.hann_window(tw * 2, periodic=False).clamp_min(0.01)
    blend = wy[:, None] * wx[None, :]
    for y in ys:
        for x in xs:
            patch = padded[:, :, y:y+th, x:x+tw].to(device)
            prediction = model(patch).float().cpu().squeeze(0)
            if prediction.shape != (3, th * 2, tw * 2) or not torch.isfinite(prediction).all():
                raise RuntimeError("SR model returned invalid output")
            output[:, y*2:(y+th)*2, x*2:(x+tw)*2] += prediction * blend
            weights[:, y*2:(y+th)*2, x*2:(x+tw)*2] += blend
            del patch, prediction
    return (output / weights)[:, :h*2, :w*2].clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    validate_setup(args.root, args.checkpoint, args.tile)
    model = load_model(args.root, args.checkpoint, args.device)
    paths = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Could not read SR input: {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        output = upscale_tiled(model, image, args.tile, args.device)
        rgb_out = (output.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        target = args.output / Path(path).name
        if not cv2.imwrite(str(target), cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write SR output: {target}")
        print(f"Super-resolution: {i+1}/{len(paths)} ({100*(i+1)/len(paths):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
