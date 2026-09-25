"""Optional SwinIR-S 2x inference worker, isolated from the training CUDA context."""
from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
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


def _predict(model: torch.nn.Module, patches: torch.Tensor, half: bool) -> torch.Tensor:
    """Run a batch of tiles; fp16 output that is not finite is recomputed in fp32."""
    if half:
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(patches).float()
        if torch.isfinite(prediction).all():
            return prediction
    prediction = model(patches).float()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("SR model returned invalid output")
    return prediction


@torch.inference_mode()
def upscale_tiled(model: torch.nn.Module, image: torch.Tensor, tile: int,
                  device: str, batch: int = 8, half: bool | None = None) -> torch.Tensor:
    """CPU CHW RGB [0,1] -> CPU CHW RGB.

    Tiles are run `batch` at a time (fp16 on CUDA unless `half=False`) and
    blended on `device`; only the input and output images cross to the CPU.
    """
    if tile < 32 or tile % 8:
        raise ValueError("Tile size must be a multiple of 8 and at least 32")
    if batch < 1:
        raise ValueError("Tile batch must be at least 1")
    half = device == "cuda" if half is None else half
    _, h, w = image.shape
    padded = F.pad(image.unsqueeze(0).to(device), (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    ph, pw = padded.shape[-2:]
    th, tw = min(tile, ph), min(tile, pw)
    overlap_h, overlap_w = min(16, th // 2), min(16, tw // 2)
    ys = list(range(0, ph - th, th - overlap_h)) + [ph - th]
    xs = list(range(0, pw - tw, tw - overlap_w)) + [pw - tw]
    output = torch.zeros(3, ph * 2, pw * 2, device=device)
    weights = torch.zeros(1, ph * 2, pw * 2, device=device)
    # Positive tapered weights suppress tile-boundary seams without zero divisions.
    wy = torch.hann_window(th * 2, periodic=False, device=device).clamp_min(0.01)
    wx = torch.hann_window(tw * 2, periodic=False, device=device).clamp_min(0.01)
    blend = wy[:, None] * wx[None, :]
    tiles = [(y, x) for y in ys for x in xs]
    for start in range(0, len(tiles), batch):
        chunk = tiles[start:start + batch]
        patches = torch.cat([padded[:, :, y:y+th, x:x+tw] for y, x in chunk])
        prediction = _predict(model, patches, half)
        if prediction.shape != (len(chunk), 3, th * 2, tw * 2):
            raise RuntimeError("SR model returned invalid output")
        for (y, x), tile_out in zip(chunk, prediction):
            output[:, y*2:(y+th)*2, x*2:(x+tw)*2] += tile_out * blend
            weights[:, y*2:(y+th)*2, x*2:(x+tw)*2] += blend
    return (output / weights)[:, :h*2, :w*2].clamp(0, 1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch", type=int, default=8, help="Tiles per forward pass.")
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 inference on CUDA.")
    args = parser.parse_args()
    validate_setup(args.root, args.checkpoint, args.tile)
    model = load_model(args.root, args.checkpoint, args.device)
    paths = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    def read(path):
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Could not read SR input: {path}")
        return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255

    def write(target, bgr):
        if not cv2.imwrite(str(target), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"Could not write SR output: {target}")

    # Reading the next input and PNG encoding run in threads while the GPU works.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending, next_image = [], pool.submit(read, paths[0]) if paths else None
        for i, path in enumerate(paths):
            image = next_image.result()
            if i + 1 < len(paths):
                next_image = pool.submit(read, paths[i + 1])
            output = upscale_tiled(model, image, args.tile, args.device, args.batch, half=not args.fp32 and None)
            rgb_out = (output.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            pending.append(pool.submit(write, args.output / Path(path).name, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)))
            if len(pending) > 4:
                pending.pop(0).result()
            print(f"Super-resolution: {i+1}/{len(paths)} ({100*(i+1)/len(paths):.1f}%)", flush=True)
        for future in pending:
            future.result()


if __name__ == "__main__":
    main()
