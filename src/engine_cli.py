"""Command-line options of src.engine.

Kept free of heavy imports (torch, pycolmap) so the upload server can reuse the
same definitions to validate browser-supplied arguments and show their help.
"""
import argparse
from pathlib import Path


def build_parser(exclude=(), parser_class=argparse.ArgumentParser, **parser_kwargs):
    """The engine's argument parser; options whose first flag is in `exclude` are left out."""
    parser_kwargs.setdefault('description', "Build a Gaussian splat model from a drone video or image directory.")
    parser = parser_class(**parser_kwargs)

    def add(*flags, **kwargs):
        if flags[0] not in exclude:
            parser.add_argument(*flags, **kwargs)

    add("video", type=Path, metavar='INPUT', help='Video file or directory of photos (non-recursive).')
    add("--output", type=Path, default=Path("runs/gsplat"),
        help="Folder for model.glb and the run reports (default: runs/gsplat).")
    add("--every", type=int, default=5,
        help="Keep every Nth video frame (videos only; see --photo-every for photo directories).")
    add('--keyframe-max-gap', type=int, default=6,
        help='Video: maximum gap in candidates AFTER --every; 1 disables selection (default: 6).')
    add('--keyframe-motion', type=float, default=.06,
        help='Video: tracked displacement / image diagonal triggering a keyframe (default: .06).')
    add('--sequential-overlap', type=int, default=12,
        help='Video COLMAP matching overlap (default: 12).')
    add('--mapper-tracks-per-view', type=int, default=1000,
        help='Tracks global_mapper keeps per image; fewer is faster (default: 1000, 0 = all).')
    add('--eval-every', type=int, default=0,
        help='Hold out every Nth registered view and report PSNR/SSIM (0 = off, 8 is standard).')
    add('--ssim-weight', type=float, default=0.2,
        help='D-SSIM weight in the training loss (0 = pure L1; ignored with SR).')
    add('--no-scale-means-lr', dest='scale_means_lr', action='store_false',
        help='Do not scale the position learning rate by the scene extent.')
    add("--photo-every", type=int, default=2,
        help="Use every Nth photo for SfM/training in image directories "
             "(default: 2; 1 uses all photos). Matching cost grows ~quadratically with image count.")
    add('--image-matching', choices=('auto', 'exhaustive'), default='auto',
        help='Photos: GPS + sequential matching when all images have GPS; otherwise vocab-tree '
             '(if --vocab-tree is given), exhaustive for <=150 photos, or sequential.')
    add('--spatial-neighbors', type=int, default=12,
        help='GPS nearest neighbors per photo in auto mode (default: 12).')
    add("--max-features", type=int, default=4096,
        help="Max SIFT features per image (default: 4096).")
    add("--gp-iterations", type=int, default=50,
        help="global_mapper position-solver iterations (lower = faster).")
    add("--ba-iterations", type=int, default=3,
        help="global_mapper bundle-adjustment rounds (lower = faster).")
    add("--no-colmap-cache", dest="colmap_cache", action="store_false",
        help="Always re-run COLMAP even if inputs and settings are unchanged.")
    add("--max-width", type=int, default=1920,
        help="Downscale wider frames/photos to this width in px; 0 keeps full size. "
             "1280 trains about 2x faster than 1920 (default: 1920).")
    add("--steps", type=int, default=5000,
        help="Training steps (default 5000: ~8 min for ~1800 video frames at 1280 px with MCMC).")
    add("--device", choices=("cuda", "cpu"), default="cuda",
        help="Training device (default: cuda).")
    add("--view-batch-size", type=int, default=None,
        help="Defaults to 1 with SR, otherwise 4.")
    add("--max-points", type=int, default=200_000,
        help="Random subset of COLMAP points used to initialize the splats (default: 200000).")
    add("--densify", choices=("mcmc", "voxel"), default="mcmc",
        help="mcmc: relocate/grow Gaussians up to --max-gaussians (3DGS-MCMC); "
             "voxel: the earlier DroneSplat-style voxel growth.")
    add("--max-gaussians", type=int, default=1_000_000,
        help="Gaussian budget (MCMC cap, and the voxel method's limit).")
    add("--refine-every", type=int, default=50,
        help="MCMC: relocate and grow (by 5%%) every N steps.")
    add("--max-axis-ratio", type=float, default=10.0,
        help="Axis ratio above which splats receive a soft shape penalty.")
    add("--scale-regularization", type=float, default=0.01,
        help="Strength of the anti-streak shape penalty; 0 disables it.")
    add("--vocab-tree", type=Path, default="vocab_tree.bin",
        help="COLMAP vocabulary tree (.bin). Enables retrieval matching for photos "
             "without GPS and loop detection for videos.")
    add(
        "--no-voxel-guided", dest="voxel_guided", action="store_false",
        help="Disable the DroneSplat-style voxel-guided optimization (floater fix).",
    )
    add(
        "--voxel-n", type=int, default=80,
        help="Divide the scene's shortest bbox edge into this many voxels (paper's N).",
    )
    add(
        "--voxel-tau", type=float, default=3.5,
        help="Voxel-lengths a Gaussian may drift/scale before being flagged unconstrained.",
    )
    add(
        "--voxel-gamma1", type=float, default=1e-4,
        help="Accumulated world-space gradient norm needed to grow into an empty voxel "
        "(scene-dependent - see src/voxel_guided.py's module docstring).",
    )
    add(
        "--voxel-gamma2", type=int, default=2,
        help="Prune sparse voxels only when their average opacity is also below voxel-gamma3.",
    )
    add(
        "--voxel-gamma3", type=float, default=0.075,
        help="Average opacity threshold for pruning sparse voxels.",
    )
    add(
        "--voxel-stride", type=int, default=1,
        help="Record visibility/accumulate voxel statistics every N steps (1 = every step).",
    )
    add(
        "--means-lr", type=float, default=1.6e-4,
        help="Initial learning rate for Gaussian positions.",
    )
    add(
        "--means-lr-final-ratio", type=float, default=0.01,
        help="Means LR is annealed exponentially to (means-lr * this ratio) by the last step.",
    )
    add(
        "--opacity-reset-interval", type=int, default=3000,
        help="Reset all opacities every N steps to flush unearned floaters. 0 disables.",
    )
    add(
        "--no-mixed-precision", dest="mixed_precision", action="store_false",
        help="Disable bf16 autocast during rasterization/loss (CUDA only).",
    )
    add(
        "--brightness", type=float, default=0.0,
        help="Added to every pixel in 0-255 units (0 = unchanged).",
    )
    add(
        "--contrast", type=float, default=1.0,
        help="Pixel multiplier (1 = unchanged).",
    )
    add(
        "--sharpness", type=float, default=0.5,
        help="CAS sharpening strength, 0-1, applied to training images only (default: 0.5).",
    )
    add("--super-resolution", action="store_true",
        help="Use tiled SwinIR 2x, CAS and dual-resolution supervision.")
    default_root = Path(__file__).resolve().parent.parent
    add("--swinir-root", type=Path,
        default=default_root / "third_party" / "SwinIR",
        help="Official SwinIR source checkout (same default as upload_server).")
    add("--sr-checkpoint", type=Path,
        default=default_root / "weights" / "swinir-lightweight-x2.pth",
        help="SwinIR-S lightweight 2x checkpoint (same default as upload_server).")
    add("--sr-tile", type=int, default=128, help="Input tile size, multiple of 8, >= 32.")
    add("--sr-prior-weight", type=float, default=0.5,
        help="Enhanced target weight; remaining weight anchors to base images.")
    add("--unsharp", action=argparse.BooleanOptionalAction, default=None,
        help="Unsharp masking: defaults off with SR, on otherwise.")
    add("--use-server", action="store_true",
        help="Use the persistent browser viewer (used by upload_server).")
    add("--headless", action="store_true",
        help="No live viewer or preview snapshots (fastest; use for benchmarking).")
    return parser
