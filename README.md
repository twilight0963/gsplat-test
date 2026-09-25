# gsplat-test

Test repository for Gaussian Splat model

## Browser upload interface

```bash
.venv/bin/python -m src.upload_server
```

Open **http://localhost:8001/**. Select a video and set steps, frame interval,
maximum width, output folder, brightness, contrast, and sharpness. Output must be
a new folder inside `runs/`. Uploaded videos are retained in `runs/.uploads/`.
The page shows upload progress followed by the latest engine stdout/stderr line.

Clicking **Upload and build** keeps you on the upload page. A new tab opens at
**http://localhost:8000/** only when the training viewer reports it is ready.
Browsers may block delayed popups; if so, use the **Open viewer** link. The viewer remains open after
export; closing a browser tab does not stop the process.

The engine exits after training and export. The upload page then enables another
upload and suggests a fresh output folder automatically; no server restart is
needed between jobs. One training job runs at a time.

The live viewer runs as a separate persistent process on port 8000 and keeps the
last model visible. When the next job reaches training, that same viewer switches
to its previews and resets its camera and bounding box. It remains alive even if
the upload server exits. Stop the `src.live_viewer` process to shut it down.

Preview snapshots and status are exchanged atomically through `runs/.viewer/`;
viewer startup errors are logged in `runs/.viewer/viewer.log`. On the first launch
after upgrading, close any older viewer already using port 8000. The usual CUDA,
COLMAP, and sharpening dependencies are still required. Stopping the upload server
while training terminates its engine process, so wait for `Model saved:` if you
want the exported result.

Use `--port 8002` to change the upload port (8000 is reserved for the viewer).
For another device on the same network, launch with `--host 0.0.0.0` and open
`http://<computer-IP>:8001/`; the viewer link uses that same hostname on port 8000.


### Video keyframes and runtime validation

`--every 10` first samples the source video. Adaptive selection then operates on
those candidates, using tracked image displacement and sharpness. It retains
endpoints, keeps both sides of tracking breaks, and limits candidate gaps to six
by default. Tracking failure retains extra views; there is no hard frame-count
cap or guaranteed runtime. Image motion is a coverage heuristic, not metric
parallax or proof of reconstruction accuracy. Photo selection is unchanged.

```bash
.venv/bin/python -m src.engine INPUT.mp4 --output runs/keyframes \
  --every 10 --max-width 960 --steps 2000 --headless
```

Use `--keyframe-max-gap 1` to disable adaptive selection. Adjust
`--keyframe-max-gap` (default 6), `--keyframe-motion` (default 0.06 of the image
diagonal), and `--sequential-overlap` (default 12) for different capture motion.
`--vocab-tree vocab_tree.bin` enables COLMAP retrieval-based loop detection;
retaining revisited frames alone does not ensure loop closure. The original
video matching overlap can be restored with `--sequential-overlap 30`.

Selected frames and their candidate indices/reasons are recorded in
`capture/keyframes/<input-and-settings-hash>/keyframes.json`. Original candidates
are preserved. Benchmark reports distinguish `keyframe_candidates` from
`frames_before_colmap`, report keyframe-selection time and individual COLMAP
command subtotals, and state whether the completed run took at most 600 seconds.
Command subtotals are included in COLMAP time and must not be added again.

Video matching uses a dense window (`quadratic_overlap=0`): COLMAP's quadratic
mode replaces the window with gaps 1, 2, 4, 8, ..., which verified about half as
many pairs and made `global_mapper` about twice as slow on test footage.
`--mapper-tracks-per-view` (default 1000, `0` = keep all) limits the tracks
`global_mapper` optimizes; on a 319-view test it cut mapping time by ~25% with
the same registration, reprojection error and held-out PSNR.

### Training quality

Training uses L1 plus a D-SSIM term (`--ssim-weight`, default 0.2, evaluated at
half resolution for speed; `0` restores pure L1) and scales the position learning
rate by the camera extent, as in 3DGS (`--no-scale-means-lr` disables it). Use
`--eval-every 8` to hold out every 8th registered view and report PSNR/SSIM in
`benchmark.log`/`benchmark.json` under `quality`; held-out views are not trained on,
so leave it off for production models.

Absolute geographic accuracy is reported as **unverified**. Photo EXIF GPS is
used for neighbor matching, not an implemented geographic alignment or checkpoint
evaluation. Video-only inputs have no geographic reference. A runtime pass is
not a claim of <=1 m absolute error.

### Photo directories and metadata

Every completed engine run writes `benchmark.log` (readable summary) and
`benchmark.json` (machine-readable timings) in its output directory. Reports
include UTC start/end timestamps, total wall time, decoded video frames,
selected/imported frames before COLMAP, usable frames from the selected COLMAP
reconstruction, and separate durations for input preparation, preprocessing,
CAS, SR, COLMAP, reconstruction loading, viewer startup, training and export.
CAS/SR are not double-counted in other stages; preprocessing total is their
subtotal plus brightness/contrast, unsharp and target preparation. COLMAP time
includes undistortion. Training time includes target loading, preview publishing
and final box fitting. Failed runs do not generate a completed-run report.

The engine also accepts a directory of JPEG, PNG or TIFF photographs:

```bash
.venv/bin/python -m src.engine Dataset/Images --output runs/tower-photos --max-width 1920
```

All photos directly inside the directory are used; `--every` applies only to
video. Use a new output directory and photos with consistent dimensions and
upright EXIF orientation. The originals are copied byte-for-byte into
`capture/photos`, preserving EXIF and DJI XMP; `photo_metadata.json` records the
source-to-capture mapping. Files are ordered by capture time when every image has
a timestamp, otherwise by filename.

When all photos have valid EXIF GPS, auto matching combines 20 GPS neighbors per
image (100 m search radius, horizontal distance) with sequential overlap 10.
Change the neighbor budget with `--spatial-neighbors`. With incomplete GPS,
matching falls back to exhaustive. `--image-matching exhaustive` explicitly
selects all-pairs matching for a quality comparison. GPS selects candidate pairs;
it is not treated as exact geometry, and gimbal angles are not imposed as poses.

Consistent DJI calibrated focal length and optical center seed a shared
SIMPLE_RADIAL camera, with lens distortion and focal length still refined by
COLMAP. If calibration is missing or differs, COLMAP estimates the appropriate
parameters instead. Metadata is read at original resolution: feature extraction
and undistortion apply the image-size limit inside COLMAP so camera coordinates
remain consistent. For portrait photos the equivalent longest-side limit is used.
Brightness/contrast, unsharp masking and CAS run on the undistorted training
targets. `--super-resolution` also works with photo directories; its 2x targets
and base-image loss use the same undistorted cameras.

This optimization reduces the candidate matching workload; end-to-end runtime
and reconstruction completeness still depend on overlap and scene content.

### Optional SwinIR super-resolution training

`src.engine --super-resolution` enables an SRGS-inspired 2× pipeline. COLMAP
uses unenhanced extracted frames. After undistortion, brightness/contrast produce
lossless base targets; SwinIR-S creates enhanced targets, then CAS sharpens them.
Training blends L1 against enhanced targets with L1 between area-downsampled
renders and base targets (`--sr-prior-weight`, default 0.5). This is an adaptation,
not a reproduction of the paper's full training recipe or a guarantee of mapping
accuracy. Camera intrinsics are doubled with image dimensions.

Install the optional dependency and obtain the official architecture and weights:

```bash
.venv/bin/python -m pip install -r requirements-sr.txt
git clone https://github.com/JingyunLiang/SwinIR.git third_party/SwinIR
mkdir -p weights
curl -fL https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth -o weights/swinir-lightweight-x2.pth
```

Use a new output directory for each run (COLMAP databases are not interchangeable
between preprocessing modes). Run from the project root:

```bash
.venv/bin/python -m src.engine input.mp4 --output runs/sr-example \
  --super-resolution \
  --max-width 640 --sr-tile 128 --sharpness 0.2 --steps 5000
```

The engine CLI uses the same default SwinIR source and checkpoint locations as
the upload server (`third_party/SwinIR` and `weights/swinir-lightweight-x2.pth`).
The path flags are only needed for a custom installation.

SR defaults to one camera per training step, with unsharp masking disabled.
`--unsharp` explicitly enables it after SR, before CAS; `--no-unsharp` disables
it in the existing mode too. Brightness remains an additive pixel offset and
contrast a multiplier. CAS remains enabled, including at sharpness 0 (minimum
extra sharpening, not a bypass). SR inference uses overlapping tiles with CPU
assembly, and runs in a separate process that exits before training begins.
Training reads only the selected SR image batch from disk. The existing Gaussian
budget controls still apply. These defaults are a starting point for 6 GB GPUs;
actual peak memory depends on scene complexity and the live viewer as well.

Without `--super-resolution`, the enhancement mode uses batch size 4 by default.
For both videos and photo directories, COLMAP receives unenhanced images;
brightness/contrast, unsharp and CAS are applied afterward to registered,
undistorted training images only. Video extraction still applies `--every` and
`--max-width` before COLMAP. Use a fresh output directory when changing pipelines.
The web upload interface includes the SR mode, tile size, enhanced-image weight,
and automatic/on/off unsharp controls. After the setup above, restart
`python -m src.upload_server` and select **On — SwinIR 2× + CAS**. Set maximum
width to 640 and CAS to 0.2 as an initial 6 GB configuration. The existing
brightness/contrast controls apply in both modes. The viewer still opens only
when splat training starts, after SR preprocessing finishes.

For custom installation locations, launch the upload server with
`--swinir-root /path/to/SwinIR --sr-checkpoint /path/to/weights.pth`.
These paths refer to the server machine, not the uploading phone/computer.
SwinIR's source and weights are external prerequisites; they are not downloaded
during training. Missing source/weights are reported before accepting the job.

Jobs launched by `src.upload_server` pass `--use-server`, selecting the
persistent browser viewer. Direct `src.engine` runs default to the desktop
OpenCV viewer; use `--use-server` explicitly when a browser viewer is desired.
